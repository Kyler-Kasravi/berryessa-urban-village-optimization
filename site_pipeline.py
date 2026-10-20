"""
Real-site NSGA-III land use optimization: Berryessa BART Urban Village, San Jose
================================================================================

1. Study area. With the official data in data/raw/ (fetch_official.py), the grid covers the City's
   Berryessa BART Urban Village boundary plus a 360 ft margin. Without it, the grid falls back to the
   OpenStreetMap download box (the October 2026 prototype).
2. Classify every 120 ft cell (California State Plane Zone III, US feet):
     UNBUILDABLE  roads, rail, creek setback (Council Policy 6-34), FEMA floodway -> -1
     EXISTING     existing homes / parks / schools -> fixed 0 / 1 / 4, count in the objectives
     SITE         redevelopable land: parcels inside the Urban Village that are surface parking, the
                  flea market, vacant, or lightly built (OSM building coverage under 30%)  -> -2
     OUTSIDE      everything else -> -1
   Manual overrides from site_edits.gpkg are applied last.
3. Put the City's adopted plan (General Plan 2040 designations) on the same cells. By default the
   optimizer gets the same amount of each land use as the plan (--program city), so the comparison
   is about arrangement, not quantity.
4. Optimize with Kyler's NSGA-III code (cluster-growth encoding, operators, livability objective):
     F1  average walk from each home to the nearest job cell        (network or grid distance)
     F2  average walk from each home to the nearest school/service  (network or grid distance)
     F3  livability (maximized)
     F4  average walk from each new home to the BART station        (network or grid distance)
     G1  building collisions (hard)      G2  homes touching industry (hard, rule R5)
   Rules R3/R4 (FEMA 1% flood zone, rail noise contour) are counted by default, as the City allows
   homes there with mitigation; --flood-rule hard / --noise-rule hard keep new homes out instead.
5. Score the City's plan with exactly the same functions.
6. Write everything to one GeoPackage for QGIS, with styles and layer metadata embedded.

Usage:
    python site_pipeline.py                      # defaults: walking distances, City's land-use mix
    python site_pipeline.py --distance grid      # optimize on the capstone's grid (taxicab) distances
    python site_pipeline.py --program capstone   # the capstone's land-use mix (with light industry)
    python site_pipeline.py --flood-rule hard --noise-rule hard   # no new homes in either area
    python site_pipeline.py --site-source osm    # ignore the official data (October 6 prototype)
"""

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from xml.sax.saxutils import quoteattr

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely import STRtree
from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import unary_union

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.operators.selection.tournament import TournamentSelection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_to_gis import load_optimizer, nearest_taxicab  # noqa: E402
import official_data as od  # noqa: E402
import site_rules  # noqa: E402

CRS_OUT = "EPSG:2227"     # NAD83 / California zone 3 (ftUS) - Santa Clara County's State Plane zone
FT_PER_M = 3.280833333    # US survey foot
GPKG_DEFAULT = os.path.join(HERE, "berryessa.gpkg")
MARGIN_FT = 360           # grid margin around the Urban Village boundary (3 cells)
TEN_MIN_FT = 2640         # 10-minute walk at 3 mph
REDEV_MAX_COVERAGE = 0.30 # parcel counts as redevelopable below this OSM building coverage

# Approximate paved widths (feet) used to turn OSM road centerlines into areas
ROAD_WIDTH_FT = {
    "motorway": 80, "trunk": 70, "primary": 60, "secondary": 50, "tertiary": 40,
    "residential": 32, "unclassified": 32, "living_street": 24,
    "motorway_link": 30, "trunk_link": 30, "primary_link": 30, "secondary_link": 26, "tertiary_link": 26,
}
RAIL_WIDTH_FT = 40
# OSM-only fallback creek buffers (assumptions; the official run uses rule R1 instead)
STREAM_SETBACK_FT = {"stream": 100, "canal": 50, "river": 150}

REDEVELOPABLE = {  # tag: values that count as land available for the new development
    "amenity": {"parking", "marketplace"},
    "landuse": {"brownfield", "greenfield", "construction", "vacant"},
}
PARK_TAGS = {"leisure": {"park", "playground", "pitch", "garden", "sports_centre"},
             "landuse": {"recreation_ground", "village_green"}}
ESSENTIAL_TAGS = {"amenity": {"school", "college", "hospital", "clinic", "library", "community_centre",
                              "fire_station", "police", "kindergarten"}}

# Cell classes in the constraint grid
C_UNBUILDABLE, C_OUTSIDE, C_SITE, C_EX_RES, C_EX_PARK, C_EX_ESS = range(6)
CLASS_NAMES = {C_UNBUILDABLE: "Road / rail / creek setback / floodway", C_OUTSIDE: "Outside site (unchanged)",
               C_SITE: "Redevelopable site", C_EX_RES: "Existing housing",
               C_EX_PARK: "Existing park", C_EX_ESS: "Existing school / public service"}

LAND_USE_NAMES = {0: "Residential", 1: "Parks & Open Space", 2: "Commercial / Mixed Use",
                  3: "Light Industrial", 4: "Public & Institutional"}
# Planning-map colors (APA land-based classification convention, softened for imagery)
LAND_USE_COLORS = {0: "#F2D95C", 1: "#5DAE4B", 2: "#D9534F", 3: "#8E6BB8", 4: "#3F7FCF"}
# highlighted Pareto solutions: key, map title, objective column it is best on (None = balanced)
REPS = (("balanced", "Balanced Solution", None), ("best_job_commute", "Best Job Access", 0),
        ("best_livability", "Best Livability", 2), ("best_transit_access", "Most Homes Near BART", 3))
FIVE_MIN_FT, FIFTEEN_MIN_FT = 1320, 3960


# -----------------------------------------------------------------------------
# OpenStreetMap -> GeoDataFrames
# -----------------------------------------------------------------------------

def _matches(tags, spec):
    return any(tags.get(k) in vals for k, vals in spec.items())


def _way_geom(el):
    pts = [(p["lon"], p["lat"]) for p in el.get("geometry", []) if p]
    if len(pts) < 2:
        return None
    if len(pts) >= 4 and pts[0] == pts[-1]:
        return Polygon(pts)
    return LineString(pts)


def _relation_polys(el):
    outers = []
    for m in el.get("members", []):
        if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and m.get("geometry"):
            pts = [(p["lon"], p["lat"]) for p in m["geometry"]]
            if len(pts) >= 4 and pts[0] == pts[-1]:
                outers.append(Polygon(pts))
    return unary_union(outers) if outers else None


def load_osm(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way":
            geom = _way_geom(el)
        elif el["type"] == "relation":
            geom = _relation_polys(el)
        elif el["type"] == "node":
            geom = Point(el["lon"], el["lat"])
        else:
            geom = None
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        rows.append({"osm_id": f"{el['type']}/{el['id']}", "name": tags.get("name"),
                     "tags": tags, "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(CRS_OUT)
    site = data.get("site", {})
    site["osm_base"] = (data.get("osm3s") or {}).get("timestamp_osm_base", "")
    return gdf, site


def categorize(osm, official_creeks=False):
    """Split raw OSM features into the layers the grid classification needs (all in feet).
    With official_creeks=True the OSM waterway buffers are skipped (rule R1 replaces them)."""
    cats = {k: [] for k in ("roads", "rail", "water", "parks", "essential", "residential",
                            "redevelopable", "buildings", "stations")}
    road_lines, rail_lines, water_lines = [], [], []
    for _, r in osm.iterrows():
        t, g = r.tags, r.geometry
        hw = t.get("highway")
        underground = t.get("tunnel") in ("yes", "culvert") or str(t.get("layer", "0")).startswith("-")
        if hw in ROAD_WIDTH_FT and g.geom_type == "LineString" and not underground:
            cats["roads"].append(g.buffer(ROAD_WIDTH_FT[hw] / 2, cap_style="flat"))
            road_lines.append({"name": r["name"], "class": hw, "geometry": g})
        if t.get("railway") in ("rail", "subway", "light_rail") and g.geom_type == "LineString" and not underground:
            cats["rail"].append(g.buffer(RAIL_WIDTH_FT / 2, cap_style="flat"))
            rail_lines.append(g)
        if t.get("waterway") in STREAM_SETBACK_FT and g.geom_type == "LineString" and t.get("tunnel") != "culvert":
            water_lines.append({"name": r["name"], "waterway": t["waterway"], "geometry": g})
            if not official_creeks:
                cats["water"].append(g.buffer(STREAM_SETBACK_FT[t["waterway"]]))
        if t.get("natural") == "water" and g.geom_type in ("Polygon", "MultiPolygon"):
            cats["water"].append(g)
        if g.geom_type in ("Polygon", "MultiPolygon"):
            if _matches(t, PARK_TAGS):
                cats["parks"].append(g)
            if _matches(t, ESSENTIAL_TAGS):
                cats["essential"].append(g)
            if t.get("landuse") == "residential":
                cats["residential"].append(g)
            if _matches(t, REDEVELOPABLE):
                cats["redevelopable"].append(g)
            if "building" in t:
                cats["buildings"].append(g)
        elif g.geom_type == "Point" and _matches(t, ESSENTIAL_TAGS):
            cats["essential"].append(g.buffer(100))
        if t.get("railway") == "station" or t.get("public_transport") == "station":
            cats["stations"].append({"name": r["name"], "geometry": g.centroid})
    merged = {k: (unary_union(v) if v and k not in ("stations",) else (v if k == "stations" else None))
              for k, v in cats.items()}
    merged["road_lines"] = road_lines
    merged["rail_lines"] = unary_union(rail_lines) if rail_lines else None
    merged["osm_waterways"] = water_lines
    merged["building_list"] = cats["buildings"]
    return merged


# -----------------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------------

class SiteGrid:
    """nr x nc cells of cs feet; row 0 is the north edge."""

    def __init__(self, x0, y_top, nr, nc, cell_ft):
        self.x0, self.y_top, self.nr, self.nc, self.cs = x0, y_top, nr, nc, cell_ft

    @classmethod
    def square(cls, center_xy, n, cell_ft):
        cx, cy = center_xy
        return cls(cx - n * cell_ft / 2, cy + n * cell_ft / 2, n, n, cell_ft)

    @classmethod
    def around(cls, geom, margin_ft, cell_ft):
        minx, miny, maxx, maxy = geom.bounds
        x0 = np.floor((minx - margin_ft) / cell_ft) * cell_ft
        y0 = np.floor((miny - margin_ft) / cell_ft) * cell_ft
        nc = int(np.ceil((maxx + margin_ft - x0) / cell_ft))
        nr = int(np.ceil((maxy + margin_ft - y0) / cell_ft))
        return cls(float(x0), float(y0 + nr * cell_ft), nr, nc, cell_ft)

    @property
    def shape(self):
        return (self.nr, self.nc)

    def cell(self, r, c):
        x = self.x0 + c * self.cs
        y = self.y_top - r * self.cs
        return box(x, y - self.cs, x + self.cs, y)

    def centroid(self, r, c):
        return (self.x0 + (c + 0.5) * self.cs, self.y_top - (r + 0.5) * self.cs)

    def extent(self):
        return box(self.x0, self.y_top - self.nr * self.cs, self.x0 + self.nc * self.cs, self.y_top)

    def cells(self):
        return [self.cell(r, c) for r in range(self.nr) for c in range(self.nc)]


def coverage_grid(grid, geom):
    """Share of each cell covered by geom (exact areas; parts indexed with an STRtree for speed)."""
    out = np.zeros(grid.shape)
    if geom is None or geom.is_empty:
        return out
    parts = [p for p in getattr(geom, "geoms", [geom]) if not p.is_empty]
    tree = STRtree(parts)
    area = grid.cs * grid.cs
    for r in range(grid.nr):
        for c in range(grid.nc):
            cell = grid.cell(r, c)
            hits = tree.query(cell)
            if len(hits):
                out[r, c] = sum(cell.intersection(parts[i]).area for i in hits) / area
    return out


# -----------------------------------------------------------------------------
# Official data: redevelopable parcels
# -----------------------------------------------------------------------------

def redevelopable_parcels(uv, cats):
    """Parcels inside the Urban Village that are available for change: surface parking, the flea
    market, vacant land, or lightly built (OSM building coverage under 30%), and not existing homes,
    parks or schools."""
    parcels = od.load("parcels")
    if parcels is None or uv is None:
        return None, None
    p = parcels[parcels.geometry.representative_point().within(uv)].copy()
    p["geometry"] = p.geometry.intersection(uv)
    p = p[~p.geometry.is_empty & (p.geometry.area > 100)]
    b_list = cats["building_list"]
    b_tree = STRtree(b_list) if b_list else None
    keep_out = unary_union([g for g in (cats["residential"], cats["parks"], cats["essential"]) if g is not None])
    redev_tags = cats["redevelopable"]
    cov, tag_share, existing_share, why = [], [], [], []
    for g in p.geometry:
        a = g.area
        bc = sum(g.intersection(b_list[i]).area for i in b_tree.query(g)) / a if b_tree is not None else 0.0
        ts = g.intersection(redev_tags).area / a if redev_tags is not None else 0.0
        es = g.intersection(keep_out).area / a if keep_out is not None and not keep_out.is_empty else 0.0
        cov.append(round(bc, 3))
        tag_share.append(round(ts, 3))
        existing_share.append(round(es, 3))
    p["bldg_coverage"] = cov
    p["parking_market_share"] = tag_share
    p["existing_use_share"] = existing_share
    ok = (p.existing_use_share < 0.5) & ((p.parking_market_share >= 0.5) | (p.bldg_coverage < REDEV_MAX_COVERAGE))
    p["redevelopable"] = ok.astype(int)
    p["reason"] = np.where(p.existing_use_share >= 0.5, "existing homes / park / school",
                           np.where(p.parking_market_share >= 0.5, "surface parking / flea market",
                                    np.where(p.bldg_coverage < REDEV_MAX_COVERAGE,
                                             "lightly built (< 30% building coverage)", "built out")))
    return p, unary_union(list(p[ok].geometry)) if ok.any() else None


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------

def classify(grid, cats, extra=None):
    """extra (official mode): blocked, parks, essential, redevelopable geometries + rule areas."""
    extra = extra or {}
    blocked = unary_union([g for g in (cats["roads"], cats["rail"], cats["water"], extra.get("blocked")) if g])
    ess = unary_union([g for g in (cats["essential"], extra.get("essential")) if g])
    parks = unary_union([g for g in (cats["parks"], extra.get("parks")) if g])
    redev = extra.get("redevelopable", cats["redevelopable"]) if "redevelopable" in extra else cats["redevelopable"]
    cov = {"blocked": coverage_grid(grid, blocked), "ess": coverage_grid(grid, ess),
           "park": coverage_grid(grid, parks), "site": coverage_grid(grid, redev),
           "res": coverage_grid(grid, cats["residential"])}
    cls = np.full(grid.shape, C_OUTSIDE, dtype=int)
    cls[cov["res"] > 0.5] = C_EX_RES
    cls[cov["site"] > 0.5] = C_SITE
    cls[cov["park"] > 0.5] = C_EX_PARK
    cls[cov["ess"] > 0.5] = C_EX_ESS
    cls[cov["blocked"] > 0.35] = C_UNBUILDABLE
    return cls, cov


# Manual edits a technician digitizes in QGIS (site_edits.gpkg, layer "overrides")
EDIT_CLASSES = {"unbuildable": C_UNBUILDABLE, "site": C_SITE, "existing_housing": C_EX_RES,
                "existing_park": C_EX_PARK, "existing_public": C_EX_ESS, "outside": C_OUTSIDE}


def apply_overrides(cls, grid, path):
    """Cells covered >50% by an override polygon take that polygon's force_class."""
    edited = np.zeros(cls.shape, dtype=bool)
    if not os.path.exists(path):
        return cls, edited
    try:
        ov = gpd.read_file(path, layer="overrides")
    except Exception:
        return cls, edited
    if ov.empty:
        return cls, edited
    ov = ov.to_crs(CRS_OUT) if ov.crs else ov.set_crs(CRS_OUT)
    cls = cls.copy()
    for _, o in ov.iterrows():
        g, fc = o.geometry, o.get("force_class")
        if g is None or g.is_empty or not g.is_valid or fc not in EDIT_CLASSES:
            continue          # gis_qaqc.py reports these as EDIT_INVALID / EDIT_INCOMPLETE
        minx, miny, maxx, maxy = g.bounds
        for r in range(grid.nr):
            for c in range(grid.nc):
                cell = grid.cell(r, c)
                cx0, cy0, cx1, cy1 = cell.bounds
                if cx1 < minx or cx0 > maxx or cy1 < miny or cy0 > maxy:
                    continue
                if cell.intersection(g).area / cell.area > 0.5 and cls[r, c] != EDIT_CLASSES[fc]:
                    cls[r, c] = EDIT_CLASSES[fc]
                    edited[r, c] = True
    return cls, edited


def base_grid_from_classes(cls):
    """Translate cell classes into the optimizer's land use codes."""
    base = np.full(cls.shape, -1, dtype=int)
    base[cls == C_SITE] = -2          # empty: optimizer decides
    base[cls == C_EX_RES] = 0
    base[cls == C_EX_PARK] = 1
    base[cls == C_EX_ESS] = 4
    return base


# -----------------------------------------------------------------------------
# Optimizer: Kyler's CityBuilder, starting from the real map
# -----------------------------------------------------------------------------

def grow_building_rect(opt, grid, anchor_row, anchor_col, target_size, growth_genes, land_type):
    """Kyler's grow_building (Nsga3 Z.py), unchanged except that it accepts a rectangular grid."""
    nr, nc = grid.shape
    if anchor_row < 0 or anchor_row >= nr or anchor_col < 0 or anchor_col >= nc:
        return set()
    if grid[anchor_row, anchor_col] != -2:
        return set()
    D = opt.DIRECTIONS
    cells = {(anchor_row, anchor_col)}
    frontier = [(anchor_row, anchor_col)]
    gene_idx = 0
    while len(cells) < target_size and frontier and gene_idx < len(growth_genes):
        direction = int(growth_genes[gene_idx]) % 4
        gene_idx += 1
        grew = False
        for fr, fc in list(frontier):
            dr, dc = D[direction]
            nr_, nc_ = fr + dr, fc + dc
            if 0 <= nr_ < nr and 0 <= nc_ < nc and grid[nr_, nc_] == -2 and (nr_, nc_) not in cells:
                cells.add((nr_, nc_))
                frontier.append((nr_, nc_))
                grew = True
                break
        if not grew:
            for alt_dir in range(4):
                if alt_dir == direction:
                    continue
                for fr, fc in list(frontier):
                    dr, dc = D[alt_dir]
                    nr_, nc_ = fr + dr, fc + dc
                    if 0 <= nr_ < nr and 0 <= nc_ < nc and grid[nr_, nc_] == -2 and (nr_, nc_) not in cells:
                        cells.add((nr_, nc_))
                        frontier.append((nr_, nc_))
                        grew = True
                        break
                if grew:
                    break
        new_frontier = []
        for fr, fc in frontier:
            for dr, dc in D:
                nr_, nc_ = fr + dr, fc + dc
                if 0 <= nr_ < nr and 0 <= nc_ < nc and grid[nr_, nc_] == -2 and (nr_, nc_) not in cells:
                    new_frontier.append((fr, fc))
                    break
        frontier = new_frontier if new_frontier else frontier[-3:]
    if len(cells) >= target_size:
        for r, c in list(cells)[:target_size]:
            grid[r, c] = land_type
        return set(list(cells)[:target_size])
    return set()


PLACE_ORDER = (3, 2, 4)        # industrial, commercial, essential: the capstone's order


def make_site_builder(opt):
    class SiteCityBuilder(opt.CityBuilder):
        """Same decoding as CityBuilder.build, except:
        - the grid starts from the real-site base grid (fixed cells already filled in), and may be rectangular
        - building anchors snap to the nearest site cell that is still empty (trying the next nearest
          ones when a building can't grow to full size there), so genes still map to legal spots
        - cells where homes aren't allowed (rules R3/R4) are filled with open space first
        """

        def __init__(self, base_grid, *args, no_res=None, **kwargs):
            super().__init__(max(base_grid.shape), *args, **kwargs)
            self.base_grid = base_grid
            self.nr, self.nc = base_grid.shape
            self.encoding_length = (self.n_industrial * self.ind_genes + self.n_commercial * self.com_genes +
                                    self.n_essential * self.ess_genes + self.nr * self.nc)
            if not (base_grid == -2).any():
                raise ValueError("No redevelopable cells on this site.")
            self.no_res = np.zeros(base_grid.shape, dtype=bool) if no_res is None else no_res & (base_grid == -2)

        def build(self, encoding):
            grid = self.base_grid.copy()
            buildings, collisions, idx = [], 0, 0
            kinds = {3: (self.n_industrial, self.ind_genes, self.industrial_size, 3),
                     2: (self.n_commercial, self.com_genes, self.commercial_size, 2),
                     4: (self.n_essential, self.ess_genes, self.essential_size, 4)}
            for n_b, genes, size, land_type in [kinds[k] for k in PLACE_ORDER]:
                for _ in range(n_b):
                    r0 = int(encoding[idx]) % self.nr
                    c0 = int(encoding[idx + 1]) % self.nc
                    growth = encoding[idx + 2:idx + genes]
                    idx += genes
                    cells = set()
                    free = np.argwhere(grid == -2)
                    if len(free):
                        d = np.abs(free[:, 0] - r0) + np.abs(free[:, 1] - c0)
                        for k in np.argsort(d, kind="stable"):     # nearest first, ties in grid order
                            cells = grow_building_rect(opt, grid, int(free[k, 0]), int(free[k, 1]), size, growth,
                                                       land_type)
                            if cells:
                                break
                    if cells and len(cells) == size:
                        buildings.append((land_type, cells))
                    else:
                        collisions += 1
            flat = grid.ravel()
            empty = np.flatnonzero(flat == -2)
            n_green = int(len(empty) * self.green_ratio)
            pri = np.asarray(encoding[idx:idx + self.nr * self.nc], dtype=float)[empty]
            forced = self.no_res.ravel()[empty]
            # same ordering as the capstone (highest priority gene first, ties keep grid order),
            # except cells where homes aren't allowed take the open-space slots first
            order = np.lexsort((np.arange(len(empty)), -pri, ~forced))
            fill = np.zeros(len(empty), dtype=int)
            fill[order[:n_green]] = 1
            fill[forced] = 1
            flat[empty] = fill
            return grid, buildings, collisions

    return SiteCityBuilder


def comp_by_cv_then_random_seeded(pop, P, random_state=None, **kwargs):
    """NSGA-III's tournament rule (fewer constraint violations wins, otherwise a coin flip),
    rewritten so the coin flip uses the run's seeded generator. pymoo 0.6.2's built-in version
    calls an inner helper without the generator, which makes every run different."""
    cv = pop.get("CV").ravel()
    S = np.empty(P.shape[0], dtype=int)
    for i, (a, b) in enumerate(P[:, :2]):
        if (cv[a] > 0 or cv[b] > 0) and cv[a] != cv[b]:
            S[i] = a if cv[a] < cv[b] else b
        else:
            S[i] = a if random_state.random() < 0.5 else b
    return S[:, None]


class Objectives:
    """F1-F4 for a decoded grid. distance='network' uses walking distances along streets and paths
    (walk_network.py); 'grid' uses the capstone's taxicab cell distances. Both report feet.
    base: the base grid (-2 = open site cell); bart_ft: walk from every cell to BART (network mode);
    bart_cells: (row, col) of the BART platforms (grid mode)."""

    def __init__(self, opt, cell_ft, base, distance="grid", D=None, ridx=None, bart_ft=None, bart_cells=None):
        self.opt, self.cs, self.mode, self.base = opt, cell_ft, distance, base
        self.D, self.ridx, self.bart_ft = D, ridx, bart_ft
        self.bart_cells = np.asarray(bart_cells if bart_cells is not None else [], dtype=float).reshape(-1, 2)

    def _net(self, g, mask_to):
        res = self.ridx[(g == 0).ravel()]
        dst = self.ridx[mask_to.ravel()]
        if len(res) == 0 or len(dst) == 0:
            return 1e5
        return float(self.D[np.ix_(res, dst)].min(axis=1).mean())

    def job(self, g, mode=None):
        if (mode or self.mode) == "network":
            return self._net(g, (g == 2) | (g == 3))
        return self.opt.calc_job_commute(g) * self.cs

    def essential(self, g, mode=None):
        if (mode or self.mode) == "network":
            return self._net(g, g == 4)
        return self.opt.calc_essential_commute(g) * self.cs

    def bart(self, g, mode=None):
        """Average distance from each NEW home (a residential cell on the redevelopment site) to BART."""
        new_res = (g == 0) & (self.base == -2)
        if not new_res.any():
            return 1e5
        if (mode or self.mode) == "network" and self.bart_ft is not None:
            return float(self.bart_ft[new_res].mean())
        if not len(self.bart_cells):
            return 0.0
        rc = np.argwhere(new_res)
        d = np.abs(rc[:, None, :] - self.bart_cells[None, :, :]).sum(-1).min(axis=1)
        return float(d.mean() * self.cs)

    @staticmethod
    def near_industry(g, cells=site_rules.SEP_SOFT_FT // 120):
        ind = g == 3
        if not ind.any():
            return 0
        d = nearest_taxicab(g, g == 0, ind)
        return int(np.nansum(d <= cells))

    def __call__(self, g):
        return [self.job(g), self.essential(g), -self.opt.calc_livability(g), self.bart(g)]


class SiteProblem(Problem):
    def __init__(self, builder, objectives, xu):
        self.builder, self.objectives = builder, objectives
        super().__init__(n_var=builder.encoding_length, n_obj=4, n_ieq_constr=2, xl=0, xu=xu, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), 4))
        G = np.zeros((len(X), 2))
        for i, x in enumerate(X):
            g, _, coll = self.builder.build(x)
            F[i] = self.objectives(g)
            G[i] = (coll, self.builder.opt_calc_adjacent(g))      # G2 = rule R5
        out["F"], out["G"] = F, G


def program_config(opt, base, plan, program, no_industrial=False, green_ratio=None):
    """How many buildings of each kind the optimizer places on the open cells.
    city:     the same amount of each use as the City's plan on those cells (4-cell blocks)
    capstone: the capstone's proportions (compute_scaled_config), which include light industry"""
    open_ = base == -2
    n_open = int(open_.sum())
    if program == "city" and plan is not None:
        count = {k: int((plan[open_] == k).sum()) for k in range(5)}
        cfg = {"n_industrial": int(round(count[3] / 8)), "n_commercial": int(round(count[2] / 4)),
               "n_essential": int(round(count[4] / 4)), "industrial_size": 8, "commercial_size": 4,
               "essential_size": 4}
        building = 8 * cfg["n_industrial"] + 4 * cfg["n_commercial"] + 4 * cfg["n_essential"]
        cfg["green_ratio"] = count[1] / max(1, n_open - building)
    else:
        equiv = max(5, int(round(np.sqrt(n_open))))     # scale building counts to the open area
        cfg = {k: v for k, v in opt.compute_scaled_config(equiv).items()
               if k in ("n_industrial", "n_commercial", "n_essential",
                        "industrial_size", "commercial_size", "essential_size", "green_ratio")}
    if no_industrial:
        cfg["n_industrial"] = 0
    if green_ratio is not None:
        cfg["green_ratio"] = green_ratio
    return cfg


def run_optimizer(opt, base, n_gen, pop, seed, cfg, objectives, no_res, history=False):
    Builder = make_site_builder(opt)
    builder = Builder(base, cfg["n_industrial"], cfg["n_commercial"], cfg["n_essential"],
                      industrial_size=cfg["industrial_size"], commercial_size=cfg["commercial_size"],
                      essential_size=cfg["essential_size"], green_ratio=cfg["green_ratio"], no_res=no_res)
    builder.opt_calc_adjacent = opt.calc_industrial_adjacent
    problem = SiteProblem(builder, objectives, xu=max(max(base.shape) - 1, 3))
    algorithm = NSGA3(ref_dirs=get_reference_directions("das-dennis", 4, n_partitions=6), pop_size=pop,
                      selection=TournamentSelection(func_comp=comp_by_cv_then_random_seeded),
                      sampling=opt.IntegerSampling(), crossover=opt.IntegerCrossover(prob=0.9),
                      mutation=opt.IntegerMutation(prob=0.2, eta=4))
    # pymoo's `seed` only seeds its own generator; Kyler's integer operators draw from NumPy's
    # global generator, so seed that too or two runs with the same seed give different layouts
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()
    tracker = FrontTracker() if history else None
    extra = {"callback": tracker} if tracker else {}
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False, **extra)
    if res.X is None:
        raise RuntimeError("No feasible layout found; try more generations or relax a rule.")
    print(f"NSGA-III: {len(np.atleast_2d(res.X))} Pareto solutions, {n_gen} generations, "
          f"{time.perf_counter() - t0:.0f}s")
    return res, problem, cfg, (tracker.fronts if tracker else None)


class FrontTracker:
    """Keeps a copy of the feasible non-dominated objective vectors after every generation, so the
    convergence (hypervolume) can be measured afterwards. Reads the population only; changes nothing."""

    def __init__(self):
        self.fronts = []

    def __call__(self, algorithm):
        opt_ = algorithm.opt
        F = opt_.get("F")
        cv = opt_.get("CV")
        if F is None:
            self.fronts.append(np.zeros((0, 4)))
            return
        ok = (cv.ravel() <= 0) if cv is not None else np.ones(len(F), dtype=bool)
        self.fronts.append(np.array(F[ok], dtype=float))


def pick_representatives(F):
    """Balanced = closest to the best value on all four objectives at once (normalized distance to the
    ideal point), plus the best solution on job access, livability and walk to BART."""
    f_min, f_max = F.min(0), F.max(0)
    F_norm = (F - f_min) / (f_max - f_min + 1e-12)
    out = {}
    for key, _, col in REPS:
        out[key] = int(np.argmin(np.linalg.norm(F_norm, axis=1))) if col is None else int(np.argmin(F[:, col]))
    return out


# -----------------------------------------------------------------------------
# Walking network
# -----------------------------------------------------------------------------

def build_walk(grid, cls, cats, rules, stations):
    """Cell-to-cell walking distances for every cell that can hold homes, jobs or services, plus
    walk distance from every cell to BART, groceries, parks and schools. Returns None without data."""
    import walk_network as wn
    from scipy.sparse.csgraph import connected_components, dijkstra
    osm = od.load_walk_osm()
    if osm is None:
        return None
    t0 = time.perf_counter()
    G = wn.WalkGraph(osm, CRS_OUT)
    # only the largest connected piece of the network takes connectors (private loops stay out)
    csr0 = G.matrix()
    _, lab = connected_components(csr0, directed=False)
    main = np.bincount(lab[:G.n_osm]).argmax()
    keep = lab[:G.n_osm] == main
    relevant = np.isin(cls, (C_SITE, C_EX_RES, C_EX_PARK, C_EX_ESS))
    rc = np.argwhere(relevant)
    pts = [grid.centroid(r, c) for r, c in rc]
    cell_nodes = G.add_nodes(pts)
    node_of = -np.ones(grid.shape, dtype=int)
    node_of[rc[:, 0], rc[:, 1]] = cell_nodes
    # future walkable blocks inside the redevelopment area: neighbouring site cells are linked
    site = cls == C_SITE
    pairs = []
    for dr, dc in ((0, 1), (1, 0)):
        a = site[:grid.nr - dr, :grid.nc - dc] & site[dr:, dc:]
        for r, c in np.argwhere(a):
            pairs.append((node_of[r, c], node_of[r + dr, c + dc]))
    G.link(pairs, grid.cs)
    barrier = unary_union([g for g in (rules.get("creek_lines"), cats.get("rail_lines")) if g is not None])
    # connectors only to the main network piece
    saved_xy = G.xy
    main_idx = np.flatnonzero(keep)
    G_conn = wn.WalkGraph.__new__(wn.WalkGraph)
    G_conn.xy, G_conn.n_osm, G_conn.extra = saved_xy[main_idx], len(main_idx), []
    G_conn.xy = np.vstack([G_conn.xy, saved_xy[cell_nodes]])
    local_cells = np.arange(len(main_idx), len(main_idx) + len(cell_nodes))
    joined = G_conn.connect(local_cells, barrier=barrier)
    for a, b, d in G_conn.extra:
        G.extra.append((int(cell_nodes[a - len(main_idx)]), int(main_idx[b]), d))
    unjoined = len(cell_nodes) - joined
    if unjoined:            # no barrier-free connector within 800 ft: join to the nearest node anyway
        done = {e[0] for e in G.extra}
        tree_pts = saved_xy[main_idx]
        from scipy.spatial import cKDTree
        t = cKDTree(tree_pts)
        for nd in cell_nodes:
            if nd not in done:
                d, j = t.query(saved_xy[nd])
                G.extra.append((int(nd), int(main_idx[j]), float(d)))
    csr = G.matrix()
    # cell-to-cell distances, in chunks to keep memory small
    # distances are rounded to 0.1 ft: last-bit differences in map projection or square-root math
    # between computers (Windows vs Linux) then can't change which layout the optimizer prefers
    D = np.empty((len(cell_nodes), len(cell_nodes)), dtype=np.float32)
    for i in range(0, len(cell_nodes), 200):
        D[i:i + 200] = np.round(dijkstra(csr, indices=cell_nodes[i:i + 200])[:, cell_nodes], 1)
    ridx = -np.ones(grid.nr * grid.nc, dtype=int)
    ridx[(rc[:, 0] * grid.nc + rc[:, 1])] = np.arange(len(rc))

    # destinations
    dest = {}
    bart = od.bart_stops()
    bart_pts = [(p.x, p.y) for p in bart.geometry] if bart is not None and len(bart) else \
        [(s["geometry"].x, s["geometry"].y) for s in stations]
    dest["bart"] = G.nearest_osm_nodes(bart_pts)
    groceries = []
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", CRS_OUT, always_xy=True)
    for el in osm["elements"]:
        t = el.get("tags", {})
        if t.get("shop") in ("supermarket", "grocery", "greengrocer"):
            if el["type"] == "node":
                groceries.append(tf.transform(el["lon"], el["lat"]))
            elif "bounds" in el:
                b = el["bounds"]
                groceries.append(tf.transform((b["minlon"] + b["maxlon"]) / 2, (b["minlat"] + b["maxlat"]) / 2))
    dest["grocery"] = G.nearest_osm_nodes(groceries)
    parks = od.open_parks()
    dest["park"] = wn.nodes_near_polygons(G, list(parks.geometry) if parks is not None else [])
    schools = od.open_schools()
    dest["school"] = wn.nodes_near_polygons(G, list(schools.geometry) if schools is not None else [])
    to_dest = {}
    for k, nodes in dest.items():
        dist = dijkstra(csr, indices=sorted(set(nodes)), min_only=True) if nodes else np.full(len(G.xy), np.inf)
        to_dest[k] = dist
    cell_to = {}
    allc = np.argwhere(np.ones(grid.shape, dtype=bool))
    # every cell (not only relevant ones) gets walk times for the access map, via its nearest node
    from scipy.spatial import cKDTree
    t = cKDTree(saved_xy[main_idx])
    dd, jj = t.query([grid.centroid(r, c) for r, c in allc])
    for k, dist in to_dest.items():
        v = np.full(grid.shape, np.nan)
        v[allc[:, 0], allc[:, 1]] = dist[main_idx[jj]] + dd
        rel = node_of >= 0
        v[rel] = dist[node_of[rel]]
        cell_to[k] = np.round(v, 1)
    bands, _ = wn.service_areas(G, csr, sorted(set(dest["bart"])), CRS_OUT)
    edges = gpd.GeoDataFrame({"geometry": G.lines}, crs=CRS_OUT)
    pts_dest = []
    for k, nodes in dest.items():
        for nd in sorted(set(nodes)):
            pts_dest.append({"type": k, "geometry": Point(G.xy[nd])})
    print(f"Walk network: {G.n_osm:,} nodes, {len(G.edges):,} segments, {G.snapped} dangles joined; "
          f"{len(cell_nodes)} cells connected ({unjoined} without a barrier-free link); "
          f"{time.perf_counter() - t0:.0f}s")
    return {"D": D, "ridx": ridx, "cell_to": cell_to, "bands": bands, "edges": edges,
            "dest_points": gpd.GeoDataFrame(pts_dest, geometry="geometry", crs=CRS_OUT),
            "stats": {"nodes": G.n_osm, "segments": len(G.edges), "dangles_joined": G.snapped,
                      "cells": len(cell_nodes), "unjoined": unjoined}}


# -----------------------------------------------------------------------------
# Metrics shared by the optimized layouts and the City's plan
# -----------------------------------------------------------------------------

def layout_metrics(g, base, objectives, walk, masks, cs):
    new = base == -2
    res = g == 0
    new_res = res & new
    row = {
        "job_walk_ft": round(objectives.job(g, "network"), 1) if walk else None,
        "services_walk_ft": round(objectives.essential(g, "network"), 1) if walk else None,
        "job_grid_ft": round(objectives.job(g, "grid"), 1),
        "services_grid_ft": round(objectives.essential(g, "grid"), 1),
        "livability": round(objectives.opt.calc_livability(g), 2),
        "homes_near_industry": Objectives.near_industry(g),
        "industrial_adjacent": int(objectives.opt.calc_industrial_adjacent(g)),
        "new_home_cells": int(new_res.sum()),
        "new_home_acres": round(new_res.sum() * cs * cs / 43560, 1),
        "job_cells": int(((g == 2) | (g == 3))[new].sum()),
        "park_cells_new": int((g == 1)[new].sum()),
        "public_cells_new": int((g == 4)[new].sum()),
        "new_homes_in_noise_contour": int((new_res & masks["noise"]).sum()),
        "new_homes_in_flood_zone": int((new_res & masks["sfha"]).sum()),
        "new_homes_in_flood_x": int((new_res & masks["flood_x"]).sum()),
        "new_homes_grid_to_bart_ft": round(objectives.bart(g, "grid"), 1),
    }
    if walk:
        ct = walk["cell_to"]
        row["new_homes_walk_to_bart_ft"] = round(objectives.bart(g, "network"), 1)
        row["new_homes_5min_bart"] = int((new_res & (ct["bart"] <= FIVE_MIN_FT)).sum())
        row["new_homes_10min_bart"] = int((new_res & (ct["bart"] <= TEN_MIN_FT)).sum())
        row["new_homes_beyond_10min_bart"] = int(new_res.sum()) - row["new_homes_10min_bart"]
        row["jobs_10min_bart"] = int((((g == 2) | (g == 3)) & new & (ct["bart"] <= TEN_MIN_FT)).sum())
        nres = max(1, int(res.sum()))
        row["pct_homes_10min_grocery"] = round(100 * (res & (ct["grocery"] <= TEN_MIN_FT)).sum() / nres, 1)
        row["pct_homes_10min_school"] = round(100 * (res & (ct["school"] <= TEN_MIN_FT)).sum() / nres, 1)
        # parks: existing official parks, or a new park cell within a 10-minute walk
        park_ok = res & (ct["park"] <= TEN_MIN_FT)
        newpark = (g == 1) & new
        if newpark.any() and res.any():
            r_res = walk["ridx"][res.ravel()]
            r_pk = walk["ridx"][newpark.ravel()]
            near = walk["D"][np.ix_(r_res, r_pk)].min(axis=1) <= TEN_MIN_FT
            flat = park_ok.ravel().copy()
            flat[np.flatnonzero(res.ravel())] |= near
            park_ok = flat.reshape(g.shape)
        row["pct_homes_10min_park"] = round(100 * park_ok.sum() / nres, 1)
    return row


# -----------------------------------------------------------------------------
# The City's adopted plan
# -----------------------------------------------------------------------------

def city_plan_grid(grid, base, uv, uv_as=2):
    """General Plan 2040 designations inside the Urban Village, on the same 120 ft cells.
    Every redevelopable cell takes the designation covering most of it; fixed cells stay as they are.
    uv_as: model code for the generic 'Urban Village' (mixed use) designation (2, or 0 for a housing reading)."""
    gp = od.load("general_plan")
    if gp is None or uv is None:
        return None, None
    gp = gp[gp.geometry.intersects(uv)].copy()
    lut = {k: v[0] for k, v in od.GP_TO_LU.items()}
    lut["UV"] = uv_as
    plan = base.copy()
    desig = np.full(base.shape, "", dtype=object)
    tree = STRtree(list(gp.geometry))
    abbr = list(gp["GPABBREVIATION"])
    for r, c in np.argwhere(base == -2):
        cell = grid.cell(r, c)
        hits = tree.query(cell)
        best, best_a = None, 0.0
        for i in hits:
            a = cell.intersection(gp.geometry.iloc[i]).area
            if a > best_a:
                best, best_a = abbr[i], a
        if best is None:                      # cell on a street gap: nearest designation
            d = gp.geometry.distance(Point(grid.centroid(r, c)))
            best = abbr[int(np.argmin(d.values))]
        code = lut.get(best, 2)
        plan[r, c] = 4 if code == -1 else code          # TU (transit facility) -> public / institutional
        desig[r, c] = best
    return plan, desig


# -----------------------------------------------------------------------------
# Export helpers (QML styles, GeoPackage)
# -----------------------------------------------------------------------------

def _rgba(h, a=255):
    h = h.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a}"


def _fill(name, fill, outline, width, style="solid", ostyle="solid"):
    return (f'<symbol type="fill" name="{name}" alpha="1" clip_to_extent="1" force_rhr="0">'
            f'<layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="color" type="QString" value="{fill}"/>'
            f'<Option name="outline_color" type="QString" value="{outline}"/>'
            f'<Option name="outline_style" type="QString" value="{ostyle}"/>'
            f'<Option name="outline_width" type="QString" value="{width}"/>'
            f'<Option name="outline_width_unit" type="QString" value="MM"/>'
            f'<Option name="joinstyle" type="QString" value="miter"/>'
            f'<Option name="style" type="QString" value="{style}"/></Option></layer></symbol>')


def _line(name, color, width, style="solid"):
    return (f'<symbol type="line" name="{name}" alpha="1" clip_to_extent="1" force_rhr="0">'
            f'<layer class="SimpleLine" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="line_color" type="QString" value="{color}"/>'
            f'<Option name="line_width" type="QString" value="{width}"/>'
            f'<Option name="line_width_unit" type="QString" value="MM"/>'
            f'<Option name="line_style" type="QString" value="{style}"/>'
            f'<Option name="capstyle" type="QString" value="round"/>'
            f'</Option></layer></symbol>')


def _marker(name, color, size, shape="circle", outline="255,255,255,255"):
    return (f'<symbol type="marker" name="{name}" alpha="1" clip_to_extent="1" force_rhr="0">'
            '<layer class="SimpleMarker" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="name" type="QString" value="{shape}"/>'
            f'<Option name="color" type="QString" value="{color}"/>'
            f'<Option name="outline_color" type="QString" value="{outline}"/>'
            '<Option name="outline_width" type="QString" value="0.4"/>'
            f'<Option name="size" type="QString" value="{size}"/>'
            '<Option name="size_unit" type="QString" value="MM"/></Option></layer></symbol>')


def _qml(renderer, opacity=1.0):
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
            f'<qgis version="3.34.0" styleCategories="AllStyleCategories">{renderer}'
            f"<layerOpacity>{opacity}</layerOpacity></qgis>")


def qml_categorized(field, cats, opacity=1.0, outline="255,255,255,0", width=0.0, kind="fill"):
    c_xml = "".join(f'<category value="{v}" label={quoteattr(lbl)} symbol="{i}" render="true"/>'
                    for i, (v, lbl, _) in enumerate(cats))
    if kind == "fill":
        s_xml = "".join(_fill(str(i), col, outline, width) for i, (_, _, col) in enumerate(cats))
    elif kind == "line":
        s_xml = "".join(_line(str(i), col, width or 0.6) for i, (_, _, col) in enumerate(cats))
    else:
        s_xml = "".join(_marker(str(i), col, width or 2.6) for i, (_, _, col) in enumerate(cats))
    return _qml(f'<renderer-v2 type="categorizedSymbol" attr="{field}" symbollevels="0" enableorderby="0" '
                f'forceraster="0"><categories>{c_xml}</categories><symbols>{s_xml}</symbols></renderer-v2>',
                opacity)


def qml_single_fill(fill, outline, width, style="solid", ostyle="solid", opacity=1.0):
    return _qml('<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">'
                f'<symbols>{_fill("0", fill, outline, width, style, ostyle)}</symbols></renderer-v2>', opacity)


def qml_single_line(color, width, opacity=1.0, style="solid"):
    return _qml('<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">'
                f'<symbols>{_line("0", color, width, style)}</symbols></renderer-v2>', opacity)


def qml_single_marker(color, size, shape="circle"):
    return _qml('<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">'
                f'<symbols>{_marker("0", color, size, shape)}</symbols></renderer-v2>')


def embed_styles(gpkg, styles):
    """Write default QML styles into the GeoPackage's layer_styles table, keeping other layers' rows."""
    with sqlite3.connect(gpkg) as con:
        geom_cols = dict(con.execute("SELECT table_name, column_name FROM gpkg_geometry_columns"))
    old = None
    try:
        old = pyogrio.read_dataframe(gpkg, layer="layer_styles", read_geometry=False)
        old = old[~old["f_table_name"].isin(list(styles))]
    except Exception:
        pass
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    df = pd.DataFrame([{"f_table_catalog": "", "f_table_schema": "", "f_table_name": lyr,
                        "f_geometry_column": geom_cols.get(lyr, "geom"), "styleName": lyr, "styleQML": q,
                        "styleSLD": "", "useAsDefault": True, "description": "site_pipeline.py",
                        "owner": "", "ui": "", "update_time": now} for lyr, q in styles.items()])
    if old is not None and len(old):
        df = pd.concat([old[df.columns], df], ignore_index=True)
    pyogrio.write_dataframe(df, gpkg, layer="layer_styles", driver="GPKG")


def write_layer_metadata(gpkg, rows):
    """layer_metadata table (one row per layer: title, source, date, CRS, how it was made) and the
    GeoPackage's own identifier/description columns, which QGIS shows in the Browser panel."""
    new = pd.DataFrame(rows)
    try:
        old = pyogrio.read_dataframe(gpkg, layer="layer_metadata", read_geometry=False)
        new = pd.concat([old[~old["layer"].isin(new["layer"])], new], ignore_index=True)
    except Exception:
        pass
    pyogrio.write_dataframe(new, gpkg, layer="layer_metadata", driver="GPKG")
    with sqlite3.connect(gpkg) as con:
        for r in rows:
            con.execute("UPDATE gpkg_contents SET identifier = ?, description = ? WHERE table_name = ?",
                        (r["title"], f"{r['abstract']} Source: {r['source']}. CRS: {r['crs']}.", r["layer"]))


def solution_layers(grid_obj, grid, buildings, base, name, sid, walk, desig=None):
    """Proposed land use for the open site cells, as dissolved zones + building footprints + cells."""
    cs = grid_obj.cs
    open_mask = base == -2
    res = grid == 0
    job_d = nearest_taxicab(grid, res, (grid == 2) | (grid == 3)) * cs
    ess_d = nearest_taxicab(grid, res, grid == 4) * cs
    cells, by_type = [], {}
    for r, c in np.argwhere(open_mask):
        code = int(grid[r, c])
        g = grid_obj.cell(r, c)
        by_type.setdefault(code, []).append(g)
        rec = {"solution": name, "solution_id": sid, "row": int(r), "col": int(c), "lu_code": code,
               "land_use": LAND_USE_NAMES[code],
               "job_grid_ft": float(job_d[r, c]) if code == 0 else None,
               "services_grid_ft": float(ess_d[r, c]) if code == 0 else None,
               "area_acres": round(g.area / 43560, 3), "geometry": g}
        if walk:
            rec["walk_min_bart"] = round(float(walk["cell_to"]["bart"][r, c]) / 264.0, 1)
        if desig is not None:
            rec["gp_designation"] = desig[r, c]
        cells.append(rec)
    zones = []
    for code, geoms in by_type.items():
        merged = unary_union(geoms)
        for part in getattr(merged, "geoms", [merged]):
            zones.append({"solution": name, "solution_id": sid, "lu_code": code, "land_use": LAND_USE_NAMES[code],
                          "acres": round(part.area / 43560, 2), "geometry": part})
    footprints = []
    for i, (lt, cl) in enumerate([b for b in buildings if b[0] in (2, 3, 4) and len(b[1]) > 1], start=1):
        geom = unary_union([grid_obj.cell(r, c) for r, c in cl])
        footprints.append({"solution": name, "building_id": i, "lu_code": int(lt),
                           "land_use": LAND_USE_NAMES[int(lt)], "acres": round(geom.area / 43560, 2),
                           "geometry": geom})
    mk = lambda rows: gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_OUT)
    if not footprints:
        footprints = gpd.GeoDataFrame({"solution": [], "building_id": [], "lu_code": [], "land_use": [],
                                       "acres": []}, geometry=[], crs=CRS_OUT)
        return mk(zones), footprints, mk(cells)
    return mk(zones), mk(footprints), mk(cells)


def pareto_chart(pareto, plan_row, path, distance):
    """Trade-off charts (job walk vs services walk, livability) with the City's plan marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jk = "job_walk_ft" if distance == "network" and pareto["job_walk_ft"].notna().all() else "job_grid_ft"
    sk = "services_walk_ft" if jk == "job_walk_ft" else "services_grid_ft"
    unit = "walking" if jk == "job_walk_ft" else "grid"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (x, y, xl, yl) in zip(axes, ((jk, sk, f"Avg. {unit} distance to nearest job (ft)",
                                          f"Avg. {unit} distance to school / services (ft)"),
                                         (jk, "livability", f"Avg. {unit} distance to nearest job (ft)",
                                          "Livability score (higher is better)"))):
        ax.scatter(pareto[x], pareto[y], s=26, c="#4C6A92", alpha=0.75, label="Optimized layouts (Pareto front)",
                   edgecolors="white", linewidths=0.5)
        names = {k: t for k, t, _ in REPS}
        xs_ = pd.concat([pareto[x], pd.Series([plan_row[x]] if plan_row is not None else [])])
        ys_ = pd.concat([pareto[y], pd.Series([plan_row[y]] if plan_row is not None else [])])
        dx, dy = 0.03 * (xs_.max() - xs_.min() or 1), 0.04 * (ys_.max() - ys_.min() or 1)
        groups = []                     # highlighted points that sit on top of each other share one label
        for _, r in pareto[pareto["label"].notna()].iterrows():
            labs = [names.get(k.strip(), k) for k in str(r["label"]).split(",")]
            for g in groups:
                if abs(g[0] - r[x]) < dx and abs(g[1] - r[y]) < dy:
                    g[2].extend(labs)
                    break
            else:
                groups.append([r[x], r[y], labs])
        hl = pareto[pareto["label"].notna()]
        ax.scatter(hl[x], hl[y], s=42, facecolors="none", edgecolors="#1a1a1a", linewidths=0.9, zorder=4)
        for gx, gy, labs in groups:
            ax.annotate(" / ".join(dict.fromkeys(labs)), (gx, gy), fontsize=7, xytext=(6, 5),
                        textcoords="offset points", color="#333")
        if plan_row is not None:
            ax.scatter([plan_row[x]], [plan_row[y]], s=110, marker="*", c="#C0392B", zorder=5,
                       label="City's adopted plan (scored by the model)")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Berryessa BART Urban Village: trade-offs among optimized layouts vs. the City's plan", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def convergence_chart(conv, path):
    """Hypervolume of the non-dominated set by generation (the right convergence measure for NSGA-III:
    single-objective bests can dip as niching trims the front, the hypervolume does not)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(conv["generation"], conv["hypervolume"], color="#1F5A86", lw=1.8)
    final = conv["hypervolume"].iloc[-1]
    g99 = int(conv.generation[conv.hypervolume >= 0.99 * final].iloc[0])
    ax.axhline(0.99 * final, color="#999999", lw=0.8, ls="--")
    ax.annotate(f"99% of final by generation {g99}", (g99, 0.99 * final), xytext=(8, -14),
                textcoords="offset points", fontsize=8, color="#444")
    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("Hypervolume (normalized)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("NSGA-III convergence: hypervolume of the trade-off set", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def census_within(geom):
    """Residents and housing units (2020 Census blocks) inside geom, allocating each block by area."""
    blocks = od.load("census_blocks")
    if blocks is None or geom is None or geom.is_empty:
        return None, None
    hit = blocks[blocks.intersects(geom)]
    if hit.empty:
        return 0, 0
    share = hit.geometry.intersection(geom).area / hit.geometry.area.where(hit.geometry.area > 0, 1)
    return int(round(float((hit["POP100"].fillna(0) * share).sum()))), \
        int(round(float((hit["HU100"].fillna(0) * share).sum())))


def hypervolume_series(fronts, ref_point):
    """Hypervolume of the non-dominated set after each generation, objectives normalized to [0, 1] by
    the final front's range so all four count equally (reference point 1.1). Higher is better. Small dips
    happen because NSGA-III keeps no archive: niching can drop a non-dominated layout from a crowded front."""
    from pymoo.indicators.hv import HV
    allF = np.vstack([f for f in fronts if len(f)])
    lo = allF.min(0)
    hv = HV(ref_point=np.ones(allF.shape[1]) * 1.1)
    out = []
    for gen, f in enumerate(fronts, start=1):
        if not len(f):
            out.append((gen, 0.0, 0))
            continue
        z = (f - lo) / np.where(ref_point - lo > 0, ref_point - lo, 1)
        z = z[(z <= 1.1).all(1)]
        out.append((gen, float(hv(z)) if len(z) else 0.0, len(f)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm", default=os.path.join(HERE, "site_osm.json"))
    ap.add_argument("--module", default=None, help='optimizer file, e.g. "Nsga3 Z.py" (default: auto-detect)')
    ap.add_argument("--cell-ft", type=float, default=120.0, help="cell size in feet (default 120 ft = 36.6 m)")
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--program", choices=("city", "capstone"), default="city",
                    help="land-use mix: the City's plan (default) or the capstone's proportions")
    ap.add_argument("--no-industrial", action="store_true")
    ap.add_argument("--green-ratio", type=float, default=None)
    ap.add_argument("--site-source", choices=("auto", "official", "osm"), default="auto")
    ap.add_argument("--distance", choices=("network", "grid"), default="network")
    ap.add_argument("--flood-rule", choices=("report", "hard"), default="report",
                    help="FEMA 1%% flood zone: count new homes (default) or keep them out")
    ap.add_argument("--noise-rule", choices=("report", "hard"), default="report",
                    help="rail noise contour: count new homes (default) or keep them out")
    ap.add_argument("--out", default=GPKG_DEFAULT)
    args = ap.parse_args()
    t_start = time.perf_counter()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    main_run = os.path.abspath(args.out) == os.path.abspath(GPKG_DEFAULT)
    stem = "" if main_run else os.path.splitext(os.path.basename(args.out))[0] + "_"

    opt = load_optimizer(args.module, HERE)
    official = args.site_source == "official" or (args.site_source == "auto" and od.available())
    if args.site_source == "official" and not od.available():
        sys.exit("data/raw/ is missing; run fetch_official.py first or use --site-source osm")
    osm, site = load_osm(args.osm)
    cats = categorize(osm, official_creeks=official)
    print(f"Loaded {len(osm)} OSM features for {site.get('name', 'site')} "
          f"({'official boundary + parcels' if official else 'OpenStreetMap only'})")

    rules, uv, parcels = {}, None, None
    extra = {}
    if official:
        uv = od.uv_boundary()
        grid_obj = SiteGrid.around(uv, MARGIN_FT, args.cell_ft)
        rules = site_rules.constraint_layers(od, args.noise_rule, args.flood_rule)
        parcels, redev = redevelopable_parcels(uv, cats)
        parks, schools = od.open_parks(), od.open_schools()
        extra = {"blocked": unary_union([g for g in (rules.get("R1_CREEK_SETBACK"), rules.get("R2_FLOODWAY"))
                                         if g is not None]),
                 "parks": parks.geometry.union_all() if parks is not None and len(parks) else None,
                 "essential": schools.geometry.union_all() if schools is not None and len(schools) else None,
                 "redevelopable": redev}
    else:
        s, w, n_, e = site["bbox"]
        bb = gpd.GeoSeries([box(w, s, e, n_)], crs="EPSG:4326").to_crs(CRS_OUT).iloc[0]
        side = min(bb.bounds[2] - bb.bounds[0], bb.bounds[3] - bb.bounds[1])
        n = int(side // args.cell_ft)
        grid_obj = SiteGrid.square((bb.centroid.x, bb.centroid.y), n, args.cell_ft)
    frame = grid_obj.extent()

    cls, cov = classify(grid_obj, cats, extra if official else None)
    cls, edited = apply_overrides(cls, grid_obj, os.path.join(HERE, "site_edits.gpkg"))
    if edited.any():
        print(f"Applied manual overrides from site_edits.gpkg to {int(edited.sum())} cells")
    base = base_grid_from_classes(cls)
    counts = {CLASS_NAMES[k]: int((cls == k).sum()) for k in CLASS_NAMES}
    print(f"Grid {grid_obj.nr}x{grid_obj.nc} of {args.cell_ft:g} ft cells "
          f"({grid_obj.nc * args.cell_ft:,.0f} x {grid_obj.nr * args.cell_ft:,.0f} ft): " +
          ", ".join(f"{k} {v}" for k, v in counts.items()))

    masks = {"noise": np.zeros(base.shape, bool), "flood_x": np.zeros(base.shape, bool),
             "sfha": np.zeros(base.shape, bool)}
    if official:
        masks["noise"] = coverage_grid(grid_obj, rules.get("R4_RAIL_NOISE")) > 0.5
        masks["flood_x"] = coverage_grid(grid_obj, rules.get("R7_FLOOD_X")) > 0.5
        masks["sfha"] = coverage_grid(grid_obj, rules.get("R3_SFHA")) > 0.5
    no_res = (masks["sfha"] if args.flood_rule == "hard" else np.zeros(base.shape, bool)) | \
             (masks["noise"] if args.noise_rule == "hard" else np.zeros(base.shape, bool))
    site_open = base == -2
    print(f"Rules: FEMA 1% flood zone {args.flood_rule} ({int((masks['sfha'] & site_open).sum())} site cells), "
          f"rail noise contour {args.noise_rule} ({int((masks['noise'] & site_open).sum())} site cells); "
          f"{int((no_res & site_open).sum())} site cells closed to new homes")

    # ---- the City's plan on the same cells (needed first: it sets the land-use mix)
    plan, desig = (city_plan_grid(grid_obj, base, uv, uv_as=2) if official else (None, None))
    if args.program == "city" and plan is None:
        print("No General Plan data: using the capstone's land-use mix")
    cfg = program_config(opt, base, plan, args.program, args.no_industrial, args.green_ratio)

    walk = build_walk(grid_obj, cls, cats, rules, cats["stations"]) if official else None
    distance = args.distance if walk is not None else "grid"
    if args.distance == "network" and walk is None:
        print("No walk network data (data/raw/osm_walk.json); using grid distances")
    bart = od.bart_stops() if official else None
    bart_pts = [(p.x, p.y) for p in bart.geometry] if bart is not None and len(bart) else \
        [(s_["geometry"].x, s_["geometry"].y) for s_ in cats["stations"]]
    bart_cells = sorted({(int((grid_obj.y_top - y) // grid_obj.cs), int((x - grid_obj.x0) // grid_obj.cs))
                         for x, y in bart_pts})
    objectives = Objectives(opt, args.cell_ft, base, distance, walk["D"] if walk else None,
                            walk["ridx"] if walk else None, walk["cell_to"]["bart"] if walk else None, bart_cells)

    res, problem, cfg, fronts = run_optimizer(opt, base, args.gens, args.pop, args.seed, cfg, objectives, no_res,
                                              history=True)
    X, F = np.atleast_2d(res.X), np.atleast_2d(res.F)
    builder = problem.builder
    print("Buildings: " + ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in cfg.items()))

    reps = pick_representatives(F)
    titles = {k: t for k, t, _ in REPS}
    labels = {}
    for nm, i in reps.items():
        labels.setdefault(i, []).append(nm)
    built, rows = {}, []
    for i, x in enumerate(X):
        g, b, coll = builder.build(x)
        built[i] = (g, b)
        assert np.all(g[base != -2] == base[base != -2]), "fixed cells changed"
        row = {"solution_id": i, "label": ", ".join(labels.get(i, [])) or None, "collisions": int(coll)}
        row.update(layout_metrics(g, base, objectives, walk, masks, args.cell_ft))
        rows.append(row)
    pareto = pd.DataFrame(rows)
    conv = pd.DataFrame(hypervolume_series(fronts, F.max(0)), columns=["generation", "hypervolume", "front_size"])
    print(f"Convergence: hypervolume {conv.hypervolume.iloc[-1]:.4f} at generation {len(conv)}; "
          f"reached 99% of it by generation {int(conv.generation[conv.hypervolume >= 0.99 * conv.hypervolume.iloc[-1]].iloc[0])}")

    # ---- the City's plan, scored the same way (UV designation as mixed use; housing reading too)
    plan_rows = []
    if plan is not None:
        plan_h, _ = city_plan_grid(grid_obj, base, uv, uv_as=0)
        for nm, pg in (("city_plan", plan), ("city_plan_uv_as_housing", plan_h)):
            r = {"scenario": nm}
            r.update(layout_metrics(pg, base, objectives, walk, masks, args.cell_ft))
            plan_rows.append(r)
    plan_table = pd.DataFrame(plan_rows)
    if len(plan_table):
        p0 = plan_table.iloc[0]
        jk, sk = ("job_walk_ft", "services_walk_ft") if distance == "network" else ("job_grid_ft", "services_grid_ft")
        pareto["beats_plan_services_and_livability"] = ((pareto[sk] < p0[sk]) &
                                                       (pareto["livability"] > p0["livability"])).astype(int)
        pareto["beats_plan_job"] = (pareto[jk] < p0[jk]).astype(int)

    # ---- write the GeoPackage (keeping the QA reviewer decisions from the last run)
    prev_qa = None
    if os.path.exists(args.out):
        try:
            prev_qa = gpd.read_file(args.out, layer="qa_errors")
        except Exception:
            prev_qa = None
        os.remove(args.out)
    W = lambda gdf, layer: pyogrio.write_dataframe(gdf, args.out, layer=layer, driver="GPKG")
    styles, meta = {}, []
    lu_cats = [(k, LAND_USE_NAMES[k], _rgba(LAND_USE_COLORS[k])) for k in (0, 1, 2, 3, 4)]
    osm_src = f"OpenStreetMap (site_osm.json, data as of {site.get('osm_base', '')[:10]})"
    off_src = "official layers in data/raw (see data/manifest.csv)"
    run_desc = (f"NSGA-III seed {args.seed}, {args.gens} generations, population {args.pop}, {distance} distances, "
                f"{args.program} land-use mix")

    def M(layer, title, abstract, source, made="site_pipeline.py"):
        meta.append({"layer": layer, "title": title, "abstract": abstract, "source": source,
                     "crs": "EPSG:2227 (NAD83 / California zone 3, US ft)", "produced_by": made,
                     "produced": datetime.now(timezone.utc).strftime("%Y-%m-%d")})

    site_poly = unary_union([grid_obj.cell(r, c) for r, c in np.argwhere(base == -2)])
    W(gpd.GeoDataFrame([{"name": site.get("name"), "acres": round(site_poly.area / 43560, 1),
                         "cells": int((base == -2).sum()), "geometry": site_poly}],
                       geometry="geometry", crs=CRS_OUT), "site_boundary")
    styles["site_boundary"] = qml_single_fill("0,0,0,0", "255,165,0,255", 1.4, style="no")
    M("site_boundary", "Redevelopment site", "Union of the grid cells the optimizer may assign.",
      off_src if official else osm_src)
    W(gpd.GeoDataFrame([{"name": "study frame", "rows": grid_obj.nr, "cols": grid_obj.nc,
                         "cell_ft": args.cell_ft, "geometry": frame}], geometry="geometry", crs=CRS_OUT), "study_frame")
    styles["study_frame"] = qml_single_fill("0,0,0,0", "60,60,60,255", 0.3, style="no", ostyle="dash")
    M("study_frame", "Study frame", "Extent of the analysis grid.", "site_pipeline.py")
    if uv is not None:
        ua = od.uv_attributes()
        W(gpd.GeoDataFrame([{"name": "Berryessa BART Urban Village", "acres": round(uv.area / 43560, 1),
                             "plan_housing": ua.get("HOUSING"), "plan_jobs": ua.get("JOBS"),
                             "plan_url": ua.get("PLANURL"), "geometry": uv}], geometry="geometry", crs=CRS_OUT),
          "uv_boundary")
        styles["uv_boundary"] = qml_single_fill("0,0,0,0", "200,30,30,255", 0.9, style="no", ostyle="dash")
        M("uv_boundary", "Berryessa BART Urban Village (official boundary)",
          "City of San Jose Growth Areas 2040 layer, Urban Village 'Berryessa BART'.",
          "City of San Jose GIS, OPN_OpenDataService layer 520")

    rep_rows = []
    for order, (nm, title, _) in enumerate(REPS):
        i = reps[nm]
        zones, fps, cells = solution_layers(grid_obj, *built[i], base, nm, i, walk)
        W(zones, f"{nm}_zones")
        W(fps, f"{nm}_buildings")
        W(cells, f"{nm}_cells")
        styles[f"{nm}_zones"] = qml_categorized("lu_code", lu_cats, opacity=0.78, outline="40,40,40,120", width=0.15)
        styles[f"{nm}_cells"] = qml_categorized("lu_code", lu_cats, opacity=0.78, outline="255,255,255,90", width=0.1)
        styles[f"{nm}_buildings"] = qml_single_fill("0,0,0,0", "20,20,20,255", 0.45, style="no")
        for suf, t in (("zones", "proposed land use zones"), ("buildings", "building footprints"),
                       ("cells", "proposed land use by cell")):
            M(f"{nm}_{suf}", f"{title}: {t}", f"Pareto solution #{i}, {run_desc}.", "model output")
        r = pareto.loc[i]
        rep_rows.append({"order": order, "key": nm, "title": title, "solution_id": i})
        jk = r.job_walk_ft if distance == "network" else r.job_grid_ft
        sk = r.services_walk_ft if distance == "network" else r.services_grid_ft
        bk = r.get("new_homes_walk_to_bart_ft", r.new_homes_grid_to_bart_ft) if distance == "network" \
            else r.new_homes_grid_to_bart_ft
        print(f"  {nm:<22} #{i:>3}: job {jk:>6.0f} ft, services {sk:>6.0f} ft, livability {r.livability:>7.1f}, "
              f"new homes to BART {bk:>5.0f} ft, new homes in noise / flood zone "
              f"{r.new_homes_in_noise_contour} / {r.new_homes_in_flood_zone}")
    pyogrio.write_dataframe(pd.DataFrame(rep_rows), args.out, layer="representatives", driver="GPKG")
    M("representatives", "Highlighted solutions", "The four Pareto solutions shown on the maps, in map order.",
      "model output")

    if plan is not None:
        zones, fps, cells = solution_layers(grid_obj, plan, [], base, "city_plan", -1, walk, desig)
        W(zones, "city_plan_zones")
        W(cells, "city_plan_cells")
        styles["city_plan_zones"] = qml_categorized("lu_code", lu_cats, opacity=0.78, outline="40,40,40,120", width=0.15)
        styles["city_plan_cells"] = qml_categorized("lu_code", lu_cats, opacity=0.78, outline="255,255,255,90", width=0.1)
        M("city_plan_zones", "City's adopted plan on the model grid",
          "General Plan 2040 designations inside the Urban Village, mapped to the five model land uses "
          "(see GP_TO_LU in official_data.py).", "City of San Jose GIS, General Plan 2040 (layer 404)")
        M("city_plan_cells", "City's adopted plan by cell", "As city_plan_zones, one row per site cell, with "
          "the General Plan designation.", "City of San Jose GIS, General Plan 2040 (layer 404)")
        bal = built[reps["balanced"]][0]
        diff = [{"row": int(r), "col": int(c), "plan_code": int(plan[r, c]), "plan_use": LAND_USE_NAMES[int(plan[r, c])],
                 "balanced_code": int(bal[r, c]), "balanced_use": LAND_USE_NAMES[int(bal[r, c])],
                 "gp_designation": desig[r, c], "same": int(plan[r, c] == bal[r, c]), "geometry": grid_obj.cell(r, c)}
                for r, c in np.argwhere(base == -2)]
        W(gpd.GeoDataFrame(diff, geometry="geometry", crs=CRS_OUT), "plan_vs_balanced")
        styles["plan_vs_balanced"] = qml_categorized("same", [(1, "Same land use as the City's plan", "185,194,204,150"),
                                                             (0, "Different from the City's plan", "232,89,12,215")],
                                                     outline="255,255,255,120", width=0.1)
        M("plan_vs_balanced", "Balanced layout vs. the City's plan", "Cell-by-cell comparison of land use.",
          "model output + City General Plan")

    # Constraint grid: what the optimizer "saw", plus rule flags and walk times
    crow = []
    for rr in range(grid_obj.nr):
        for cc in range(grid_obj.nc):
            rec = {"row": rr, "col": cc, "class_code": int(cls[rr, cc]), "class": CLASS_NAMES[int(cls[rr, cc])],
                   "manual_edit": int(edited[rr, cc]), "no_new_homes": int(no_res[rr, cc] and base[rr, cc] == -2),
                   "in_flood_zone": int(masks["sfha"][rr, cc]), "in_noise_contour": int(masks["noise"][rr, cc]),
                   "in_flood_x": int(masks["flood_x"][rr, cc]), "geometry": grid_obj.cell(rr, cc)}
            if walk:
                for k in ("bart", "grocery", "park", "school"):
                    v = walk["cell_to"][k][rr, cc]
                    rec[f"walk_min_{k}"] = round(float(v) / 264.0, 1) if np.isfinite(v) else None
            crow.append(rec)
    W(gpd.GeoDataFrame(crow, geometry="geometry", crs=CRS_OUT), "constraint_grid")
    styles["constraint_grid"] = qml_categorized("class_code", [
        (C_UNBUILDABLE, CLASS_NAMES[C_UNBUILDABLE], "90,90,90,200"),
        (C_OUTSIDE, CLASS_NAMES[C_OUTSIDE], "255,255,255,40"),
        (C_SITE, CLASS_NAMES[C_SITE], "255,170,0,160"),
        (C_EX_RES, CLASS_NAMES[C_EX_RES], "242,217,92,160"),
        (C_EX_PARK, CLASS_NAMES[C_EX_PARK], "93,174,75,160"),
        (C_EX_ESS, CLASS_NAMES[C_EX_ESS], "63,127,207,160")], outline="255,255,255,120", width=0.1)
    M("constraint_grid", "Constraint grid (what the optimizer saw)",
      "One 120 ft cell per row: class, manual edits, rule flags, walk minutes to BART / grocery / park / school.",
      "site_pipeline.py from OSM + official layers")

    # Rules
    if official:
        W(site_rules.rules_layer(rules, frame, CRS_OUT), "rules")
        styles["rules"] = qml_categorized("rule_id", [
            ("R1_CREEK_SETBACK", "Creek setback (CP 6-34): no development", "40,110,200,90"),
            ("R2_FLOODWAY", "FEMA floodway: no development", "20,60,160,150"),
            ("R3_SFHA", "FEMA 1% flood zone: homes counted" if args.flood_rule == "report"
             else "FEMA 1% flood zone: no new homes", "90,150,230,90"),
            ("R4_RAIL_NOISE", "Rail noise 70-74 dBA: homes counted" if args.noise_rule == "report"
             else "Rail noise 70-74 dBA: no new homes", "200,60,160,60"),
            ("R7_FLOOD_X", "FEMA Zone X shaded (reported)", "150,190,240,50")], outline="60,60,60,160", width=0.25)
        M("rules", "Rules as geometry", "One polygon per written rule (site_rules.py), clipped to the frame.",
          "Council Policy 6-34, FEMA NFHL, City GP noise contours")
        rt = site_rules.rules_table(od.firm_effective_dates(), args.noise_rule, args.flood_rule)
        pyogrio.write_dataframe(rt, args.out, layer="rules_table", driver="GPKG")
        M("rules_table", "Rules table", "Rule id, description, value, units, enforcement, source citation, "
          "date checked.", "site_rules.py")
        # official context layers, clipped to the frame
        man = od.manifest()
        man_url = dict(zip(man["layer"], man["url"])) if len(man) else {}
        for key, layer, title, style in (
                ("general_plan", "gp_landuse", "General Plan 2040 land use (City)", None),
                ("zoning", "zoning", "Zoning districts (City)", qml_single_fill("0,0,0,0", "120,120,120,200", 0.2, style="no")),
                ("streets", "city_streets", "Street centerlines (City)", qml_single_line("90,90,90,220", 0.3)),
                ("streams", "creeks", "Creek centerlines (City / Valley Water)", None),
                ("flood_zones", "fema_flood", "FEMA flood hazard zones", None),
                ("noise", "noise_contours", "General Plan noise contours (City)", None),
                ("census_blocks", "census_blocks", "2020 Census blocks (population, housing units)", None),
                ("transit_stops", "transit_stops", "Transit stops (Caltrans GTFS compilation)", None)):
            g = od.load(key)
            if g is None:
                continue
            g = g.clip(frame.buffer(1500) if key in ("transit_stops", "census_blocks") else frame)
            g = g[~g.is_empty]
            if key == "general_plan":
                g = g.assign(model_code=g["GPABBREVIATION"].map(lambda a: od.GP_TO_LU.get(a, (2, ""))[0]))
            if key == "transit_stops":
                g = g.assign(mode=np.where(g["route_type"].astype(str) == "1", "BART", "VTA bus"))
            if key == "census_blocks":
                g = g[["GEOID", "POP100", "HU100", "geometry"]]
            W(g, layer)
            if style:
                styles[layer] = style
            M(layer, title, f"Clipped to the study frame from {od.FILES[key]}.", man_url.get(key, off_src))
        styles["creeks"] = qml_categorized("COVEREDSTATUS", [("Exposed", "Creek (open channel)", "30,110,210,255"),
                                                            ("Covered", "Creek (culvert)", "120,160,210,200")],
                                           kind="line", width=0.8)
        styles["fema_flood"] = qml_categorized("FLD_ZONE", [("AE", "Zone AE (1% annual chance / floodway)", "30,80,200,120"),
                                                           ("X", "Zone X (0.2% / minimal)", "150,190,240,50")],
                                               outline="30,80,200,160", width=0.2)
        styles["noise_contours"] = qml_single_fill("200,60,160,30", "200,60,160,220", 0.35, ostyle="dash")
        styles["census_blocks"] = qml_single_fill("0,0,0,0", "120,120,120,150", 0.15, style="no", ostyle="dot")
        styles["transit_stops"] = qml_categorized("mode", [("BART", "BART platform", "0,120,190,255"),
                                                          ("VTA bus", "VTA bus stop", "80,80,80,255")],
                                                  kind="marker", width=2.2)
        gp_cats = [("TR", "Transit Residential", "242,217,92,190"), ("UR", "Urban Residential", "232,197,71,190"),
                   ("MUN", "Mixed Use Neighborhood", "240,160,80,190"), ("RN", "Residential Neighborhood", "255,243,176,190"),
                   ("UV", "Urban Village", "217,83,79,190"), ("TEC", "Transit Employment Center", "160,128,208,190"),
                   ("OSPH", "Open Space, Parklands & Habitat", "93,174,75,190"), ("PQP", "Public / Quasi-Public", "63,127,207,190"),
                   ("TU", "Transportation & Utilities", "150,150,150,190"), ("HI", "Heavy Industrial", "90,90,90,190"),
                   ("LI", "Light Industrial", "130,130,130,190")]
        styles["gp_landuse"] = qml_categorized("GPABBREVIATION", gp_cats, outline="255,255,255,200", width=0.15)
        if parcels is not None:
            W(parcels, "parcels_uv")
            styles["parcels_uv"] = qml_categorized("redevelopable", [(1, "Redevelopable parcel", "255,170,0,70"),
                                                                     (0, "Not redevelopable", "0,0,0,0")],
                                                   outline="80,80,80,200", width=0.15)
            M("parcels_uv", "Urban Village parcels and redevelopability",
              "City parcels inside the Urban Village with OSM building coverage, parking/market share, and the "
              "redevelopable decision with its reason.", "City of San Jose GIS Parcels (layer 270) + OSM buildings")

    # Walk network outputs
    census = []
    if walk:
        bands = walk["bands"].copy()
        cum = None
        pop, hu = [], []
        for gm in bands.geometry:                   # residents within the band's outer edge (cumulative)
            cum = gm if cum is None else cum.union(gm)
            p_, h_ = census_within(cum)
            pop.append(p_)
            hu.append(h_)
        bands["residents_2020_within"] = pop
        bands["homes_2020_within"] = hu
        census = bands.drop(columns="geometry").to_dict("records")
        W(bands, "walk_bart")
        styles["walk_bart"] = qml_categorized("minutes", [(5, "5-minute walk", "0,104,55,110"),
                                                         (10, "10-minute walk", "49,163,84,90"),
                                                         (15, "15-minute walk", "161,217,155,80")],
                                              outline="255,255,255,0")
        M("walk_bart", "Walk time to Berryessa BART", "5 / 10 / 15 minute walking areas at 3 mph along OSM streets "
          "and paths (service-area method, 100 ft trim), with 2020 Census residents and housing units inside each "
          "band's outer edge (blocks allocated by area).", "OpenStreetMap walk network (data/raw/osm_walk.json); "
          "US Census Bureau 2020 blocks")
        we = walk["edges"].clip(frame.buffer(2000))
        W(we[~we.is_empty].explode(index_parts=False), "walk_network")
        styles["walk_network"] = qml_single_line("110,110,110,160", 0.18)
        M("walk_network", "Walk network", "Walkable OSM ways used for walking distances.", "OpenStreetMap")
        W(walk["dest_points"], "walk_destinations")
        styles["walk_destinations"] = qml_categorized("type", [("bart", "BART", "0,120,190,255"),
                                                               ("grocery", "Grocery", "230,120,20,255"),
                                                               ("park", "Park entrance", "60,150,60,255"),
                                                               ("school", "School entrance", "60,90,200,255")],
                                                      kind="marker", width=2.0)
        M("walk_destinations", "Walk destinations", "Network nodes used as BART, grocery, park and school "
          "destinations.", "OSM groceries; City parks and schools; Caltrans GTFS stops")

    # Real-world context from OSM
    def ctx(geom, label):
        geom = geom.intersection(frame)            # clip real-world context to the map frame
        parts = [p for p in getattr(geom, "geoms", [geom]) if p is not None and not p.is_empty]
        return gpd.GeoDataFrame([{"type": label, "geometry": p} for p in parts], geometry="geometry", crs=CRS_OUT)
    if cats["road_lines"]:
        roads = gpd.GeoDataFrame(cats["road_lines"], geometry="geometry", crs=CRS_OUT).clip(frame)
        W(roads[~roads.is_empty].explode(index_parts=False), "osm_roads")
        styles["osm_roads"] = qml_single_line("255,255,255,200", 0.5)
        M("osm_roads", "Street centerlines (OSM)", "Vehicular streets used for the road mask.", osm_src)
    water = unary_union([g for g in (cats["water"], rules.get("R1_CREEK_SETBACK")) if g is not None]) if official \
        else cats["water"]
    for key, geom, label, style in (
            ("water", water, "Creek + riparian setback",
             qml_single_fill("70,140,220,140", "40,100,180,255", 0.3)),
            ("parks", unary_union([g for g in (cats["parks"], extra.get("parks")) if g is not None]) or None,
             "Existing park", qml_single_fill("0,0,0,0", "93,174,75,255", 0.6, style="no")),
            ("essential", unary_union([g for g in (cats["essential"], extra.get("essential")) if g is not None]) or None,
             "Existing school / public", qml_single_fill("0,0,0,0", "63,127,207,255", 0.6, style="no"))):
        if geom is not None and not geom.is_empty:
            W(ctx(geom, label), f"osm_{key}")
            styles[f"osm_{key}"] = style
            M(f"osm_{key}", label, "Context layer clipped to the frame.", osm_src + (" + City layers" if official else ""))
    stations = cats["stations"]
    if stations:
        W(gpd.GeoDataFrame(stations, geometry="geometry", crs=CRS_OUT).clip(frame), "osm_stations")
        styles["osm_stations"] = qml_single_marker("0,120,190,255", 4.2)
        M("osm_stations", "BART station", "Station point.", osm_src)

    pyogrio.write_dataframe(pareto, args.out, layer="pareto_front", driver="GPKG")
    M("pareto_front", "Pareto front (all solutions)", "Every non-dominated layout: walking and grid distances, "
      "livability, walk to BART, hazard exposure, access shares. " + run_desc + ".", "model output")
    pyogrio.write_dataframe(conv, args.out, layer="convergence", driver="GPKG")
    M("convergence", "Convergence (hypervolume by generation)", "Normalized hypervolume of the feasible "
      "non-dominated set after each generation; it should rise and level off.", "model output")
    if len(plan_table):
        pyogrio.write_dataframe(plan_table, args.out, layer="plan_comparison", driver="GPKG")
        M("plan_comparison", "City's plan scored by the model", "Same metrics as pareto_front for the City's "
          "plan (Urban Village designation read as mixed use, and as housing).", "model output")
    if prev_qa is not None and len(prev_qa):
        pyogrio.write_dataframe(prev_qa, args.out, layer="qa_errors", driver="GPKG", geometry_type="Point")
    embed_styles(args.out, styles)
    write_layer_metadata(args.out, meta)

    os.makedirs(os.path.join(out_dir, "maps"), exist_ok=True)
    pareto_chart(pareto, plan_table.iloc[0] if len(plan_table) else None,
                 os.path.join(out_dir, "maps", f"{stem}pareto_vs_city_plan.png"), distance)
    convergence_chart(conv, os.path.join(out_dir, "maps", f"{stem}convergence.png"))
    np.savez(os.path.join(out_dir, f"{stem}site_results.npz"), X=X, F=F, base_grid=base, classes=cls,
             city_plan=plan if plan is not None else np.zeros(0), cell_ft=args.cell_ft, x0=grid_obj.x0,
             y_top=grid_obj.y_top, distance=distance, **cfg)
    summary = {"site_source": "official" if official else "osm", "distance": distance, "program": args.program,
               "flood_rule": args.flood_rule, "noise_rule": args.noise_rule, "seed": args.seed,
               "generations": args.gens, "population": args.pop,
               "grid": [grid_obj.nr, grid_obj.nc], "cell_ft": args.cell_ft, "classes": counts,
               "manual_edit_cells": int(edited.sum()), "pareto_solutions": len(pareto),
               "representatives": {nm: int(i) for nm, i in reps.items()},
               "walk": walk["stats"] if walk else None, "walk_bart_census": census,
               "buildings": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in cfg.items()},
               "hypervolume_final": round(float(conv.hypervolume.iloc[-1]), 6),
               "hypervolume_99pct_generation": int(conv.generation[conv.hypervolume >=
                                                                   0.99 * conv.hypervolume.iloc[-1]].iloc[0]),
               "osm_base": site.get("osm_base", ""), "seconds": round(time.perf_counter() - t_start)}
    with open(os.path.join(out_dir, f"{stem}run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if len(plan_table):
        p = plan_table.iloc[0]
        print(f"  City's plan (scored):  job {p.job_walk_ft if distance == 'network' else p.job_grid_ft:.0f} ft, "
              f"services {p.services_walk_ft if distance == 'network' else p.services_grid_ft:.0f} ft, "
              f"livability {p.livability:.1f}, new homes in noise / flood zone "
              f"{p.new_homes_in_noise_contour} / {p.new_homes_in_flood_zone}")
        print(f"  {int(pareto.beats_plan_services_and_livability.sum())} of {len(pareto)} layouts beat the plan on "
              f"both services and livability; {int(pareto.beats_plan_job.sum())} beat it on walk to jobs")
    print(f"Wrote {args.out} ({time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
