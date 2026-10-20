"""
GIS data QA/QC for the Berryessa site model
===========================================

Checks the real-world source data (OpenStreetMap) on its own and against the official City, FEMA and
GTFS layers, any manual edits in site_edits.gpkg, and the optimizer's output layers. Writes an error
layer a technician can review in QGIS, a summary table and a corrected road network, all into
berryessa.gpkg next to the model layers.

Checks
  Source data (site_osm.json)
    GEOM_INVALID      polygon is not valid (self-intersection, bad ring)       Error    auto-fix: make_valid
    DUPLICATE_GEOM    two features with identical geometry and type            Error    auto-fix: drop copy
    ROAD_UNDERSHOOT   road ends within 25 ft of another road without joining   Error    auto-fix: snap
    ROAD_DEAD_END     road ends with nothing nearby (cul-de-sac or missing)    Info     review
    CROSSING_NO_NODE  two roads cross with no shared junction and no           Warning  review
                      bridge/tunnel/layer tag
    MISSING_NAME      public street without a name (City centerline name       Warning  review
                      suggested when one runs alongside)
    NOT_IN_DICTIONARY highway value not in the project's data dictionary       Info     review
    LANDUSE_CONFLICT  redevelopable lot overlaps existing homes/park/school     Warning  review
  OpenStreetMap vs official layers (data/raw, see data/manifest.csv)
    OSM_BUILDING_NO_PARCEL   OSM building not inside any City parcel           Info     review
    PARCEL_LANDUSE_MISMATCH  OSM land use disagrees with the General Plan      Info     review
    STOP_NOT_IN_OSM          GTFS stop with no OSM stop within 100 ft          Warning  review
    STOP_NOT_IN_GTFS         OSM bus stop with no GTFS stop within 100 ft      Info     review
    CREEK_OFFSET             OSM creek more than 50 ft from the official       Warning  review
                             (City / Valley Water) centerline
  Manual edits (site_edits.gpkg, layer "overrides")
    EDIT_INCOMPLETE   override missing its class or reason                     Error    review
    EDIT_INVALID      override geometry invalid or empty                       Error    review
  Model outputs (berryessa.gpkg)
    CELL_AMBIGUOUS    site cell where another class also covers 30%+          Info     review
    ZONE_DOMAIN       land use code outside the allowed list                   Error
    ZONE_OVERLAP      proposed zones overlap each other                        Error
    ZONE_OUTSIDE_SITE proposed zone extends past the site boundary             Error
    SITE_GAP          part of the site has no proposed land use                Error
    AREA_RECONCILE    zone acreage does not add up to site acreage             Error
    FOOTPRINT_ZONE    building footprint not inside a zone of its own use      Error

Usage:  python gis_qaqc.py            (after site_pipeline.py)
Output: layers qa_errors, qa_summary, roads_corrected in berryessa.gpkg; qa_summary.csv
Reviewer decisions (status "Fixed (manual)" / "Not an error" and notes) carry over between runs.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import unary_union, nearest_points
from shapely.validation import explain_validity, make_valid
from shapely import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import site_pipeline as sp  # noqa: E402
import official_data as od  # noqa: E402

CRS = sp.CRS_OUT
SNAP_FT = 25.0          # undershoot tolerance
EDGE_FT = 60.0          # ignore dangles this close to the download frame (roads continue off-map)
VEHICULAR = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified",
             "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link",
             "tertiary_link", "busway"}
NAMED_CLASSES = {"primary", "secondary", "tertiary", "residential", "unclassified", "living_street"}
DATA_DICTIONARY = VEHICULAR | {"footway", "cycleway", "path", "pedestrian", "steps", "track", "crossing"}
STOP_MATCH_FT = 100.0   # a transit stop in one source should have a partner this close in the other
CREEK_TOL_FT = 50.0     # OSM creek lines farther than this from the official centerline are flagged
STREET_MATCH_FT = 40.0  # an official street centerline this close and parallel names an unnamed OSM street
# OSM landuse / leisure / amenity -> model land use, for comparison with the General Plan
OSM_TO_LU = {"residential": 0, "commercial": 2, "retail": 2, "industrial": 3, "park": 1, "recreation_ground": 1,
             "school": 4, "college": 4}
MIXED_GP = {"UV", "MUN", "TR"}            # designations that allow both homes and commercial uses

CHECKS = {  # code: (severity, description, fix)
    "GEOM_INVALID": ("Error", "Polygon geometry is invalid", "auto: make_valid"),
    "DUPLICATE_GEOM": ("Error", "Duplicate feature (identical geometry and type)", "auto: drop duplicate"),
    "ROAD_UNDERSHOOT": ("Error", f"Road ends within {SNAP_FT:g} ft of another road without connecting", "auto: snap endpoint"),
    "ROAD_DEAD_END": ("Info", "Road ends with no road nearby (cul-de-sac or missing segment?)", "review"),
    "CROSSING_NO_NODE": ("Warning", "Roads cross without a shared junction or bridge/tunnel tag", "review"),
    "MISSING_NAME": ("Warning", "Public street has no name", "review"),
    "NOT_IN_DICTIONARY": ("Info", "highway value not in the project data dictionary", "review"),
    "LANDUSE_CONFLICT": ("Warning", "Redevelopable lot overlaps existing homes / park / school", "review"),
    "OSM_BUILDING_NO_PARCEL": ("Info", "OpenStreetMap building is not inside any City parcel", "review"),
    "PARCEL_LANDUSE_MISMATCH": ("Info", "OpenStreetMap land use disagrees with the General Plan designation", "review"),
    "STOP_NOT_IN_OSM": ("Warning", f"GTFS transit stop has no OpenStreetMap stop within {STOP_MATCH_FT:g} ft", "review"),
    "STOP_NOT_IN_GTFS": ("Info", f"OpenStreetMap bus stop has no GTFS stop within {STOP_MATCH_FT:g} ft", "review"),
    "CREEK_OFFSET": ("Warning", f"OpenStreetMap creek is more than {CREEK_TOL_FT:g} ft from the official centerline",
                     "review"),
    "EDIT_INCOMPLETE": ("Error", "Manual override is missing its class or reason", "review"),
    "EDIT_INVALID": ("Error", "Manual override geometry is invalid or empty", "review"),
    "CELL_AMBIGUOUS": ("Info", "Site cell where another class also covers 30% or more", "review"),
    "ZONE_DOMAIN": ("Error", "Land use code outside the allowed list", "none"),
    "ZONE_OVERLAP": ("Error", "Proposed zones overlap each other", "none"),
    "ZONE_OUTSIDE_SITE": ("Error", "Proposed zone extends past the site boundary", "none"),
    "SITE_GAP": ("Error", "Part of the site has no proposed land use", "none"),
    "AREA_RECONCILE": ("Error", "Zone acreage does not add up to site acreage", "none"),
    "FOOTPRINT_ZONE": ("Error", "Building footprint not inside a zone of its own land use", "none"),
}
SOLUTIONS = ("balanced", "best_job_commute", "best_livability", "best_transit_access")   # fallback list


class Findings:
    def __init__(self):
        self.rows = []

    def add(self, check, geom, layer, ref, detail="", fixed=False):
        sev, desc, fix = CHECKS[check]
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        self.rows.append({"error_id": len(self.rows) + 1, "check_code": check, "severity": sev,
                          "source_layer": layer, "feature_ref": str(ref),
                          "description": desc + (f": {detail}" if detail else ""),
                          "fix": fix, "status": "Fixed (auto)" if fixed else "Open",
                          "reviewer_note": None, "geometry": pt})


# ----------------------------------------------------------------------------- source data
def raw_features(path):
    """Parse OSM JSON without repairing geometry, so invalid inputs can be reported."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        g = None
        if el["type"] == "way":
            g = sp._way_geom(el)
        elif el["type"] == "relation":
            g = sp._relation_polys(el)
        if g is not None and not g.is_empty:
            rows.append({"ref": f"{el['type']}/{el['id']}", "name": tags.get("name"), "tags": tags, "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(CRS)
    return gdf, data["site"]


def check_polygons(f, feats):
    polys = feats[feats.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    for _, r in polys[~polys.geometry.is_valid].iterrows():
        f.add("GEOM_INVALID", r.geometry, "osm", r.ref, explain_validity(r.geometry), fixed=True)
    polys["kind"] = polys.tags.apply(lambda t: next((f"{k}={t[k]}" for k in ("landuse", "amenity", "leisure",
                                                                             "building", "natural") if k in t), "other"))
    polys["wkb"] = polys.geometry.apply(lambda g: g.normalize().wkb)
    for _, r in polys[polys.duplicated(["wkb", "kind"])].iterrows():
        f.add("DUPLICATE_GEOM", r.geometry, "osm", r.ref, r.kind, fixed=True)


def city_name_for(line, streets, tree):
    """Name of the official City centerline running alongside an OSM street (most length within
    STREET_MATCH_FT of it), or None."""
    if streets is None:
        return None
    zone = line.buffer(STREET_MATCH_FT, cap_style="flat")
    best, best_len = None, 0.0
    for j in tree.query(zone):
        g = streets.geometry.iloc[j]
        inside = g.intersection(zone).length
        name = streets["FULLNAME"].iloc[j]
        if inside > best_len and isinstance(name, str) and name.strip():
            best, best_len = name.strip(), inside
    return best if best_len >= min(0.5 * line.length, 100.0) else None


def check_roads(f, feats, frame, city_streets=None):
    roads = feats[feats.tags.apply(lambda t: "highway" in t) & (feats.geometry.geom_type == "LineString")].copy()
    roads["hw"] = roads.tags.apply(lambda t: t["highway"])
    s_tree = STRtree(list(city_streets.geometry)) if city_streets is not None and len(city_streets) else None
    for _, r in roads.iterrows():
        if r.hw not in DATA_DICTIONARY:
            f.add("NOT_IN_DICTIONARY", r.geometry, "osm_roads", r.ref, f"highway={r.hw}")
        if r.hw in NAMED_CLASSES and (pd.isna(r["name"]) or not str(r["name"]).strip()):
            hint = ""
            if s_tree is not None:
                nm = city_name_for(r.geometry, city_streets, s_tree)
                hint = f"; City centerline name: {nm}" if nm else "; no City centerline nearby"
            f.add("MISSING_NAME", r.geometry, "osm_roads", r.ref, f"highway={r.hw}{hint}")

    veh = roads[roads.hw.isin(VEHICULAR)].reset_index(drop=True)
    geoms = list(veh.geometry)
    tree = STRtree(geoms)
    inner_frame = frame.buffer(-EDGE_FT)
    corrected = [list(g.coords) for g in geoms]
    snapped = [False] * len(geoms)

    for i, g in enumerate(geoms):
        for end_idx, end in ((0, Point(g.coords[0])), (-1, Point(g.coords[-1]))):
            if not inner_frame.contains(end):
                continue
            # STRtree returns bounding-box candidates; keep only roads truly within tolerance
            others = [j for j in tree.query(end.buffer(SNAP_FT)) if j != i and geoms[j].distance(end) <= SNAP_FT]
            touching = [j for j in others if geoms[j].distance(end) < 0.5]
            # also count the way's own other end (closed loops) and self-touching
            rest = list(g.coords[1:] if end_idx == 0 else g.coords[:-1])
            loops_back = len(rest) >= 2 and LineString(rest).distance(end) < 0.5
            if touching or loops_back:
                continue
            if others:
                j = min(others, key=lambda k: geoms[k].distance(end))
                target = nearest_points(end, geoms[j])[1]
                corrected[i][end_idx] = (target.x, target.y)
                snapped[i] = True
                f.add("ROAD_UNDERSHOOT", end, "osm_roads", veh.ref[i],
                      f"{veh.hw[i]} gap of {end.distance(geoms[j]):.1f} ft to {veh.ref[j]}", fixed=True)
            elif veh.hw[i] != "service":     # driveways and parking aisles end all the time
                f.add("ROAD_DEAD_END", end, "osm_roads", veh.ref[i], f"highway={veh.hw[i]}")

    grade_sep = lambda t: any(t.get(k) not in (None, "no", "0") for k in ("bridge", "tunnel", "layer"))
    for i, j in combinations(range(len(geoms)), 2):
        a, b = geoms[i], geoms[j]
        if not a.envelope.intersects(b.envelope) or not a.crosses(b):
            continue
        if grade_sep(veh.tags[i]) or grade_sep(veh.tags[j]):
            continue
        shared = set(a.coords) & set(b.coords)
        if not shared:
            x = a.intersection(b)
            f.add("CROSSING_NO_NODE", x if x.geom_type == "Point" else x.representative_point(), "osm_roads",
                  f"{veh.ref[i]} x {veh.ref[j]}", f"{veh.hw[i]} / {veh.hw[j]}")

    fixed = gpd.GeoDataFrame({"osm_ref": veh.ref, "name": veh["name"], "highway": veh.hw,
                              "snapped": [int(x) for x in snapped],     # 0/1 integer so QGIS styles match
                              "geometry": [LineString(c) for c in corrected]}, geometry="geometry", crs=CRS)
    return fixed


def check_landuse_conflicts(f, osm_clean):
    cats = sp.categorize(osm_clean)
    existing = unary_union([g for g in (cats["residential"], cats["parks"], cats["essential"]) if g])
    redev = osm_clean[osm_clean.tags.apply(lambda t: sp._matches(t, sp.REDEVELOPABLE)) &
                      osm_clean.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    for _, r in redev.iterrows():
        share = r.geometry.intersection(existing).area / r.geometry.area if r.geometry.area else 0
        if share >= 0.2:
            f.add("LANDUSE_CONFLICT", r.geometry, "osm", r.osm_id,
                  f"{r.tags.get('name') or r.tags.get('amenity') or r.tags.get('landuse')}: {share:.0%} overlaps existing use")
    return cats


# ----------------------------------------------------------------------------- OSM vs official layers
def _osm_stops(osm_json):
    """Bus / transit stops from the wider OSM extract (data/raw/osm_walk.json), in feet."""
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    rows = []
    for el in (osm_json or {}).get("elements", []):
        t = el.get("tags", {})
        if el["type"] != "node":
            continue
        if t.get("highway") == "bus_stop" or t.get("public_transport") in ("platform", "stop_position", "station") \
                or t.get("railway") in ("station", "stop"):
            x, y = tf.transform(el["lon"], el["lat"])
            kind = "rail" if (t.get("railway") or t.get("train") == "yes" or t.get("subway") == "yes") else "bus"
            rows.append({"ref": f"node/{el['id']}", "name": t.get("name"), "kind": kind,
                         "is_bus_stop": int(t.get("highway") == "bus_stop" or
                                            (t.get("public_transport") == "platform" and t.get("bus") == "yes")),
                         "geometry": Point(x, y)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS) if rows else None


def check_official(f, raw, frame):
    """Where OpenStreetMap and the official layers disagree. Each check is skipped when its official
    layer is missing (the pipeline still runs on OSM alone)."""
    done = []
    area = frame.buffer(-EDGE_FT)

    # buildings vs parcels
    parcels = od.load("parcels")
    if parcels is not None and len(parcels):
        p_tree = STRtree(list(parcels.geometry))
        p_extent = parcels.geometry.union_all().envelope
        bld = raw[raw.tags.apply(lambda t: "building" in t) & raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        for _, b in bld.iterrows():
            pt = b.geometry.representative_point()
            if not area.contains(pt) or not p_extent.contains(pt):
                continue
            hits = [j for j in p_tree.query(pt) if parcels.geometry.iloc[j].contains(pt)]
            if not hits:
                f.add("OSM_BUILDING_NO_PARCEL", pt, "osm", b.ref,
                      f"building={b.tags.get('building')}" + (f", {b['name']}" if isinstance(b["name"], str) else ""))
        done.append("parcels")

    # OSM land use vs General Plan
    gp = od.load("general_plan")
    if gp is not None and len(gp):
        gp_tree = STRtree(list(gp.geometry))
        lu = raw[raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        for _, r in lu.iterrows():
            t = r.tags
            val = t.get("landuse") or t.get("leisure") or t.get("amenity")
            code = OSM_TO_LU.get(val)
            if code is None or not area.contains(r.geometry.representative_point()):
                continue
            g = r.geometry
            best, best_a = None, 0.0
            for j in gp_tree.query(g):
                a = g.intersection(gp.geometry.iloc[j]).area
                if a > best_a:
                    best, best_a = j, a
            if best is None or best_a < 0.5 * g.area:
                continue
            abbr = gp["GPABBREVIATION"].iloc[best]
            gp_code = od.GP_TO_LU.get(abbr, (None,))[0]
            if gp_code in (None, -1) or gp_code == code:
                continue
            if abbr in MIXED_GP and code in (0, 2):         # mixed-use designations allow homes and shops
                continue
            if code == 1 and gp_code in (0, 4):             # parks are allowed in any designation
                continue
            f.add("PARCEL_LANDUSE_MISMATCH", g, "osm", r.ref,
                  f"OSM {val} vs General Plan {abbr} ({gp['GPDESIGNATION'].iloc[best]})")
        done.append("general_plan")

    # transit stops: GTFS (Caltrans) vs OSM
    gtfs = od.load("transit_stops")
    osm_stops = _osm_stops(od.load_walk_osm())
    if gtfs is not None and osm_stops is not None:
        gt = gtfs[gtfs.geometry.within(area)]
        os_tree = STRtree(list(osm_stops.geometry))
        for _, s_ in gt.iterrows():
            near = [j for j in os_tree.query(s_.geometry.buffer(STOP_MATCH_FT))
                    if osm_stops.geometry.iloc[j].distance(s_.geometry) <= STOP_MATCH_FT]
            if not near:
                mode = "BART" if str(s_.get("route_type")) == "1" else "bus"
                f.add("STOP_NOT_IN_OSM", s_.geometry, "transit_stops", s_.get("stop_id"),
                      f"{s_.get('stop_name')} ({mode}, {s_.get('n_arrivals')} arrivals/day)")
        g_tree = STRtree(list(gtfs.geometry))
        ob = osm_stops[(osm_stops.is_bus_stop == 1) & osm_stops.geometry.within(area)]
        for _, o in ob.iterrows():
            near = [j for j in g_tree.query(o.geometry.buffer(STOP_MATCH_FT))
                    if gtfs.geometry.iloc[j].distance(o.geometry) <= STOP_MATCH_FT]
            if not near:
                f.add("STOP_NOT_IN_GTFS", o.geometry, "osm_walk", o.ref,
                      (o["name"] if isinstance(o["name"], str) else "unnamed stop"))
        done.append("transit_stops")

    # creek centerlines: OSM vs City / Valley Water
    streams = od.load("streams")
    if streams is not None and len(streams):
        official = streams.geometry.union_all()
        wl = raw[raw.tags.apply(lambda t: t.get("waterway") in sp.STREAM_SETBACK_FT and t.get("tunnel") != "culvert")
                 & (raw.geometry.geom_type == "LineString")]
        for _, w_ in wl.iterrows():
            g = w_.geometry.intersection(area)
            if g.is_empty or g.length < 50:
                continue
            pts = [g.interpolate(d) for d in np.arange(0, g.length, 20.0)]
            d = np.array([official.distance(p) for p in pts])
            if d.max() > 500:           # no official centerline near this line at all: a different feature
                continue
            if d.max() > CREEK_TOL_FT:
                worst = pts[int(d.argmax())]
                f.add("CREEK_OFFSET", worst, "osm", w_.ref,
                      f"largest offset {d.max():.0f} ft" + (f" ({w_['name']})" if isinstance(w_["name"], str) else ""))
        done.append("streams")
    return done


# ----------------------------------------------------------------------------- manual edits
def check_edits(f, path):
    if not os.path.exists(path):
        return 0
    try:
        ov = gpd.read_file(path, layer="overrides")
    except Exception:
        return 0
    for _, r in ov.iterrows():
        ref = r.get("override_id") or r.name
        if r.geometry is None or r.geometry.is_empty or not r.geometry.is_valid:
            g = r.geometry if (r.geometry is not None and not r.geometry.is_empty) else None
            if g is not None:
                f.add("EDIT_INVALID", g, "overrides", ref, explain_validity(g))
            continue
        missing = [k for k in ("force_class", "reason") if not r.get(k)]
        if missing:
            f.add("EDIT_INCOMPLETE", r.geometry, "overrides", ref, "missing " + ", ".join(missing))
    return len(ov)


# ----------------------------------------------------------------------------- model outputs
def check_outputs(f, gpkg, cats, summary_extra):
    site = gpd.read_file(gpkg, layer="site_boundary").geometry.union_all()
    grid = gpd.read_file(gpkg, layer="constraint_grid")
    blocked = unary_union([g for g in (cats["roads"], cats["rail"], cats["water"]) if g])
    others = {"road/rail/creek": blocked, "existing homes": cats["residential"], "existing park": cats["parks"],
              "existing school/public": cats["essential"]}
    for _, c in grid[grid.class_code == sp.C_SITE].iterrows():
        cell = c.geometry
        for nm, g in others.items():
            if g is not None and not g.is_empty:
                share = cell.intersection(g).area / cell.area
                if share >= 0.30:
                    f.add("CELL_AMBIGUOUS", cell, "constraint_grid", f"row {c.row}, col {c.col}",
                          f"{nm} covers {share:.0%}")
                    break

    allowed = set(sp.LAND_USE_NAMES)
    try:
        sols = list(pyogrio.read_dataframe(gpkg, layer="representatives", read_geometry=False)
                    .sort_values("order")["key"])
    except Exception:
        sols = list(SOLUTIONS)
    for sol in sols:
        zones = gpd.read_file(gpkg, layer=f"{sol}_zones")
        fps = gpd.read_file(gpkg, layer=f"{sol}_buildings")
        lyr = f"{sol}_zones"
        for _, z in zones[~zones.lu_code.isin(allowed)].iterrows():
            f.add("ZONE_DOMAIN", z.geometry, lyr, z.name, f"lu_code={z.lu_code}")
        tree = STRtree(list(zones.geometry))
        for i, g in enumerate(zones.geometry):
            for j in tree.query(g):
                if j > i:
                    ov = g.intersection(zones.geometry.iloc[j]).area
                    if ov > 1.0:
                        f.add("ZONE_OVERLAP", g.intersection(zones.geometry.iloc[j]), lyr, f"{i} & {j}", f"{ov:,.0f} sq ft")
        zu = zones.geometry.union_all()
        outside = zu.difference(site)
        if outside.area > 1.0:
            f.add("ZONE_OUTSIDE_SITE", outside, lyr, "-", f"{outside.area:,.0f} sq ft")
        gap = site.difference(zu)
        if gap.area > 1.0:
            f.add("SITE_GAP", gap, lyr, "-", f"{gap.area:,.0f} sq ft")
        z_ac, s_ac = zones.geometry.area.sum() / 43560, site.area / 43560
        if abs(z_ac - s_ac) > 0.01:
            f.add("AREA_RECONCILE", site, lyr, "-", f"zones {z_ac:.2f} ac vs site {s_ac:.2f} ac")
        for _, b in fps.iterrows():
            same = zones[zones.lu_code == b.lu_code].geometry.union_all()
            if same is None or b.geometry.difference(same).area > 1.0:
                f.add("FOOTPRINT_ZONE", b.geometry, f"{sol}_buildings", b.building_id, b.land_use)
        summary_extra[sol] = round(z_ac, 2)


# ----------------------------------------------------------------------------- styles
def _qml_points():
    def sym(i, rgb, size):
        return (f'<symbol type="marker" name="{i}" alpha="1" clip_to_extent="1" force_rhr="0">'
                '<layer class="SimpleMarker" enabled="1" locked="0" pass="0"><Option type="Map">'
                '<Option name="name" type="QString" value="circle"/>'
                f'<Option name="color" type="QString" value="{rgb}"/>'
                '<Option name="outline_color" type="QString" value="255,255,255,255"/>'
                '<Option name="outline_width" type="QString" value="0.4"/>'
                f'<Option name="size" type="QString" value="{size}"/>'
                '<Option name="size_unit" type="QString" value="MM"/></Option></layer></symbol>')
    cats = [("Error", "Error", "214,39,40,255", 3.4), ("Warning", "Warning", "255,152,0,255", 2.8),
            ("Info", "Info", "33,150,243,255", 2.2)]
    c_xml = "".join(f'<category value="{v}" label="{lbl}" symbol="{i}" render="true"/>' for i, (v, lbl, _, _) in enumerate(cats))
    s_xml = "".join(sym(i, rgb, size) for i, (_, _, rgb, size) in enumerate(cats))
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
            '<qgis version="3.34.0" styleCategories="AllStyleCategories">'
            '<renderer-v2 type="categorizedSymbol" attr="severity" symbollevels="0" enableorderby="0" forceraster="0">'
            f'<categories>{c_xml}</categories><symbols>{s_xml}</symbols></renderer-v2>'
            '<layerOpacity>1</layerOpacity></qgis>')


def _qml_roads():
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"
            '<qgis version="3.34.0" styleCategories="AllStyleCategories">'
            '<renderer-v2 type="categorizedSymbol" attr="snapped" symbollevels="0" enableorderby="0" forceraster="0">'
            '<categories><category value="0" label="Unchanged" symbol="0" render="true"/>'
            '<category value="1" label="Snapped (corrected)" symbol="1" render="true"/></categories><symbols>'
            + sp._line("0", "255,255,255,170", 0.45) + sp._line("1", "255,0,200,255", 1.2) +
            '</symbols></renderer-v2><layerOpacity>1</layerOpacity></qgis>')


# ----------------------------------------------------------------------------- main
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="QA/QC checks for the Berryessa model")
    ap.add_argument("--gpkg", default=os.path.join(HERE, "berryessa.gpkg"))
    args = ap.parse_args(argv)
    osm_path = os.path.join(HERE, "site_osm.json")
    model = args.gpkg
    edits = os.path.join(HERE, "site_edits.gpkg")
    if not os.path.exists(model):
        sys.exit(f"{model} not found; run site_pipeline.py first")

    f = Findings()
    raw, site = raw_features(osm_path)
    s, w, n, e = site["bbox"]
    frame = gpd.GeoSeries([box(w, s, e, n)], crs="EPSG:4326").to_crs(CRS).iloc[0]
    check_polygons(f, raw)
    roads_fixed = check_roads(f, raw, frame, od.load("streets"))
    osm_clean, _ = sp.load_osm(osm_path)
    cats = check_landuse_conflicts(f, osm_clean)
    compared = check_official(f, raw, frame)
    n_edits = check_edits(f, edits)
    acres = {}
    check_outputs(f, model, cats, acres)

    cols = ["error_id", "check_code", "severity", "source_layer", "feature_ref", "description", "fix", "status",
            "reviewer_note"]
    errors = gpd.GeoDataFrame(f.rows, geometry="geometry", crs=CRS) if f.rows else \
        gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs=CRS)
    errors["reviewer_note"] = errors["reviewer_note"].astype(object)
    counts = errors.groupby("check_code").size().to_dict() if len(errors) else {}
    summary = pd.DataFrame([{"check_code": k, "severity": v[0], "description": v[1], "fix": v[2],
                             "count": int(counts.get(k, 0)), "result": "PASS" if counts.get(k, 0) == 0 else
                             ("FIXED" if v[2].startswith("auto") else ("REVIEW" if v[2] == "review" else "FAIL"))}
                            for k, v in CHECKS.items()])

    # Keep reviewers' decisions from the previous run (matched on check + feature + location).
    # site_pipeline.py copies the previous qa_errors layer into the new GeoPackage for this.
    prev = None
    for src in (model, os.path.join(HERE, "site_qaqc.gpkg")):          # the second is the Oct 6 layout
        if prev is None and os.path.exists(src):
            try:
                prev = gpd.read_file(src, layer="qa_errors")
            except Exception:
                prev = None
    if prev is not None and len(prev) and len(errors):
        try:
            key = lambda d: d.check_code + "|" + d.feature_ref.astype(str) + "|" + \
                d.geometry.x.round(0).astype(str) + "," + d.geometry.y.round(0).astype(str)
            prev = prev.assign(k=key(prev)).drop_duplicates("k").set_index("k")
            errors["k"] = key(errors)
            kept = errors.k.isin(prev.index)
            reviewed = prev.status.reindex(errors.k).values
            take = kept & pd.Series(reviewed, index=errors.index).isin(["Fixed (manual)", "Not an error"])
            errors.loc[take, "status"] = prev.status.reindex(errors.k[take]).values
            errors.loc[kept, "reviewer_note"] = prev.reviewer_note.reindex(errors.k[kept]).values
            errors = errors.drop(columns="k")
            print(f"Carried over {int(take.sum())} reviewer decisions from the previous run")
        except Exception as exc:
            print(f"(Could not carry over previous review: {exc})")
            errors = errors.drop(columns="k", errors="ignore")

    pyogrio.write_dataframe(errors, model, layer="qa_errors", driver="GPKG", geometry_type="Point")
    roads_fixed = roads_fixed.clip(frame)
    roads_fixed = roads_fixed[~roads_fixed.is_empty].explode(index_parts=False)
    pyogrio.write_dataframe(roads_fixed, model, layer="roads_corrected", driver="GPKG")
    pyogrio.write_dataframe(summary, model, layer="qa_summary", driver="GPKG")
    sp.embed_styles(model, {"qa_errors": _qml_points(), "roads_corrected": _qml_roads()})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sp.write_layer_metadata(model, [
        {"layer": "qa_errors", "title": "QA findings", "abstract": "One point per QA/QC finding: check, severity, "
         "feature, description, fix, review status and note.", "source": "gis_qaqc.py",
         "crs": "EPSG:2227 (NAD83 / California zone 3, US ft)", "produced_by": "gis_qaqc.py", "produced": today},
        {"layer": "qa_summary", "title": "QA summary by check", "abstract": "Count and result (PASS / FIXED / "
         "REVIEW / FAIL) for every check.", "source": "gis_qaqc.py", "crs": "", "produced_by": "gis_qaqc.py",
         "produced": today},
        {"layer": "roads_corrected", "title": "Road network (QA-corrected)", "abstract": "OSM vehicular roads with "
         "undershoots snapped (snapped = 1).", "source": "OpenStreetMap + gis_qaqc.py",
         "crs": "EPSG:2227 (NAD83 / California zone 3, US ft)", "produced_by": "gis_qaqc.py", "produced": today}])
    summary.to_csv(os.path.join(HERE, "qa_summary.csv"), index=False)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"QA/QC run {stamp}: {len(raw)} source features, {n_edits} manual edits, "
          f"{len(acres)} solutions checked, official layers compared: {', '.join(compared) or 'none'}")
    print(summary[["check_code", "severity", "count", "result"]].to_string(index=False))
    print(f"\n{len(errors)} findings -> {model} (layer qa_errors)")


if __name__ == "__main__":
    main()
