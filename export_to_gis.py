"""
Export NSGA-III land use solutions to a georeferenced GeoPackage for QGIS
=========================================================================

Takes the pareto_results.npz written by the optimizer, rebuilds each layout with
the same CityBuilder decoder, places the grid on a real location, and writes a
GeoPackage that opens in QGIS already styled.

Layers written:
  <solution>_cells      one square polygon per grid cell, with land use and
                        per-cell metrics (commute distance in meters, livability)
  <solution>_buildings  multi-cell buildings dissolved into footprints
  study_area            outline of the whole grid
  pareto_front          attribute table: every Pareto solution and its objectives

<solution> is one of: balanced, best_job_commute, best_livability,
min_industrial_adjacency (the same four the GUI shows).

Usage (run from the folder that has your optimizer .py and pareto_results.npz):
  python export_to_gis.py pareto_results.npz
  python export_to_gis.py pareto_results.npz --module "Nsga3 Z.py"
  python export_to_gis.py pareto_results.npz --site mercer_island --cell-size 40
  python export_to_gis.py pareto_results.npz --lat 37.33 --lon -121.89
  python export_to_gis.py pareto_results.npz --all      # also export every solution

Requires: numpy, geopandas, shapely, pyproj, pyogrio
"""

import argparse
import importlib
import importlib.util
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union


# Approximate site centers (lat, lon). Nudge with --lat/--lon after looking in QGIS.
SITES = {
    # Largely undeveloped valley in south San Jose, long debated for development
    "coyote_valley": (37.2100, -121.7530),
    # Middle of Mercer Island, WA (the site named in the original project scope)
    "mercer_island": (47.5600, -122.2250),
}

OBJECTIVE_KEYS = ("job_commute", "essential_commute", "livability", "industrial_adjacent")
CONFIG_KEYS = ("n_industrial", "n_commercial", "n_essential",
               "industrial_size", "commercial_size", "essential_size", "green_ratio")
REPRESENTATIVES = ("best_job_commute", "best_livability", "min_industrial_adjacency", "balanced")

# Clean names for the attribute table (the optimizer's names include "(cluster)")
LAND_USE_NAMES = {-1: "Unbuildable", 0: "Residential", 1: "Green Space",
                  2: "Commercial", 3: "Industrial", 4: "Essential Services"}


# -----------------------------------------------------------------------------
# Loading the optimizer module
# -----------------------------------------------------------------------------

REQUIRED_NAMES = ("CityBuilder", "compute_scaled_config", "LAND_TYPES", "calc_job_commute",
                  "calc_essential_commute", "calc_livability", "calc_industrial_adjacent")


def _import_file(path):
    name = "optimizer_" + "".join(ch if ch.isalnum() else "_" for ch in os.path.splitext(os.path.basename(path))[0])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_optimizer(module_arg, search_dir):
    """Accepts a file path ("Nsga3 Z.py"), a module name, or nothing (auto-detect)."""
    if module_arg:
        candidates = [module_arg, module_arg + ".py", os.path.join(search_dir, module_arg),
                      os.path.join(search_dir, module_arg + ".py")]
        path = next((c for c in candidates if os.path.isfile(c)), None)
        if path is None:
            sys.exit(f"Couldn't find optimizer file '{module_arg}'.")
    else:
        found = []
        for folder in dict.fromkeys([os.getcwd(), search_dir]):
            for f in os.listdir(folder):
                full = os.path.join(folder, f)
                if f.endswith(".py") and f != os.path.basename(__file__) and os.path.isfile(full):
                    try:
                        text = open(full, encoding="utf-8", errors="ignore").read()
                    except OSError:
                        continue
                    if "class CityBuilder" in text and "def compute_scaled_config" in text:
                        found.append(full)
        if not found:
            sys.exit("No optimizer file found here. Pass it with --module \"Your File.py\".")
        path = max(found, key=os.path.getmtime)
    mod = _import_file(path)
    missing = [n for n in REQUIRED_NAMES if not hasattr(mod, n)]
    if missing:
        sys.exit(f"{os.path.basename(path)} is missing {', '.join(missing)}; pass the right file with --module.")
    print(f"Using optimizer code from {os.path.basename(path)}")
    return mod


# -----------------------------------------------------------------------------
# Rebuilding the layouts
# -----------------------------------------------------------------------------

def load_results(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    X = np.atleast_2d(data["X"])
    F = np.atleast_2d(data["F"])
    saved = {k: data[k].item() for k in data.files if k not in ("X", "F", "weights")}
    for short, full in (("n_ind", "n_industrial"), ("n_com", "n_commercial"), ("n_ess", "n_essential")):
        if short in saved and full not in saved:
            saved[full] = saved.pop(short)
    weights = data["weights"] if "weights" in data.files else None
    return X, F, saved, weights


def resolve_config(opt, X, saved, args):
    """Work out the grid size and building config that produced X.

    Order: values saved in the .npz -> command-line overrides -> auto-scaled
    defaults. If the result doesn't match the chromosome length, search for the
    auto-scaled grid size that does (covers older .npz files with no config).
    """
    grid_size = int(args.grid_size or saved.get("grid_size") or 20)
    cfg = {k: saved[k] for k in CONFIG_KEYS if k in saved}
    if len(cfg) < len(CONFIG_KEYS):
        scaled = opt.compute_scaled_config(grid_size)
        cfg = {k: cfg.get(k, scaled[k]) for k in CONFIG_KEYS}
    for k in CONFIG_KEYS:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    if cfg["green_ratio"] > 1:          # GUI shows a percent; builder wants a fraction
        cfg["green_ratio"] /= 100.0

    def length_for(gs, c):
        return opt.CityBuilder(gs, **{k: int(v) if k != "green_ratio" else v for k, v in c.items()}).encoding_length

    if length_for(grid_size, cfg) != X.shape[1]:
        for gs in range(5, 101):
            scaled = opt.compute_scaled_config(gs)
            c = {k: scaled[k] for k in CONFIG_KEYS}
            if length_for(gs, c) == X.shape[1]:
                print(f"Note: config didn't match the saved chromosomes; using auto-scaled {gs}x{gs} config.")
                grid_size, cfg = gs, c
                break
        else:
            sys.exit("Could not match the chromosome length to a grid configuration.\n"
                     "Pass --grid-size and the building counts/sizes you used for this run.")
    return grid_size, cfg


def neighbor_counts(grid, value):
    match = (grid == value).astype(float)
    counts = np.zeros_like(match)
    counts[1:, :] += match[:-1, :]
    counts[:-1, :] += match[1:, :]
    counts[:, 1:] += match[:, :-1]
    counts[:, :-1] += match[:, 1:]
    return counts


def nearest_taxicab(grid, mask_from, mask_to):
    """Taxicab distance (in cells) from every cell in mask_from to the nearest cell in mask_to."""
    out = np.full(grid.shape, np.nan)
    src, dst = np.argwhere(mask_from), np.argwhere(mask_to)
    if len(src) and len(dst):
        d = np.abs(src[:, None, :] - dst[None, :, :]).sum(-1).min(axis=1)
        out[src[:, 0], src[:, 1]] = d
    return out


def objectives_for(opt, grid):
    return {
        "job_commute": opt.calc_job_commute(grid),
        "essential_commute": opt.calc_essential_commute(grid),
        "livability": opt.calc_livability(grid),
        "industrial_adjacent": int(opt.calc_industrial_adjacent(grid)),
    }


def pick_representatives(F):
    """Same four picks the GUI's 'Show Final Results' makes."""
    f_min, f_max = F.min(0), F.max(0)
    F_norm = (F - f_min) / (f_max - f_min + 1e-12)
    return {
        "best_job_commute": int(np.argmin(F[:, 0])),
        "best_livability": int(np.argmin(F[:, 2])),   # F[:,2] is negative livability
        "min_industrial_adjacency": int(np.argmin(F[:, 3])),
        "balanced": int(np.argmin(np.linalg.norm(F_norm, axis=1))),
    }


# -----------------------------------------------------------------------------
# Georeferencing
# -----------------------------------------------------------------------------

def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class GridGeoref:
    """Maps grid (row, col) to a square in projected meters, centered on (lat, lon).
    Row 0 is the north edge, matching how the optimizer draws its grids."""

    def __init__(self, lat, lon, grid_size, cell_size):
        self.epsg = utm_epsg(lat, lon)
        cx, cy = Transformer.from_crs(4326, self.epsg, always_xy=True).transform(lon, lat)
        half = grid_size * cell_size / 2
        self.x0, self.y_top = cx - half, cy + half
        self.n, self.cs = grid_size, cell_size

    def cell(self, r, c):
        x = self.x0 + c * self.cs
        y = self.y_top - r * self.cs
        return box(x, y - self.cs, x + self.cs, y)

    def extent(self):
        return box(self.x0, self.y_top - self.n * self.cs, self.x0 + self.n * self.cs, self.y_top)


# -----------------------------------------------------------------------------
# Layer construction
# -----------------------------------------------------------------------------

def layout_layers(grid, buildings, geo, solution_name, solution_id, cell_size):
    n = grid.shape[0]
    res = grid == 0
    job_d = nearest_taxicab(grid, res, (grid == 2) | (grid == 3))
    ess_d = nearest_taxicab(grid, res, grid == 4)
    liv = (1.0 + 0.5 * neighbor_counts(grid, 1) + 0.3 * neighbor_counts(grid, 4)
           - 0.1 * neighbor_counts(grid, 2) - 2.0 * neighbor_counts(grid, 3))
    ind_adj = neighbor_counts(grid, 3) > 0

    bldg_id = np.full(grid.shape, -1, dtype=int)
    footprints = []
    next_id = 1
    for land_type, cells in buildings:
        if land_type in (2, 3, 4) and len(cells) > 0:
            for r, c in cells:
                bldg_id[r, c] = next_id
            geom = unary_union([geo.cell(r, c) for r, c in cells])
            footprints.append({
                "solution": solution_name, "solution_id": solution_id,
                "building_id": next_id, "lu_code": int(land_type),
                "land_use": LAND_USE_NAMES[int(land_type)], "n_cells": len(cells),
                "area_m2": round(geom.area, 1), "geometry": geom,
            })
            next_id += 1

    rows = []
    for r in range(n):
        for c in range(n):
            code = int(grid[r, c])
            is_res = code == 0
            rows.append({
                "solution": solution_name, "solution_id": solution_id,
                "row": r, "col": c, "lu_code": code,
                "land_use": LAND_USE_NAMES.get(code, str(code)),
                "building_id": int(bldg_id[r, c]) if bldg_id[r, c] > 0 else None,
                "job_dist_m": float(job_d[r, c] * cell_size) if is_res and not np.isnan(job_d[r, c]) else None,
                "ess_dist_m": float(ess_d[r, c] * cell_size) if is_res and not np.isnan(ess_d[r, c]) else None,
                "livability": round(float(liv[r, c]), 2) if is_res else None,
                "next_to_industrial": int(ind_adj[r, c]) if is_res else None,
                "geometry": geo.cell(r, c),
            })
    crs = f"EPSG:{geo.epsg}"
    cells_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    bldg_gdf = gpd.GeoDataFrame(footprints, geometry="geometry", crs=crs)
    return cells_gdf, bldg_gdf


# -----------------------------------------------------------------------------
# QGIS styles (embedded in the GeoPackage so layers open pre-styled)
# -----------------------------------------------------------------------------

def _rgba(hex_color, alpha=255):
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha}"


def _fill_symbol(name, fill, outline, width_mm, style="solid", outline_style="solid"):
    return f"""      <symbol type="fill" name="{name}" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="{fill}"/>
            <Option name="outline_color" type="QString" value="{outline}"/>
            <Option name="outline_style" type="QString" value="{outline_style}"/>
            <Option name="outline_width" type="QString" value="{width_mm}"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="joinstyle" type="QString" value="miter"/>
            <Option name="style" type="QString" value="{style}"/>
          </Option>
        </layer>
      </symbol>
"""


def _wrap_qml(renderer, opacity=1.0):
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="AllStyleCategories">
{renderer}  <layerOpacity>{opacity}</layerOpacity>
</qgis>
"""


def qml_cells(land_types):
    cats, syms = [], []
    for i, code in enumerate(sorted(land_types)):
        cats.append(f'      <category value="{code}" label="{LAND_USE_NAMES[code]}" symbol="{i}" render="true"/>')
        syms.append(_fill_symbol(str(i), _rgba(land_types[code]["color"]), "255,255,255,160", 0.1))
    renderer = (f'  <renderer-v2 type="categorizedSymbol" attr="lu_code" symbollevels="0" enableorderby="0" forceraster="0">\n'
                f'    <categories>\n' + "\n".join(cats) + '\n    </categories>\n'
                f'    <symbols>\n' + "".join(syms) + '    </symbols>\n  </renderer-v2>\n')
    return _wrap_qml(renderer, opacity=0.85)


def qml_outline(color, width_mm, outline_style="solid"):
    renderer = ('  <renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0" forceraster="0">\n'
                '    <symbols>\n' + _fill_symbol("0", "0,0,0,0", color, width_mm, style="no",
                                                  outline_style=outline_style) +
                '    </symbols>\n  </renderer-v2>\n')
    return _wrap_qml(renderer)


def embed_styles(gpkg_path, styles):
    """Write QGIS default styles into the GeoPackage's layer_styles table.
    QGIS looks here when a layer is loaded and applies the style automatically."""
    with sqlite3.connect(gpkg_path) as con:
        geom_cols = dict(con.execute("SELECT table_name, column_name FROM gpkg_geometry_columns"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    table = pd.DataFrame([{
        "f_table_catalog": "", "f_table_schema": "", "f_table_name": layer,
        "f_geometry_column": geom_cols.get(layer, "geom"), "styleName": layer,
        "styleQML": qml, "styleSLD": "", "useAsDefault": True,
        "description": "Generated by export_to_gis.py", "owner": "", "ui": "",
        "update_time": now,
    } for layer, qml in styles.items()])
    pyogrio.write_dataframe(table, gpkg_path, layer="layer_styles", driver="GPKG")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Export NSGA-III land use solutions to a QGIS-ready GeoPackage.")
    p.add_argument("npz", nargs="?", default="pareto_results.npz", help="pareto_results.npz from the optimizer")
    p.add_argument("--module", default=None,
                   help="your optimizer file, e.g. \"Nsga3 Z.py\" (default: newest .py in this folder "
                        "that defines CityBuilder)")
    p.add_argument("--site", choices=sorted(SITES), default="coyote_valley")
    p.add_argument("--lat", type=float, help="custom grid center latitude (overrides --site)")
    p.add_argument("--lon", type=float, help="custom grid center longitude (overrides --site)")
    p.add_argument("--cell-size", type=float, default=50.0, help="cell edge length in meters (default 50)")
    p.add_argument("--out", help="output .gpkg path (default: land_use_<site>.gpkg next to the .npz)")
    p.add_argument("--all", action="store_true", help="also write every Pareto solution into one layer")
    p.add_argument("--grid-size", type=int)
    for k in CONFIG_KEYS:
        p.add_argument("--" + k.replace("_", "-"), dest=k, type=float if k == "green_ratio" else int)
    args = p.parse_args()

    # Load the optimizer so we decode with the exact same code
    opt = load_optimizer(args.module, os.path.dirname(os.path.abspath(args.npz)))

    X, F, saved, weights = load_results(args.npz)
    grid_size, cfg = resolve_config(opt, X, saved, args)
    builder = opt.CityBuilder(grid_size, **{k: int(v) if k != "green_ratio" else v for k, v in cfg.items()})

    if args.lat is not None and args.lon is not None:
        lat, lon, site = args.lat, args.lon, "custom"
    else:
        (lat, lon), site = SITES[args.site], args.site
    geo = GridGeoref(lat, lon, grid_size, args.cell_size)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.npz)), f"land_use_{site}.gpkg")

    print(f"{len(X)} Pareto solutions | grid {grid_size}x{grid_size} | {args.cell_size:g} m cells "
          f"({grid_size * args.cell_size / 1000:g} km across) | site {site} ({lat:.4f}, {lon:.4f}) | EPSG:{geo.epsg}")

    # Pareto table, with objectives recomputed (unweighted) and converted to meters
    reps = pick_representatives(F)
    rep_lookup = {}
    for name, idx in reps.items():
        rep_lookup.setdefault(idx, []).append(name)
    built, records = {}, []
    for i, x in enumerate(X):
        grid, buildings, collisions = builder.build(x)
        built[i] = (grid, buildings)
        obj = objectives_for(opt, grid)
        records.append({
            "solution_id": i,
            "label": ", ".join(rep_lookup.get(i, [])) or None,
            "job_commute_cells": round(obj["job_commute"], 3),
            "job_commute_m": round(obj["job_commute"] * args.cell_size, 1),
            "essential_commute_cells": round(obj["essential_commute"], 3),
            "essential_commute_m": round(obj["essential_commute"] * args.cell_size, 1),
            "livability": round(obj["livability"], 2),
            "industrial_adjacent": obj["industrial_adjacent"],
            "collisions": int(collisions),
        })
    pareto = pd.DataFrame(records)

    # Sanity check: recomputed objectives should match the optimizer's F (when weights are 1)
    w = np.asarray(weights, float) if weights is not None else np.ones(4)
    if np.all(w > 0):
        F_unw = F / w
        recomputed = np.column_stack([pareto.job_commute_cells, pareto.essential_commute_cells,
                                      -pareto.livability, pareto.industrial_adjacent])
        if not np.allclose(F_unw, recomputed, atol=1e-2):
            print("Warning: recomputed objectives differ from the saved F values. If you edited "
                  "objectives or weights for this run, the attribute values reflect the default objectives.")

    if os.path.exists(out):
        os.remove(out)

    styles = {}
    write = lambda gdf, layer: pyogrio.write_dataframe(gdf, out, layer=layer, driver="GPKG")

    study = gpd.GeoDataFrame([{"site": site, "grid_size": grid_size, "cell_size_m": args.cell_size,
                               "area_km2": round(geo.extent().area / 1e6, 3), "geometry": geo.extent()}],
                             geometry="geometry", crs=f"EPSG:{geo.epsg}")
    write(study, "study_area")
    styles["study_area"] = qml_outline("40,40,40,255", 0.8, outline_style="dash")

    cells_qml = qml_cells(opt.LAND_TYPES)
    for name in REPRESENTATIVES:
        idx = reps[name]
        grid, buildings = built[idx]
        cells, bldgs = layout_layers(grid, buildings, geo, name, idx, args.cell_size)
        write(cells, f"{name}_cells")
        write(bldgs, f"{name}_buildings")
        styles[f"{name}_cells"] = cells_qml
        styles[f"{name}_buildings"] = qml_outline("0,0,0,255", 0.5)
        r = pareto.loc[idx]
        print(f"  {name:<26} solution {idx:>3}: job {r.job_commute_m:>6.0f} m, essential "
              f"{r.essential_commute_m:>6.0f} m, livability {r.livability:>6.1f}, ind. adjacent {r.industrial_adjacent}")

    if args.all:
        all_cells = pd.concat([layout_layers(g, b, geo, f"solution_{i}", i, args.cell_size)[0]
                               for i, (g, b) in built.items()], ignore_index=True)
        write(gpd.GeoDataFrame(all_cells, geometry="geometry", crs=f"EPSG:{geo.epsg}"), "all_solutions_cells")
        styles["all_solutions_cells"] = cells_qml

    pyogrio.write_dataframe(pareto, out, layer="pareto_front", driver="GPKG")
    embed_styles(out, styles)

    # Also save .qml files in case you want to re-apply a style by hand
    style_dir = os.path.splitext(out)[0] + "_styles"
    os.makedirs(style_dir, exist_ok=True)
    for fname, qml in {"cells.qml": cells_qml, "buildings.qml": styles["balanced_buildings"],
                       "study_area.qml": styles["study_area"]}.items():
        with open(os.path.join(style_dir, fname), "w", encoding="utf-8") as fh:
            fh.write(qml)

    print(f"\nWrote {out}")
    print(f"Style files in {style_dir}")


if __name__ == "__main__":
    main()
