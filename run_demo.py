"""
One-command reproducible run of the Berryessa land use project
==============================================================

    python run_demo.py                  full run: 300 generations, seed 7, QA, web map, maps, check vs reference
    python run_demo.py --quick          60 generations (about 3 minutes in all) for a live demo
    python run_demo.py --no-maps        skip the QGIS step (Python results, QA and web map only)
    python run_demo.py --no-basemap     maps without satellite / street tiles (offline)
    python run_demo.py --fetch          download fresh OpenStreetMap and official data first (the numbers will
                                        then differ from the reference, because the data changed)
    python run_demo.py --update-reference --reason "why"   accept this run's numbers as the new reference

Steps
  1. Check the Python packages
  2. Data: the frozen snapshots (site_osm.json, data/raw/), or --fetch for fresh downloads
  3. Site grid, walk network, City's plan and NSGA-III      site_pipeline.py   -> berryessa.gpkg
  4. QA/QC checks                                           gis_qaqc.py        -> berryessa.gpkg, qa_summary.csv
  5. Interactive web map                                    make_webmap.py     -> webmap/index.html
  6. Compare the numbers with expected_results.json (same data + same seed = same answer)
  7. QGIS project, map series and QA maps, headless         qgis_export_all.py -> berryessa_map.qgz, maps/
  8. Write demo_report.html (opens in a browser)

Everything is read from and written to the folder this script is in.
"""
import argparse
import datetime as dt
import glob
import html
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_FILE = "Nsga3 Z.py"          # Kyler's NSGA-III code (CityBuilder, objectives, operators)
GPKG = os.path.join(HERE, "berryessa.gpkg")
SEED = 7
FULL_GENS, QUICK_GENS = 300, 60
REFERENCE = os.path.join(HERE, "expected_results.json")
LOG = os.path.join(HERE, "demo_run.log")
PACKAGES = ["numpy", "pandas", "shapely", "pyproj", "pyogrio", "geopandas", "pymoo", "scipy", "matplotlib"]
SCRIPTS = (OPTIMIZER_FILE, "site_pipeline.py", "export_to_gis.py", "official_data.py", "site_rules.py",
           "walk_network.py", "gis_qaqc.py", "make_webmap.py")
MAP_SERIES = [("01_existing_conditions", "Existing conditions"), ("02_rules_constraints", "Rules and constraints"),
              ("03_walk_access_bart", "Walk access to BART"), ("04_city_plan", "The City's plan on the model grid"),
              ("05_balanced", "Balanced solution"), ("06_best_job_commute", "Best job access"),
              ("07_best_livability", "Best livability"), ("08_best_transit_access", "Most homes near BART"),
              ("09_balanced_vs_city_plan", "Balanced layout vs. the City's plan")]


def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def step(n, title):
    log(f"\n[{n}] {title}")


NOISE = ("Axes3D", "warnings.warn(", "PNG driver does not support update", "QStandardPaths",
         "Application path not initialized", "propagateSizeHints")


def run(cmd, env=None, timeout=3600, fatal=True):
    """Run a command, stream its output to the console and the log, stop on failure."""
    t0 = time.perf_counter()
    env = dict(env if env is not None else os.environ, PYTHONIOENCODING="utf-8")   # names with accents on Windows
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = line.rstrip()
        if not line.strip() or any(n in line for n in NOISE):
            continue
        log("    " + line)
    proc.wait(timeout=timeout)
    if proc.returncode != 0:
        log(f"    FAILED (exit code {proc.returncode}). See {LOG}.")
        if fatal:
            sys.exit(proc.returncode)
    return time.perf_counter() - t0, proc.returncode


# ----------------------------------------------------------------------------- results
def collect():
    """Pull the headline numbers out of the run's outputs."""
    import pandas as pd
    import pyogrio
    grid = pyogrio.read_dataframe(GPKG, layer="constraint_grid", read_geometry=False)
    pareto = pyogrio.read_dataframe(GPKG, layer="pareto_front", read_geometry=False)
    reps_t = pyogrio.read_dataframe(GPKG, layer="representatives", read_geometry=False).sort_values("order")
    qa = pd.read_csv(os.path.join(HERE, "qa_summary.csv"))
    summary = json.load(open(os.path.join(HERE, "run_summary.json")))
    keys = ["job_walk_ft", "services_walk_ft", "livability", "new_homes_walk_to_bart_ft", "new_home_cells",
            "new_homes_10min_bart", "new_homes_in_noise_contour", "new_homes_in_flood_zone", "pct_homes_10min_park"]

    def pick(row):
        return {k: (None if pd.isna(row.get(k)) else (int(row[k]) if float(row[k]).is_integer() and "ft" not in k
                                                       and k != "livability" else float(row[k])))
                for k in keys if k in row}
    reps = {}
    for _, r in reps_t.iterrows():
        row = pareto[pareto.solution_id == r.solution_id].iloc[0]
        reps[r.key] = {"solution_id": int(r.solution_id), **pick(row)}
    out = {
        "grid_classes": {k: int(v) for k, v in grid["class"].value_counts().sort_index().items()},
        "manual_edit_cells": int(grid["manual_edit"].sum()) if "manual_edit" in grid else 0,
        "pareto_solutions": int(len(pareto)),
        "representatives": reps,
        "hypervolume_final": summary.get("hypervolume_final"),
        "qa_counts": {r.check_code: int(r["count"]) for _, r in qa.iterrows()},
    }
    try:
        plan = pyogrio.read_dataframe(GPKG, layer="plan_comparison", read_geometry=False)
        out["city_plan"] = pick(plan[plan.scenario == "city_plan"].iloc[0])
        out["layouts_beating_plan_on_services_and_livability"] = int(pareto.beats_plan_services_and_livability.sum())
    except Exception:
        pass
    if summary.get("walk"):
        out["walk_network"] = {k: summary["walk"][k] for k in ("nodes", "segments", "dangles_joined")}
    if summary.get("walk_bart_census"):
        out["residents_2020_within_walk_of_bart"] = {f"{r['minutes']}_min": r["residents_2020_within"]
                                                     for r in summary["walk_bart_census"]}
    return out


def compare(ref, got, path=""):
    """List every difference between the reference and this run (numbers within 1e-6)."""
    diffs = []
    if isinstance(ref, dict):
        for k in sorted(set(ref) | set(got or {})):
            if k not in (got or {}):
                diffs.append(f"{path}{k}: missing in this run")
            elif k not in ref:
                diffs.append(f"{path}{k}: not in reference")
            else:
                diffs += compare(ref[k], got[k], f"{path}{k}.")
    elif isinstance(ref, (int, float)) and isinstance(got, (int, float)):
        if abs(ref - got) > 1e-6 * max(1.0, abs(ref)):
            diffs.append(f"{path[:-1]}: expected {ref}, got {got}")
    elif ref != got:
        diffs.append(f"{path[:-1]}: expected {ref!r}, got {got!r}")
    return diffs


# ----------------------------------------------------------------------------- QGIS
def find_qgis_python():
    """Windows: the QGIS launcher that sets up PyQGIS. Linux/macOS: a python that can import qgis."""
    if os.name == "nt":
        hits = sorted(glob.glob(r"C:\Program Files\QGIS*\bin\python-qgis-ltr.bat") +
                      glob.glob(r"C:\Program Files\QGIS*\bin\python-qgis.bat") +
                      glob.glob(r"C:\OSGeo4W*\bin\python-qgis*.bat"))
        return [hits[-1]] if hits else None
    for exe in (["/usr/bin/python3", shutil.which("python3") or ""] + sorted(glob.glob("/usr/bin/python3.*[0-9]")) +
                ["/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3",
                 "/Applications/QGIS.app/Contents/MacOS/bin/python3"]):
        if exe and os.path.exists(exe):
            ok = subprocess.run([exe, "-c", "import qgis.core"], capture_output=True).returncode == 0
            if ok:
                return [exe]
    return None


def merge_series():
    """maps/01..09 PDFs -> maps/berryessa_map_series.pdf (needs pypdf; skipped quietly without it)."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        return None
    parts = [os.path.join(HERE, "maps", f + ".pdf") for f, _ in MAP_SERIES]
    parts = [p for p in parts if os.path.exists(p)]
    if not parts:
        return None
    out = os.path.join(HERE, "maps", "berryessa_map_series.pdf")
    w = PdfWriter()
    for p in parts:
        w.append(p)
    with open(out, "wb") as fh:
        w.write(fh)
    return out, len(parts)


# ----------------------------------------------------------------------------- report
def write_report(settings, results, verdict, diffs, versions, maps_made, reason=None):
    import pyogrio
    reps = results["representatives"]
    plan = results.get("city_plan")
    titles = {}
    try:
        rt = pyogrio.read_dataframe(GPKG, layer="representatives", read_geometry=False)
        titles = dict(zip(rt.key, rt.title))
    except Exception:
        pass
    fmt_ft = lambda v: "&ndash;" if v is None else f"{v:,.0f}"
    rows = "".join(
        f"<tr><td>{html.escape(titles.get(n, n))}</td><td>#{r['solution_id']}</td><td>{fmt_ft(r.get('job_walk_ft'))}</td>"
        f"<td>{fmt_ft(r.get('services_walk_ft'))}</td><td>{r['livability']:,.1f}</td>"
        f"<td>{fmt_ft(r.get('new_homes_walk_to_bart_ft'))}</td><td>{r.get('new_homes_in_noise_contour')}</td>"
        f"<td>{r.get('new_homes_in_flood_zone')}</td></tr>" for n, r in reps.items())
    if plan:
        rows += (f"<tr class='plan'><td>City's adopted plan (scored)</td><td>&ndash;</td><td>{fmt_ft(plan.get('job_walk_ft'))}</td>"
                 f"<td>{fmt_ft(plan.get('services_walk_ft'))}</td><td>{plan['livability']:,.1f}</td>"
                 f"<td>{fmt_ft(plan.get('new_homes_walk_to_bart_ft'))}</td><td>{plan.get('new_homes_in_noise_contour')}</td>"
                 f"<td>{plan.get('new_homes_in_flood_zone')}</td></tr>")
    grid_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>" for k, v in results["grid_classes"].items())
    qa_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in results["qa_counts"].items() if v)
    census = results.get("residents_2020_within_walk_of_bart") or {}
    census_txt = ", ".join(f"{v:,} within {k.replace('_min', '')} min" for k, v in census.items())
    imgs = ""
    figs = [("pareto_vs_city_plan.png", "Trade-offs among the optimized layouts, with the City's plan marked"),
            ("convergence.png", "Convergence: hypervolume of the trade-off set by generation")]
    if maps_made:
        figs += [(f + ".png", f"Map {f[:2]}: {t}") for f, t in MAP_SERIES] + [("qa_overview.png", "QA/QC overview")]
    for f, cap in figs:
        if os.path.exists(os.path.join(HERE, "maps", f)):
            imgs += f'<figure><a href="maps/{f}"><img src="maps/{f}" alt="{html.escape(cap)}"></a><figcaption>{html.escape(cap)}</figcaption></figure>'
    color = {"PASS": "#1a7f37", "DIFFERENT": "#b35900", "NEW REFERENCE": "#0b5cad"}[verdict]
    diff_html = ("<ul>" + "".join(f"<li>{html.escape(d)}</li>" for d in diffs[:30]) + "</ul>") if diffs else ""
    reason_html = f"<p>Reason recorded: {html.escape(reason)}</p>" if reason else ""
    beat = results.get("layouts_beating_plan_on_services_and_livability")
    beat_txt = (f" {beat} of them beat the City's plan on both services access and livability." if beat is not None else "")
    links = "".join(f'<li><a href="{p}">{html.escape(t)}</a></li>' for p, t in (
        ("webmap/index.html", "Interactive web map"), ("maps/berryessa_map_series.pdf", "Map series (PDF, 9 maps)"),
        ("maps/qa_error_atlas.pdf", "QA error atlas (PDF)"), ("qa_summary.csv", "QA summary (CSV)"),
        ("data/manifest.csv", "Data sources (manifest)")) if os.path.exists(os.path.join(HERE, p)))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Berryessa land use demo run</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1a1a1a;background:#fff}}
table{{border-collapse:collapse;margin:8px 0 20px}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
th{{background:#f2f2f2}}tr.plan td{{background:#fff7e6}}td{{font-variant-numeric:tabular-nums}}
table.num td+td{{text-align:right}}figure{{margin:0 0 24px}}img{{width:100%;border:1px solid #ddd}}
.badge{{display:inline-block;padding:4px 10px;border-radius:4px;color:#fff;background:{color};font-weight:600}}
code{{background:#f4f4f4;padding:1px 4px}}figcaption{{color:#555;font-size:14px}}</style></head><body>
<h1>Berryessa BART Urban Village: reproducible run</h1>
<p>{settings['when']} &middot; seed {settings['seed']} &middot; {settings['gens']} generations &middot; data: {settings['data']}
&middot; Python {versions.get('python')} on {versions.get('platform')}</p>
<p>Check against reference ({settings['mode']}): <span class="badge">{verdict}</span></p>{reason_html}{diff_html}
<ul>{links}</ul>
<h2>Highlighted layouts and the City's plan</h2>
<table class="num"><tr><th>Layout</th><th>Pareto #</th><th>Avg. walk, home to job (ft)</th><th>Avg. walk, home to services (ft)</th>
<th>Livability</th><th>Avg. walk, new home to BART (ft)</th><th>New homes in noise contour (cells)</th>
<th>New homes in 1% flood zone (cells)</th></tr>{rows}</table>
<p>{results['pareto_solutions']} non-dominated layouts.{beat_txt} Walks follow real streets and paths at 3 mph.
{results['manual_edit_cells']} grid cells changed by manual overrides in site_edits.gpkg.
{('2020 Census residents near BART today: ' + census_txt + '.') if census_txt else ''}</p>
<h2>What the optimizer saw (grid cells)</h2><table class="num"><tr><th>Class</th><th>Cells</th></tr>{grid_rows}</table>
<h2>QA/QC findings</h2><table class="num"><tr><th>Check</th><th>Count</th></tr>{qa_rows or '<tr><td colspan=2>None</td></tr>'}</table>
<p>All other checks passed. Full list: <code>qa_summary.csv</code>; one page per Error or Warning: <code>maps/qa_error_atlas.pdf</code>.</p>
<h2>Figures and maps</h2>{imgs}{'' if maps_made else '<p>QGIS maps were not produced in this run.</p>'}
<h2>Package versions</h2><p>{html.escape(', '.join(f'{k} {v}' for k, v in versions.items()))}</p>
</body></html>"""
    with open(os.path.join(HERE, "demo_report.html"), "w", encoding="utf-8") as fh:
        fh.write(page)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--no-basemap", action="store_true", help="maps without satellite imagery (offline)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--update-reference", action="store_true")
    ap.add_argument("--reason", default=None, help="why the reference changed (stored with it)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    gens = QUICK_GENS if args.quick else FULL_GENS
    mode = f"{'quick' if args.quick else 'full'}_seed{args.seed}"

    open(LOG, "w").close()
    log(f"Berryessa land use demo  {dt.datetime.now():%Y-%m-%d %H:%M}  ({mode}, folder {HERE})")

    step(1, "Checking Python packages")
    versions = {"python": platform.python_version(), "platform": platform.system()}
    missing = []
    for m in PACKAGES:
        try:
            versions[m] = getattr(importlib.import_module(m), "__version__", "?")
        except ImportError:
            missing.append(m)
    log("    " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    if missing:
        log(f"    Missing: {', '.join(missing)}. Double-click setup_env.bat, or run:  "
            f"{sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)
    for f in SCRIPTS:
        if not os.path.exists(os.path.join(HERE, f)):
            log(f"    Missing file: {f}")
            sys.exit(1)

    py = [sys.executable, "-u"]
    if args.fetch:
        step(2, "Downloading fresh OpenStreetMap and official data")
        run(py + ["fetch_osm_site.py"])
        run(py + ["fetch_official.py"])
        data_note = "freshly downloaded OpenStreetMap + official layers"
    else:
        step(2, "Using the frozen data snapshots")
        if not os.path.exists(os.path.join(HERE, "site_osm.json")):
            log("    site_osm.json not found; run with --fetch")
            sys.exit(1)
        n_raw = len(glob.glob(os.path.join(HERE, "data", "raw", "*")))
        if not n_raw:
            log("    data/raw is empty: the run falls back to OpenStreetMap only (the October 6 prototype). "
                "Run with --fetch to download the official layers.")
        data_note = f"frozen snapshots (site_osm.json + {n_raw} files in data/raw)"
    log("    " + data_note)

    step(3, f"Site grid, walk network, City's plan and NSGA-III ({gens} generations, seed {args.seed})")
    t, _ = run(py + ["site_pipeline.py", "--gens", str(gens), "--seed", str(args.seed), "--module", OPTIMIZER_FILE])
    log(f"    done in {t:.0f}s")

    step(4, "QA/QC checks")
    run(py + ["gis_qaqc.py"])

    step(5, "Interactive web map")
    run(py + ["make_webmap.py"], fatal=False)

    step(6, "Comparing with the reference results")
    results = collect()
    refs = json.load(open(REFERENCE)) if os.path.exists(REFERENCE) else {}
    diffs = []
    if args.update_reference or mode not in refs:
        refs[mode] = {"results": results, "versions": versions, "data": data_note,
                      "recorded": f"{dt.datetime.now():%Y-%m-%d %H:%M}",
                      "reason": args.reason or ("first run with this mode" if mode not in refs else "")}
        with open(REFERENCE, "w") as fh:
            json.dump(refs, fh, indent=2)
        verdict = "NEW REFERENCE"
        log(f"    Saved this run as the reference for '{mode}' in expected_results.json")
    else:
        diffs = compare(refs[mode]["results"], results)
        verdict = "PASS" if not diffs else "DIFFERENT"
        if diffs:
            log(f"    {len(diffs)} differences from the reference:")
            for d in diffs[:15]:
                log("      - " + d)
            rv = refs[mode].get("versions", {})
            changed = [k for k in ("pymoo", "numpy", "scipy", "shapely") if rv.get(k) != versions.get(k)]
            if changed:
                log(f"    Note: package versions differ from the reference run ({', '.join(changed)}); "
                    "that alone can change results.")
            if args.fetch:
                log("    Expected: --fetch downloaded new data.")
        else:
            log("    PASS: identical to the reference results")

    maps_made = False
    if args.no_maps:
        step(7, "Maps skipped (--no-maps)")
    else:
        step(7, "QGIS project, map series and QA maps (headless)")
        qpy = find_qgis_python()
        if not qpy:
            log("    QGIS Python not found; skipping maps. Run qgis_site_map.py and qgis_qaqc_review.py "
                "from the QGIS Python Console instead.")
        else:
            env = dict(os.environ, LANDUSE_PROJECT_DIR=HERE)
            for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE"):     # the .venv's must not leak into QGIS
                env.pop(k, None)
            if args.no_basemap:
                env["LANDUSE_NO_BASEMAP"] = "1"
            if os.name != "nt":
                env.setdefault("QT_QPA_PLATFORM", "offscreen")
            t, rc = run(qpy + [os.path.join(HERE, "qgis_export_all.py")], env=env, fatal=False)
            maps_made = rc == 0
            merged = merge_series() if maps_made else None
            if merged:
                log(f"    Map series merged into {merged[0]} ({merged[1]} maps)")
            log(f"    done in {t:.0f}s; maps in {os.path.join(HERE, 'maps')}")

    step(8, "Writing demo_report.html")
    write_report({"when": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "seed": args.seed, "gens": gens, "data": data_note,
                  "mode": mode}, results, verdict, diffs, versions, maps_made,
                 refs.get(mode, {}).get("reason") if verdict == "NEW REFERENCE" else None)
    log(f"    Open {os.path.join(HERE, 'demo_report.html')} in a browser.")
    log(f"\nFinished: {verdict}")
    sys.exit(0 if verdict != "DIFFERENT" else 2)


if __name__ == "__main__":
    main()
