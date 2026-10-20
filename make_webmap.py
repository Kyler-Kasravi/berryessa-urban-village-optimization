"""
Interactive web map of the results (roadmap item 5): webmap/index.html
=====================================================================

One self-contained HTML page (Leaflet, data inlined as GeoJSON in WGS84) that opens from disk, from
GitHub Pages, or as a claude.ai artifact (webmap/fragment.html is the same page with the libraries
loaded from cdnjs, for hosting). Switch between the four highlighted optimized layouts, the
City's plan, and a cell-by-cell difference; turn on walk-time bands, rules, transit stops and QA
findings; click any cell for its details.

Usage:  python make_webmap.py          (after site_pipeline.py and gis_qaqc.py)
"""
import json
import os

import geopandas as gpd
import pandas as pd
import pyogrio

HERE = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(HERE, "berryessa.gpkg")
OUT = os.path.join(HERE, "webmap", "index.html")
LEAFLET_CSS = os.path.join(HERE, "webmap", "leaflet.css")
CDN_LEAFLET = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"
CDN_PROJ4 = "https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.9.0/proj4.js"
LU = {0: ["Residential", "#F2D95C"], 1: ["Parks & Open Space", "#5DAE4B"], 2: ["Commercial / Mixed Use", "#D9534F"],
      3: ["Light Industrial", "#8E6BB8"], 4: ["Public & Institutional", "#3F7FCF"]}


def gj(gdf, cols, simplify=0.0, digits=6):
    """GeoJSON in WGS84 with only the columns the page uses (coordinates rounded to ~10 cm)."""
    if gdf is None or len(gdf) == 0:
        return {"type": "FeatureCollection", "features": []}
    g = gdf[[c for c in cols if c in gdf.columns] + ["geometry"]].copy()
    if simplify:
        g["geometry"] = g.geometry.simplify(simplify, preserve_topology=True)
    g = g.to_crs(4326)
    data = json.loads(g.to_json(drop_id=True))

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(v, digits) for v in c]
    for f in data["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
        f["properties"] = {k: (None if isinstance(v, float) and v != v else v) for k, v in f["properties"].items()}
    return data


def layer(name):
    try:
        return gpd.read_file(GPKG, layer=name)
    except Exception:
        return None


def table(name):
    try:
        return pyogrio.read_dataframe(GPKG, layer=name, read_geometry=False)
    except Exception:
        return None


def clean(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def main():
    reps = table("representatives").sort_values("order")
    pareto = table("pareto_front")
    plan_t = table("plan_comparison")
    walk = layer("walk_bart")
    summary = json.load(open(os.path.join(HERE, "run_summary.json"))) if os.path.exists(os.path.join(HERE, "run_summary.json")) else {}
    metric_keys = ["job_walk_ft", "services_walk_ft", "livability", "new_home_cells", "new_homes_5min_bart",
                   "new_homes_10min_bart", "new_homes_walk_to_bart_ft",
                   "new_homes_in_noise_contour", "new_homes_in_flood_zone", "pct_homes_10min_park",
                   "pct_homes_10min_grocery", "pct_homes_10min_school"]
    scenarios = []
    for _, r in reps.iterrows():
        row = pareto[pareto.solution_id == r.solution_id].iloc[0]
        scenarios.append({"key": r.key, "title": r.title, "solution_id": int(r.solution_id),
                          "metrics": {k: clean(row.get(k)) for k in metric_keys},
                          "cells": gj(layer(f"{r.key}_cells"), ["lu_code", "walk_min_bart"])})
    if plan_t is not None:
        p = plan_t[plan_t.scenario == "city_plan"].iloc[0]
        scenarios.append({"key": "city_plan", "title": "City's plan", "solution_id": None,
                          "metrics": {k: clean(p.get(k)) for k in metric_keys},
                          "cells": gj(layer("city_plan_cells"), ["lu_code", "walk_min_bart", "gp_designation"])})
    rules = layer("rules")
    qa = layer("qa_errors")
    if qa is not None:
        qa = qa[qa.severity.isin(["Error", "Warning"])]
    data = {
        "scenarios": scenarios,
        "diff": gj(layer("plan_vs_balanced"), ["plan_use", "balanced_use", "same"]),
        "uv": gj(layer("uv_boundary"), ["name", "acres"]),
        "site": gj(layer("site_boundary"), ["acres"]),
        "walk": gj(walk, ["minutes", "label", "residents_2020_within", "homes_2020_within"], simplify=12),
        "rules": gj(rules, ["rule_id", "enforcement", "description", "acres"], simplify=6),
        "stops": gj(layer("transit_stops"), ["stop_name", "agency", "mode", "n_arrivals"]),
        "qa": gj(qa, ["check_code", "severity", "description", "status"]),
        "streets": gj(layer("city_streets"), ["FULLNAME"], simplify=3),
        "creeks": gj(layer("creeks"), ["STREAMNAME", "COVEREDSTATUS"], simplify=3),
        "parks": gj(layer("osm_parks"), ["type"], simplify=3),
        "parcels": gj(layer("parcels_uv")[lambda d: d.redevelopable == 1] if layer("parcels_uv") is not None else None,
                      ["APN", "reason", "bldg_coverage"], simplify=2),
        "lu": {str(k): v for k, v in LU.items()},
        "run": {k: summary.get(k) for k in ("distance", "program", "flood_rule", "noise_rule", "pareto_solutions",
                                             "osm_base")},
    }
    css = open(LEAFLET_CSS, encoding="utf-8").read() if os.path.exists(LEAFLET_CSS) else ""
    page = TEMPLATE.replace("/*LEAFLET_CSS*/", css).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # artifact / web-hosting fragment: libraries from cdnjs
    with open(os.path.join(os.path.dirname(OUT), "fragment.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    # local page: a complete document with Leaflet and proj4 inlined, so it opens offline from disk
    # (only the basemap tiles need the internet; without them the vector layers still draw)
    for lib, url in (("leaflet.js", CDN_LEAFLET), ("proj4.js", CDN_PROJ4)):
        p = os.path.join(os.path.dirname(OUT), lib)
        if os.path.exists(p):
            code = open(p, encoding="utf-8").read().replace("</script", "<\\/script")
            page = page.replace(f'<script src="{url}"></script>', f"<script>{code}</script>")
    cut = page.index("</style>") + len("</style>")           # title, fonts and styles go in <head>
    doc = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">\n' + page[:cut] +
           "\n</head><body>" + page[cut:] + "\n</body></html>\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB): {len(scenarios)} scenarios, "
          f"{len(data['qa']['features'])} QA findings, {len(data['stops']['features'])} transit stops")


TEMPLATE = r"""<title>Berryessa Land Use Explorer</title>
<meta name="description" content="NSGA-III land use layouts for the Berryessa BART Urban Village, San Jose, compared with the City's adopted plan.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/*LEAFLET_CSS*/
/* Layout: full-height map with a docked control sheet on the left (stacks under the map on phones). */
:root {
  --paper: #eef1f4; --panel: #ffffff; --ink: #17212b; --muted: #5a6877; --line: #d3dae2;
  --accent: #1f5a86; --accent-ink: #ffffff; --chip: #e6edf3; --map-bg: #e4e8ec; --halo: rgba(255,255,255,.9);
  --street: #9aa6b2; --creek: #3b82c4; --better: #23794a; --worse: #b4462a;
  --display: "Barlow Semi Condensed", "Arial Narrow", Arial, sans-serif;
  --body: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
  --mono: "IBM Plex Mono", Consolas, "Courier New", monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #0e141a; --panel: #151d25; --ink: #e5ebf1; --muted: #98a6b5; --line: #2a3541;
    --accent: #7db7e3; --accent-ink: #0e141a; --chip: #1d2731; --map-bg: #1a222b; --halo: rgba(14,20,26,.85);
    --street: #5d6b79; --creek: #5fa8e6; --better: #5cc98a; --worse: #f08a6c; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #0e141a; --panel: #151d25; --ink: #e5ebf1; --muted: #98a6b5; --line: #2a3541;
  --accent: #7db7e3; --accent-ink: #0e141a; --chip: #1d2731; --map-bg: #1a222b; --halo: rgba(14,20,26,.85);
  --street: #5d6b79; --creek: #5fa8e6; --better: #5cc98a; --worse: #f08a6c; color-scheme: dark;
}
html, body { height: 100%; }
body { background: var(--paper); color: var(--ink); font: 14px/1.45 var(--body); }
.app { display: grid; grid-template-columns: minmax(300px, 380px) 1fr; height: 100%; }
.sheet { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; padding-block: 18px 24px;
  padding-inline: 18px; display: flex; flex-direction: column; gap: 18px; min-width: 0; }
#map { background: var(--map-bg); min-height: 320px; }
.eyebrow { font: 600 11px/1 var(--body); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font: 700 28px/1.05 var(--display); margin: 6px 0 6px; text-wrap: balance; }
.lede { color: var(--muted); margin: 0; max-width: 60ch; }
h2 { font: 600 17px/1.2 var(--display); margin: 0 0 8px; letter-spacing: .01em; }
.seg { display: flex; flex-wrap: wrap; gap: 6px; }
.seg button { font: 500 13px/1 var(--body); padding: 8px 10px; border-radius: 4px; border: 1px solid var(--line);
  background: var(--chip); color: var(--ink); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.seg button:focus-visible, label:focus-within { outline: 2px solid var(--accent); outline-offset: 2px; }
table.m { width: 100%; border-collapse: collapse; font-size: 13px; }
table.m th, table.m td { padding: 5px 0; border-bottom: 1px solid var(--line); text-align: right; }
table.m th:first-child, table.m td:first-child { text-align: left; font-weight: 400; color: var(--muted); }
table.m thead th { font: 600 11px/1.2 var(--body); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
table.m td { font-family: var(--mono); font-variant-numeric: tabular-nums; }
td.better { color: var(--better); } td.worse { color: var(--worse); }
.legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 6px 12px; font-size: 13px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 7px; vertical-align: -1px;
  border: 1px solid rgba(0,0,0,.15); }
.toggles { display: grid; gap: 6px; }
.toggles label { display: flex; gap: 8px; align-items: baseline; cursor: pointer; }
.toggles small { color: var(--muted); }
.note { font-size: 12px; color: var(--muted); margin: 0; }
.note code { font-family: var(--mono); font-size: 11.5px; }
.readout { position: absolute; z-index: 500; right: 10px; bottom: 26px; background: var(--halo); color: var(--ink);
  font: 12px/1.3 var(--mono); padding: 5px 8px; border-radius: 3px; pointer-events: none; }
.mapwrap { position: relative; min-width: 0; }
.leaflet-container { font-family: var(--body); background: var(--map-bg); }
.leaflet-popup-content-wrapper, .leaflet-popup-tip { background: var(--panel); color: var(--ink); }
.leaflet-popup-content { font-size: 13px; margin: 10px 12px; }
.leaflet-popup-content b { font-family: var(--display); font-size: 15px; }
.leaflet-control-scale-line { background: var(--halo); color: var(--ink); border-color: var(--ink); }
.leaflet-bar a { background: var(--panel); color: var(--ink); border-color: var(--line); }
@media (max-width: 760px) {
  .app { grid-template-columns: 1fr; grid-template-rows: 58vh auto; height: auto; }
  .mapwrap { grid-row: 1; height: 58vh; } #map { height: 100%; }
  .sheet { grid-row: 2; border-right: 0; border-top: 1px solid var(--line); overflow: visible; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
<div class="app">
  <aside class="sheet">
    <header>
      <div class="eyebrow">San Jose &middot; Berryessa / North San Jose BART</div>
      <h1>Berryessa Land Use Explorer</h1>
      <p class="lede">Land use layouts from my NSGA-III optimizer for the redevelopable land in the City's Berryessa BART
      Urban Village, next to the City's adopted plan scored with the same functions. Each square is a 120 ft grid cell.</p>
    </header>
    <section>
      <h2>Show</h2>
      <div class="seg" id="seg" role="group" aria-label="Layout shown on the map"></div>
    </section>
    <section>
      <h2 id="mtitle">Balanced vs. City's plan</h2>
      <table class="m" id="metrics"></table>
      <p class="note" id="mnote"></p>
    </section>
    <section>
      <h2>Land use</h2>
      <div class="legend" id="legend"></div>
    </section>
    <section>
      <h2>Layers</h2>
      <div class="toggles" id="toggles"></div>
    </section>
    <section>
      <p class="note">Walking distances follow real streets and paths at 3 mph (264 ft per minute); a 10-minute walk is
      half a mile. Analysis in NAD83 / California State Plane Zone III, US survey feet (EPSG:2227); the pointer readout
      shows those coordinates. Data: City of San Jose GIS, FEMA NFHL, US Census 2020, Caltrans GTFS stops,
      &copy; OpenStreetMap contributors. Illustrative planning exercise by Kyler Kasravi, not an adopted or proposed plan.</p>
    </section>
  </aside>
  <div class="mapwrap"><div id="map" style="height:100%"></div><div class="readout" id="readout">EPSG:2227 &middot; move the pointer</div></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.9.0/proj4.js"></script>
<script>
const D = __DATA__;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const map = L.map("map", { zoomControl: true, preferCanvas: false, attributionControl: true });
map.attributionControl.setPrefix(false);
map.attributionControl.addAttribution("City of San Jose GIS &middot; FEMA NFHL &middot; US Census &middot; Caltrans &middot; &copy; OpenStreetMap contributors");
const base = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
  maxZoom: 19, maxNativeZoom: 16, attribution: "Basemap &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors" });
let tileFailures = 0;
base.on("tileerror", () => { if (++tileFailures === 4) map.removeLayer(base); });   // blocked host: vector context only
base.addTo(map);
L.control.scale({ imperial: true, metric: false, maxWidth: 160 }).addTo(map);

const fmtFt = (v) => v == null ? "&ndash;" : Math.round(v).toLocaleString() + " ft";
const fmt1 = (v) => v == null ? "&ndash;" : (+v).toFixed(1);
const fmt0 = (v) => v == null ? "&ndash;" : Math.round(v).toLocaleString();
const pct = (v) => v == null ? "&ndash;" : (+v).toFixed(0) + "%";

// context layers (always available, independent of the basemap)
const ctx = {
  streets: L.geoJSON(D.streets, { style: () => ({ color: css("--street"), weight: 1, opacity: .9 }), interactive: false }),
  creeks: L.geoJSON(D.creeks, { style: (f) => ({ color: css("--creek"), weight: 2.2, dashArray: f.properties.COVEREDSTATUS === "Covered" ? "4 4" : null }),
    onEachFeature: (f, l) => l.bindTooltip(f.properties.STREAMNAME || "Creek") }),
  parks: L.geoJSON(D.parks, { style: () => ({ color: "#4f9a41", weight: 1, fillColor: "#5DAE4B", fillOpacity: .18 }), interactive: false }),
};
const uv = L.geoJSON(D.uv, { style: () => ({ color: "#c0392b", weight: 2.4, dashArray: "8 5", fill: false }), interactive: false });
const site = L.geoJSON(D.site, { style: () => ({ color: "#e69500", weight: 1.6, fill: false }), interactive: false });

const ruleStyle = { R1_CREEK_SETBACK: ["#2f6fbf", .30], R2_FLOODWAY: ["#163f8f", .45], R3_SFHA: ["#5a96e6", .22],
  R4_RAIL_NOISE: ["#c43c9c", .16], R7_FLOOD_X: ["#8db7ee", .14] };
const rules = L.geoJSON(D.rules, { style: (f) => { const s = ruleStyle[f.properties.rule_id] || ["#777", .2];
  return { color: s[0], weight: 1, fillColor: s[0], fillOpacity: s[1] }; },
  onEachFeature: (f, l) => l.bindPopup(`<b>${f.properties.rule_id.replace(/_/g, " ")}</b><br>${f.properties.description}<br>
    Enforcement: ${f.properties.enforcement} &middot; ${f.properties.acres} ac in frame`) });
const walkCol = { 5: "#00683a", 10: "#31a354", 15: "#a1d99b" };
const walk = L.geoJSON(D.walk, { style: (f) => ({ color: walkCol[f.properties.minutes], weight: 0, fillColor: walkCol[f.properties.minutes], fillOpacity: .32 }),
  onEachFeature: (f, l) => l.bindPopup(`<b>${f.properties.label} to BART</b><br>${fmt0(f.properties.residents_2020_within)} residents and
    ${fmt0(f.properties.homes_2020_within)} homes within this time (2020 Census, area-weighted)`) });
const stops = L.geoJSON(D.stops, { pointToLayer: (f, ll) => L.circleMarker(ll, { radius: f.properties.mode === "BART" ? 7 : 4,
  color: "#fff", weight: 1.2, fillColor: f.properties.mode === "BART" ? "#0079be" : "#4b5563", fillOpacity: 1 }),
  onEachFeature: (f, l) => l.bindPopup(`<b>${f.properties.stop_name}</b><br>${f.properties.mode} &middot; ${fmt0(f.properties.n_arrivals)} arrivals a day (GTFS)`) });
const sevCol = { Error: "#d62728", Warning: "#ff9800" };
const qa = L.geoJSON(D.qa, { pointToLayer: (f, ll) => L.circleMarker(ll, { radius: 5, color: "#fff", weight: 1, fillColor: sevCol[f.properties.severity] || "#2196f3", fillOpacity: 1 }),
  onEachFeature: (f, l) => l.bindPopup(`<b>${f.properties.check_code}</b> (${f.properties.severity})<br>${f.properties.description}<br>Status: ${f.properties.status}`) });
const parcels = L.geoJSON(D.parcels, { style: () => ({ color: "#e69500", weight: 1, fillColor: "#ffb000", fillOpacity: .12 }),
  onEachFeature: (f, l) => l.bindPopup(`<b>APN ${f.properties.APN}</b><br>Redevelopable: ${f.properties.reason}<br>Building coverage ${Math.round(100 * f.properties.bldg_coverage)}%`) });

// scenarios
const scen = {};
D.scenarios.forEach((s) => {
  scen[s.key] = L.geoJSON(s.cells, { style: (f) => ({ color: "#ffffff", weight: .4, fillColor: D.lu[f.properties.lu_code][1], fillOpacity: .82 }),
    onEachFeature: (f, l) => l.bindPopup(`<b>${D.lu[f.properties.lu_code][0]}</b><br>${s.title}` +
      (f.properties.gp_designation ? `<br>General Plan: ${f.properties.gp_designation}` : "") +
      `<br>Walk to BART: ${f.properties.walk_min_bart == null ? "&ndash;" : f.properties.walk_min_bart + " min"}`) });
});
scen.diff = L.geoJSON(D.diff, { style: (f) => ({ color: "#ffffff", weight: .4, fillColor: f.properties.same ? "#b9c2cc" : "#e8590c", fillOpacity: f.properties.same ? .55 : .85 }),
  onEachFeature: (f, l) => l.bindPopup(`<b>${f.properties.same ? "Same use" : "Different use"}</b><br>Balanced: ${f.properties.balanced_use}<br>City's plan: ${f.properties.plan_use}`) });

const order = D.scenarios.map((s) => [s.key, s.key === "city_plan" ? "City's plan" : s.title]).concat([["diff", "Difference"]]);
const seg = document.getElementById("seg");
let current = null;
function show(key) {
  if (current) map.removeLayer(scen[current]);
  current = key; scen[key].addTo(map); site.bringToFront(); uv.bringToFront();
  [...seg.children].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
  renderMetrics(key);
  try { localStorage.setItem("berryessa-scenario", key); } catch (e) {}
}
order.forEach(([k, t]) => { const b = document.createElement("button"); b.type = "button"; b.dataset.key = k; b.id = "seg-" + k;
  b.textContent = t; b.addEventListener("click", () => show(k)); seg.appendChild(b); });

const plan = D.scenarios.find((s) => s.key === "city_plan");
const rows = [["Avg. walk, home to job", "job_walk_ft", fmtFt, -1], ["Avg. walk, home to services", "services_walk_ft", fmtFt, -1],
  ["Livability score", "livability", fmt1, 1], ["New-home cells", "new_home_cells", fmt0, 0],
  ["&hellip;within 5 min of BART", "new_homes_5min_bart", fmt0, 1], ["&hellip;within 10 min of BART", "new_homes_10min_bart", fmt0, 1],
  ["&hellip;in rail noise contour", "new_homes_in_noise_contour", fmt0, -1],
  ["&hellip;in FEMA 1% flood zone", "new_homes_in_flood_zone", fmt0, -1], ["Avg. walk, new home to BART", "new_homes_walk_to_bart_ft", fmtFt, -1],
  ["Homes &le;10 min to a park", "pct_homes_10min_park", pct, 1], ["Homes &le;10 min to a grocery", "pct_homes_10min_grocery", pct, 1]];
function renderMetrics(key) {
  const s = key === "diff" ? D.scenarios[0] : D.scenarios.find((x) => x.key === key);
  const t = document.getElementById("metrics");
  const same = s.key === "city_plan";
  document.getElementById("mtitle").textContent = same ? "City's plan, scored by the model" : `${s.title} vs. City's plan`;
  let h = `<thead><tr><th></th><th>${same ? "City's plan" : "Layout"}</th>${same ? "" : "<th>City's plan</th>"}</tr></thead><tbody>`;
  rows.forEach(([lab, k, f, dir]) => {
    const a = s.metrics[k], b = plan ? plan.metrics[k] : null;
    let cls = "";
    if (!same && dir && a != null && b != null && a !== b) cls = (dir > 0 ? a > b : a < b) ? "better" : "worse";
    h += `<tr><th>${lab}</th><td class="${cls}">${f(a)}</td>${same ? "" : `<td>${f(b)}</td>`}</tr>`;
  });
  t.innerHTML = h + "</tbody>";
  document.getElementById("mnote").textContent = same ? "General Plan 2040 designations mapped to the five model land uses."
    : `Pareto solution #${s.solution_id} of ${D.run.pareto_solutions}. Green = better than the plan, orange = worse. Both use the same amount of each land use.`;
}

// legend + toggles
const lg = document.getElementById("legend");
const used = new Set(D.scenarios.flatMap((s) => s.cells.features.map((f) => String(f.properties.lu_code))));
Object.entries(D.lu).filter(([k]) => used.has(k)).forEach(([, [n, c]]) => { const d = document.createElement("div");
  d.innerHTML = `<span class="sw" style="background:${c}"></span>${n}`; lg.appendChild(d); });
const tg = document.getElementById("toggles");
[["uv", "Urban Village boundary (City)", uv, true], ["site", "Redevelopment site", site, true],
 ["walk", "Walk time to BART", walk, false, "5 / 10 / 15 min"], ["rules", "Rules", rules, false, "creek setback, floodway, flood zone, noise"],
 ["stops", "Transit stops (GTFS)", stops, true], ["parcels", "Redevelopable parcels", parcels, false],
 ["qa", "QA findings", qa, false, "errors and warnings"], ["streets", "Streets (City)", ctx.streets, true],
 ["creeks", "Creeks", ctx.creeks, true], ["parks", "Existing parks", ctx.parks, true]].forEach(([id, label, lyr, on, hint]) => {
  const l = document.createElement("label");
  l.innerHTML = `<input type="checkbox" id="t-${id}" ${on ? "checked" : ""}><span>${label}${hint ? ` <small>${hint}</small>` : ""}</span>`;
  const cb = l.querySelector("input");
  cb.addEventListener("change", () => { cb.checked ? lyr.addTo(map) : map.removeLayer(lyr); site.bringToFront(); uv.bringToFront(); });
  if (on) lyr.addTo(map);
  tg.appendChild(l);
});

map.fitBounds(uv.getBounds(), { padding: [12, 12] });
let start = "balanced";
try { const k = localStorage.getItem("berryessa-scenario"); if (k && scen[k]) start = k; } catch (e) {}
show(start);

// pointer readout in California State Plane Zone III (US ft)
if (window.proj4) {
  proj4.defs("EPSG:2227", "+proj=lcc +lat_0=36.5 +lon_0=-120.5 +lat_1=38.4333333333333 +lat_2=37.0666666666667 +x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs");
  const ro = document.getElementById("readout");
  map.on("mousemove", (e) => { const [x, y] = proj4("EPSG:4326", "EPSG:2227", [e.latlng.lng, e.latlng.lat]);
    ro.textContent = `EPSG:2227  E ${Math.round(x).toLocaleString()}  N ${Math.round(y).toLocaleString()} ft`; });
}
</script>
"""

if __name__ == "__main__":
    main()
