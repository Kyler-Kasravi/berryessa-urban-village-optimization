"""
Builds the Berryessa QGIS project and the map series (one print-layout template, many maps), then
exports each map to PDF + PNG.

Run inside QGIS: Plugins > Python Console, then
    exec(open(r"C:\\Users\\gradd\\Desktop\\nsga3_city_test\\qgis_site_map.py").read())
(or let run_demo.py do it headless). Inputs: berryessa.gpkg from site_pipeline.py. Safe to re-run.

Map series (maps/NN_*.pdf / .png). Every map has the same frame: title block, legend, scale bar in feet,
north arrow, a sources line, and the coordinate system note (NAD83 / California State Plane Zone III,
US survey feet, EPSG:2227).
    01 Existing conditions            05-08 Optimized layouts (balanced, best job access,
    02 Rules and constraints                best livability, most homes near BART)
    03 Walk access to BART            09 Balanced layout vs. the City's plan
    04 The City's adopted plan
"""
import os
from collections import defaultdict
from qgis.core import (Qgis, QgsProject, QgsVectorLayer, QgsRasterLayer, QgsCoordinateReferenceSystem,
                       QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel, QgsLayoutItemLegend,
                       QgsLayoutItemScaleBar, QgsLayoutItemPicture, QgsLayoutPoint, QgsLayoutSize,
                       QgsLayoutExporter, QgsTextFormat, QgsSymbolLayerUtils, QgsLegendStyle,
                       QgsMapLayerLegendUtils, QgsLayerMetadata, QgsLayoutItemShape, QgsFillSymbol)
from qgis.PyQt.QtGui import QFont, QColor
from qgis.PyQt.QtCore import Qt

FOLDER = os.environ.get("LANDUSE_PROJECT_DIR", r"C:\Users\gradd\Desktop\nsga3_city_test")
GPKG = os.path.join(FOLDER, "berryessa.gpkg")
PROJECT_FILE = os.path.join(FOLDER, "berryessa_map.qgz")
OUT_DIR = os.path.join(FOLDER, "maps")
AUTHOR = "Kyler Kasravi"
USE_BASEMAP = os.environ.get("LANDUSE_NO_BASEMAP") != "1"   # run_demo.py --no-basemap turns it off
CRS_NOTE = "NAD83 / California State Plane Zone III, US survey feet (EPSG:2227)"
MM = Qgis.LayoutUnit.Millimeters if hasattr(Qgis, "LayoutUnit") else 0
RECTANGLE = getattr(QgsLayoutItemShape, "Rectangle", None)          # name moved in newer QGIS releases
if RECTANGLE is None:
    RECTANGLE = QgsLayoutItemShape.Shape.Rectangle

GROUPS = ("Proposed land use", "Alternative solutions", "City's plan", "Walk access", "Rules and constraints",
          "Official data", "OpenStreetMap context", "Basemap")
OLD_GROUPS = ("Real-world constraints (OpenStreetMap)",)       # from the October 6 version

project = QgsProject.instance()
root = project.layerTreeRoot()
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(GPKG):
    raise FileNotFoundError(f"{GPKG} not found; run site_pipeline.py first")

# ---------------------------------------------------------------- layers
manager = project.layoutManager()
for lay in list(manager.layouts()):                    # layouts first: they reference the layers below
    if lay.name().startswith(("Map ", "Site - ")):
        manager.removeLayout(lay)
for name in GROUPS + OLD_GROUPS:
    grp = root.findGroup(name)
    if grp:
        for lyr in [n.layer() for n in grp.findLayers()]:
            project.removeMapLayer(lyr.id())
        root.removeChildNode(grp)

meta_layer = QgsVectorLayer(f"{GPKG}|layername=layer_metadata", "meta", "ogr")
META = {f["layer"]: {k: f[k] for k in ("title", "abstract", "source", "crs", "produced_by", "produced")}
        for f in meta_layer.getFeatures()} if meta_layer.isValid() else {}


def apply_metadata(lyr, layer_name):
    m = META.get(layer_name)
    if not m:
        return
    md = QgsLayerMetadata()
    md.setTitle(m["title"] or layer_name)
    md.setAbstract(f"{m['abstract']}\n\nSource: {m['source']}\nProduced by {m['produced_by']} on {m['produced']}.")
    md.setRights([f"Source: {m['source']}"])
    md.setCrs(QgsCoordinateReferenceSystem("EPSG:2227"))
    lyr.setMetadata(md)


ALL = {}


def vec(layer, title, group, visible=True):
    lyr = QgsVectorLayer(f"{GPKG}|layername={layer}", title, "ogr")
    if not lyr.isValid():
        return None
    apply_metadata(lyr, layer)
    project.addMapLayer(lyr, False)
    group.addLayer(lyr).setItemVisibilityChecked(visible)
    ALL[layer] = lyr
    return lyr


rep_layer = QgsVectorLayer(f"{GPKG}|layername=representatives", "reps", "ogr")
REPS = sorted(((f["order"], f["key"], f["title"], f["solution_id"]) for f in rep_layer.getFeatures()))

g_main = root.insertGroup(0, GROUPS[0])
station = vec("osm_stations", "BART station", g_main)
key0, title0 = REPS[0][1], REPS[0][2]
vec(f"{key0}_buildings", "Building footprints", g_main)
vec(f"{key0}_zones", f"Proposed land use ({title0})", g_main)
site = vec("site_boundary", "Redevelopment site", g_main)
uvb = vec("uv_boundary", "Urban Village boundary (City)", g_main)

g_alt = root.insertGroup(1, GROUPS[1])
for _, key, title, _sid in REPS[1:]:
    sub = g_alt.addGroup(title)
    vec(f"{key}_buildings", "Building footprints", sub)
    vec(f"{key}_zones", "Proposed land use", sub)
    sub.setItemVisibilityChecked(False)
    sub.setExpanded(False)
pareto = vec("pareto_front", "Pareto front (all solutions)", g_alt)

g_plan = root.insertGroup(2, GROUPS[2])
g_plan.setItemVisibilityChecked(False)
vec("plan_vs_balanced", "Balanced layout vs. City's plan", g_plan, visible=False)
vec("city_plan_zones", "City's plan on the model grid", g_plan)
vec("gp_landuse", "General Plan 2040 designations", g_plan, visible=False)
plan_tbl = vec("plan_comparison", "City's plan scored by the model", g_plan)

g_walk = root.insertGroup(3, GROUPS[3])
g_walk.setItemVisibilityChecked(False)
vec("walk_destinations", "Walk destinations", g_walk)
vec("walk_network", "Walk network (OSM)", g_walk)
vec("walk_bart", "Walk time to BART", g_walk)

g_rules = root.insertGroup(4, GROUPS[4])
vec("creeks", "Creeks (City / Valley Water)", g_rules)
vec("rules", "Rules as geometry", g_rules, visible=False)
vec("noise_contours", "General Plan noise contours", g_rules, visible=False)
vec("fema_flood", "FEMA flood hazard zones", g_rules, visible=False)
cgrid = vec("constraint_grid", "Constraint grid (what the optimizer saw)", g_rules, visible=False)
vec("rules_table", "Rules table", g_rules)

g_off = root.insertGroup(5, GROUPS[5])
g_off.setItemVisibilityChecked(False)
vec("transit_stops", "Transit stops (GTFS)", g_off)
vec("city_streets", "Street centerlines (City)", g_off)
vec("parcels_uv", "Urban Village parcels", g_off)
vec("zoning", "Zoning districts (City)", g_off, visible=False)
vec("study_frame", "Study frame", g_off)

g_osm = root.insertGroup(6, GROUPS[6])
vec("osm_water", "Creek + riparian setback", g_osm)
vec("osm_parks", "Existing parks", g_osm)
vec("osm_essential", "Existing schools / public", g_osm)
vec("osm_roads", "Street centerlines (OSM)", g_osm, visible=False)

g_base = root.addGroup(GROUPS[7])
imagery = light = None
if USE_BASEMAP:
    imagery = QgsRasterLayer("type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
                             "MapServer/tile/{z}/{y}/{x}&zmin=0&zmax=19", "Esri World Imagery", "wms")
    # CARTO's free light tiles now return "API KEY REQUIRED" images; Esri Light Gray needs no key (tiles up to z16)
    light = QgsRasterLayer("type=xyz&url=https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                           "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}&zmin=0&zmax=16",
                           "Esri World Light Gray", "wms")
    for b in (imagery, light):
        project.addMapLayer(b, False)
        g_base.addLayer(b)
    g_base.findLayer(light.id()).setItemVisibilityChecked(False)

project.setCrs(QgsCoordinateReferenceSystem("EPSG:2227"))
frame_layer = ALL.get("study_frame") or cgrid
extent = frame_layer.extent()
try:
    iface.mapCanvas().setExtent(extent)  # noqa: F821
    iface.mapCanvas().refresh()          # noqa: F821
except NameError:
    pass

# ---------------------------------------------------------------- stats
def rows(lyr):
    if lyr is None:
        return []
    names = [f.name() for f in lyr.fields()]
    return [dict(zip(names, f.attributes())) for f in lyr.getFeatures()]


P_ROWS = rows(pareto)
PLAN = {r["scenario"]: r for r in rows(plan_tbl)}
RULES = rows(ALL.get("rules_table"))
WALK = sorted(rows(ALL.get("walk_bart")), key=lambda r: r["minutes"])


def stats_for(sid):
    return next(r for r in P_ROWS if r["solution_id"] == sid)


def acres_by_use(zones):
    out = defaultdict(float)
    for f in zones.getFeatures():
        out[f["land_use"]] += f["acres"]
    return out


def ft(v):
    return "n/a" if v is None else f"{v:,.0f} ft"


def mins(v):
    return "n/a" if v is None else f"{v / 264:.1f} min"


# ---------------------------------------------------------------- layout template
def fmt(size, bold=False, color="#1a1a1a", family="Arial"):
    t = QgsTextFormat()
    f = QFont(family)
    f.setBold(bold)
    t.setFont(f)
    t.setSize(size)
    t.setColor(QColor(color))
    return t


def label(layout, text, x, y, w, h, size, bold=False, color="#1a1a1a", mono=False):
    item = QgsLayoutItemLabel(layout)
    item.setText(text)
    item.setTextFormat(fmt(size, bold, color, "Courier New" if mono else "Arial"))
    item.setVAlign(Qt.AlignTop)
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y, MM))
    item.attemptResize(QgsLayoutSize(w, h, MM))
    return item


SERIES = []          # (number, key, title) for the series index


def map_page(number, key, title, subtitle, layers, legend, side_title, side_text, sources, basemap="imagery",
             zoom=None, side_mono=False):
    """One page of the series. layers: top-to-bottom list for the map; legend: [(layer, label, keep_values)]."""
    name = f"Map {number:02d} - {title}"
    old = manager.layoutByName(name)
    if old:
        manager.removeLayout(old)
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(name)
    manager.addLayout(layout)

    band = QgsLayoutItemShape(layout)                     # title band
    band.setShapeType(RECTANGLE)
    band.setSymbol(QgsFillSymbol.createSimple({"color": "31,58,95,255", "outline_style": "no"}))
    layout.addLayoutItem(band)
    band.attemptMove(QgsLayoutPoint(0, 0, MM))
    band.attemptResize(QgsLayoutSize(297, 5, MM))
    label(layout, f"MAP {number:02d}", 10, 8, 30, 6, 9, bold=True, color="#1F3A5F")
    label(layout, f"Berryessa BART Urban Village: {title}", 10, 13, 277, 10, 16, bold=True)
    label(layout, subtitle, 10, 22, 277, 9, 8.5, color="#555555")

    m = QgsLayoutItemMap(layout)
    m.setRect(0, 0, 10, 10)
    layout.addLayoutItem(m)
    m.attemptMove(QgsLayoutPoint(10, 31, MM))
    m.attemptResize(QgsLayoutSize(183, 160, MM))
    base = {"imagery": imagery, "light": light}.get(basemap)
    m.setLayers([l for l in layers + [base] if l])
    m.setKeepLayerSet(True)
    m.setCrs(QgsCoordinateReferenceSystem("EPSG:2227"))
    m.setFrameEnabled(True)
    m.zoomToExtent(zoom or extent)

    arrow = QgsLayoutItemPicture(layout)
    arrow.setPicturePath(QgsSymbolLayerUtils.svgSymbolNameToPath("arrows/NorthArrow_02.svg", project.pathResolver()))
    layout.addLayoutItem(arrow)
    arrow.attemptMove(QgsLayoutPoint(178, 34, MM))
    arrow.attemptResize(QgsLayoutSize(12, 12, MM))
    arrow.setLinkedMap(m)
    arrow.setBackgroundEnabled(True)
    arrow.setBackgroundColor(QColor(255, 255, 255, 230))

    sb = QgsLayoutItemScaleBar(layout)
    sb.setStyle("Single Box")
    layout.addLayoutItem(sb)
    sb.setLinkedMap(m)
    sb.applyDefaultSize()
    sb.setUnits(Qgis.DistanceUnit.Feet if hasattr(Qgis, "DistanceUnit") else 3)
    sb.setUnitLabel("ft")
    sb.setSegmentSizeMode(Qgis.ScaleBarSegmentSizeMode.Fixed if hasattr(Qgis, "ScaleBarSegmentSizeMode") else 0)
    sb.setUnitsPerSegment(500)
    sb.setNumberOfSegments(4)
    sb.setNumberOfSegmentsLeft(0)
    sb.setTextFormat(fmt(7.5))
    sb.setBackgroundEnabled(True)
    sb.setBackgroundColor(QColor(255, 255, 255, 230))
    sb.update()
    sb.attemptMove(QgsLayoutPoint(13, 177, MM))

    lg = QgsLayoutItemLegend(layout)
    lg.setTitle("Legend")
    lg.setAutoUpdateModel(False)
    lroot = lg.model().rootGroup()
    lroot.clear()
    n_lines = 0                                         # for the legend height estimate below
    for entry in legend:
        lyr, ltitle, field = entry[:3]
        only = {str(v) for v in entry[3]} if len(entry) > 3 else None     # optional: categories to list
        if not lyr:
            continue
        node = lroot.addLayer(lyr)
        node.setCustomProperty("legend/title-label", ltitle)
        r = lyr.renderer()
        if field:                                       # only categories present in the data
            present = {str(v) for v in lyr.uniqueValues(lyr.fields().indexOf(field))}
            if only is not None:
                present &= only
            if hasattr(r, "categories"):
                keep = [i for i, c in enumerate(r.categories()) if str(c.value()) in present]
                QgsMapLayerLegendUtils.setLegendNodeOrder(node, keep)
                lg.model().refreshLayerLegend(node)
                n_lines += 1 + len(keep)
                continue
        n_lines += (1 + len(r.categories())) if hasattr(r, "categories") else 1
    for sid, size, bold in ((QgsLegendStyle.Title, 11, True), (QgsLegendStyle.Group, 9, True),
                            (QgsLegendStyle.Subgroup, 8.5, True), (QgsLegendStyle.SymbolLabel, 7.8, False)):
        lg.rstyle(sid).setTextFormat(fmt(size, bold))
    lg.setSymbolHeight(3.6)
    lg.setLinkedMap(m)
    layout.addLayoutItem(lg)
    lg.attemptMove(QgsLayoutPoint(199, 30, MM))
    lg.adjustBoxSize()
    lg_h = lg.sizeWithUnits().height()
    est_h = 12 + 6.2 * n_lines             # headless exports can under-report the legend's size

    y = max(30 + lg_h + 4, 30 + est_h + 4, 96)
    label(layout, side_title, 199, y, 90, 6, 10, bold=True)
    label(layout, side_text, 199, y + 6, 92, 186 - y, 6.9 if side_mono else 7.6, mono=side_mono)
    label(layout, f"Map {number:02d} of the Berryessa map series  \u2022  {AUTHOR}, Oct 2026  \u2022  "
                  f"Python (pymoo NSGA-III) + QGIS\nSources: {sources}\n{CRS_NOTE}  \u2022  "
                  "Illustrative planning exercise, not an adopted or proposed plan.",
          10, 193, 277, 14, 6.4, color="#555555")

    exporter = QgsLayoutExporter(layout)
    base_path = os.path.join(OUT_DIR, f"{number:02d}_{key}")
    for ext_ in (".pdf", ".png", ".pgw", ".png.aux.xml"):
        if os.path.exists(base_path + ext_):
            os.remove(base_path + ext_)
    ok_pdf = exporter.exportToPdf(base_path + ".pdf", QgsLayoutExporter.PdfExportSettings()) == QgsLayoutExporter.Success
    img = QgsLayoutExporter.ImageExportSettings()
    img.dpi = 170
    ok_png = exporter.exportToImage(base_path + ".png", img) == QgsLayoutExporter.Success
    SERIES.append((number, key, title, ok_pdf, ok_png))
    return layout


IMAGERY_SRC = "Imagery: Esri, Maxar, Earthstar Geographics"
LIGHT_SRC = "Basemap: Esri, HERE, Garmin, \u00a9 OpenStreetMap contributors"
OSM_SRC = "\u00a9 OpenStreetMap contributors"
CITY_SRC = "City of San Jose GIS (Urban Village boundary, General Plan 2040, parcels, creeks, noise contours)"

L = ALL.get
uv_acres = next((r["acres"] for r in rows(uvb)), None)
site_rows = rows(site)
site_acres = site_rows[0]["acres"] if site_rows else None
cls_counts = defaultdict(int)
for f in cgrid.getFeatures():
    cls_counts[f["class"]] += 1

# 01 existing conditions
map_page(1, "existing_conditions", "Existing Conditions",
         "Where the model may build: parcels inside the City's Urban Village boundary that are surface parking, the "
         "flea market, vacant or lightly built (under 30% building coverage). Existing homes, parks and schools stay.",
         [station, L("transit_stops"), uvb, site, L("parcels_uv"), L("osm_parks"), L("osm_essential")],
         [(uvb, "Urban Village boundary (City)", None), (site, "Redevelopment site (model)", None),
          (L("parcels_uv"), "Parcels", "redevelopable"), (L("transit_stops"), "Transit stops (GTFS)", "mode"),
          (station, "BART station (OSM)", None), (L("osm_parks"), "Existing park", None),
          (L("osm_essential"), "Existing school / public", None)],
         "Study area",
         f"Urban Village: {uv_acres:,.0f} acres (City boundary)\nRedevelopment site: {site_acres:,.0f} acres "
         f"({cls_counts.get('Redevelopable site', 0)} cells of 120 ft)\nExisting housing cells: "
         f"{cls_counts.get('Existing housing', 0):,}\nRoad / rail / creek / floodway cells: "
         f"{cls_counts.get('Road / rail / creek setback / floodway', 0):,}\n\nThe City's plan for this village "
         f"(adopted June 22, 2021) expects about 4,800 homes and 22,100 jobs; the model only decides which "
         f"use goes where on the redevelopable land.",
         f"{CITY_SRC}; Caltrans GTFS stops; {OSM_SRC}; {IMAGERY_SRC}")

# 02 rules and constraints
SHORT_RULES = {"R1_CREEK_SETBACK": "Creek setback, 125 ft from centerline", "R2_FLOODWAY": "FEMA floodway",
               "R3_SFHA": "FEMA 1% flood zone (new homes)", "R4_RAIL_NOISE": "Rail noise 70-74 dBA (new homes)",
               "R5_INDUSTRY_HARD": "Homes touching industry", "R7_FLOOD_X": "FEMA Zone X shaded (new homes)"}
rule_lines = []
for r in RULES:
    if r["rule_id"] not in SHORT_RULES or r["rule_id"] == "R5_INDUSTRY_HARD":
        continue
    rule_lines.append(f"{r['rule_id'].split('_')[0]}  {SHORT_RULES[r['rule_id']]}: {r['enforcement']}")
map_page(2, "rules_constraints", "Rules and Constraints",
         "Written rules turned into geometry: creek setback (Council Policy 6-34), FEMA floodway and flood zones, and "
         "the General Plan rail noise contour. Grey cells are off-limits to the optimizer.",
         [uvb, L("creeks"), L("rules"), L("constraint_grid")],
         [(L("rules"), "Rules", "rule_id"), (L("creeks"), "Creeks", "COVEREDSTATUS"),
          (L("constraint_grid"), "Grid cells", "class_code", (0, 2)), (uvb, "Urban Village boundary", None)],
         "How each rule is enforced", "\n".join(rule_lines) + "\n\nunbuildable = no development.\nreport = allowed "
         "with mitigation, as the City does, and counted.\nno_residential = no new homes (strict runs).\n"
         "Citations and dates checked: layer rules_table.",
         f"Council Policy 6-34; FEMA NFHL ({'FIRM panels in rules_table'}); {CITY_SRC}; {LIGHT_SRC}",
         basemap="light")

# 03 walk access
walk_txt = "\n".join(f"{r['label']}: {r.get('residents_2020_within', 0):,} residents, "
                     f"{r.get('homes_2020_within', 0):,} homes (2020 Census)" for r in WALK)
map_page(3, "walk_access_bart", "Walk Access to BART",
         "Walking times along real streets and paths from the Berryessa / North San Jose BART station at 3 mph "
         "(264 ft per minute), the service-area method used in QGIS and ArcGIS Network Analyst.",
         [station, L("walk_destinations"), uvb, L("walk_network"), L("walk_bart")],
         [(L("walk_bart"), "Walk time to BART", "minutes"), (L("walk_destinations"), "Destinations", "type"),
          (L("walk_network"), "Walk network (OSM)", None), (uvb, "Urban Village boundary", None)],
         "Existing residents within walking distance",
         walk_txt + "\n\nPopulation is allocated from 2020 Census blocks by area. Walk times to groceries, parks and "
         "schools for every cell are in the constraint_grid layer.",
         f"{OSM_SRC} (walk network); US Census Bureau 2020 blocks; Caltrans GTFS; {LIGHT_SRC}", basemap="light")

# 04 city plan
plan = PLAN.get("city_plan")
plan_txt = ("" if not plan else
            f"New homes: {plan['new_home_acres']:.1f} ac ({plan['new_home_cells']} cells)\n"
            f"Avg. walk, home to nearest job: {ft(plan['job_walk_ft'])}\n"
            f"Avg. walk, home to school / services: {ft(plan['services_walk_ft'])}\n"
            f"Livability score: {plan['livability']:.1f}\n"
            f"New-home cells within 10 min of BART: {plan['new_homes_10min_bart']}\n"
            f"New-home cells in rail noise contour: {plan['new_homes_in_noise_contour']}\n"
            f"New-home cells in FEMA 1% flood zone: {plan['new_homes_in_flood_zone']}\n\n"
            "Mapping: TR, UR, MUN -> Residential; UV, TEC -> Commercial / Mixed Use; OSPH -> Parks; PQP, TU -> "
            "Public. The Urban Village designation also allows housing; read that way the plan has more homes "
            "(layer plan_comparison, row 2).")
map_page(4, "city_plan", "The City's Adopted Plan on the Model Grid",
         "General Plan 2040 designations inside the Urban Village, placed on the same 120 ft cells and scored with the "
         "same objective functions as the optimized layouts.",
         [station, site, L("gp_landuse"), L("city_plan_zones")],
         [(L("city_plan_zones"), "Plan, as model land use", "lu_code"), (site, "Redevelopment site", None)],
         "The plan, scored by the model", plan_txt, f"{CITY_SRC}; {IMAGERY_SRC}")

# 05-08 optimized layouts
for k, (_, key, title, sid) in enumerate(REPS):
    rec = stats_for(sid)
    zones = L(f"{key}_zones")
    bldg = L(f"{key}_buildings")
    mix = acres_by_use(zones)
    total = sum(mix.values()) or 1
    mix_txt = "\n".join(f"{lu}: {a:.1f} ac ({a / total:.0%})" for lu, a in sorted(mix.items(), key=lambda kv: -kv[1]))
    obj_txt = (f"Avg. walk, home to nearest job: {ft(rec['job_walk_ft'])}\n"
               f"Avg. walk, home to school / services: {ft(rec['services_walk_ft'])}\n"
               f"Livability score: {rec['livability']:.1f}\n"
               f"Avg. walk, new homes to BART: {ft(rec.get('new_homes_walk_to_bart_ft'))}\n"
               f"New-home cells within 5 / 10 min of BART: {rec.get('new_homes_5min_bart')} / "
               f"{rec.get('new_homes_10min_bart')} of {rec.get('new_home_cells')}\n"
               f"New-home cells in noise contour / 1% flood zone: {rec.get('new_homes_in_noise_contour')} / "
               f"{rec.get('new_homes_in_flood_zone')}\n"
               f"Homes within 10 min of a park: {rec.get('pct_homes_10min_park')}%\n"
               f"Pareto solution #{sid} of {len(P_ROWS)}")
    map_page(5 + k, key, title,
             "Optimized land use for the redevelopable cells (NSGA-III, 300 generations, seed 7), with the same amount "
             "of each use as the City's plan. Walking distances follow real streets and paths.",
             [station, bldg, zones, site, uvb, L("osm_water"), L("osm_parks"), L("osm_essential")],
             [(zones, "Proposed land use", "lu_code"), (bldg, "Building footprint", None),
              (site, "Redevelopment site", None), (uvb, "Urban Village boundary", None), (station, "BART station", None),
              (L("osm_water"), "Creek + riparian setback", None), (L("osm_parks"), "Existing park", None)],
             "Land use mix and objective values", mix_txt + "\n\n" + obj_txt,
             f"Model output; {CITY_SRC}; {OSM_SRC}; {IMAGERY_SRC}")

# 09 balanced vs plan
bal = stats_for(REPS[0][3])
cmp_rows = [("Avg. walk to job", ft(bal["job_walk_ft"]), ft(plan["job_walk_ft"]) if plan else "-"),
            ("Avg. walk, new homes-BART", ft(bal.get("new_homes_walk_to_bart_ft")),
             ft(plan.get("new_homes_walk_to_bart_ft")) if plan else "-"),
            ("Avg. walk to services", ft(bal["services_walk_ft"]), ft(plan["services_walk_ft"]) if plan else "-"),
            ("Livability", f"{bal['livability']:.1f}", f"{plan['livability']:.1f}" if plan else "-"),
            ("New-home cells", str(bal["new_home_cells"]), str(plan["new_home_cells"]) if plan else "-"),
            ("  within 10 min of BART", str(bal.get("new_homes_10min_bart")), str(plan.get("new_homes_10min_bart")) if plan else "-"),
            ("  in noise contour", str(bal["new_homes_in_noise_contour"]), str(plan["new_homes_in_noise_contour"]) if plan else "-"),
            ("  in 1% flood zone", str(bal["new_homes_in_flood_zone"]), str(plan["new_homes_in_flood_zone"]) if plan else "-"),
            ("Homes <=10 min to park", f"{bal.get('pct_homes_10min_park')}%", f"{plan.get('pct_homes_10min_park')}%" if plan else "-")]
n_same = sum(1 for f in L("plan_vs_balanced").getFeatures() if f["same"] == 1) if L("plan_vs_balanced") else 0
n_cells = L("plan_vs_balanced").featureCount() if L("plan_vs_balanced") else 0
cmp_txt = (f"{'':<26}{'Balanced':>9}{'City plan':>11}\n" +
           "\n".join(f"{a:<26}{b:>9}{c:>11}" for a, b, c in cmp_rows) +
           f"\n\nSame use in {n_same} of {n_cells} site cells.\nLower walks and hazard counts are better;"
           f"\nhigher livability and access are better.")
map_page(9, "balanced_vs_city_plan", "Balanced Layout vs. the City's Plan",
         "Orange cells: the balanced optimized layout uses the land differently from the City's plan. Grey: same "
         "use. Both have the same amount of each use, so differences are about arrangement.",
         [station, uvb, L("plan_vs_balanced")],
         [(L("plan_vs_balanced"), "Cell comparison", "same"), (uvb, "Urban Village boundary", None)],
         "Same metrics, side by side", cmp_txt, f"Model output; {CITY_SRC}; {LIGHT_SRC}", basemap="light",
         side_mono=True)

project.write(PROJECT_FILE)
for n, k, t, a, b in SERIES:
    print(f"Map {n:02d} {t:<44} PDF {'ok' if a else 'FAILED'}   PNG {'ok' if b else 'FAILED'}")
print(f"Project saved to {PROJECT_FILE}; maps in {OUT_DIR}")
try:
    iface.openLayoutDesigner(manager.layoutByName(f"Map 05 - {REPS[0][2]}"))  # noqa: F821
except NameError:
    pass
