"""
QA/QC review workspace in QGIS for the Berryessa site model.

Run inside QGIS (after gis_qaqc.py, and after qgis_site_map.py has built the site project):
    exec(open(r"C:\\Users\\gradd\\Desktop\\nsga3_city_test\\qgis_qaqc_review.py").read())

What it sets up
  * "QA/QC review" group: error points (by severity), corrected road network, summary table
    (all from berryessa.gpkg, written by gis_qaqc.py)
  * "Manual edits" group: the overrides layer, ready to digitize, with a dropdown form
    (force_class), required reason, and editor/date filled in automatically
  * Dropdown review form on the error layer: status = Open / Fixed (manual) / Not an error ...
  * Two print layouts, exported to maps/:
      QA - Overview      map of all findings + check-by-check results table
      QA - Error Atlas   one page per Error/Warning finding, zoomed in (QGIS atlas)
Safe to re-run.
"""
import os
from qgis.core import (Qgis, QgsProject, QgsVectorLayer, QgsEditorWidgetSetup, QgsDefaultValue,
                       QgsFieldConstraints, QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel,
                       QgsLayoutItemLegend,
                       QgsLayoutPoint, QgsLayoutSize, QgsLayoutExporter, QgsTextFormat, QgsLegendStyle,
                       QgsLayoutItemScaleBar, QgsCoordinateReferenceSystem, QgsLayoutMeasurement)
from qgis.PyQt.QtGui import QFont, QColor
from qgis.PyQt.QtCore import Qt

FOLDER = os.environ.get("LANDUSE_PROJECT_DIR", r"C:\Users\gradd\Desktop\nsga3_city_test")
QA_GPKG = os.path.join(FOLDER, "berryessa.gpkg")          # gis_qaqc.py writes its layers here
EDITS_GPKG = os.path.join(FOLDER, "site_edits.gpkg")
OUT_DIR = os.path.join(FOLDER, "maps")
MM = Qgis.LayoutUnit.Millimeters if hasattr(Qgis, "LayoutUnit") else 0
G_QA, G_EDIT = "QA/QC review", "Manual edits"

project = QgsProject.instance()
root = project.layerTreeRoot()
os.makedirs(OUT_DIR, exist_ok=True)

# Remove the previous run's layouts first: they reference the layers removed below
for nm in ("QA - Overview", "QA - Error Atlas"):
    old = project.layoutManager().layoutByName(nm)
    if old:
        project.layoutManager().removeLayout(old)

for name in (G_QA, G_EDIT):
    grp = root.findGroup(name)
    if grp:
        for lyr in [n.layer() for n in grp.findLayers()]:
            project.removeMapLayer(lyr.id())
        root.removeChildNode(grp)


def add(path, layer, title, group, visible=True):
    lyr = QgsVectorLayer(f"{path}|layername={layer}", title, "ogr")
    if not lyr.isValid():
        raise RuntimeError(f"Could not load {layer} from {path}")
    project.addMapLayer(lyr, False)
    group.addLayer(lyr).setItemVisibilityChecked(visible)
    return lyr


def value_map(layer, field, pairs):
    """Dropdown editor: pairs = [(label, stored value), ...]"""
    idx = layer.fields().indexOf(field)
    layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup("ValueMap", {"map": [{lbl: val} for lbl, val in pairs]}))


def read_only(layer, *fields):
    cfg = layer.editFormConfig()
    for f in fields:
        cfg.setReadOnly(layer.fields().indexOf(f), True)
    layer.setEditFormConfig(cfg)


g_qa = root.insertGroup(0, G_QA)
errors = add(QA_GPKG, "qa_errors", "QA findings", g_qa)
roads_fixed = add(QA_GPKG, "roads_corrected", "Road network (corrected)", g_qa)
summary = add(QA_GPKG, "qa_summary", "QA summary by check", g_qa)

value_map(errors, "status", [("Open", "Open"), ("Fixed (auto)", "Fixed (auto)"),
                             ("Fixed (manual)", "Fixed (manual)"), ("Not an error", "Not an error")])
read_only(errors, "error_id", "check_code", "severity", "source_layer", "feature_ref", "description", "fix")
errors.setDisplayExpression('"check_code" || \': \' || "feature_ref"')

g_edit = root.insertGroup(1, G_EDIT)
overrides = add(EDITS_GPKG, "overrides", "Site overrides (digitize here)", g_edit)
value_map(overrides, "force_class", [("Unbuildable (road / rail / creek)", "unbuildable"),
                                     ("Redevelopable site", "site"),
                                     ("Existing housing (keep)", "existing_housing"),
                                     ("Existing park (keep)", "existing_park"),
                                     ("Existing school / public (keep)", "existing_public"),
                                     ("Outside site (no change)", "outside")])
for fld in ("force_class", "reason"):
    i = overrides.fields().indexOf(fld)
    overrides.setFieldConstraint(i, QgsFieldConstraints.ConstraintNotNull, QgsFieldConstraints.ConstraintStrengthHard)
overrides.setDefaultValueDefinition(overrides.fields().indexOf("override_id"),
                                    QgsDefaultValue('coalesce(maximum("override_id"), 0) + 1'))
overrides.setDefaultValueDefinition(overrides.fields().indexOf("editor"), QgsDefaultValue("@user_full_name"))
overrides.setDefaultValueDefinition(overrides.fields().indexOf("edit_date"),
                                    QgsDefaultValue("format_date(now(), 'yyyy-MM-dd HH:mm')"))
read_only(overrides, "override_id", "editor", "edit_date")
sym = overrides.renderer().symbol()
sym.setColor(QColor(255, 0, 200, 60))
sym.symbolLayer(0).setStrokeColor(QColor(255, 0, 200))
sym.symbolLayer(0).setStrokeWidth(0.8)

# ------------------------------------------------------------------ layouts
def fmt(size, bold=False, color="#1a1a1a"):
    t = QgsTextFormat()
    f = QFont("Arial")
    f.setBold(bold)
    t.setFont(f)
    t.setSize(size)
    t.setColor(QColor(color))
    return t


def label(layout, text, x, y, w, h, size, bold=False, color="#1a1a1a"):
    item = QgsLayoutItemLabel(layout)
    item.setText(text)
    item.setTextFormat(fmt(size, bold, color))
    item.setVAlign(Qt.AlignTop)
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y, MM))
    item.attemptResize(QgsLayoutSize(w, h, MM))
    return item


def find_layer(group_name, title):
    grp = root.findGroup(group_name)
    for n in (grp.findLayers() if grp else []):
        if n.layer().name() == title:
            return n.layer()
    return None


def find_prefix(group_name, prefix):
    grp = root.findGroup(group_name)
    for n in (grp.findLayers() if grp else []):
        if n.layer().name().startswith(prefix):
            return n.layer()
    return None


site = find_layer("Proposed land use", "Redevelopment site")
zones = find_prefix("Proposed land use", "Proposed land use")
imagery = find_layer("Basemap", "Esri World Imagery")
base_layers = [l for l in (site, zones, imagery) if l]

manager = project.layoutManager()

n_err = errors.featureCount()
counts = {}
for f in errors.getFeatures():
    counts[f["severity"]] = counts.get(f["severity"], 0) + 1

# --- Overview
ov = QgsPrintLayout(project)
ov.initializeDefaults()
ov.setName("QA - Overview")
manager.addLayout(ov)
label(ov, "Berryessa Site Model: Data QA/QC Review", 10, 7, 277, 10, 17, bold=True)
label(ov, f"{n_err} findings: OpenStreetMap on its own and against the City, FEMA and GTFS layers, manual edits, "
          "model outputs  \u2022  "
          + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())), 10, 17, 277, 7, 9.5, color="#555555")
m = QgsLayoutItemMap(ov)
m.setRect(0, 0, 10, 10)
ov.addLayoutItem(m)
m.attemptMove(QgsLayoutPoint(10, 26, MM))
m.attemptResize(QgsLayoutSize(150, 176, MM))
m.setLayers([errors, overrides, roads_fixed] + base_layers)
m.setKeepLayerSet(True)
m.setCrs(QgsCoordinateReferenceSystem("EPSG:2227"))
m.setFrameEnabled(True)
cg = find_layer("Rules and constraints", "Constraint grid (what the optimizer saw)")
ext = cg.extent() if cg else roads_fixed.extent()
m.zoomToExtent(ext)

sb = QgsLayoutItemScaleBar(ov)
sb.setStyle("Single Box")
ov.addLayoutItem(sb)
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
sb.attemptMove(QgsLayoutPoint(13, 189, MM))

legend = QgsLayoutItemLegend(ov)
legend.setTitle("")
legend.setAutoUpdateModel(False)
lroot = legend.model().rootGroup()
lroot.clear()
for lyr, t in ((errors, "QA finding (severity)"), (roads_fixed, "Road network"), (overrides, "Manual override")):
    lroot.addLayer(lyr).setCustomProperty("legend/title-label", t)
for sid, size, bold in ((QgsLegendStyle.Group, 9, True), (QgsLegendStyle.Subgroup, 9, True),
                        (QgsLegendStyle.SymbolLabel, 8, False)):
    legend.rstyle(sid).setTextFormat(fmt(size, bold))
legend.setLinkedMap(m)
ov.addLayoutItem(legend)
legend.attemptMove(QgsLayoutPoint(165, 26, MM))
legend.adjustBoxSize()

label(ov, "Results by check", 165, 104, 120, 6, 10.5, bold=True)
rows = sorted((f["check_code"], f["severity"], int(f["count"]), f["result"]) for f in summary.getFeatures())
order = {"FAIL": 0, "REVIEW": 1, "FIXED": 2, "PASS": 3}
rows.sort(key=lambda r: (order.get(r[3], 9), r[0]))
lines = [f"{'Check':<24}{'Severity':<9}{'Count':>5}  Result", "-" * 46]
lines += [f"{c:<24}{sv:<9}{n:>5}  {r}" for c, sv, n, r in rows]
tbl = label(ov, "\n".join(lines), 165, 111, 122, 74, 6.6)
mono = QgsTextFormat(tbl.textFormat())
mono.setFont(QFont("Courier New"))
mono.setSize(6.6)
tbl.setTextFormat(mono)
label(ov, "PASS = no findings  \u2022  FIXED = corrected automatically (see magenta road segments)  \u2022  "
          "REVIEW = needs a technician's decision; set status in the QA findings form\n"
          "Data: \u00a9 OpenStreetMap contributors; City of San Jose GIS; FEMA NFHL; Caltrans GTFS  \u2022  "
          "Imagery: Esri  \u2022  EPSG:2227 (CA State Plane III, ftUS)",
      165, 188, 122, 16, 6.3, color="#555555")

# --- Error atlas: one page per Error / Warning finding
at = QgsPrintLayout(project)
at.initializeDefaults()
at.setName("QA - Error Atlas")
manager.addLayout(at)
atlas = at.atlas()
atlas.setCoverageLayer(errors)
atlas.setEnabled(True)
atlas.setFilterFeatures(True)
atlas.setFilterExpression("\"severity\" IN ('Error', 'Warning')")
atlas.setSortFeatures(True)
atlas.setSortExpression('"error_id"')
atlas.setPageNameExpression("'finding_' || \"error_id\"")
am = QgsLayoutItemMap(at)
am.setRect(0, 0, 10, 10)
at.addLayoutItem(am)
am.attemptMove(QgsLayoutPoint(10, 26, MM))
am.attemptResize(QgsLayoutSize(175, 176, MM))
am.setLayers([l for l in (errors, overrides, roads_fixed, site, imagery) if l])   # no zones: show the source data
am.setKeepLayerSet(True)
am.setCrs(QgsCoordinateReferenceSystem("EPSG:2227"))
am.setFrameEnabled(True)
am.zoomToExtent(ext)
am.setAtlasDriven(True)
am.setAtlasScalingMode(QgsLayoutItemMap.Fixed)
am.setScale(1200)
label(at, "QA/QC Finding [% @atlas_featurenumber %] of [% @atlas_totalfeatures %] (Errors and Warnings)  \u2022  "
          "ID [% \"error_id\" %]", 10, 7, 277, 10, 16, bold=True)
label(at, "[% \"check_code\" %]  \u2022  [% \"severity\" %]  \u2022  status: [% \"status\" %]", 10, 17, 277, 7, 10,
      color="#555555")
label(at, "Description", 192, 26, 95, 6, 10.5, bold=True)
label(at, "[% \"description\" %]", 192, 33, 95, 30, 9)
label(at, "Feature", 192, 66, 95, 6, 10.5, bold=True)
label(at, "Layer: [% \"source_layer\" %]\nReference: [% \"feature_ref\" %]\n"
          "Location: [% format_number($x, 0) %] E, [% format_number($y, 0) %] N (ft)", 192, 73, 95, 20, 9)
label(at, "Correction", 192, 96, 95, 6, 10.5, bold=True)
label(at, "Fix type: [% \"fix\" %]\nReviewer note: [% coalesce(\"reviewer_note\", '(none)') %]", 192, 103, 95, 20, 9)
label(at, "Map scale 1:1,200 \u2022 magenta = corrected road geometry \u2022 EPSG:2227", 192, 190, 95, 10, 7,
      color="#555555")

exp = QgsLayoutExporter(ov)
for e_ in (".pdf", ".png"):
    p_ = os.path.join(OUT_DIR, "qa_overview" + e_)
    if os.path.exists(p_):
        os.remove(p_)
ok1 = exp.exportToPdf(os.path.join(OUT_DIR, "qa_overview.pdf"), QgsLayoutExporter.PdfExportSettings())
img = QgsLayoutExporter.ImageExportSettings()
img.dpi = 170
ok2 = exp.exportToImage(os.path.join(OUT_DIR, "qa_overview.png"), img)
atlas_pdf = os.path.join(OUT_DIR, "qa_error_atlas.pdf")
if os.path.exists(atlas_pdf):
    os.remove(atlas_pdf)
res = QgsLayoutExporter.exportToPdf(atlas, atlas_pdf, QgsLayoutExporter.PdfExportSettings())
ok3 = res[0] if isinstance(res, tuple) else res
project.write()
print(f"QA overview PDF {'ok' if ok1 == 0 else 'FAILED'}, PNG {'ok' if ok2 == 0 else 'FAILED'}; "
      f"error atlas ({atlas.count()} pages) {'ok' if ok3 == 0 else 'FAILED'}")
print(f"{n_err} findings loaded. Open the 'QA findings' attribute table to review; "
      "toggle editing on 'Site overrides' to digitize corrections.")
