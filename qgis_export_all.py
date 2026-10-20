"""
Headless QGIS step for run_demo.py: builds the site project and exports every map without opening
the QGIS window. Started by run_demo.py through QGIS's own Python (python-qgis-ltr.bat on Windows).

Reads LANDUSE_PROJECT_DIR (the project folder) and LANDUSE_NO_BASEMAP=1 (skip satellite imagery).
"""
import os
import sys

from qgis.core import QgsApplication

FOLDER = os.environ.get("LANDUSE_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
app = QgsApplication([], False)
app.initQgis()
try:
    for script in ("qgis_site_map.py", "qgis_qaqc_review.py"):
        path = os.path.join(FOLDER, script)
        print(f"Running {script}", flush=True)
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), {"__name__": "__qgis_export__"})
    code = 0
except Exception as exc:  # report and fail the step
    import traceback
    traceback.print_exc()
    code = 1
finally:
    app.exitQgis()
sys.stdout.flush()
os._exit(code)   # skip Qt teardown, which can crash after a headless run
