"""
Download real-world constraints for the study site from OpenStreetMap (Overpass API).

Pure standard library, so it runs in plain Python OR inside the QGIS Python Console:
    exec(open(r"C:\\Users\\gradd\\Desktop\\nsga3_city_test\\fetch_osm_site.py").read())

Writes site_osm.json next to this script. site_pipeline.py turns it into a grid mask.

The download box covers the whole official Berryessa BART Urban Village boundary (City of San Jose
Growth Areas 2040 layer, see fetch_official.py) plus a 360 ft modelling margin and 300 ft of context.
The first version (Oct 6, 2026) covered only a ~1.1 km square around the flea market; that snapshot is
kept as data/archive/site_osm_2026-10-06.json.
"""
import json
import os
import urllib.parse
import urllib.request

SITE_NAME = "Berryessa BART Urban Village, San Jose"
CENTER_LAT, CENTER_LON = 37.36972, -121.87393
BBOX = (37.36132, -121.88729, 37.37811, -121.86056)      # S, W, N, E

try:
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                   # exec() in the QGIS console has no __file__
    OUT_DIR = r"C:\Users\gradd\Desktop\nsga3_city_test"
OUT_FILE = os.path.join(OUT_DIR, "site_osm.json")
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]


def build_query():
    bbox = ",".join(str(v) for v in BBOX)
    return f"""
[out:json][timeout:90];
(
  way["highway"]({bbox});
  way["railway"]({bbox});
  way["waterway"]({bbox});
  way["natural"="water"]({bbox});
  relation["natural"="water"]({bbox});
  way["landuse"]({bbox});
  relation["landuse"]({bbox});
  way["leisure"~"park|playground|pitch|sports_centre|garden"]({bbox});
  way["amenity"~"parking|marketplace|school|college|hospital|clinic|library|community_centre|fire_station|police|place_of_worship|kindergarten"]({bbox});
  node["amenity"~"school|hospital|clinic|library|community_centre|fire_station|police"]({bbox});
  way["building"]({bbox});
  way["public_transport"="station"]({bbox});
  node["railway"="station"]({bbox});
);
out geom tags;
"""


def save(data):
    data["site"] = {"name": SITE_NAME, "center": [CENTER_LAT, CENTER_LON], "bbox": list(BBOX)}
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    kinds = {}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        key = next((k for k in ("highway", "railway", "waterway", "natural", "landuse", "leisure",
                                "amenity", "building", "public_transport") if k in tags), "other")
        kinds[key] = kinds.get(key, 0) + 1
    print(f"Saved {len(data.get('elements', []))} OSM features to {OUT_FILE}")
    print(", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())))


def download():
    body = urllib.parse.urlencode({"data": build_query()}).encode()
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": "nsga3-land-use-portfolio/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # try the next mirror
            print(f"{url} failed: {exc}")
    raise RuntimeError("All Overpass endpoints failed; check your internet connection and retry.")


if __name__ == "__main__":
    save(download())
