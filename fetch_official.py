"""
Download the official (authoritative) GIS layers for the Berryessa site
=======================================================================

Pure standard library, so it runs in plain Python OR inside the QGIS Python Console:
    .venv\\Scripts\\python.exe fetch_official.py
    exec(open(r"C:\\Users\\gradd\\Desktop\\nsga3_city_test\\fetch_official.py").read())

Writes every layer to data/raw/ exactly as the agency's server returned it (GeoJSON in WGS84) and
records where each one came from in data/manifest.csv. The pipeline only ever reads these frozen
files, so a run is reproducible until you choose to refresh them by running this script again.

    python fetch_official.py                         download everything now
    python fetch_official.py --from-bundle a.json    rebuild data/raw from saved server responses
                                                     (a JSON object of {request URL: response text},
                                                     e.g. responses saved through a web browser)

Sources
  City of San Jose GIS (geo.sanjoseca.gov)   Urban Village boundary, General Plan 2040 land use,
                                             zoning, parcels, street centerlines, creeks, General
                                             Plan noise contours, parks, schools
  FEMA National Flood Hazard Layer           flood hazard zones, FIRM panels (effective dates)
  US Census Bureau                           2020 Census blocks (population, housing units),
                                             block groups (TIGERweb), ACS 5-year counts (optional)
  Caltrans (Cal-ITP GTFS compilation)        transit stops for every agency (VTA, BART, ...)
  OpenStreetMap (Overpass API)               walk network + destinations for walk-time analysis
                                             (a wider area than site_osm.json)
"""
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                   # exec() in the QGIS console has no __file__
    HERE = r"C:\Users\gradd\Desktop\nsga3_city_test"
RAW = os.path.join(HERE, "data", "raw")
MANIFEST = os.path.join(HERE, "data", "manifest.csv")

# site: the Urban Village plus the 360 ft model margin and ~300 ft of context (same box as
# fetch_osm_site.py); access: ~4,000 ft further each way, so walks that leave the site can be measured
SITE_BBOX = (37.36132, -121.88729, 37.37811, -121.86056)          # S, W, N, E
ACCESS_PAD = (0.011, 0.0138)                                      # degrees lat, lon

SJ = "https://geo.sanjoseca.gov/server/rest/services/"
FEMA = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/"
TIGER = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
CALTRANS = "https://gisdata.dot.ca.gov/arcgis/rest/services/CHrailroad/CA_Transit_Stops/FeatureServer/0"
ACS_API = "https://api.census.gov/data/2024/acs/acs5"
OVERPASS = "https://overpass-api.de/api/interpreter"

SJ_TERMS = "City of San Jose open GIS data (public record); see data.sanjoseca.gov terms of use"
FED = "Public domain (US federal government work)"

# key, file, agency, title, service url, where, extent, license
ARCGIS = [
    ("uv_boundary", "sj_urban_village.geojson", "City of San Jose PBCE", "Growth Areas 2040 (Urban Village boundary)",
     SJ + "OPN/OPN_OpenDataService/MapServer/520", "NAME = 'Berryessa BART'", None, SJ_TERMS),
    ("general_plan", "sj_general_plan.geojson", "City of San Jose PBCE", "Envision San Jose 2040 General Plan land use",
     SJ + "OPN/OPN_OpenDataService/MapServer/404", "1=1", "site", SJ_TERMS),
    ("zoning", "sj_zoning.geojson", "City of San Jose PBCE", "Zoning districts",
     SJ + "OPN/OPN_OpenDataService/MapServer/401", "1=1", "site", SJ_TERMS),
    ("parcels", "sj_parcels.geojson", "City of San Jose (from Santa Clara County Assessor parcels)", "Parcels",
     SJ + "OPN/OPN_OpenDataService/MapServer/270", "1=1", "site", SJ_TERMS),
    ("streets", "sj_streets.geojson", "City of San Jose DOT", "Street centerlines",
     SJ + "OPN/OPN_OpenDataService/MapServer/60", "1=1", "site", SJ_TERMS),
    ("streams", "sj_streams.geojson", "City of San Jose DPW / Valley Water", "San Jose Streams (creek centerlines)",
     SJ + "DPW/DPW_TransportationAndHydraulicsServicesLayers/MapServer/21", "1=1", "site", SJ_TERMS),
    ("noise", "sj_noise_contours.geojson", "City of San Jose PBCE", "General Plan noise contours (Road Noise)",
     SJ + "OPN/OPN_OpenDataService/MapServer/367", "1=1", "site", SJ_TERMS),
    ("parks", "sj_parks.geojson", "City of San Jose PRNS", "Parks",
     SJ + "OPN/OPN_OpenDataService/MapServer/188", "1=1", "access", SJ_TERMS),
    ("schools", "sj_schools.geojson", "City of San Jose", "Schools",
     SJ + "OPN/OPN_OpenDataService/MapServer/342", "1=1", "access", SJ_TERMS),
    ("flood_zones", "fema_flood_zones.geojson", "FEMA", "NFHL Flood Hazard Zones (S_FLD_HAZ_AR)",
     FEMA + "28", "1=1", "site", FED),
    ("firm_panels", "fema_firm_panels.geojson", "FEMA", "NFHL FIRM Panels (effective dates)",
     FEMA + "3", "1=1", "site", FED),
    ("block_groups", "census_block_groups.geojson", "US Census Bureau", "TIGERweb ACS 2024 block groups",
     TIGER + "tigerWMS_ACS2024/MapServer/10", "1=1", "access", FED),
    ("census_blocks", "census_blocks_2020.geojson", "US Census Bureau",
     "TIGERweb 2020 Census blocks (POP100 population, HU100 housing units)",
     TIGER + "tigerWMS_Census2020/MapServer/10", "1=1", "access", FED),
    ("transit_stops", "caltrans_transit_stops.geojson", "Caltrans / Cal-ITP", "CA Transit Stops (compiled GTFS)",
     CALTRANS, "1=1", "access", "Caltrans GIS open data (public)"),
]
ACS_VARS = {"B01003_001E": "population", "B11001_001E": "households", "B25001_001E": "housing_units",
            "B19013_001E": "median_hh_income"}
CHUNK = 250


def bbox(extent):
    s, w, n, e = SITE_BBOX
    if extent == "access":
        s, w, n, e = s - ACCESS_PAD[0], w - ACCESS_PAD[1], n + ACCESS_PAD[0], e + ACCESS_PAD[1]
    return tuple(round(v, 5) for v in (s, w, n, e))


def envelope(extent):
    s, w, n, e = bbox(extent)
    return json.dumps({"xmin": w, "ymin": s, "xmax": e, "ymax": n, "spatialReference": {"wkid": 4326}},
                      separators=(",", ":"))


def http_get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nsga3-land-use-portfolio/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            if k == tries - 1:
                raise
            print(f"  retry after error: {exc}")
            time.sleep(3 * (k + 1))


def bundle_get(paths):
    """A get() that answers from saved {url: response} files instead of the network."""
    cache = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            cache.update(json.load(fh))

    def get(url):
        if url not in cache:
            raise KeyError(f"not in the saved responses: {url[:120]}...")
        v = cache[url]
        return v if isinstance(v, str) else json.dumps(v)
    return get


def q(url, params):
    return url + "?" + urllib.parse.urlencode(params, safe="*")


def arcgis_layer(get, service, where, extent):
    """Every feature matching where + extent, as one GeoJSON FeatureCollection in WGS84.
    Object ids first, then features in fixed-size id chunks, so paging is complete and in a stable order."""
    meta = json.loads(get(q(service, {"f": "json"})))
    params = {"where": where, "returnIdsOnly": "true", "f": "json"}
    if extent:
        params.update({"geometry": envelope(extent), "geometryType": "esriGeometryEnvelope", "inSR": "4326",
                       "spatialRel": "esriSpatialRelIntersects"})
    idres = json.loads(get(q(service + "/query", params)))
    if "error" in idres:
        raise RuntimeError(f"{service}: {idres['error']}")
    ids = sorted(idres.get("objectIds") or [])
    feats = []
    for i in range(0, len(ids), CHUNK):
        part = json.loads(get(q(service + "/query", {"objectIds": ",".join(map(str, ids[i:i + CHUNK])),
                                                      "outFields": "*", "outSR": "4326", "f": "geojson"})))
        if "error" in part:
            raise RuntimeError(f"{service}: {part['error']}")
        feats += part.get("features", [])
    sr = (meta.get("extent") or {}).get("spatialReference") or meta.get("sourceSpatialReference") or {}
    ei = meta.get("editingInfo") or {}
    edit = ei.get("dataLastEditDate") or ei.get("lastEditDate")
    info = {"name": meta.get("name"), "native_crs": f"EPSG:{sr.get('latestWkid') or sr.get('wkid')}",
            "data_date": dt.datetime.fromtimestamp(edit / 1000, dt.timezone.utc).strftime("%Y-%m-%d") if edit else ""}
    return {"type": "FeatureCollection", "features": feats}, info


def acs_url(tracts):
    return q(ACS_API, [("get", "NAME," + ",".join(ACS_VARS)), ("for", "block group:*"),
                       ("in", "state:06"), ("in", "county:085"), ("in", "tract:" + ",".join(sorted(tracts)))])


def overpass_url():
    s, w, n, e = bbox("access")
    b = f"{s},{w},{n},{e}"
    walk = ("primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential|unclassified|"
            "living_street|service|footway|path|pedestrian|steps|cycleway|track|corridor|busway|trunk|trunk_link")
    dest = "school|library|community_centre|clinic|hospital|pharmacy"
    query = (f'[out:json][timeout:120];('
             f'way["highway"~"^({walk})$"]({b});'
             f'nwr["shop"~"^(supermarket|grocery|greengrocer)$"]({b});'
             f'nwr["amenity"~"^({dest})$"]({b});'
             f'nwr["leisure"~"^(park|playground)$"]({b});'
             f'node["highway"="bus_stop"]({b});'
             f'node["public_transport"~"^(platform|stop_position|station)$"]({b});'
             f'node["railway"~"^(station|stop)$"]({b});'
             f');out body geom;')
    return q(OVERPASS, {"data": query})


def write_json(path, obj):
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_all(get=http_get, today=None, how="downloaded with fetch_official.py"):
    os.makedirs(RAW, exist_ok=True)
    today = today or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    rows = []
    tracts = set()
    for key, fname, agency, title, url, where, extent, lic in ARCGIS:
        print(f"{key:<14} {title} ...", flush=True)
        fc, info = arcgis_layer(get, url, where, extent)
        sha = write_json(os.path.join(RAW, fname), fc)
        if key == "block_groups":
            tracts = {f["properties"]["TRACT"] for f in fc["features"]}
        ext = bbox(extent) if extent else None
        rows.append({"layer": key, "file": f"data/raw/{fname}", "source_agency": agency, "title": title,
                     "url": url, "query": where + (f"; bbox S,W,N,E {ext}" if ext else ""),
                     "download_date": today, "data_date": info["data_date"], "original_crs": info["native_crs"],
                     "delivered_crs": "EPSG:4326", "features": len(fc["features"]), "license": lic,
                     "how": how, "sha256": sha})
        print(f"               {len(fc['features'])} features (native {info['native_crs']})")

    print("acs            ACS 2020-2024 5-year, block groups ...", flush=True)
    try:
        acs = json.loads(get(acs_url(tracts)))
        sha = write_json(os.path.join(RAW, "acs5_2024_block_groups.json"), acs)
        rows.append({"layer": "acs", "file": "data/raw/acs5_2024_block_groups.json",
                     "source_agency": "US Census Bureau",
                     "title": "ACS 2020-2024 5-year: " + ", ".join(f"{k} {v}" for k, v in ACS_VARS.items()),
                     "url": ACS_API, "query": f"block groups in tracts {','.join(sorted(tracts))}",
                     "download_date": today, "data_date": "2025-12 release", "original_crs": "",
                     "delivered_crs": "", "features": len(acs) - 1, "license": FED, "how": how, "sha256": sha})
    except Exception as exc:          # optional: nothing in the model needs ACS
        print(f"               skipped ({str(exc)[:80]})")

    print("osm_walk       OpenStreetMap walk network + destinations ...", flush=True)
    osm = json.loads(get(overpass_url()))
    sha = write_json(os.path.join(RAW, "osm_walk.json"), osm)
    rows.append({"layer": "osm_walk", "file": "data/raw/osm_walk.json", "source_agency": "OpenStreetMap contributors",
                 "title": "Walkable ways, grocery, schools, parks, transit stops (Overpass API)",
                 "url": OVERPASS, "query": f"bbox S,W,N,E {bbox('access')}", "download_date": today,
                 "data_date": (osm.get("osm3s") or {}).get("timestamp_osm_base", "")[:10],
                 "original_crs": "EPSG:4326", "delivered_crs": "EPSG:4326", "features": len(osm.get("elements", [])),
                 "license": "ODbL 1.0 (c) OpenStreetMap contributors", "how": how, "sha256": sha})

    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} layers to {RAW}\nManifest: {MANIFEST}")
    return rows


def main(argv):
    if "--from-bundle" in argv:
        i = argv.index("--from-bundle")
        date = next((argv[j + 1] for j, a in enumerate(argv) if a == "--date"), None)
        paths = []
        for a in argv[i + 1:]:
            if a.startswith("--"):
                break
            paths.append(a)
        fetch_all(bundle_get(paths), today=date,
                  how="requested through a web browser, saved, and replayed with --from-bundle")
    else:
        fetch_all()


if __name__ == "__main__":          # also true for exec() in the QGIS Python Console
    main(sys.argv[1:])
