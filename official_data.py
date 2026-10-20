"""
Loads the frozen official layers in data/raw/ (written by fetch_official.py) in State Plane feet.

Every loader returns None when its file is missing, so the pipeline still runs on OpenStreetMap alone
(the original October 2026 behaviour) until fetch_official.py has been run.
"""
import csv
import json
import os
from functools import lru_cache

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
MANIFEST = os.path.join(HERE, "data", "manifest.csv")
CRS = "EPSG:2227"

FILES = {
    "uv_boundary": "sj_urban_village.geojson", "general_plan": "sj_general_plan.geojson",
    "zoning": "sj_zoning.geojson", "parcels": "sj_parcels.geojson", "streets": "sj_streets.geojson",
    "streams": "sj_streams.geojson", "noise": "sj_noise_contours.geojson", "parks": "sj_parks.geojson",
    "schools": "sj_schools.geojson", "flood_zones": "fema_flood_zones.geojson",
    "firm_panels": "fema_firm_panels.geojson", "block_groups": "census_block_groups.geojson",
    "census_blocks": "census_blocks_2020.geojson", "transit_stops": "caltrans_transit_stops.geojson",
}
UV_NAME = "Berryessa BART"

# Envision San Jose 2040 General Plan designations -> the model's five land uses.
# The model has one class per cell, so mixed-use designations go to their primary use.
GP_TO_LU = {
    "TR": (0, "Transit Residential: high-density housing (ground-floor retail allowed)"),
    "UR": (0, "Urban Residential"),
    "RN": (0, "Residential Neighborhood"),
    "MUN": (0, "Mixed Use Neighborhood: mainly residential with some commercial"),
    "OSPH": (1, "Open Space, Parklands and Habitat"),
    "UV": (2, "Urban Village: mixed residential / commercial, counted as mixed use (jobs)"),
    "TEC": (2, "Transit Employment Center: office / R&D employment"),
    "NCC": (2, "Neighborhood / Community Commercial"),
    "CIC": (2, "Combined Industrial / Commercial"),
    "LI": (3, "Light Industrial"),
    "HI": (3, "Heavy Industrial"),
    "PQP": (4, "Public / Quasi-Public"),
    "TU": (-1, "Transportation and Utilities (BART station, rail): not a land use the model assigns"),
}
GROWTH_DESIGNATIONS = {"UV", "TR", "TEC", "UR", "MUN", "OSPH", "PQP", "NCC", "CIC"}
PARK_TYPES = {"NEIG", "REG", "CTY", "COMGRD", "DOG"}      # walkable parks (not trails, golf, libraries)


def available():
    return os.path.exists(os.path.join(RAW, FILES["uv_boundary"])) and \
        os.path.exists(os.path.join(RAW, FILES["parcels"]))


@lru_cache(maxsize=None)
def load(key):
    path = os.path.join(RAW, FILES[key])
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(CRS)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def uv_boundary():
    uv = load("uv_boundary")
    if uv is None or uv.empty:
        return None
    hit = uv[uv["NAME"] == UV_NAME]
    return (hit if len(hit) else uv).geometry.union_all()


def uv_attributes():
    uv = load("uv_boundary")
    return {} if uv is None or uv.empty else uv.drop(columns="geometry").iloc[0].to_dict()


def open_parks():
    p = load("parks")
    if p is None:
        return None
    return p[(p["CURRENTSTATUS"] == "Open") & p["PARKTYPE"].isin(PARK_TYPES)]


def open_schools():
    s = load("schools")
    return None if s is None else s[s["STATUS"].fillna("").str.strip() == "Open"]


def bart_stops():
    t = load("transit_stops")
    if t is None:
        return None
    return t[t["route_type"].astype(str) == "1"]


def manifest():
    if not os.path.exists(MANIFEST):
        return pd.DataFrame()
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return pd.DataFrame(list(csv.DictReader(fh)))


def firm_effective_dates():
    p = load("firm_panels")
    if p is None or p.empty:
        return ""
    out = []
    for _, r in p.iterrows():
        d = pd.to_datetime(r["EFF_DATE"], unit="ms", utc=True).strftime("%Y-%m-%d") if pd.notna(r["EFF_DATE"]) else "?"
        out.append(f"{r['FIRM_PAN']} (effective {d})")
    return ", ".join(sorted(out))


def walk_osm_path():
    p = os.path.join(RAW, "osm_walk.json")
    return p if os.path.exists(p) else None


def load_walk_osm():
    p = walk_osm_path()
    if p is None:
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
