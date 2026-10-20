"""
Written rules turned into geometry (roadmap item 4)
===================================================

Each rule has a citation and says how the model enforces it:
  unbuildable      cells covered by the rule's area cannot be developed at all
  no_residential   the optimizer may place jobs, services or open space there, but not homes
                   (a home that would land there becomes open space instead); R3 and R4 switch
                   between this and 'report' with --flood-rule / --noise-rule
  constraint       hard pymoo inequality constraint (G <= 0); any violation makes a layout infeasible
  objective        minimized as one of the four objectives
  report           measured and reported, not enforced

`kind` separates adopted rules from data layers and from my own modelling assumptions, so nobody
mistakes an assumption for city code.
"""
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

CHECKED = "2026-10-12"
CREEK_SETBACK_FT = 100           # Council Policy 6-34
BANK_ALLOWANCE_FT = 25           # assumption: centerline to top of bank (no open top-of-bank layer)
SEP_HARD_CELLS = 1               # homes may not touch industry (one 120 ft cell)
SEP_SOFT_FT = 360                # objective: homes within 360 ft (3 cells) of industry

RULES = [
    {"rule_id": "R1_CREEK_SETBACK", "description": "No development within the riparian setback of exposed "
     "(open-channel) creeks: Upper Penitencia Creek and Coyote Creek",
     "value": CREEK_SETBACK_FT + BANK_ALLOWANCE_FT, "units": "ft from creek centerline",
     "enforcement": "unbuildable", "kind": "Adopted rule + assumption",
     "source": "City of San Jose Council Policy 6-34, Riparian Corridor Protection and Bird-Safe Design "
               "(adopted Aug 23, 2016): minimum 100 ft setback measured from the riparian dripline or top of "
               "bank, whichever is greater. Model: 100 ft + 25 ft assumed centerline-to-top-of-bank, since no "
               "open top-of-bank layer exists. Creek centerlines: City of San Jose 'San Jose Streams' layer "
               "(Valley Water IDs); culverted ('Covered') reaches are excluded.",
     "source_url": "https://www.sanjoseca.gov/home/showpublisheddocument/12815/636669915138100000"},
    {"rule_id": "R2_FLOODWAY", "description": "No development in the FEMA regulatory floodway",
     "value": None, "units": "FEMA AE Floodway polygons",
     "enforcement": "unbuildable", "kind": "Data layer (FEMA NFHL) + modelling choice",
     "source": "FEMA National Flood Hazard Layer, Flood Hazard Zones (FIRM panels {panels}). Encroachment in a "
               "floodway needs a no-rise certification (44 CFR 60.3(d)(3)); the model treats it as unbuildable.",
     "source_url": "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28"},
    {"rule_id": "R3_SFHA", "description": "New homes in a Special Flood Hazard Area outside the floodway "
     "(FEMA 1% annual chance flood zone: Zones A, AE, AH, AO)", "value": None, "units": "FEMA SFHA polygons",
     "enforcement": "report", "kind": "Data layer + modelling choice",
     "source": "FEMA NFHL, SFHA_TF = 'T'. San Jose Municipal Code Ch. 17.08 (floodplain management) allows homes "
               "in an SFHA when the lowest floor is elevated, so by default the model allows them and counts them; "
               "--flood-rule hard keeps new homes out (stricter than code).",
     "source_url": "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28"},
    {"rule_id": "R4_RAIL_NOISE", "description": "New homes inside the General Plan 70-74 dBA heavy-rail noise "
     "contour along the BART / former UPRR corridor", "value": 500, "units": "ft each side of the tracks "
     "(the City's contour)", "enforcement": "report", "kind": "Data layer + modelling choice",
     "source": "City of San Jose General Plan noise contours ('Road Noise' open-data layer: Heavy Rail Train, "
               "70-74 dBA, 500 ft). Envision San Jose 2040 Policy EC-1.1 sets 60 dBA DNL as the exterior "
               "objective for homes and 45 dBA DNL indoors, so homes inside the contour need a noise study and "
               "attenuation. By default the model allows them and counts them, as the City does with mitigation; "
               "--noise-rule hard keeps new homes out.",
     "source_url": "https://geo.sanjoseca.gov/server/rest/services/OPN/OPN_OpenDataService/MapServer/367"},
    {"rule_id": "R5_INDUSTRY_HARD", "description": "No home may share an edge with a light-industrial cell",
     "value": SEP_HARD_CELLS * 120, "units": "ft (one grid cell)", "enforcement": "constraint",
     "kind": "Modelling assumption",
     "source": "No single City standard applies to light industry. CARB's Air Quality and Land Use Handbook "
               "(2005) recommends separation distances for specific sources (for example 300 ft from large gas "
               "stations, 1,000 ft from distribution centers); this one-cell buffer is a minimal stand-in.",
     "source_url": "https://ww2.arb.ca.gov/our-work/programs/resource-center/strategy-development/land-use-resources"},
    {"rule_id": "R6_INDUSTRY_SOFT", "description": "Count homes within 360 ft of light industry",
     "value": SEP_SOFT_FT, "units": "ft (3 cells, grid distance)", "enforcement": "report",
     "kind": "Modelling assumption",
     "source": "The capstone minimized homes next to industry; here R5 forces that to zero, and the fourth "
               "objective is the walk from new homes to BART instead. Only matters with --program capstone "
               "(the City's plan puts no light industry on the site).",
     "source_url": ""},
    {"rule_id": "R7_FLOOD_X", "description": "Report new homes in FEMA Zone X shaded (0.2% annual chance, or 1% "
     "with depths under 1 ft, or levee-protected)", "value": None, "units": "FEMA Zone X subtypes",
     "enforcement": "report", "kind": "Data layer",
     "source": "FEMA NFHL. Not a Special Flood Hazard Area; no building restriction, flagged for disclosure.",
     "source_url": "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28"},
]


def enforcement(rule_id, noise_rule="report", flood_rule="report"):
    """How a rule is enforced in this run (R3 / R4 switch between 'report' and 'no_residential')."""
    if rule_id == "R3_SFHA":
        return "no_residential" if flood_rule == "hard" else "report"
    if rule_id == "R4_RAIL_NOISE":
        return "no_residential" if noise_rule == "hard" else "report"
    return next(r["enforcement"] for r in RULES if r["rule_id"] == rule_id)


def rules_table(panels="", noise_rule="report", flood_rule="report"):
    rows = []
    for r in RULES:
        r = dict(r)
        r["source"] = r["source"].replace("{panels}", panels or "see manifest")
        r["enforcement"] = enforcement(r["rule_id"], noise_rule, flood_rule)
        r["date_checked"] = CHECKED
        rows.append(r)
    return pd.DataFrame(rows)


def _u(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return unary_union(geoms) if geoms else None


def constraint_layers(off, noise_rule="report", flood_rule="report"):
    """Polygons for each rule from the official layers (None where a layer is missing).
    off: the official_data module."""
    out = {}
    streams = off.load("streams")
    if streams is not None:
        exposed = streams[streams["COVEREDSTATUS"].fillna("Exposed") != "Covered"]
        out["R1_CREEK_SETBACK"] = _u(exposed.geometry.buffer(CREEK_SETBACK_FT + BANK_ALLOWANCE_FT))
        out["creek_lines"] = _u(exposed.geometry)
    flood = off.load("flood_zones")
    if flood is not None:
        sub = flood["ZONE_SUBTY"].fillna("")
        out["R2_FLOODWAY"] = _u(flood[sub.str.contains("FLOODWAY")].geometry)
        out["R3_SFHA"] = _u(flood[(flood["SFHA_TF"] == "T") & ~sub.str.contains("FLOODWAY")].geometry)
        out["R7_FLOOD_X"] = _u(flood[(flood["FLD_ZONE"] == "X") & sub.str.contains("0.2 PCT|1 PCT|LEVEE")].geometry)
    noise = off.load("noise")
    if noise is not None:
        rail = noise[noise["NAME"].fillna("").str.contains("Rail", case=False)]
        out["R4_RAIL_NOISE"] = _u(rail.geometry)
    out["noise_rule"] = noise_rule
    out["flood_rule"] = flood_rule
    return out


def rules_layer(layers, frame, crs):
    """One polygon per rule, clipped to the study frame, for the map and the GeoPackage."""
    rows = []
    for r in RULES:
        g = layers.get(r["rule_id"])
        if g is None or g.is_empty:
            continue
        g = g.intersection(frame)
        if g.is_empty:
            continue
        how = enforcement(r["rule_id"], layers.get("noise_rule", "report"), layers.get("flood_rule", "report"))
        rows.append({"rule_id": r["rule_id"], "enforcement": how, "description": r["description"],
                     "acres": round(g.area / 43560, 1), "geometry": g})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
