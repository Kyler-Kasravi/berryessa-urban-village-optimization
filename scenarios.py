"""
Scenario runs for the planner summary
=====================================

Reruns site_pipeline.py with different settings (same data) and tabulates how the results change, so
every number quoted in the planner summary comes from a script.

    python scenarios.py                    # every scenario, two at a time (about an hour)
    python scenarios.py strict grid        # only some
    python scenarios.py seeds              # only the four extra seeds (robustness check)
    python scenarios.py --summary-only     # re-tabulate scenarios.csv from runs already made

Scenarios (the main run is the default berryessa.gpkg and is not rerun here)
  long      600 generations instead of 300: did the main run converge?
  strict    no new homes in the FEMA 1% flood zone or the rail noise contour (--flood-rule hard --noise-rule hard)
  grid      optimize with the capstone's grid (taxicab) distances, then measure the result by walking distance
  capstone  the capstone's own land-use mix (includes light industry) instead of the City's
  seed1..seed4   the main settings with other random seeds: are the conclusions an accident of seed 7?

Output: scenarios/<name>.gpkg (+ <name>_run_summary.json, maps/<name>_*.png) and scenarios.csv
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyogrio

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scenarios")
SCENARIOS = {
    "long": (["--gens", "600"], "600 generations (convergence check)"),
    "strict": (["--flood-rule", "hard", "--noise-rule", "hard"], "No new homes in the 1% flood zone or rail noise contour"),
    "grid": (["--distance", "grid"], "Optimized with grid distances, measured by walking distance"),
    "capstone": (["--program", "capstone"], "Capstone land-use mix (includes light industry)"),
}
SEEDS = (1, 2, 3, 4)
for _s in SEEDS:
    SCENARIOS[f"seed{_s}"] = (["--seed", str(_s)], f"Main settings, seed {_s}")


def run(name):
    args, _ = SCENARIOS[name]
    out = os.path.join(OUT, f"{name}.gpkg")
    log = os.path.join(OUT, f"{name}.log")
    cmd = [sys.executable, "-u", os.path.join(HERE, "site_pipeline.py"), "--module", "Nsga3 Z.py", "--out", out] + args
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"{name:<9} {'ok' if rc == 0 else f'FAILED (see {log})'}", flush=True)
    return name, rc


def summarize(name, gpkg, label):
    p = pyogrio.read_dataframe(gpkg, layer="pareto_front", read_geometry=False)
    plan = pyogrio.read_dataframe(gpkg, layer="plan_comparison", read_geometry=False)
    reps = pyogrio.read_dataframe(gpkg, layer="representatives", read_geometry=False)
    stem = "" if name == "main" else f"{name}_"
    folder = HERE if name == "main" else OUT
    try:
        summ = json.load(open(os.path.join(folder, f"{stem}run_summary.json")))
    except OSError:
        summ = {}
    pl = plan[plan.scenario == "city_plan"].iloc[0]
    bal = p[p.solution_id == int(reps[reps.key == "balanced"].solution_id.iloc[0])].iloc[0]
    return {
        "scenario": name, "description": label, "pareto_solutions": len(p),
        "best_job_walk_ft": p.job_walk_ft.min(), "best_services_walk_ft": p.services_walk_ft.min(),
        "best_livability": p.livability.max(),
        "best_new_home_walk_to_bart_ft": p.new_homes_walk_to_bart_ft.min(),
        "balanced_job_walk_ft": bal.job_walk_ft, "balanced_services_walk_ft": bal.services_walk_ft,
        "balanced_livability": bal.livability, "balanced_new_home_walk_to_bart_ft": bal.new_homes_walk_to_bart_ft,
        "balanced_new_home_cells": bal.new_home_cells,
        "balanced_new_homes_5min_bart": bal.new_homes_5min_bart,
        "balanced_homes_in_noise_contour": bal.new_homes_in_noise_contour,
        "balanced_homes_in_flood_zone": bal.new_homes_in_flood_zone,
        "layouts_beating_plan_on_services_and_livability": int(((p.services_walk_ft < pl.services_walk_ft) &
                                                                 (p.livability > pl.livability)).sum()),
        "layouts_beating_plan_on_job_walk": int((p.job_walk_ft < pl.job_walk_ft).sum()),
        "rank_corr_job_grid_vs_walk": round(p.job_grid_ft.corr(p.job_walk_ft, method="spearman"), 2),
        "rank_corr_services_grid_vs_walk": round(p.services_grid_ft.corr(p.services_walk_ft, method="spearman"), 2),
        "hypervolume_final": summ.get("hypervolume_final"),
        "hypervolume_99pct_generation": summ.get("hypervolume_99pct_generation"),
        "plan_job_walk_ft": pl.job_walk_ft, "plan_services_walk_ft": pl.services_walk_ft,
        "plan_livability": pl.livability, "plan_new_home_walk_to_bart_ft": pl.new_homes_walk_to_bart_ft,
        "plan_new_home_cells": pl.new_home_cells,
    }


def main():
    want = sys.argv[1:]
    if "seeds" in want:
        want = [w for w in want if w != "seeds"] + [f"seed{s}" for s in SEEDS]
    os.makedirs(OUT, exist_ok=True)
    if "--summary-only" not in want:                 # --summary-only: just re-tabulate existing runs
        names = [a for a in want if a in SCENARIOS] or list(SCENARIOS)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, names))
    rows = []
    main_gpkg = os.path.join(HERE, "berryessa.gpkg")
    if os.path.exists(main_gpkg):
        rows.append(summarize("main", main_gpkg, "Defaults: walking distances, City's land-use mix, rules reported, seed 7"))
    for n in SCENARIOS:
        g = os.path.join(OUT, f"{n}.gpkg")
        if os.path.exists(g):
            rows.append(summarize(n, g, SCENARIOS[n][1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "scenarios.csv"), index=False)
    seeds = df[df.scenario.isin(["main"] + [f"seed{s}" for s in SEEDS])]
    if len(seeds) > 1:
        cols = ["best_job_walk_ft", "best_services_walk_ft", "best_livability", "best_new_home_walk_to_bart_ft",
                "layouts_beating_plan_on_services_and_livability", "layouts_beating_plan_on_job_walk"]
        spread = seeds[cols].agg(["min", "max", "mean"]).T
        spread.to_csv(os.path.join(HERE, "scenarios_seed_spread.csv"))
        print(f"\nAcross {len(seeds)} seeds:\n{spread.round(1).to_string()}")
    pd.set_option("display.width", 220)
    print(df.set_index("scenario").T.to_string())


if __name__ == "__main__":
    main()
