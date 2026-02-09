"""Single instance benchmark execution."""

import streamlit as st
import statistics
import random
from benchmark_utils import execute_and_measure


def run_single_benchmark(instance_data, settings, bks_cost):
    """
    Run a single instance benchmark with the given settings.
    Returns a DataFrame with results.
    """
    import pandas as pd
    
    experiments = []
    for fs in settings["sel_fs"]:
        for mh in settings["sel_mh"]:
            algo = f"{mh.split('(')[0].strip()} [{fs.split('(')[0].strip()}]"
            kw = {
                "first_solution_strategy": settings["fs_enum"][fs],
                "local_search_metaheuristic": settings["mh_enum"][mh],
                "time_limit_seconds": settings["time_limit"],
                "solution_limit": settings["sol_limit"],
                "lns_time_limit_seconds": settings["lns_limit"],
                "target_cost": settings["target_val"],
                "no_improvement_limit": settings["no_improv"],
                "no_improvement_iterations_limit": settings["no_improv_iter"]
            }
            experiments.append({
                "name": algo,
                "func": settings["solver_func"],
                "kwargs": kw,
                "reps": settings["reps"]
            })

    results_list = []
    total = sum(e["reps"] for e in experiments)
    prog = st.progress(0.0)
    cnt = 0
    stat = st.empty()

    for exp in experiments:
        stat.text(f"Running: {exp['name']}")
        costs, times, iters_list, best_routes = [], [], [], None
        best_cost_run = float('inf')
        exp_histories = []

        for _ in range(exp["reps"]):
            cnt += 1
            prog.progress(cnt / total)
            cur_kw = exp["kwargs"].copy()
            cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
            
            res = execute_and_measure(exp["func"], instance_data, **cur_kw)
            
            if res["cpu_time"] is not None:
                times.append(res["cpu_time"])
            if res["objective_value"] is not None:
                costs.append(res["objective_value"])
                if res["objective_value"] < best_cost_run:
                    best_cost_run = res["objective_value"]
                    best_routes = res["routes"]
            if res["iterations"] is not None:
                iters_list.append(res["iterations"])
            
            if res.get("history"):
                exp_histories.append(res["history"])
        
        st.session_state.all_histories[exp["name"]] = exp_histories

        if costs:
            best = min(costs)
            avg = statistics.mean(costs)
            vehicles_used = len(best_routes) if best_routes else None
            row = {
                "Algorithm": exp["name"],
                "Best Cost": best,
                "Avg Cost": avg,
                "CPU Time (s)": statistics.mean(times) if times else None,
                "Vehicles Used": vehicles_used,
                "Accepted Neighbors": int(statistics.mean(iters_list)) if iters_list else None,
                "Repetitions": exp["reps"],
                "_routes": best_routes
            }
            if bks_cost:
                row["Best Gap (%)"] = ((best - bks_cost) / bks_cost) * 100.0
                row["Avg Gap (%)"] = ((avg - bks_cost) / bks_cost) * 100.0
            results_list.append(row)
        else:
            row = {
                "Algorithm": exp["name"],
                "Best Cost": "No Solution",
                "Avg Cost": "No Solution",
                "CPU Time (s)": statistics.mean(times) if times else None,
                "Vehicles Used": None,
                "Accepted Neighbors": None,
                "Repetitions": exp["reps"],
                "_routes": None
            }
            if bks_cost:
                row["Best Gap (%)"] = "N/A"
                row["Avg Gap (%)"] = "N/A"
            results_list.append(row)

    df = pd.DataFrame(results_list)
    
    # Calculate best run solution gap
    global_best_cost = None
    for row in results_list:
        if isinstance(row["Best Cost"], (int, float)) and row["Best Cost"] < float('inf'):
            if global_best_cost is None or row["Best Cost"] < global_best_cost:
                global_best_cost = row["Best Cost"]
    
    if global_best_cost is not None:
        for i, row in enumerate(results_list):
            if isinstance(row["Best Cost"], (int, float)):
                row["Best Run Gap (%)"] = ((row["Best Cost"] - global_best_cost) / global_best_cost) * 100.0
            else:
                row["Best Run Gap (%)"] = "N/A"
        df = pd.DataFrame(results_list)
    
    stat.empty()
    prog.empty()
    
    return df
