"""Common benchmark logic shared between single and bulk benchmarks."""

import statistics
import random
from components.utils.benchmark_utils import execute_and_measure
from components.utils.helpers import get_cost_at_time, parse_gaetano_metadata


# Time checkpoints for tracking convergence
TIME_CHECKPOINTS = [5, 10, 15, 20, 30, 60]


def prepare_experiments(settings):
    """Prepare experiment configurations from settings."""
    experiments = []
    for fs in settings["sel_fs"]:
        for mh in settings["sel_mh"]:
            fs_label = fs.split('(')[0].strip()
            mh_label = mh.split('(')[0].strip()
            kw = {
                "first_solution_strategy": settings["fs_enum"][fs],
                "local_search_metaheuristic": settings["mh_enum"][mh],
                "time_limit_seconds": settings["time_limit"],
                "solution_limit": settings["sol_limit"],
                "lns_time_limit_seconds": settings["lns_limit"],
                "target_cost": settings.get("target_gap"),
                "no_improvement_limit": settings["no_improv"],
                "no_improvement_iterations_limit": settings["no_improv_iter"]
            }
            experiments.append({
                "name": f"{mh_label} [{fs_label}]",
                "fs_label": fs_label,
                "mh_label": mh_label,
                "func": settings["solver_func"],
                "kwargs": kw,
                "reps": settings["reps"]
            })
    return experiments


def extract_instance_metadata(instance_data, instance_name):
    """Extract metadata from instance data."""
    meta_features = parse_gaetano_metadata(instance_name)
    return {
        "name": instance_name,
        "meta": meta_features,
        "customers": instance_data.get('num_nodes', 0) - 1,
        "vehicles": instance_data.get('num_vehicles', 0),
        "capacity": instance_data.get('capacity', 0)
    }


def build_result_row(exp, instance_meta, costs, times, iters_list, best_routes, bks_cost, checkpoint_data):
    """Build a result row with all standard columns."""
    if costs:
        best = min(costs)
        avg = statistics.mean(costs)
    else:
        best = None
        avg = None
    
    best_gap = ((best - bks_cost) / bks_cost * 100) if (bks_cost and best) else None
    avg_gap = ((avg - bks_cost) / bks_cost * 100) if (bks_cost and avg) else None
    
    row = {
        "Instance": instance_meta["name"],
        "Depot Layout": instance_meta["meta"]["Depot Layout"],
        "Cust Layout": instance_meta["meta"]["Customer Layout"],
        "Demand Type": instance_meta["meta"]["Demand Profile ID"],
        "Route Class": instance_meta["meta"]["Route Length Class"],
        "Climate": instance_meta["meta"]["Climate"],
        "Customers": instance_meta["customers"],
        "Vehicles": instance_meta["vehicles"],
        "Capacity": instance_meta["capacity"],
        "First Solution": exp.get("fs_label"),
        "Metaheuristic": exp.get("mh_label"),
        "Repetitions": exp["reps"],
        "Avg Cost": avg,
        "Best Cost": best,
        "BKS Cost": bks_cost,
        "Best Gap (%)": best_gap,
        "Avg Gap (%)": avg_gap,
        "Avg CPU Time (s)": statistics.mean(times) if times else None
    }
    
    # Add time checkpoint columns
    for t_chk in TIME_CHECKPOINTS:
        costs_at_t = checkpoint_data.get(t_chk, [])
        if costs_at_t:
            row[f"Avg Cost @ {t_chk}s"] = statistics.mean(costs_at_t)
            row[f"Best Cost @ {t_chk}s"] = min(costs_at_t)
        else:
            row[f"Avg Cost @ {t_chk}s"] = None
            row[f"Best Cost @ {t_chk}s"] = None
    
    return row


def run_experiment_reps(exp, instance_data, reps, progress_callback=None):
    """Run an experiment for given repetitions and return collected data."""
    costs, times, iters_list, best_routes = [], [], [], None
    checkpoint_collectors = {t: [] for t in TIME_CHECKPOINTS}
    
    for rep in range(reps):
        if progress_callback:
            progress_callback(rep, reps)
        
        cur_kw = exp["kwargs"].copy()
        cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
        
        res = execute_and_measure(exp["func"], instance_data, **cur_kw)
        
        if res["cpu_time"] is not None:
            times.append(res["cpu_time"])
        if res["objective_value"] is not None:
            costs.append(res["objective_value"])
            if not best_routes:
                best_routes = res["routes"]
        if res["iterations"] is not None:
            iters_list.append(res["iterations"])
        
        # Collect costs at time checkpoints
        if res.get("history"):
            for t_chk in TIME_CHECKPOINTS:
                cost_at_t = get_cost_at_time(res["history"], t_chk)
                if cost_at_t is not None:
                    checkpoint_collectors[t_chk].append(cost_at_t)
    
    return {
        "costs": costs,
        "times": times,
        "iters_list": iters_list,
        "best_routes": best_routes,
        "checkpoints": checkpoint_collectors
    }
