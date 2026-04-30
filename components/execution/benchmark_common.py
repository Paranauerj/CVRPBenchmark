"""Common benchmark logic shared between single and bulk benchmarks."""

import statistics
import random
from components.utils.benchmark_utils import execute_and_measure
from components.utils.helpers import get_cost_at_time, parse_gaetano_metadata
from components.models import (
    ExperimentConfig, InstanceMetadata, RunStatistics, BenchmarkResult
)
from components import constants as C


# Time checkpoints for tracking convergence
TIME_CHECKPOINTS = [5, 10, 15, 20, 30, 60]


def prepare_experiments(settings) -> list[ExperimentConfig]:
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
                "no_improvement_neighbors_limit": settings["no_improv_iter"]
            }
            experiments.append(ExperimentConfig(
                name=f"{mh_label} [{fs_label}]",
                fs_label=fs_label,
                mh_label=mh_label,
                func=settings["solver_func"],
                kwargs=kw,
                reps=settings["reps"]
            ))
    return experiments


def extract_instance_metadata(instance_data, instance_name) -> InstanceMetadata:
    """Extract metadata from instance data."""
    # Determine if Gaetano (num_nodes includes depot) or Uchoa (num_nodes doesn't)
    is_gaetano = instance_name.startswith("LDG")
    num_nodes = instance_data.get('num_nodes', 0)
    customers = (num_nodes - 1) if is_gaetano else num_nodes
    
    meta_features = parse_gaetano_metadata(instance_name)
    
    return InstanceMetadata(
        name=instance_name,
        customers=customers,
        vehicles=instance_data.get('num_vehicles', 0),
        capacity=instance_data.get('capacity', 0),
        depot_layout=meta_features.get("Depot Layout", "Unknown"),
        customer_layout=meta_features.get("Customer Layout", "Unknown"),
        demand_profile=meta_features.get("Demand Profile ID", "Unknown"),
        route_class=meta_features.get("Route Length Class", "Unknown"),
        climate=meta_features.get("Climate", "Unknown")
    )


def build_result_row(exp: ExperimentConfig, instance_meta: InstanceMetadata, 
                     costs, times, neighbors_list, best_routes, bks_cost, 
                     checkpoint_data) -> BenchmarkResult:
    """Build a result row with all standard columns."""
    if costs:
        best = min(costs)
        avg = statistics.mean(costs)
    else:
        best = None
        avg = None
    
    best_gap = ((best - bks_cost) / bks_cost * 100) if (bks_cost and best) else None
    avg_gap = ((avg - bks_cost) / bks_cost * 100) if (bks_cost and avg) else None
    
    # Build checkpoint columns dictionary
    checkpoints = {}
    for t_chk in TIME_CHECKPOINTS:
        costs_at_t = checkpoint_data.get(t_chk, [])
        if costs_at_t:
            checkpoints[C.get_checkpoint_avg_cost_col(t_chk)] = statistics.mean(costs_at_t)
            checkpoints[C.get_checkpoint_best_cost_col(t_chk)] = min(costs_at_t)
        else:
            checkpoints[C.get_checkpoint_avg_cost_col(t_chk)] = None
            checkpoints[C.get_checkpoint_best_cost_col(t_chk)] = None
    
    return BenchmarkResult(
        instance=instance_meta.name,
        depot_layout=instance_meta.depot_layout,
        customer_layout=instance_meta.customer_layout,
        demand_type=instance_meta.demand_profile,
        route_class=instance_meta.route_class,
        climate=instance_meta.climate,
        customers=instance_meta.customers,
        vehicles=instance_meta.vehicles,
        capacity=instance_meta.capacity,
        first_solution=exp.fs_label,
        metaheuristic=exp.mh_label,
        repetitions=exp.reps,
        avg_cost=avg,
        best_cost=best,
        bks_cost=bks_cost,
        best_gap_percent=best_gap,
        avg_gap_percent=avg_gap,
        avg_cpu_time=statistics.mean(times) if times else None,
        checkpoints=checkpoints
    )


def run_experiment_reps(exp: ExperimentConfig, instance_data, reps, progress_callback=None) -> RunStatistics:
    """Run an experiment for given repetitions and return collected data."""
    costs, times, neighbors_list, best_routes = [], [], [], None
    checkpoint_collectors = {t: [] for t in TIME_CHECKPOINTS}
    all_histories = []  # Store full convergence history from each run
    
    for rep in range(reps):
        if progress_callback:
            progress_callback(rep, reps)
        
        cur_kw = exp.kwargs.copy()
        cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
        
        res = execute_and_measure(exp.func, instance_data, **cur_kw)
        
        if res.cpu_time is not None:
            times.append(res.cpu_time)
        if res.objective_value is not None:
            costs.append(res.objective_value)
            if not best_routes:
                best_routes = res.routes
        if res.accepted_neighbors is not None:
            neighbors_list.append(res.accepted_neighbors)
        
        # Collect costs at time checkpoints
        if res.history:
            # Store full history for convergence visualization
            all_histories.append(res.history)
            
            for t_chk in TIME_CHECKPOINTS:
                cost_at_t = get_cost_at_time(res.history, t_chk)
                if cost_at_t is not None:
                    checkpoint_collectors[t_chk].append(cost_at_t)
    
    return RunStatistics(
        costs=costs,
        times=times,
        neighbors_list=neighbors_list,
        best_routes=best_routes,
        checkpoints=checkpoint_collectors,
        all_histories=all_histories
    )
