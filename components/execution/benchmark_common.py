"""Common benchmark logic shared between single and bulk benchmarks."""

import statistics
import random
import copy
from components.utils.benchmark_utils import execute_and_measure
from components.utils.helpers import get_cost_at_time, parse_gaetano_metadata
from components.models import (
    ExperimentConfig, InstanceMetadata, RunStatistics, BenchmarkResult
)
from components import constants as C


from components.execution.hygese_solver import solve_hgs

# Time checkpoints for tracking convergence
TIME_CHECKPOINTS = [1, 5, 10, 15, 20, 30, 60]


def prepare_experiments(settings) -> list[ExperimentConfig]:
    """Prepare experiment configurations from settings."""
    experiments = []
    engine = settings.get("engine", "ortools")
    
    if engine == "ortools":
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
                    "target_gap_percent": settings.get("target_gap"), # Store as percentage
                    "no_improvement_limit": settings["no_improv"],
                    "no_improvement_neighbors_limit": settings["no_improv_iter"],
                    "continue_after_target": settings.get("continue_after_gap", False)
                }
                experiments.append(ExperimentConfig(
                    name=f"{mh_label} [{fs_label}]",
                    fs_label=fs_label,
                    mh_label=mh_label,
                    func=settings["solver_func"],
                    kwargs=kw,
                    reps=settings["reps"]
                ))
    elif engine == "hgs":
        # HGS uses its own internal logic, usually one experiment per run
        kw = {
            "time_limit_seconds": settings["time_limit"],
            "no_improvement_limit_iterations": settings["no_improv_iter"],
            "target_gap_percent": settings.get("target_gap")
        }
        # Include HGS specific parameters
        kw.update(settings.get("hgs_params", {}))
        
        experiments.append(ExperimentConfig(
            name="HGS-CVRP",
            fs_label="HGS-CVRP",
            mh_label="HGS-CVRP",
            func=solve_hgs,
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
                     checkpoint_data, time_checkpoints=TIME_CHECKPOINTS,
                     time_to_target_list=None, engine="ortools") -> BenchmarkResult:
    """Build a result row with all standard columns."""
    
    if costs:
        best = min(costs)
        avg = statistics.mean(costs)
    else:
        best = None
        avg = None
    
    best_gap = ((best - bks_cost) / bks_cost * 100) if (bks_cost and best) else None
    avg_gap = ((avg - bks_cost) / bks_cost * 100) if (bks_cost and avg) else None
    
    avg_time_to_target = None
    if time_to_target_list:
        # Filter out None values
        valid_ttt = [t for t in time_to_target_list if t is not None]
        if valid_ttt:
            avg_time_to_target = statistics.mean(valid_ttt)

    # Build checkpoint columns dictionary
    checkpoints = {}
    for t_chk in time_checkpoints:
        costs_at_t = checkpoint_data.get(t_chk, [])
        if costs_at_t:
            checkpoints[C.get_checkpoint_avg_cost_col(t_chk)] = statistics.mean(costs_at_t)
            checkpoints[C.get_checkpoint_best_cost_col(t_chk)] = min(costs_at_t)
        else:
            checkpoints[C.get_checkpoint_avg_cost_col(t_chk)] = None
            checkpoints[C.get_checkpoint_best_cost_col(t_chk)] = None
    
    # Use the number of routes in the best solution as the vehicle count
    used_vehicles = len(best_routes) if best_routes else instance_meta.vehicles

    return BenchmarkResult(
        instance=instance_meta.name,
        depot_layout=instance_meta.depot_layout,
        customer_layout=instance_meta.customer_layout,
        demand_type=instance_meta.demand_profile,
        route_class=instance_meta.route_class,
        climate=instance_meta.climate,
        customers=instance_meta.customers,
        vehicles=used_vehicles,
        capacity=instance_meta.capacity,
        first_solution=exp.fs_label,
        metaheuristic=exp.mh_label,
        engine=engine,
        repetitions=exp.reps,
        avg_cost=avg,
        best_cost=best,
        bks_cost=bks_cost,
        best_gap_percent=best_gap,
        avg_gap_percent=avg_gap,
        avg_cpu_time=statistics.mean(times) if times else None,
        avg_time_to_target=avg_time_to_target,
        checkpoints=checkpoints
    )


def run_experiment_reps(exp: ExperimentConfig, instance_data, reps, bks_cost=None, progress_callback=None, time_checkpoints=TIME_CHECKPOINTS) -> RunStatistics:
    """Run an experiment for given repetitions and return collected data."""
    
    costs, times, neighbors_list, best_routes = [], [], [], None
    best_cost_found = float('inf')
    checkpoint_collectors = {t: [] for t in time_checkpoints}
    all_histories = []  # Store full convergence history from each run
    ttt_list = []  # Time to target list
    
    for rep in range(reps):
        if progress_callback:
            progress_callback(rep, reps)
        
        cur_kw = exp.kwargs.copy()
        cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
        
        # Calculate target_cost from percentage and BKS if available
        target_gap_pct = cur_kw.pop("target_gap_percent", None)
        if target_gap_pct is not None and bks_cost is not None:
            cur_kw["target_cost"] = bks_cost * (1 + target_gap_pct / 100)
        else:
            cur_kw["target_cost"] = None
        
        res = execute_and_measure(exp.func, instance_data, **cur_kw)
        
        if res.cpu_time is not None:
            times.append(res.cpu_time)
        if res.objective_value is not None:
            costs.append(res.objective_value)
            
            # Keep track of the BEST routes found across all repetitions
            if res.objective_value < best_cost_found:
                best_cost_found = res.objective_value
                best_routes = res.routes
                
        if res.accepted_neighbors is not None:
            neighbors_list.append(res.accepted_neighbors)
        
        # Track time to target
        if hasattr(res, 'time_to_target'):
             ttt_list.append(res.time_to_target)

        # Collect costs at time checkpoints
        if res.history:
            # Store full history for convergence visualization
            all_histories.append(res.history)
            
            for t_chk in time_checkpoints:
                cost_at_t = get_cost_at_time(res.history, t_chk)
                if cost_at_t is not None:
                    checkpoint_collectors[t_chk].append(cost_at_t)
    
    return RunStatistics(
        costs=costs,
        times=times,
        neighbors_list=neighbors_list,
        best_routes=best_routes,
        checkpoints=checkpoint_collectors,
        all_histories=all_histories,
        time_to_target_list=ttt_list
    )

def run_experiment_with_vehicle_retry(exp: ExperimentConfig, inst_data, bks_val, instance_meta, 
                                      max_retries=5, log_fn=None, progress_fn=None, time_checkpoints=TIME_CHECKPOINTS,
                                      engine="ortools"):
    """
    Core logic for running one experiment with vehicle retries.
    Returns: (dict result_row, bool found_solution, int final_vehicle_attempt)
    """
    
    for vehicle_attempt in range(max_retries + 1):
        working_instance = inst_data
        current_meta = instance_meta
        
        if vehicle_attempt > 0:
            working_instance = copy.deepcopy(inst_data)
            original_num = working_instance['num_vehicles']
            new_num = original_num + vehicle_attempt
            working_instance['num_vehicles'] = new_num
            working_instance['vehicle_capacities'] = [working_instance['capacity']] * working_instance['num_vehicles']
            
            # Create a copy of meta to update the reported vehicle count
            current_meta = copy.copy(instance_meta)
            current_meta.vehicles = new_num
            
            if log_fn:
                log_fn(f"Retrying {instance_meta.name} | {exp.name} with {new_num} vehicles (+{vehicle_attempt})")

        # progress_fn needs to handle rep, reps, and vehicle_attempt
        def wrapped_progress(rep, reps):
            if progress_fn:
                progress_fn(rep, reps, vehicle_attempt)

        data = run_experiment_reps(exp, working_instance, exp.reps, bks_val, wrapped_progress, time_checkpoints)
        
        # If we found a solution, or if this was the last attempt
        if data.costs or vehicle_attempt == max_retries:
            # If solution was found, use the ACTUAL number of routes from the solution
            # instead of the fleet size limit we gave the solver
            if data.costs and data.best_routes:
                current_meta = copy.copy(instance_meta)
                current_meta.vehicles = len(data.best_routes)

            result_row = build_result_row(exp, current_meta, data.costs, data.times,
                                          data.neighbors_list, data.best_routes, bks_val, data.checkpoints, 
                                          time_checkpoints, data.time_to_target_list, engine=engine)
            return result_row.to_dict(), bool(data.costs), vehicle_attempt, data.best_routes
            
    return None, False, 0, None

