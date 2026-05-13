import os
from ortools.constraint_solver import routing_enums_pb2 as RE
from components.execution.benchmark_runner import BenchmarkRunner
from components.execution import configurable_solver
from components.execution.benchmark_common import (
    extract_instance_metadata, run_experiment_with_vehicle_retry
)
from components.models import ExperimentConfig
from components.ui.sidebar import FIRST_SOLUTIONS, METAHEURISTICS
from components.utils import instance_data_parser, solution_parser
from components import constants as C

# --- Configuration ---
INSTANCES_DIR = "instances/gaetano"
RESULTS_DIR = "gaetano_chunk_results"
MAX_INSTANCES = 5      # Limit for testing
NUM_PARALLEL = 5       # Number of concurrent workers
CHUNK_SIZE = 10        # Number of instances per chunk

# Reverse maps for labeling
FS_NAME_MAP = {v: k for k, v in FIRST_SOLUTIONS.items()}
MH_NAME_MAP = {v: k for k, v in METAHEURISTICS.items()}

BENCHMARK_PARAMS = {
    "time_limit_seconds": 10,
    "solution_limit": None,
    "lns_time_limit_seconds": None,
    "no_improvement_limit": None,
    "no_improvement_neighbors_limit": None,
    "continue_after_target": True,
    "reps": 1,
}

# Define strategies to test
SELECTED_FS = [
    RE.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    RE.FirstSolutionStrategy.SAVINGS,
]

SELECTED_MH = [
    RE.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    RE.LocalSearchMetaheuristic.TABU_SEARCH,
]

def get_experiments() -> list[ExperimentConfig]:
    """Generates a list of OR-Tools experiments based on selected strategies."""
    experiments = []
    for fs in SELECTED_FS:
        for mh in SELECTED_MH:
            fs_label = FS_NAME_MAP.get(fs, str(fs))
            mh_label = MH_NAME_MAP.get(mh, str(mh))
            kwargs = BENCHMARK_PARAMS.copy()
            reps = kwargs.pop("reps")
            kwargs.update({
                "first_solution_strategy": fs, 
                "local_search_metaheuristic": mh
            })
            experiments.append(ExperimentConfig(
                name=f"OR-Tools: {mh_label} [{fs_label}]",
                fs_label=fs_label, 
                mh_label=mh_label,
                func=configurable_solver.solve_cvrp,
                kwargs=kwargs, 
                reps=reps
            ))
    return experiments

def process_instance(vrp_path: str, experiments: list[ExperimentConfig]):
    """
    Standard processing for OR-Tools instances.
    Handles BKS loading and vehicle retries.
    """
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    try:
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        # Look for BKS (Best Known Solution)
        sol_path = vrp_path.replace(".vrp", ".sol")
        bks_val = None
        if os.path.exists(sol_path):
            try:
                bks_val = solution_parser.parse_solution_file(sol_path)
            except Exception: 
                pass

        results = []
        for exp in experiments:
            result_row_dict, _, _, _ = run_experiment_with_vehicle_retry(
                exp, inst_data, bks_val, instance_meta, max_retries=5, engine="ortools"
            )
            if result_row_dict:
                results.append(result_row_dict)
        return results
    except Exception as e:
        print(f"Error processing {inst_name}: {e}")
        return []

def main():
    """Main execution block for OR-Tools benchmark."""
    runner = BenchmarkRunner(
        name="OR-Tools Gaetano Benchmark",
        instances_dir=INSTANCES_DIR,
        results_dir=RESULTS_DIR,
        max_instances=MAX_INSTANCES,
        num_parallel=NUM_PARALLEL,
        chunk_size=CHUNK_SIZE
    )
    
    # Define columns to drop for performance-only report
    filter_cols = [
        C.COL_DEPOT_LAYOUT, C.COL_CUSTOMER_LAYOUT, C.COL_DEMAND_TYPE, 
        C.COL_ROUTE_CLASS, C.COL_CLIMATE, C.COL_CUSTOMERS, C.COL_CAPACITY
    ]
    
    runner.run(
        experiments=get_experiments(),
        process_instance_fn=process_instance,
        filter_cols=filter_cols
    )

if __name__ == "__main__":
    main()
