import os
from components.execution.benchmark_runner import BenchmarkRunner
from components.execution.hygese_solver import solve_hgs
from components.execution.benchmark_common import (
    extract_instance_metadata, run_experiment_with_vehicle_retry
)
from components.models import ExperimentConfig
from components.utils import instance_data_parser, solution_parser
from components import constants as C

# --- Configuration ---
INSTANCES_DIR = "instances/gaetano"
RESULTS_DIR = "hgs_gaetano_results"
MAX_INSTANCES = None   # Run all instances
SAVE_SOLUTIONS = True  # Flag to generate .sol files
NUM_PARALLEL = 8       # Number of concurrent workers
CHUNK_SIZE = 5         # Smaller chunk size for longer runs
SOLVE_ONLY_UNSOLVED = True # Only run on instances without .sol files
MAX_VEHICLE_RETRIES = 10    # Increased to 10 for harder instances

BENCHMARK_PARAMS = {
    "time_limit_seconds": 600,
    "no_improvement_limit_iterations": 1000000000,
    "reps": 3,
}

def get_experiment() -> ExperimentConfig:
    """Creates the HGS experiment configuration."""
    kwargs = BENCHMARK_PARAMS.copy()
    reps = kwargs.pop("reps")
    return ExperimentConfig(
        name="HGS-CVRP",
        fs_label="HGS-CVRP",
        mh_label="HGS-CVRP",
        func=solve_hgs,
        kwargs=kwargs,
        reps=reps
    )

def process_instance(vrp_path: str, experiments: list[ExperimentConfig]):
    """
    Specialized processing for HGS benchmark.
    Loads instance, runs experiment, and handles vehicle retries.
    """
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    exp = experiments[0]
    
    # Check if already solved
    sol_path = vrp_path.replace(".vrp", ".sol")
    if SOLVE_ONLY_UNSOLVED and os.path.exists(sol_path):
        print(f"Skipping {inst_name}: already solved.")
        return []

    try:
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        # Look for BKS (Best Known Solution)
        bks_val = None
        if os.path.exists(sol_path):
            try:
                bks_val = solution_parser.parse_solution_file(sol_path)
            except Exception: 
                pass

        # Run HGS with vehicle retry
        result_row_dict, found, final_attempt, best_routes = run_experiment_with_vehicle_retry(
            exp, inst_data, bks_val, instance_meta, max_retries=MAX_VEHICLE_RETRIES, engine="hgs"
        )
        
        if not found or not result_row_dict:
            return []

        best_cost = result_row_dict[C.COL_BEST_COST]

        # Save .sol file if it's better than existing or no existing
        if SAVE_SOLUTIONS and best_routes:
            save_needed = True
            if os.path.exists(sol_path):
                try:
                    existing_cost = solution_parser.parse_solution_file(sol_path)
                    if existing_cost is not None and best_cost >= existing_cost:
                        save_needed = False
                except Exception:
                    pass
            
            if save_needed:
                action = "Updating" if os.path.exists(sol_path) else "Saving new"
                print(f"  [SOL] {action} best solution for {inst_name}: {best_cost:.2f}")
                solution_parser.save_solution_file(sol_path, best_routes, best_cost)
        
        return [result_row_dict]
    except Exception as e:
        print(f"Error processing {inst_name}: {e}")
        return []

def main():
    """Main execution block for HGS benchmark."""
    runner = BenchmarkRunner(
        name="HGS Gaetano Benchmark",
        instances_dir=INSTANCES_DIR,
        results_dir=RESULTS_DIR,
        max_instances=MAX_INSTANCES,
        num_parallel=NUM_PARALLEL,
        chunk_size=CHUNK_SIZE
    )
    
    runner.run(
        experiments=[get_experiment()],
        process_instance_fn=process_instance
    )

if __name__ == "__main__":
    main()
