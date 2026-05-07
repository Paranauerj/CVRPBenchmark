import os
from components.execution.benchmark_runner import BenchmarkRunner
from components.execution.hygese_solver import solve_hgs
from components.execution.benchmark_common import extract_instance_metadata, run_experiment_reps
from components.models import ExperimentConfig
from components.utils import instance_data_parser, solution_parser
from components import constants as C

# Configuration
INSTANCES_DIR = "instances/gaetano"
RESULTS_DIR = "hgs_gaetano_results"
MAX_INSTANCES = None 
SAVE_SOLUTIONS = True  # Flag to generate .sol files
NUM_PARALLEL = 4       # Number of concurrent workers
CHUNK_SIZE = 10        # Number of instances per chunk


BENCHMARK_PARAMS = {
    "time_limit_seconds": 10,
    "no_improvement_limit_iterations": 20000,
    "reps": 1,
}

def get_experiment():
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

def process_instance(vrp_path, experiments):
    """Specialized processing for HGS (no features, no gaps)."""
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    exp = experiments[0] # HGS only has one
    
    try:
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        # Run HGS
        data = run_experiment_reps(exp, inst_data, exp.reps, bks_cost=None, time_checkpoints=[])
        
        if not data.costs:
            return []

        avg_cost = sum(data.costs) / len(data.costs)
        best_cost = min(data.costs)
        avg_time = sum(data.times) / len(data.times)

        # Save .sol file if requested and it's better than existing or no existing
        if SAVE_SOLUTIONS and data.best_routes:
            sol_path = vrp_path.replace(".vrp", ".sol")
            save_needed = True
            
            if os.path.exists(sol_path):
                try:
                    existing_cost = solution_parser.parse_solution_file(sol_path)
                    if existing_cost is not None and best_cost >= existing_cost:
                        save_needed = False
                except:
                    # If error parsing, assume we should overwrite
                    pass
            
            if save_needed:
                action = "Updating" if os.path.exists(sol_path) else "Saving new"
                print(f"  [SOL] {action} best solution for {inst_name}: {best_cost}")
                solution_parser.save_solution_file(sol_path, data.best_routes, best_cost)
        
        result = {
            C.COL_INSTANCE: inst_name,
            C.COL_SOLVER: "HGS-CVRP",
            C.COL_VEHICLES: instance_meta.vehicles,
            C.COL_REPETITIONS: exp.reps,
            C.COL_BEST_COST: best_cost,
            C.COL_AVG_COST: avg_cost,
            C.COL_AVG_CPU_TIME: avg_time,
        }
        return [result]
    except Exception as e:
        print(f"Error processing {inst_name}: {e}")
        return []

def main():
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
