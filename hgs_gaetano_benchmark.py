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
        max_instances=MAX_INSTANCES
    )
    
    runner.run(
        experiments=[get_experiment()],
        process_instance_fn=process_instance
    )

if __name__ == "__main__":
    main()
