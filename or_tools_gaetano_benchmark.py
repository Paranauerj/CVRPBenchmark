import os
import copy
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
RESULTS_DIR = "gaetano_ortools_results"
MAX_INSTANCES = 1000   # 1,000 instances sampled randomly
RANDOM_SAMPLE = True   # Sample 1,000 random instances
NUM_PARALLEL = 8       # Number of concurrent workers
CHUNK_SIZE = 10        # Number of instances per chunk
TARGET_GAP_PCT = 5.0   # Record time to reach 5% gap

# Time intervals for progress monitoring: 1%, 2%, 5%, 10%, 15%, 20%, 30%, 50%, 75%, 100%
OR_TOOLS_CHECKPOINTS_PCT = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]

# Reverse maps for labeling
FS_NAME_MAP = {v: k for k, v in FIRST_SOLUTIONS.items()}
MH_NAME_MAP = {v: k for k, v in METAHEURISTICS.items()}

# Selected strategies
SELECTED_FS = [
    RE.FirstSolutionStrategy.SAVINGS,
    RE.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    RE.FirstSolutionStrategy.CHRISTOFIDES,
]

SELECTED_MH = [
    RE.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    RE.LocalSearchMetaheuristic.TABU_SEARCH,
    RE.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
]

def get_experiments() -> list[ExperimentConfig]:
    """Generates a template list of OR-Tools experiments based on selected strategies."""
    experiments = []
    for fs in SELECTED_FS:
        for mh in SELECTED_MH:
            fs_label = FS_NAME_MAP.get(fs, str(fs))
            mh_label = MH_NAME_MAP.get(mh, str(mh))
            
            kwargs = {
                "first_solution_strategy": fs, 
                "local_search_metaheuristic": mh,
                "target_gap_percent": TARGET_GAP_PCT,
                "continue_after_target": True,
            }
            
            experiments.append(ExperimentConfig(
                name=f"OR-Tools: {mh_label} [{fs_label}]",
                fs_label=fs_label, 
                mh_label=mh_label,
                func=None, 
                kwargs=kwargs, 
                reps=3 # 3 independent runs
            ))
    return experiments

def process_instance(vrp_path: str, experiments: list[ExperimentConfig]):
    """
    Standard processing for OR-Tools instances.
    Implements dynamic time limit T = N * 0.5.
    """
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    try:
        from components.execution import configurable_solver
        from components.execution.benchmark_common import TIME_CHECKPOINTS
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        # Dynamic Time Limit: T = N * 0.5
        N = instance_meta.customers
        T = N * 0.5
        
        # Define Checkpoints
        checkpoint_configs = []
        probe_times = set()
        for p in OR_TOOLS_CHECKPOINTS_PCT:
            t_sec = round(T * p, 3)
            probe_times.add(t_sec)
            checkpoint_configs.append({"time": t_sec, "label": f"{int(p*100)}", "is_pct": True})
            
        for t_fixed in TIME_CHECKPOINTS:
            probe_times.add(float(t_fixed))
            checkpoint_configs.append({"time": float(t_fixed), "label": t_fixed, "is_pct": False})
        
        time_checkpoints = sorted(list(probe_times))
        
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
            # Localize experiment with dynamic time limit
            inst_exp = copy.copy(exp)
            inst_exp.func = configurable_solver.solve_cvrp
            inst_exp.kwargs = exp.kwargs.copy()
            inst_exp.kwargs["time_limit_seconds"] = T
            
            # Execute with centralized logic
            result_row_dict, _, _, _ = run_experiment_with_vehicle_retry(
                inst_exp, inst_data, bks_val, instance_meta, 
                max_retries=10, 
                time_checkpoints=time_checkpoints,
                checkpoint_configs=checkpoint_configs,
                use_permutations=True,
                engine="ortools"
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
        name="OR-Tools Gaetano Benchmark (1,000 Random Instances)",
        instances_dir=INSTANCES_DIR,
        results_dir=RESULTS_DIR,
        max_instances=MAX_INSTANCES,
        num_parallel=NUM_PARALLEL,
        chunk_size=CHUNK_SIZE,
        random_sample=RANDOM_SAMPLE
    )
    
    runner.run(
        experiments=get_experiments(),
        process_instance_fn=process_instance,
        filter_cols=[]
    )

if __name__ == "__main__":
    main()
