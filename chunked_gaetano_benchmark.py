import os
import glob
import json
import copy
import pandas as pd
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ortools.constraint_solver import routing_enums_pb2 as RE

# Import project components
from components.utils import instance_data_parser, solution_parser
from components.execution import configurable_solver
from components.execution.benchmark_common import (
    extract_instance_metadata, build_result_row, 
    run_experiment_reps, run_experiment_with_vehicle_retry
)
from components.models import ExperimentConfig
from components.ui.sidebar import FIRST_SOLUTIONS, METAHEURISTICS

# Reverse maps for readable reporting
FS_NAME_MAP = {v: k for k, v in FIRST_SOLUTIONS.items()}
MH_NAME_MAP = {v: k for k, v in METAHEURISTICS.items()}

# Configuration
INSTANCES_DIR = "instances/gaetano"
RESULTS_DIR = "gaetano_chunk_results"
CHUNK_SIZE = 10
FINAL_OUTPUT_DIR = "server_output"
NUM_PARALLEL = 2
MAX_INSTANCES = 2 # Set to an integer to limit the total number of instances (e.g., 50)

# Custom time checkpoints for convergence analysis
TIME_CHECKPOINTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60]

# Professional Benchmark Configuration
# Use OR-Tools constants directly here
SELECTED_FS = [
    RE.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    RE.FirstSolutionStrategy.SAVINGS,
    RE.FirstSolutionStrategy.CHRISTOFIDES
]

SELECTED_MH = [
    RE.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    RE.LocalSearchMetaheuristic.TABU_SEARCH,
    RE.LocalSearchMetaheuristic.SIMULATED_ANNEALING
]

BENCHMARK_PARAMS = {
    "time_limit_seconds": 6,
    "solution_limit": None,
    "lns_time_limit_seconds": None,
    "no_improvement_limit": None,
    "no_improvement_neighbors_limit": None,
    "reps": 1,
}

def get_experiments():
    """Generates a list of ExperimentConfig objects from constants."""
    experiments = []
    for fs in SELECTED_FS:
        for mh in SELECTED_MH:
            fs_label = FS_NAME_MAP.get(fs, str(fs))
            mh_label = MH_NAME_MAP.get(mh, str(mh))
            
            # Prepare kwargs for the solver
            kwargs = BENCHMARK_PARAMS.copy()
            reps = kwargs.pop("reps") # Remove from kwargs
            
            kwargs.update({
                "first_solution_strategy": fs,
                "local_search_metaheuristic": mh,
            })
            
            experiments.append(ExperimentConfig(
                name=f"{mh_label} [{fs_label}]",
                fs_label=fs_label,
                mh_label=mh_label,
                func=configurable_solver.solve_cvrp,
                kwargs=kwargs,
                reps=reps
            ))
    return experiments

def get_instance_files():
    """Returns a list of all .vrp files in the Gaetano directory."""
    files = sorted(glob.glob(os.path.join(INSTANCES_DIR, "*.vrp")))
    return files

def process_instance(vrp_path, experiments):
    """Processes a single instance and returns its results."""
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    print(f"  Processing instance: {inst_name}")
    
    try:
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        # Look for BKS if it exists
        sol_path = vrp_path.replace(".vrp", ".sol")
        bks_val = None
        if os.path.exists(sol_path):
            try:
                bks_val = solution_parser.parse_solution_file(sol_path)
            except:
                pass

        results = []
        for exp in experiments:
            try:
                # Use shared logic from benchmark_common
                result_row_dict, found, final_attempt = run_experiment_with_vehicle_retry(
                    exp, inst_data, bks_val, instance_meta, 
                    max_retries=5, 
                    log_fn=print,
                    time_checkpoints=TIME_CHECKPOINTS
                )
                
                if result_row_dict:
                    results.append(result_row_dict)
                    if found:
                        if final_attempt > 0:
                            print(f"    ✓ Solution found for {inst_name} | {exp.name} with +{final_attempt} vehicles.")
                    else:
                        print(f"    ✗ No solution found for {inst_name} | {exp.name} even with +5 vehicles.")
                
            except Exception as e:
                print(f"    Error in experiment {exp.name} for {inst_name}: {e}")
        
        return results
    except Exception as e:
        print(f"    Critical error loading {inst_name}: {e}")
        return []

def run_chunk(chunk_id, chunk_files, experiments):
    """Runs a chunk of instances and saves to a JSON file."""
    print(f"Starting Chunk {chunk_id} ({len(chunk_files)} instances)...")
    
    chunk_results = []
    
    if NUM_PARALLEL > 1:
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL) as executor:
            futures = [executor.submit(process_instance, f, experiments) for f in chunk_files]
            for future in as_completed(futures):
                chunk_results.extend(future.result())
    else:
        for f in chunk_files:
            chunk_results.extend(process_instance(f, experiments))
            
    # Save chunk to JSON
    chunk_file = os.path.join(RESULTS_DIR, f"chunk_{chunk_id:04d}.json")
    with open(chunk_file, 'w') as f:
        json.dump(chunk_results, f)
    
    print(f"Chunk {chunk_id} completed and saved to {chunk_file}")

def aggregate_results():
    """Combines all chunk files into a single Excel file."""
    print("Aggregating all chunk results...")
    all_results = []
    chunk_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "chunk_*.json")))
    
    for cf in chunk_files:
        with open(cf, 'r') as f:
            all_results.extend(json.load(f))
            
    if not all_results:
        print("No results found to aggregate.")
        return None

    df = pd.DataFrame(all_results)
    
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(FINAL_OUTPUT_DIR, f"gaetano_chunked_benchmark_{timestamp}.xlsx")
    
    df.to_excel(output_path, index=False)
    print(f"Final results saved to {output_path}")
    return output_path

def cleanup():
    """Deletes temporary chunk files."""
    print("Cleaning up temporary files...")
    chunk_files = glob.glob(os.path.join(RESULTS_DIR, "chunk_*.json"))
    for cf in chunk_files:
        os.remove(cf)
    try:
        os.rmdir(RESULTS_DIR)
        print(f"Removed directory: {RESULTS_DIR}")
    except:
        pass

def main():
    # 1. Setup
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instance_files = get_instance_files()
    if not instance_files:
        print(f"No instances found in {INSTANCES_DIR}")
        return

    # Apply instance limit if set
    if MAX_INSTANCES is not None:
        instance_files = instance_files[:MAX_INSTANCES]
        print(f"Limited benchmark to {len(instance_files)} instances (MAX_INSTANCES={MAX_INSTANCES})")

    print(f"Total instances to process: {len(instance_files)}. Chunk size: {CHUNK_SIZE}.")
    
    # 2. Prepare experiments directly from constants
    experiments = get_experiments()
    
    # 3. Chunking logic
    chunks = [instance_files[i:i + CHUNK_SIZE] for i in range(0, len(instance_files), CHUNK_SIZE)]
    
    # 4. Resumption logic
    existing_chunks = sorted(glob.glob(os.path.join(RESULTS_DIR, "chunk_*.json")))
    start_chunk_idx = 0
    if existing_chunks:
        latest_chunk_file = existing_chunks[-1]
        latest_chunk_name = os.path.basename(latest_chunk_file)
        latest_idx = int(latest_chunk_name.split('_')[1].split('.')[0])
        
        print(f"Found existing results up to chunk {latest_idx}.")
        start_chunk_idx = latest_idx
        print(f"Resuming from chunk {start_chunk_idx} (will overwrite {latest_chunk_file})")
    
    # 5. Execution
    try:
        for i in range(start_chunk_idx, len(chunks)):
            run_chunk(i, chunks[i], experiments)
            
        output_file = aggregate_results()
        
        if output_file:
            cleanup()
            print("Benchmark finished successfully!")
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. You can resume later.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")

if __name__ == "__main__":
    main()
