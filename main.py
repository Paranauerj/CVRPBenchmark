import time
import statistics
import baseline_solver
import gls_solver
import instance_data_parser
import sa_solver
import ts_solver
import solution_parser # --- NEW IMPORT ---
from benchmark_utils import execute_and_measure
import sys 

def run_agnostic_benchmark(instance_filepath, solution_filepath):
    """
    Main function to orchestrate the benchmarking experiment.
    Takes a filepath to a .vrp instance and a .sol solution.
    """
    print("Starting Hardware-Agnostic CVRP Benchmark...\n")

    # --- A. Experiment Configuration ---
    PASSMARK_SINGLE_THREAD_BASE = 2000.0 
    PASSMARK_SINGLE_THREAD_LOCAL = 2476.0 
    
    print(f"--- Benchmark Configuration ---")
    print(f"  Reference Score (s_base): {PASSMARK_SINGLE_THREAD_BASE}")
    print(f"  Local Score (s_local):    {PASSMARK_SINGLE_THREAD_LOCAL}\n")

    # --- Load problem instance from file ---
    try:
        with open(instance_filepath, 'r') as f:
            instance_data = instance_data_parser.load_vrp_instance(f)
        instance_name = instance_filepath
    except FileNotFoundError:
        print(f"Error: Instance file not found at '{instance_filepath}'")
        return
    except Exception as e:
        print(f"Error parsing instance file: {e}")
        return
        
    # --- NEW: Load BKS from solution file ---
    try:
        with open(solution_filepath, 'r') as f:
            bks_cost = solution_parser.parse_solution_file(f)
        print(f"  Best Known Solution (BKS) Cost: {bks_cost}")
    except Exception as e:
        print(f"Warning: Could not parse solution file '{solution_filepath}'. BKS Gap will not be calculated.")
        bks_cost = None
    
    NUM_REPETITIONS = 5 

    algorithms_to_test = {
        "Baseline (C&W)": baseline_solver.solve_baseline,
        "Guided Local Search": gls_solver.solve_gls,
        "Simulated Annealing": sa_solver.solve_sa,
        "Tabu Search": ts_solver.solve_ts,
    }
    
    results = {}  
    print(f"--- Running Benchmark ---")
    print(f"  Instance: {instance_name}")
    print(f"  Repetitions: {NUM_REPETITIONS}\n")

    # --- B. Execution and Data Collection ---
    for algo_name, algo_func in algorithms_to_test.items():
        cpu_times, objectives = [], []
        print(f"  Testing Algorithm: {algo_name}...")
        
        solver_kwargs = {} 
        
        for i in range(NUM_REPETITIONS):
            print(f"    Running Repetition {i+1}/{NUM_REPETITIONS}...")
            measurement = execute_and_measure(
                algo_func, 
                instance_data,
                **solver_kwargs
            )
            cpu_times.append(measurement["cpu_time"])
            objectives.append(measurement["objective_value"])

        results[algo_name] = {
            "avg_cpu_time": statistics.mean(cpu_times),
            "avg_objective": statistics.mean(objectives),
        }

    print("\n--- Benchmark Execution Complete ---")

    # --- C. Metric Calculation & Reporting ---
    baseline_time = results["Baseline (C&W)"]["avg_cpu_time"]
    
    for algo_name, data in results.items():
        avg_time = data["avg_cpu_time"]
        data["relative_time"] = avg_time / baseline_time
        data["normalized_time"] = avg_time * (
            PASSMARK_SINGLE_THREAD_LOCAL / PASSMARK_SINGLE_THREAD_BASE)
            
        # --- NEW: BKS Gap Calculation ---
        if bks_cost is not None:
            data["bks_gap"] = ((data["avg_objective"] - bks_cost) / bks_cost) * 100.0
        else:
            data["bks_gap"] = None # Set to None if BKS cost wasn't loaded

    # --- Final Results Table ---
    print("\n--- Final Results Summary ---")
    
    col_algo = "Algorithm"
    col_cost = "Cost"
    col_bks_gap = "BKS Gap (%)" # --- NEW ---
    col_cpu_local = "CPU Time (s)" 
    col_cpu_norm = "Normalized Time (s)"
    col_relative = "Time vs. Baseline"

    # Print header
    print(f"{col_algo:<22} | {col_cost:<10} | {col_bks_gap:<12} | {col_cpu_local:<15} | {col_cpu_norm:<20} | {col_relative:<20}")
    print(f"{'-'*22} | {'-'*10} | {'-'*12} | {'-'*15} | {'-'*20} | {'-'*20}")

    # Print data rows
    for algo_name, data in results.items():
        # Format BKS gap or show N/A
        gap_str = f"{data['bks_gap']:<12.4f}" if data["bks_gap"] is not None else f"{'N/A':<12}"
        
        print(f"{algo_name:<22} | "
              f"{data['avg_objective']:<10.2f} | "
              f"{gap_str} | "
              f"{data['avg_cpu_time']:<15.6f} | "
              f"{data['normalized_time']:<20.6f} | "
              f"{data['relative_time']:<20.4f}")

    print("\n'Normalized Time (s)' = How long the algorithm would have run on the i9-13900KS reference machine.")

# --- Script Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) > 2:
        vrp_filepath = sys.argv[1]
        sol_filepath = sys.argv[2]
    else:
        # Fallback to default filenames if none are provided
        vrp_filepath = "P-n16-k8.vrp" 
        sol_filepath = "P-n16-k8.sol"
        print(f"No instance files provided. Defaulting to '{vrp_filepath}' and '{sol_filepath}'")
        print(f"Usage: python {sys.argv[0]} <path_to.vrp> <path_to.sol>\n")
        
    run_agnostic_benchmark(vrp_filepath, sol_filepath)