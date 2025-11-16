import time
import statistics
import data_model 
import baseline_solver
import gls_solver
import sa_solver
import ts_solver
from benchmark_utils import execute_and_measure # Import shared function

def run_agnostic_benchmark():
    """Main function to orchestrate the benchmarking experiment."""
    print("Starting Hardware-Agnostic CVRP Benchmark...\n")

    # --- A. Experiment Configuration ---

    # --- 1. PASSMARK NORMALIZATION SCORES (SINGLE-THREAD) ---
    PASSMARK_SINGLE_THREAD_BASE = 2000.0 
    
    # Your Machine: (Intel i5-9400F) - Single Thread Rating
    PASSMARK_SINGLE_THREAD_LOCAL = 2476.0 
    
    print(f"--- Benchmark Configuration ---")
    print(f"  Reference Score (s_base): {PASSMARK_SINGLE_THREAD_BASE}")
    print(f"  Local Score (s_local):    {PASSMARK_SINGLE_THREAD_LOCAL}\n")

    # Load problem instance
    instance_data = data_model.create_data_model()
    instance_name = "P-n16-k8"
    
    # Number of repetitions for stable averages
    NUM_REPETITIONS = 5 

    # Map algorithm names to their functions
    algorithms_to_test = {
        "Baseline (C&W)": baseline_solver.solve_baseline,
        "Guided Local Search": gls_solver.solve_gls,
        "Simulated Annealing": sa_solver.solve_sa,
        "Tabu Search": ts_solver.solve_ts,
    }
    
    results = {}  # To store results
    print(f"--- Running Benchmark ---")
    print(f"  Instance: {instance_name}")
    print(f"  Repetitions: {NUM_REPETITIONS}\n")

    # --- B. Execution and Data Collection ---

    for algo_name, algo_func in algorithms_to_test.items():
        cpu_times, objectives = [], []
        print(f"  Testing Algorithm: {algo_name}...")
        
        for i in range(NUM_REPETITIONS):
            measurement = execute_and_measure(
                algo_func, 
                instance_data
            )
            cpu_times.append(measurement["cpu_time"])
            objectives.append(measurement["objective_value"])

        # Store the averages
        results[algo_name] = {
            "avg_cpu_time": statistics.mean(cpu_times),
            "avg_objective": statistics.mean(objectives),
        }

    print("\n--- Benchmark Execution Complete ---")

    # --- C. Metric Calculation & Reporting ---
    
    # Get the baseline time for relative calculations
    baseline_time = results["Baseline (C&W)"]["avg_cpu_time"]
    
    # Calculate final metrics for all algorithms
    for algo_name, data in results.items():
        avg_time = data["avg_cpu_time"]
        
        # 1. Time Relative to the Baseline
        data["relative_time"] = avg_time / baseline_time

        # 2. Normalized Runtime (Eq 2)
        # t_norm = t_local * (s_local / s_base)
        data["normalized_time"] = avg_time * (
            PASSMARK_SINGLE_THREAD_LOCAL / PASSMARK_SINGLE_THREAD_BASE)

    # --- Final Results Table ---
    print("\n--- Final Results Summary ---")
    
    # Define column headers
    col_algo = "Algorithm"
    col_cost = "Cost"
    col_cpu_local = "Local CPU Time (s)"
    col_cpu_norm = "Normalized Time (s)"
    col_relative = "Time vs. Baseline"

    # Print header
    print(f"{col_algo:<22} | {col_cost:<10} | {col_cpu_local:<18} | {col_cpu_norm:<20} | {col_relative:<20}")
    print(f"{'-'*22} | {'-'*10} | {'-'*18} | {'-'*20} | {'-'*20}")

    # Print data rows
    for algo_name, data in results.items():
        print(f"{algo_name:<22} | "
              f"{data['avg_objective']:<10.2f} | "
              f"{data['avg_cpu_time']:<18.6f} | "
              f"{data['normalized_time']:<20.6f} | "
              f"{data['relative_time']:<20.4f}")

# --- Script Entry Point ---
if __name__ == "__main__":
    run_agnostic_benchmark()