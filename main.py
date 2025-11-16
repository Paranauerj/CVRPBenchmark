# main_benchmark.py
import time
import statistics
import data_model 
import baseline_solver
import gls_solver
import sa_solver
import ts_solver

def execute_and_measure(algorithm_func, instance_data):
    """
    Executes a given algorithm function (with its own manual settings)
    and measures its CPU process time and objective value.
    """
    start_cpu_time = time.process_time()

    # --- Execute the algorithm ---
    # We no longer pass a time limit. The function is called directly.
    objective_value = algorithm_func(instance_data)
    # -----------------------------

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time

    return {
        "cpu_time": cpu_time,
        "objective_value": objective_value,
    }

def run_agnostic_benchmark():
    """Main function to orchestrate the benchmarking experiment."""
    print("Starting Hardware-Agnostic CVRP Benchmark...\n")

    # --- A. Experiment Configuration ---

    # --- 1. PASSMARK NORMALIZATION SCORES ---
    PASSMARK_SINGLE_THREAD_BASE = 2000
    
    # Your Machine: (Intel i5-9400F) - Single Thread Rating
    PASSMARK_SINGLE_THREAD_LOCAL = 2476.0 
    
    print(f"--- Benchmark Configuration ---")
    print(f"  Reference Score (s_base): {PASSMARK_SINGLE_THREAD_BASE}")
    print(f"  Local Score (s_local):    {PASSMARK_SINGLE_THREAD_LOCAL}\n")

    instance_data = data_model.create_data_model()
    instance_name = "P-n16-k8"
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
        
        for i in range(NUM_REPETITIONS):
            # The call is now simplified
            measurement = execute_and_measure(
                algo_func, 
                instance_data
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
    
    col_algo = "Algorithm"
    col_cost = "Cost"
    col_cpu_local = "Local CPU Time (s)"
    col_cpu_norm = "Normalized Time (s)"
    col_relative = "Time vs. Baseline"

    print(f"{col_algo:<22} | {col_cost:<10} | {col_cpu_local:<18} | {col_cpu_norm:<20} | {col_relative:<20}")
    print(f"{'-'*22} | {'-'*10} | {'-'*18} | {'-'*20} | {'-'*20}")

    for algo_name, data in results.items():
        print(f"{algo_name:<22} | "
              f"{data['avg_objective']:<10.2f} | "
              f"{data['avg_cpu_time']:<18.6f} | "
              f"{data['normalized_time']:<20.6f} | "
              f"{data['relative_time']:<20.4f}")

# --- Script Entry Point ---
if __name__ == "__main__":
    run_agnostic_benchmark()