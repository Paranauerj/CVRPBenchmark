import time

def execute_and_measure(algorithm_func, instance_data):
    """
    Executes a given algorithm function (with its own manual settings)
    and measures its CPU process time and objective value.
    """
    start_cpu_time = time.process_time()

    # --- Execute the algorithm ---
    # We no longer pass a time limit; the function is called directly.
    # The function is expected to have its own internal parameters
    # (e.g., a time_limit.seconds = 5 set inside gls_solver.py)
    objective_value = algorithm_func(instance_data)
    # -----------------------------

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time

    return {
        "cpu_time": cpu_time,
        "objective_value": objective_value,
    }