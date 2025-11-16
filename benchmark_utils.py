import time

def execute_and_measure(algorithm_func, instance_data, **kwargs):
    """
    Executes a given algorithm function, passing through optional
    keyword arguments (like 'time_limit_seconds'), and measures
    its CPU process time and objective value.
    
    Returns a dict containing cpu_time and objective_value.
    """
    start_cpu_time = time.process_time()

    # --- Execute the algorithm ---
    # --- UPDATED: Solvers now only return one value: objective_value ---
    objective_value = algorithm_func(instance_data, **kwargs)
    # -----------------------------

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time

    return {
        "cpu_time": cpu_time,
        "objective_value": objective_value,
    }