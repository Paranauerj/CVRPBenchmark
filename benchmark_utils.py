import time

def execute_and_measure(algorithm_func, instance_data, **solver_kwargs):
    """
    Executes a given algorithm function and measures its CPU process time and objective value.

    Any solver-specific parameters (e.g., time_limit_seconds, time_limit_nanos,
    solution_limit) are accepted as keyword arguments and forwarded to the
    algorithm function.
    """
    start_cpu_time = time.process_time()

    # Execute the algorithm and forward any solver configuration kwargs
    objective_value = algorithm_func(instance_data, **solver_kwargs)

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time

    return {
        "cpu_time": cpu_time,
        "objective_value": objective_value,
    }