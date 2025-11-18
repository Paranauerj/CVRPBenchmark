import time

def execute_and_measure(algorithm_func, instance_data, **kwargs):
    """
    Executes a given algorithm function, passing through optional
    keyword arguments (like 'time_limit_seconds'), and measures
    its execution metrics.
    
    Solvers return (objective_value, memory_usage).
    
    Returns a dict containing cpu_time, objective_value, and memory_usage.
    """
    start_cpu_time = time.process_time()

    # Execute the algorithm
    objective_value, memory_usage = algorithm_func(instance_data, **kwargs)
    
    end_cpu_time = time.process_time()
    elapsed_cpu_time = end_cpu_time - start_cpu_time

    return {
        "cpu_time": elapsed_cpu_time,
        "objective_value": objective_value,
        "memory_usage": memory_usage,
    }