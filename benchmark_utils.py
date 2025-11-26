import time

def execute_and_measure(algorithm_func, instance_data, **kwargs):
    """
    Executes algorithm and handles tuple return (cost, memory).
    Returns dict with 'cpu_time', 'objective_value', 'memory_usage_mb'.
    """
    start_cpu_time = time.process_time()

    # Execute: solver returns (cost, memory_bytes)
    # Or just cost if it's the baseline solver
    result = algorithm_func(instance_data, **kwargs)

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time
    
    objective_value = None
    memory_usage_mb = None

    # Handle tuple return (cost, memory) vs single value (cost)
    if isinstance(result, tuple):
        objective_value = result[0]
        memory_bytes = result[1]
        if memory_bytes is not None:
            memory_usage_mb = memory_bytes / (1024 * 1024)
    else:
        objective_value = result
        memory_usage_mb = 0.0 # Baseline or fallback
    
    # --- NEW: Log the solution found to CLI ---
    if objective_value is not None:
        print(f"    > Solution Found: Cost={objective_value} (Time={cpu_time:.4f}s)")
    else:
        print(f"    > No Solution Found (Time={cpu_time:.4f}s)")
    # ------------------------------------------

    return {
        "cpu_time": cpu_time,
        "objective_value": objective_value,
        "memory_usage_mb": memory_usage_mb 
    }