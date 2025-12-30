import time

def execute_and_measure(algorithm_func, instance_data, **kwargs):
    """
    Executes algorithm.
    Expected return from solver: (cost, iterations, routes, history).
    Returns dict with 'cpu_time', 'objective_value', 'iterations', 'routes', 'history'.
    """
    start_cpu_time = time.process_time()

    # Execute
    result = algorithm_func(instance_data, **kwargs)

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time
    
    objective_value = None
    iterations = 0
    routes = None
    history = []

    # Handle tuple return (cost, iterations, routes, history)
    if isinstance(result, tuple):
        if len(result) >= 1: objective_value = result[0]
        if len(result) >= 2: iterations = result[1]
        if len(result) >= 3: routes = result[2]
        if len(result) >= 4: history = result[3] # --- NEW ---
    else:
        objective_value = result

    if objective_value is not None:
        print(f"    > Found: Cost={objective_value:.2f} (Time={cpu_time:.4f}s)")
    else:
        print(f"    > ❌ No Solution Found (Time={cpu_time:.4f}s)")

    return {
        "cpu_time": cpu_time,
        "objective_value": objective_value,
        "iterations": iterations,
        "routes": routes,
        "history": history # --- NEW ---
    }