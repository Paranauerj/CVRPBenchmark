import time
from components.models import ExecutionResult


def execute_and_measure(algorithm_func, instance_data, **kwargs):
    """
    Executes algorithm.
    Expected return from solver: (cost, accepted_neighbors, routes, history).
    Returns ExecutionResult with 'cpu_time', 'objective_value', 'accepted_neighbors', 'routes', 'history'.
    """
    start_cpu_time = time.process_time()

    # Execute
    result = algorithm_func(instance_data, **kwargs)

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time
    
    objective_value = None
    accepted_neighbors = 0
    routes = None
    history = []

    # Handle tuple return (cost, accepted_neighbors, routes, history)
    if isinstance(result, tuple):
        if len(result) >= 1: objective_value = result[0]
        if len(result) >= 2: accepted_neighbors = result[1]
        if len(result) >= 3: routes = result[2]
        if len(result) >= 4: history = result[3] # --- NEW ---
    else:
        objective_value = result

    if objective_value is not None:
        print(f"    > Found: Cost={objective_value:.2f} (Time={cpu_time:.4f}s)")
    else:
        print(f"    > No Solution Found (Time={cpu_time:.4f}s)")

    return ExecutionResult(
        cpu_time=cpu_time,
        objective_value=objective_value,
        accepted_neighbors=accepted_neighbors,
        routes=routes,
        history=history
    )