import time
from components.models import ExecutionResult


def execute_and_measure(algorithm_func, instance_data, **kwargs):
    """
    Executes algorithm and measures duration.
    Uses perf_counter for accurate wall-clock timing in multi-threaded environments.
    """
    start_time = time.perf_counter()

    # Execute
    result = algorithm_func(instance_data, **kwargs)

    end_time = time.perf_counter()
    duration = end_time - start_time

    objective_value = None
    accepted_neighbors = 0
    routes = None
    history = []
    time_to_target = None

    # Handle tuple return (cost, accepted_neighbors, routes, history, time_to_target)
    if isinstance(result, tuple):
        if len(result) >= 1: objective_value = result[0]
        if len(result) >= 2: accepted_neighbors = result[1]
        if len(result) >= 3: routes = result[2]
        if len(result) >= 4: history = result[3]
        if len(result) >= 5: time_to_target = result[4]
    else:
        objective_value = result

    if objective_value is not None:
        print(f"    > Found: Cost={objective_value:.2f} (Time={duration:.4f}s)")
    else:
        print(f"    > No Solution Found (Time={duration:.4f}s)")

    return ExecutionResult(
        cpu_time=duration, # We keep the field name 'cpu_time' in the model for compatibility
        objective_value=objective_value,
        accepted_neighbors=accepted_neighbors,
        routes=routes,
        history=history,
        time_to_target=time_to_target
    )