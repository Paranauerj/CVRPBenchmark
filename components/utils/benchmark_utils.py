import time
from components.models import ExecutionResult


def execute_and_measure(algorithm_func, instance_data, **kwargs):
    """
    Executes algorithm and measures duration using CPU time.
    """
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    # Execute
    result = algorithm_func(instance_data, **kwargs)

    end_cpu = time.process_time()
    end_wall = time.perf_counter()

    cpu_duration = end_cpu - start_cpu
    wall_duration = end_wall - start_wall

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
        print(f"    > Found: Cost={objective_value:.2f} (CPU Time={cpu_duration:.4f}s, Wall Time={wall_duration:.4f}s)")
    else:
        print(f"    > No Solution Found (CPU Time={cpu_duration:.4f}s, Wall Time={wall_duration:.4f}s)")

    return ExecutionResult(
        cpu_time=cpu_duration, 
        objective_value=objective_value,
        accepted_neighbors=accepted_neighbors,
        routes=routes,
        history=history,
        time_to_target=time_to_target
    )