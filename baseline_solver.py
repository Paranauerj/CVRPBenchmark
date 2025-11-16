# baseline_solver.py
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def _create_routing_model(data):
    """Creates the common routing model and manager."""
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance Callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Demand Callback
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle capacity
        True,  # start cumul to zero
        'Capacity')
        
    return manager, routing

def solve_baseline(data, time_limit_seconds=5, solution_limit=None):
    """
    Solves the CVRP using the Clarke-Wright (SAVINGS) heuristic.
    Local search is disabled to keep it a pure, deterministic baseline.
    """
    manager, routing = _create_routing_model(data)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # --- Baseline Configuration ---
    # Use SAVINGS (Clarke & Wright) as the construction heuristic
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
        
    # **CRITICAL**: Disable local search for a pure baseline
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.UNSET)
    # ---

    # Only set time/solution parameters if caller provided them. If not provided,
    # leave the routing search parameters untouched so OR-Tools defaults apply.
    if time_limit_seconds is not None:
        search_parameters.time_limit.seconds = int(time_limit_seconds)

    if solution_limit is not None:
        search_parameters.solution_limit = int(solution_limit)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        return solution.ObjectiveValue()
    else:
        return None