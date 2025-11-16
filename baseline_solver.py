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

def solve_baseline(data, **kwargs):
    """
    Solves the CVRP using the Clarke-Wright (SAVINGS) heuristic.
    
    **kwargs is included to absorb any parameters from the benchmarker,
    but they are NOT used, ensuring a consistent baseline.
    
    This solver is hard-coded to return the *first solution*
    (the pure heuristic) with no local search and no time limit.
    """
    manager, routing = _create_routing_model(data)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # --- Baseline Configuration ---
    
    # 1. Use SAVINGS (Clarke & Wright) as the construction heuristic
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
        
    # 2. Disable all local search to get the pure heuristic result
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.UNSET)
        
    # 3. Force the solver to stop after *ten* solutions are found
    search_parameters.time_limit.seconds = 20
    
    # NOTE: We explicitly DO NOT set a time limit.
    
    # ---
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # --- UPDATED: Extract and return routes ---
        routes_list = []
        for vehicle_id in range(routing.vehicles()):
            route_nodes = []
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != data['depot']: # Do not include the depot in the route string
                    route_nodes.append(str(node_index + 1)) # +1 to match .sol 1-indexing
                index = solution.Value(routing.NextVar(index))
            
            if route_nodes: # If the route wasn't empty
                routes_list.append("Route: " + " ".join(route_nodes))
        
        return solution.ObjectiveValue(), routes_list # Return a tuple
        # ---
    else:
        return None, None