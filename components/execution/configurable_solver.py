from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

def _create_routing_model(data):
    """Creates the common routing model and manager."""
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')
        
    return manager, routing

class SmartLimitCallback:
    def __init__(self, routing, no_improvement_limit=None, target_cost=None, no_improvement_iterations_limit=None):
        self.routing = routing
        self.solver = routing.solver()
        self.no_improvement_limit = no_improvement_limit
        self.target_cost = target_cost
        self.no_improvement_iterations_limit = no_improvement_iterations_limit
        
        self.best_objective = float('inf')
        self.last_improvement_time = time.time()
        self.last_improvement_neighbors = 0
        self.start_time = time.time()
        
        # --- NEW: History Tracking ---
        # List of tuples: (time_elapsed, iterations, cost)
        self.history = []

    def on_solution_callback(self):
        try:
            current_cost = self.routing.CostVar().Value()
        except:
            return 
        
        # Record every solution found? Or just improvements?
        # Usually recording every solution gives a better "convergence" curve,
        # but recording only improvements is cleaner for "best so far".
        # Let's record improvements to show the "step" function.
        
        current_time = time.time() - self.start_time
        current_iters = self.solver.AcceptedNeighbors()

        if current_cost < self.best_objective:
            self.best_objective = current_cost
            self.last_improvement_time = time.time()
            self.last_improvement_neighbors = current_iters
            # Record Improvement
            self.history.append((current_time, current_iters, current_cost))
            
    def check_limit_callback(self):
        if self.target_cost is not None and self.best_objective <= self.target_cost:
            return True
            
        if self.best_objective != float('inf'):
            if self.no_improvement_limit is not None:
                if time.time() - self.last_improvement_time > self.no_improvement_limit:
                    return True
            
            if self.no_improvement_iterations_limit is not None:
                 current_neighbors = self.solver.AcceptedNeighbors()
                 if current_neighbors - self.last_improvement_neighbors > self.no_improvement_iterations_limit:
                     return True
        return False

def solve_cvrp(data, 
               first_solution_strategy, 
               local_search_metaheuristic, 
               time_limit_seconds=None, 
               solution_limit=None, 
               lns_time_limit_seconds=None, 
               target_cost=None, 
               no_improvement_limit=None,
               no_improvement_iterations_limit=None,
               random_seed=None):
    
    manager, routing = _create_routing_model(data)
    
    if random_seed is not None:
        routing.solver().ReSeed(random_seed)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = first_solution_strategy
    search_parameters.local_search_metaheuristic = local_search_metaheuristic
    
    if time_limit_seconds is not None: search_parameters.time_limit.seconds = int(time_limit_seconds)
    if solution_limit is not None: search_parameters.solution_limit = int(solution_limit)
    if lns_time_limit_seconds is not None: search_parameters.lns_time_limit.seconds = int(lns_time_limit_seconds)

    routing.CloseModelWithParameters(search_parameters)

    # Always create the callback to track history, even if no custom limits are set
    limit_handler = SmartLimitCallback(routing, no_improvement_limit, target_cost, no_improvement_iterations_limit)
    routing.AddAtSolutionCallback(limit_handler.on_solution_callback)
    
    # Only add custom limit if needed
    if target_cost is not None or no_improvement_limit is not None or no_improvement_iterations_limit is not None:
        solver = routing.solver()
        custom_limit = solver.CustomLimit(limit_handler.check_limit_callback)
        routing.AddSearchMonitor(custom_limit)

    solution = routing.SolveWithParameters(search_parameters)
    
    if solution is not None:
        # Extract Routes
        routes = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != data['depot']:
                    route.append(node_index) 
                index = solution.Value(routing.NextVar(index))
            if route:
                routes.append(route)
                
        # Return tuple: (Cost, Iterations, Routes, History)
        return solution.ObjectiveValue(), routing.solver().AcceptedNeighbors(), routes, limit_handler.history
    else:
        return None, None, None, None