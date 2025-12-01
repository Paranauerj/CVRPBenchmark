from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from baseline_solver import _create_routing_model
import time

class SmartLimitCallback:
    """
    A helper class to track the best solution found so far.
    Used by the CustomLimit callback to decide when to stop.
    """
    def __init__(self, routing, no_improvement_limit=None, target_cost=None, no_improvement_iterations_limit=None):
        self.routing = routing
        self.solver = routing.solver() # Access the underlying CP solver
        
        self.no_improvement_limit = no_improvement_limit # Time based
        self.target_cost = target_cost
        self.no_improvement_iterations_limit = no_improvement_iterations_limit # Iteration based
        
        self.best_objective = float('inf')
        self.last_improvement_time = time.time()
        self.last_improvement_neighbors = 0 # Tracks accepted neighbors (iterations)
        self.start_time = time.time()

    def on_solution_callback(self):
        """
        Called by the RoutingModel every time a solution is found.
        We use this to update our 'best found so far' stats.
        """
        try:
            # Access the cost variable directly from the model
            current_cost = self.routing.CostVar().Value()
        except:
            return 

        # Check if this solution is an improvement
        if current_cost < self.best_objective:
            self.best_objective = current_cost
            self.last_improvement_time = time.time()
            # Reset the iteration counter baseline to the current count
            self.last_improvement_neighbors = self.solver.AcceptedNeighbors()
            
    def check_limit_callback(self):
        """
        Called periodically by the solver to check if we should stop.
        Returns True to stop.
        """
        # 1. Check Target Cost
        if self.target_cost is not None:
            if self.best_objective <= self.target_cost:
                return True

        # 2. Check No Improvement (Time)
        if self.no_improvement_limit is not None:
            if self.best_objective != float('inf'):
                if time.time() - self.last_improvement_time > self.no_improvement_limit:
                    return True
        
        # 3. Check No Improvement (Iterations / Accepted Neighbors)
        if self.no_improvement_iterations_limit is not None:
             if self.best_objective != float('inf'):
                 current_neighbors = self.solver.AcceptedNeighbors()             
                 # Stop if we've accepted N neighbors since the last improvement
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
               no_improvement_iterations_limit=None, # --- NEW ARGUMENT ---
               random_seed=None):
    """
    A generic solver function that can run ANY combination of strategies.
    """
    manager, routing = _create_routing_model(data)
    
    # --- APPLY RANDOM SEED ---
    if random_seed is not None:
        routing.solver().ReSeed(random_seed)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # --- 1. Apply Strategies ---
    search_parameters.first_solution_strategy = first_solution_strategy
    search_parameters.local_search_metaheuristic = local_search_metaheuristic
    
    # --- 2. Apply Limits ---
    if time_limit_seconds is not None:
        search_parameters.time_limit.seconds = int(time_limit_seconds)
    if solution_limit is not None:
        search_parameters.solution_limit = int(solution_limit)
    if lns_time_limit_seconds is not None:
        search_parameters.lns_time_limit.seconds = int(lns_time_limit_seconds)

    # --- CRITICAL FIX: Close model to initialize variables like CostVar ---
    routing.CloseModelWithParameters(search_parameters)

    # --- 3. Custom Monitor Setup (Smart Limits) ---
    if target_cost is not None or no_improvement_limit is not None or no_improvement_iterations_limit is not None:
        # Create our helper object
        limit_handler = SmartLimitCallback(
            routing, 
            no_improvement_limit, 
            target_cost,
            no_improvement_iterations_limit
        )
        
        # 1. Register the "At Solution" callback to update stats
        routing.AddAtSolutionCallback(limit_handler.on_solution_callback)
        
        # 2. Register the "Custom Limit" to stop the search
        solver = routing.solver()
        custom_limit = solver.CustomLimit(limit_handler.check_limit_callback)
        routing.AddSearchMonitor(custom_limit)

    # --- 4. Solve ---
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Return tuple: (Cost, Memory Usage Bytes)
        return solution.ObjectiveValue(), routing.solver().MemoryUsage()
    else:
        return None, None