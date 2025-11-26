from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from baseline_solver import _create_routing_model
import time

class SmartLimitCallback:
    """
    A helper class to track the best solution found so far.
    Used by the CustomLimit callback to decide when to stop.
    """
    def __init__(self, routing, no_improvement_limit=None, target_cost=None):
        self.routing = routing
        self.no_improvement_limit = no_improvement_limit
        self.target_cost = target_cost
        
        self.best_objective = float('inf')
        self.last_improvement_time = time.time()
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

        if current_cost < self.best_objective:
            self.best_objective = current_cost
            self.last_improvement_time = time.time()
            
    def check_limit_callback(self):
        """
        Called periodically by the solver to check if we should stop.
        Returns True to stop.
        """
        # 1. Check Target Cost
        if self.target_cost is not None:
            if self.best_objective <= self.target_cost:
                return True

        # 2. Check No Improvement
        if self.no_improvement_limit is not None:
            if self.best_objective != float('inf'):
                # Check if time since last improvement exceeds limit
                if time.time() - self.last_improvement_time > self.no_improvement_limit:
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
               random_seed=None):
    """
    A generic solver function that can run ANY combination of strategies.
    """
    manager, routing = _create_routing_model(data)
    
    # --- APPLY RANDOM SEED ---
    if random_seed is not None:
        # Note: Using routing.solver() just for ReSeed seems to be generally accepted
        # but if it fails similarly, we might skip it. Usually this works though.
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

    # --- CRITICAL FIX: Close model to initialize variables ---
    routing.CloseModelWithParameters(search_parameters)

    # --- 3. Custom Monitor Setup (Smart Limits) ---
    if target_cost is not None or no_improvement_limit is not None:
        # Create our helper object
        limit_handler = SmartLimitCallback(routing, no_improvement_limit, target_cost)
        
        # 1. Register the "At Solution" callback to update stats
        routing.AddAtSolutionCallback(limit_handler.on_solution_callback)
        
        # 2. Register the "Custom Limit" to stop the search
        # FIX: Use routing.AddSearchMonitor instead of solver.AddSearchMonitor
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