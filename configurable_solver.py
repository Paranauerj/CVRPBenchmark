from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from baseline_solver import _create_routing_model
import time

class SmartSearchLimit(pywrapcp.SearchMonitor):
    """
    A custom search monitor that terminates the search based on:
    1. Reaching a target cost (gap limit).
    2. No improvement for N seconds.
    """
    def __init__(self, solver, objective_var, no_improvement_limit=None, target_cost=None):
        super().__init__(solver)
        self._objective_var = objective_var
        self._no_improvement_limit = no_improvement_limit
        self._target_cost = target_cost
        
        self._best_objective = float('inf')
        self._last_improvement_time = time.time()
        
    def AtSolution(self):
        """Called every time a valid solution is found."""
        try:
            # Access the value of the cost variable directly.
            # AtSolution fires when a solution is found, so the variable is bound.
            current_objective = self._objective_var.Value()
        except:
            return False
        
        # Check for improvement
        if current_objective < self._best_objective:
            self._best_objective = current_objective
            self._last_improvement_time = time.time()

        # Check Target Cost Stop
        if self._target_cost is not None and current_objective <= self._target_cost:
            self.solver().FinishCurrentSearch()
            return True
            
        return False

    def Check(self):
        """Called periodically by the solver."""
        # Check No Improvement Stop
        if self._no_improvement_limit is not None:
            if self._best_objective != float('inf'): 
                if time.time() - self._last_improvement_time > self._no_improvement_limit:
                    self.solver().FinishCurrentSearch()
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
               random_seed=None): # --- NEW ARGUMENT ---
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

    # --- CRITICAL FIX: Close model to initialize CostVar ---
    routing.CloseModelWithParameters(search_parameters)

    # --- 3. Custom Monitor Setup (Smart Limits) ---
    if target_cost is not None or no_improvement_limit is not None:
        solver = routing.solver()
        
        # Pass the COST VARIABLE (routing.CostVar()) to the monitor
        custom_limit = SmartSearchLimit(
            solver, 
            routing.CostVar(), 
            no_improvement_limit, 
            target_cost
        )
        routing.AddSearchMonitor(custom_limit)

    # --- 4. Solve ---
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Return tuple: (Cost, Memory Usage Bytes)
        return solution.ObjectiveValue(), routing.solver().MemoryUsage()
    else:
        return None, None