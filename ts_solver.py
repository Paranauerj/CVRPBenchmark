# ts_solver.py
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from baseline_solver import _create_routing_model # Reuse the setup function

def solve_ts(data, time_limit_seconds=None, solution_limit=None):
    """Solves the CVRP using Tabu Search (TS)."""
    
    manager, routing = _create_routing_model(data)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # --- TS Configuration ---
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # **Enable Tabu Search**
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
        
    # Only set parameters when explicitly provided; otherwise keep OR-Tools defaults
    if time_limit_seconds is not None:
        search_parameters.time_limit.seconds = int(time_limit_seconds)

    if solution_limit is not None:
        search_parameters.solution_limit = int(solution_limit)
    # ---
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        return solution.ObjectiveValue()
    else:
        return None