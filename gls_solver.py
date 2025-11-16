# gls_solver.py
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from baseline_solver import _create_routing_model # Reuse the setup function!

def solve_gls(data):
    """Solves the CVRP using Guided Local Search (GLS)."""
    
    manager, routing = _create_routing_model(data)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # --- GLS Configuration ---
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # **Enable Guided Local Search**
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    # Set a time limit for the search
    search_parameters.time_limit.seconds = 5
    search_parameters.solution_limit = 1000
    # ---
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        return solution.ObjectiveValue()
    else:
        return None