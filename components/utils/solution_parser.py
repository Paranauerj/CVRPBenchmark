import vrplib

def parse_solution_file(file_path):
    """
    Parses CVRPLIB .sol file for Cost.
    """
    
    solution = vrplib.read_solution(file_path)
    return solution.get("cost", None)