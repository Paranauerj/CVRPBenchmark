import vrplib

def parse_solution_file(file_path):
    """
    Parses CVRPLIB .sol file for Cost.
    """

    solution = vrplib.read_solution(file_path)
    return solution.get("cost", None)

def save_solution_file(file_path, routes, cost):
    """
    Saves solution in CVRPLIB .sol format.
    routes: list of lists of customer indices.
    cost: float
    """
    with open(file_path, 'w') as f:
        for i, route in enumerate(routes):
            # CVRPLIB format: Route i: c1 c2 ...
            f.write(f"Route {i+1}: {' '.join(map(str, route))}\n")
        f.write(f"Cost {cost}\n")