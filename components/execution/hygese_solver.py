import time
import numpy as np
import os
import platform

# Windows specific: PyHygese requires HGS C++ library which may depend on MinGW runtimes
if platform.system() == "Windows":
    # Try to add MinGW to DLL search path if installed via Scoop
    scoop_mingw = os.path.join(os.environ.get("USERPROFILE", ""), "scoop", "apps", "mingw", "current", "bin")
    if os.path.exists(scoop_mingw):
        try:
            os.add_dll_directory(scoop_mingw)
        except AttributeError:
            # Python < 3.8
            os.environ["PATH"] = scoop_mingw + os.pathsep + os.environ["PATH"]

import hygese as hgs

def solve_hgs(data, 
              time_limit_seconds=None, 
              no_improvement_limit_iterations=20000,
              random_seed=1,
              **kwargs):
    """
    Solves CVRP using PyHygese (Hybrid Genetic Search).
    
    Returns:
        tuple: (Cost, Accepted Neighbors, Routes, History)
    """

    # Prepare data for hygese
    coords = data.get('coordinates', {})
    if not coords:
        # If no coordinates, we must have distance_matrix
        if 'distance_matrix' not in data:
            raise ValueError("HGS requires either coordinates or a distance matrix.")
        
    # demands should have depot at index 0 with demand 0
    demands = np.array(data['demands'])
    
    hgs_data = {
        'demands': demands,
        'vehicle_capacity': data['capacity'],
        'num_vehicles': data['num_vehicles'],
        'depot': data.get('depot', 0)
    }
    
    if coords:
        # Sort keys to ensure correct order
        sorted_indices = sorted(coords.keys())
        hgs_data['x_coordinates'] = np.array([coords[i][0] for i in sorted_indices])
        hgs_data['y_coordinates'] = np.array([coords[i][1] for i in sorted_indices])
    
    if 'distance_matrix' in data:
         hgs_data['distance_matrix'] = np.array(data['distance_matrix'])

    # Map parameters
    # Note: hgs.AlgorithmParameters has timeLimit in seconds
    ap = hgs.AlgorithmParameters(
        timeLimit=float(time_limit_seconds) if time_limit_seconds else 0.0,
        nbIter=int(no_improvement_limit_iterations),
        seed=int(random_seed),
        nbGranular=kwargs.get('nbGranular', 20),
        mu=kwargs.get('mu', 25),
        lambda_=kwargs.get('lambda_', 40),
        nbElite=kwargs.get('nbElite', 4),
        nbClose=kwargs.get('nbClose', 5),
        targetFeasible=kwargs.get('targetFeasible', 0.2),
        useSwapStar=kwargs.get('useSwapStar', True)
    )
    
    solver = hgs.Solver(parameters=ap, verbose=False)
    
    start_cpu = time.process_time()
    result = solver.solve_cvrp(hgs_data)
    end_cpu = time.process_time()
    
    if result.cost > 0.01 and result.routes: # Check for a valid cost and routes
        # result.routes is a list of routes, where each route is a list of customer indices
        # result.cost is the total distance
        
        # Accepted neighbors is not directly provided by HGS in a compatible way, return 0
        # History: HGS doesn't provide a step-by-step history easily via the wrapper
        # We'll provide a single entry for the final result using CPU time
        history = [(end_cpu - start_cpu, 0, result.cost)]
        
        return result.cost, 0, result.routes, history, None
    else:
        return None, None, None, None, None
