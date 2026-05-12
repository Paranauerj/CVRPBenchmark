"""Helper functions for CVRPBenchmark."""

import os
import glob


def find_instance_files(directory="instances"):
    """Find all VRP instance files in a directory."""
    if not os.path.exists(directory):
        return [], {}
    vrp_files = sorted(glob.glob(os.path.join(directory, "**", "*.vrp"), recursive=True))
    valid_names, path_map = [], {}
    for p in vrp_files:
        base = os.path.basename(p).replace(".vrp", "")
        sol = p.replace(".vrp", ".sol")
        valid_names.append(base)
        path_map[base] = {
            "vrp": p,
            "sol": sol if os.path.exists(sol) else None
        }
    return valid_names, path_map


def get_climate_from_filename(filename):
    """
    Extracts climate from Gaetano filenames like 'LDG95_3376_rain_95_0088'.
    Assumes format: Series_Depot_Climate_...
    """
    parts = filename.split('_')
    valid_climates = {'rain', 'fog', 'snow', 'none'}
    for part in parts:
        if part.lower() in valid_climates:
            return part.lower()
    return "unknown"


def get_cost_at_time(history, time_limit_sec):
    """
    Finds the best cost found within the time_limit_sec based on history.
    History is a list of tuples: (time_elapsed, accepted_neighbors, cost)
    """
    if not history:
        return None
    
    best_cost_at_t = None
    
    for t, iters, cost in history:
        if t <= time_limit_sec:
            best_cost_at_t = cost
        else:
            # Since history is sorted by time, we can stop early
            break
            
    return best_cost_at_t


def parse_gaetano_metadata(filename):
    """
    Parses filenames like 'LDG95_3376_rain_95_0088' to extract structural features.
    Returns a dictionary of features for ML model training.
    
    Filename structure: PREFIX_PARAMS_CLIMATE_SIZE_SEED
    PARAMS digits breakdown:
    - Position 0: Depot layout (1=Random, 2=Center, 3=Corner)
    - Position 1: Customer layout (1=Random, 2=Cluster, 3=Mix)
    - Position 2: Demand profile type (0-9)
    - Position 3: Route size class (1-6)
    """
    try:
        # Split: ['LDG95', '3376', 'rain', '95', '0088']
        parts = filename.split('_')
        
        # 1. Parse encoded structural parameters (e.g., '3376')
        code_block = parts[1] 
        root_pos_id = int(code_block[0])
        cust_pos_id = int(code_block[1])
        demand_type_id = int(code_block[2])
        route_size_id = int(code_block[3])

        # Mappings based on generator script
        root_map = {1: "Random", 2: "Center", 3: "Corner"}
        cust_map = {1: "Random", 2: "Cluster", 3: "Mix"}
        route_map = {
            1: "Very Short (3-5)", 2: "Short (5-8)", 3: "Medium (8-12)",
            4: "Long (12-16)", 5: "Very Long (16-25)", 6: "Ultra Long (25-50)"
        }
        
        # 2. Extract Climate
        climate = parts[2]

        return {
            "Depot Layout": root_map.get(root_pos_id, "Unknown"),
            "Customer Layout": cust_map.get(cust_pos_id, "Unknown"),
            "Demand Profile ID": demand_type_id,
            "Route Length Class": route_map.get(route_size_id, "Unknown"),
            "Climate": climate
        }
    except Exception as e:
        return {
            "Depot Layout": "Error",
            "Customer Layout": "Error", 
            "Demand Profile ID": 0,
            "Route Length Class": "Error", 
            "Climate": "Unknown"
        }

