import math
import os

def load_vrp_instance(file_obj):
    """
    Parses CVRPLIB .vrp file manually.
    """
    data = {}
    coords = {}
    demands_dict = {}
    section = "HEADER"
    
    # Read all lines
    # file_obj is a text file object from open(..., 'r')
    # We iterate directly or read()
    if hasattr(file_obj, 'read'):
        content = file_obj.read()
        if isinstance(content, bytes):
            lines = content.decode('utf-8').splitlines()
        else:
            lines = content.splitlines()
    else:
        lines = []

    for line in lines:
        line = line.strip()
        if not line: continue

        if "NODE_COORD_SECTION" in line: 
            section = "COORDS"
            continue
        elif "DEMAND_SECTION" in line: 
            section = "DEMANDS"
            continue
        elif "DEPOT_SECTION" in line: 
            section = "DEPOT"
            continue
        elif "EOF" in line: 
            break
            
        if section == "HEADER":
            if line.startswith("NAME"):
                try:
                    k_val = line.split('-k')[-1].split()[0]
                    data['num_vehicles'] = int(k_val)
                except: 
                    data['num_vehicles'] = 1
            elif line.startswith("CAPACITY"): 
                data['capacity'] = int(line.split()[-1])
            elif line.startswith("DIMENSION"):
                data['num_nodes'] = int(line.split()[-1])
                
        elif section == "COORDS":
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)
            
        elif section == "DEMANDS":
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                node_id = int(parts[0])
                demand = int(parts[1])
                demands_dict[node_id] = demand

        elif section == "DEPOT":
            parts = line.split()
            if parts[0].isdigit():
                val = int(parts[0])
                if val != -1:
                    data['depot_id_raw'] = val

    # --- Post Processing ---
    # Ensure capacity list
    if 'capacity' in data:
        n_vehicles = data.get('num_vehicles', 1)
        data['vehicle_capacities'] = [data['capacity']] * n_vehicles
    else:
        data['vehicle_capacities'] = [100] * 1
        data['num_vehicles'] = 1

    # In CVRPLIB, nodes are 1..N. Depot is usually 1.
    # OR-Tools needs 0-based indexing.
    # We assume the file nodes 1..N map to indices 0..N-1.
    
    num_nodes = len(coords)
    if num_nodes == 0:
        # Fallback if parsing failed or DIMENSION was used but COORDS missing
        num_nodes = data.get('num_nodes', 0)

    # Build Demands (0-based)
    # Index 0 corresponds to Node 1, Index 1 to Node 2, etc.
    demands = []
    for i in range(1, num_nodes + 1):
        demands.append(demands_dict.get(i, 0))
    data['demands'] = demands

    # Build Distance Matrix (0-based)
    distance_matrix = []
    for i in range(1, num_nodes + 1):
        row = []
        for j in range(1, num_nodes + 1):
            if i == j:
                row.append(0)
            else:
                c1 = coords.get(i, (0,0))
                c2 = coords.get(j, (0,0))
                dist = int(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) + 0.5)
                row.append(dist)
        distance_matrix.append(row)
    data['distance_matrix'] = distance_matrix
    
    # Depot is index 0 (Node 1)
    data['depot'] = 0
    
    # --- ADD COORDINATES FOR PLOTTING ---
    # Map 0-based index back to (x, y)
    # Index 0 -> Node 1's coords
    mapped_coords = {}
    for i in range(1, num_nodes + 1):
        if i in coords:
            mapped_coords[i-1] = coords[i]
            
    data['coordinates'] = mapped_coords
    # ------------------------------------

    return data