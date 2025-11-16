import math

def load_vrp_instance(file_obj):
    """
    Parses a CVRPLIB .vrp file (like Uchoa's) from a file-like object.
    
    The file_obj can be from open() or a Streamlit FileUploader.
    
    Returns a data dictionary compatible with our OR-Tools solvers.
    """
    data = {}
    coords = {}
    demands_dict = {}
    
    section = "HEADER"
    
    # Process the file line by line
    for line in file_obj:
        # Handle both binary (from upload) and text (from open)
        if hasattr(line, 'decode'):
            line = line.decode('utf-8')
        line = line.strip()

        if not line:
            continue

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
            
        # --- Parse Header ---
        if section == "HEADER":
            if line.startswith("NAME"):
                # Try to parse 'k' (num_vehicles) from NAME line, e.g., "P-n16-k8"
                try:
                    k_val = line.split('-k')[-1]
                    data['num_vehicles'] = int(k_val)
                except (ValueError, IndexError):
                    print("Warning: Could not parse num_vehicles from NAME. Defaulting to 1.")
                    data['num_vehicles'] = 1
            elif line.startswith("CAPACITY"):
                data['capacity'] = int(line.split()[-1])
            elif line.startswith("DIMENSION"):
                data['num_nodes'] = int(line.split()[-1])
                
        # --- Parse Coords ---
        elif section == "COORDS":
            parts = line.split()
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coords[node_id] = (x, y)
            
        # --- Parse Demands ---
        elif section == "DEMANDS":
            parts = line.split()
            node_id = int(parts[0])
            demand = int(parts[1])
            demands_dict[node_id] = demand

        # --- Parse Depot ---
        elif section == "DEPOT":
            depot_id = int(line)
            if depot_id != -1:
                # CVRPLIB files are 1-indexed, our model is 0-indexed
                data['depot'] = depot_id - 1
                
    # --- Post-Processing ---
    
    # 1. Create vehicle capacity list
    if 'capacity' in data and 'num_vehicles' in data:
        data['vehicle_capacities'] = [data['capacity']] * data['num_vehicles']
    else:
        # Fallback if header parsing failed
        if 'num_vehicles' not in data: data['num_vehicles'] = 1
        if 'capacity' not in data: data['capacity'] = 100
        data['vehicle_capacities'] = [data['capacity']] * data['num_vehicles']

    # 2. Create demands list in correct 0-indexed order
    num_nodes = data.get('num_nodes', len(coords))
    data['demands'] = [demands_dict.get(i + 1, 0) for i in range(num_nodes)]

    # 3. Calculate distance matrix (using Euclidean distance, NINT)
    data['distance_matrix'] = []
    for i in range(1, num_nodes + 1):
        row = []
        for j in range(1, num_nodes + 1):
            c1 = coords[i]
            c2 = coords[j]
            # NINT (Nearest Integer) as per CVRPLIB spec
            dist = int(math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) + 0.5)
            row.append(dist)
        data['distance_matrix'].append(row)
        
    if 'depot' not in data:
        data['depot'] = 0 # Default to node 0

    return data