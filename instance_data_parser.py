import math

def load_vrp_instance(file_obj):
    """Parses CVRPLIB .vrp file."""
    data = {}
    coords = {}
    demands_dict = {}
    section = "HEADER"
    
    for line in file_obj:
        if hasattr(line, 'decode'): line = line.decode('utf-8')
        line = line.strip()
        if not line: continue

        if "NODE_COORD_SECTION" in line: section = "COORDS"; continue
        elif "DEMAND_SECTION" in line: section = "DEMANDS"; continue
        elif "DEPOT_SECTION" in line: section = "DEPOT"; continue
        elif "EOF" in line: break
            
        if section == "HEADER":
            if line.startswith("NAME"):
                try:
                    k_val = line.split('-k')[-1].split()[0]
                    data['num_vehicles'] = int(k_val)
                except: data['num_vehicles'] = 1
            elif line.startswith("CAPACITY"): data['capacity'] = int(line.split()[-1])
                
        elif section == "COORDS":
            parts = line.split()
            if not parts[0].isdigit(): continue
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            
        elif section == "DEMANDS":
            parts = line.split()
            if not parts[0].isdigit(): continue
            demands_dict[int(parts[0])] = int(parts[1])

        elif section == "DEPOT":
            parts = line.split()
            if parts[0].isdigit() and int(parts[0]) != -1:
                data['depot'] = int(parts[0]) - 1
                
    if 'capacity' in data and 'num_vehicles' in data:
        data['vehicle_capacities'] = [data['capacity']] * data['num_vehicles']
    else:
        data['vehicle_capacities'] = [100] * 1

    num_nodes = len(coords)
    data['demands'] = [demands_dict.get(i + 1, 0) for i in range(num_nodes)]
    data['distance_matrix'] = []
    
    for i in range(1, num_nodes + 1):
        row = []
        for j in range(1, num_nodes + 1):
            c1, c2 = coords[i], coords[j]
            dist = int(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) + 0.5)
            row.append(dist)
        data['distance_matrix'].append(row)
        
    if 'depot' not in data: data['depot'] = 0
    return data