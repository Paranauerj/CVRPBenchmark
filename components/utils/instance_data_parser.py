import math
import numpy as np
import vrplib

def load_vrp_instance(file_path):
    """
    Loads a VRP instance from a .vrp file and returns a structured data dictionary.
    """

    instance = vrplib.read_instance(file_path)
    data = {}

    data['demands'] = instance["demand"]
    data['depot'] = 0
    data['capacity'] = instance.get("capacity")

    total_demand = np.sum(data['demands'])
    min_vehicles = int(np.ceil(total_demand / data['capacity']))
    
    # Store minimum vehicles needed
    data['min_vehicles'] = min_vehicles
    # Default to minimum, but can be overridden
    data['num_vehicles'] = min_vehicles
    data['vehicle_capacities'] = np.full(data['num_vehicles'], data['capacity'])

    raw_coords = instance.get("node_coord")
    data['num_nodes'] = len(raw_coords)

    # Map coordinates to a dictionary with node indices
    mapped_coords = {}
    if raw_coords is not None:
        for i in range(len(raw_coords)):
            mapped_coords[i] = (raw_coords[i][0], raw_coords[i][1])
            
    # Compute distance matrix
    data['coordinates'] = mapped_coords
    distance_matrix = np.zeros((data['num_nodes'], data['num_nodes']), dtype=int)

    for i in range(data['num_nodes']):
        for j in range(data['num_nodes']):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                c1 = mapped_coords.get(i, (0, 0))
                c2 = mapped_coords.get(j, (0, 0))
                dist = int(math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) + 0.5)
                distance_matrix[i][j] = dist
    
    data['distance_matrix'] = distance_matrix.tolist()
    return data