import math
import numpy as np
import vrplib
from components import constants as C


def load_vrp_instance(file_path):
    """
    Loads a VRP instance from a .vrp file and returns a structured data dictionary.
    """

    instance = vrplib.read_instance(file_path)
    data = {}

    data[C.KEY_DEMANDS] = instance[C.VRPLIB_KEY_DEMAND]
    data[C.KEY_DEPOT] = 0
    data[C.KEY_CAPACITY] = instance.get(C.VRPLIB_KEY_CAPACITY)

    total_demand = np.sum(data[C.KEY_DEMANDS])
    min_vehicles = int(np.ceil(total_demand / data[C.KEY_CAPACITY]))
    
    # Store minimum vehicles needed
    data[C.KEY_MIN_VEHICLES] = min_vehicles
    # Default to minimum, but can be overridden
    data[C.KEY_NUM_VEHICLES] = min_vehicles
    data[C.KEY_VEHICLE_CAPACITIES] = np.full(data[C.KEY_NUM_VEHICLES], data[C.KEY_CAPACITY])

    raw_coords = instance.get(C.VRPLIB_KEY_NODE_COORD)
    
    # Ensure raw_coords is not None before using it
    if raw_coords is None:
        raise ValueError(C.ERROR_NO_COORDS)
    
    data[C.KEY_NUM_NODES] = len(raw_coords)

    # Map coordinates to a dictionary with node indices
    mapped_coords = {}
    for i in range(len(raw_coords)):
        mapped_coords[i] = (raw_coords[i][0], raw_coords[i][1])
            
    # Compute distance matrix
    data[C.KEY_COORDINATES] = mapped_coords
    distance_matrix = np.zeros((data[C.KEY_NUM_NODES], data[C.KEY_NUM_NODES]), dtype=int)

    for i in range(data[C.KEY_NUM_NODES]):
        for j in range(data[C.KEY_NUM_NODES]):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                c1 = mapped_coords.get(i, (0, 0))
                c2 = mapped_coords.get(j, (0, 0))
                dist = int(math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) + 0.5)
                distance_matrix[i][j] = dist
    
    data[C.KEY_DISTANCE_MATRIX] = distance_matrix.tolist()
    return data