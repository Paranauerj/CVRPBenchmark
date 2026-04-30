"""Constants for CVRP Benchmark - Column names, field names, and other hardcoded strings."""

# ============================================================================
# INSTANCE METADATA COLUMNS
# ============================================================================

# Instance information
COL_INSTANCE = "Instance"
COL_DEPOT_LAYOUT = "Depot Layout"
COL_CUSTOMER_LAYOUT = "Cust Layout"
COL_DEMAND_TYPE = "Demand Type"
COL_ROUTE_CLASS = "Route Class"
COL_CLIMATE = "Climate"
COL_CUSTOMERS = "Customers"
COL_VEHICLES = "Vehicles"
COL_CAPACITY = "Capacity"

# ============================================================================
# ALGORITHM COLUMNS
# ============================================================================
COL_FIRST_SOLUTION = "First Solution"
COL_METAHEURISTIC = "Metaheuristic"

# ============================================================================
# METRICS COLUMNS
# ============================================================================

# Count metrics
COL_REPETITIONS = "Repetitions"

# Cost metrics
COL_BEST_COST = "Best Cost"
COL_AVG_COST = "Avg Cost"
COL_BKS_COST = "BKS Cost"

# Gap metrics
COL_BEST_GAP = "Best Gap (%)"
COL_AVG_GAP = "Avg Gap (%)"

# Time metrics
COL_AVG_CPU_TIME = "Avg CPU Time (s)"

# ============================================================================
# TIME CHECKPOINT COLUMNS (Dynamic - use template)
# ============================================================================
CHECKPOINT_BEST_COST_TEMPLATE = "Best Cost @ {}s"
CHECKPOINT_AVG_COST_TEMPLATE = "Avg Cost @ {}s"

def get_checkpoint_best_cost_col(time_sec: int) -> str:
    """Get the column name for best cost at a specific time checkpoint."""
    return CHECKPOINT_BEST_COST_TEMPLATE.format(time_sec)

def get_checkpoint_avg_cost_col(time_sec: int) -> str:
    """Get the column name for average cost at a specific time checkpoint."""
    return CHECKPOINT_AVG_COST_TEMPLATE.format(time_sec)

# ============================================================================
# INSTANCE DATA KEYS (Dictionary keys used internally)
# ============================================================================

KEY_DEMANDS = "demands"
KEY_DEPOT = "depot"
KEY_CAPACITY = "capacity"
KEY_MIN_VEHICLES = "min_vehicles"
KEY_NUM_VEHICLES = "num_vehicles"
KEY_VEHICLE_CAPACITIES = "vehicle_capacities"
KEY_NUM_NODES = "num_nodes"
KEY_COORDINATES = "coordinates"
KEY_DISTANCE_MATRIX = "distance_matrix"

# ============================================================================
# VRPLIB INSTANCE KEYS
# ============================================================================

VRPLIB_KEY_DEMAND = "demand"
VRPLIB_KEY_CAPACITY = "capacity"
VRPLIB_KEY_NODE_COORD = "node_coord"

# ============================================================================
# RESULT AGGREGATION GROUPS
# ============================================================================

ALGORITHM_COLS = [COL_METAHEURISTIC, COL_FIRST_SOLUTION]
METADATA_COLS = [
    COL_INSTANCE,
    COL_DEPOT_LAYOUT,
    COL_CUSTOMER_LAYOUT,
    COL_DEMAND_TYPE,
    COL_ROUTE_CLASS,
    COL_CLIMATE,
    COL_CUSTOMERS,
    COL_VEHICLES,
    COL_CAPACITY,
]
COST_COLS = [COL_BEST_COST, COL_AVG_COST, COL_BKS_COST]
GAP_COLS = [COL_BEST_GAP, COL_AVG_GAP]
TIME_COLS = [COL_AVG_CPU_TIME]
COUNT_COLS = [COL_REPETITIONS]

# All base columns (without checkpoints)
ALL_BASE_COLS = METADATA_COLS + ALGORITHM_COLS + COST_COLS + GAP_COLS + TIME_COLS + COUNT_COLS

# ============================================================================
# UI DISPLAY SETTINGS
# ============================================================================

# Format strings for column configuration
FORMAT_COST = "%.2f"
FORMAT_GAP = "%.4f%%"
FORMAT_TIME = "%.6f"

# Display column groups
DISPLAY_COLS_ALGORITHM = ALGORITHM_COLS
DISPLAY_COLS_COST = COST_COLS
DISPLAY_COLS_TIME = TIME_COLS

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_NO_COORDS = "Instance must contain node coordinates"
ERROR_NO_RESULTS = "No results to display."
ERROR_NO_INSTANCES = "No instances selected for bulk run."
ERROR_NO_ALGORITHMS = "Please select at least one algorithm in the sidebar."
ERROR_NO_INSTANCES_FOUND = "No instances found in '{}' folder."
ERROR_NO_CHECKPOINT_DATA = "No convergence data available."
ERROR_NO_VISUALIZATION_DATA = "Route visualization requires both convergence history and best routes data."

# ============================================================================
# HELPER FUNCTIONS FOR COMMON PATTERNS
# ============================================================================

def get_display_columns(include_gaps: bool = True, include_time: bool = True) -> list[str]:
    """
    Get standard display columns for results table.
    
    Args:
        include_gaps: Include gap columns if True
        include_time: Include time columns if True
        
    Returns:
        List of column names to display
    """
    cols = ALGORITHM_COLS + DISPLAY_COLS_COST
    
    if include_gaps:
        cols.extend(GAP_COLS)
    
    if include_time:
        cols.extend(TIME_COLS)
    
    cols.extend(COUNT_COLS)
    
    return cols


def get_column_config_for_display(display_columns: list[str]) -> dict:
    """
    Get Streamlit column configuration for number formatting.
    
    Args:
        display_columns: List of column names to configure
        
    Returns:
        Dictionary mapping column names to st.column_config objects
    """
    import streamlit as st
    
    config = {}
    
    for col in display_columns:
        if col in [COL_BEST_COST, COL_AVG_COST, COL_BKS_COST]:
            config[col] = st.column_config.NumberColumn(col, format=FORMAT_COST)
        elif col in [COL_BEST_GAP, COL_AVG_GAP]:
            config[col] = st.column_config.NumberColumn(col, format=FORMAT_GAP)
        elif col == COL_AVG_CPU_TIME:
            config[col] = st.column_config.NumberColumn(col, format=FORMAT_TIME)
    
    return config
