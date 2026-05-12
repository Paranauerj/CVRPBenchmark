"""UI-specific utility functions for Streamlit."""

import streamlit as st
from components import constants as C

def get_column_config_for_display(display_columns: list[str]) -> dict:
    """
    Get Streamlit column configuration for number formatting.
    
    Args:
        display_columns: List of column names to configure
        
    Returns:
        Dictionary mapping column names to st.column_config objects
    """
    config = {}
    
    for col in display_columns:
        if col in [C.COL_BEST_COST, C.COL_AVG_COST, C.COL_BKS_COST]:
            config[col] = st.column_config.NumberColumn(col, format=C.FORMAT_COST)
        elif col in [C.COL_BEST_GAP, C.COL_AVG_GAP]:
            config[col] = st.column_config.NumberColumn(col, format=C.FORMAT_GAP)
        elif col in [C.COL_AVG_CPU_TIME, C.COL_TIME_TO_TARGET]:
            config[col] = st.column_config.NumberColumn(col, format=C.FORMAT_TIME)
    
    return config
