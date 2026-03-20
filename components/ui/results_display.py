"""Results display and visualization."""

import streamlit as st
from components.utils import instance_data_parser
from components.ui.plotting import plot_routes, plot_convergence


def display_results(results_df, p_map, sel_inst, bks_cost, time_limit, all_histories):
    """Display benchmark results with tables, plots, and routes."""
    if results_df is None or results_df.empty:
        st.warning("No results to display.")
        return
    
    df = results_df
    
    # 1. Results Table - Show key columns
    st.subheader("Benchmark Results")
    
    # Build columns list based on what's available
    display_cols = []
    
    # Algorithm info
    if "Metaheuristic" in df.columns:
        display_cols.append("Metaheuristic")
    if "First Solution" in df.columns:
        display_cols.append("First Solution")
    
    # Cost columns
    if "Best Cost" in df.columns:
        display_cols.append("Best Cost")
    if "Avg Cost" in df.columns:
        display_cols.append("Avg Cost")
    
    # Gap columns (if we have BKS)
    if bks_cost is not None:
        if "Best Gap (%)" in df.columns:
            display_cols.append("Best Gap (%)")
        if "Avg Gap (%)" in df.columns:
            display_cols.append("Avg Gap (%)")
    
    # Time columns
    if "Avg CPU Time (s)" in df.columns:
        display_cols.append("Avg CPU Time (s)")
    
    # Repetitions
    if "Repetitions" in df.columns:
        display_cols.append("Repetitions")
    
    # Filter to only existing columns
    display_cols = [c for c in display_cols if c in df.columns]
    
    if display_cols:
        df_display = df[display_cols].copy()
        
        # Configure columns formatting
        column_config = {}
        if "Best Cost" in df_display.columns:
            column_config["Best Cost"] = st.column_config.NumberColumn("Best Cost", format="%.2f")
        if "Avg Cost" in df_display.columns:
            column_config["Avg Cost"] = st.column_config.NumberColumn("Avg Cost", format="%.2f")
        if "Best Gap (%)" in df_display.columns:
            column_config["Best Gap (%)"] = st.column_config.NumberColumn("Best Gap (%)", format="%.4f%%")
        if "Avg Gap (%)" in df_display.columns:
            column_config["Avg Gap (%)"] = st.column_config.NumberColumn("Avg Gap (%)", format="%.4f%%")
        if "Avg CPU Time (s)" in df_display.columns:
            column_config["Avg CPU Time (s)"] = st.column_config.NumberColumn("Avg CPU Time (s)", format="%.6f")
        
        st.dataframe(df_display, width='stretch', column_config=column_config if column_config else None)
    
    # 2. Time Checkpoint Analysis
    st.subheader("Convergence Analysis")
    
    checkpoint_cols = [col for col in df.columns if col.startswith("Best Cost @")]
    if checkpoint_cols:
        st.write("**Cost at Time Checkpoints:**")
        checkpoint_display = df[["Metaheuristic"] + checkpoint_cols] if "Metaheuristic" in df.columns else df[checkpoint_cols]
        st.dataframe(checkpoint_display, width='stretch')
    else:
        st.info("No convergence data available.")

