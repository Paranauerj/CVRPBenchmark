"""Results display and visualization."""

import streamlit as st
from components.utils import instance_data_parser
from components.visualization.plotting import plot_routes, plot_convergence_comparison, plot_single_benchmark_results
from components import constants as C


from components.ui.ui_utils import get_column_config_for_display


def display_results(results_df, p_map, sel_inst, bks_cost, time_limit, all_histories, instance_data=None, all_best_routes=None, sidebar_settings=None):
    """Display benchmark results with tables, plots, and routes."""
    if results_df is None or results_df.empty:
        st.warning("No results to display.")
        return
    
    df = results_df
    engine = sidebar_settings.get("engine", "ortools") if sidebar_settings else "ortools"
    
    # 1. Results Table - Show key columns
    st.subheader("Benchmark Results")
    
    # Build columns list based on what's available
    display_cols = []
    
    # Algorithm info
    if C.COL_SOLVER in df.columns:
        display_cols.append(C.COL_SOLVER)
    if engine == "ortools":
        if C.COL_METAHEURISTIC in df.columns:
            display_cols.append(C.COL_METAHEURISTIC)
        if C.COL_FIRST_SOLUTION in df.columns:
            display_cols.append(C.COL_FIRST_SOLUTION)
    
    # Cost columns
    if C.COL_BEST_COST in df.columns:
        display_cols.append(C.COL_BEST_COST)
    if C.COL_AVG_COST in df.columns:
        display_cols.append(C.COL_AVG_COST)
    
    # Gap columns (if we have BKS and not HGS)
    if engine != "hgs" and bks_cost is not None:
        if C.COL_BEST_GAP in df.columns:
            display_cols.append(C.COL_BEST_GAP)
        if C.COL_AVG_GAP in df.columns:
            display_cols.append(C.COL_AVG_GAP)
    
    # Time columns
    if C.COL_AVG_CPU_TIME in df.columns:
        display_cols.append(C.COL_AVG_CPU_TIME)
    if engine != "hgs" and C.COL_TIME_TO_TARGET in df.columns:
        display_cols.append(C.COL_TIME_TO_TARGET)
    
    # Repetitions
    if C.COL_REPETITIONS in df.columns:
        display_cols.append(C.COL_REPETITIONS)
    
    # Filter to only existing columns
    display_cols = [c for c in display_cols if c in df.columns]
    
    if display_cols:
        df_display = df[display_cols].copy()
        
        # Use helper function to get column config
        column_config = get_column_config_for_display(display_cols)
        
        st.dataframe(df_display, width='stretch', column_config=column_config if column_config else None)
    
    # 2. Time Checkpoint Analysis (Skip for HGS)
    if engine != "hgs":
        st.subheader("Convergence Analysis")
        
        checkpoint_cols = [col for col in df.columns if col.startswith(C.CHECKPOINT_BEST_COST_TEMPLATE.split("{}")[0])]
        if checkpoint_cols:
            st.write("**Cost at Time Checkpoints:**")
            display_cols_checkpoint = []
            if C.COL_METAHEURISTIC in df.columns:
                display_cols_checkpoint.append(C.COL_METAHEURISTIC)
            if C.COL_FIRST_SOLUTION in df.columns:
                display_cols_checkpoint.append(C.COL_FIRST_SOLUTION)
            display_cols_checkpoint.extend(checkpoint_cols)
            checkpoint_display = df[display_cols_checkpoint]
            st.dataframe(checkpoint_display, width='stretch')
        else:
            st.info(C.ERROR_NO_CHECKPOINT_DATA)
    
    # 3. Route and Convergence Visualization
    st.divider()
    if instance_data and all_histories:
        # Get experiment names from results
        exp_names = []
        if engine == "hgs":
            exp_names = ["HGS-CVRP"]
        else:
            if C.COL_METAHEURISTIC in df.columns and C.COL_FIRST_SOLUTION in df.columns:
                for _, row in df.iterrows():
                    mh = row.get(C.COL_METAHEURISTIC)
                    fs = row.get(C.COL_FIRST_SOLUTION)
                    if mh and fs:
                        exp_names.append(f"{mh} [{fs}]")
        
        if exp_names and all_best_routes:
            # Custom logic for HGS: Hide convergence, show only best routes
            from components.visualization.plotting import plot_routes
            if engine == "hgs":
                st.subheader("🗺️ Best Routes")
                coordinates = instance_data.get('coordinates', {})
                customers = []
                depot_pos = coordinates.get(0, (0, 0))
                if coordinates:
                    num_nodes = instance_data.get('num_nodes', 0)
                    for i in range(1, num_nodes):
                        if i in coordinates:
                            customers.append(coordinates[i])
                
                if customers:
                    for exp_name in exp_names:
                        if exp_name in all_best_routes and all_best_routes[exp_name]:
                            routes = all_best_routes[exp_name]
                            fig_routes = plot_routes(customers, routes, depot_pos=depot_pos, title=f"Best Routes - {exp_name}", figsize=(7, 5))
                            
                            # Use a column layout to make the plot smaller
                            col_plot, _ = st.columns([0.6, 0.4])
                            with col_plot:
                                st.pyplot(fig_routes)
            else:
                plot_single_benchmark_results(instance_data, all_histories, all_best_routes, exp_names)
        else:
            st.info(C.ERROR_NO_VISUALIZATION_DATA)


