"""Results display and visualization."""

from components.utils import instance_data_parser
from components.visualization.plotting import plot_routes, plot_convergence


def display_results(results_df, p_map, sel_inst, bks_cost, time_limit, all_histories):
    """Display benchmark results with tables, plots, and routes."""
    import streamlit as st
    if results_df is None:
        return
    
    df = results_df
    
    # 1. Results Table
    st.subheader("Benchmark Results")
    
    base_cols = ["Algorithm", "Best Cost", "Avg Cost", "Best Run Gap (%)"]
    bks_gap_cols = ["Best Gap (%)", "Avg Gap (%)"]
    other_cols = ["CPU Time (s)", "Vehicles Used", "Accepted Neighbors", "Repetitions"]
    
    cols = base_cols.copy()
    
    if bks_cost is not None and "Best Gap (%)" in df.columns:
        cols.extend(bks_gap_cols)
    
    cols.extend(other_cols)
    cols = [c for c in cols if c in df.columns]
    
    df_display = df[cols].copy()
    
    # Configure columns
    column_config = {}
    if "Best Run Gap (%)" in df_display.columns:
        column_config["Best Run Gap (%)"] = st.column_config.NumberColumn("Best Run Gap (%)", format="%.4f%%")
    if "Best Gap (%)" in df_display.columns:
        column_config["Best Gap (%)"] = st.column_config.NumberColumn("Best Gap (%)", format="%.4f%%")
    if "Avg Gap (%)" in df_display.columns:
        column_config["Avg Gap (%)"] = st.column_config.NumberColumn("Avg Gap (%)", format="%.4f%%")
    if "Best Cost" in df_display.columns:
        column_config["Best Cost"] = st.column_config.NumberColumn("Best Cost", format="%.2f")
    if "Avg Cost" in df_display.columns:
        column_config["Avg Cost"] = st.column_config.NumberColumn("Avg Cost", format="%.2f")
    if "CPU Time (s)" in df_display.columns:
        column_config["CPU Time (s)"] = st.column_config.NumberColumn("CPU Time (s)", format="%.6f")
    
    st.dataframe(df_display, width='stretch', column_config=column_config if column_config else None)

    # 2. Convergence Plots
    st.subheader("Convergence Analysis")
    
    max_time_val = df["CPU Time (s)"].max() if not df.empty else 10
    if time_limit:
        max_time_val = max(max_time_val, time_limit)
    
    max_iter_val = df["Accepted Neighbors"].max() if not df.empty else 100

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Cost vs. Time")
        fig_time = plot_convergence(all_histories, "time", max_val=max_time_val)
        st.pyplot(fig_time)
        
    with col2:
        st.markdown("#### Cost vs. Accepted Neighbors")
        fig_iter = plot_convergence(all_histories, "iterations", max_val=max_iter_val)
        st.pyplot(fig_iter)

    # 3. Route Plots
    st.subheader("Route Visualization (Best Run)")
    cols = st.columns(2)
    for i, row in df.iterrows():
        with cols[i % 2]:
            cost_str = f"(Cost: {row['Best Cost']:.2f})" if isinstance(row['Best Cost'], (int, float)) else "(No Solution)"
            st.markdown(f"**{row['Algorithm']}** {cost_str}")
            if row.get("_routes"):
                paths = p_map.get(sel_inst)
                inst_data = instance_data_parser.load_vrp_instance(paths["vrp"])
                fig = plot_routes(inst_data, row["_routes"], title="")
                st.pyplot(fig)
            else:
                st.warning("No solution found for this algorithm.")
