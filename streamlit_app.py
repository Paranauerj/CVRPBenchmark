"""CVRP Benchmarker - Main Application."""

import streamlit as st
import instance_data_parser
import configurable_solver
from uuid import uuid4

from components.ui.sidebar import render_sidebar, FIRST_SOLUTIONS, METAHEURISTICS
from components.execution.single_benchmark import run_single_benchmark, run_single_benchmark_background
from components.execution.bulk_benchmark import run_bulk_benchmark, run_bulk_benchmark_background
from components.execution.background_task import run_background_task
from components.visualization.results_display import display_results
from components.visualization.monitoring import render_monitor_page


# --- Helper Functions ---
def _init_session_state():
    """Initialize session state variables."""
    defaults = {
        'results_df': None,
        'all_histories': {},
        'show_bulk_config': False,
        'selected_bulk_instances': [],
        'run_bulk': False,
        'active_tab': 'Run Benchmarks'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _prepare_benchmark_settings(sidebar_settings):
    """Prepare common benchmark settings."""
    return {
        "sel_fs": sidebar_settings["sel_fs"],
        "sel_mh": sidebar_settings["sel_mh"],
        "fs_enum": FIRST_SOLUTIONS,
        "mh_enum": METAHEURISTICS,
        "solver_func": configurable_solver.solve_cvrp,
        "time_limit": sidebar_settings["time_limit"],
        "sol_limit": sidebar_settings["sol_limit"],
        "lns_limit": sidebar_settings["lns_limit"],
        "no_improv": sidebar_settings["no_improv"],
        "no_improv_iter": sidebar_settings["no_improv_iter"],
        "reps": sidebar_settings["reps"],
    }


def _prepare_single_benchmark_settings(sidebar_settings):
    """Prepare settings for single instance benchmark."""
    settings = _prepare_benchmark_settings(sidebar_settings)
    
    # Calculate target value
    target_val = None
    if sidebar_settings["target_gap"] and sidebar_settings["bks_cost"]:
        target_val = sidebar_settings["bks_cost"] * (1.0 + sidebar_settings["target_gap"] / 100.0)
    
    settings["target_val"] = target_val
    return settings


def _prepare_bulk_benchmark_settings(sidebar_settings):
    """Prepare settings for bulk benchmark."""
    settings = _prepare_benchmark_settings(sidebar_settings)
    settings["save_to_server"] = sidebar_settings.get("save_to_server", False)
    settings["selected_instances"] = st.session_state.get('selected_bulk_instances', [])
    settings["num_parallel_instances"] = st.session_state.get('num_parallel_instances', 2)
    return settings


# --- Page Configuration ---
st.set_page_config(page_title="CVRP Benchmarker", layout="wide")

# --- Session State Initialization ---
_init_session_state()

# --- Main Page ---
st.title("CVRP Benchmarker 📊")

# Render sidebar OUTSIDE tabs (so it's always accessible)
sidebar_settings = render_sidebar()

# Create tabs
tab1, tab2 = st.tabs(["🚀 Run Benchmarks", "📊 Monitor"])

# --- RUN BENCHMARKS TAB ---
with tab1:
    st.session_state.active_tab = 'Run Benchmarks'

    # Add Run button
    if st.button("🚀 Run", type="primary", 
                 disabled=not (sidebar_settings["sel_inst"] and sidebar_settings["sel_fs"] and sidebar_settings["sel_mh"]),
                 width='stretch'):
        # Start single benchmark in background
        task_id = str(uuid4())[:8]
        instance_file = sidebar_settings["p_map"].get(sidebar_settings["sel_inst"], {}).get("vrp")
        task_name = f"Single - {sidebar_settings['sel_inst']}"
        
        # Prepare settings
        benchmark_settings = _prepare_single_benchmark_settings(sidebar_settings)
        
        # Run in background
        run_background_task(
            run_single_benchmark_background,
            task_id,
            task_name,
            benchmark_settings,
            instance_file,
            sidebar_settings["bks_cost"]
        )
        
        st.success(f"✅ Benchmark started! Task ID: `{task_id}`")
        st.info("Switch to the **Monitor** tab to follow the progress.")

    # --- Bulk Benchmark Execution ---
    if st.session_state.run_bulk:
        selected_instances = st.session_state.get('selected_bulk_instances', [])
        
        if selected_instances:
            st.subheader("🚀 Starting Bulk Benchmark")
            
            # Prepare bulk settings
            bulk_settings = _prepare_bulk_benchmark_settings(sidebar_settings)
            
            # Start in background
            task_id = str(uuid4())[:8]
            task_name = f"Bulk - {len(selected_instances)} instances"
            run_background_task(
                run_bulk_benchmark_background,
                task_id,
                task_name,
                bulk_settings
            )
            
            st.success(f"✅ Bulk benchmark started! Task ID: `{task_id}`")
            st.info("📊 Switch to the **Monitor** tab to follow the progress.")
            
            st.session_state.run_bulk = False
        else:
            st.error("❌ No instances selected for bulk run.")
            st.session_state.run_bulk = False

    # --- Display quick note ---
    st.divider()
    st.markdown("""
    ### How to use:
    1. Configure your benchmark settings using the sidebar
    2. Click **Run** to start the benchmark
    3. Switch to the **Monitor** tab to track progress
    4. Results are automatically saved and can be downloaded
    """)

# --- MONITOR TAB ---
with tab2:
    st.session_state.active_tab = 'Monitor'
    render_monitor_page()

