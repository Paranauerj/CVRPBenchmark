"""CVRP Benchmarker - Main Application."""

import streamlit as st
import instance_data_parser
import configurable_solver
from uuid import uuid4

from components.ui.sidebar import render_sidebar, FIRST_SOLUTIONS, METAHEURISTICS
from components.execution.single_benchmark import run_single_benchmark
from components.execution.bulk_benchmark import run_bulk_benchmark, run_bulk_benchmark_background
from components.execution.background_task import run_background_task
from components.visualization.results_display import display_results


# --- Helper Functions ---
def _init_session_state():
    """Initialize session state variables."""
    defaults = {
        'run_benchmark': False,
        'results_df': None,
        'all_histories': {},
        'run_bulk': False,
        'show_bulk_config': False,
        'selected_bulk_instances': []
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
    return settings


# --- Page Configuration ---
st.set_page_config(page_title="CVRP Benchmarker", layout="wide")

# --- Session State Initialization ---
_init_session_state()

# --- Main Page ---
st.title("CVRP Benchmarker 📊")

# Render sidebar and get settings
sidebar_settings = render_sidebar()

# Add Run button
if st.button("🚀 Run", type="primary", 
             disabled=not (sidebar_settings["sel_inst"] and sidebar_settings["sel_fs"] and sidebar_settings["sel_mh"]) or st.session_state.run_benchmark,
             width='stretch'):
    st.session_state.run_benchmark = True
    st.session_state.results_df = None
    st.session_state.all_histories = {}
    st.rerun()


# --- Single Instance Benchmark Execution ---
if st.session_state.run_benchmark:
    paths = sidebar_settings["p_map"].get(sidebar_settings["sel_inst"])
    instance_data = instance_data_parser.load_vrp_instance(paths["vrp"])
    
    # Override num_vehicles with user selection for single benchmark
    if sidebar_settings["num_vehicles"] is not None:
        instance_data['num_vehicles'] = sidebar_settings["num_vehicles"]
        instance_data['vehicle_capacities'] = [instance_data['capacity']] * sidebar_settings["num_vehicles"]
    
    # Prepare and run single benchmark
    benchmark_settings = _prepare_single_benchmark_settings(sidebar_settings)
    st.session_state.results_df = run_single_benchmark(instance_data, benchmark_settings, sidebar_settings["bks_cost"])
    
    st.balloons()
    st.session_state.run_benchmark = False
    st.rerun()


# --- Bulk Benchmark Execution ---
if st.session_state.run_bulk:
    st.title("📦 Bulk Benchmark Execution")
    bulk_settings = _prepare_bulk_benchmark_settings(sidebar_settings)
    
    # Check if running in background
    if sidebar_settings.get("save_to_server", False):
        task_id = str(uuid4())[:8]
        run_background_task(run_bulk_benchmark_background, task_id, bulk_settings)
        
        st.success(f"✅ Benchmark started in background! Task ID: `{task_id}`")
        st.info("You can now close this page. The benchmark will continue running on the server.")
        st.markdown("Your results will be saved to the server when complete.")
        
        st.session_state.run_bulk = False
        st.rerun()
    else:
        run_bulk_benchmark(bulk_settings)


# --- Display Results ---
if st.session_state.results_df is not None:
    display_results(
        st.session_state.results_df,
        sidebar_settings["p_map"],
        sidebar_settings["sel_inst"],
        sidebar_settings["bks_cost"],
        sidebar_settings["time_limit"],
        st.session_state.all_histories
    )

