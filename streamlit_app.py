"""CVRP Benchmarker - Main Application."""

import streamlit as st
import instance_data_parser
import configurable_solver

from components.ui.sidebar import render_sidebar, FIRST_SOLUTIONS, METAHEURISTICS
from components.execution.single_benchmark import run_single_benchmark
from components.execution.bulk_benchmark import run_bulk_benchmark
from components.visualization.results_display import display_results

# --- Page Configuration ---
st.set_page_config(page_title="CVRP Benchmarker", layout="wide")

# --- Session State Initialization ---
if 'run_benchmark' not in st.session_state:
    st.session_state.run_benchmark = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'all_histories' not in st.session_state:
    st.session_state.all_histories = {}
if 'run_bulk' not in st.session_state:
    st.session_state.run_bulk = False
if 'show_bulk_config' not in st.session_state:
    st.session_state.show_bulk_config = False
if 'selected_bulk_instances' not in st.session_state:
    st.session_state.selected_bulk_instances = []


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
    
    # Calculate target value
    target_val = None
    if sidebar_settings["target_gap"] and sidebar_settings["bks_cost"]:
        target_val = sidebar_settings["bks_cost"] * (1.0 + sidebar_settings["target_gap"] / 100.0)
    
    # Prepare settings dict for single benchmark
    benchmark_settings = {
        "sel_fs": sidebar_settings["sel_fs"],
        "sel_mh": sidebar_settings["sel_mh"],
        "fs_enum": FIRST_SOLUTIONS,
        "mh_enum": METAHEURISTICS,
        "solver_func": configurable_solver.solve_cvrp,
        "time_limit": sidebar_settings["time_limit"],
        "sol_limit": sidebar_settings["sol_limit"],
        "lns_limit": sidebar_settings["lns_limit"],
        "target_val": target_val,
        "no_improv": sidebar_settings["no_improv"],
        "no_improv_iter": sidebar_settings["no_improv_iter"],
        "reps": sidebar_settings["reps"]
    }
    
    # Run single benchmark
    st.session_state.results_df = run_single_benchmark(instance_data, benchmark_settings, sidebar_settings["bks_cost"])
    
    st.balloons()
    st.session_state.run_benchmark = False
    st.rerun()


# --- Bulk Benchmark Execution ---
if st.session_state.run_bulk:
    st.title("📦 Bulk Benchmark Execution")
    
    # Prepare settings dict for bulk benchmark
    bulk_settings = {
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
        "save_to_server": sidebar_settings.get("save_to_server", False)
    }
    
    # Run bulk benchmark
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
