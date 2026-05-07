"""CVRP Benchmarker - Main Application."""

import streamlit as st
import os
import glob
from uuid import uuid4
from components.utils import instance_data_parser, solution_parser
from components.execution import configurable_solver
from components.ui.sidebar import render_shared_sidebar, FIRST_SOLUTIONS, METAHEURISTICS
from components.execution.single_benchmark import run_single_benchmark
from components.execution.bulk_benchmark import run_bulk_benchmark_background
from components.execution.background_task import run_background_task
from components.ui.results_display import display_results
from components.ui.monitoring import render_monitor_page


# --- Helper Functions ---
def _init_session_state():
    """Initialize session state variables."""
    defaults = {
        'results_df': None,
        'all_histories': {},
        'all_best_routes': {},
        'active_tab': 'Single'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def find_instance_files(directory="instances"):
    """Find all VRP instance files in a directory."""
    if not os.path.exists(directory):
        return [], {}
    vrp_files = sorted(glob.glob(os.path.join(directory, "*.vrp")))
    valid_names, path_map = [], {}
    for p in vrp_files:
        base = os.path.basename(p).replace(".vrp", "")
        sol = os.path.join(directory, base + ".sol")
        valid_names.append(base)
        path_map[base] = {"vrp": p, "sol": sol if os.path.exists(sol) else None}
    return valid_names, path_map


def get_instance_sources():
    """Get list of available instance sources (subdirectories in instances/)"""
    instances_dir = "instances"
    sources = []
    if os.path.exists(instances_dir):
        for item in sorted(os.listdir(instances_dir)):
            path = os.path.join(instances_dir, item)
            if os.path.isdir(path):
                sources.append(item)
    return sources


def _prepare_benchmark_settings(sidebar_settings, target_gap=None, reps=1):
    """Prepare common benchmark settings."""
    return {
        "engine": sidebar_settings.get("engine", "ortools"),
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
        "continue_after_gap": sidebar_settings.get("continue_after_gap", False),
        "hgs_params": sidebar_settings.get("hgs_params", {}),
        "target_gap": target_gap,
        "reps": reps,
    }


# --- Page Configuration ---
st.set_page_config(page_title="CVRP Benchmarker", layout="wide")

# --- Session State Initialization ---
_init_session_state()

# --- Main Layout ---
st.title("🚚 CVRP Benchmarker")

# Render shared sidebar (algorithms and limits only)
sidebar_settings = render_shared_sidebar()

# Create main tabs
tab_single, tab_bulk, tab_monitor = st.tabs(["🚀 Single Benchmark", "📊 Bulk Benchmark", "📈 Monitor"])

# ============ SINGLE BENCHMARK TAB ============
with tab_single:
    st.session_state.active_tab = 'Single'
    st.header("Single Instance Benchmark")
    
    # Initialize variables to avoid scoping issues
    sel_inst = None
    p_map = {}
    bks_cost = None
    min_vehicles = None
    num_nodes = None
    num_vehicles = None
    target_gap = None
    instance_data = None
    
    # Two-column layout for instance selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Instance source selector
        sources = get_instance_sources()
        sel_source: str | None = None
        
        if sources:
            sel_source = st.selectbox(
                "📁 Instance Source",
                options=sources,
                help="Select instance collection (Uchoa, Gaetano, etc.)"
            )
            assert sel_source is not None  # Type guard for Pylance
            source_dir = os.path.join("instances", sel_source)
            # Get instances from selected source
            names, p_map = find_instance_files(source_dir)
        else:
            st.error("No instance sources found. Please ensure instances/uchoa or instances/gaetano exist.")
            source_dir = "instances"
            names, p_map = [], {}
        
        # Instance dropdown
        if names:
            sel_inst = st.selectbox(
                "🎯 Instance",
                options=names,
                help="Select instance to benchmark"
            )
        else:
            st.warning(f"No instances found in '{sel_source or 'selected'}' folder.")
            sel_inst = None
    
    with col2:
        # Instance information
        bks_cost = None
        min_vehicles = None
        num_nodes = None
        
        if sel_inst and sel_inst in p_map:
            try:
                if p_map[sel_inst]["sol"]:
                    bks_cost = solution_parser.parse_solution_file(p_map[sel_inst]["sol"])
            except:
                bks_cost = None
            
            try:
                temp_instance = instance_data_parser.load_vrp_instance(p_map[sel_inst]["vrp"])
                min_vehicles = temp_instance.get("min_vehicles", 1)
                num_nodes = temp_instance.get("num_nodes", 0)
            except:
                min_vehicles = 1
                num_nodes = 0
        
        # Display instance info
        if sel_inst:
            # Customers is always total nodes - 1 (the depot)
            customers_count = (num_nodes - 1) if num_nodes else 0
            st.metric("📍 Customers", customers_count if customers_count else "?")
            if bks_cost:
                st.metric("🎖️ BKS Cost", f"{bks_cost:.0f}")
            else:
                st.info("ℹ️ No BKS available (gap comparison disabled)")
    
    # Configuration section
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_vehicles = None
        if min_vehicles is not None:
            num_vehicles = st.number_input(
                "🚛 Number of Vehicles",
                min_value=min_vehicles,
                value=min_vehicles,
                step=1,
                help=f"Minimum required: {min_vehicles}"
            )
    
    with col2:
        reps = st.number_input(
            "🔄 Repetitions",
            min_value=1,
            max_value=20,
            value=3,
            help="Number of times to run the algorithm"
        )
    
    with col3:
        # Gap target (if BKS available)
        target_gap = None
        if bks_cost is not None:
            target_gap = st.number_input(
                "🎯 Target Gap %",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help="Stop when this gap is reached"
            ) if st.checkbox("Enable Gap Target") else None
        else:
            st.text("Gap Target: N/A")
    
    # Run button
    if st.button(
        "🚀 Run Benchmark",
        type="primary",
        disabled=not (sel_inst and sidebar_settings["sel_fs"] and sidebar_settings["sel_mh"]),
        width='stretch'
    ):
        # Run synchronously (blocking) - Single benchmark is always sync
        with st.spinner("🔄 Running benchmark..."):
            instance_data = instance_data_parser.load_vrp_instance(p_map[sel_inst]["vrp"])
            
            # Prepare instance data with selected vehicle count
            if num_vehicles and num_vehicles != min_vehicles:
                instance_data['num_vehicles'] = num_vehicles
                instance_data['vehicle_capacities'] = [instance_data['capacity']] * num_vehicles
            
            benchmark_settings = _prepare_benchmark_settings(sidebar_settings, target_gap, reps)
            
            if sel_inst is not None:
                results_df, all_histories, all_best_routes = run_single_benchmark(instance_data, benchmark_settings, bks_cost, sel_inst)
                
                st.session_state.results_df = results_df
                st.session_state.all_histories = all_histories
                st.session_state.all_best_routes = all_best_routes
                st.session_state.instance_data = instance_data
                st.success("✅ Benchmark completed!")
    
    # Results display
    st.divider()
    if st.session_state.results_df is not None and sel_inst is not None:
        st.subheader("📊 Results")
        display_results(
            st.session_state.results_df, 
            p_map, 
            sel_inst, 
            bks_cost, 
            sidebar_settings["time_limit"],
            st.session_state.get('all_histories', {}),
            st.session_state.get('instance_data'),
            st.session_state.get('all_best_routes', {}),
            sidebar_settings=sidebar_settings
        )


# ============ BULK BENCHMARK TAB ============
with tab_bulk:
    st.session_state.active_tab = 'Bulk'
    st.header("Bulk Benchmark (Gaetano Instances)")
    
    # Instance selection (Gaetano only)
    gaetano_dir = os.path.join("instances", "gaetano")
    
    if os.path.exists(gaetano_dir):
        g_names, g_map = find_instance_files(gaetano_dir)
        
        if g_names:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_instances = st.multiselect(
                    "📁 Select Instances",
                    options=g_names,
                    default=g_names[:5],  # Default to first 5
                    help="Choose which instances to benchmark. Deselect to skip."
                )
            
            with col2:
                st.metric("📊 Count", len(selected_instances))
            
            # Configuration
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                reps_bulk = st.number_input(
                    "🔄 Repetitions (per instance)",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="Number of times to run each algorithm on each instance"
                )
            
            with col2:
                num_parallel = st.number_input(
                    "⚙️ Parallel Workers",
                    min_value=1,
                    max_value=16,
                    value=4,
                    help="Number of instances to process simultaneously"
                )
            
            # Run button
            if st.button(
                "🚀 Run Bulk Benchmark",
                type="primary",
                disabled=not (selected_instances and sidebar_settings["sel_fs"] and sidebar_settings["sel_mh"]),
                width='stretch'
            ):
                task_id = str(uuid4())[:8]
                task_name = f"Bulk - {len(selected_instances)} instances"
                
                bulk_settings = _prepare_benchmark_settings(sidebar_settings, None, reps_bulk)
                bulk_settings["selected_instances"] = selected_instances
                bulk_settings["num_parallel_instances"] = num_parallel
                
                run_background_task(
                    run_bulk_benchmark_background,
                    task_id,
                    task_name,
                    bulk_settings
                )
                
                st.success(f"✅ Bulk benchmark started in background! Task ID: `{task_id}`")
                st.info(f"📊 Processing {len(selected_instances)} instances with {num_parallel} parallel workers")
                st.info("📈 View progress in the **Monitor** tab")
        else:
            st.warning("No instances found in Gaetano directory.")
    else:
        st.error("Gaetano instances directory not found. Please ensure instances/gaetano exists.")


# ============ MONITOR TAB ============
with tab_monitor:
    st.session_state.active_tab = 'Monitor'
    render_monitor_page()
