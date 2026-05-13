"""
CVRP Benchmarker - Main Application.
A professional tool for benchmarking VRP algorithms across standard and custom instances.
"""

import streamlit as st
import os
from uuid import uuid4
from typing import Dict, Any, Tuple, Optional

# Components and Utilities
from components.utils import instance_data_parser, solution_parser
from components.execution import configurable_solver
from components.ui.sidebar import render_shared_sidebar
from components.constants import FIRST_SOLUTIONS, METAHEURISTICS
from components.execution.single_benchmark import run_single_benchmark
from components.execution.bulk_benchmark import run_bulk_benchmark_background
from components.execution.background_task import run_background_task
from components.ui.results_display import display_results
from components.ui.monitoring import render_monitor_page
from components.utils.helpers import find_instance_files


def _init_session_state() -> None:
    """Initialize session state variables if running in Streamlit runtime."""
    if not st.runtime.exists():
        return
        
    defaults = {
        'results_df': None,
        'all_histories': {},
        'all_best_routes': {},
        'active_tab': 'Single',
        'instance_data': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_instance_sources() -> list[str]:
    """Get list of available instance sources (subdirectories in instances/)"""
    instances_dir = "instances"
    if not os.path.exists(instances_dir):
        return []
    return [item for item in sorted(os.listdir(instances_dir)) 
            if os.path.isdir(os.path.join(instances_dir, item))]


def _prepare_benchmark_settings(sidebar_settings: Dict[str, Any], 
                               target_gap: Optional[float] = None, 
                               reps: int = 1) -> Dict[str, Any]:
    """Prepare a standardized settings dictionary for benchmark execution."""
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


def render_single_benchmark_tab(sidebar_settings: Dict[str, Any]) -> None:
    """Logic and UI for the Single Instance Benchmark tab."""
    st.header("Single Instance Benchmark")
    
    # 1. Instance Selection
    col1, col2 = st.columns(2)
    with col1:
        sources = get_instance_sources()
        if not sources:
            st.error("No instance sources found. Please ensure instances/ folder exists.")
            return

        sel_source = st.selectbox("📁 Instance Source", options=sources)
        source_dir = os.path.join("instances", sel_source)
        names, path_map = find_instance_files(source_dir)
        
        sel_inst = st.selectbox("🎯 Instance", options=names) if names else None
        if not names:
            st.warning(f"No instances found in '{sel_source}' folder.")

    # 2. Instance Metadata & BKS
    bks_cost, min_vehicles, num_nodes = None, 1, 0
    if sel_inst and sel_inst in path_map:
        paths = path_map[sel_inst]
        if paths["sol"]:
            try: bks_cost = solution_parser.parse_solution_file(paths["sol"])
            except: bks_cost = None
        
        try:
            instance_data = instance_data_parser.load_vrp_instance(paths["vrp"])
            min_vehicles = instance_data.get("min_vehicles", 1)
            num_nodes = instance_data.get("num_nodes", 0)
        except Exception as e:
            st.error(f"Error loading instance: {e}")
            return

    with col2:
        if sel_inst:
            cust_count = (num_nodes - 1) if num_nodes > 0 else 0
            st.metric("📍 Customers", cust_count)
            if bks_cost: st.metric("🎖️ BKS Cost", f"{bks_cost:.0f}")
            else: st.info("ℹ️ No BKS available")

    # 3. Local Configuration
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        num_vehicles = st.number_input("🚛 Number of Available Vehicles", min_value=min_vehicles, value=min_vehicles)
    with c2:
        reps = st.number_input("🔄 Repetitions", min_value=1, max_value=20, value=3)
    with c3:
        target_gap = None
        if bks_cost:
            target_gap = st.number_input("🎯 Target Gap %", 0.0, 100.0, 1.0, 0.1) if st.checkbox("Enable Target") else None
        else:
            st.text("Gap Target: N/A")

    # 4. Execution
    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True,
                 disabled=not (sel_inst and sidebar_settings["sel_fs"] and sidebar_settings["sel_mh"])):
        with st.spinner("🔄 Running benchmark..."):
            instance_data = instance_data_parser.load_vrp_instance(path_map[sel_inst]["vrp"])
            if num_vehicles != min_vehicles:
                instance_data['num_vehicles'] = num_vehicles
                instance_data['vehicle_capacities'] = [instance_data['capacity']] * num_vehicles
            
            settings = _prepare_benchmark_settings(sidebar_settings, target_gap, reps)
            res_df, histories, routes = run_single_benchmark(instance_data, settings, bks_cost, sel_inst)
            
            st.session_state.update({
                'results_df': res_df, 'all_histories': histories,
                'all_best_routes': routes, 'instance_data': instance_data
            })
            st.success("✅ Benchmark completed!")

    # 5. Results
    if st.session_state.results_df is not None and sel_inst:
        st.divider()
        display_results(
            st.session_state.results_df, path_map, sel_inst, bks_cost, 
            sidebar_settings["time_limit"], st.session_state.all_histories,
            st.session_state.instance_data, st.session_state.all_best_routes,
            sidebar_settings=sidebar_settings
        )


def render_bulk_benchmark_tab(sidebar_settings: Dict[str, Any]) -> None:
    """Logic and UI for the Bulk Benchmark tab."""
    st.header("Bulk Benchmark (Gaetano Instances)")
    
    gaetano_dir = os.path.join("instances", "gaetano")
    if not os.path.exists(gaetano_dir):
        st.error("Gaetano instances directory not found.")
        return

    g_names, _ = find_instance_files(gaetano_dir)
    if not g_names:
        st.warning("No instances found in Gaetano directory.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.multiselect("📁 Select Instances", options=g_names, default=g_names[:5])
    with col2:
        st.metric("📊 Count", len(selected))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        reps = st.number_input("🔄 Repetitions", 1, 20, 1)
    with c2:
        workers = st.number_input("⚙️ Parallel Workers", 1, 16, 4)

    if st.button("🚀 Run Bulk Benchmark", type="primary", use_container_width=True,
                 disabled=not (selected and sidebar_settings["sel_fs"] and sidebar_settings["sel_mh"])):
        task_id = str(uuid4())[:8]
        settings = _prepare_benchmark_settings(sidebar_settings, None, reps)
        settings.update({"selected_instances": selected, "num_parallel_instances": workers})
        
        run_background_task(run_bulk_benchmark_background, task_id, f"Bulk - {len(selected)} instances", settings)
        
        st.success(f"✅ Started! Task ID: `{task_id}`")
        st.info(f"📊 Processing {len(selected)} instances with {workers} workers. View progress in **Monitor**.")


def main() -> None:
    """Main application entry point."""
    if not st.runtime.exists():
        return

    st.set_page_config(page_title="CVRP Benchmarker", layout="wide", page_icon="🚚")
    _init_session_state()

    st.title("🚚 CVRP Benchmarker")
    sidebar_settings = render_shared_sidebar()

    tab_single, tab_bulk, tab_monitor = st.tabs(["🚀 Single Benchmark", "📊 Bulk Benchmark", "📈 Monitor"])

    with tab_single:
        st.session_state.active_tab = 'Single'
        render_single_benchmark_tab(sidebar_settings)

    with tab_bulk:
        st.session_state.active_tab = 'Bulk'
        render_bulk_benchmark_tab(sidebar_settings)

    with tab_monitor:
        st.session_state.active_tab = 'Monitor'
        render_monitor_page()


if __name__ == "__main__":
    main()
