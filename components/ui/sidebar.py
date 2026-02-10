"""Sidebar configuration and UI components."""

import streamlit as st
import os
import glob
import instance_data_parser
import solution_parser
from ortools.constraint_solver import routing_enums_pb2


# Constants
# Some first solutions need to pass a callback (like Sweep) - only available in C++
# https://github.com/google/or-tools/issues/2004#issuecomment-623913505
# https://github.com/google/or-tools/issues/3593#issuecomment-1347828378
# https://stackoverflow.com/questions/50137182/ortools-how-to-use-search-strategies-sweep-and-best-insertion
FIRST_SOLUTIONS = {
    "Automatic": routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
    "Path Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "Path Most Constrained Arc": routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
    #"Evaluator Strategy": routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY, # C++ only
    "Savings (Clarke-Wright)": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    #"Sweep": routing_enums_pb2.FirstSolutionStrategy.SWEEP, # C++ only
    "Christofides": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    #"All Unperformed": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED, # C++ only
    #"Best Insertion": routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION, # C++ only
    "Parallel Cheapest Insertion": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    "Local Cheapest Insertion": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
    #"Global Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC, # C++ only
    "Local Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
    "First Unbound Min Value": routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
}

METAHEURISTICS = {
    "Automatic": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    "Greedy Descent": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    "Guided Local Search (GLS)": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "Simulated Annealing": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    "Tabu Search": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    "Generic Tabu Search": routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
}


@st.cache_data
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


def render_sidebar():
    """Render the main configuration sidebar and return all settings."""
    with st.sidebar:
        st.header("Configuration")
        
        # Select instance source
        sources = get_instance_sources()
        if sources:
            sel_source = st.selectbox("Instance Source:", options=sources)
            source_dir = os.path.join("instances", sel_source)
        else:
            st.error("No instance sources found. Please ensure instances/uchoa or instances/gaetano exist.")
            sel_source = None
            source_dir = "instances"
        
        names, p_map = find_instance_files(source_dir) if sel_source else ([], {})
        
        if names:
            sel_inst = st.selectbox("Instance:", options=names)
        else:
            st.warning(f"No instances found in '{sel_source}' folder.")
            sel_inst = None
        
        # Load BKS and instance info
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
        
        # Display info
        if num_nodes is not None:
            st.write(f"**Nodes to Visit:** {num_nodes}")
        
        # Vehicle selector
        num_vehicles = None
        if min_vehicles is not None:
            num_vehicles = st.number_input(
                "Number of Vehicles",
                min_value=min_vehicles,
                value=min_vehicles,
                step=1,
                help=f"Minimum required: {min_vehicles}"
            )
        
        # Algorithm selection
        st.subheader("Algorithms")
        sel_fs = st.multiselect("First Solution", list(FIRST_SOLUTIONS.keys()), ["Parallel Cheapest Insertion"])
        sel_mh = st.multiselect("Metaheuristics", list(METAHEURISTICS.keys()), ["Guided Local Search (GLS)"])
        
        # Limits
        st.subheader("Limits")
        reps = st.number_input("Repetitions", 1, 20, 3)
        time_limit = st.number_input("Time (s)", 1, 3600, 5) if st.checkbox("Time Limit", True) else None
        sol_limit = st.number_input("Count", 1, 100000, 2000) if st.checkbox("Solution Limit", False) else None
        lns_limit = st.number_input("LNS (s)", 1, 100, 1) if st.checkbox("LNS Limit", False) else None
        
        # Gap option
        if bks_cost is not None:
            target_gap = st.number_input("Gap %", 0.0, 100.0, 1.0) if st.checkbox("Stop at Gap", False) else None
        else:
            target_gap = None
            if sel_inst:
                st.warning("Best Known Solution not available. Gap comparison disabled.")
        
        # No improvement options
        no_improv = st.number_input("No Improv (s)", 1, 300, 5) if st.checkbox("Stop No Improv (s)", False) else None
        no_improv_iter = st.number_input("No Improv Accepted Neighbors", 20, 10000, 100) if st.checkbox("Stop No Improv (Accepted Neighbors)", False) else None
        
        # Bulk Operations
        render_bulk_operations()
    
    return {
        "sel_source": sel_source,
        "sel_inst": sel_inst,
        "p_map": p_map,
        "bks_cost": bks_cost,
        "num_vehicles": num_vehicles,
        "sel_fs": sel_fs,
        "sel_mh": sel_mh,
        "reps": reps,
        "time_limit": time_limit,
        "sol_limit": sol_limit,
        "lns_limit": lns_limit,
        "target_gap": target_gap,
        "no_improv": no_improv,
        "no_improv_iter": no_improv_iter,
        "save_to_server": st.session_state.get('save_bulk_to_server', False)
    }


def render_bulk_operations():
    """Render the bulk operations UI."""
    st.markdown("---")
    st.subheader("Bulk Operations")

    if st.button("Bulk Benchmark (Gaetano)", help="Configure and run on Gaetano instances"):
        st.session_state.show_bulk_config = True

    if st.session_state.show_bulk_config:
        st.markdown("#### Select Instances")
        gaetano_dir = os.path.join("instances", "gaetano")
        
        if os.path.exists(gaetano_dir):
            g_names, _ = find_instance_files(gaetano_dir)
            
            if g_names:
                selected = st.multiselect(
                    "Choose instances to run:",
                    options=g_names,
                    default=g_names,
                    help="Remove instances you want to skip."
                )
                
                st.markdown("#### Options")
                save_to_server = st.checkbox(
                    "Run in Background (continue if page closes)",
                    value=False,
                    help="Run benchmark in the background. The benchmark will continue even if you close the page. Results will be saved to the server."
                )
                
                st.warning(f"You are about to run {len(selected)} instances. This may take significant time.")
                
                col_run, col_cancel = st.columns(2)
                
                if col_run.button("Run Selected", type="primary"):
                    st.session_state.selected_bulk_instances = selected
                    st.session_state.save_bulk_to_server = save_to_server
                    st.session_state.run_bulk = True
                    st.session_state.show_bulk_config = False
                    st.rerun()
                    
                if col_cancel.button("Cancel"):
                    st.session_state.show_bulk_config = False
                    st.rerun()
            else:
                st.error("No instances found in 'instances/gaetano'.")
        else:
            st.error(f"Folder not found: {gaetano_dir}")
