"""Simplified sidebar with only shared algorithm and limit parameters."""

import streamlit as st
import os
import glob
from ortools.constraint_solver import routing_enums_pb2


from components.constants import FIRST_SOLUTIONS, METAHEURISTICS


from components.utils.helpers import find_instance_files as _find_instance_files


def find_instance_files(directory="instances"):
    """Cached wrapper for find_instance_files."""
    if st.runtime.exists():
        # Apply caching only if running in Streamlit
        @st.cache_data
        def _cached_find(d):
            return _find_instance_files(d)
        return _cached_find(directory)
    return _find_instance_files(directory)


SOLVER_ENGINES = {
    "OR-Tools": "ortools",
    "HGS-CVRP": "hgs"
}

def render_shared_sidebar():
    """Render only the shared algorithm and limit parameters in sidebar."""
    with st.sidebar:
        st.header("⚙️ Solver Configuration")
        
        # Solver engine selection
        sel_engine = st.selectbox(
            "Solver Engine",
            options=list(SOLVER_ENGINES.keys()),
            index=0,
            help="Select the core solver algorithm"
        )
        engine_key = SOLVER_ENGINES[sel_engine]

        st.divider()
        st.header("⚙️ Algorithm Settings")
        
        # Algorithm selection - Only show FS/MH for OR-Tools
        sel_fs = []
        sel_mh = []
        if engine_key == "ortools":
            st.subheader("Algorithms")
            sel_fs = st.multiselect(
                "First Solution Strategy", 
                list(FIRST_SOLUTIONS.keys()), 
                ["Parallel Cheapest Insertion"],
                help="Initial solution method"
            )
            sel_mh = st.multiselect(
                "Metaheuristic", 
                list(METAHEURISTICS.keys()), 
                ["Guided Local Search (GLS)"],
                help="Local search improvement method"
            )
        else:
            st.info("HGS uses its own hybrid genetic search strategy (no FS/MH selection required).")
            # For HGS, we'll use a dummy FS/MH to keep the data structure consistent
            sel_fs = ["HGS Default"]
            sel_mh = ["Hybrid Genetic Search"]
        
        # Limits
        st.subheader("Execution Limits")
        time_limit = st.number_input(
            "Time Limit (seconds)", 
            min_value=1, 
            max_value=3600, 
            value=20,
            help="Maximum execution time per run"
        ) if st.checkbox("Enable Time Limit", value=True) else None
        
        sol_limit = None
        lns_limit = None
        if engine_key == "ortools":
            sol_limit = st.number_input(
                "Solution Limit", 
                min_value=1, 
                max_value=100000, 
                value=2000,
                help="Stop after finding N solutions"
            ) if st.checkbox("Enable Solution Limit", value=False) else None
            
            lns_limit = st.number_input(
                "LNS Time Limit (seconds)", 
                min_value=1, 
                max_value=100, 
                value=1,
                help="Time limit for Large Neighborhood Search"
            ) if st.checkbox("Enable LNS Limit", value=False) else None
        
        # Stopping conditions
        st.subheader("Stopping Conditions")
        
        no_improv = None
        no_improv_iter = None
        hgs_params = {}

        if engine_key == "ortools":
            no_improv = st.number_input(
                "No Improvement Timeout (seconds)", 
                min_value=1, 
                max_value=300, 
                value=60,
                help="Stop if no improvement for N seconds"
            ) if st.checkbox("Stop on No Improvement (Time)", value=False) else None
            
            no_improv_iter = st.number_input(
                "No Improvement Accepted Neighbors", 
                min_value=20, 
                max_value=10000, 
                value=100,
                help="Stop if no improvement for N accepted neighbors. An 'accepted neighbor' is a move in the local search that was accepted (improving or allowed by metaheuristic)."
            ) if st.checkbox("Stop on No Improvement (Neighbors)", value=False) else None

            continue_after_gap = st.checkbox(
                "Continue after gap reached",
                value=False,
                help="If enabled, OR-Tools will continue running until the time limit even if the target gap is reached. The time to reach the gap will be recorded."
            )
        else:
            continue_after_gap = False
            # HGS uses iterations without improvement
            no_improv_iter = st.number_input(
                "Iterations without improvement",
                min_value=1000,
                max_value=1000000,
                value=20000,
                step=1000,
                help="Stop if no improvement for N iterations"
            )

            st.divider()
            st.subheader("🧬 HGS Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                hgs_params['mu'] = st.number_input("mu (Min Population)", min_value=1, value=25, help="Minimum population size")
                hgs_params['lambda_'] = st.number_input("lambda (Offspring)", min_value=1, value=40, help="Number of individuals created each generation")
                hgs_params['nbElite'] = st.number_input("Elite Individuals", min_value=1, value=4, help="Number of elite individuals preserved")
            with col2:
                hgs_params['nbClose'] = st.number_input("Close Individuals", min_value=1, value=5, help="Number of closest individuals used for diversity")
                hgs_params['nbGranular'] = st.number_input("Granular Neighbors", min_value=1, value=20, help="Number of granular neighbors in local search")
                hgs_params['targetFeasible'] = st.number_input("Target Feasible", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Target proportion of feasible individuals")
            
            hgs_params['useSwapStar'] = st.checkbox("Use SWAP* Operator", value=True, help="Enable SWAP* local search operator")
    
    return {
        "engine": engine_key,
        "sel_fs": sel_fs,
        "sel_mh": sel_mh,
        "fs_enum": FIRST_SOLUTIONS,
        "mh_enum": METAHEURISTICS,
        "time_limit": time_limit,
        "sol_limit": sol_limit,
        "lns_limit": lns_limit,
        "no_improv": no_improv,
        "no_improv_iter": no_improv_iter,
        "continue_after_gap": continue_after_gap,
        "hgs_params": hgs_params,
    }
