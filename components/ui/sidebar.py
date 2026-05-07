"""Simplified sidebar with only shared algorithm and limit parameters."""

import streamlit as st
import os
import glob
from ortools.constraint_solver import routing_enums_pb2


# Algorithm Constants
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
