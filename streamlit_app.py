import instance_data_parser
import streamlit as st
import pandas as pd
import baseline_solver
import configurable_solver 
import solution_parser
from benchmark_utils import execute_and_measure
import statistics
import glob
import os
import math
import random
from ortools.constraint_solver import routing_enums_pb2

# --- Page Configuration ---
st.set_page_config(page_title="CVRP Benchmarker", layout="wide")

# --- Session State ---
if 'run_benchmark' not in st.session_state:
    st.session_state.run_benchmark = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- Constants ---
FIRST_SOLUTIONS = {
    "Path Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "Savings (Clarke-Wright)": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    "Parallel Cheapest Insertion": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    "Global Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    "Local Cheapest Insertion": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
    "Automatic": routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
}

METAHEURISTICS = {
    "Guided Local Search (GLS)": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "Tabu Search": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    "Simulated Annealing": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    "Greedy Descent": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
}

@st.cache_data
def find_instance_files(directory="instances"):
    if not os.path.exists(directory): return [], {}
    vrp_pattern = os.path.join(directory, "*.vrp")
    vrp_files = sorted(glob.glob(vrp_pattern))
    valid_names, path_map = [], {}
    for p in vrp_files:
        base = os.path.basename(p).replace(".vrp", "")
        sol = os.path.join(directory, base + ".sol")
        if os.path.exists(sol):
            valid_names.append(base)
            path_map[base] = {"vrp": p, "sol": sol}
    return valid_names, path_map

# --- Sidebar ---
with st.sidebar:
    st.header("1. Configuration")
    instance_names, instance_path_map = find_instance_files("instances")
    if not instance_names:
        st.warning("No .vrp/.sol files found in 'instances/' folder.")
    selected_instance_name = st.selectbox("Select Instance:", options=instance_names)
    
    st.subheader("2. Algorithm Composition")
    st.checkbox("Include Baseline (Pure Savings)", value=True, disabled=True)
    selected_first_sols = st.multiselect("First Solution Strategies", options=list(FIRST_SOLUTIONS.keys()), default=["Parallel Cheapest Insertion"])
    selected_metaheuristics = st.multiselect("Local Search Metaheuristics", options=list(METAHEURISTICS.keys()), default=["Guided Local Search (GLS)"])
    
    st.subheader("3. Limits & Stops")
    reps = st.number_input("Repetitions", 1, 20, 3)
    
    use_time = st.checkbox("Time Limit (s)", value=True)
    time_limit = st.number_input("Seconds", 1, 3600, 5) if use_time else None
    
    use_sol = st.checkbox("Solution Limit (count)", value=False)
    sol_limit = st.number_input("Count", 1, 100000, 2000) if use_sol else None
    
    use_lns = st.checkbox("LNS Time Limit (s)", value=False)
    lns_limit = st.number_input("LNS Seconds", 1, 100, 1) if use_lns else None
    
    use_gap = st.checkbox("Stop at Gap (%)", value=False)
    target_gap = st.number_input("Gap %", 0.0, 100.0, 1.0) if use_gap else None
    
    use_no_improv = st.checkbox("Stop at No Improvement (s)", value=False)
    no_improv_limit = st.number_input("No Improv Seconds", 1, 300, 5) if use_no_improv else None
    
    # --- NEW: No Improvement (Iterations) Input ---
    use_no_improv_iter = st.checkbox("Stop at No Improvement (iterations)", value=False,
        help="Stops if no better solution is found after N accepted neighbors (iterations).")
    no_improv_iter_limit = st.number_input(
        "No Improv Iterations Limit", min_value=100, value=1000, step=100
    ) if use_no_improv_iter else None
    # ---------------------------------------------

# --- Main Content ---
if selected_instance_name:
    st.title(f"Benchmark: {selected_instance_name} 📊")
else:
    st.title("CVRP Solver Benchmarker 📊")

col1, col2 = st.columns(2)
can_run = (selected_instance_name is not None) and (selected_first_sols and selected_metaheuristics)

if col1.button("🚀 Run Benchmark", type="primary", disabled=not can_run, width='stretch'):
    st.session_state.run_benchmark = True
    st.session_state.results_df = None
    st.rerun()

if col2.button("⏹️ Stop", width='stretch'):
    st.session_state.run_benchmark = False
    st.toast("Stopping...")

if st.session_state.run_benchmark:
    paths = instance_path_map.get(selected_instance_name)
    try:
        with open(paths["vrp"], 'r') as f: instance_data = instance_data_parser.load_vrp_instance(f)
        with open(paths["sol"], 'r') as f: bks_cost = solution_parser.parse_solution_file(f)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    st.info(f"Instance: **{selected_instance_name}** | BKS: **{bks_cost}**")
    
    target_cost_val = bks_cost * (1.0 + target_gap/100.0) if target_gap is not None else None

    experiments = []
    experiments.append({"name": "Baseline (C&W)", "func": baseline_solver.solve_baseline, "kwargs": {}, "reps": 1})
    
    for fs_name in selected_first_sols:
        for mh_name in selected_metaheuristics:
            algo_name = f"{mh_name.split('(')[0].strip()} [{fs_name.split('(')[0].strip()}]"
            kwargs = {
                "first_solution_strategy": FIRST_SOLUTIONS[fs_name],
                "local_search_metaheuristic": METAHEURISTICS[mh_name],
                "time_limit_seconds": time_limit,
                "solution_limit": sol_limit,
                "lns_time_limit_seconds": lns_limit,
                "target_cost": target_cost_val,
                "no_improvement_limit": no_improv_limit,
                "no_improvement_iterations_limit": no_improv_iter_limit # --- NEW ARGUMENT ---
            }
            experiments.append({"name": algo_name, "func": configurable_solver.solve_cvrp, "kwargs": kwargs, "reps": reps})

    # Validate Limits
    for exp in experiments:
        if exp["name"] == "Baseline (C&W)": continue
        k = exp["kwargs"]
        if all(v is None for v in [
            k.get("time_limit_seconds"), 
            k.get("solution_limit"), 
            k.get("target_cost"), 
            k.get("no_improvement_limit"),
            k.get("no_improvement_iterations_limit") # --- CHECK THIS ---
        ]):
            st.error(f"Error: Algorithm '{exp['name']}' has NO stopping limits set.")
            st.stop()

    results_list = []
    total_runs = sum(e["reps"] for e in experiments)
    progress_bar = st.progress(0.0)
    run_count = 0
    status_text = st.empty()

    for exp in experiments:
        if not st.session_state.run_benchmark: break
        status_text.text(f"Running: {exp['name']}")
        
        costs, times, mems = [], [], []
        
        for i in range(exp["reps"]):
            if not st.session_state.run_benchmark: break
            run_count += 1
            progress_bar.progress(run_count / total_runs)
            
            current_kwargs = exp["kwargs"].copy()
            if exp["name"] != "Baseline (C&W)":
                current_kwargs["random_seed"] = random.randint(0, 2**31 - 1)
            
            res = execute_and_measure(exp["func"], instance_data, **current_kwargs)
            
            if res["cpu_time"] is not None: times.append(res["cpu_time"])
            if res["objective_value"] is not None: costs.append(res["objective_value"])
            if res.get("memory_usage_mb") is not None: mems.append(res["memory_usage_mb"])
        
        if costs:
            best_cost = min(costs)
            avg_cost = statistics.mean(costs)
            
            row = {
                "Algorithm": exp["name"],
                "Best Cost": best_cost,
                "Avg Cost": avg_cost,
                "CPU Time (s)": statistics.mean(times) if times else None,
                "Memory (MB)": statistics.mean(mems) if mems else None,
                "Repetitions": exp["reps"]
            }
            if bks_cost:
                row["Best Gap (%)"] = ((best_cost - bks_cost)/bks_cost)*100.0
                row["Avg Gap (%)"] = ((avg_cost - bks_cost)/bks_cost)*100.0
            else:
                row["Best Gap (%)"] = None
                row["Avg Gap (%)"] = None
            results_list.append(row)

    if st.session_state.run_benchmark and results_list:
        st.session_state.results_df = pd.DataFrame(results_list)
        st.balloons()
        st.session_state.run_benchmark = False

if st.session_state.results_df is not None:
    df = st.session_state.results_df
    try:
        base_time = df[df["Algorithm"] == "Baseline (C&W)"]["CPU Time (s)"].iloc[0]
        df["Time vs. Baseline"] = df["CPU Time (s)"] / base_time
    except:
        df["Time vs. Baseline"] = None

    cols = ["Algorithm", "Best Cost", "Best Gap (%)", "Avg Cost", "Avg Gap (%)", 
            "CPU Time (s)", "Time vs. Baseline", "Memory (MB)", "Repetitions"]
    
    st.dataframe(
        df[cols].style.format({
            "Best Cost": "{:,.2f}", "Avg Cost": "{:,.2f}",
            "Best Gap (%)": "{:.4f}%", "Avg Gap (%)": "{:.4f}%",
            "CPU Time (s)": "{:.6f}", "Time vs. Baseline": "{:.4f}",
            "Memory (MB)": "{:.2f}"
        }, na_rep="N/A"),
        width='stretch'
    )