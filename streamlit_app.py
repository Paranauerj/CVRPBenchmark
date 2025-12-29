import instance_data_parser
import streamlit as st
import pandas as pd
import configurable_solver 
import solution_parser
from benchmark_utils import execute_and_measure
import statistics
import glob
import os
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from ortools.constraint_solver import routing_enums_pb2

# --- Page Configuration ---
st.set_page_config(page_title="CVRP Benchmarker", layout="wide")

if 'run_benchmark' not in st.session_state: st.session_state.run_benchmark = False
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'benchmark_results' not in st.session_state: st.session_state.benchmark_results = []
if 'all_histories' not in st.session_state: st.session_state.all_histories = {} # Store history for plotting

# --- Plotting Functions ---
def plot_routes(instance_data, routes, title="Routes"):
    # (Same as before, omitted for brevity but preserved in full file)
    coords = instance_data.get('coordinates', {})
    if not coords:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No coordinates available", ha='center')
        return fig
    fig, ax = plt.subplots(figsize=(8, 6))
    depot_id = instance_data.get('depot', 0)
    if depot_id in coords:
        depot_pos = coords[depot_id]
        ax.scatter(depot_pos[0], depot_pos[1], c='red', s=100, marker='s', label='Depot', zorder=10)
    x_vals = [pos[0] for idx, pos in coords.items() if idx != depot_id]
    y_vals = [pos[1] for idx, pos in coords.items() if idx != depot_id]
    ax.scatter(x_vals, y_vals, c='gray', s=10, alpha=0.5)
    if routes:
        colors = cm.rainbow(np.linspace(0, 1, len(routes)))
        for route, color in zip(routes, colors):
            full_route = [depot_id] + route + [depot_id]
            route_x = [coords[n][0] for n in full_route if n in coords]
            route_y = [coords[n][1] for n in full_route if n in coords]
            ax.plot(route_x, route_y, c=color, linewidth=1.5, alpha=0.8)
    ax.set_title(title)
    ax.legend()
    return fig

def plot_convergence(histories_dict, metric_type="time", max_val=None):
    """
    Plots convergence curves.
    metric_type: "time" or "iterations"
    max_val: The maximum x-axis value (time or iterations) to stretch lines to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo_name, runs in histories_dict.items():
        # Pick the "best" run (lowest final cost) to plot, or average them?
        # Plotting the best run is usually cleaner for convergence analysis.
        best_run_idx = -1
        best_run_final_cost = float('inf')
        
        for idx, history in enumerate(runs):
            if history and history[-1][2] < best_run_final_cost:
                best_run_final_cost = history[-1][2]
                best_run_idx = idx
                
        if best_run_idx != -1:
            history = runs[best_run_idx]
            
            # Prepare data
            x_data = []
            y_data = []
            
            # (time, iterations, cost)
            for pt in history:
                if metric_type == "time":
                    x_data.append(pt[0])
                else:
                    x_data.append(pt[1])
                y_data.append(pt[2])
            
            # Stretch logic
            if max_val is not None and x_data[-1] < max_val:
                x_data.append(max_val)
                y_data.append(y_data[-1]) # Repeat last cost
                
            ax.step(x_data, y_data, where='post', label=algo_name, linewidth=2)
            
    ax.set_xlabel("Time (s)" if metric_type == "time" else "Iterations")
    ax.set_ylabel("Cost")
    ax.set_title(f"Convergence over {metric_type.capitalize()}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

# --- Constants & Sidebar (Restored Full List) ---
# Some first solutions need to pass a callback (like Sweep) - only available in C++
# https://github.com/google/or-tools/issues/2004#issuecomment-623913505
# https://github.com/google/or-tools/issues/3593#issuecomment-1347828378
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
    if not os.path.exists(directory): return [], {}
    vrp_files = sorted(glob.glob(os.path.join(directory, "*.vrp")))
    valid_names, path_map = [], {}
    for p in vrp_files:
        base = os.path.basename(p).replace(".vrp", "")
        sol = os.path.join(directory, base + ".sol")
        if os.path.exists(sol):
            valid_names.append(base)
            path_map[base] = {"vrp": p, "sol": sol}
    return valid_names, path_map

with st.sidebar:
    st.header("Configuration")
    names, p_map = find_instance_files("instances")
    sel_inst = st.selectbox("Instance:", options=names) if names else None
    st.subheader("Algorithms")
    sel_fs = st.multiselect("First Solution", list(FIRST_SOLUTIONS.keys()), ["Parallel Cheapest Insertion"])
    sel_mh = st.multiselect("Metaheuristics", list(METAHEURISTICS.keys()), ["Guided Local Search (GLS)"])
    st.subheader("Limits")
    reps = st.number_input("Repetitions", 1, 20, 3)
    time_limit = st.number_input("Time (s)", 1, 3600, 5) if st.checkbox("Time Limit", True) else None
    sol_limit = st.number_input("Count", 1, 100000, 2000) if st.checkbox("Solution Limit", False) else None
    lns_limit = st.number_input("LNS (s)", 1, 100, 1) if st.checkbox("LNS Limit", False) else None
    target_gap = st.number_input("Gap %", 0.0, 100.0, 1.0) if st.checkbox("Stop at Gap", False) else None
    no_improv = st.number_input("No Improv (s)", 1, 300, 5) if st.checkbox("Stop No Improv (s)", False) else None
    no_improv_iter = st.number_input("No Improv Iterations", 100, 10000, 100) if st.checkbox("Stop No Improv (Iter)", False) else None

st.title("CVRP Benchmarker 📊")
if st.button("🚀 Run", type="primary", disabled=not (sel_inst and sel_fs and sel_mh), width='stretch'):
    st.session_state.run_benchmark = True
    st.session_state.results_df = None
    st.session_state.all_histories = {} # Reset histories
    st.rerun()

if st.session_state.run_benchmark:
    paths = p_map.get(sel_inst)
    with open(paths["vrp"], 'r') as f: instance_data = instance_data_parser.load_vrp_instance(f)
    with open(paths["sol"], 'r') as f: bks_cost = solution_parser.parse_solution_file(f)

    target_val = bks_cost * (1.0 + target_gap/100.0) if target_gap else None

    experiments = []
    for fs in sel_fs:
        for mh in sel_mh:
            algo = f"{mh.split('(')[0].strip()} [{fs.split('(')[0].strip()}]"
            kw = {
                "first_solution_strategy": FIRST_SOLUTIONS[fs],
                "local_search_metaheuristic": METAHEURISTICS[mh],
                "time_limit_seconds": time_limit, "solution_limit": sol_limit,
                "lns_time_limit_seconds": lns_limit, "target_cost": target_val,
                "no_improvement_limit": no_improv, "no_improvement_iterations_limit": no_improv_iter
            }
            experiments.append({"name": algo, "func": configurable_solver.solve_cvrp, "kwargs": kw, "reps": reps})

    results_list = []
    total = sum(e["reps"] for e in experiments)
    prog = st.progress(0.0)
    cnt = 0
    stat = st.empty()

    for exp in experiments:
        stat.text(f"Running: {exp['name']}")
        costs, times, iters_list, best_routes = [], [], [], None
        best_cost_run = float('inf')
        
        # Store all histories for this algorithm
        exp_histories = []

        for _ in range(exp["reps"]):
            cnt += 1
            prog.progress(cnt / total)
            cur_kw = exp["kwargs"].copy()
            cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
            
            res = execute_and_measure(exp["func"], instance_data, **cur_kw)
            
            if res["cpu_time"] is not None: times.append(res["cpu_time"])
            if res["objective_value"] is not None: 
                costs.append(res["objective_value"])
                if res["objective_value"] < best_cost_run:
                    best_cost_run = res["objective_value"]
                    best_routes = res["routes"]
            if res["iterations"] is not None: iters_list.append(res["iterations"])
            
            # --- Capture History ---
            if res.get("history"):
                exp_histories.append(res["history"])
        
        # Save histories to session state
        st.session_state.all_histories[exp["name"]] = exp_histories

        if costs:
            best = min(costs)
            avg = statistics.mean(costs)
            row = {
                "Algorithm": exp["name"], "Best Cost": best, "Avg Cost": avg,
                "CPU Time (s)": statistics.mean(times) if times else None,
                "Iterations": int(statistics.mean(iters_list)) if iters_list else None,
                "Repetitions": exp["reps"], "_routes": best_routes 
            }
            if bks_cost:
                row["Best Gap (%)"] = ((best - bks_cost)/bks_cost)*100.0
                row["Avg Gap (%)"] = ((avg - bks_cost)/bks_cost)*100.0
            results_list.append(row)

    st.session_state.results_df = pd.DataFrame(results_list)
    st.balloons()
    st.session_state.run_benchmark = False
    st.rerun() # Rerun to show results

if st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    # 1. Table
    cols = ["Algorithm", "Best Cost", "Best Gap (%)", "Avg Cost", "Avg Gap (%)", "CPU Time (s)", "Iterations", "Repetitions"]
    st.dataframe(df[cols].style.format({
        "Best Cost": "{:,.2f}", "Avg Cost": "{:,.2f}", "Best Gap (%)": "{:.4f}%", "Avg Gap (%)": "{:.4f}%",
        "CPU Time (s)": "{:.6f}", "Iterations": "{:d}"
    }, na_rep="N/A"), width='stretch')

    # 2. Convergence Plots
    st.subheader("Convergence Analysis")
    
    # Calculate max time for stretching
    max_time_val = df["CPU Time (s)"].max() if not df.empty else 10
    if time_limit: max_time_val = max(max_time_val, time_limit)
    
    # Calculate max iterations for stretching
    max_iter_val = df["Iterations"].max() if not df.empty else 100

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Cost vs. Time")
        fig_time = plot_convergence(st.session_state.all_histories, "time", max_val=max_time_val)
        st.pyplot(fig_time)
        
    with col2:
        st.markdown("#### Cost vs. Iterations")
        fig_iter = plot_convergence(st.session_state.all_histories, "iterations", max_val=max_iter_val)
        st.pyplot(fig_iter)

    # 3. Route Plots
    st.subheader("Route Visualization (Best Run)")
    cols = st.columns(2)
    for i, row in df.iterrows():
        with cols[i % 2]:
            # Updated Title to include Cost
            st.markdown(f"**{row['Algorithm']}** (Cost: {row['Best Cost']:.2f})")
            if row.get("_routes"):
                paths = p_map.get(sel_inst)
                with open(paths["vrp"], 'r') as f: inst_data = instance_data_parser.load_vrp_instance(f)
                fig = plot_routes(inst_data, row["_routes"], title="")
                st.pyplot(fig)
            else:
                st.warning("No routes.")