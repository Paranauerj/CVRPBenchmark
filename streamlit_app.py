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

# --- Plotting Function ---
def plot_routes(instance_data, routes, title="Routes"):
    """Generates a matplotlib figure for the VRP routes."""
    coords = instance_data.get('coordinates', {})
    
    if not coords:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No coordinates available in instance data.", ha='center')
        return fig

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Depot
    depot_id = instance_data.get('depot', 0)
    if depot_id in coords:
        depot_pos = coords[depot_id]
        ax.scatter(depot_pos[0], depot_pos[1], c='red', s=100, marker='s', label='Depot', zorder=10)
    
    # Plot Clients
    x_vals = []
    y_vals = []
    for idx, pos in coords.items():
        if idx != depot_id:
            x_vals.append(pos[0])
            y_vals.append(pos[1])
    ax.scatter(x_vals, y_vals, c='gray', s=10, alpha=0.5)

    # Plot Routes
    if routes:
        colors = cm.rainbow(np.linspace(0, 1, len(routes)))
        for route, color in zip(routes, colors):
            full_route = [depot_id] + route + [depot_id]
            route_x = []
            route_y = []
            for n in full_route:
                if n in coords:
                    route_x.append(coords[n][0])
                    route_y.append(coords[n][1])
            ax.plot(route_x, route_y, c=color, linewidth=1.5, alpha=0.8)
        
    ax.set_title(title)
    ax.legend()
    return fig

# --- Constants ---
FIRST_SOLUTIONS = {
    "Automatic": routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
    "Path Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "Path Most Constrained Arc": routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
    "Evaluator Strategy": routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY,
    "Savings (Clarke-Wright)": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    "Sweep": routing_enums_pb2.FirstSolutionStrategy.SWEEP,
    "Christofides": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    "All Unperformed": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
    "Best Insertion": routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
    "Parallel Cheapest Insertion": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    "Local Cheapest Insertion": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
    "Global Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    "Local Cheapest Arc": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
    "First Unbound Min Value": routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
}

METAHEURISTICS = {
    "Guided Local Search (GLS)": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "Tabu Search": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    "Simulated Annealing": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    "Greedy Descent": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    "Generic Tabu Search": routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
    "Automatic": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
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
    st.session_state.benchmark_results = []
    st.rerun()

if st.button("⏹️ Stop", width='stretch'):
    st.session_state.run_benchmark = False
    st.toast("Stopping...")

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
        if not st.session_state.run_benchmark: break
        stat.text(f"Running: {exp['name']}")
        
        costs, times, iters_list, best_routes = [], [], [], None
        best_cost_run = float('inf')

        for _ in range(exp["reps"]):
            if not st.session_state.run_benchmark: break
            cnt += 1
            prog.progress(cnt / total)
            
            cur_kw = exp["kwargs"].copy()
            cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
            
            res = execute_and_measure(exp["func"], instance_data, **cur_kw)
            
            if res["cpu_time"] is not None: times.append(res["cpu_time"])
            if res["objective_value"] is not None: 
                costs.append(res["objective_value"])
                # Track best route for plotting
                if res["objective_value"] < best_cost_run:
                    best_cost_run = res["objective_value"]
                    best_routes = res["routes"]
            if res["iterations"] is not None: iters_list.append(res["iterations"])

        if costs:
            best = min(costs)
            avg = statistics.mean(costs)
            row = {
                "Algorithm": exp["name"], "Best Cost": best, "Avg Cost": avg,
                "CPU Time (s)": statistics.mean(times) if times else None,
                "Iterations": int(statistics.mean(iters_list)) if iters_list else None,
                "Repetitions": exp["reps"],
                # Store extra data for plotting, but don't show in table
                "_routes": best_routes 
            }
            if bks_cost:
                row["Best Gap (%)"] = ((best - bks_cost)/bks_cost)*100.0
                row["Avg Gap (%)"] = ((avg - bks_cost)/bks_cost)*100.0
            
            results_list.append(row)
            st.session_state.benchmark_results = results_list

    st.session_state.results_df = pd.DataFrame(results_list)
    st.balloons()
    st.session_state.run_benchmark = False

if st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    # Display Table
    cols = ["Algorithm", "Best Cost", "Best Gap (%)", "Avg Cost", "Avg Gap (%)", 
            "CPU Time (s)", "Iterations", "Repetitions"]
    st.dataframe(df[cols].style.format({
        "Best Cost": "{:,.2f}", "Avg Cost": "{:,.2f}",
        "Best Gap (%)": "{:.4f}%", "Avg Gap (%)": "{:.4f}%",
        "CPU Time (s)": "{:.6f}", "Iterations": "{:d}"
    }, na_rep="N/A"), width='stretch')

    # Display Plots Side-by-Side
    st.subheader("Visualization (Best Run)")
    
    # Create columns for the plots (e.g., 2 columns)
    cols = st.columns(2)
    
    for i, row in df.iterrows():
        # Cycle through columns
        with cols[i % 2]:
            st.markdown(f"#### {row['Algorithm']}")
            st.caption(f"Cost: {row['Best Cost']:,.2f}")
            
            if row.get("_routes"):
                paths = p_map.get(sel_inst)
                with open(paths["vrp"], 'r') as f: 
                    inst_data_plot = instance_data_parser.load_vrp_instance(f)
                
                fig = plot_routes(inst_data_plot, row["_routes"], title="")
                st.pyplot(fig)
            else:
                st.warning("No solution routes available to plot.")