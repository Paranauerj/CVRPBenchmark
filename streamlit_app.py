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
import io
from ortools.constraint_solver import routing_enums_pb2

# --- Helper Functions for Bulk Benchmark ---
def get_climate_from_filename(filename):
    """
    Extracts climate from Gaetano filenames like 'LDG95_3376_rain_95_0088'.
    Assumes format: Series_Depot_Climate_...
    """
    parts = filename.split('_')
    # Usually index 2 is climate (rain, fog, snow, none), but let's be safe
    valid_climates = {'rain', 'fog', 'snow', 'none'}
    for part in parts:
        if part.lower() in valid_climates:
            return part.lower()
    return "unknown"

def get_cost_at_time(history, time_limit_sec):
    """
    Finds the best cost found within the time_limit_sec based on history.
    History is a list of tuples: (time_elapsed, iterations, cost)
    """
    if not history:
        return None
    
    # History is strictly chronological. Find the last entry where t <= limit
    best_cost_at_t = None
    
    for t, iters, cost in history:
        if t <= time_limit_sec:
            best_cost_at_t = cost
        else:
            # Since history is sorted by time, we can stop early
            break
            
    # If the first solution took longer than time_limit_sec, return None
    return best_cost_at_t

# --- Page Configuration ---
st.set_page_config(page_title="CVRP Benchmarker", layout="wide")

if 'run_benchmark' not in st.session_state: st.session_state.run_benchmark = False
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'benchmark_results' not in st.session_state: st.session_state.benchmark_results = []
if 'all_histories' not in st.session_state: st.session_state.all_histories = {} # Store history for plotting
if 'run_bulk' not in st.session_state: st.session_state.run_bulk = False
if 'confirm_bulk' not in st.session_state: st.session_state.confirm_bulk = False
if 'show_bulk_config' not in st.session_state: st.session_state.show_bulk_config = False
if 'selected_bulk_instances' not in st.session_state: st.session_state.selected_bulk_instances = []

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
    
    has_data = False
    
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
            has_data = True
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
    
    if not has_data:
        ax.text(0.5, 0.5, "No convergence data available\n(No solutions found)", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
            
    ax.set_xlabel("Time (s)" if metric_type == "time" else "Iterations")
    ax.set_ylabel("Cost")
    ax.set_title(f"Convergence over {metric_type.capitalize()}")
    if has_data:
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

# --- Constants & Sidebar (Restored Full List) ---
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
    if not os.path.exists(directory): return [], {}
    vrp_files = sorted(glob.glob(os.path.join(directory, "*.vrp")))
    valid_names, path_map = [], {}
    for p in vrp_files:
        base = os.path.basename(p).replace(".vrp", "")
        sol = os.path.join(directory, base + ".sol")
        # Accept instances even without .sol file (for Gaetano's instances)
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

with st.sidebar:
    st.header("Configuration")
    
    # Select instance source (uchoa, gaetano, etc.)
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
        st.warning(f"⚠️ No instances found in '{sel_source}' folder.")
        sel_inst = None
    
    # Try to load BKS early to determine if gap option should be shown
    bks_cost = None
    min_vehicles = None
    num_nodes = None
    if sel_inst and sel_inst in p_map:
        try:
            if p_map[sel_inst]["sol"]:  # Only try to load if .sol file exists
                bks_cost = solution_parser.parse_solution_file(p_map[sel_inst]["sol"])
        except:
            bks_cost = None
        # Load instance to get minimum vehicles and num_nodes
        try:
            temp_instance = instance_data_parser.load_vrp_instance(p_map[sel_inst]["vrp"])
            min_vehicles = temp_instance.get("min_vehicles", 1)
            num_nodes = temp_instance.get("num_nodes", 0)
        except:
            min_vehicles = 1
            num_nodes = 0
    
    # Display number of nodes (read-only)
    if num_nodes is not None:
        st.write(f"**Nodes to Visit:** {num_nodes}")
    
    # Vehicle count selector
    num_vehicles = None
    if min_vehicles is not None:
        num_vehicles = st.number_input(
            "Number of Vehicles",
            min_value=min_vehicles,
            value=min_vehicles,
            step=1,
            help=f"Minimum required: {min_vehicles}"
        )
    
    st.subheader("Algorithms")
    sel_fs = st.multiselect("First Solution", list(FIRST_SOLUTIONS.keys()), ["Parallel Cheapest Insertion"])
    sel_mh = st.multiselect("Metaheuristics", list(METAHEURISTICS.keys()), ["Guided Local Search (GLS)"])
    st.subheader("Limits")
    reps = st.number_input("Repetitions", 1, 20, 3)
    time_limit = st.number_input("Time (s)", 1, 3600, 5) if st.checkbox("Time Limit", True) else None
    sol_limit = st.number_input("Count", 1, 100000, 2000) if st.checkbox("Solution Limit", False) else None
    lns_limit = st.number_input("LNS (s)", 1, 100, 1) if st.checkbox("LNS Limit", False) else None
    
    # Only show gap option if BKS is available
    if bks_cost is not None:
        target_gap = st.number_input("Gap %", 0.0, 100.0, 1.0) if st.checkbox("Stop at Gap", False) else None
    else:
        target_gap = None
        if sel_inst:
            st.warning("⚠️ Best Known Solution not available. Gap comparison disabled.")
    
    no_improv = st.number_input("No Improv (s)", 1, 300, 5) if st.checkbox("Stop No Improv (s)", False) else None
    no_improv_iter = st.number_input("No Improv Accepted Neighbors", 20, 10000, 100) if st.checkbox("Stop No Improv (Accepted Neighbors)", False) else None
    
    # --- Bulk Operations Section ---
    st.markdown("---")
    st.subheader("Bulk Operations")

    if st.button("📦 Bulk Benchmark (Gaetano)", help="Configure and run on Gaetano instances"):
        st.session_state.show_bulk_config = True

    # Configuration Block (Appears after button click)
    if st.session_state.show_bulk_config:
        st.markdown("#### Select Instances")
        gaetano_dir = os.path.join("instances", "gaetano")
        
        if os.path.exists(gaetano_dir):
            # 1. Find all available instances
            g_names, _ = find_instance_files(gaetano_dir)
            
            if g_names:
                # 2. Multiselect Widget (Default = All)
                selected = st.multiselect(
                    "Choose instances to run:",
                    options=g_names,
                    default=g_names,  # Pre-select all
                    help="Remove instances you want to skip."
                )
                
                st.warning(f"⚠️ You are about to run {len(selected)} instances. This may take significant time.")
                
                col_run, col_cancel = st.columns(2)
                
                # 3. Confirmation Buttons
                if col_run.button("✅ Run Selected", type="primary"):
                    st.session_state.selected_bulk_instances = selected
                    st.session_state.run_bulk = True
                    st.session_state.show_bulk_config = False  # Close config
                    st.rerun()
                    
                if col_cancel.button("❌ Cancel"):
                    st.session_state.show_bulk_config = False
                    st.rerun()
            else:
                st.error("No instances found in 'instances/gaetano'.")
        else:
            st.error(f"Folder not found: {gaetano_dir}")

st.title("CVRP Benchmarker 📊")
if st.button("🚀 Run", type="primary", disabled=not (sel_inst and sel_fs and sel_mh) or st.session_state.run_benchmark, width='stretch'):
    st.session_state.run_benchmark = True
    st.session_state.results_df = None
    st.session_state.all_histories = {} # Reset histories
    st.rerun()

if st.session_state.run_benchmark:
    paths = p_map.get(sel_inst)
    instance_data = instance_data_parser.load_vrp_instance(paths["vrp"])
    
    # Override num_vehicles with user selection
    if num_vehicles is not None:
        instance_data['num_vehicles'] = num_vehicles
        instance_data['vehicle_capacities'] = [instance_data['capacity']] * num_vehicles
    
    # bks_cost is already loaded from sidebar

    target_val = bks_cost * (1.0 + target_gap/100.0) if (target_gap and bks_cost) else None

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
            vehicles_used = len(best_routes) if best_routes else None
            row = {
                "Algorithm": exp["name"], "Best Cost": best, "Avg Cost": avg,
                "CPU Time (s)": statistics.mean(times) if times else None,
                "Vehicles Used": vehicles_used,
                "Accepted Neighbors": int(statistics.mean(iters_list)) if iters_list else None,
                "Repetitions": exp["reps"], "_routes": best_routes 
            }
            if bks_cost:
                row["Best Gap (%)"] = ((best - bks_cost)/bks_cost)*100.0
                row["Avg Gap (%)"] = ((avg - bks_cost)/bks_cost)*100.0
            results_list.append(row)
        else:
            # No solution found for any run
            row = {
                "Algorithm": exp["name"], "Best Cost": "No Solution", "Avg Cost": "No Solution",
                "CPU Time (s)": statistics.mean(times) if times else None,
                "Vehicles Used": None,
                "Accepted Neighbors": None,
                "Repetitions": exp["reps"], "_routes": None
            }
            if bks_cost:
                row["Best Gap (%)"] = "N/A"
                row["Avg Gap (%)"] = "N/A"
            results_list.append(row)

    st.session_state.results_df = pd.DataFrame(results_list)
    
    # Calculate best run solution gap (gap from global best cost found in this execution)
    global_best_cost = None
    for row in results_list:
        if isinstance(row["Best Cost"], (int, float)) and row["Best Cost"] < float('inf'):
            if global_best_cost is None or row["Best Cost"] < global_best_cost:
                global_best_cost = row["Best Cost"]
    
    # Add best run solution gap to each row
    if global_best_cost is not None:
        for i, row in enumerate(results_list):
            if isinstance(row["Best Cost"], (int, float)):
                row["Best Run Gap (%)"] = ((row["Best Cost"] - global_best_cost) / global_best_cost) * 100.0
            else:
                row["Best Run Gap (%)"] = "N/A"
        st.session_state.results_df = pd.DataFrame(results_list)
    
    st.balloons()
    st.session_state.run_benchmark = False
    st.rerun() # Rerun to show results

# --- Bulk Benchmark Execution ---
if st.session_state.run_bulk:
    st.title("📦 Bulk Benchmark Execution")
    
    # 1. Retrieve Selected Instances & Path Map
    target_instances = st.session_state.get('selected_bulk_instances', [])
    gaetano_dir = os.path.join("instances", "gaetano")
    
    # We need the path map to find the actual file paths for the selected names
    _, path_map = find_instance_files(gaetano_dir)

    if not target_instances:
        st.error("No instances selected for bulk run.")
        st.session_state.run_bulk = False
        st.stop()

    # 2. Prepare Algorithms (from Sidebar selection)
    bulk_experiments = []
    for fs in sel_fs:
        for mh in sel_mh:
            algo_name = f"{mh.split('(')[0].strip()} [{fs.split('(')[0].strip()}]"
            kw = {
                "first_solution_strategy": FIRST_SOLUTIONS[fs],
                "local_search_metaheuristic": METAHEURISTICS[mh],
                "time_limit_seconds": time_limit, 
                "solution_limit": sol_limit,
                "lns_time_limit_seconds": lns_limit,
                "target_cost": None,  # Cannot use gap target in bulk
                "no_improvement_limit": no_improv,
                "no_improvement_iterations_limit": no_improv_iter
            }
            bulk_experiments.append({"name": algo_name, "func": configurable_solver.solve_cvrp, "kwargs": kw})

    if not bulk_experiments:
        st.error("Please select at least one algorithm in the sidebar.")
        st.session_state.run_bulk = False
        st.stop()

    # 3. Run Loop
    bulk_results = []
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    total_steps = len(target_instances) * len(bulk_experiments) * reps
    current_step = 0
    
    # Time checkpoints for feature extraction
    time_checkpoints = [5, 10, 15, 20, 30, 60]

    for inst_name in target_instances:
        
        # Safety check if file still exists
        if inst_name not in path_map:
            continue
        
        # Load Instance Data
        p_info = path_map[inst_name]
        try:
            inst_data = instance_data_parser.load_vrp_instance(p_info["vrp"])
            # For bulk runs we keep the instance default number of vehicles
            # (do not override with the sidebar `num_vehicles`).
        except Exception as e:
            st.warning(f"❌ Skipping {inst_name}: {e}")
            continue

        # Extract Static Features
        climate = get_climate_from_filename(inst_name)
        n_customers = inst_data.get('num_nodes', 0) - 1  # Exclude depot
        n_vehicles = inst_data.get('num_vehicles', 0)
        capacity = inst_data.get('capacity', 0)
        
        # Load BKS if available
        bks_val = None
        if p_info["sol"]:
            try:
                bks_val = solution_parser.parse_solution_file(p_info["sol"])
            except: 
                pass

        for exp in bulk_experiments:
            algo_costs = []
            algo_times = []
            
            # Collect costs at each time checkpoint across repetitions
            checkpoint_collectors = {t: [] for t in time_checkpoints}

            for r in range(reps):
                current_step += 1
                progress_bar.progress(min(current_step / total_steps, 1.0))
                status_text.text(f"Processing: {inst_name} | {exp['name']} | Rep {r+1}/{reps}")
                
                # Randomize seed
                cur_kw = exp["kwargs"].copy()
                cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
                
                # Run Solver
                res = execute_and_measure(exp["func"], inst_data, **cur_kw)
                
                if res["objective_value"] is not None:
                    algo_costs.append(res["objective_value"])
                    algo_times.append(res["cpu_time"])
                    
                    # Extract history features
                    history = res.get("history", [])
                    for t_chk in time_checkpoints:
                        cost_at_t = get_cost_at_time(history, t_chk)
                        if cost_at_t is not None:
                            checkpoint_collectors[t_chk].append(cost_at_t)
                
            # Aggregate Results for this Algo on this Instance
            if algo_costs:
                avg_cost = statistics.mean(algo_costs)
                best_cost = min(algo_costs)
                avg_time = statistics.mean(algo_times)
                
                # Calculate Gaps
                best_gap = ((best_cost - bks_val) / bks_val * 100) if bks_val else None
                avg_gap = ((avg_cost - bks_val) / bks_val * 100) if bks_val else None
                
                row = {
                    "Instance": inst_name,
                    "Climate": climate,
                    "Customers": n_customers,
                    "Vehicles": n_vehicles,
                    "Capacity": capacity,
                    "Algorithm": exp["name"],
                    "Repetitions": reps,
                    "Avg Cost": avg_cost,
                    "Best Cost": best_cost,
                    "BKS Cost": bks_val,
                    "Best Gap (%)": best_gap,
                    "Avg Gap (%)": avg_gap,
                    "Avg CPU Time (s)": avg_time
                }
                
                # Add time checkpoints (Average cost at T across reps)
                for t_chk in time_checkpoints:
                    costs_at_t = checkpoint_collectors[t_chk]
                    if costs_at_t:
                        row[f"Avg Cost @ {t_chk}s"] = statistics.mean(costs_at_t)
                        row[f"Best Cost @ {t_chk}s"] = min(costs_at_t)
                    else:
                        row[f"Avg Cost @ {t_chk}s"] = None
                        row[f"Best Cost @ {t_chk}s"] = None
                        
                bulk_results.append(row)

    # 4. Finish and Export
    st.success("✅ Bulk Benchmark Complete!")
    st.session_state.run_bulk = False  # Reset state
    
    status_text.empty()
    progress_bar.empty()
    
    if bulk_results:
        df_bulk = pd.DataFrame(bulk_results)
        
        # Display Preview
        st.subheader("Results Preview")
        st.dataframe(df_bulk.head(10), use_container_width=True)
        
        # Excel Export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_bulk.to_excel(writer, sheet_name='Benchmark', index=False)
            
        st.download_button(
            label="📥 Download Results (Excel)",
            data=buffer.getvalue(),
            file_name="vrp_bulk_benchmark_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("⚠️ No results were generated. Check your configurations.")
    
    st.stop()

if st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    # 1. Table - conditionally show gap columns based on BKS availability
    base_cols = ["Algorithm", "Best Cost", "Avg Cost", "Best Run Gap (%)"]
    bks_gap_cols = ["Best Gap (%)", "Avg Gap (%)"]
    other_cols = ["CPU Time (s)", "Vehicles Used", "Accepted Neighbors", "Repetitions"]
    
    cols = base_cols.copy()
    
    # Only include BKS gap columns if BKS was available and not None
    if bks_cost is not None and "Best Gap (%)" in df.columns:
        cols.extend(bks_gap_cols)
    
    cols.extend(other_cols)
    
    # Ensure all columns exist in dataframe
    cols = [c for c in cols if c in df.columns]
    
    # Create a copy for display
    df_display = df[cols].copy()
    
    # Configure columns for proper sorting and display (keep as numbers, don't convert to strings)
    column_config = {}
    if "Best Run Gap (%)" in df_display.columns:
        column_config["Best Run Gap (%)"] = st.column_config.NumberColumn("Best Run Gap (%)", format="%.4f%%")
    if "Best Gap (%)" in df_display.columns:
        column_config["Best Gap (%)"] = st.column_config.NumberColumn("Best Gap (%)", format="%.4f%%")
    if "Avg Gap (%)" in df_display.columns:
        column_config["Avg Gap (%)"] = st.column_config.NumberColumn("Avg Gap (%)", format="%.4f%%")
    if "Best Cost" in df_display.columns:
        column_config["Best Cost"] = st.column_config.NumberColumn("Best Cost", format="%.2f")
    if "Avg Cost" in df_display.columns:
        column_config["Avg Cost"] = st.column_config.NumberColumn("Avg Cost", format="%.2f")
    if "CPU Time (s)" in df_display.columns:
        column_config["CPU Time (s)"] = st.column_config.NumberColumn("CPU Time (s)", format="%.6f")
    
    st.dataframe(df_display, width='stretch', column_config=column_config if column_config else None)

    # 2. Convergence Plots
    st.subheader("Convergence Analysis")
    
    # Calculate max time for stretching
    max_time_val = df["CPU Time (s)"].max() if not df.empty else 10
    if time_limit: max_time_val = max(max_time_val, time_limit)
    
    # Calculate max iterations for stretching
    max_iter_val = df["Accepted Neighbors"].max() if not df.empty else 100

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
            cost_str = f"(Cost: {row['Best Cost']:.2f})" if isinstance(row['Best Cost'], (int, float)) else "(No Solution)"
            st.markdown(f"**{row['Algorithm']}** {cost_str}")
            if row.get("_routes"):
                paths = p_map.get(sel_inst)
                inst_data = instance_data_parser.load_vrp_instance(paths["vrp"])
                fig = plot_routes(inst_data, row["_routes"], title="")
                st.pyplot(fig)
            else:
                st.warning("No solution found for this algorithm.")