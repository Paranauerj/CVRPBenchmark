import instance_data_parser
import streamlit as st
import pandas as pd
import baseline_solver
import gls_solver
import sa_solver
import ts_solver
import solution_parser
from benchmark_utils import execute_and_measure
import statistics
import glob
import os
import math
# --- plotting import removed ---

# --- Page Configuration ---
st.set_page_config(
    page_title="CVRP Benchmarker",
    layout="wide"
)

# --- Session State Initialization ---
if 'run_benchmark' not in st.session_state:
    st.session_state.run_benchmark = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- (find_instance_files function is unchanged) ---
@st.cache_data # Cache this so it only runs once
def find_instance_files(directory="instances"):
    """
    Scans a directory for .vrp files and finds their matching .sol files.
    Returns a list of valid base names and a map to their full paths.
    """
    vrp_pattern = os.path.join(directory, "*.vrp")
    vrp_files = sorted(glob.glob(vrp_pattern))
    
    valid_instance_names = []
    instance_path_map = {}
    
    for vrp_path in vrp_files:
        base_name = os.path.basename(vrp_path).replace(".vrp", "")
        sol_path = os.path.join(directory, base_name + ".sol")
        
        # Only add the instance if BOTH files exist
        if os.path.exists(sol_path):
            valid_instance_names.append(base_name)
            instance_path_map[base_name] = {
                "vrp": vrp_path,
                "sol": sol_path
            }
            
    return valid_instance_names, instance_path_map

# --- NEW: Plotting Function (REMOVED) ---

st.title("CVRP Solver Benchmarker 📊")
st.write("""
This app runs various CVRP solvers and compares their performance 
in terms of solution cost (BKS Gap) and execution time.
""")

# --- Algorithm Mapping ---
ALGORITHM_MAP = {
    "Baseline (C&W)": baseline_solver.solve_baseline,
    "Guided Local Search": gls_solver.solve_gls,
    "Simulated Annealing": sa_solver.solve_sa,
    "Tabu Search": ts_solver.solve_ts,
}

other_algorithms = list(ALGORITHM_MAP.keys())
other_algorithms.remove("Baseline (C&W)")


# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Benchmark Configuration")
    
    st.subheader("Experiment Settings")
    
    # --- UPDATED: Load and match files from "instances" directory ---
    instance_names, instance_path_map = find_instance_files("instances")

    if not instance_names:
        st.warning(
            "No matching `.vrp` / `.sol` pairs found in the 'instances' directory. "
            "Please create it and add files (e.g., `X-n641-k35.vrp` and `X-n641-k35.sol`)."
        )

    # --- UPDATED: Single selectbox for instance name ---
    selected_instance_name = st.selectbox(
        "Select Instance to Run:",
        options=instance_names
    )
    
    st.subheader("2. Select Algorithms")
    st.checkbox("Baseline (C&W)", value=True, disabled=True, help="The Baseline is required for relative time calculations.")
    
    algorithms_to_run_other = st.multiselect(
        "Choose additional algorithms to test:",
        options=other_algorithms,
        default=other_algorithms
    )
    
    algorithms_to_run = ["Baseline (C&W)"] + algorithms_to_run_other

    st.subheader("3. General Optional Parameters")
    st.write("These settings apply to all algorithms unless overridden below.")

    set_reps_general = st.checkbox("Set custom repetitions", value=False)
    reps_general = st.number_input(
        "Repetitions per Algorithm", min_value=1, value=5,
        help="Number of times each algorithm is run to average results."
    ) if set_reps_general else 1

    set_time_limit_general = st.checkbox("Set time limit (seconds)", value=False)
    time_general = st.number_input(
            "Time limit (seconds)", min_value=1, value=5, step=1
        ) if set_time_limit_general else None

    set_solution_limit_general = st.checkbox("Set solution limit (number of solutions)", value=False)
    solution_general = st.number_input(
        "Solution limit", min_value=1, value=1000, step=1
    ) if set_solution_limit_general else None
    
    st.subheader("4. Per-Algorithm Overrides")
    st.write("Set specific parameters for individual algorithms.")
    
    per_algo_overrides = {}
    
    # --- UPDATED: Loop over other_algorithms to hide baseline config ---
    for algo_name in algorithms_to_run_other:
        with st.expander(f"Overrides for {algo_name}"):
            st.write(f"Set parameters to use *only* for {algo_name}.")
            
            override_reps = st.checkbox(f"Override Repetitions##{algo_name}", value=False)
            rep_val = st.number_input(f"Repetitions##{algo_name}", min_value=1, value=reps_general, step=1) if override_reps else None

            override_time = st.checkbox(f"Override Time Limit##{algo_name}", value=False)
            time_val = st.number_input(f"Time limit (s)##{algo_name}", min_value=1, value=time_general or 5, step=1) if override_time else None
            
            override_solution = st.checkbox(f"Override Solution Limit##{algo_name}", value=False)
            solution_val = st.number_input(f"Solution limit##{algo_name}", min_value=1, value=solution_general or 1000, step=1) if override_solution else None
            
            per_algo_overrides[algo_name] = {
                "reps": rep_val,
                "time": time_val,
                "solution": solution_val
            }


# --- Main Page ---
st.header("Controls")
col1, col2 = st.columns(2)

# --- UPDATED: Disable button if no instances are found ---
files_selected = (selected_instance_name is not None)
if col1.button("🚀 Run Benchmark", type="primary", use_container_width=True, disabled=(not files_selected)):
    if not algorithms_to_run_other:
            st.toast("Warning: Running baseline only.", icon="⚠️")
    st.session_state.run_benchmark = True
    st.session_state.results_df = None # Clear old results
    st.rerun()

if col2.button("⏹️ Stop Benchmark", use_container_width=True):
    st.session_state.run_benchmark = False
    st.toast("Stopping benchmark...")

if not files_selected:
    st.warning("No instance files found. Please create an 'instances' folder and add matching .vrp and .sol files.")


# ... (rest of UI layout, sidebar, and controls are unchanged) ...
if st.session_state.run_benchmark:
    
    # --- UPDATED: Get file paths from map ---
    instance_paths = instance_path_map.get(selected_instance_name)

    if instance_paths is None:
        st.error(f"File paths not found for instance: {selected_instance_name}. Please check the 'instances' folder.")
        st.session_state.run_benchmark = False
    else:
        vrp_path = instance_paths["vrp"]
        sol_path = instance_paths["sol"]
        
        # --- UPDATED: Load data by opening file paths ---
        try:
            with open(vrp_path, 'r') as f:
                instance_data = instance_data_parser.load_vrp_instance(f)
            instance_name = selected_instance_name
            
            with open(sol_path, 'r') as f:
                bks_cost = solution_parser.parse_solution_file(f) # <--- UPDATED
            
        except Exception as e:
            st.error(f"Error parsing instance files: {vrp_path} or {sol_path}")
            st.exception(e)
            st.session_state.run_benchmark = False
            st.stop() # Stop execution
        
        st.header("Running Experiment...")

        # --- NEW: VALIDATION STEP ---
        # Check that every non-baseline algorithm has at least one limit
        validation_passed = True
        # --- UPDATED: Loop over other_algorithms to skip baseline ---
        for algo_name in algorithms_to_run_other:
            # Determine final limits for this algo
            time_limit = per_algo_overrides.get(algo_name, {}).get("time") or time_general
            solution_limit = per_algo_overrides.get(algo_name, {}).get("solution") or solution_general
            
            # If no limits are set, this is an error
            if time_limit is None and solution_limit is None:
                st.error(
                    f"**Validation Error:** Algorithm '{algo_name}' has no limits set. "
                    "Please set a general time/solution limit, or set an override for this algorithm, "
                    "to prevent a potentially infinite run."
                )
                st.session_state.run_benchmark = False
                validation_passed = False
                break # Stop checking other algorithms

        if not validation_passed:
            st.stop() # Stop the script execution if validation failed
        
        # --- END OF VALIDATION STEP ---
        
        st.info(f"Instance: **{instance_name}** | Best Known Solution (BKS) Cost: **{bks_cost}**")
        status_bar = st.container()
        results_list = [] 

        # --- B. Execution and Data Collection ---
        
        total_runs = 0
        reps_to_run = {}
        for algo_name in algorithms_to_run:
            reps = per_algo_overrides.get(algo_name, {}).get("reps") or reps_general
            reps_to_run[algo_name] = reps
            total_runs += reps

        progress_bar = st.progress(0.0)
        current_run = 0

        for algo_name in algorithms_to_run:
            if not st.session_state.run_benchmark:
                status_bar.warning("Benchmark stopped by user.")
                break
                
            algo_func = ALGORITHM_MAP[algo_name]
            cpu_times, objectives = [], []
            status_bar.text(f"Testing Algorithm: {algo_name}...")
            
            # --- REMOVED: "Best solution" tracking ---
            
            reps = reps_to_run[algo_name]
            
            # --- UPDATED: Special handling for baseline ---
            # This is the *single* definition of solver_kwargs,
            # defined *outside* the repetitions loop.
            if algo_name == "Baseline (C&W)":
                # Baseline is not configurable and should have no limits
                solver_kwargs = {}
            else:
                # Get final parameters for metaheuristics
                time_limit = per_algo_overrides.get(algo_name, {}).get("time") or time_general
                solution_limit = per_algo_overrides.get(algo_name, {}).get("solution") or solution_general
                
                solver_kwargs = {}
                if time_limit is not None:
                    solver_kwargs["time_limit_seconds"] = time_limit
                if solution_limit is not None:
                    solver_kwargs["solution_limit"] = solution_limit
            
            for i in range(reps):
                if not st.session_state.run_benchmark:
                    status_bar.warning("Benchmark stopped by user.")
                    break
                    
                current_run += 1
                progress_bar.progress(
                    current_run / total_runs, 
                    text=f"Running {algo_name} (Rep {i+1}/{reps})"
                )
                
                measurement = execute_and_measure(algo_func, instance_data, **solver_kwargs)
                cpu_times.append(measurement["cpu_time"])
                
                # --- UPDATED: Check for None ---
                if measurement["objective_value"] is not None:
                    objectives.append(measurement["objective_value"])
                # ---
                
            if not st.session_state.run_benchmark:
                break
            
            if cpu_times: 
                results_list.append({
                    "Algorithm": algo_name,
                    "Avg Cost": statistics.mean(objectives) if objectives else None,
                    # "Best Cost" and "Best Routes" removed
                    "CPU Time (s)": statistics.mean(cpu_times) if cpu_times else None,
                    "Repetitions": reps, 
                })
        
        # --- C. Metric Calculation & Reporting ---
        if results_list: 
            st.header("Benchmark Results")
            df = pd.DataFrame(results_list)
            
            # --- UPDATED: Calculate gap for Avg cost only ---
            df["Avg Cost Gap (%)"] = ((df["Avg Cost"] - bks_cost) / bks_cost) * 100.0
            
            try:
                baseline_time = df[df["Algorithm"] == "Baseline (C&W)"]["CPU Time (s)"].iloc[0]
                df["Time vs. Baseline"] = df["CPU Time (s)"] / baseline_time
                
                # --- UPDATED: New column order ---
                df = df[[
                    "Algorithm", "Avg Cost", "Avg Cost Gap (%)", 
                    "CPU Time (s)", "Time vs. Baseline", "Repetitions"
                ]]
                format_dict = {
                    "Avg Cost": "{:,.2f}",
                    "Avg Cost Gap (%)": "{:.4f}%",
                    "CPU Time (s)": "{:.6f}",
                    "Time vs. Baseline": "{:.4f}",
                    "Repetitions": "{:d}"
                }
            except (IndexError, KeyError):
                st.warning("Baseline (C&W) was not run. Cannot calculate 'Time vs. Baseline'.")
                df = df[["Algorithm", "Avg Cost", "Avg Cost Gap (%)", "CPU Time (s)", "Repetitions"]]
                format_dict = {
                    "Avg Cost": "{:,.2f}",
                    "Avg Cost Gap (%)": "{:.4f}%",
                    "CPU Time (s)": "{:.6f}",
                    "Repetitions": "{:d}"
                }
            
            st.session_state.results_df = df
            
            st.dataframe(
                df.style.format(format_dict),
                use_container_width=True
            )
            
            st.info(f"""
            **How to Read This Table:**
            * **Avg Cost:** The average solution cost across all repetitions.
            * **Avg Cost Gap (%):** Percent difference from the BKS ({bks_cost}).
            * **CPU Time (s):** The average CPU time per run.
            * **Time vs. Baseline:** How many times slower/faster the algorithm was compared to the Baseline.
            """)

            # --- NEW: Solution Comparison Section (REMOVED) ---
            
            with st.expander("Show Configuration"):
                st.subheader("General Settings Used")
                st.json({
                    "Instance": instance_name,
                    "BKS Cost": bks_cost,
                    "General Repetitions": reps_general,
                    "General Time Limit": time_general,
                    "General Solution Limit": solution_general
                })
                st.subheader("Per-Algorithm Overrides Applied")
                st.json(per_algo_overrides)
        
        if st.session_state.run_benchmark:
            st.balloons() 
        st.session_state.run_benchmark = False

# --- Show previous results if they exist and we are not running ---
elif st.session_state.results_df is not None:
    st.header("Last Benchmark Results")
    
    format_dict = {
        # "Cost" key removed, "Avg Cost" is used
        "CPU Time (s)": "{:.6f}",
        "Repetitions": "{:d}"
    }
    if "Avg Cost Gap (%)" in st.session_state.results_df.columns:
         format_dict["Avg Cost Gap (%)"] = "{:.4f}%"
    if "Avg Cost" in st.session_state.results_df.columns:
         format_dict["Avg Cost"] = "{:,.2f}"
    if "Time vs. Baseline" in st.session_state.results_df.columns:
         format_dict["Time vs. Baseline"] = "{:.4f}"

    st.dataframe(
        st.session_state.results_df.style.format(format_dict),
        use_container_width=True
    )