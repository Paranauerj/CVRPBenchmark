import streamlit as st
import pandas as pd
import data_model 
import baseline_solver
import gls_solver
import sa_solver
import ts_solver
from benchmark_utils import execute_and_measure # Import shared function
import statistics

# --- Page Configuration ---
st.set_page_config(
    page_title="CVRP Benchmarker",
    layout="wide"
)

st.title("CVRP Solver Benchmarker 📊")
st.write("""
This app runs various CVRP solvers and compares their performance 
in terms of solution cost and execution time.
""")

# --- Algorithm Mapping ---
# Maps the string name to the actual solver function
ALGORITHM_MAP = {
    "Baseline (C&W)": baseline_solver.solve_baseline,
    "Guided Local Search": gls_solver.solve_gls,
    "Simulated Annealing": sa_solver.solve_sa,
    "Tabu Search": ts_solver.solve_ts,
}

# Get all algorithms EXCEPT the baseline for the multi-select
other_algorithms = list(ALGORITHM_MAP.keys())
other_algorithms.remove("Baseline (C&W)")


# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Benchmark Configuration")
    
    st.subheader("Experiment Settings")
    instance_name = st.selectbox(
        "Problem Instance", 
        ["P-n16-k8"], 
        help="Select the problem instance to solve."
    )
    
    st.subheader("2. Select Algorithms")
    
    st.checkbox("Baseline (C&W)", value=True, disabled=True, help="The Baseline is required for relative time calculations.")
    
    algorithms_to_run_other = st.multiselect(
        "Choose additional algorithms to test:",
        options=other_algorithms,
        default=other_algorithms
    )
    
    # Combine the (always-on) baseline with the user's other selections
    algorithms_to_run = ["Baseline (C&W)"] + algorithms_to_run_other

    # --- UPDATED: Renamed to "General" ---
    st.subheader("3. General Optional Parameters")
    st.write("These settings apply to all algorithms unless overridden below.")

    set_reps_general = st.checkbox("Set custom repetitions", value=False)
    if set_reps_general:
        reps_general = st.number_input(
            "Repetitions per Algorithm", min_value=1, value=5,
            help="Number of times each algorithm is run to average results."
        )
    else:
        reps_general = 1 # Default to 1 if not set

    # --- UPDATED: Default time limit checkbox is now True and value is 5 ---
    set_time_limit_general = st.checkbox("Set time limit (seconds)", value=True)
    if set_time_limit_general:
        time_general = st.number_input(
            "Time limit (seconds)", min_value=1, value=5, step=1
        )
    else:
        time_general = None

    set_solution_limit_general = st.checkbox("Set solution limit (number of solutions)", value=False)
    if set_solution_limit_general:
        solution_general = st.number_input(
            "Solution limit", min_value=1, value=1000, step=1
        )
    else:
        solution_general = None
    
    # --- NEW: Per-Algorithm Overrides ---
    st.subheader("4. Per-Algorithm Overrides")
    st.write("Set specific parameters for individual algorithms.")
    
    per_algo_overrides = {}
    
    for algo_name in algorithms_to_run:
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
if st.button("🚀 Run Benchmark", type="primary"):
    
    # Load data
    instance_data = data_model.create_data_model()
    
    st.header("Running Experiment...")
    status_bar = st.container()
    results_list = [] # To store dicts for the DataFrame

    # --- B. Execution and Data Collection ---
    
    # 1. Pre-calculate total runs for the progress bar
    total_runs = 0
    reps_to_run = {}
    for algo_name in algorithms_to_run:
        reps = per_algo_overrides.get(algo_name, {}).get("reps") or reps_general
        reps_to_run[algo_name] = reps
        total_runs += reps

    progress_bar = st.progress(0.0)
    current_run = 0

    # 2. Main execution loop
    for algo_name in algorithms_to_run:
        algo_func = ALGORITHM_MAP[algo_name]
        cpu_times, objectives = [], []
        status_bar.text(f"Testing Algorithm: {algo_name}...")
        
        # Get the final parameters based on priority
        reps = reps_to_run[algo_name]
        time_limit = per_algo_overrides.get(algo_name, {}).get("time") or time_general
        solution_limit = per_algo_overrides.get(algo_name, {}).get("solution") or solution_general
        
        for i in range(reps):
            # Update progress
            current_run += 1
            progress_bar.progress(
                current_run / total_runs, 
                text=f"Running {algo_name} (Rep {i+1}/{reps})"
            )
            
            # Build solver_kwargs only with provided (non-None) values
            solver_kwargs = {}
            if time_limit is not None:
                solver_kwargs["time_limit_seconds"] = time_limit
            if solution_limit is not None:
                solver_kwargs["solution_limit"] = solution_limit
            
            # Pass kwargs to the execution function
            measurement = execute_and_measure(algo_func, instance_data, **solver_kwargs)
            cpu_times.append(measurement["cpu_time"])
            objectives.append(measurement["objective_value"])

        # Store the averages
        results_list.append({
            "Algorithm": algo_name,
            "Cost": statistics.mean(objectives),
            "CPU Time (s)": statistics.mean(cpu_times), # <--- RENAMED
            "Repetitions": reps, # Also store reps in result
        })
    
    progress_bar.progress(1.0, text="Benchmark Complete!")
    st.header("Benchmark Results")

    # --- C. Metric Calculation & Reporting ---
    df = pd.DataFrame(results_list)
    
    # --- UPDATED: Using new column name ---
    baseline_time = df[df["Algorithm"] == "Baseline (C&W)"]["CPU Time (s)"].iloc[0]
    df["Time vs. Baseline"] = df["CPU Time (s)"] / baseline_time
    
    # Reorder columns for display
    df = df[[
        "Algorithm", 
        "Cost", 
        "CPU Time (s)", # <--- RENAMED
        "Time vs. Baseline",
        "Repetitions"
    ]]

    st.dataframe(
        df.style.format({
            "Cost": "{:,.2f}",
            "CPU Time (s)": "{:.6f}", # <--- RENAMED
            "Time vs. Baseline": "{:.4f}",
            "Repetitions": "{:d}"
        }),
        use_container_width=True
    )
    
    st.info(f"""
    **How to Read This Table:**
    * **Cost:** The final solution (total distance). **Lower is better.**
    * **CPU Time (s):** The actual CPU time taken by the algorithm. # <--- RENAMED
    * **Time vs. Baseline:** How many times slower/faster the algorithm was compared to the simple Baseline (C&W).
    """)

    with st.expander("Show Configuration"):
        # Show the final "general" settings
        st.subheader("General Settings Used")
        st.json({
            "Instance": instance_name,
            "General Repetitions": reps_general,
            "General Time Limit": time_general,
            "General Solution Limit": solution_general
        })
        # Show the overrides that were applied
        st.subheader("Per-Algorithm Overrides Applied")
        st.json(per_algo_overrides)