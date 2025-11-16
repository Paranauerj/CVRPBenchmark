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
    
    # --- UPDATED: Baseline is always selected and disabled ---
    st.checkbox("Baseline (C&W)", value=True, disabled=True, help="The Baseline is required for relative time calculations.")
    
    algorithms_to_run_other = st.multiselect(
        "Choose additional algorithms to test:",
        options=other_algorithms,
        default=other_algorithms
    )
    
    # Combine the (always-on) baseline with the user's other selections
    algorithms_to_run = ["Baseline (C&W)"] + algorithms_to_run_other

    # --- UPDATED: Moved Repetitions here ---
    st.subheader("3. Optional Parameters")
    st.write("Leave unchecked to use defaults (1 repetition, no limits).")

    set_reps = st.checkbox("Set custom repetitions", value=False)
    if set_reps:
        num_repetitions = st.number_input(
            "Repetitions per Algorithm",
            min_value=1,
            value=5,
            help="Number of times each algorithm is run to average results."
        )
    else:
        num_repetitions = 1 # Default to 1 if not set

    set_time_limit = st.checkbox("Set time limit (seconds)", value=False)
    if set_time_limit:
        ui_time_limit_seconds = st.number_input(
            "Time limit (seconds)", min_value=1, value=5, step=1
        )
    else:
        ui_time_limit_seconds = None

    set_solution_limit = st.checkbox("Set solution limit (number of solutions)", value=False)
    if set_solution_limit:
        ui_solution_limit = st.number_input(
            "Solution limit", min_value=1, value=1000, step=1
        )
    else:
        ui_solution_limit = None
        
    st.warning("""
    **Note:** If you don't set optional limits here, the app will use
    the hard-coded limits inside each solver file (e.g., `gls_solver.py`).
    """)


# --- Main Page ---
if st.button("🚀 Run Benchmark", type="primary"):
    
    # We know at least the baseline is selected, so no need to check for empty list
    
    # Load data
    instance_data = data_model.create_data_model()
    
    st.header("Running Experiment...")
    status_bar = st.container()
    results_list = [] # To store dicts for the DataFrame

    # --- B. Execution and Data Collection ---
    progress_bar = st.progress(0.0)
    total_runs = len(algorithms_to_run) * num_repetitions
    current_run = 0

    for algo_name in algorithms_to_run:
        algo_func = ALGORITHM_MAP[algo_name]
        cpu_times, objectives = [], []
        status_bar.text(f"Testing Algorithm: {algo_name}...")
        
        for i in range(num_repetitions):
            # Update progress
            current_run += 1
            progress_bar.progress(
                current_run / total_runs, 
                text=f"Running {algo_name} (Rep {i+1}/{num_repetitions})"
            )
            
            # Build solver_kwargs only with provided (non-None) values
            solver_kwargs = {}
            if ui_time_limit_seconds is not None:
                solver_kwargs["time_limit_seconds"] = ui_time_limit_seconds
            if ui_solution_limit is not None:
                solver_kwargs["solution_limit"] = ui_solution_limit
            
            # Pass kwargs to the execution function
            measurement = execute_and_measure(algo_func, instance_data, **solver_kwargs)
            cpu_times.append(measurement["cpu_time"])
            objectives.append(measurement["objective_value"])

        # Store the averages
        results_list.append({
            "Algorithm": algo_name,
            "Cost": statistics.mean(objectives),
            "Local CPU Time (s)": statistics.mean(cpu_times),
        })
    
    progress_bar.progress(1.0, text="Benchmark Complete!")
    st.header("Benchmark Results")

    # --- C. Metric Calculation & Reporting ---
    df = pd.DataFrame(results_list)
    
    # Get baseline time for relative calculations
    # We know baseline is in the list, so we can safely get its time
    baseline_time = df[df["Algorithm"] == "Baseline (C&W)"]["Local CPU Time (s)"].iloc[0]
    
    # --- REMOVED Normalized Time ---
    
    # 2. Time Relative to the Baseline
    df["Time vs. Baseline"] = df["Local CPU Time (s)"] / baseline_time
    
    # Reorder columns for display
    df = df[[
        "Algorithm", 
        "Cost", 
        "Local CPU Time (s)", 
        "Time vs. Baseline"
    ]]

    st.dataframe(
        df.style.format({
            "Cost": "{:,.2f}",
            "Local CPU Time (s)": "{:.6f}",
            "Time vs. Baseline": "{:.4f}",
        }),
        use_container_width=True
    )
    
    st.info(f"""
    **How to Read This Table:**
    * **Cost:** The final solution (total distance). **Lower is better.**
    * **Local CPU Time (s):** The actual CPU time taken by the algorithm.
    * **Time vs. Baseline:** How many times slower/faster the algorithm was compared to the simple Baseline (C&W).
    """)

    with st.expander("Show Configuration"):
        st.json({
            "Instance": instance_name,
            "Repetitions": num_repetitions,
            "Time Limit (override)": ui_time_limit_seconds,
            "Solution Limit (override)": ui_solution_limit
        })