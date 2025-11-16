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

st.title("Hardware-Agnostic CVRP Benchmarker 📊")
st.write("""
This app runs CVRP solvers and normalizes their execution time using PassMark 
Single Thread scores to allow for fair, hardware-agnostic comparisons.
""")

# --- Algorithm Mapping ---
# Maps the string name to the actual solver function
ALGORITHM_MAP = {
    "Baseline (C&W)": baseline_solver.solve_baseline,
    "Guided Local Search": gls_solver.solve_gls,
    "Simulated Annealing": sa_solver.solve_sa,
    "Tabu Search": ts_solver.solve_ts,
}

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. Benchmark Configuration")
    
    st.subheader("PassMark Scores")
    s_base = st.number_input(
        "Reference Score (s_base)", 
        min_value=1, 
        value=2000, # <--- UPDATED DEFAULT VALUE
        help="Single-thread score of the reference machine."
    )
    s_local = st.number_input(
        "Your Local Score (s_local)", 
        min_value=1, 
        value=2476, 
        help="Single-thread score of this machine (e.g., i5-9400F)."
    )
    
    st.subheader("Experiment Settings")
    instance_name = st.selectbox(
        "Problem Instance", 
        ["P-n16-k8"], 
        help="Select the problem instance to solve."
    )
    # Repetitions: allow the user to optionally set repetitions (iterations)
    custom_reps = st.checkbox("Set custom repetitions (iterations)", value=False)
    if custom_reps:
        num_repetitions = st.number_input(
            "Repetitions per Algorithm",
            min_value=1,
            value=5,
            help="Number of times each algorithm is run to average results."
        )
    else:
        # Keep None to indicate not set; the caller can decide default behavior
        num_repetitions = None
    
    st.subheader("2. Select Algorithms")
    algorithms_to_run = st.multiselect(
        "Choose algorithms to test:",
        options=list(ALGORITHM_MAP.keys()),
        default=list(ALGORITHM_MAP.keys())
    )
    
    st.warning("""
    **Note:** This app assumes you have manually set the time limits 
    inside the `gls_solver.py`, `sa_solver.py`, etc. files.
    """)

    st.subheader("Optional Solver Limits")
    st.write("Leave unchecked to keep OR-Tools defaults (no override).")
    set_time_limit = st.checkbox("Set time limit (seconds)", value=False)
    if set_time_limit:
        ui_time_limit_seconds = st.number_input(
            "Time limit (seconds)", min_value=0, value=5, step=1
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

# --- Main Page ---
if st.button("🚀 Run Benchmark", type="primary"):
    
    if not algorithms_to_run:
        st.error("Please select at least one algorithm to run.")
    else:
        # Load data
        instance_data = data_model.create_data_model()

        st.header("Running Experiment...")
        status_bar = st.container()
        results_list = []  # To store dicts for the DataFrame

        # --- B. Execution and Data Collection ---
        progress_bar = st.progress(0.0)
        # If repetitions not set, default to 1 run per algorithm
        reps = num_repetitions if num_repetitions is not None else 1
        total_runs = len(algorithms_to_run) * reps
        current_run = 0

        for algo_name in algorithms_to_run:
            algo_func = ALGORITHM_MAP[algo_name]
            cpu_times, objectives = [], []
            status_bar.text(f"Testing Algorithm: {algo_name}...")

            for i in range(reps):
                # Update progress
                current_run += 1
                progress_bar.progress(
                    current_run / total_runs,
                    text=f"Running {algo_name} (Rep {i+1}/{reps})",
                )

                # Build solver_kwargs only with provided (non-None) values
                solver_kwargs = {}
                if ui_time_limit_seconds is not None:
                    solver_kwargs["time_limit_seconds"] = ui_time_limit_seconds
                if ui_solution_limit is not None:
                    solver_kwargs["solution_limit"] = ui_solution_limit

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
        try:
            baseline_time = df[df["Algorithm"] == "Baseline (C&W)"]["Local CPU Time (s)"].iloc[0]
        except (IndexError, KeyError):
            st.warning("Baseline (C&W) not run. Cannot calculate 'Time vs. Baseline'.")
            baseline_time = 1.0  # Avoid division by zero

        # 1. Normalized Runtime (Eq 2)
        # t_norm = t_local * (s_local / s_base)
        df["Normalized Time (s)"] = df["Local CPU Time (s)"] * (s_local / s_base)

        # 2. Time Relative to the Baseline
        df["Time vs. Baseline"] = df["Local CPU Time (s)"] / baseline_time

        # Reorder columns for display
        df = df[[
            "Algorithm",
            "Cost",
            "Local CPU Time (s)",
            "Normalized Time (s)",
            "Time vs. Baseline",
        ]]

        st.dataframe(
            df.style.format({
                "Cost": "{:,.2f}",
                "Local CPU Time (s)": "{:.6f}",
                "Normalized Time (s)": "{:.6f}",
                "Time vs. Baseline": "{:.4f}",
            }),
            use_container_width=True,
        )

        st.info(f"""
        **How to Read This Table:**
        * **Cost:** The final solution (total distance). **Lower is better.**
        * **Local CPU Time (s):** The actual time your CPU (i5-9400F) took.
        * **Normalized Time (s):** The *virtual* time it would have taken on the reference machine (i9-13900KS). **This is the number to use when comparing to other research.**
        * **Time vs. Baseline:** How many times slower/faster the algorithm was compared to the simple Baseline (C&W).
        """)

        with st.expander("Show Configuration"):
            st.json({
                "Reference Score (s_base)": s_base,
                "Local Score (s_local)": s_local,
                "Instance": instance_name,
                "Repetitions": num_repetitions,
                "Repetitions (used)": reps,
                "Normalization Ratio (s_local / s_base)": (s_local / s_base),
            })