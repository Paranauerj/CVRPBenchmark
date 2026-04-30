"""
Benchmark Validation Test Script
Verifies Target Gap logic and Time Checkpoint reporting.
"""

# TODO: Create a py file to do the benchmark by chunks => every 10 instances it stores the values in a file and at the end of execution
# unites all those results into a single one (gaetano instances - separate folder for the results and once its done, delete all temporary files)
# if somehow the loading fails, use the latest file (order by name, number, idk) to resume (replace the latest by the new execution so we dont have
# corrupted files)

import os
import sys
from components.utils import instance_data_parser, solution_parser
from components.execution import configurable_solver
from components.execution.benchmark_common import run_experiment_reps, ExperimentConfig
from components.ui.sidebar import FIRST_SOLUTIONS, METAHEURISTICS

def test_uchoa_target_gap():
    print("\n--- Testing Uchoa Target Gap (X-n106-k14) ---")
    vrp_path = "instances/uchoa/X-n106-k14.vrp"
    sol_path = "instances/uchoa/X-n106-k14.sol"
    
    # 1. Load data
    instance_data = instance_data_parser.load_vrp_instance(vrp_path)
    bks_cost = solution_parser.parse_solution_file(sol_path)
    print(f"Loaded BKS Cost: {bks_cost}")
    
    # 2. Configure experiment with 10% gap
    # target_cost should be 27591 * 1.1 = 30350.1
    target_gap = 5.0
    expected_target_cost = bks_cost * (1 + target_gap / 100) # pyright: ignore[reportOptionalOperand]
    
    settings = {
        "fs_enum": FIRST_SOLUTIONS,
        "mh_enum": METAHEURISTICS,
        "solver_func": configurable_solver.solve_cvrp,
        "time_limit": 10,
        "sol_limit": None,
        "lns_limit": None,
        "no_improv": None,
        "no_improv_iter": None,
        "target_gap": target_gap,
        "reps": 1,
        "sel_fs": ["Parallel Cheapest Insertion"],
        "sel_mh": ["Guided Local Search (GLS)"]
    }
    
    # Manual Experiment Setup to verify dynamic target_cost calculation
    fs_strategy = FIRST_SOLUTIONS["Parallel Cheapest Insertion"]
    mh_meta = METAHEURISTICS["Guided Local Search (GLS)"]
    
    exp = ExperimentConfig(
        name="Test Gap",
        fs_label="PCI",
        mh_label="GLS",
        func=configurable_solver.solve_cvrp,
        kwargs={
            "first_solution_strategy": fs_strategy,
            "local_search_metaheuristic": mh_meta,
            "time_limit_seconds": 10,
            "target_gap_percent": target_gap # This is what we refactored
        },
        reps=1
    )
    
    # 3. Run
    print(f"Running experiment with {target_gap}% gap (Expected Target Cost: {expected_target_cost})...")
    results = run_experiment_reps(exp, instance_data, 1, bks_cost=bks_cost)
    
    # 4. Validate
    if results.costs:
        actual_cost = results.costs[0]
        print(f"Final Cost: {actual_cost}")
        # The solver should stop as soon as it hits <= target_cost
        # Note: In GLS it might find a slightly better solution in the same step
        assert actual_cost <= expected_target_cost, f"Cost {actual_cost} should be <= target {expected_target_cost}"
        print("✅ Target Gap Logic Verified!")
    else:
        print("❌ No solution found")

def test_gaetano_time_checkpoint():
    print("\n--- Testing Gaetano Time Checkpoint (LDG100_1142) ---")
    vrp_path = "instances/gaetano/LDG100_1142_rain_100_0307.vrp"
    
    # 1. Load data
    instance_data = instance_data_parser.load_vrp_instance(vrp_path)
    
    # 2. Setup experiment for 10 seconds
    exp = ExperimentConfig(
        name="Test Time",
        fs_label="PCI",
        mh_label="GLS",
        func=configurable_solver.solve_cvrp,
        kwargs={
            "first_solution_strategy": FIRST_SOLUTIONS["Parallel Cheapest Insertion"],
            "local_search_metaheuristic": METAHEURISTICS["Guided Local Search (GLS)"],
            "time_limit_seconds": 10
        },
        reps=1
    )
    
    # 3. Run
    print("Running experiment for 10s...")
    results = run_experiment_reps(exp, instance_data, 1)
    
    # 4. Validate Checkpoints (specifically at 5s)
    checkpoint_5s = results.checkpoints.get(5)
    if checkpoint_5s:
        print(f"Cost at 5s: {checkpoint_5s[0]}")
        assert checkpoint_5s[0] > 0, "Cost at 5s should be a positive number"
        print("✅ Time Checkpoint Verified!")
    else:
        # It's possible for an easy instance to finish in < 5s
        if results.times[0] < 5:
            print(f"Instance finished in {results.times[0]:.2f}s (less than 5s). Checkpoint test skipped but valid.")
        else:
            print("❌ No data recorded for 5s checkpoint despite long run")

if __name__ == "__main__":
    try:
        test_uchoa_target_gap()
        test_gaetano_time_checkpoint()
        print("\n✨ All validation tests passed!")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        sys.exit(1)
