#!/usr/bin/env python3
"""
USAL Server Benchmark Script for Major Revision (COMPELECENG-D-26-08714)
========================================================================
Runs the two new experiments requested by Reviewers 1, 2, 3, and 4:

  1. Experiment A (External Generalization on CVRPLIB - Uchoa et al., 2017):
     Evaluates all 9 OR-Tools heuristic-metaheuristic combinations, the OR-Tools Default
     configuration (AUTOMATIC), and HGS-CVRP under the operational time budget T = 0.5 * N
     across the CVRPLIB X-benchmark instances in the study's scale regime (N <= 200, 22 instances,
     or up to --max-uchoa-n). Records full 0.5s trajectories.

  2. Experiment B (Matched-Budget Baselines on the 200 Unseen T-CVRP Test Instances):
     Evaluates HGS-CVRP and OR-Tools Default (AUTOMATIC) under the exact same dynamic budget
     T = 0.5 * N on the 200 test instances of the T-CVRP benchmark.

Features:
  - Multiprocessing pool with configurable workers (--workers 8)
  - Automatic per-instance JSON checkpointing & resumption (safe against interruptions)
  - Exports consolidated CSV and Excel files into server_output/
"""
import os
import re
import sys
import json
import time
import glob
import copy
import math
import random
import argparse
import statistics
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2 as RE

from components.execution import configurable_solver
from components.execution.hygese_solver import solve_hgs
from components.execution.benchmark_common import run_experiment_with_vehicle_retry
from components.models import ExperimentConfig, InstanceMetadata
from components.utils import instance_data_parser, solution_parser
from components.utils.helpers import parse_gaetano_metadata

CHECKPOINT_INTERVAL_SEC = 0.5

ORTOOLS_9_COMBOS = [
    ("Savings (Clarke-Wright)", RE.FirstSolutionStrategy.SAVINGS,
     "Guided Local Search (GLS)", RE.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
    ("Savings (Clarke-Wright)", RE.FirstSolutionStrategy.SAVINGS,
     "Tabu Search", RE.LocalSearchMetaheuristic.TABU_SEARCH),
    ("Savings (Clarke-Wright)", RE.FirstSolutionStrategy.SAVINGS,
     "Simulated Annealing", RE.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
    ("Parallel Cheapest Insertion", RE.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
     "Guided Local Search (GLS)", RE.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
    ("Parallel Cheapest Insertion", RE.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
     "Tabu Search", RE.LocalSearchMetaheuristic.TABU_SEARCH),
    ("Parallel Cheapest Insertion", RE.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
     "Simulated Annealing", RE.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
    ("Christofides", RE.FirstSolutionStrategy.CHRISTOFIDES,
     "Guided Local Search (GLS)", RE.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
    ("Christofides", RE.FirstSolutionStrategy.CHRISTOFIDES,
     "Tabu Search", RE.LocalSearchMetaheuristic.TABU_SEARCH),
    ("Christofides", RE.FirstSolutionStrategy.CHRISTOFIDES,
     "Simulated Annealing", RE.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
]


def build_metadata_corrected(inst_data: dict, inst_name: str) -> InstanceMetadata:
    """
    Correctly extracts InstanceMetadata for both Gaetano (LDG*) and Uchoa (X-n*-k*) instances.
    Fixes the node-count vs customer-count offset (N = num_nodes - 1) and parses k from X-n*-k*.
    """
    num_nodes = inst_data.get('num_nodes', 1)
    customers = max(1, num_nodes - 1)

    # Check if Uchoa format X-n101-k25
    m_uchoa = re.match(r'^X-n(\d+)-k(\d+)', inst_name)
    if m_uchoa:
        k_fleet = int(m_uchoa.group(2))
        if k_fleet > inst_data.get('num_vehicles', 0):
            inst_data['num_vehicles'] = k_fleet
            inst_data['vehicle_capacities'] = [inst_data['capacity']] * k_fleet
        return InstanceMetadata(
            name=inst_name,
            customers=customers,
            vehicles=inst_data.get('num_vehicles', k_fleet),
            capacity=inst_data.get('capacity', 0),
            depot_layout="CVRPLIB",
            customer_layout="CVRPLIB",
            demand_profile="CVRPLIB",
            route_class="CVRPLIB",
            climate="none"
        )

    meta_features = parse_gaetano_metadata(inst_name)
    return InstanceMetadata(
        name=inst_name,
        customers=customers,
        vehicles=inst_data.get('num_vehicles', 0),
        capacity=inst_data.get('capacity', 0),
        depot_layout=meta_features.get("Depot Layout", "Unknown"),
        customer_layout=meta_features.get("Customer Layout", "Unknown"),
        demand_profile=meta_features.get("Demand Profile ID", "Unknown"),
        route_class=meta_features.get("Route Length Class", "Unknown"),
        climate=meta_features.get("Climate", "Unknown")
    )


def build_checkpoints_for_budget(T: float):
    checkpoint_configs = []
    probe_times = set()
    step_idx = 1
    while True:
        t_val = round(step_idx * CHECKPOINT_INTERVAL_SEC, 3)
        if t_val > T:
            break
        probe_times.add(t_val)
        label_str = str(int(t_val)) if t_val.is_integer() else f"{t_val:.1f}"
        checkpoint_configs.append({"time": t_val, "label": label_str, "is_pct": False})
        step_idx += 1

    t_final = round(T, 3)
    if t_final not in probe_times:
        probe_times.add(t_final)
        label_str = str(int(t_final)) if t_final.is_integer() else f"{t_final:.1f}"
        checkpoint_configs.append({"time": t_final, "label": label_str, "is_pct": False})

    return sorted(list(probe_times)), checkpoint_configs


def solve_single_instance_worker(task_tuple):
    """
    Worker function executed in parallel for a single instance.
    task_tuple: (vrp_path, include_9_ortools, include_default_ortools, include_hgs, reps, cache_dir)
    """
    vrp_path, include_9_ortools, include_default_ortools, include_hgs, reps, cache_dir = task_tuple
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    cache_file = os.path.join(cache_dir, f"{inst_name}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_rows = json.load(f)
            if cached_rows:
                print(f"  [SKIP/RESUME] {inst_name} loaded from checkpoint ({len(cached_rows)} rows).")
                return cached_rows
        except Exception:
            pass

    t_inst_start = time.time()
    try:
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = build_metadata_corrected(inst_data, inst_name)
        N = instance_meta.customers
        T = round(N * 0.5, 3)

        time_checkpoints, checkpoint_configs = build_checkpoints_for_budget(T)

        sol_path = vrp_path.replace(".vrp", ".sol")
        bks_val = None
        if os.path.exists(sol_path):
            try:
                bks_val = solution_parser.parse_solution_file(sol_path)
            except Exception:
                pass

        experiments = []
        if include_9_ortools:
            for fs_label, fs_enum, mh_label, mh_enum in ORTOOLS_9_COMBOS:
                experiments.append((
                    "ortools",
                    ExperimentConfig(
                        name=f"OR-Tools: {mh_label} [{fs_label}]",
                        fs_label=fs_label,
                        mh_label=mh_label,
                        func=configurable_solver.solve_cvrp,
                        kwargs={
                            "first_solution_strategy": fs_enum,
                            "local_search_metaheuristic": mh_enum,
                            "time_limit_seconds": T,
                            "target_gap_percent": 5.0,
                            "continue_after_target": True,
                        },
                        reps=reps
                    )
                ))

        if include_default_ortools:
            experiments.append((
                "ortools",
                ExperimentConfig(
                    name="OR-Tools: AUTOMATIC [AUTOMATIC]",
                    fs_label="Default (AUTOMATIC)",
                    mh_label="Default (AUTOMATIC)",
                    func=configurable_solver.solve_cvrp,
                    kwargs={
                        "first_solution_strategy": RE.FirstSolutionStrategy.AUTOMATIC,
                        "local_search_metaheuristic": RE.LocalSearchMetaheuristic.AUTOMATIC,
                        "time_limit_seconds": T,
                        "target_gap_percent": 5.0,
                        "continue_after_target": True,
                    },
                    reps=reps
                )
            ))

        if include_hgs:
            experiments.append((
                "hgs",
                ExperimentConfig(
                    name="HGS-CVRP (Matched Budget)",
                    fs_label="HGS-CVRP",
                    mh_label="HGS-CVRP",
                    func=solve_hgs,
                    kwargs={
                        "time_limit_seconds": T,
                        "no_improvement_limit_iterations": 1000000000,
                    },
                    reps=reps
                )
            ))

        rows = []
        for engine_name, exp in experiments:
            res_dict, _, _, _ = run_experiment_with_vehicle_retry(
                exp, inst_data, bks_val, instance_meta,
                max_retries=10,
                time_checkpoints=time_checkpoints,
                checkpoint_configs=checkpoint_configs,
                use_permutations=(engine_name == "ortools"),
                engine=engine_name
            )
            if res_dict:
                # For HGS-CVRP, since pyhygese returns final cost at timeLimit T,
                # also evaluate HGS at fixed short checkpoints (5s, 10s, 15s, 30s, 60s <= T)
                if engine_name == "hgs":
                    hgs_sub_budgets = [tb for tb in [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0] if tb < T]
                    for tb in hgs_sub_budgets:
                        sub_exp = copy.deepcopy(exp)
                        sub_exp.kwargs["time_limit_seconds"] = tb
                        sub_exp.reps = 1  # 1 fast probe for intermediate HGS checkpoints
                        sub_res, _, _, _ = run_experiment_with_vehicle_retry(
                            sub_exp, inst_data, bks_val, instance_meta,
                            max_retries=5,
                            time_checkpoints=[tb],
                            checkpoint_configs=[{"time": tb, "label": str(int(tb)), "is_pct": False}],
                            use_permutations=False,
                            engine="hgs"
                        )
                        if sub_res:
                            lbl = str(int(tb))
                            res_dict[f"Avg Cost @ {lbl}s"] = sub_res.get("Avg Cost")
                            res_dict[f"Best Cost @ {lbl}s"] = sub_res.get("Best Cost")
                    lbl_T = str(int(T)) if float(T).is_integer() else f"{T:.1f}"
                    res_dict[f"Avg Cost @ {lbl_T}s"] = res_dict.get("Avg Cost")
                    res_dict[f"Best Cost @ {lbl_T}s"] = res_dict.get("Best Cost")
                rows.append(res_dict)

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(rows, f)

        elapsed = time.time() - t_inst_start
        print(f"  [DONE] {inst_name} (N={N}, T={T}s) finished in {elapsed:.1f}s ({len(rows)} solver records).")
        return rows
    except Exception as e:
        print(f"  [ERROR] {inst_name}: {e}")
        return []


def get_200_test_instances() -> list[str]:
    """Loads the exact 200 unseen test instance names from CSV or Excel."""
    split_csv = "benchmark_1000_train_test_split_ids.csv"
    if os.path.exists(split_csv):
        df_s = pd.read_csv(split_csv)
        return df_s[df_s["Split"] == "Test"]["Instance"].tolist()

    excel_file = "or-tools_gaetano_benchmark_(1,000_random_instances)_20501208_174656.xlsx"
    if os.path.exists(excel_file):
        from sklearn.model_selection import train_test_split
        df_e = pd.read_excel(excel_file, usecols=["Instance"]).drop_duplicates()
        u_inst = df_e["Instance"].unique()
        _, test_inst = train_test_split(u_inst, test_size=0.2, random_state=68)
        return list(test_inst)

    raise FileNotFoundError(
        "Could not find 'benchmark_1000_train_test_split_ids.csv' or the benchmark Excel file."
    )


def run_batch(tasks: list, workers: int, out_prefix: str):
    print(f"\nStarting batch '{out_prefix}' with {len(tasks)} instances across {workers} workers...")
    t0 = time.time()
    all_rows = []
    if workers > 1:
        with Pool(processes=workers) as pool:
            for inst_rows in pool.imap_unordered(solve_single_instance_worker, tasks):
                all_rows.extend(inst_rows)
    else:
        for t in tasks:
            all_rows.extend(solve_single_instance_worker(t))

    if not all_rows:
        print(f"No rows produced for {out_prefix}.")
        return

    df_out = pd.DataFrame(all_rows)
    df_out.sort_values(by=["Instance", "First Solution", "Metaheuristic"], inplace=True)
    os.makedirs("server_output", exist_ok=True)
    csv_path = os.path.join("server_output", f"{out_prefix}.csv")
    xlsx_path = os.path.join("server_output", f"{out_prefix}.xlsx")
    df_out.to_csv(csv_path, index=False)
    try:
        df_out.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"Note: CSV saved; Excel export skipped ({e}).")
    print(f"Batch '{out_prefix}' completed in {(time.time() - t0)/60.0:.1f} min! Saved to:\n  - {csv_path}\n  - {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Major Revision Experiments A (Uchoa) & B (Gaetano Test Baselines)")
    parser.add_argument("--mode", choices=["all", "uchoa", "gaetano_test"], default="all",
                        help="Which experiment to run (default: all)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel single-thread workers (default: 8)")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per solver (default: 3, or 1 for fast run)")
    parser.add_argument("--max-uchoa-n", type=int, default=200,
                        help="Max customers N in Uchoa set (default: 200 -> 22 instances matching T-CVRP scale)")
    args = parser.parse_args()

    print("=" * 90)
    print(f"ELSEVIER REVISION BENCHMARK RUNNER | Mode: {args.mode} | Workers: {args.workers} | Reps: {args.reps}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    if args.mode in ("all", "uchoa"):
        uchoa_files = sorted(glob.glob(os.path.join("instances", "uchoa", "*.vrp")))
        selected_uchoa = []
        for f in uchoa_files:
            m = re.search(r"X-n(\d+)-k(\d+)", os.path.basename(f))
            if m:
                n_cust = int(m.group(1)) - 1
                if n_cust <= args.max_uchoa_n:
                    selected_uchoa.append(f)
        print(f"[Experiment A - Uchoa CVRPLIB] Selected {len(selected_uchoa)} instances with N <= {args.max_uchoa_n}.")
        cache_dir_a = os.path.join("server_output", "cache_exp_a_uchoa")
        tasks_a = [
            (f, True, True, True, args.reps, cache_dir_a)
            for f in selected_uchoa
        ]
        run_batch(tasks_a, args.workers, "review_expA_uchoa_cvrplib_results")

    if args.mode in ("all", "gaetano_test"):
        test_inst_names = get_200_test_instances()
        gaetano_files = [
            os.path.join("instances", "gaetano", f"{name}.vrp")
            for name in test_inst_names
            if os.path.exists(os.path.join("instances", "gaetano", f"{name}.vrp"))
        ]
        print(f"[Experiment B - Gaetano 200 Test Set] Found {len(gaetano_files)}/200 test .vrp files.")
        cache_dir_b = os.path.join("server_output", "cache_exp_b_gaetano_test")
        tasks_b = [
            (f, False, True, True, args.reps, cache_dir_b)
            for f in gaetano_files
        ]
        run_batch(tasks_b, args.workers, "review_expB_gaetano_test200_hgs_and_default")

    print("\nAll requested experiments finished successfully!")


if __name__ == "__main__":
    main()
