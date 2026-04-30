"""Single instance benchmark execution (Synchronous only)."""

import streamlit as st
import pandas as pd
import statistics
import random
import os
import copy
from components.execution.benchmark_common import (
    prepare_experiments, extract_instance_metadata, build_result_row, 
    run_experiment_reps
)


def run_single_benchmark(instance_data, settings, bks_cost, instance_name="Unknown"):
    """
    Run a single instance benchmark with the given settings.
    Returns a tuple: (DataFrame with results, all_histories dict, all_best_routes dict)
    """
    experiments = prepare_experiments(settings)
    instance_meta = extract_instance_metadata(instance_data, instance_name)
    
    results_list = []
    all_histories = {}  # exp_name -> history data
    all_best_routes = {}  # exp_name -> best routes
    
    total = sum(e.reps for e in experiments)
    prog = st.progress(0.0)
    cnt = 0
    stat = st.empty()

    for exp in experiments:
        stat.text(f"Running: {exp.name}")
        
        def progress_callback(rep, reps):
            nonlocal cnt
            cnt += 1
            prog.progress(cnt / total)
        
        # Pass bks_cost to ensure target_cost is calculated if gap is set
        data = run_experiment_reps(exp, instance_data, exp.reps, bks_cost, progress_callback)
        
        # Store histories and best routes for visualization
        all_histories[exp.name] = data.all_histories
        all_best_routes[exp.name] = data.best_routes
        
        result_row = build_result_row(exp, instance_meta, data.costs, data.times, 
                                      data.neighbors_list, data.best_routes, bks_cost, data.checkpoints)
        results_list.append(result_row.to_dict())

    df = pd.DataFrame(results_list)
    stat.empty()
    prog.empty()
    
    return df, all_histories, all_best_routes
