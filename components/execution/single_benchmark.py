"""Single instance benchmark execution."""

import streamlit as st
import statistics
import random
import json
import os
import copy
from datetime import datetime
from components.execution.benchmark_common import (
    prepare_experiments, extract_instance_metadata, build_result_row, 
    run_experiment_reps, TIME_CHECKPOINTS
)
from components.utils import instance_data_parser


def run_single_benchmark(instance_data, settings, bks_cost, instance_name="Unknown"):
    """
    Run a single instance benchmark with the given settings.
    Returns a DataFrame with results.
    """
    import pandas as pd
    
    experiments = prepare_experiments(settings)
    instance_meta = extract_instance_metadata(instance_data, instance_name)
    
    results_list = []
    total = sum(e["reps"] for e in experiments)
    prog = st.progress(0.0)
    cnt = 0
    stat = st.empty()

    for exp in experiments:
        stat.text(f"Running: {exp['name']}")
        
        def progress_callback(rep, reps):
            nonlocal cnt
            cnt += 1
            prog.progress(cnt / total)
        
        data = run_experiment_reps(exp, instance_data, exp["reps"], progress_callback)
        
        result_row = build_result_row(exp, instance_meta, data["costs"], data["times"], 
                                      data["iters_list"], data["best_routes"], bks_cost, data["checkpoints"])
        results_list.append(result_row)

    df = pd.DataFrame(results_list)
    stat.empty()
    prog.empty()
    
    return df


def run_single_benchmark_background(task, settings, instance_data_file, bks_cost):
    """
    Run single benchmark as a background task.
    Loads instance data from file since it can't be serialized easily.
    """
    import pandas as pd
    from components.utils import instance_data_parser
    
    task.log(f"==================== SINGLE BENCHMARK START ====================")
    task.log(f"Task ID: {task.task_id}")
    
    try:
        # Extract instance name from file path
        instance_name = os.path.splitext(os.path.basename(instance_data_file))[0]
        
        # Load instance data
        task.log(f"Loading instance: {instance_data_file}")
        instance_data = instance_data_parser.load_vrp_instance(instance_data_file)
        task.log(f"Instance loaded successfully")
        
        # Prepare experiments and metadata
        experiments = prepare_experiments(settings)
        instance_meta = extract_instance_metadata(instance_data, instance_name)
        
        task.log(f"Prepared {len(experiments)} experiments")
        
        results_list = []
        total = sum(e["reps"] for e in experiments) * 6  # Account for 6 vehicle attempts (0 to +5)
        current_step = 0
        
        for exp in experiments:
            # Try with increasing number of vehicles (0 = original, 1-5 = +1 to +5)
            for vehicle_attempt in range(6):
                # Prepare instance data with modified vehicle count
                working_instance = instance_data.copy()
                if vehicle_attempt > 0:
                    working_instance = copy.deepcopy(instance_data)
                    original_num = working_instance['num_vehicles']
                    working_instance['num_vehicles'] = original_num + vehicle_attempt
                    working_instance['vehicle_capacities'] = [working_instance['capacity']] * working_instance['num_vehicles']
                    task.log(f"Retrying {exp['name']} with {working_instance['num_vehicles']} vehicles (+{vehicle_attempt})")
                
                # Custom progress callback for background task
                def bg_progress_callback(rep, reps):
                    nonlocal current_step
                    current_step += 1
                    step_name = f"{exp['name']} | Rep {rep+1}/{reps}"
                    if vehicle_attempt > 0:
                        step_name += f" | Vehicles +{vehicle_attempt}"
                    task.update_progress(current_step, total, step_name)
                    
                    # Check for stop signal
                    if task.should_stop():
                        raise InterruptedError("Benchmark stopped by user")
                
                try:
                    data = run_experiment_reps(exp, working_instance, exp["reps"], bg_progress_callback)
                    
                    # If we found a solution, break out of vehicle retry loop
                    if data["costs"]:
                        result_row = build_result_row(exp, instance_meta, data["costs"], data["times"],
                                                      data["iters_list"], data["best_routes"], bks_cost, data["checkpoints"])
                        results_list.append(result_row)
                        task.log(f"Solution found for {exp['name']} with {vehicle_attempt} additional vehicles")
                        break  # Exit vehicle retry loop since we found a solution
                    
                    # If last attempt with no solution yet
                    if vehicle_attempt == 5:
                        result_row = build_result_row(exp, instance_meta, [], data["times"],
                                                      data["iters_list"], None, bks_cost, data["checkpoints"])
                        results_list.append(result_row)
                        task.log(f"No solution found for {exp['name']} even with +5 vehicles", level="warning")
                
                except InterruptedError:
                    task.set_completed(error="Benchmark stopped by user")
                    return
        
        # Save results
        task.log(f"Saving {len(results_list)} results")
        if results_list:
            df = pd.DataFrame(results_list)
            
            server_dir = "server_output"
            os.makedirs(server_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"vrp_single_benchmark_results_{timestamp}.xlsx"
            results_path = os.path.join(server_dir, results_filename)
            
            task.log(f"Writing results to {results_path}")
            df.to_excel(results_path, sheet_name='Benchmark', index=False)
            
            task.log(f"Results saved successfully!")
            task.set_completed(results_file=results_path)
            task.log(f"==================== SINGLE BENCHMARK COMPLETE ====================")
        else:
            error_msg = "No results generated. Check your configurations."
            task.set_completed(error=error_msg)
            task.log(error_msg, level="error")
    except Exception as e:
        error_msg = f"Error during benchmark: {str(e)}"
        task.set_completed(error=error_msg)
        task.log(error_msg, level="error")
