"""Bulk benchmark execution across multiple instances (Background only)."""

import pandas as pd
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from components.execution.benchmark_common import (
    prepare_experiments, extract_instance_metadata, build_result_row, 
    run_experiment_reps, run_experiment_with_vehicle_retry
)
from components.ui.sidebar import find_instance_files
from components.utils import instance_data_parser
from components.utils import solution_parser


def _process_instance_benchmark(inst_name, p_info, bulk_experiments, settings, path_map, task, shared_state):
    """
    Process a single instance across all experiments.
    Returns list of result rows for this instance.
    shared_state: dict with 'lock', 'current_step', 'total_steps', 'results' keys
    """
    results = []
    task.log(f"Starting processing for instance: {inst_name}")
    
    try:
        inst_data = instance_data_parser.load_vrp_instance(p_info["vrp"])
        task.log(f"Loaded instance: {inst_name}")
        
        # Extract features
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        bks_val = None
        if p_info["sol"]:
            try:
                bks_val = solution_parser.parse_solution_file(p_info["sol"])
            except:
                pass

        for exp in bulk_experiments:
            # Custom progress callback for background task
            def bg_progress_callback(rep, reps, vehicle_attempt):
                # Check for stop signal
                if task.should_stop():
                    task.log(f"Stop signal received in {inst_name}, skipping", level="warning")
                    raise InterruptedError("Benchmark stopped by user")
                
                # Only increment counter on first rep of first vehicle attempt
                if vehicle_attempt == 0 and rep == 0:
                    with shared_state['lock']:
                        shared_state['current_step'] += 1
                        current_step = shared_state['current_step']
                        total_steps = shared_state['total_steps']
                else:
                    with shared_state['lock']:
                        current_step = shared_state['current_step']
                        total_steps = shared_state['total_steps']
                
                step_name = f"{inst_name} | {exp.name} | Rep {rep+1}/{reps}"
                if vehicle_attempt > 0:
                    step_name += f" | Vehicles +{vehicle_attempt}"
                task.update_progress(current_step, total_steps, step_name)
            
            try:
                # Use shared logic from benchmark_common
                result_row_dict, found, final_attempt = run_experiment_with_vehicle_retry(
                    exp, inst_data, bks_val, instance_meta, 
                    max_retries=5, 
                    log_fn=task.log, 
                    progress_fn=bg_progress_callback
                )
                
                if result_row_dict:
                    results.append(result_row_dict)
                    if found:
                        if final_attempt > 0:
                            task.log(f"✓ Solution found for {inst_name} | {exp.name} with {final_attempt} additional vehicles.")
                    else:
                        task.log(f"✗ No solution found for {inst_name} | {exp.name} even with +5 vehicles", level="warning")
                
            except InterruptedError:
                return results
            except Exception as e:
                task.log(f"Error in experiment {exp.name} for {inst_name}: {str(e)}", level="error")
        
        return results
    except Exception as e:
        task.log(f"Error processing {inst_name}: {str(e)}", level="error")
        return results


def run_bulk_benchmark_background(task, settings):
    """
    Run bulk benchmark execution as a background task.
    """
    task.log(f"==================== BENCHMARK START ====================")
    task.log(f"Task ID: {task.task_id}")
    
    target_instances = settings.get('selected_instances', [])
    gaetano_dir = "instances/gaetano"
    
    task.log(f"Loading instances from: {gaetano_dir}")
    _, path_map = find_instance_files(gaetano_dir)
    task.log(f"Found {len(target_instances)} instances to process")

    if not target_instances:
        task.set_completed(error="No instances selected for bulk run.")
        return

    # Prepare experiments
    bulk_experiments = prepare_experiments(settings)
    
    if not bulk_experiments:
        task.set_completed(error="Please select at least one algorithm in the sidebar.")
        return

    # Prepare loop variables
    bulk_results = []
    total_steps = len(target_instances) * len(bulk_experiments)
    
    # Get number of parallel workers from settings
    num_parallel = settings.get('num_parallel_instances', 2)
    task.log(f"Using {num_parallel} parallel workers for instance processing")
    
    # Run loop
    task.log(f"Total experiments: {len(bulk_experiments)}")
    task.log(f"Total instances to process: {len(target_instances)}")
    task.log(f"================== EXECUTION START ====================")
    
    # Prepare shared state for thread-safe progress tracking
    shared_state = {
        'lock': threading.Lock(),
        'current_step': 0,
        'total_steps': total_steps,
        'results': []
    }
    
    # Filter valid instances and prepare tasks
    valid_instances = []
    for inst_name in target_instances:
        if inst_name not in path_map:
            task.log(f"Skipping {inst_name}: not found in path map")
            continue
        valid_instances.append((inst_name, path_map[inst_name]))
    
    # Run instances in parallel
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        # Submit all instance processing tasks
        futures = []
        for inst_name, p_info in valid_instances:
            # Check for stop signal before submitting
            if task.should_stop():
                task.log("Stop signal received, halting execution...", level="error")
                task.set_completed(error="Benchmark stopped by user")
                return
            
            future = executor.submit(
                _process_instance_benchmark,
                inst_name, p_info, bulk_experiments, settings, path_map, task, shared_state
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            if task.should_stop():
                task.log("Stop signal received, halting execution...", level="error")
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                task.set_completed(error="Benchmark stopped by user")
                return
            
            try:
                instance_results = future.result()
                bulk_results.extend(instance_results)
            except Exception as e:
                task.log(f"Error in instance processing: {str(e)}", level="error")

    # Save results to Excel
    task.log(f"================== SAVING RESULTS ====================")
    task.log(f"Total results to save: {len(bulk_results)}")
    
    if bulk_results:
        df_bulk = pd.DataFrame(bulk_results)
        server_dir = "server_output"
        os.makedirs(server_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        server_filename = f"vrp_bulk_benchmark_results_{timestamp}.xlsx"
        server_path = os.path.join(server_dir, server_filename)
        
        task.log(f"Creating Excel file...")
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_bulk.to_excel(writer, sheet_name='Benchmark', index=False)
            
            task.log(f"Writing to disk: {server_path}")
            with open(server_path, "wb") as f:
                f.write(buffer.getvalue())
            
            task.log(f"Results saved successfully!")
            task.set_completed(results_file=server_path)
            task.log(f"==================== BENCHMARK COMPLETE ====================")
        except Exception as e:
            task.set_completed(error=f"Failed to save results: {str(e)}")
            task.log(f"Failed to save results: {str(e)}", level="error")
    else:
        error_msg = "No results were generated. Check your configurations."
        task.set_completed(error=error_msg)
        task.log(error_msg, level="error")
        task.log(f"==================== BENCHMARK FAILED ====================")
