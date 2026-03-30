"""Bulk benchmark execution across multiple instances."""

import streamlit as st
import pandas as pd
import io
import base64
import os
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import streamlit.components.v1 as components
from components.execution.benchmark_common import (
    prepare_experiments, extract_instance_metadata, build_result_row, 
    run_experiment_reps, TIME_CHECKPOINTS
)
from components.ui.sidebar import find_instance_files
from components.utils import instance_data_parser
from components.utils import solution_parser
from components.execution.background_task import run_background_task, BackgroundTask


def run_bulk_benchmark(settings):
    """
    Run bulk benchmark on selected instances.
    Returns a DataFrame with aggregated results.
    """
    target_instances = st.session_state.get('selected_bulk_instances', [])
    gaetano_dir = "instances/gaetano"
    
    _, path_map = find_instance_files(gaetano_dir)

    if not target_instances:
        st.error("No instances selected for bulk run.")
        st.session_state.run_bulk = False
        st.stop()

    # Prepare experiments
    bulk_experiments = prepare_experiments(settings)

    if not bulk_experiments:
        st.error("Please select at least one algorithm in the sidebar.")
        st.session_state.run_bulk = False
        st.stop()

    # Run loop
    bulk_results = []
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    total_steps = len(target_instances) * len(bulk_experiments) * settings["reps"]
    current_step = 0

    for inst_name in target_instances:
        if inst_name not in path_map:
            continue
        
        p_info = path_map[inst_name]
        try:
            inst_data = instance_data_parser.load_vrp_instance(p_info["vrp"])
            # Bulk runs keep instance default vehicle counts
        except Exception as e:
            st.warning(f"Skipping {inst_name}: {e}")
            continue

        # Extract features (richer metadata from filename)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        bks_val = None
        if p_info["sol"]:
            try:
                bks_val = solution_parser.parse_solution_file(p_info["sol"])
            except:
                pass

        for exp in bulk_experiments:
            def progress_callback(rep, reps):
                nonlocal current_step
                current_step += 1
                progress_bar.progress(min(current_step / total_steps, 1.0))
                status_text.text(f"Processing: {inst_name} | {exp['name']} | Rep {rep+1}/{reps}")
            
            data = run_experiment_reps(exp, inst_data, exp["reps"], progress_callback)
            
            if data["costs"]:
                result_row = build_result_row(exp, instance_meta, data["costs"], data["times"],
                                             data["neighbors_list"], data["best_routes"], bks_val, data["checkpoints"])
                bulk_results.append(result_row)

    st.success("Bulk Benchmark Complete!")
    st.session_state.run_bulk = False
    
    status_text.empty()
    progress_bar.empty()
    
    if bulk_results:
        df_bulk = pd.DataFrame(bulk_results)

        st.subheader("Results Preview")
        st.dataframe(df_bulk.head(10), use_container_width=True)

        # Excel file will be downloaded when execution is finished
        st.info("Excel file will be downloaded when execution is finished")

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_bulk.to_excel(writer, sheet_name='Benchmark', index=False)

        data = buffer.getvalue()
        
        # Save to server if enabled
        if settings.get("save_to_server", False):
            server_dir = "server_output"
            os.makedirs(server_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            server_filename = f"vrp_bulk_benchmark_results_{timestamp}.xlsx"
            server_path = os.path.join(server_dir, server_filename)
            
            try:
                with open(server_path, "wb") as f:
                    f.write(data)
                st.success(f"Results saved to server: {server_path}")
            except Exception as e:
                st.error(f"Failed to save to server: {e}")
        
        # Create a base64 data URI and auto-click a hidden link via an HTML component
        b64 = base64.b64encode(data).decode()
        href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
        html = f'<a id="dl" href="{href}" download="vrp_bulk_benchmark_results.xlsx"></a>\n'
        html += '<script>document.getElementById("dl").click();</script>'
        components.html(html, height=0)
        st.success("Excel file download started in your browser.")

        # Fallback download button if automatic download is blocked
        st.download_button(
            label="If automatic download failed, click to download",
            data=data,
            file_name="vrp_bulk_benchmark_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No results were generated. Check your configurations.")


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
            # Try with increasing number of vehicles (0 = original, 1-5 = +1 to +5)
            for vehicle_attempt in range(6):
                # Prepare instance data with modified vehicle count
                working_instance = inst_data.copy()
                if vehicle_attempt > 0:
                    working_instance = copy.deepcopy(inst_data)
                    original_num = working_instance['num_vehicles']
                    working_instance['num_vehicles'] = original_num + vehicle_attempt
                    working_instance['vehicle_capacities'] = [working_instance['capacity']] * working_instance['num_vehicles']
                    task.log(f"Retrying {inst_name} | {exp['name']} with {working_instance['num_vehicles']} vehicles (+{vehicle_attempt})")

                # Custom progress callback for background task
                # Only count progress on first vehicle attempt to avoid exceeding 100%
                def bg_progress_callback(rep, reps):
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
                    
                    step_name = f"{inst_name} | {exp['name']} | Rep {rep+1}/{reps}"
                    if vehicle_attempt > 0:
                        step_name += f" | Vehicles +{vehicle_attempt}"
                    task.update_progress(current_step, total_steps, step_name)
                
                try:
                    data = run_experiment_reps(exp, working_instance, exp["reps"], bg_progress_callback)
                    
                    # If we found a solution, break out of vehicle retry loop
                    if data["costs"]:
                        result_row = build_result_row(exp, instance_meta, data["costs"], data["times"],
                                                      data["neighbors_list"], data["best_routes"], bks_val, data["checkpoints"])
                        results.append(result_row)
                        task.log(f"✓ Solution found for {inst_name} | {exp['name']} with {vehicle_attempt} additional vehicles. Cost: {min(data['costs']):.2f}")
                        break  # Exit vehicle retry loop since we found a solution
                    
                    # If this is the last vehicle attempt and still no solution, record as failed
                    if vehicle_attempt == 5:
                        result_row = build_result_row(exp, instance_meta, [], data["times"],
                                                      data["neighbors_list"], None, bks_val, data["checkpoints"])
                        results.append(result_row)
                        task.log(f"✗ No solution found for {inst_name} | {exp['name']} even with +5 vehicles", level="warning")
                
                except InterruptedError:
                    return results
        
        return results
    except Exception as e:
        task.log(f"Error processing {inst_name}: {str(e)}", level="error")
        return results


def run_bulk_benchmark_background(task, settings):
    """
    Run bulk benchmark execution as a background task.
    Updates task progress instead of using Streamlit UI.
    Uses parallel processing for instances.
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
    total_steps = len(target_instances) * len(bulk_experiments) * settings["reps"]
    current_step = 0
    
    # Get number of parallel workers from settings
    num_parallel = settings.get('num_parallel_instances', 2)
    task.log(f"Using {num_parallel} parallel workers for instance processing")
    
    # Run loop
    task.log(f"Total experiments: {len(bulk_experiments)}")
    task.log(f"Total steps: {total_steps} ({len(target_instances)} instances × {len(bulk_experiments)} algorithms × {settings['reps']} reps)")
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
                task.log(f"Received {len(instance_results)} results from instance")
                bulk_results.extend(instance_results)
                task.log(f"Total results collected so far: {len(bulk_results)}")
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


