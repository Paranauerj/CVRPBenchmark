"""Bulk benchmark execution across multiple instances (Background only)."""

import os
import sys

# Crucial: Disable streamlit script run context check for child processes
if "streamlit" in sys.modules:
    # If streamlit is already loaded, we want to ensure we're not in a worker process
    # attempting to use UI features.
    pass

import pandas as pd
import os
import glob
import json
from datetime import datetime
from components.execution.benchmark_runner import BenchmarkRunner
from components.execution.benchmark_common import (
    prepare_experiments, extract_instance_metadata, 
    run_experiment_with_vehicle_retry
)
from components.utils.helpers import find_instance_files
from components.utils import instance_data_parser
from components.utils import solution_parser
from components.utils.logging_utils import get_task_logger


def _bulk_process_instance(vrp_path, experiments, engine="ortools"):
    """
    Picklable function to process a single instance in a worker process.
    """
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    results = []
    
    try:
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        
        bks_val = None
        sol_path = vrp_path.replace(".vrp", ".sol")
        if os.path.exists(sol_path):
            try:
                bks_val = solution_parser.parse_solution_file(sol_path)
            except:
                pass

        for exp in experiments:
            try:
                # Progress functions are tricky with ProcessPoolExecutor (can't pickle callbacks easily)
                # So we disable per-rep progress for bulk runs and only do per-instance progress
                result_row_dict, found, final_attempt, _ = run_experiment_with_vehicle_retry(
                    exp, inst_data, bks_val, instance_meta, 
                    max_retries=5, 
                    log_fn=lambda x: None,
                    progress_fn=None,
                    engine=engine
                )
                
                if result_row_dict:
                    results.append(result_row_dict)
            except Exception:
                pass
        
        return results
    except Exception:
        return results


def run_bulk_benchmark_background(task, settings):
    """
    Run bulk benchmark execution as a background task using BenchmarkRunner.
    """
    logger = get_task_logger(task.task_id)
    logger.info(f"==================== BENCHMARK START ====================")
    logger.info(f"Task ID: {task.task_id}")
    
    target_instances = settings.get('selected_instances', [])
    gaetano_dir = "instances/gaetano"
    
    logger.info(f"Loading instances from: {gaetano_dir}")
    _, path_map = find_instance_files(gaetano_dir)
    logger.info(f"Found {len(target_instances)} instances to process")

    if not target_instances:
        task.set_completed(error="No instances selected for bulk run.")
        return

    # Prepare experiments
    bulk_experiments = prepare_experiments(settings)
    
    if not bulk_experiments:
        task.set_completed(error="Please select at least one algorithm in the sidebar.")
        return

    # Build instance list for BenchmarkRunner
    valid_instances = []
    for inst_name in target_instances:
        if inst_name not in path_map:
            logger.info(f"Skipping {inst_name}: not found in path map")
            continue
        valid_instances.append((inst_name, path_map[inst_name]))
    
    total_instances = len(valid_instances)
    logger.info(f"Total experiments: {len(bulk_experiments)}")
    logger.info(f"Total instances to process: {total_instances}")
    logger.info(f"Using {settings.get('num_parallel_instances', 2)} parallel workers (ProcessPool)")
    logger.info(f"================== EXECUTION START ====================")
    
    temp_results_dir = f"temp_bulk_{task.task_id}"
    
    # Create runner
    runner = BenchmarkRunner(
        name="Bulk Benchmark",
        instances_dir=gaetano_dir,
        results_dir=temp_results_dir,
        output_dir="server_output",
        num_parallel=settings.get('num_parallel_instances', 2),
        chunk_size=total_instances,  # Process all in one chunk for simplicity in bulk
        use_processes=True           # True for ProcessPoolExecutor as requested
    )
    
    # Progress callback for the runner
    def progress_cb(current, total, step_name):
        task.update_progress(current, total, f"Processed {current}/{total} instances")

    # Wrap process function to include engine setting using partial (picklable)
    from functools import partial
    engine = settings.get("engine", "ortools")
    process_wrapper = partial(_bulk_process_instance, engine=engine)
    
    # Run
    try:
        runner.run(
            experiments=bulk_experiments,
            process_instance_fn=process_wrapper,
            task=task,
            progress_callback=progress_cb,
            instance_list=valid_instances
        )
        
        # If stopped by user, runner.run returns None
        if task.should_stop():
            task.set_completed(error="Benchmark stopped by user")
            runner.cleanup()
            return

        # Post-process results and save to Excel (Custom logic for bulk benchmark)
        logger.info(f"================== SAVING RESULTS ====================")
        bulk_results = []
        chunk_files = sorted(glob.glob(os.path.join(temp_results_dir, "chunk_*.json")))
        
        for cf in chunk_files:
            with open(cf, 'r') as f:
                bulk_results.extend(json.load(f))
        
        logger.info(f"Total results to save: {len(bulk_results)}")
        
        if bulk_results:
            df_bulk = pd.DataFrame(bulk_results)
            
            # Remove instance feature columns
            from components import constants as C
            feature_cols = [
                C.COL_DEPOT_LAYOUT, C.COL_CUSTOMER_LAYOUT, C.COL_DEMAND_TYPE, 
                C.COL_ROUTE_CLASS, C.COL_CLIMATE, C.COL_CUSTOMERS, 
                C.COL_CAPACITY
            ]
            df_bulk = df_bulk.drop(columns=[col for col in feature_cols if col in df_bulk.columns])

            # If HGS, remove progress columns and gaps
            if engine == "hgs":
                checkpoint_cols = [col for col in df_bulk.columns if "Cost @" in col]
                hgs_remove = [C.COL_BEST_GAP, C.COL_AVG_GAP, C.COL_TIME_TO_TARGET]
                cols_to_drop = checkpoint_cols + [c for c in hgs_remove if c in df_bulk.columns]
                df_bulk = df_bulk.drop(columns=cols_to_drop)

            server_dir = "server_output"
            os.makedirs(server_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            server_filename = f"vrp_bulk_benchmark_results_{timestamp}.xlsx"
            server_path = os.path.join(server_dir, server_filename)
            
            logger.info(f"Creating Excel file...")
            with pd.ExcelWriter(server_path, engine='xlsxwriter') as writer:
                df_bulk.to_excel(writer, sheet_name='Benchmark', index=False)
            
            logger.info(f"Results saved successfully to {server_path}!")
            task.set_completed(results_file=server_path)
            logger.info(f"==================== BENCHMARK COMPLETE ====================")
        else:
            error_msg = "No results were generated. Check your configurations."
            logger.error(error_msg)
            task.set_completed(error=error_msg)
            logger.info(f"==================== BENCHMARK FAILED ====================")
            
        # Cleanup
        runner.cleanup()
    
    except Exception as e:
        logger.error(f"Benchmark execution failed: {str(e)}")
        task.set_completed(error=f"Benchmark execution failed: {str(e)}")
        logger.info(f"==================== BENCHMARK FAILED ====================")
        runner.cleanup()
