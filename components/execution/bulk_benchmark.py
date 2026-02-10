"""Bulk benchmark execution across multiple instances."""

import streamlit as st
import pandas as pd
import statistics
import random
import io
import base64
import os
from datetime import datetime
import streamlit.components.v1 as components
from benchmark_utils import execute_and_measure
from components.ui.sidebar import find_instance_files
from components.utils.helpers import get_cost_at_time, parse_gaetano_metadata
from components.execution.background_task import run_background_task, BackgroundTask
import instance_data_parser
import solution_parser


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
    bulk_experiments = []
    for fs in settings["sel_fs"]:
        for mh in settings["sel_mh"]:
            fs_label = fs.split('(')[0].strip()
            mh_label = mh.split('(')[0].strip()
            algo_name = f"{mh_label} [{fs_label}]"
            kw = {
                "first_solution_strategy": settings["fs_enum"][fs],
                "local_search_metaheuristic": settings["mh_enum"][mh],
                "time_limit_seconds": settings["time_limit"],
                "solution_limit": settings["sol_limit"],
                "lns_time_limit_seconds": settings["lns_limit"],
                "target_cost": None,  # Cannot use gap target in bulk
                "no_improvement_limit": settings["no_improv"],
                "no_improvement_iterations_limit": settings["no_improv_iter"]
            }
            bulk_experiments.append({
                "name": algo_name,
                "func": settings["solver_func"],
                "kwargs": kw,
                "fs_label": fs_label,
                "mh_label": mh_label
            })

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
    time_checkpoints = [5, 10, 15, 20, 30, 60]

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
        meta_features = parse_gaetano_metadata(inst_name)
        n_customers = inst_data.get('num_nodes', 0) - 1
        n_vehicles = inst_data.get('num_vehicles', 0)
        capacity = inst_data.get('capacity', 0)
        
        bks_val = None
        if p_info["sol"]:
            try:
                bks_val = solution_parser.parse_solution_file(p_info["sol"])
            except:
                pass

        for exp in bulk_experiments:
            algo_costs = []
            algo_times = []
            checkpoint_collectors = {t: [] for t in time_checkpoints}

            for r in range(settings["reps"]):
                current_step += 1
                progress_bar.progress(min(current_step / total_steps, 1.0))
                status_text.text(f"Processing: {inst_name} | {exp['name']} | Rep {r+1}/{settings['reps']}")
                
                cur_kw = exp["kwargs"].copy()
                cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
                
                res = execute_and_measure(exp["func"], inst_data, **cur_kw)
                
                if res["objective_value"] is not None:
                    algo_costs.append(res["objective_value"])
                    algo_times.append(res["cpu_time"])
                    
                    history = res.get("history", [])
                    for t_chk in time_checkpoints:
                        cost_at_t = get_cost_at_time(history, t_chk)
                        if cost_at_t is not None:
                            checkpoint_collectors[t_chk].append(cost_at_t)
            
            if algo_costs:
                avg_cost = statistics.mean(algo_costs)
                best_cost = min(algo_costs)
                avg_time = statistics.mean(algo_times)
                
                best_gap = ((best_cost - bks_val) / bks_val * 100) if bks_val else None
                avg_gap = ((avg_cost - bks_val) / bks_val * 100) if bks_val else None
                
                row = {
                    "Instance": inst_name,
                    "Depot Layout": meta_features["Depot Layout"],
                    "Cust Layout": meta_features["Customer Layout"],
                    "Demand Type": meta_features["Demand Profile ID"],
                    "Route Class": meta_features["Route Length Class"],
                    "Climate": meta_features["Climate"],
                    "Customers": n_customers,
                    "Vehicles": n_vehicles,
                    "Capacity": capacity,
                    "First Solution": exp.get("fs_label"),
                    "Metaheuristic": exp.get("mh_label"),
                    "Repetitions": settings["reps"],
                    "Avg Cost": avg_cost,
                    "Best Cost": best_cost,
                    "BKS Cost": bks_val,
                    "Best Gap (%)": best_gap,
                    "Avg Gap (%)": avg_gap,
                    "Avg CPU Time (s)": avg_time
                }
                
                for t_chk in time_checkpoints:
                    costs_at_t = checkpoint_collectors[t_chk]
                    if costs_at_t:
                        row[f"Avg Cost @ {t_chk}s"] = statistics.mean(costs_at_t)
                        row[f"Best Cost @ {t_chk}s"] = min(costs_at_t)
                    else:
                        row[f"Avg Cost @ {t_chk}s"] = None
                        row[f"Best Cost @ {t_chk}s"] = None
                        
                bulk_results.append(row)

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
    
    st.stop()


def run_bulk_benchmark_background(task, settings):
    """
    Run bulk benchmark execution as a background task.
    Updates task progress instead of using Streamlit UI.
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
    bulk_experiments = []
    task.log(f"Preparing experiments...")
    for fs in settings["sel_fs"]:
        for mh in settings["sel_mh"]:
            fs_label = fs.split('(')[0].strip()
            mh_label = mh.split('(')[0].strip()
            algo_name = f"{mh_label} [{fs_label}]"
            kw = {
                "first_solution_strategy": settings["fs_enum"][fs],
                "local_search_metaheuristic": settings["mh_enum"][mh],
                "time_limit_seconds": settings["time_limit"],
                "solution_limit": settings["sol_limit"],
                "lns_time_limit_seconds": settings["lns_limit"],
                "target_cost": None,  # Cannot use gap target in bulk
                "no_improvement_limit": settings["no_improv"],
                "no_improvement_iterations_limit": settings["no_improv_iter"]
            }
            bulk_experiments.append({
                "name": algo_name,
                "func": settings["solver_func"],
                "kwargs": kw,
                "fs_label": fs_label,
                "mh_label": mh_label
            })

    if not bulk_experiments:
        task.set_completed(error="Please select at least one algorithm in the sidebar.")
        return

    # Prepare loop variables
    bulk_results = []
    total_steps = len(target_instances) * len(bulk_experiments) * settings["reps"]
    current_step = 0
    time_checkpoints = [5, 10, 15, 20, 30, 60]
    
    # Run loop
    task.log(f"Total experiments: {len(bulk_experiments)}")
    task.log(f"Total steps: {total_steps} ({len(target_instances)} instances × {len(bulk_experiments)} algorithms × {settings['reps']} reps)")
    task.log(f"================== EXECUTION START ====================")


    for inst_name in target_instances:
        if inst_name not in path_map:
            task.log(f"Skipping {inst_name}: not found in path map")
            continue

        p_info = path_map[inst_name]
        try:
            inst_data = instance_data_parser.load_vrp_instance(p_info["vrp"])
            task.log(f"Loaded instance: {inst_name}")
        except Exception as e:
            task.log(f"Error loading {inst_name}: {e}", level="error")
            continue

        # Extract features
        meta_features = parse_gaetano_metadata(inst_name)
        n_customers = inst_data.get('num_nodes', 0) - 1
        n_vehicles = inst_data.get('num_vehicles', 0)
        capacity = inst_data.get('capacity', 0)
        
        bks_val = None
        if p_info["sol"]:
            try:
                bks_val = solution_parser.parse_solution_file(p_info["sol"])
            except:
                pass

        for exp in bulk_experiments:
            algo_costs = []
            algo_times = []
            checkpoint_collectors = {t: [] for t in time_checkpoints}

            for r in range(settings["reps"]):
                current_step += 1
                progress = current_step / total_steps * 100
                step_name = f"{inst_name} | {exp['name']} | Rep {r+1}/{settings['reps']}"
                task.update_progress(current_step, total_steps, step_name)
                cur_kw = exp["kwargs"].copy()
                cur_kw["random_seed"] = random.randint(0, 2**31 - 1)
                
                res = execute_and_measure(exp["func"], inst_data, **cur_kw)
                
                if res["objective_value"] is not None:
                    algo_costs.append(res["objective_value"])
                    algo_times.append(res["cpu_time"])
                    
                    history = res.get("history", [])
                    for t_chk in time_checkpoints:
                        cost_at_t = get_cost_at_time(history, t_chk)
                        if cost_at_t is not None:
                            checkpoint_collectors[t_chk].append(cost_at_t)
            
            if algo_costs:
                avg_cost = statistics.mean(algo_costs)
                best_cost = min(algo_costs)
                avg_time = statistics.mean(algo_times)
                
                best_gap = ((best_cost - bks_val) / bks_val * 100) if bks_val else None
                avg_gap = ((avg_cost - bks_val) / bks_val * 100) if bks_val else None
                
                row = {
                    "Instance": inst_name,
                    "Depot Layout": meta_features["Depot Layout"],
                    "Cust Layout": meta_features["Customer Layout"],
                    "Demand Type": meta_features["Demand Profile ID"],
                    "Route Class": meta_features["Route Length Class"],
                    "Climate": meta_features["Climate"],
                    "Customers": n_customers,
                    "Vehicles": n_vehicles,
                    "Capacity": capacity,
                    "First Solution": exp.get("fs_label"),
                    "Metaheuristic": exp.get("mh_label"),
                    "Repetitions": settings["reps"],
                    "Avg Cost": avg_cost,
                    "Best Cost": best_cost,
                    "BKS Cost": bks_val,
                    "Best Gap (%)": best_gap,
                    "Avg Gap (%)": avg_gap,
                    "Avg CPU Time (s)": avg_time
                }
                
                for t_chk in time_checkpoints:
                    costs_at_t = checkpoint_collectors[t_chk]
                    if costs_at_t:
                        row[f"Avg Cost @ {t_chk}s"] = statistics.mean(costs_at_t)
                        row[f"Best Cost @ {t_chk}s"] = min(costs_at_t)
                    else:
                        row[f"Avg Cost @ {t_chk}s"] = None
                        row[f"Best Cost @ {t_chk}s"] = None
                        
                bulk_results.append(row)

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


