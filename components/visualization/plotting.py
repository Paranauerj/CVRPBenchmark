"""Route and convergence visualization for benchmark results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from collections import defaultdict


def plot_routes(customers, routes, depot_pos=(0, 0), title="Routes"):
    """
    Plot routes on 2D space.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if not customers or not routes:
        ax.text(0.5, 0.5, 'No route data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Plot depot
    ax.plot(depot_pos[0], depot_pos[1], 'r*', markersize=20, label='Depot', zorder=5)
    
    # Colors for different routes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 1)))
    
    # Plot each route
    for route_idx, route in enumerate(routes):
        if not route:
            continue
        
        color = colors[route_idx % len(colors)]
        
        # Get positions for this route
        route_points = [depot_pos]
        for cust_idx in route:
            if cust_idx > 0 and cust_idx <= len(customers):
                route_points.append(customers[cust_idx - 1])
        route_points.append(depot_pos)
        
        # Plot route as lines
        xs = [p[0] for p in route_points]
        ys = [p[1] for p in route_points]
        ax.plot(xs, ys, '-', color=color, linewidth=2, label=f'Route {route_idx + 1}')
        
        # Plot customers on this route
        for cust_idx in route:
            if cust_idx > 0 and cust_idx <= len(customers):
                ax.plot(customers[cust_idx - 1][0], customers[cust_idx - 1][1], 
                       'o', color=color, markersize=6)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_convergence(histories_dict, metric_type="time", max_val=None):
    """
    Plots convergence curves.
    metric_type: "time" or "accepted_neighbors"
    max_val: The maximum x-axis value (time or accepted_neighbors) to stretch lines to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    has_data = False
    
    for exp_name, runs in histories_dict.items():
        if not runs:
            continue
            
        best_run_idx = -1
        best_run_final_cost = float('inf')
        
        for idx, history in enumerate(runs):
            if history and history[-1][2] < best_run_final_cost:
                best_run_final_cost = history[-1][2]
                best_run_idx = idx
                
        if best_run_idx != -1:
            has_data = True
            history = runs[best_run_idx]
            
            x_data = []
            y_data = []
            
            for pt in history:
                if metric_type == "time":
                    x_data.append(pt[0])
                else:
                    x_data.append(pt[1])
                y_data.append(pt[2])
            
            if max_val is not None and x_data and x_data[-1] < max_val:
                x_data.append(max_val)
                y_data.append(y_data[-1])
                
            ax.step(x_data, y_data, where='post', label=exp_name, linewidth=2)
    
    if not has_data:
        ax.text(0.5, 0.5, "No convergence data available\n(No solutions found)", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
            
    ax.set_xlabel("Time (s)" if metric_type == "time" else "Accepted Neighbors")
    ax.set_ylabel("Cost")
    ax.set_title(f"Convergence over {metric_type.capitalize().replace('_', ' ')}")
    if has_data:
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


def plot_convergence_comparison(all_histories, exp_names, title="Convergence Comparison"):
    """
    Plot convergence curves for multiple experiments (using best run from each).
    all_histories: Dict mapping exp_name -> list of history lists (one per rep)
    """
    # Increased width and height for better visibility
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(exp_names), 1)))
    
    # 1. First, find best run for each experiment
    best_histories = {}
    max_time = 0
    max_neighbors = 0
    
    for exp_name in exp_names:
        runs = all_histories.get(exp_name, [])
        if not runs: continue
        
        best_run = None
        min_cost = float('inf')
        for run in runs:
            if run and run[-1][2] < min_cost:
                min_cost = run[-1][2]
                best_run = run
        
        if best_run:
            best_histories[exp_name] = best_run
            max_time = max(max_time, best_run[-1][0])
            max_neighbors = max(max_neighbors, best_run[-1][1])

    # 2. Plot
    for idx, exp_name in enumerate(exp_names):
        history = best_histories.get(exp_name)
        if not history: continue
        
        times = [h[0] for h in history]
        neighbors = [h[1] for h in history]
        costs = [h[2] for h in history]
        
        # Extend to max for visual consistency in step plot
        if times[-1] < max_time:
            times.append(max_time)
            costs.append(costs[-1])
            neighbors.append(neighbors[-1]) # Neighbors don't increase while waiting
            
        color = colors[idx % len(colors)]
        
        # Time Plot
        ax1.step(times, costs, where='post', label=exp_name, color=color, linewidth=2)
        
        # Neighbors Plot
        ax2.step(neighbors, costs, where='post', label=exp_name, color=color, linewidth=2)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Best Cost')
    ax1.set_title('Cost Over Time')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Accepted Neighbors')
    ax2.set_ylabel('Best Cost')
    ax2.set_title('Cost Over Accepted Neighbors')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Increase space between subplots
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    return fig


def plot_single_benchmark_results(instance_data, all_histories, all_best_routes, exp_names):
    """
    Create a comprehensive visualization for single benchmark results.
    """
    if not all_histories:
        st.warning("No convergence history data available for visualization")
        return
    
    # Get customer locations
    coordinates = instance_data.get('coordinates', {})
    customers = []
    depot_pos = coordinates.get(0, (0, 0))
    if coordinates:
        num_nodes = instance_data.get('num_nodes', 0)
        for i in range(1, num_nodes):
            if i in coordinates:
                customers.append(coordinates[i])
    
    # Show convergence comparison
    st.subheader("📊 Convergence Analysis")
    
    fig_conv = plot_convergence_comparison(all_histories, exp_names, "Convergence Comparison")
    # Make convergence analysis even bigger (95% width)
    col_l, col_m, col_r = st.columns([0.025, 0.95, 0.025])
    with col_m:
        st.pyplot(fig_conv, use_container_width=True)
    
    # Add spacing between sections
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Show routes for each algorithm
    if all_best_routes and customers:
        st.subheader("🗺️ Best Routes by Algorithm")
        
        # Create tabs for each algorithm
        tabs = st.tabs(exp_names)
        
        for tab, exp_name in zip(tabs, exp_names):
            with tab:
                if exp_name in all_best_routes and all_best_routes[exp_name]:
                    routes = all_best_routes[exp_name]
                    
                    # Calculate cost for this route (from best run)
                    runs = all_histories.get(exp_name, [])
                    best_cost = None
                    if runs:
                        min_c = float('inf')
                        for r in runs:
                            if r and r[-1][2] < min_c:
                                min_c = r[-1][2]
                        best_cost = min_c
                    
                    if best_cost is not None:
                        st.metric("Best Cost Found", f"{best_cost:.0f}")
                    
                    fig_routes = plot_routes(customers, routes, depot_pos=depot_pos, title=f"Routes - {exp_name}")
                    # Realign to the beginning (left) and make it smaller (50% width)
                    col_l2, col_r2 = st.columns([0.5, 0.5])
                    with col_l2:
                        st.pyplot(fig_routes, use_container_width=True)
                else:
                    st.info(f"No route data for {exp_name}")
