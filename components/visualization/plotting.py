"""Route and convergence visualization for benchmark results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from collections import defaultdict


def plot_routes(customers, routes, depot_pos=(0, 0), title="Routes"):
    """
    Plot routes on 2D space.
    
    Args:
        customers: List of (x, y) tuples for customer positions
        routes: List of routes, where each route is a list of customer indices
        depot_pos: Position of the depot (default 0,0)
        title: Title for the plot
    
    Returns:
        matplotlib figure
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
    ax.set_title(f"Convergence over {metric_type.capitalize()}")
    if has_data:
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


def plot_convergence_comparison(all_histories, exp_names, title="Convergence Comparison"):
    """
    Plot convergence curves for multiple experiments.
    
    Args:
        all_histories: Dict mapping exp_name -> list of (time, accepted_neighbors, cost) tuples
        exp_names: List of experiment names in order
        title: Title for the plot
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))
    
    # Find max time across all histories (for extending lines to end of plot)
    max_time = 0
    for exp_name in exp_names:
        if exp_name in all_histories and all_histories[exp_name]:
            history = all_histories[exp_name]
            if history:
                max_time = max(max_time, history[-1][0])
    
    for idx, exp_name in enumerate(exp_names):
        if exp_name not in all_histories or not all_histories[exp_name]:
            continue
        
        history = all_histories[exp_name]
        times = [h[0] for h in history]      # First element: time
        costs = [h[2] for h in history]      # Third element: cost
        
        # Extend to end of time axis for step plot
        if history and times[-1] < max_time:
            times = times + [max_time]
            costs = costs + [costs[-1]]
        
        # Convert to "best cost found so far" (running minimum)
        best_so_far = []
        min_cost = float('inf')
        for cost in costs:
            min_cost = min(min_cost, cost)
            best_so_far.append(min_cost)
        
        color = colors[idx % len(colors)]
        ax1.plot(times, best_so_far, '-o', label=exp_name, color=color, linewidth=2, markersize=4, drawstyle='steps-post')
        
        # Also plot accepted neighbors on second axis
        accepted_neighbors = list(range(len(best_so_far)))
        ax2.plot(accepted_neighbors, best_so_far, '-o', label=exp_name, color=color, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Best Cost Found')
    ax1.set_title('Cost Over Time')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Accepted Neighbors')
    ax2.set_ylabel('Best Cost Found')
    ax2.set_title('Cost Over Accepted Neighbors')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_single_benchmark_results(instance_data, all_histories, all_best_routes, exp_names):
    """
    Create a comprehensive visualization for single benchmark results.
    
    Args:
        instance_data: Instance data dict with 'coordinates' (dict mapping node_id -> (x, y))
        all_histories: Dict mapping exp_name -> list of history lists (each history is list of (time, cost) tuples)
        all_best_routes: Dict mapping exp_name -> best routes (list of lists of customer indices)
        exp_names: List of experiment names
    
    Returns:
        None (shows content in Streamlit)
    """
    if not all_histories:
        st.warning("No convergence history data available for visualization")
        return
    
    # Get customer locations (skip depot at index 0)
    coordinates = instance_data.get('coordinates', {})
    customers = []
    depot_pos = coordinates.get(0, (0, 0))  # Extract actual depot position
    if coordinates:
        # Extract customer coordinates (indices 1 to n), skip depot (index 0)
        num_nodes = instance_data.get('num_nodes', 0)
        for i in range(1, num_nodes):
            if i in coordinates:
                customers.append(coordinates[i])
    
    # Show convergence comparison
    st.subheader("📊 Convergence Analysis")
    
    # Build consolidated history for comparison using best run
    consolidated_histories = {}
    for exp_name, history_list in all_histories.items():
        # all_histories[exp_name] is a list of histories (one per repetition)
        # Select the best run (lowest final cost)
        if history_list and len(history_list) > 0:
            best_history = None
            best_final_cost = float('inf')
            
            for history in history_list:
                if history and len(history) > 0:
                    final_cost = history[-1][1]  # Last tuple's cost
                    if final_cost < best_final_cost:
                        best_final_cost = final_cost
                        best_history = history
            
            if best_history:
                consolidated_histories[exp_name] = best_history
    
    if consolidated_histories:
        fig_conv = plot_convergence_comparison(consolidated_histories, exp_names, "Convergence Comparison")
        st.pyplot(fig_conv, width='stretch')
    else:
        st.warning("No convergence history available.")
        return
    
    # Show routes for each algorithm
    if all_best_routes and customers:
        st.subheader("🗺️ Best Routes by Algorithm")
        
        # Create tabs for each algorithm
        tabs = st.tabs(exp_names)
        
        for tab, exp_name in zip(tabs, exp_names):
            with tab:
                if exp_name in all_best_routes and all_best_routes[exp_name]:
                    routes = all_best_routes[exp_name]
                    
                    # Calculate cost for this route
                    if exp_name in consolidated_histories and consolidated_histories[exp_name]:
                        best_cost = consolidated_histories[exp_name][-1][2]  # Last tuple's cost (index 2)
                        st.metric("Best Cost Found", f"{best_cost:.2f}")
                    
                    fig_routes = plot_routes(customers, routes, depot_pos=depot_pos, title=f"Routes - {exp_name}")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.pyplot(fig_routes)
                else:
                    st.info(f"No route data for {exp_name}")
