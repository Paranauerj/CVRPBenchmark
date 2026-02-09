"""Plotting and visualization functions."""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_routes(instance_data, routes, title="Routes"):
    """Plot VRP routes on a map."""
    coords = instance_data.get('coordinates', {})
    if not coords:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No coordinates available", ha='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 6))
    depot_id = instance_data.get('depot', 0)
    
    if depot_id in coords:
        depot_pos = coords[depot_id]
        ax.scatter(depot_pos[0], depot_pos[1], c='red', s=100, marker='s', label='Depot', zorder=10)
    
    x_vals = [pos[0] for idx, pos in coords.items() if idx != depot_id]
    y_vals = [pos[1] for idx, pos in coords.items() if idx != depot_id]
    ax.scatter(x_vals, y_vals, c='gray', s=10, alpha=0.5)
    
    if routes:
        colors = cm.rainbow(np.linspace(0, 1, len(routes)))
        for route, color in zip(routes, colors):
            full_route = [depot_id] + route + [depot_id]
            route_x = [coords[n][0] for n in full_route if n in coords]
            route_y = [coords[n][1] for n in full_route if n in coords]
            ax.plot(route_x, route_y, c=color, linewidth=1.5, alpha=0.8)
    
    ax.set_title(title)
    ax.legend()
    return fig


def plot_convergence(histories_dict, metric_type="time", max_val=None):
    """
    Plots convergence curves.
    metric_type: "time" or "iterations"
    max_val: The maximum x-axis value (time or iterations) to stretch lines to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    has_data = False
    
    for algo_name, runs in histories_dict.items():
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
            
            if max_val is not None and x_data[-1] < max_val:
                x_data.append(max_val)
                y_data.append(y_data[-1])
                
            ax.step(x_data, y_data, where='post', label=algo_name, linewidth=2)
    
    if not has_data:
        ax.text(0.5, 0.5, "No convergence data available\n(No solutions found)", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
            
    ax.set_xlabel("Time (s)" if metric_type == "time" else "Iterations")
    ax.set_ylabel("Cost")
    ax.set_title(f"Convergence over {metric_type.capitalize()}")
    if has_data:
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig
