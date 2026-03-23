# CVRP Benchmarker

A comprehensive Streamlit web application for benchmarking Capacitated Vehicle Routing Problem (CVRP) solvers with dynamic algorithm configuration, route visualization, and convergence analysis.

## Features

✨ **Interactive Web Interface** — Streamlit-based UI with intuitive tabs and real-time results  
🚀 **Single Instance Benchmarking** — Run algorithms on individual instances with parametric control  
📊 **Bulk Benchmarking** — Benchmark multiple instances in parallel with progress tracking  
📈 **Convergence Visualization** — Plot cost improvements over time and iterations  
🗺️ **Route Visualization** — Display vehicle routes with color-coded vehicles  
⚙️ **Configurable Solvers** — Support for multiple first solution strategies and metaheuristics  
🔄 **Background Execution** — Run benchmarks asynchronously without blocking the UI  
💾 **Automatic Exports** — Save results to Excel with comprehensive metrics  

## Architecture

```
components/
  ├── execution/
  │   ├── configurable_solver.py    # OR-Tools wrapper with SmartLimitCallback
  │   ├── benchmark_common.py        # Shared benchmark functions
  │   ├── single_benchmark.py        # Single instance execution
  │   ├── bulk_benchmark.py          # Multi-instance with ThreadPoolExecutor
  │   ├── background_task.py         # Async task management
  │   └── task_manager.py            # Task persistence
  ├── ui/
  │   ├── sidebar.py                 # Shared algorithm/limit parameters
  │   ├── results_display.py         # Results table and checkpoint analysis
  │   └── monitoring.py              # Task progress tracking
  ├── utils/
  │   ├── benchmark_utils.py         # execute_and_measure() wrapper
  │   ├── instance_data_parser.py    # VRP file parsing
  │   ├── solution_parser.py         # Best known solution parsing
  │   └── helpers.py                 # Utility functions
  └── visualization/
      └── plotting.py                # Route and convergence visualizations
instances/
  ├── gaetano/                       # Gaetano benchmark instances
  └── uchoa/                         # Uchoa benchmark instances
app.py                               # Main Streamlit application
requirements.txt                     # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit` — Web framework
- `ortools` — Google OR-Tools solver
- `pandas` — Data manipulation
- `matplotlib` — Visualization
- `vrplib` — VRP instance parsing

### 2. Run the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### 3. Benchmark an Instance

**Single Benchmark Tab:**
1. Select instance source (Uchoa/Gaetano)
2. Choose an instance
3. Configure algorithms (First Solution + Metaheuristic)
4. Set execution limits (time, solutions, LNS time)
5. Run synchronously or in background
6. View convergence plots and route visualization

**Bulk Benchmark Tab:**
1. Multi-select instances from Gaetano
2. Configure parallel workers and repetitions
3. Run in background
4. Monitor progress in the Monitor tab

## Key Components

### Shared Sidebar (`components/ui/sidebar.py`)
- Algorithm selection: First Solution strategies & Metaheuristics
- Execution limits: Time, solution count, LNS time
- Stopping conditions: No-improvement timeouts

### Benchmark Common (`components/execution/benchmark_common.py`)
Four core functions shared between single/bulk execution:
- `prepare_experiments(settings)` — Build experiment configurations
- `extract_instance_metadata(instance_data, instance_name)` — Parse instance features
- `build_result_row(...)` — Create 31-column result records
- `run_experiment_reps(...)` — Execute repetitions with checkpoint collection

### Convergence Tracking
- **Time checkpoints:** [5, 10, 15, 20, 30, 60] seconds
- **History format:** (time_elapsed, iterations, cost) tuples
- **Best-of-reps:** Visualization shows best iteration (lowest final cost)
- **Step-function plot:** Cost stays constant until next improvement

### Results Structure (31 Columns)
```
Metadata (11):
  Instance, Depot Layout, Customer Layout, Demand Type, Route Class,
  Climate, Customers, Vehicles, Capacity, First Solution, Metaheuristic

Metrics (8):
  Repetitions, Best Cost, Avg Cost, BKS Cost, Best Gap (%), Avg Gap (%),
  Avg CPU Time (s), (Reserved)

Time Checkpoints (12):
  Best Cost @ 5s, Best Cost @ 10s, ..., Best Cost @ 60s,
  Avg Cost @ 5s, ... (compressed to 12 columns)
```

## Vehicle Retry Logic

If an instance fails with the initial vehicle count, the solver automatically retries with:
- +1 vehicle (attempt 1)
- +2 vehicles (attempt 2)
- ... up to +5 vehicles (attempt 5)

Stops on first successful solution.

## Results Display

After benchmarking, results show:
1. **Benchmark Results Table** — Key metrics with formatting
2. **Convergence Analysis** — Time checkpoint costs
3. **Convergence Plots** — Cost over time + Cost over iterations
4. **Route Visualization** — Tabbed view per algorithm

## Background Tasks

Tasks automatically saved to `server_output/`:
- `tasks.json` — Task registry
- `latest_task.json` — Most recent task status
- Excel files — Benchmark results (one per task)

## Supported Algorithms

**First Solution Strategies:**
- Automatic, Path Cheapest Arc, Path Most Constrained Arc
- Savings (Clarke-Wright), Christofides
- Parallel/Local Cheapest Insertion, Local Cheapest Arc
- First Unbound Min Value

**Metaheuristics:**
- Automatic, Greedy Descent
- Guided Local Search (GLS)
- Simulated Annealing
- Tabu Search, Generic Tabu Search

## Configuration Examples

### Quick Benchmark (20 seconds, 3 reps)
- First Solution: Parallel Cheapest Insertion
- Metaheuristic: Guided Local Search
- Time Limit: 20 seconds
- Repetitions: 3

### Intensive Search (60 seconds, 10 reps)
- First Solution: Christofides
- Metaheuristic: Tabu Search
- Time Limit: 60 seconds
- Solution Limit: 5000
- Repetitions: 10

## Requirements

- Python 3.8+
- OR-Tools 9.0+
- Streamlit 1.20+
- pandas 1.3+
- matplotlib 3.4+

## License

Specify your license here (MIT, Apache 2.0, etc.)
