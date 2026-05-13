# HGS Gaetano Benchmark: BKS Generation Plan

This document outlines the strategy and configuration for generating Best Known Solutions (BKS) for the Gaetano instance set using the Hybrid Genetic Search (HGS-CVRP) solver.

## 1. Objective
The goal is to establish high-quality, high-confidence baseline results for ~500 CVRP instances. These results will serve as the "Best Known Solutions" (BKS) for future comparative studies against other solvers (like OR-Tools).

## 2. Configuration Parameters

The benchmark is configured in `hgs_gaetano_benchmark.py` with the following parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Time Limit** | 600s (10 min) | Maximum wall-clock time per repetition. |
| **Repetitions** | 3 | Independent runs with different random seeds. |
| **No-Improvement Limit** | 1,000,000,000 | Forces the solver to use the full time window. |
| **Parallel Workers** | 8 | Concurrent instances solved (one per core). |
| **Instances** | ~500 | Total count of Gaetano instances in `instances/gaetano`. |

## 3. Execution Strategy

### Hardware Allocation
To prevent CPU contention and ensure reproducible results, each of the 8 workers will be restricted to a single thread. This maximizes throughput by solving 8 instances simultaneously without them "fighting" for the same CPU resources.

### Recommended Command
Run the benchmark via the CLI using a "Master Environment" setup to ensure stable threading:

```powershell
# Windows PowerShell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; python hgs_gaetano_benchmark.py
```

## 4. Time Estimation

*   **Total Instance-Minutes:** 500 instances × 3 reps × 10 mins = **15,000 minutes**.
*   **Total Clock Time (8 Workers):** 15,000 ÷ 8 = **1,875 minutes**.
*   **Total Duration:** **~31 hours and 15 minutes**.

## 5. Reliability & Resumption
*   **Chunking:** Results are saved every 5 instances (`CHUNK_SIZE = 5`).
*   **Resumption:** If interrupted, the `BenchmarkRunner` will automatically detect existing `chunk_*.json` files in `hgs_gaetano_results` and resume from the last incomplete chunk.
*   **Solution Saving:** The script automatically updates `.sol` files in the instances directory only if a new best-ever cost is found.

## 6. Output
*   **Excel Report:** A consolidated report will be generated in `server_output/` upon completion.
*   **Solution Files:** Individual `.sol` files for each instance will be created/updated in `instances/gaetano/`.
