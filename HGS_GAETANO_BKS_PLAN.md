# HGS Gaetano Benchmark: BKS Generation Plan

This document outlines the strategy and configuration for generating Best Known Solutions (BKS) for the Gaetano instance set using the Hybrid Genetic Search (HGS-CVRP) solver.

## 1. Objective
The goal is to establish high-quality, high-confidence baseline results for the complete set of **10,000 CVRP instances** in the Gaetano collection. These results will serve as the "Best Known Solutions" (BKS) for future comparative studies against other solvers (like OR-Tools).

## 2. Why HGS-CVRP?
The Hybrid Genetic Search (HGS) for the CVRP, originally proposed by Vidal et al., is widely considered one of the state-of-the-art metaheuristics for this problem. It is used in this project because:
*   **Superior Solution Quality:** It consistently finds optimal or near-optimal solutions across diverse benchmark sets.
*   **Efficiency:** Its advanced population management and local search (including the specialized "Split" algorithm) make it highly effective for instances with hundreds of customers.
*   **Standard for BKS:** It is the industry-standard tool for establishing high-quality reference baselines in academic research.

## 3. Configuration Parameters

The benchmark is configured in `hgs_gaetano_benchmark.py` with the following parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Time Limit** | 600s (10 min) | Maximum wall-clock time per repetition. |
| **Repetitions** | 3 | Independent runs with different random seeds. |
| **No-Improvement Limit** | 1,000,000,000 | Forces the solver to use the full time window. |
| **Parallel Workers** | 8 | Concurrent instances solved (one per core). |
| **Instances** | 10,000 | Total count of Gaetano instances in `instances/gaetano`. |

## 4. Execution Strategy

### Hardware Allocation
To prevent CPU contention and ensure reproducible results, each of the 8 workers will be restricted to a single thread. This maximizes throughput by solving 8 instances simultaneously without them "fighting" for the same CPU resources.

### Execution Commands

#### Windows (PowerShell)
```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; python hgs_gaetano_benchmark.py
```

#### Linux (Bash)
Ensure the environment is set up using `setup_linux.sh` first. For long-running benchmarks, use `nohup` to keep the process running after closing the SSH session:

```bash
source venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nohup python hgs_gaetano_benchmark.py > benchmark.log 2>&1 &
```

**Monitoring progress:**
```bash
tail -f benchmark.log
```

## 5. Time Estimation

*   **Total Instance-Minutes:** 10,000 instances × 3 reps × 10 mins = **300,000 minutes**.
*   **Total Clock Time (8 Workers):** 300,000 ÷ 8 = **37,500 minutes**.
*   **Total Duration:** **~625 hours (approx. 26 days)**.

*Note: The script is configured to skip already solved instances (`SOLVE_ONLY_UNSOLVED = True`). If 500 instances are already solved, the remaining 9,500 will take ~24.7 days.*

## 6. Reliability & Resumption
*   **Chunking:** Results are saved every 5 instances (`CHUNK_SIZE = 5`).
*   **Resumption:** If interrupted, the `BenchmarkRunner` will automatically detect existing `chunk_*.json` files in `hgs_gaetano_results` and resume from the last incomplete chunk.
*   **Solution Saving:** The script automatically updates `.sol` files in the instances directory only if a new best-ever cost is found.

## 7. Output
*   **Excel Report:** A consolidated report will be generated in `server_output/` upon completion.
*   **Solution Files:** Individual `.sol` files for each instance will be created/updated in `instances/gaetano/`.
