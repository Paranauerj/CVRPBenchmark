# OR-Tools Gaetano Benchmark: Dynamic BKS Generation Plan

This document outlines the strategy for benchmarking Google OR-Tools on the Gaetano instance set using dynamic time limits and solution progress tracking.

## 1. Objective
The goal is to evaluate OR-Tools performance on a representative sample of 1,000 randomly selected Gaetano instances using a standardized methodology that accounts for instance size and monitors convergence over time.

## 2. Why Google OR-Tools?
Google OR-Tools is one of the most widely used open-source libraries for combinatorial optimization. Benchmarking it against the HGS-CVRP (Best Known Solutions) is crucial because:
*   **Industry Standard:** It is a go-to tool for real-world routing applications.
*   **Order Sensitivity:** By using customer permutations, we evaluate its robustness. For deterministic solvers like OR-Tools, input permutation effectively acts as a search seed.
*   **Performance vs. Quality:** It provides a strong baseline for the trade-off between computation time and solution quality in dynamic LMD.

## 3. Methodology

### Dynamic Time Limit
The time limit $T$ for each instance is determined by the number of customers $N$:
$$T = N \times 0.5 \text{ seconds}$$

### Statistical Significance
*   **Repetitions:** 3 independent runs per experiment.
*   **Randomization:** Customer permutation per run. This ensures each repetition explores a different part of the search space starting from a different constructive solution.

### Progress Monitoring
We record the best and average solution values at every second of execution to analyze convergence continuously over time:
*   **Second-by-Second Benchmarks:** Recorded at every 1 second ($1\text{s}, 2\text{s}, 3\text{s}, \dots, \lfloor T \rfloor\text{s}$, and at the exact time limit $T$) until the dynamic time limit is reached.
*   **Target Gap:** We track the exact time required to reach a **5% gap** relative to BKS.

## 4. Configuration Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Time Limit (T)** | $N \times 0.5$s | Dynamic limit based on customer count. |
| **Repetitions** | 3 | Independent runs per experiment. |
| **Randomization** | Permutation | Shuffling customers per repetition. |
| **Parallel Workers** | 8 | Concurrent instances solved. |
| **Scope** | 1,000 | 1,000 randomly sampled Gaetano instances. |
| **Progress Logging** | Every 1 second | Second-by-second cost recording ($1\text{s}, 2\text{s}, \dots$) until time limit $T$. |

### Solver Strategies (OR-Tools)
We test all combinations of:
*   **First Solution:** `SAVINGS`, `PARALLEL_CHEAPEST_INSERTION`, `CHRISTOFIDES`.
*   **Metaheuristics:** `GUIDED_LOCAL_SEARCH`, `TABU_SEARCH`, `SIMULATED_ANNEALING`.

## 5. Execution Strategy

### Hardware Allocation
Each of the 8 parallel workers is restricted to a single thread (`OMP_NUM_THREADS=1`).

### Execution Command
```bash
source venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nohup python or_tools_gaetano_benchmark.py > ortools_benchmark.log 2>&1 &
```

## 6. Time Estimation (Reduced Set: 1,000 Instances)

*   **Instance Profile:** 1,000 randomly sampled Gaetano instances.
*   **Combinations:** 9 strategy combos $\times$ 3 reps = **27,000 total runs**.
*   **Total Clock Time (8 Workers):** **~54 hours (approx. 2.25 days)**.

## 7. Reliability & Resumption
*   **Chunking:** Results are saved every 10 instances in `gaetano_ortools_results`.
*   **Resumption:** Automatically resumes from the last incomplete chunk.
