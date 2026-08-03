import os
import glob
import copy
from components.execution.benchmark_runner import BenchmarkRunner
from components.execution.hygese_solver import solve_hgs
from components.execution.benchmark_common import (
    extract_instance_metadata, run_experiment_reps, build_result_row
)
from components.models import ExperimentConfig
from components.utils import instance_data_parser, solution_parser
from components import constants as C

# --- Configuration ---
INSTANCES_DIR = "instances/gaetano"
RESULTS_DIR = "bks_retry_results"
NUM_PARALLEL = 8               # Concurrent workers
TIME_LIMIT_SECONDS = 600       # 10 minutes per repetition
REPETITIONS = 3                # 3 independent runs per vehicle attempt
START_ADDITIONAL_VEHICLES = 10 # Start from base vehicles + 10 (current max additional vehicles)
MAX_EXTRA_VEHICLES_TO_TRY = 10 # Try up to base vehicles + 20 (10 extra beyond current)

# The 10 known instances missing BKS / .sol files
UNSOLVED_INSTANCES = [
    "LDG130_2271_snow_130_0274.vrp",
    "LDG150_1371_fog_150_0111.vrp",
    "LDG170_2271_rain_170_0064.vrp",
    "LDG170_3371_rain_170_0226.vrp",
    "LDG175_1371_fog_175_0206.vrp",
    "LDG180_2371_fog_180_0120.vrp",
    "LDG180_3171_rain_180_0112.vrp",
    "LDG195_3271_none_195_0202.vrp",
    "LDG200_1171_rain_200_0008.vrp",
    "LDG200_3271_rain_200_0006.vrp",
]

def get_unsolved_instance_paths(instances_dir: str) -> list[str]:
    """
    Finds all .vrp files in instances_dir that do NOT have a corresponding .sol file.
    Falls back to UNSOLVED_INSTANCES if present.
    """
    missing = []
    all_vrps = sorted(glob.glob(os.path.join(instances_dir, "*.vrp")))
    for vrp_path in all_vrps:
        sol_path = vrp_path.replace(".vrp", ".sol")
        if not os.path.exists(sol_path):
            missing.append(vrp_path)
    return missing

def get_hgs_experiment() -> ExperimentConfig:
    """Creates the HGS experiment configuration for finding BKS."""
    kwargs = {
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "no_improvement_limit_iterations": 1000000000,
    }
    return ExperimentConfig(
        name="HGS-CVRP (BKS Search)",
        fs_label="HGS-CVRP",
        mh_label="HGS-CVRP",
        func=solve_hgs,
        kwargs=kwargs,
        reps=REPETITIONS
    )

def process_unsolved_instance(vrp_path: str, experiments: list[ExperimentConfig]):
    """
    Processes an instance by iteratively trying vehicle counts starting from 
    base_vehicles + START_ADDITIONAL_VEHICLES up to START_ADDITIONAL_VEHICLES + MAX_EXTRA_VEHICLES_TO_TRY.
    """
    inst_name = os.path.basename(vrp_path).replace(".vrp", "")
    sol_path = vrp_path.replace(".vrp", ".sol")
    exp = experiments[0]

    try:
        inst_data = instance_data_parser.load_vrp_instance(vrp_path)
        instance_meta = extract_instance_metadata(inst_data, inst_name)
        base_vehicles = inst_data.get('num_vehicles', 0)

        print(f"🔍 Searching BKS for {inst_name} | Base vehicles: {base_vehicles}")
        
        # Try additional vehicles starting from START_ADDITIONAL_VEHICLES (+10)
        # up to START_ADDITIONAL_VEHICLES + MAX_EXTRA_VEHICLES_TO_TRY (+20)
        min_add = START_ADDITIONAL_VEHICLES
        max_add = START_ADDITIONAL_VEHICLES + MAX_EXTRA_VEHICLES_TO_TRY

        for add_v in range(min_add, max_add + 1):
            num_vehicles_attempt = base_vehicles + add_v
            
            working_instance = copy.deepcopy(inst_data)
            working_instance['num_vehicles'] = num_vehicles_attempt
            working_instance['vehicle_capacities'] = [working_instance['capacity']] * num_vehicles_attempt
            
            current_meta = copy.copy(instance_meta)
            current_meta.vehicles = num_vehicles_attempt

            print(f"  Attempting {inst_name} with {num_vehicles_attempt} vehicles (+{add_v} extra)...")

            data = run_experiment_reps(
                exp, working_instance, exp.reps, bks_cost=None, use_permutations=False
            )

            if data.costs and data.best_routes:
                best_cost = min(data.costs)
                routes_used = len(data.best_routes)
                print(f"  ✅ SUCCESS: Found solution for {inst_name} with {num_vehicles_attempt} vehicles! Best cost: {best_cost:.2f} (used {routes_used} routes)")
                
                # Save .sol file
                print(f"  💾 Saving solution to {sol_path}")
                solution_parser.save_solution_file(sol_path, data.best_routes, best_cost)

                result_row = build_result_row(
                    exp, current_meta, data.costs, data.times,
                    data.neighbors_list, data.best_routes, best_cost, data.checkpoints, engine="hgs"
                )
                return [result_row.to_dict()]

        print(f"  ❌ FAILED: No feasible solution found for {inst_name} up to +{max_add} extra vehicles.")
        return []

    except Exception as e:
        print(f"Error processing {inst_name}: {e}")
        return []

def main():
    """Main execution entry point."""
    missing_files = get_unsolved_instance_paths(INSTANCES_DIR)
    print(f"Found {len(missing_files)} instances missing Best Known Solutions (.sol files):")
    for f in missing_files:
        print(f"  - {os.path.basename(f)}")

    if not missing_files:
        print("All instances already have .sol files!")
        return

    runner = BenchmarkRunner(
        name="Find BKS For Unknown Gaetano Instances",
        instances_dir=INSTANCES_DIR,
        results_dir=RESULTS_DIR,
        max_instances=len(missing_files),
        num_parallel=NUM_PARALLEL,
        chunk_size=1
    )

    runner.run(
        experiments=[get_hgs_experiment()],
        process_instance_fn=process_unsolved_instance,
        instance_list=[(os.path.basename(p).replace(".vrp", ""), {"vrp": p}) for p in missing_files]
    )

if __name__ == "__main__":
    main()
