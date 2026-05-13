import os
import glob
import pandas as pd
from components.utils import instance_data_parser
from components.execution.benchmark_common import extract_instance_metadata
from datetime import datetime

# Configuration
INSTANCES_DIRS = ["instances/gaetano", "instances/uchoa"]
OUTPUT_DIR = "server_output"

def main():
    """
    Scans instance directories and generates a comprehensive feature set file.
    This includes metadata like Depot Layout, Climate, etc., but NO performance metrics.
    """
    all_features = []
    
    print("🚀 Starting instance feature extraction...")
    
    for directory in INSTANCES_DIRS:
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found, skipping: {directory}")
            continue
            
        print(f"📂 Scanning directory: {directory}")
        vrp_files = sorted(glob.glob(os.path.join(directory, "*.vrp")))
        
        for vrp_path in vrp_files:
            inst_name = os.path.basename(vrp_path).replace(".vrp", "")
            
            try:
                # Load instance data to get capacity/vehicles
                inst_data = instance_data_parser.load_vrp_instance(vrp_path)
                
                # Extract metadata features (Layout, Climate, etc.)
                meta = extract_instance_metadata(inst_data, inst_name)
                
                # Look for BKS
                bks_val = None
                sol_path = vrp_path.replace(".vrp", ".sol")
                if os.path.exists(sol_path):
                    try:
                        from components.utils import solution_parser
                        bks_val = solution_parser.parse_solution_file(sol_path)
                    except:
                        pass

                # Create a feature-only row
                row = {
                    "Instance": inst_name,
                    "Source": os.path.basename(directory),
                    "Customers": meta.customers,
                    "Vehicles": meta.vehicles,
                    "Capacity": meta.capacity,
                    "Depot Layout": meta.depot_layout,
                    "Customer Layout": meta.customer_layout,
                    "Demand Type": meta.demand_profile,
                    "Route Class": meta.route_class,
                    "Climate": meta.climate,
                    "BKS": bks_val
                }
                all_features.append(row)
                print(f"  ✅ Extracted: {inst_name}")
            except Exception as e:
                print(f"  ❌ Error processing {inst_name}: {e}")

    if all_features:
        df = pd.DataFrame(all_features)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"instances_features_set_{timestamp}.xlsx")
        
        df.to_excel(output_path, index=False, engine='xlsxwriter')
        print(f"\n✨ Success! Feature set saved to: {output_path}")
        print(f"Total instances documented: {len(all_features)}")
    else:
        print("\n❌ No instances found or processed.")

if __name__ == "__main__":
    main()
