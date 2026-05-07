import os
import glob
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from components.utils.logging_utils import setup_logger

class BenchmarkRunner:
    """
    Standardizes the orchestration of large-scale benchmarks.
    Handles instance loading, parallel execution, chunking, and reporting.
    """
    
    def __init__(self, name, instances_dir, results_dir, output_dir="server_output", 
                 chunk_size=10, num_parallel=2, max_instances=None):
        self.name = name
        self.instances_dir = instances_dir
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.num_parallel = num_parallel
        self.max_instances = max_instances
        
        self.logger = setup_logger(name)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_instance_files(self):
        """Returns a list of all .vrp files in the instances directory."""
        files = sorted(glob.glob(os.path.join(self.instances_dir, "*.vrp")))
        if self.max_instances:
            files = files[:self.max_instances]
        return files

    def run(self, experiments, process_instance_fn, time_checkpoints=None, filter_cols=None):
        """
        Runs the benchmark.
        process_instance_fn: function(vrp_path, experiments, **kwargs) -> list of result dicts
        """
        instance_files = self.get_instance_files()
        if not instance_files:
            self.logger.error(f"No instances found in {self.instances_dir}")
            return None

        self.logger.info(f"🚀 Starting Benchmark: {self.name}")
        self.logger.info(f"Total instances: {len(instance_files)}. Chunk size: {self.chunk_size}.")

        chunks = [instance_files[i:i + self.chunk_size] for i in range(0, len(instance_files), self.chunk_size)]
        
        # Resumption logic
        start_chunk_idx = self._get_resume_index()
        
        try:
            for i in range(start_chunk_idx, len(chunks)):
                self._run_chunk(i, chunks[i], experiments, process_instance_fn)
                
            output_path = self.aggregate_results(filter_cols=filter_cols)
            if output_path:
                self.cleanup()
                self.logger.info(f"✨ Benchmark '{self.name}' finished successfully!")
                return output_path
                
        except KeyboardInterrupt:
            self.logger.warning("Terminated by user. You can resume later.")
        except Exception as e:
            self.logger.exception(f"Benchmark failed: {e}")
            
        return None

    def _get_resume_index(self):
        existing_chunks = sorted(glob.glob(os.path.join(self.results_dir, "chunk_*.json")))
        if existing_chunks:
            latest_chunk_file = existing_chunks[-1]
            latest_idx = int(os.path.basename(latest_chunk_file).split('_')[1].split('.')[0])
            self.logger.info(f"Found existing results up to chunk {latest_idx}. Resuming...")
            return latest_idx
        return 0

    def _run_chunk(self, chunk_id, chunk_files, experiments, process_instance_fn):
        self.logger.info(f"📦 Starting Chunk {chunk_id} ({len(chunk_files)} instances)...")
        chunk_results = []
        
        if self.num_parallel > 1:
            with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
                futures = [executor.submit(process_instance_fn, f, experiments) for f in chunk_files]
                for future in as_completed(futures):
                    chunk_results.extend(future.result())
        else:
            for f in chunk_files:
                chunk_results.extend(process_instance_fn(f, experiments))
                
        chunk_file = os.path.join(self.results_dir, f"chunk_{chunk_id:04d}.json")
        with open(chunk_file, 'w') as f:
            json.dump(chunk_results, f)
        
        self.logger.info(f"✅ Chunk {chunk_id} completed.")

    def aggregate_results(self, filter_cols=None):
        self.logger.info("📊 Aggregating results...")
        all_results = []
        chunk_files = sorted(glob.glob(os.path.join(self.results_dir, "chunk_*.json")))
        
        for cf in chunk_files:
            with open(cf, 'r') as f:
                all_results.extend(json.load(f))
                
        if not all_results:
            return None

        df = pd.DataFrame(all_results)
        
        if filter_cols:
            cols_to_drop = [c for c in filter_cols if c in df.columns]
            df = df.drop(columns=cols_to_drop)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name.lower().replace(' ', '_')}_{timestamp}.xlsx"
        output_path = os.path.join(self.output_dir, filename)
        
        df.to_excel(output_path, index=False)
        self.logger.info(f"💾 Final results saved to {output_path}")
        return output_path

    def cleanup(self):
        self.logger.info("🧹 Cleaning up temporary files...")
        chunk_files = glob.glob(os.path.join(self.results_dir, "chunk_*.json"))
        for cf in chunk_files:
            os.remove(cf)
        try:
            os.rmdir(self.results_dir)
        except:
            pass
