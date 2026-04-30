"""Data models for CVRP Benchmark - Using dataclasses instead of dictionaries."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from components import constants as C


@dataclass
class ExecutionResult:
    """Result from executing a single algorithm run."""
    cpu_time: float
    objective_value: Optional[float] = None
    accepted_neighbors: int = 0
    routes: Optional[List] = None
    history: List[tuple] = field(default_factory=list)  # List of (time, iters, cost) tuples
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class InstanceMetadata:
    """Metadata extracted from an instance."""
    name: str
    customers: int
    vehicles: int
    capacity: int
    depot_layout: str
    customer_layout: str
    demand_profile: str
    route_class: str
    climate: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment (algorithm + first solution combo)."""
    name: str
    fs_label: str
    mh_label: str
    func: Callable
    kwargs: Dict[str, Any]
    reps: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding callable func)."""
        d = asdict(self)
        d.pop('func', None)  # Remove the callable function
        return d


@dataclass
class RunStatistics:
    """Statistics from a single run of an experiment."""
    costs: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    neighbors_list: List[int] = field(default_factory=list)
    best_routes: Optional[List] = None
    checkpoints: Dict[int, List[float]] = field(default_factory=dict)  # {time_checkpoint: [costs]}
    all_histories: List[List[tuple]] = field(default_factory=list)  # Full history from each run
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete benchmark result row for a single algorithm-instance combination."""
    # Instance info
    instance: str
    depot_layout: str
    customer_layout: str
    demand_type: str
    route_class: str
    climate: str
    customers: int
    vehicles: int
    capacity: int
    
    # Algorithm info
    first_solution: str
    metaheuristic: str
    
    # Metrics
    repetitions: int
    avg_cost: Optional[float] = None
    best_cost: Optional[float] = None
    bks_cost: Optional[float] = None
    best_gap_percent: Optional[float] = None
    avg_gap_percent: Optional[float] = None
    avg_cpu_time: Optional[float] = None
    
    # Time checkpoints (dynamic, can be added as needed)
    checkpoints: Dict[str, Optional[float]] = field(default_factory=dict)  # {"Avg Cost @ 5s": value, ...}
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame compatibility."""
        d = {
            C.COL_INSTANCE: self.instance,
            C.COL_DEPOT_LAYOUT: self.depot_layout,
            C.COL_CUSTOMER_LAYOUT: self.customer_layout,
            C.COL_DEMAND_TYPE: self.demand_type,
            C.COL_ROUTE_CLASS: self.route_class,
            C.COL_CLIMATE: self.climate,
            C.COL_CUSTOMERS: self.customers,
            C.COL_VEHICLES: self.vehicles,
            C.COL_CAPACITY: self.capacity,
            C.COL_FIRST_SOLUTION: self.first_solution,
            C.COL_METAHEURISTIC: self.metaheuristic,
            C.COL_REPETITIONS: self.repetitions,
            C.COL_AVG_COST: self.avg_cost,
            C.COL_BEST_COST: self.best_cost,
            C.COL_BKS_COST: self.bks_cost,
            C.COL_BEST_GAP: self.best_gap_percent,
            C.COL_AVG_GAP: self.avg_gap_percent,
            C.COL_AVG_CPU_TIME: self.avg_cpu_time,
        }
        # Add checkpoint columns
        d.update(self.checkpoints)
        return d


@dataclass
class BenchmarkSettings:
    """Settings for a benchmark run."""
    sel_fs: List[str]  # Selected first solution strategies
    sel_mh: List[str]  # Selected metaheuristics
    fs_enum: Dict[str, Any]  # First solution enum mapping
    mh_enum: Dict[str, Any]  # Metaheuristic enum mapping
    solver_func: Callable
    time_limit: int
    sol_limit: int
    lns_limit: int
    no_improv: int
    no_improv_iter: int
    target_gap: Optional[float] = None
    reps: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding callables)."""
        d = asdict(self)
        d.pop('solver_func', None)
        return d


@dataclass
class TaskStatus:
    """Status of a background benchmark task."""
    task_id: str
    task_name: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    current_step: int = 0
    total_steps: int = 0
    error: Optional[str] = None
    results_file: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# Conversion helpers for backward compatibility
def execution_result_to_dict(result: ExecutionResult) -> dict:
    """Convert ExecutionResult to dictionary format."""
    if isinstance(result, ExecutionResult):
        return result.to_dict()
    return result


def instance_metadata_to_dict(metadata: InstanceMetadata) -> dict:
    """Convert InstanceMetadata to dictionary format."""
    if isinstance(metadata, InstanceMetadata):
        return metadata.to_dict()
    # Legacy format
    return {
        "name": metadata.get("name"),
        "meta": {
            "Depot Layout": metadata.get("depot_layout"),
            "Customer Layout": metadata.get("customer_layout"),
            "Demand Profile ID": metadata.get("demand_profile"),
            "Route Length Class": metadata.get("route_class"),
            "Climate": metadata.get("climate"),
        },
        "customers": metadata.get("customers"),
        "vehicles": metadata.get("vehicles"),
        "capacity": metadata.get("capacity"),
    }


def benchmark_result_to_legacy_dict(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to legacy dictionary format for DataFrame."""
    return result.to_dict()
