"""Task manager for tracking and managing benchmark execution."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import signal


@dataclass
class TaskInfo:
    """Information about a running or completed task."""
    task_id: str
    name: str
    status: str  # "pending", "running", "completed", "failed", "stopped"
    progress: float  # 0-100
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_step: str = ""
    total_steps: int = 0
    error: Optional[str] = None
    results_file: Optional[str] = None
    log_file: Optional[str] = None
    process_id: Optional[int] = None


class TaskManager:
    """Manages multiple benchmark tasks."""
    
    TASKS_FILE = "server_output/tasks.json"
    _file_lock = threading.Lock()  # Class-level lock for thread-safe file I/O
    
    def __init__(self):
        """Initialize task manager."""
        self._ensure_dir()
        self._tasks: Dict[str, TaskInfo] = {}
        self._running_threads: Dict[str, threading.Thread] = {}
        self._load_tasks()
    
    @staticmethod
    def _ensure_dir():
        """Ensure server_output directory exists."""
        os.makedirs("server_output", exist_ok=True)
        os.makedirs("server_output/logs", exist_ok=True)
    
    def _load_tasks(self):
        """Load tasks from persistent storage (thread-safe)."""
        if os.path.exists(self.TASKS_FILE):
            try:
                with TaskManager._file_lock:
                    with open(self.TASKS_FILE, "r") as f:
                        data = json.load(f)
                        self._tasks = {
                            task_id: TaskInfo(**task_data)
                            for task_id, task_data in data.items()
                        }
            except Exception as e:
                print(f"Failed to load tasks: {e}")
                self._tasks = {}
    
    def _save_tasks(self):
        """Save tasks to persistent storage (thread-safe)."""
        self._ensure_dir()
        try:
            with TaskManager._file_lock:
                with open(self.TASKS_FILE, "w") as f:
                    data = {
                        task_id: asdict(task_info)
                        for task_id, task_info in self._tasks.items()
                    }
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save tasks: {e}")
    
    def create_task(self, task_id: str, name: str) -> TaskInfo:
        """Create a new task."""
        task = TaskInfo(
            task_id=task_id,
            name=name,
            status="pending",
            progress=0.0,
            created_at=datetime.now().isoformat()
        )
        self._tasks[task_id] = task
        self._save_tasks()
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TaskInfo]:
        """Get all tasks."""
        return list(self._tasks.values())
    
    def get_running_tasks(self) -> List[TaskInfo]:
        """Get all currently running tasks."""
        return [t for t in self._tasks.values() if t.status == "running"]
    
    def get_recent_tasks(self, limit: int = 20) -> List[TaskInfo]:
        """Get most recent tasks (running and recently completed)."""
        # Sort by most recent first
        sorted_tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        # Return running tasks first, then recent completed/failed
        running = [t for t in sorted_tasks if t.status == "running"]
        others = [t for t in sorted_tasks if t.status != "running"][:limit - len(running)]
        return running + others
    
    def update_task_status(self, task_id: str, status: str, **kwargs) -> bool:
        """Update task status (thread-safe)."""
        with TaskManager._file_lock:
            # Reload latest state from disk
            try:
                if os.path.exists(self.TASKS_FILE):
                    with open(self.TASKS_FILE, "r") as f:
                        data = json.load(f)
                        tasks = {
                            tid: TaskInfo(**tdata)
                            for tid, tdata in data.items()
                        }
                else:
                    tasks = {}
            except Exception as e:
                print(f"Failed to load tasks for status update: {e}")
                return False
            
            if task_id not in tasks:
                return False
            
            task = tasks[task_id]
            task.status = status
            
            if status == "running" and not task.started_at:
                task.started_at = datetime.now().isoformat()
            elif status in ("completed", "failed", "stopped"):
                task.completed_at = datetime.now().isoformat()
            
            # Update any additional fields
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            # Save updated tasks back to disk
            try:
                self._ensure_dir()
                with open(self.TASKS_FILE, "w") as f:
                    data = {
                        tid: asdict(t)
                        for tid, t in tasks.items()
                    }
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Failed to save tasks: {e}")
                return False
            
            # Update local copy
            self._tasks = tasks
            return True
    
    def update_task_progress(
        self,
        task_id: str,
        progress: float,
        current_step: str = "",
        total_steps: int = 0
    ) -> bool:
        """Update task progress (thread-safe)."""
        with TaskManager._file_lock:
            # Reload latest state from disk
            try:
                if os.path.exists(self.TASKS_FILE):
                    with open(self.TASKS_FILE, "r") as f:
                        data = json.load(f)
                        tasks = {
                            tid: TaskInfo(**tdata)
                            for tid, tdata in data.items()
                        }
                else:
                    tasks = {}
            except Exception as e:
                print(f"Failed to load tasks for progress update: {e}")
                return False
            
            if task_id not in tasks:
                return False
            
            task = tasks[task_id]
            task.progress = min(100.0, max(0.0, progress))
            task.current_step = current_step
            task.total_steps = total_steps
            
            # Save updated tasks back to disk
            try:
                self._ensure_dir()
                with open(self.TASKS_FILE, "w") as f:
                    data = {
                        tid: asdict(t)
                        for tid, t in tasks.items()
                    }
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Failed to save tasks: {e}")
                return False
            
            # Update local copy
            self._tasks = tasks
            return True
    
    def set_task_running(self, task_id: str) -> bool:
        """Mark task as running."""
        return self.update_task_status(task_id, "running")
    
    def set_task_completed(self, task_id: str, results_file: Optional[str] = None) -> bool:
        """Mark task as completed."""
        return self.update_task_status(
            task_id,
            "completed",
            results_file=results_file,
            progress=100.0
        )
    
    def set_task_failed(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        return self.update_task_status(task_id, "failed", error=error)
    
    def set_task_stopped(self, task_id: str) -> bool:
        """Mark task as stopped by user."""
        return self.update_task_status(task_id, "stopped")
    
    def register_thread(self, task_id: str, thread: threading.Thread) -> None:
        """Register a thread running a task for later stopping."""
        self._running_threads[task_id] = thread
    
    def stop_task(self, task_id: str) -> bool:
        """Stop a running task."""
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        if task.status != "running":
            return False
        
        # Try to stop the thread if it's registered
        if task_id in self._running_threads:
            thread = self._running_threads[task_id]
            if thread.is_alive():
                # For Python threads, we can't forcefully stop them
                # Instead, we mark as stopped and let the implementation check
                # the task manager status periodically
                self.set_task_stopped(task_id)
                del self._running_threads[task_id]
                return True
        
        self.set_task_stopped(task_id)
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its associated files (thread-safe)."""
        with TaskManager._file_lock:
            # Reload latest state from disk
            try:
                if os.path.exists(self.TASKS_FILE):
                    with open(self.TASKS_FILE, "r") as f:
                        data = json.load(f)
                        tasks = {
                            tid: TaskInfo(**tdata)
                            for tid, tdata in data.items()
                        }
                else:
                    tasks = {}
            except Exception as e:
                print(f"Failed to load tasks for deletion: {e}")
                return False
            
            if task_id not in tasks:
                return False
            
            task = tasks[task_id]
            
            # Delete log file if it exists
            if task.log_file and os.path.exists(task.log_file):
                try:
                    os.remove(task.log_file)
                except Exception as e:
                    print(f"Warning: Failed to delete log file {task.log_file}: {e}")
            
            # Delete results file if it exists
            if task.results_file and os.path.exists(task.results_file):
                try:
                    os.remove(task.results_file)
                except Exception as e:
                    print(f"Warning: Failed to delete results file {task.results_file}: {e}")
            
            # Remove from tasks dict
            del tasks[task_id]
            
            # Save updated tasks back to disk
            try:
                self._ensure_dir()
                with open(self.TASKS_FILE, "w") as f:
                    data = {
                        tid: asdict(t)
                        for tid, t in tasks.items()
                    }
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Failed to save tasks: {e}")
                return False
            
            # Update local copy
            self._tasks = tasks
            
            # Remove from threads dict if registered
            if task_id in self._running_threads:
                del self._running_threads[task_id]
            
            return True
    
    def delete_all_tasks(self) -> int:
        """Delete all non-running tasks and their associated files (thread-safe)."""
        tasks_to_delete = [tid for tid, t in self._tasks.items() if t.status != "running"]
        count = 0
        for tid in tasks_to_delete:
            if self.delete_task(tid):
                count += 1
        return count

    def clean_old_tasks(self, days: int = 7) -> int:
        """Remove tasks older than specified days."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        tasks_to_remove = []
        
        for task_id, task in self._tasks.items():
            created = datetime.fromisoformat(task.created_at)
            if created < cutoff and task.status != "running":
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self._tasks[task_id]
        
        if tasks_to_remove:
            self._save_tasks()
        
        return len(tasks_to_remove)
