"""Background task execution for long-running benchmarks."""

import threading
import json
import os
import sys
import logging
from datetime import datetime


class BackgroundTask:
    """Manages background benchmark execution and progress tracking."""
    
    TASK_FILE = "server_output/latest_task.json"
    LOG_DIR = "server_output/logs"
    
    def __init__(self, task_id: str):
        """Initialize a background task with given ID."""
        self.task_id = task_id
        self._ensure_dir()
        self._init_logger()
        self._init_task_file()
    
    @staticmethod
    def _ensure_dir():
        """Ensure server_output and logs directories exist."""
        os.makedirs("server_output", exist_ok=True)
        os.makedirs(BackgroundTask.LOG_DIR, exist_ok=True)
    
    def _init_logger(self) -> None:
        """Initialize logger for this task."""
        log_file = os.path.join(self.LOG_DIR, f"{self.task_id}.log")
        self.logger = logging.getLogger(f"task_{self.task_id}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Also stream logs to console so they appear in the running process output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with the given level."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
    
    def _init_task_file(self):
        """Initialize the task status file with default values."""
        self.log(f"Task {self.task_id} initialized")
        self.update_status({
            "task_id": self.task_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "current_step": "",
            "total_steps": 0,
            "error": None,
            "results_file": None,
            "log_file": os.path.join(self.LOG_DIR, f"{self.task_id}.log")
        })
    
    def update_status(self, status_dict: dict) -> None:
        """Update task status file with new data."""
        self._ensure_dir()
        with open(self.TASK_FILE, "w") as f:
            json.dump(status_dict, f, indent=2)
    
    def get_status(self) -> dict:
        """Get current task status from file."""
        if os.path.exists(self.TASK_FILE):
            with open(self.TASK_FILE, "r") as f:
                return json.load(f)
        return {}
    
    def set_running(self) -> None:
        """Mark task as running and record start time."""
        status = self.get_status()
        status["status"] = "running"
        started_at = datetime.now().isoformat()
        status["started_at"] = started_at
        self.update_status(status)
        self.log(f"Task started at {started_at}")
    
    def set_completed(self, results_file: str | None = None, error: str | None = None) -> None:
        """Mark task as completed with optional results file or error message."""
        status = self.get_status()
        completed_at = datetime.now().isoformat()
        
        if error:
            status["status"] = "failed"
            status["error"] = error
            self.log(f"Task failed: {error}", level="error")
        else:
            status["status"] = "completed"
            status["results_file"] = results_file
            self.log(f"Task completed successfully at {completed_at}")
            if results_file:
                self.log(f"Results saved to: {results_file}")
        
        status["completed_at"] = completed_at
        self.update_status(status)
    
    def update_progress(self, current_step: int, total_steps: int, step_name: str = "") -> None:
        """Update progress information for current task."""
        status = self.get_status()
        progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
        status["current_step"] = step_name
        status["total_steps"] = total_steps
        status["progress"] = progress_pct
        self.update_status(status)
        
        # Log progress at regular intervals to avoid log spam
        if current_step % max(1, total_steps // 10) == 0 or current_step == total_steps:
            self.log(f"Progress: {current_step}/{total_steps} ({progress_pct:.1f}%) - {step_name}")



def run_background_task(func, task_id: str, *args, **kwargs) -> str:
    """
    Run a function in a background thread with task tracking.
    
    Args:
        func: The function to run (should accept task as first argument)
        task_id: Unique task identifier
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        The task ID for later reference
    """
    task = BackgroundTask(task_id)
    
    def wrapper():
        try:
            task.set_running()
            func(task, *args, **kwargs)
        except Exception as e:
            task.log(f"Unexpected error: {str(e)}", level="error")
            task.set_completed(error=str(e))
    
    thread = threading.Thread(target=wrapper, daemon=False)
    thread.start()
    return task_id

