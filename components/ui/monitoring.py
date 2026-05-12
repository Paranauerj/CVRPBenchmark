"""UI component for monitoring running and completed benchmarks."""

import streamlit as st
from datetime import datetime
from components.execution.task_manager import TaskManager, TaskInfo


def format_timestamp(iso_string: str) -> str:
    """Format ISO timestamp to readable string."""
    if not iso_string:
        return "Not started"
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(iso_string)


def format_duration(start_iso: str, end_iso: str = None) -> str:
    """Format duration between two timestamps."""
    if not start_iso:
        return "—"
    
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso) if end_iso else datetime.now()
        duration = end - start
        
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception:
        return "—"


def get_status_color(status: str) -> str:
    """Get status color for badge."""
    colors = {
        "pending": "🟡",
        "running": "🔵",
        "completed": "🟢",
        "failed": "🔴",
        "stopped": "⚪"
    }
    return colors.get(status, "⚪")


def show_delete_confirmation(task_id: str, task_name: str, task_manager: TaskManager):
    """Show delete confirmation dialog."""
    import streamlit as st
    st.warning(f"⚠️ Are you sure you want to delete task '{task_name}'?")
    st.write("This action cannot be undone.")
    
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Yes, Delete", width='stretch', key=f"confirm_delete_{task_id}"):
            if task_manager.delete_task(task_id):
                st.success(f"Task '{task_name}' deleted successfully")
                st.rerun()
            else:
                st.error(f"Failed to delete task '{task_name}'")
    
    with col_no:
        if st.button("❌ Cancel", width='stretch', key=f"cancel_delete_{task_id}"):
            st.rerun()

# Apply Streamlit dialog decorator only if running in Streamlit
try:
    import streamlit as st
    if st.runtime.exists():
        show_delete_confirmation = st.dialog("Delete Task", width="small")(show_delete_confirmation)
except (ImportError, AttributeError):
    pass


def render_task_card(task: TaskInfo, task_manager: TaskManager):
    """Render a card for a single task."""
    import streamlit as st
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"{get_status_color(task.status)} {task.name}")
        st.caption(f"ID: `{task.task_id}` • Created: {format_timestamp(task.created_at)}")
    
    with col2:
        st.metric("Status", task.status.upper(), delta=None)
    
    with col3:
        duration = format_duration(task.started_at, task.completed_at)
        st.metric("Duration", duration)
    
    # Progress bar
    if task.status == "running":
        progress_val = min(max(task.progress / 100.0, 0.0), 1.0)
        st.progress(progress_val)
        if task.current_step:
            st.caption(f"📍 {task.current_step} ({task.progress:.1f}%)")
    elif task.status in ("completed", "failed"):
        st.progress(1.0)
    
    # Task details
    details_col1, details_col2, details_col3 = st.columns(3)
    
    with details_col1:
        if task.started_at:
            st.caption(f"**Started:** {format_timestamp(task.started_at)}")
    
    with details_col2:
        if task.completed_at:
            st.caption(f"**Completed:** {format_timestamp(task.completed_at)}")
    
    with details_col3:
        if task.total_steps > 0:
            st.caption(f"**Progress:** {task.total_steps} steps")
    
    # Error message if failed
    if task.status == "failed" and task.error:
        st.error(f"Error: {task.error}")
    
    # Action buttons
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if task.status == "running":
            if st.button("⏹️ Stop", key=f"stop_{task.task_id}", width='stretch'):
                if task_manager.stop_task(task.task_id):
                    st.success(f"Stop signal sent to task {task.task_id}")
                    st.rerun()
                else:
                    st.error("Failed to stop task")
    
    with action_col2:
        if task.log_file:
            if st.button("📋 Logs", key=f"logs_{task.task_id}", width='stretch'):
                st.session_state['view_log_file'] = task.log_file
                st.rerun()
    
    with action_col3:
        if task.status in ("completed", "stopped") and task.results_file:
            try:
                with open(task.results_file, 'rb') as f:
                    file_data = f.read()
                st.download_button(
                    label="📥 Download",
                    data=file_data,
                    file_name=task.results_file.split('/')[-1],
                    mime="application/octet-stream",
                    key=f"download_{task.task_id}",
                    width='stretch'
                )
            except Exception as e:
                st.error(f"Error loading file: {e}", icon="❌")
    
    with action_col4:
        if task.status in ("completed", "failed", "stopped"):
            if st.button("🗑️ Delete", key=f"delete_{task.task_id}", width='stretch'):
                show_delete_confirmation(task.task_id, task.name, task_manager)
    
    st.divider()


def render_monitor_page():
    """Render the benchmark monitoring page."""
    st.header("📊 Monitor Benchmarks")
    
    try:
        task_manager = TaskManager()
    except Exception as e:
        st.error(f"❌ Failed to initialize task manager: {e}")
        return
    
    # Control section
    col_refresh, col_interval = st.columns([1, 2])
    
    with col_refresh:
        if st.button("🔄 Refresh", width='stretch'):
            st.rerun()
    
    with col_interval:
        refresh_interval = st.selectbox(
            "Auto-refresh interval",
            options=[0, 5, 10, 30, 60],
            format_func=lambda x: "Disabled" if x == 0 else f"Every {x}s",
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # Get recent tasks
    try:
        recent_tasks = task_manager.get_recent_tasks(limit=20)
        running_tasks = task_manager.get_running_tasks()
    except Exception as e:
        st.error(f"❌ Failed to load tasks: {e}")
        st.info("Start a benchmark from the 'Run Benchmarks' tab to see monitoring data.")
        return
    
    # Summary metrics
    if running_tasks or recent_tasks:
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            st.metric("Running", len(running_tasks))
        with met_col2:
            completed = len([t for t in recent_tasks if t.status == "completed"])
            st.metric("Completed", completed)
        with met_col3:
            failed = len([t for t in recent_tasks if t.status == "failed"])
            st.metric("Failed", failed)
        with met_col4:
            stopped = len([t for t in recent_tasks if t.status == "stopped"])
            st.metric("Stopped", stopped)
        
        st.divider()
    
    # Task list
    if not recent_tasks:
        st.info("📝 No benchmarks run yet. Start a benchmark from the **Run Benchmarks** tab!")
    else:
        st.subheader(f"Tasks ({len(recent_tasks)})")
        
        for task in recent_tasks:
            try:
                render_task_card(task, task_manager)
            except Exception as e:
                st.error(f"Error rendering task {task.task_id}: {str(e)}")
    
    # View logs modal
    if 'view_log_file' in st.session_state and st.session_state.view_log_file:
        st.divider()
        try:
            with open(st.session_state['view_log_file'], 'r') as f:
                logs = f.read()
            
            st.subheader("📋 Task Logs")
            st.code(logs, language="log")
            
            # Download logs button
            st.download_button(
                label="📥 Download Logs",
                data=logs,
                file_name=f"benchmark_{st.session_state['view_log_file'].split('/')[-1]}",
                mime="text/plain",
                width='stretch'
            )
            
            if st.button("Close", key="close_logs", width='stretch'):
                del st.session_state['view_log_file']
                st.rerun()
        except FileNotFoundError:
            st.error("❌ Log file not found")
        except Exception as e:
            st.error(f"❌ Error reading logs: {str(e)}")
    
    # View results file handler (if needed in future)
    
    # Auto-refresh with placeholder
    if running_tasks and refresh_interval > 0:
        st.info(f"ℹ️ {len(running_tasks)} benchmark(s) running. Auto-refreshing in {refresh_interval}s...")
        # Use a placeholder that reruns, instead of sleep
        import time
        time.sleep(refresh_interval)
        st.rerun()
