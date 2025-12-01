"""
Pipeline Orchestration Module
==============================

This module provides DAG-based pipeline orchestration similar to
Apache Airflow for the deepfake detection system.

Components:
    - Task: Base class for pipeline tasks
    - DAG: Directed Acyclic Graph for task dependencies
    - Executor: Pipeline execution engine
    - TaskResult: Container for task execution results

Example Usage:
    >>> from src.orchestration import DAG, Task, LocalExecutor
    >>>
    >>> dag = DAG("training_pipeline")
    >>> dag.add_task(Task("preprocess", preprocess_fn))
    >>> dag.add_task(Task("extract_features", extract_fn, depends_on=["preprocess"]))
    >>>
    >>> executor = LocalExecutor(max_workers=4)
    >>> results = executor.run(dag, {"video_path": "video.mp4"})
"""

from src.orchestration.task import Task, TaskStatus, TaskResult
from src.orchestration.dag import DAG
from src.orchestration.executor import LocalExecutor, TaskExecutor

__all__ = [
    "Task",
    "TaskStatus",
    "TaskResult",
    "DAG",
    "LocalExecutor",
    "TaskExecutor",
]
