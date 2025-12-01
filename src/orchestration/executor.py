"""
Pipeline Executor Module
=========================

This module provides execution engines for running DAG pipelines.

Features:
    - Local sequential execution
    - Local parallel execution with ThreadPool
    - Task status tracking
    - Error handling and propagation
    - Execution monitoring

Example Usage:
    >>> executor = LocalExecutor(max_workers=4)
    >>> run = executor.run(dag, params={"video_path": "video.mp4"})
    >>> if run.is_success:
    ...     print("Pipeline completed successfully")
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from src.utils.logging import get_logger
from src.utils.exceptions import OrchestrationError
from src.orchestration.task import Task, TaskStatus, TaskResult, TaskContext
from src.orchestration.dag import DAG, DAGRun

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Abstract Executor
# =============================================================================

class TaskExecutor(ABC):
    """
    Abstract base class for task executors.

    Executors are responsible for running DAG pipelines
    and managing task execution order.
    """

    @abstractmethod
    def run(
        self,
        dag: DAG,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DAGRun:
        """
        Execute a DAG.

        Args:
            dag: DAG to execute.
            params: User parameters.
            config: Configuration overrides.

        Returns:
            DAGRun with execution results.
        """
        pass

    @abstractmethod
    def run_task(
        self,
        task: Task,
        context: TaskContext,
    ) -> TaskResult:
        """
        Execute a single task.

        Args:
            task: Task to execute.
            context: Execution context.

        Returns:
            TaskResult with execution result.
        """
        pass


# =============================================================================
# Local Executor
# =============================================================================

class LocalExecutor(TaskExecutor):
    """
    Local pipeline executor.

    Executes tasks either sequentially or in parallel using
    a thread pool.

    Attributes:
        max_workers: Maximum parallel workers.
        sequential: Run tasks sequentially.

    Example:
        >>> executor = LocalExecutor(max_workers=4)
        >>> run = executor.run(dag, params={"video_path": "video.mp4"})
    """

    def __init__(
        self,
        max_workers: int = 4,
        sequential: bool = False,
    ) -> None:
        """
        Initialize local executor.

        Args:
            max_workers: Maximum parallel workers.
            sequential: Force sequential execution.
        """
        self.max_workers = max_workers
        self.sequential = sequential

        logger.info(
            f"LocalExecutor initialized: max_workers={max_workers}, "
            f"sequential={sequential}"
        )

    def run(
        self,
        dag: DAG,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DAGRun:
        """
        Execute a DAG.

        Args:
            dag: DAG to execute.
            params: User parameters.
            config: Configuration overrides.

        Returns:
            DAGRun with execution results.
        """
        # Validate DAG
        errors = dag.validate()
        if errors:
            raise OrchestrationError(
                f"DAG validation failed: {errors}",
                dag_id=dag.dag_id,
            )

        # Create run
        run = DAGRun(
            dag_id=dag.dag_id,
            status="running",
            start_time=datetime.now(),
            params=params or {},
        )

        logger.info(f"Starting DAG run: {run.run_id}")

        try:
            if self.sequential:
                self._run_sequential(dag, run, config or {})
            else:
                self._run_parallel(dag, run, config or {})

            # Determine final status
            if run.is_success:
                run.status = "success"
            else:
                run.status = "failed"

        except Exception as e:
            run.status = "failed"
            logger.exception(f"DAG run failed: {e}")

        finally:
            run.end_time = datetime.now()
            duration = (run.end_time - run.start_time).total_seconds()
            logger.info(
                f"DAG run completed: {run.run_id}, "
                f"status={run.status}, duration={duration:.2f}s"
            )

        return run

    def _run_sequential(
        self,
        dag: DAG,
        run: DAGRun,
        config: Dict[str, Any],
    ) -> None:
        """Execute tasks sequentially."""
        execution_order = dag.get_execution_order()

        for task_name in execution_order:
            task = dag.get_task(task_name)
            if task is None:
                continue

            # Check upstream status
            upstream_failed = False
            for upstream in task.depends_on:
                if upstream in run.task_results:
                    if run.task_results[upstream].is_failed:
                        upstream_failed = True
                        break

            if upstream_failed:
                # Mark as upstream failed
                run.task_results[task_name] = TaskResult(
                    task_name=task_name,
                    status=TaskStatus.UPSTREAM_FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )
                continue

            # Create context
            context = self._create_context(dag, run, task_name, config)

            # Execute task
            result = self.run_task(task, context)
            run.task_results[task_name] = result

    def _run_parallel(
        self,
        dag: DAG,
        run: DAGRun,
        config: Dict[str, Any],
    ) -> None:
        """Execute tasks in parallel where possible."""
        # Get parallel groups
        groups = dag.get_parallel_groups()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for group in groups:
                # Filter out tasks whose upstream failed
                runnable_tasks = []
                for task_name in group:
                    task = dag.get_task(task_name)
                    if task is None:
                        continue

                    upstream_failed = False
                    for upstream in task.depends_on:
                        if upstream in run.task_results:
                            if run.task_results[upstream].is_failed:
                                upstream_failed = True
                                break

                    if upstream_failed:
                        run.task_results[task_name] = TaskResult(
                            task_name=task_name,
                            status=TaskStatus.UPSTREAM_FAILED,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                        )
                    else:
                        runnable_tasks.append(task)

                if not runnable_tasks:
                    continue

                # Submit tasks for parallel execution
                futures = {}
                for task in runnable_tasks:
                    context = self._create_context(dag, run, task.name, config)
                    future = executor.submit(self.run_task, task, context)
                    futures[future] = task.name

                # Collect results
                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        result = future.result()
                        run.task_results[task_name] = result
                    except Exception as e:
                        run.task_results[task_name] = TaskResult(
                            task_name=task_name,
                            status=TaskStatus.FAILED,
                            error=str(e),
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                        )

    def _create_context(
        self,
        dag: DAG,
        run: DAGRun,
        task_name: str,
        config: Dict[str, Any],
    ) -> TaskContext:
        """Create task execution context."""
        # Collect upstream outputs
        task = dag.get_task(task_name)
        upstream_outputs = {}

        if task:
            for upstream in task.depends_on:
                if upstream in run.task_results:
                    result = run.task_results[upstream]
                    if result.is_success:
                        upstream_outputs[upstream] = result.output

        return TaskContext(
            dag_run_id=run.run_id,
            task_name=task_name,
            params=run.params,
            upstream_outputs=upstream_outputs,
            config=config,
        )

    def run_task(
        self,
        task: Task,
        context: TaskContext,
    ) -> TaskResult:
        """
        Execute a single task.

        Args:
            task: Task to execute.
            context: Execution context.

        Returns:
            TaskResult with execution result.
        """
        logger.info(f"Executing task: {task.name}")
        return task.execute(context)


# =============================================================================
# Async Executor (for future use)
# =============================================================================

class AsyncExecutor(TaskExecutor):
    """
    Asynchronous pipeline executor using asyncio.

    This executor is designed for I/O-bound pipelines
    where tasks can benefit from async execution.
    """

    def __init__(self, max_concurrent: int = 10):
        """Initialize async executor."""
        self.max_concurrent = max_concurrent
        logger.info(f"AsyncExecutor initialized: max_concurrent={max_concurrent}")

    def run(
        self,
        dag: DAG,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DAGRun:
        """Execute DAG asynchronously."""
        # Placeholder - would use asyncio.run() in production
        # For now, delegate to LocalExecutor
        local = LocalExecutor(max_workers=self.max_concurrent)
        return local.run(dag, params, config)

    def run_task(
        self,
        task: Task,
        context: TaskContext,
    ) -> TaskResult:
        """Execute task."""
        return task.execute(context)


# =============================================================================
# Execution Monitor
# =============================================================================

class ExecutionMonitor:
    """
    Monitor for tracking pipeline execution.

    Provides callbacks for task events and progress tracking.
    """

    def __init__(self):
        """Initialize monitor."""
        self.callbacks = {
            "on_task_start": [],
            "on_task_complete": [],
            "on_task_fail": [],
            "on_dag_start": [],
            "on_dag_complete": [],
        }

    def register_callback(self, event: str, callback) -> None:
        """Register a callback for an event."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def emit(self, event: str, **kwargs) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                logger.warning(f"Callback error for {event}: {e}")


# =============================================================================
# Service Factory
# =============================================================================

def create_executor(config: Optional[Dict[str, Any]] = None) -> TaskExecutor:
    """
    Factory function to create an executor from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured TaskExecutor instance.
    """
    if config is None:
        config = {}

    backend = config.get("backend", "local")

    if backend == "local":
        return LocalExecutor(
            max_workers=config.get("max_workers", 4),
            sequential=config.get("sequential", False),
        )
    elif backend == "async":
        return AsyncExecutor(
            max_concurrent=config.get("max_concurrent", 10),
        )
    else:
        raise OrchestrationError(
            f"Unsupported executor backend: {backend}",
        )
