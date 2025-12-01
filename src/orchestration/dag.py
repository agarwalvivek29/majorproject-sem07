"""
Directed Acyclic Graph Module
==============================

This module provides DAG definitions for pipeline orchestration.

Features:
    - Task dependency management
    - Topological sorting
    - Cycle detection
    - DAG validation
    - Visualization

Example Usage:
    >>> dag = DAG("training_pipeline")
    >>> dag.add_task(preprocess_task)
    >>> dag.add_task(feature_task)
    >>> dag.add_task(classify_task)
    >>> dag.validate()
    >>> execution_order = dag.get_execution_order()
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from src.utils.logging import get_logger
from src.utils.exceptions import OrchestrationError
from src.orchestration.task import Task, TaskStatus, TaskResult

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DAGRun:
    """
    Container for a DAG execution run.

    Attributes:
        run_id: Unique run identifier.
        dag_id: DAG identifier.
        status: Overall run status.
        start_time: Run start time.
        end_time: Run end time.
        task_results: Results for each task.
        params: Run parameters.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    dag_id: str = ""
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        if not self.task_results:
            return False
        return all(
            r.status in (TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for r in self.task_results.values()
        )

    @property
    def is_success(self) -> bool:
        """Check if all tasks succeeded."""
        if not self.task_results:
            return False
        return all(
            r.status == TaskStatus.SUCCESS
            for r in self.task_results.values()
        )

    @property
    def failed_tasks(self) -> List[str]:
        """Get list of failed task names."""
        return [
            name for name, result in self.task_results.items()
            if result.status == TaskStatus.FAILED
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "dag_id": self.dag_id,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "task_results": {
                name: result.to_dict()
                for name, result in self.task_results.items()
            },
        }


# =============================================================================
# DAG Class
# =============================================================================

class DAG:
    """
    Directed Acyclic Graph for pipeline orchestration.

    A DAG represents a collection of tasks with dependencies.
    The DAG ensures tasks are executed in the correct order
    based on their dependencies.

    Attributes:
        dag_id: Unique DAG identifier.
        tasks: Dictionary of tasks by name.
        description: DAG description.
        default_args: Default arguments for tasks.

    Example:
        >>> dag = DAG("detection_pipeline")
        >>> dag.add_task(Task("preprocess", preprocess_fn))
        >>> dag.add_task(Task("features", features_fn, depends_on=["preprocess"]))
        >>> dag.add_task(Task("classify", classify_fn, depends_on=["features"]))
        >>> order = dag.get_execution_order()
        >>> print(order)  # ["preprocess", "features", "classify"]
    """

    def __init__(
        self,
        dag_id: str,
        description: str = "",
        default_args: Optional[Dict[str, Any]] = None,
        schedule: Optional[str] = None,
        max_active_runs: int = 1,
    ) -> None:
        """
        Initialize DAG.

        Args:
            dag_id: Unique DAG identifier.
            description: Human-readable description.
            default_args: Default arguments for all tasks.
            schedule: Cron-style schedule (for future use).
            max_active_runs: Maximum concurrent runs.
        """
        self.dag_id = dag_id
        self.description = description
        self.default_args = default_args or {}
        self.schedule = schedule
        self.max_active_runs = max_active_runs

        self.tasks: Dict[str, Task] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # task -> downstream tasks
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # task -> upstream tasks

        logger.info(f"DAG '{dag_id}' created")

    def add_task(self, task: Task) -> "DAG":
        """
        Add a task to the DAG.

        Args:
            task: Task to add.

        Returns:
            Self for chaining.

        Raises:
            OrchestrationError: If task name already exists.
        """
        if task.name in self.tasks:
            raise OrchestrationError(
                f"Task '{task.name}' already exists in DAG",
                dag_id=self.dag_id,
            )

        self.tasks[task.name] = task

        # Update adjacency lists
        for upstream in task.depends_on:
            self._adjacency[upstream].add(task.name)
            self._reverse_adjacency[task.name].add(upstream)

        logger.debug(f"Added task '{task.name}' to DAG '{self.dag_id}'")
        return self

    def remove_task(self, task_name: str) -> None:
        """
        Remove a task from the DAG.

        Args:
            task_name: Name of task to remove.
        """
        if task_name not in self.tasks:
            return

        # Remove from adjacency lists
        for downstream in list(self._adjacency[task_name]):
            self._reverse_adjacency[downstream].discard(task_name)
        del self._adjacency[task_name]

        for upstream in list(self._reverse_adjacency[task_name]):
            self._adjacency[upstream].discard(task_name)
        del self._reverse_adjacency[task_name]

        # Remove task
        del self.tasks[task_name]

    def get_task(self, task_name: str) -> Optional[Task]:
        """Get task by name."""
        return self.tasks.get(task_name)

    def get_upstream_tasks(self, task_name: str) -> Set[str]:
        """Get names of upstream tasks."""
        return self._reverse_adjacency.get(task_name, set())

    def get_downstream_tasks(self, task_name: str) -> Set[str]:
        """Get names of downstream tasks."""
        return self._adjacency.get(task_name, set())

    def get_root_tasks(self) -> List[str]:
        """Get tasks with no dependencies."""
        return [
            name for name in self.tasks
            if not self._reverse_adjacency.get(name)
        ]

    def get_leaf_tasks(self) -> List[str]:
        """Get tasks with no downstream tasks."""
        return [
            name for name in self.tasks
            if not self._adjacency.get(name)
        ]

    def validate(self) -> List[str]:
        """
        Validate the DAG.

        Checks for:
            - Cycles in the graph
            - Missing dependencies
            - Duplicate task names

        Returns:
            List of validation error messages.

        Raises:
            OrchestrationError: If critical validation error.
        """
        errors = []

        # Check for missing dependencies
        for task_name, task in self.tasks.items():
            for dep in task.depends_on:
                if dep not in self.tasks:
                    errors.append(
                        f"Task '{task_name}' depends on missing task '{dep}'"
                    )

        # Check for cycles
        if self._has_cycle():
            errors.append("DAG contains a cycle")

        if errors:
            for error in errors:
                logger.error(f"DAG validation error: {error}")

        return errors

    def _has_cycle(self) -> bool:
        """Check if the DAG contains a cycle."""
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for task_name in self.tasks:
            if task_name not in visited:
                if dfs(task_name):
                    return True

        return False

    def get_execution_order(self) -> List[str]:
        """
        Get topological sort of tasks.

        Returns:
            List of task names in execution order.

        Raises:
            OrchestrationError: If DAG has cycles.
        """
        if self._has_cycle():
            raise OrchestrationError(
                "Cannot get execution order: DAG has cycles",
                dag_id=self.dag_id,
            )

        # Kahn's algorithm for topological sort
        in_degree = {name: len(self._reverse_adjacency.get(name, set()))
                     for name in self.tasks}
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in self._adjacency.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_parallel_groups(self) -> List[List[str]]:
        """
        Get groups of tasks that can run in parallel.

        Returns:
            List of task name lists, where each inner list
            contains tasks that can run concurrently.
        """
        if self._has_cycle():
            raise OrchestrationError(
                "Cannot get parallel groups: DAG has cycles",
                dag_id=self.dag_id,
            )

        # Calculate levels using BFS
        levels: Dict[str, int] = {}
        in_degree = {name: len(self._reverse_adjacency.get(name, set()))
                     for name in self.tasks}

        # Initialize root tasks at level 0
        queue = deque()
        for name, degree in in_degree.items():
            if degree == 0:
                levels[name] = 0
                queue.append(name)

        # BFS to assign levels
        while queue:
            node = queue.popleft()
            node_level = levels[node]

            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in levels:
                    levels[neighbor] = node_level + 1
                else:
                    levels[neighbor] = max(levels[neighbor], node_level + 1)

                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Group by level
        max_level = max(levels.values()) if levels else 0
        groups = [[] for _ in range(max_level + 1)]
        for name, level in levels.items():
            groups[level].append(name)

        return groups

    def visualize(self) -> str:
        """
        Generate ASCII visualization of the DAG.

        Returns:
            ASCII representation of the DAG.
        """
        lines = [f"DAG: {self.dag_id}", "=" * (len(self.dag_id) + 5), ""]

        groups = self.get_parallel_groups()
        for i, group in enumerate(groups):
            lines.append(f"Level {i}:")
            for task_name in group:
                task = self.tasks[task_name]
                deps = ", ".join(task.depends_on) if task.depends_on else "none"
                lines.append(f"  [{task_name}] <- ({deps})")
            lines.append("")

        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of tasks."""
        return len(self.tasks)

    def __contains__(self, task_name: str) -> bool:
        """Check if task exists."""
        return task_name in self.tasks

    def __repr__(self) -> str:
        """String representation."""
        return f"DAG(dag_id='{self.dag_id}', tasks={len(self.tasks)})"


# =============================================================================
# DAG Builder
# =============================================================================

class DAGBuilder:
    """
    Builder pattern for creating DAGs.

    Example:
        >>> dag = (DAGBuilder("pipeline")
        ...     .add("preprocess", preprocess_fn)
        ...     .add("features", features_fn, depends_on=["preprocess"])
        ...     .add("classify", classify_fn, depends_on=["features"])
        ...     .build())
    """

    def __init__(self, dag_id: str, **kwargs):
        """Initialize builder."""
        self.dag = DAG(dag_id, **kwargs)

    def add(
        self,
        name: str,
        callable,
        depends_on: Optional[List[str]] = None,
        **kwargs,
    ) -> "DAGBuilder":
        """
        Add a task.

        Args:
            name: Task name.
            callable: Task function.
            depends_on: Upstream dependencies.
            **kwargs: Additional task arguments.

        Returns:
            Self for chaining.
        """
        task = Task(name, callable, depends_on=depends_on, **kwargs)
        self.dag.add_task(task)
        return self

    def build(self) -> DAG:
        """
        Build and validate the DAG.

        Returns:
            Validated DAG.
        """
        errors = self.dag.validate()
        if errors:
            raise OrchestrationError(
                f"DAG validation failed: {errors}",
                dag_id=self.dag.dag_id,
            )
        return self.dag
