"""
Task Definition Module
=======================

This module provides task definitions for pipeline orchestration.

Features:
    - Task status tracking
    - Retry logic with backoff
    - Timeout handling
    - Task result serialization
    - Dependency specification

Example Usage:
    >>> task = Task(
    ...     name="extract_features",
    ...     callable=extract_features_fn,
    ...     depends_on=["preprocess"],
    ...     retries=3,
    ...     timeout=300,
    ... )
    >>> result = task.execute(context)
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from src.utils.logging import get_logger
from src.utils.exceptions import TaskError

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    UPSTREAM_FAILED = "upstream_failed"
    RETRYING = "retrying"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TaskResult:
    """
    Container for task execution results.

    Attributes:
        task_id: Unique task execution ID.
        task_name: Name of the task.
        status: Execution status.
        output: Task output data.
        error: Error message if failed.
        start_time: Execution start time.
        end_time: Execution end time.
        duration: Execution duration in seconds.
        retries: Number of retries attempted.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    output: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status.value,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "retries": self.retries,
        }

    @property
    def is_success(self) -> bool:
        """Check if task succeeded."""
        return self.status == TaskStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status in (TaskStatus.FAILED, TaskStatus.UPSTREAM_FAILED)


@dataclass
class TaskContext:
    """
    Context passed to task execution.

    Attributes:
        dag_run_id: ID of the current DAG run.
        task_name: Name of the current task.
        params: User-provided parameters.
        upstream_outputs: Outputs from upstream tasks.
        config: Configuration dictionary.
    """
    dag_run_id: str = ""
    task_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    upstream_outputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def get_upstream_output(self, task_name: str) -> Any:
        """Get output from an upstream task."""
        return self.upstream_outputs.get(task_name)


# =============================================================================
# Task Class
# =============================================================================

class Task:
    """
    Pipeline task definition.

    A task represents a single unit of work in the pipeline.
    Tasks can have dependencies on other tasks and support
    retry logic.

    Attributes:
        name: Unique task name.
        callable: Function to execute.
        depends_on: List of upstream task names.
        retries: Number of retries on failure.
        retry_delay: Delay between retries in seconds.
        timeout: Execution timeout in seconds.
        pool: Execution pool for resource limiting.

    Example:
        >>> def process_video(context):
        ...     video_path = context.params["video_path"]
        ...     return {"frames": extract_frames(video_path)}
        >>>
        >>> task = Task("process_video", process_video)
    """

    def __init__(
        self,
        name: str,
        callable: Callable[[TaskContext], Any],
        depends_on: Optional[List[str]] = None,
        retries: int = 0,
        retry_delay: float = 5.0,
        retry_exponential_backoff: bool = True,
        timeout: Optional[float] = None,
        pool: str = "default",
        trigger_rule: str = "all_success",
    ) -> None:
        """
        Initialize task.

        Args:
            name: Unique task name.
            callable: Function to execute with TaskContext argument.
            depends_on: List of upstream task names.
            retries: Number of retries on failure.
            retry_delay: Base delay between retries.
            retry_exponential_backoff: Use exponential backoff for retries.
            timeout: Execution timeout in seconds.
            pool: Execution pool name.
            trigger_rule: When to trigger ("all_success", "one_success", "all_done").
        """
        self.name = name
        self.callable = callable
        self.depends_on = depends_on or []
        self.retries = retries
        self.retry_delay = retry_delay
        self.retry_exponential_backoff = retry_exponential_backoff
        self.timeout = timeout
        self.pool = pool
        self.trigger_rule = trigger_rule

    def execute(
        self,
        context: TaskContext,
    ) -> TaskResult:
        """
        Execute the task.

        Args:
            context: Task execution context.

        Returns:
            TaskResult with execution results.
        """
        result = TaskResult(
            task_name=self.name,
            start_time=datetime.now(),
        )

        attempt = 0
        last_exception = None

        while attempt <= self.retries:
            try:
                if attempt > 0:
                    result.status = TaskStatus.RETRYING
                    delay = self._get_retry_delay(attempt)
                    logger.info(
                        f"Task '{self.name}' retry {attempt}/{self.retries} "
                        f"after {delay:.1f}s"
                    )
                    time.sleep(delay)

                result.status = TaskStatus.RUNNING
                result.retries = attempt

                # Execute the callable
                output = self.callable(context)

                # Success
                result.status = TaskStatus.SUCCESS
                result.output = output
                result.end_time = datetime.now()
                result.duration = (result.end_time - result.start_time).total_seconds()

                logger.info(
                    f"Task '{self.name}' completed in {result.duration:.2f}s"
                )
                return result

            except Exception as e:
                last_exception = e
                attempt += 1

                if attempt > self.retries:
                    # Final failure
                    result.status = TaskStatus.FAILED
                    result.error = str(e)
                    result.traceback = traceback.format_exc()
                    result.end_time = datetime.now()
                    result.duration = (result.end_time - result.start_time).total_seconds()

                    logger.error(
                        f"Task '{self.name}' failed after {attempt} attempts: {e}"
                    )
                    return result

        return result

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay."""
        if self.retry_exponential_backoff:
            return self.retry_delay * (2 ** (attempt - 1))
        return self.retry_delay

    def __repr__(self) -> str:
        """String representation."""
        deps = f", depends_on={self.depends_on}" if self.depends_on else ""
        return f"Task(name='{self.name}'{deps})"


# =============================================================================
# Predefined Tasks
# =============================================================================

def create_preprocess_task(
    video_processor,
    face_detector,
    audio_processor,
) -> Task:
    """
    Create preprocessing task.

    Args:
        video_processor: VideoProcessor instance.
        face_detector: FaceDetector instance.
        audio_processor: AudioProcessor instance.

    Returns:
        Preprocessing task.
    """
    def preprocess(context: TaskContext) -> Dict[str, Any]:
        video_path = context.params["video_path"]

        # Extract frames
        frames, fps = video_processor.extract_frames(video_path)

        # Detect faces
        face_crops = []
        landmarks_list = []
        for frame in frames:
            result = face_detector.detect_and_crop(frame)
            if result.success:
                face_crops.append(result.face_crop)
                landmarks_list.append(
                    result.landmarks.to_dict() if result.landmarks else None
                )
            else:
                face_crops.append(None)
                landmarks_list.append(None)

        # Extract audio features
        mfccs = audio_processor.extract_features(video_path, len(frames), fps)

        return {
            "frames": frames,
            "face_crops": face_crops,
            "landmarks": landmarks_list,
            "mfccs": mfccs,
            "fps": fps,
        }

    return Task(
        name="preprocess",
        callable=preprocess,
        retries=1,
        timeout=300,
    )


def create_feature_extraction_task(
    visual_encoder,
    temporal_model,
    audio_encoder,
    correlation_analyzer,
) -> Task:
    """
    Create feature extraction task.

    Args:
        visual_encoder: VisualEncoder instance.
        temporal_model: TemporalModel instance.
        audio_encoder: AudioEncoder instance.
        correlation_analyzer: CorrelationAnalyzer instance.

    Returns:
        Feature extraction task.
    """
    def extract_features(context: TaskContext) -> Dict[str, Any]:
        preprocess_output = context.get_upstream_output("preprocess")

        face_crops = preprocess_output["face_crops"]
        mfccs = preprocess_output["mfccs"]
        landmarks = preprocess_output["landmarks"]
        fps = preprocess_output["fps"]

        # Filter valid face crops
        valid_crops = [c for c in face_crops if c is not None]

        if not valid_crops:
            raise ValueError("No valid face crops extracted")

        # Visual features
        visual_embeddings = visual_encoder.encode(valid_crops)

        # Temporal features
        temporal_embedding = temporal_model.encode(visual_embeddings)

        # Audio features
        audio_embedding = audio_encoder.encode(mfccs)

        # Correlation analysis
        correlation_metrics = correlation_analyzer.analyze(
            face_crops=[c for c in face_crops if c is not None],
            mfcc_sequence=mfccs,
            landmarks_sequence=landmarks,
        )

        # Concatenate all features
        feature_vector = np.concatenate([
            temporal_embedding.flatten(),
            audio_embedding.flatten(),
            correlation_metrics.to_feature_vector(),
        ])

        return {
            "visual_embeddings": visual_embeddings,
            "temporal_embedding": temporal_embedding,
            "audio_embedding": audio_embedding,
            "correlation_metrics": correlation_metrics.to_dict(),
            "feature_vector": feature_vector,
        }

    return Task(
        name="extract_features",
        callable=extract_features,
        depends_on=["preprocess"],
        retries=1,
        timeout=600,
    )


def create_classification_task(
    classifier,
    augmenter=None,
) -> Task:
    """
    Create classification task.

    Args:
        classifier: DeepfakeClassifier instance.
        augmenter: Optional FeatureAugmenter instance.

    Returns:
        Classification task.
    """
    def classify(context: TaskContext) -> Dict[str, Any]:
        features_output = context.get_upstream_output("extract_features")
        feature_vector = features_output["feature_vector"]

        # Augment if available
        if augmenter is not None:
            augmentation_result = augmenter.augment(feature_vector)
            feature_vector = augmentation_result.augmented_vector
            neighbor_vote = augmentation_result.neighbor_vote
        else:
            neighbor_vote = None

        # Classify
        probability = classifier.predict_proba(
            feature_vector.reshape(1, -1)
        )[0]
        prediction = int(probability > classifier.threshold)

        return {
            "probability": float(probability),
            "prediction": prediction,
            "is_fake": prediction == 1,
            "neighbor_vote": neighbor_vote,
            "threshold": classifier.threshold,
        }

    return Task(
        name="classify",
        callable=classify,
        depends_on=["extract_features"],
        retries=0,
        timeout=60,
    )


# Need numpy for concatenation
import numpy as np
