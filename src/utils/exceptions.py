"""
Custom Exception Hierarchy
===========================

This module defines a comprehensive exception hierarchy for the
Deepfake Detection System. Using specific exception types enables:
    - Precise error handling at different levels
    - Informative error messages for debugging
    - Proper error propagation in microservices
    - Consistent API error responses

Exception Hierarchy:
    DeepfakeDetectionError (base)
    ├── ConfigurationError
    ├── PreprocessingError
    │   ├── VideoProcessingError
    │   ├── FaceDetectionError
    │   └── AudioProcessingError
    ├── FeatureExtractionError
    │   ├── VisualEncodingError
    │   ├── TemporalEncodingError
    │   └── AudioEncodingError
    ├── RetrievalError
    │   ├── IndexError
    │   └── SearchError
    ├── ClassificationError
    │   ├── ModelLoadError
    │   └── PredictionError
    ├── PipelineError
    │   ├── TaskError
    │   └── OrchestrationError
    └── ServiceError
        ├── ConnectionError
        └── TimeoutError

Example Usage:
    >>> from src.utils.exceptions import VideoProcessingError
    >>> try:
    ...     process_video(path)
    ... except VideoProcessingError as e:
    ...     logger.error(f"Failed to process video: {e}")
    ...     return None
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# =============================================================================
# Base Exception
# =============================================================================

class DeepfakeDetectionError(Exception):
    """
    Base exception for all Deepfake Detection System errors.

    All custom exceptions in this system inherit from this class,
    enabling catch-all error handling when needed.

    Attributes:
        message: Human-readable error message.
        code: Error code for programmatic handling.
        details: Additional error details dictionary.
        cause: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            code: Error code (e.g., "VIDEO_001").
            details: Additional context about the error.
            cause: The underlying exception.
        """
        super().__init__(message)
        self.message = message
        self.code = code or self._default_code()
        self.details = details or {}
        self.cause = cause

        # Chain exceptions
        if cause:
            self.__cause__ = cause

    def _default_code(self) -> str:
        """Generate default error code from class name."""
        # Convert CamelCase to UPPER_SNAKE_CASE
        name = self.__class__.__name__
        code = ""
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                code += "_"
            code += char.upper()
        return code

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.

        Returns:
            Dictionary representation of the error.
        """
        result = {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.cause:
            result["cause"] = str(self.cause)
        return result

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"[{self.code}] {self.message}"]
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r})"
        )


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(DeepfakeDetectionError):
    """
    Exception raised for configuration-related errors.

    This includes missing configuration files, invalid values,
    and schema validation failures.
    """

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize configuration error.

        Args:
            message: Error message.
            config_path: Path to the configuration file.
            key: The configuration key that caused the error.
            **kwargs: Additional arguments for base class.
        """
        details = kwargs.pop("details", {})
        if config_path:
            details["config_path"] = config_path
        if key:
            details["key"] = key
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Preprocessing Errors
# =============================================================================

class PreprocessingError(DeepfakeDetectionError):
    """
    Base exception for preprocessing-related errors.

    This covers errors in video processing, face detection,
    and audio extraction pipelines.
    """

    pass


class VideoProcessingError(PreprocessingError):
    """
    Exception raised when video processing fails.

    This includes errors in frame extraction, format conversion,
    and video decoding.
    """

    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        frame_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize video processing error.

        Args:
            message: Error message.
            video_path: Path to the video file.
            frame_index: Frame where error occurred.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if video_path:
            details["video_path"] = video_path
        if frame_index is not None:
            details["frame_index"] = frame_index
        super().__init__(message, details=details, **kwargs)


class FaceDetectionError(PreprocessingError):
    """
    Exception raised when face detection fails.

    This includes no face detected, multiple faces when one expected,
    and landmark extraction failures.
    """

    def __init__(
        self,
        message: str,
        frame_index: Optional[int] = None,
        faces_found: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize face detection error.

        Args:
            message: Error message.
            frame_index: Frame where error occurred.
            faces_found: Number of faces detected.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if frame_index is not None:
            details["frame_index"] = frame_index
        if faces_found is not None:
            details["faces_found"] = faces_found
        super().__init__(message, details=details, **kwargs)


class AudioProcessingError(PreprocessingError):
    """
    Exception raised when audio processing fails.

    This includes audio extraction failures, format issues,
    and MFCC computation errors.
    """

    def __init__(
        self,
        message: str,
        audio_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize audio processing error.

        Args:
            message: Error message.
            audio_path: Path to the audio file.
            sample_rate: Expected or actual sample rate.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if audio_path:
            details["audio_path"] = audio_path
        if sample_rate:
            details["sample_rate"] = sample_rate
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Feature Extraction Errors
# =============================================================================

class FeatureExtractionError(DeepfakeDetectionError):
    """
    Base exception for feature extraction errors.

    This covers errors in visual, temporal, and audio encoders.
    """

    pass


class VisualEncodingError(FeatureExtractionError):
    """
    Exception raised when visual feature extraction fails.

    This includes CNN forward pass errors and embedding computation failures.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize visual encoding error.

        Args:
            message: Error message.
            model_name: Name of the visual encoder model.
            input_shape: Shape of the input that caused the error.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        if input_shape:
            details["input_shape"] = input_shape
        super().__init__(message, details=details, **kwargs)


class TemporalEncodingError(FeatureExtractionError):
    """
    Exception raised when temporal modeling fails.

    This includes LSTM/Transformer errors and sequence processing failures.
    """

    def __init__(
        self,
        message: str,
        sequence_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize temporal encoding error.

        Args:
            message: Error message.
            sequence_length: Length of the input sequence.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if sequence_length is not None:
            details["sequence_length"] = sequence_length
        super().__init__(message, details=details, **kwargs)


class AudioEncodingError(FeatureExtractionError):
    """
    Exception raised when audio feature extraction fails.

    This includes MFCC computation and wav2vec embedding errors.
    """

    def __init__(
        self,
        message: str,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize audio encoding error.

        Args:
            message: Error message.
            method: Feature extraction method (mfcc, wav2vec).
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if method:
            details["method"] = method
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Retrieval Errors
# =============================================================================

class RetrievalError(DeepfakeDetectionError):
    """
    Base exception for retrieval-augmented detection errors.

    This covers FAISS index and similarity search errors.
    """

    pass


class IndexBuildError(RetrievalError):
    """
    Exception raised when building the vector index fails.
    """

    def __init__(
        self,
        message: str,
        index_type: Optional[str] = None,
        dimension: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize index build error.

        Args:
            message: Error message.
            index_type: Type of index being built.
            dimension: Vector dimension.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if index_type:
            details["index_type"] = index_type
        if dimension:
            details["dimension"] = dimension
        super().__init__(message, details=details, **kwargs)


class SearchError(RetrievalError):
    """
    Exception raised when similarity search fails.
    """

    def __init__(
        self,
        message: str,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize search error.

        Args:
            message: Error message.
            k: Number of neighbors requested.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if k is not None:
            details["k"] = k
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Classification Errors
# =============================================================================

class ClassificationError(DeepfakeDetectionError):
    """
    Base exception for classification errors.
    """

    pass


class ModelLoadError(ClassificationError):
    """
    Exception raised when loading a trained model fails.
    """

    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize model load error.

        Args:
            message: Error message.
            model_path: Path to the model file.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, details=details, **kwargs)


class PredictionError(ClassificationError):
    """
    Exception raised when model prediction fails.
    """

    def __init__(
        self,
        message: str,
        input_shape: Optional[tuple] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize prediction error.

        Args:
            message: Error message.
            input_shape: Shape of the input features.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if input_shape:
            details["input_shape"] = input_shape
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Pipeline Errors
# =============================================================================

class PipelineError(DeepfakeDetectionError):
    """
    Base exception for pipeline orchestration errors.
    """

    pass


class TaskError(PipelineError):
    """
    Exception raised when a pipeline task fails.
    """

    def __init__(
        self,
        message: str,
        task_name: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize task error.

        Args:
            message: Error message.
            task_name: Name of the failed task.
            task_id: Unique task identifier.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if task_name:
            details["task_name"] = task_name
        if task_id:
            details["task_id"] = task_id
        super().__init__(message, details=details, **kwargs)


class OrchestrationError(PipelineError):
    """
    Exception raised for DAG execution and orchestration errors.
    """

    def __init__(
        self,
        message: str,
        dag_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize orchestration error.

        Args:
            message: Error message.
            dag_id: DAG identifier.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if dag_id:
            details["dag_id"] = dag_id
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Service Errors
# =============================================================================

class ServiceError(DeepfakeDetectionError):
    """
    Base exception for microservice communication errors.
    """

    pass


class ServiceConnectionError(ServiceError):
    """
    Exception raised when connecting to a service fails.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize service connection error.

        Args:
            message: Error message.
            service_name: Name of the service.
            host: Service host.
            port: Service port.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if host:
            details["host"] = host
        if port:
            details["port"] = port
        super().__init__(message, details=details, **kwargs)


class ServiceTimeoutError(ServiceError):
    """
    Exception raised when a service call times out.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize service timeout error.

        Args:
            message: Error message.
            service_name: Name of the service.
            timeout: Timeout value in seconds.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if timeout:
            details["timeout"] = timeout
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(DeepfakeDetectionError):
    """
    Exception raised for input validation errors.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize validation error.

        Args:
            message: Error message.
            field: Name of the invalid field.
            value: The invalid value.
            **kwargs: Additional arguments.
        """
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate long values
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Utility Functions
# =============================================================================

def wrap_exception(
    exception: Exception,
    wrapper_class: type[DeepfakeDetectionError] = DeepfakeDetectionError,
    message: Optional[str] = None,
) -> DeepfakeDetectionError:
    """
    Wrap a standard exception in a custom exception.

    This utility helps convert third-party library exceptions into
    our custom exception hierarchy while preserving the original
    exception information.

    Args:
        exception: The original exception.
        wrapper_class: The wrapper exception class to use.
        message: Optional message override.

    Returns:
        A wrapped exception instance.

    Example:
        >>> try:
        ...     cv2.imread("nonexistent.jpg")
        ... except Exception as e:
        ...     raise wrap_exception(e, VideoProcessingError)
    """
    return wrapper_class(
        message=message or str(exception),
        cause=exception,
        details={"original_type": type(exception).__name__},
    )
