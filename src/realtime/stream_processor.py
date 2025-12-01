"""
Stream Processor Module
========================

This module provides real-time video stream processing for deepfake detection.

Features:
    - Live video stream processing (RTMP, RTSP, HLS, webcam)
    - Frame buffering and batching
    - Sliding window analysis
    - Real-time detection with alerting
    - Performance monitoring

Example Usage:
    >>> processor = StreamProcessor(model_dir="models/")
    >>> processor.process_stream("rtmp://source/stream", callback=on_detection)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from src.utils.logging import get_logger
from src.utils.exceptions import StreamingError
from src.preprocessing import VideoProcessor, FaceDetector, AudioProcessor
from src.features import VisualEncoder, TemporalModel, AudioEncoder, CorrelationAnalyzer
from src.retrieval import VectorStore, FeatureAugmenter
from src.classifier import DeepfakeClassifier

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class StreamStatus(Enum):
    """Stream processing status."""
    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamConfig:
    """
    Configuration for stream processing.

    Attributes:
        buffer_size: Number of frames to buffer.
        batch_size: Frames per analysis batch.
        window_size: Sliding window size for temporal analysis.
        window_stride: Sliding window stride.
        detection_threshold: Threshold for alerting.
        min_faces: Minimum faces required for analysis.
        fps_limit: Maximum FPS to process.
        skip_frames: Frames to skip between analyses.
    """
    buffer_size: int = 300
    batch_size: int = 30
    window_size: int = 60
    window_stride: int = 15
    detection_threshold: float = 0.5
    min_faces: int = 1
    fps_limit: float = 30.0
    skip_frames: int = 1


@dataclass
class FrameBuffer:
    """
    Circular buffer for frame storage.

    Attributes:
        max_size: Maximum buffer size.
        frames: Stored frames.
        timestamps: Frame timestamps.
    """
    max_size: int = 300
    frames: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def add(self, frame: np.ndarray, timestamp: float) -> None:
        """Add frame to buffer."""
        self.frames.append(frame)
        self.timestamps.append(timestamp)

        # Remove old frames if buffer is full
        while len(self.frames) > self.max_size:
            self.frames.pop(0)
            self.timestamps.pop(0)

    def get_window(self, size: int) -> List[np.ndarray]:
        """Get the most recent frames."""
        return self.frames[-size:] if len(self.frames) >= size else self.frames.copy()

    def clear(self) -> None:
        """Clear the buffer."""
        self.frames.clear()
        self.timestamps.clear()

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class DetectionEvent:
    """
    Container for detection events.

    Attributes:
        timestamp: Event timestamp.
        stream_url: Source stream URL.
        is_fake: Detection result.
        probability: Fake probability.
        confidence: Detection confidence.
        frame_index: Frame index in stream.
        frame_snapshot: Snapshot frame (optional).
    """
    timestamp: datetime = field(default_factory=datetime.now)
    stream_url: str = ""
    is_fake: bool = False
    probability: float = 0.0
    confidence: float = 0.0
    frame_index: int = 0
    frame_snapshot: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "stream_url": self.stream_url,
            "is_fake": self.is_fake,
            "probability": self.probability,
            "confidence": self.confidence,
            "frame_index": self.frame_index,
        }


@dataclass
class StreamStats:
    """
    Statistics for stream processing.

    Attributes:
        total_frames: Total frames processed.
        analyzed_windows: Windows analyzed.
        detections: Number of detections.
        avg_fps: Average processing FPS.
        avg_latency: Average detection latency.
        start_time: Processing start time.
    """
    total_frames: int = 0
    analyzed_windows: int = 0
    detections: int = 0
    avg_fps: float = 0.0
    avg_latency: float = 0.0
    start_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        runtime = 0.0
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_frames": self.total_frames,
            "analyzed_windows": self.analyzed_windows,
            "detections": self.detections,
            "avg_fps": self.avg_fps,
            "avg_latency_ms": self.avg_latency * 1000,
            "runtime_seconds": runtime,
        }


# =============================================================================
# Stream Processor
# =============================================================================

class StreamProcessor:
    """
    Real-time video stream processor for deepfake detection.

    This class processes live video streams and performs
    real-time deepfake detection using a sliding window approach.

    Attributes:
        model_dir: Directory containing trained models.
        config: Stream processing configuration.
        status: Current processing status.

    Example:
        >>> processor = StreamProcessor(model_dir="models/")
        >>> processor.process_stream(
        ...     "rtmp://source/stream",
        ...     callback=lambda e: print(f"Detection: {e.is_fake}")
        ... )
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "models/",
        config: Optional[StreamConfig] = None,
        device: Optional[str] = None,
        alerter: Optional[Any] = None,
    ) -> None:
        """
        Initialize stream processor.

        Args:
            model_dir: Directory containing trained models.
            config: Stream processing configuration.
            device: Compute device.
            alerter: Alert manager for notifications.
        """
        self.model_dir = Path(model_dir)
        self.config = config or StreamConfig()
        self.device = device
        self.alerter = alerter

        self.status = StreamStatus.IDLE
        self.stats = StreamStats()
        self._frame_buffer = FrameBuffer(max_size=self.config.buffer_size)
        self._audio_buffer: List[np.ndarray] = []

        # Threading
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=100)
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        # Initialize components
        self._initialized = False
        self._initialize_components()

        logger.info(f"StreamProcessor initialized: model_dir={model_dir}")

    def _initialize_components(self) -> None:
        """Initialize detection components."""
        if self._initialized:
            return

        logger.info("Initializing stream processing components...")

        try:
            # Face detection
            self.face_detector = FaceDetector(
                backend="mediapipe",
                output_size=(224, 224),
            )

            # Feature extraction
            self.visual_encoder = VisualEncoder(
                model_name="resnet50",
                device=self.device,
            )

            self.temporal_model = TemporalModel(
                architecture="lstm",
                input_dim=self.visual_encoder.embedding_dim,
                hidden_dim=256,
                device=self.device,
            )

            self.audio_encoder = AudioEncoder(method="mfcc")

            self.correlation_analyzer = CorrelationAnalyzer(
                fps=25.0,
                genuine_threshold=0.5,
            )

            # Load classifier if available
            classifier_path = self.model_dir / "classifier"
            if classifier_path.exists():
                self.classifier = DeepfakeClassifier(input_dim=1, model_type="mlp")
                self.classifier.load(classifier_path)
                logger.info(f"Loaded classifier from: {classifier_path}")
            else:
                self.classifier = None
                logger.warning("Classifier not found, using correlation-based detection")

            # Load vector store if available
            vector_store_path = self.model_dir / "vector_store"
            if vector_store_path.exists():
                self.vector_store = VectorStore(dimension=1)
                self.vector_store.load(vector_store_path)
                self.augmenter = FeatureAugmenter(self.vector_store, k=5)
                logger.info(f"Loaded vector store from: {vector_store_path}")
            else:
                self.vector_store = None
                self.augmenter = None

            self._initialized = True
            logger.info("Stream processing components initialized")

        except Exception as e:
            raise StreamingError(f"Failed to initialize components: {e}")

    def process_stream(
        self,
        source: Union[str, int],
        callback: Optional[Callable[[DetectionEvent], None]] = None,
        duration: Optional[float] = None,
    ) -> StreamStats:
        """
        Process a video stream.

        Args:
            source: Stream URL or camera index.
            callback: Function called on each detection.
            duration: Maximum processing duration in seconds.

        Returns:
            StreamStats with processing statistics.

        Example:
            >>> stats = processor.process_stream(
            ...     "rtmp://source/stream",
            ...     callback=lambda e: print(e.to_dict()),
            ...     duration=3600  # 1 hour
            ... )
        """
        self.status = StreamStatus.CONNECTING
        self.stats = StreamStats(start_time=datetime.now())
        self._stop_event.clear()

        try:
            # Open video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise StreamingError(f"Failed to open stream: {source}")

            self.status = StreamStatus.RUNNING
            logger.info(f"Started processing stream: {source}")

            # Get stream properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_time = 1.0 / min(fps, self.config.fps_limit)

            frame_count = 0
            last_analysis_frame = 0
            start_time = time.time()

            while not self._stop_event.is_set():
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    logger.info("Duration limit reached")
                    break

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

                frame_count += 1
                self.stats.total_frames = frame_count

                # Skip frames based on configuration
                if frame_count % (self.config.skip_frames + 1) != 0:
                    continue

                # Add to buffer
                self._frame_buffer.add(frame, time.time())

                # Check if we should analyze
                if (frame_count - last_analysis_frame) >= self.config.window_stride:
                    if len(self._frame_buffer) >= self.config.window_size:
                        # Perform analysis
                        event = self._analyze_window(
                            source=str(source),
                            frame_index=frame_count,
                        )

                        self.stats.analyzed_windows += 1

                        if event and event.is_fake:
                            self.stats.detections += 1

                            # Trigger callback
                            if callback:
                                callback(event)

                            # Send alert
                            if self.alerter:
                                self.alerter.send_alert(event)

                        last_analysis_frame = frame_count

                # Update FPS
                elapsed = time.time() - start_time
                self.stats.avg_fps = frame_count / elapsed if elapsed > 0 else 0

                # Rate limiting
                time.sleep(frame_time * 0.1)

        except Exception as e:
            self.status = StreamStatus.ERROR
            logger.exception(f"Stream processing error: {e}")
            raise StreamingError(f"Stream processing failed: {e}")

        finally:
            cap.release()
            self.status = StreamStatus.STOPPED
            logger.info(f"Stream processing stopped: {self.stats.to_dict()}")

        return self.stats

    def _analyze_window(
        self,
        source: str,
        frame_index: int,
    ) -> Optional[DetectionEvent]:
        """Analyze a window of frames."""
        analysis_start = time.time()

        try:
            # Get frames from buffer
            frames = self._frame_buffer.get_window(self.config.window_size)
            if len(frames) < self.config.batch_size:
                return None

            # Detect faces
            face_crops = []
            landmarks_list = []
            for frame in frames:
                result = self.face_detector.detect_and_crop(frame)
                if result.success:
                    face_crops.append(result.face_crop)
                    landmarks_list.append(
                        result.landmarks.to_dict() if result.landmarks else None
                    )

            if len(face_crops) < self.config.min_faces:
                return None

            # Extract visual features
            visual_embeddings = self.visual_encoder.encode(face_crops)
            temporal_embedding = self.temporal_model.encode(visual_embeddings)

            # Create dummy audio features (in real implementation, would capture audio)
            audio_embedding = np.zeros(128)

            # Compute correlation
            correlation = self.correlation_analyzer.analyze(
                face_crops=face_crops,
                mfcc_sequence=np.zeros((len(face_crops), 13)),  # Placeholder
                landmarks_sequence=landmarks_list,
            )

            # Combine features
            feature_vector = np.concatenate([
                temporal_embedding.flatten(),
                audio_embedding.flatten(),
                correlation.to_feature_vector(),
            ])

            # Classify
            if self.classifier is not None:
                if self.augmenter is not None:
                    aug_result = self.augmenter.augment(feature_vector)
                    feature_vector = aug_result.augmented_vector

                probability = self.classifier.predict_proba(
                    feature_vector.reshape(1, -1)
                )[0]
                is_fake = probability > self.config.detection_threshold
                confidence = abs(probability - 0.5) * 2
            else:
                # Fallback to correlation-based detection
                probability = 1.0 - max(0, min(1, (correlation.pearson_correlation + 1) / 2))
                is_fake = correlation.pearson_correlation < 0.3
                confidence = 0.5

            # Update latency stats
            latency = time.time() - analysis_start
            self.stats.avg_latency = (
                (self.stats.avg_latency * (self.stats.analyzed_windows - 1) + latency)
                / self.stats.analyzed_windows
                if self.stats.analyzed_windows > 0
                else latency
            )

            # Create event
            event = DetectionEvent(
                stream_url=source,
                is_fake=is_fake,
                probability=float(probability),
                confidence=float(confidence),
                frame_index=frame_index,
                frame_snapshot=frames[-1].copy() if is_fake else None,
            )

            return event

        except Exception as e:
            logger.warning(f"Window analysis failed: {e}")
            return None

    def start_async(
        self,
        source: Union[str, int],
        callback: Optional[Callable[[DetectionEvent], None]] = None,
    ) -> None:
        """
        Start stream processing in background thread.

        Args:
            source: Stream URL or camera index.
            callback: Function called on each detection.
        """
        if self.status == StreamStatus.RUNNING:
            logger.warning("Stream already running")
            return

        self._stop_event.clear()

        def _run():
            try:
                self.process_stream(source, callback)
            except Exception as e:
                logger.exception(f"Async stream error: {e}")

        self._process_thread = threading.Thread(target=_run, daemon=True)
        self._process_thread.start()
        logger.info("Started async stream processing")

    def stop(self) -> None:
        """Stop stream processing."""
        logger.info("Stopping stream processing...")
        self._stop_event.set()

        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=5.0)

        self.status = StreamStatus.STOPPED
        self._frame_buffer.clear()

    def pause(self) -> None:
        """Pause stream processing."""
        if self.status == StreamStatus.RUNNING:
            self.status = StreamStatus.PAUSED
            logger.info("Stream processing paused")

    def resume(self) -> None:
        """Resume stream processing."""
        if self.status == StreamStatus.PAUSED:
            self.status = StreamStatus.RUNNING
            logger.info("Stream processing resumed")

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.to_dict()


# =============================================================================
# Service Factory
# =============================================================================

def create_stream_processor(
    config: Optional[Dict[str, Any]] = None,
) -> StreamProcessor:
    """
    Factory function to create a StreamProcessor.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured StreamProcessor instance.
    """
    if config is None:
        config = {}

    stream_config = StreamConfig(
        buffer_size=config.get("buffer_size", 300),
        batch_size=config.get("batch_size", 30),
        window_size=config.get("window_size", 60),
        window_stride=config.get("window_stride", 15),
        detection_threshold=config.get("detection_threshold", 0.5),
        fps_limit=config.get("fps_limit", 30.0),
    )

    return StreamProcessor(
        model_dir=config.get("model_dir", "models/"),
        config=stream_config,
        device=config.get("device"),
    )
