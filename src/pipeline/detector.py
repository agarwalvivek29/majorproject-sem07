"""
Deepfake Detector Pipeline Module
===================================

This module provides the main inference pipeline for deepfake detection.

Features:
    - End-to-end video analysis
    - Model loading and management
    - Batch processing support
    - Confidence scoring
    - Detailed analysis results

Example Usage:
    >>> from src.pipeline.detector import DeepfakeDetector
    >>> detector = DeepfakeDetector(model_dir="models/")
    >>> result = detector.detect("suspicious_video.mp4")
    >>> if result.is_fake:
    ...     print(f"FAKE detected with {result.probability:.1%} confidence")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.config import load_config, Config
from src.utils.exceptions import DeepfakeDetectionError

from src.preprocessing import VideoProcessor, FaceDetector, AudioProcessor
from src.features import VisualEncoder, TemporalModel, AudioEncoder, CorrelationAnalyzer
from src.retrieval import VectorStore, FeatureAugmenter
from src.classifier import DeepfakeClassifier

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DetectionResult:
    """
    Container for detection results.

    Attributes:
        video_path: Path to analyzed video.
        is_fake: Whether video is classified as fake.
        probability: Probability of being fake (0-1).
        confidence: Detection confidence (0-1).
        correlation_score: Lip-audio correlation score.
        threshold: Decision threshold used.
        processing_time: Total processing time in seconds.
        frame_count: Number of frames analyzed.
        details: Additional analysis details.
    """
    video_path: str = ""
    is_fake: bool = False
    probability: float = 0.0
    confidence: float = 0.0
    correlation_score: float = 0.0
    threshold: float = 0.5
    processing_time: float = 0.0
    frame_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "is_fake": self.is_fake,
            "probability": self.probability,
            "confidence": self.confidence,
            "correlation_score": self.correlation_score,
            "threshold": self.threshold,
            "processing_time": self.processing_time,
            "frame_count": self.frame_count,
            "details": self.details,
        }

    @property
    def label(self) -> str:
        """Get human-readable label."""
        return "FAKE" if self.is_fake else "REAL"

    def __str__(self) -> str:
        """String representation."""
        return (
            f"DetectionResult({self.label}, "
            f"probability={self.probability:.2%}, "
            f"confidence={self.confidence:.2%})"
        )


# =============================================================================
# Deepfake Detector
# =============================================================================

class DeepfakeDetector:
    """
    End-to-end deepfake detection pipeline.

    This class provides a complete pipeline for analyzing videos
    and detecting deepfakes using audio-visual synchronization
    analysis and machine learning.

    Attributes:
        model_dir: Directory containing trained models.
        config: Detection configuration.
        device: Compute device.
        threshold: Decision threshold.

    Example:
        >>> detector = DeepfakeDetector(model_dir="models/")
        >>> result = detector.detect("video.mp4")
        >>> print(result)
        DetectionResult(FAKE, probability=87.3%, confidence=92.1%)
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "models/",
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        lazy_load: bool = False,
    ) -> None:
        """
        Initialize the detector.

        Args:
            model_dir: Directory containing trained models.
            config_path: Path to configuration file.
            device: Compute device ("cuda", "cpu").
            threshold: Decision threshold override.
            lazy_load: Delay model loading until first use.

        Raises:
            DeepfakeDetectionError: If initialization fails.
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.threshold = threshold

        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = self._load_default_config()

        # Apply overrides
        if threshold is not None:
            self.config.set("classifier.threshold", threshold)

        # Initialize components
        self._initialized = False
        if not lazy_load:
            self._initialize_components()

        logger.info(f"DeepfakeDetector initialized: model_dir={model_dir}")

    def _load_default_config(self) -> Config:
        """Load default configuration."""
        try:
            return load_config("config/default.yaml")
        except Exception:
            # Return minimal config if file not found
            return Config({
                "preprocessing": {
                    "video": {"sample_rate": 1, "max_frames": 300},
                    "face": {"backend": "mediapipe", "output_size": [224, 224]},
                    "audio": {"sample_rate": 16000, "n_mfcc": 13},
                },
                "features": {
                    "visual": {"model": "resnet50"},
                    "temporal": {"architecture": "lstm", "hidden_dim": 256},
                    "audio": {"method": "mfcc"},
                },
                "classifier": {"threshold": 0.5},
            })

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        logger.info("Initializing detector components...")

        try:
            # Preprocessing components
            self.video_processor = VideoProcessor(
                sample_rate=self.config.get("preprocessing.video.sample_rate", 1),
                max_frames=self.config.get("preprocessing.video.max_frames", 300),
            )

            self.face_detector = FaceDetector(
                backend=self.config.get("preprocessing.face.backend", "mediapipe"),
                output_size=tuple(self.config.get("preprocessing.face.output_size", [224, 224])),
            )

            self.audio_processor = AudioProcessor(
                sample_rate=self.config.get("preprocessing.audio.sample_rate", 16000),
                n_mfcc=self.config.get("preprocessing.audio.n_mfcc", 13),
            )

            # Feature extraction components
            self.visual_encoder = VisualEncoder(
                model_name=self.config.get("features.visual.model", "resnet50"),
                device=self.device,
            )

            self.temporal_model = TemporalModel(
                architecture=self.config.get("features.temporal.architecture", "lstm"),
                input_dim=self.visual_encoder.embedding_dim,
                hidden_dim=self.config.get("features.temporal.hidden_dim", 256),
                device=self.device,
            )

            self.audio_encoder = AudioEncoder(
                method=self.config.get("features.audio.method", "mfcc"),
            )

            self.correlation_analyzer = CorrelationAnalyzer(
                fps=25.0,
                genuine_threshold=0.5,
            )

            # Load trained models
            self._load_models()

            self._initialized = True
            logger.info("Detector components initialized successfully")

        except Exception as e:
            raise DeepfakeDetectionError(
                f"Failed to initialize detector: {e}",
                cause=e,
            )

    def _load_models(self) -> None:
        """Load trained models from disk."""
        classifier_path = self.model_dir / "classifier"
        vector_store_path = self.model_dir / "vector_store"

        # Load classifier
        if classifier_path.exists():
            self.classifier = DeepfakeClassifier(
                input_dim=1,  # Will be set on load
                model_type=self.config.get("classifier.model_type", "mlp"),
            )
            self.classifier.load(classifier_path)
            if self.threshold is not None:
                self.classifier.threshold = self.threshold
            logger.info(f"Loaded classifier from: {classifier_path}")
        else:
            logger.warning(f"Classifier not found at: {classifier_path}")
            self.classifier = None

        # Load vector store for RAD
        if vector_store_path.exists():
            self.vector_store = VectorStore(dimension=1)  # Will be set on load
            self.vector_store.load(vector_store_path)
            self.augmenter = FeatureAugmenter(
                self.vector_store,
                k=self.config.get("retrieval.k_neighbors", 5),
            )
            logger.info(f"Loaded vector store from: {vector_store_path}")
        else:
            logger.info("Vector store not found, RAD disabled")
            self.vector_store = None
            self.augmenter = None

    @log_execution_time()
    def detect(
        self,
        video_path: Union[str, Path],
        detailed: bool = False,
    ) -> DetectionResult:
        """
        Detect deepfakes in a video.

        Args:
            video_path: Path to the video file.
            detailed: Include detailed analysis in results.

        Returns:
            DetectionResult with analysis results.

        Example:
            >>> result = detector.detect("video.mp4")
            >>> if result.is_fake:
            ...     alert_user(result)
        """
        import time
        start_time = time.time()

        # Ensure initialized
        if not self._initialized:
            self._initialize_components()

        video_path = Path(video_path)
        result = DetectionResult(video_path=str(video_path))

        try:
            # Step 1: Preprocess video
            logger.info(f"Processing video: {video_path.name}")
            frames, fps = self.video_processor.extract_frames(video_path)
            result.frame_count = len(frames)

            if not frames:
                raise DeepfakeDetectionError(
                    f"No frames extracted from video",
                    details={"video_path": str(video_path)},
                )

            # Step 2: Detect and crop faces
            face_crops = []
            landmarks_list = []
            for frame in frames:
                face_result = self.face_detector.detect_and_crop(frame)
                if face_result.success:
                    face_crops.append(face_result.face_crop)
                    landmarks_list.append(
                        face_result.landmarks.to_dict() if face_result.landmarks else None
                    )
                else:
                    face_crops.append(None)
                    landmarks_list.append(None)

            # Filter valid crops
            valid_crops = [c for c in face_crops if c is not None]
            if not valid_crops:
                raise DeepfakeDetectionError(
                    "No faces detected in video",
                    details={"video_path": str(video_path)},
                )

            # Step 3: Extract audio features
            mfccs = self.audio_processor.extract_features(video_path, len(frames), fps)

            # Step 4: Extract visual features
            visual_embeddings = self.visual_encoder.encode(valid_crops)
            temporal_embedding = self.temporal_model.encode(visual_embeddings)

            # Step 5: Extract audio embedding
            audio_embedding = self.audio_encoder.encode(mfccs)

            # Step 6: Compute correlation
            correlation = self.correlation_analyzer.analyze(
                face_crops=[c for c in face_crops if c is not None],
                mfcc_sequence=mfccs,
                landmarks_sequence=landmarks_list,
            )
            result.correlation_score = correlation.pearson_correlation

            # Step 7: Combine features
            correlation_features = correlation.to_feature_vector()
            feature_vector = np.concatenate([
                temporal_embedding.flatten(),
                audio_embedding.flatten(),
                correlation_features,
            ])

            # Step 8: Augment with RAD (if available)
            if self.augmenter is not None:
                aug_result = self.augmenter.augment(feature_vector)
                feature_vector = aug_result.augmented_vector

            # Step 9: Classify
            if self.classifier is not None:
                probability = self.classifier.predict_proba(
                    feature_vector.reshape(1, -1)
                )[0]
                result.probability = float(probability)
                result.is_fake = probability > self.classifier.threshold
                result.threshold = self.classifier.threshold
                result.confidence = abs(probability - 0.5) * 2  # Scale to 0-1
            else:
                # Fallback to correlation-based detection
                result.probability = 1.0 - max(0, min(1, (correlation.pearson_correlation + 1) / 2))
                result.is_fake = correlation.pearson_correlation < 0.3
                result.confidence = 0.5

            # Add details if requested
            if detailed:
                result.details = {
                    "frames_processed": len(frames),
                    "faces_detected": len(valid_crops),
                    "correlation": correlation.to_dict(),
                    "visual_embedding_shape": visual_embeddings.shape,
                }

        except DeepfakeDetectionError:
            raise
        except Exception as e:
            logger.exception(f"Detection failed: {e}")
            raise DeepfakeDetectionError(
                f"Detection failed: {e}",
                details={"video_path": str(video_path)},
                cause=e,
            )
        finally:
            result.processing_time = time.time() - start_time

        logger.info(
            f"Detection complete: {result.label} "
            f"(prob={result.probability:.2%}, time={result.processing_time:.2f}s)"
        )

        return result

    def detect_batch(
        self,
        video_paths: List[Union[str, Path]],
        detailed: bool = False,
    ) -> List[DetectionResult]:
        """
        Detect deepfakes in multiple videos.

        Args:
            video_paths: List of video file paths.
            detailed: Include detailed analysis.

        Returns:
            List of DetectionResults.
        """
        results = []
        for path in video_paths:
            try:
                result = self.detect(path, detailed=detailed)
            except DeepfakeDetectionError as e:
                result = DetectionResult(
                    video_path=str(path),
                    is_fake=False,
                    probability=0.0,
                    confidence=0.0,
                    details={"error": str(e)},
                )
            results.append(result)
        return results


# =============================================================================
# Service Factory
# =============================================================================

def create_detector(config: Optional[Dict[str, Any]] = None) -> DeepfakeDetector:
    """
    Factory function to create a DeepfakeDetector.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured DeepfakeDetector instance.
    """
    if config is None:
        config = {}

    return DeepfakeDetector(
        model_dir=config.get("model_dir", "models/"),
        device=config.get("device"),
        threshold=config.get("threshold"),
    )
