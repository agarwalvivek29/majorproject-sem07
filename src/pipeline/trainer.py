"""
Training Pipeline Module
=========================

This module provides end-to-end training pipeline for deepfake detection models.

Features:
    - Dataset preprocessing and feature extraction
    - Vector store building for RAD
    - Classifier training with validation
    - Model evaluation and metrics
    - Model persistence

Example Usage:
    >>> from src.pipeline.trainer import TrainingPipeline
    >>> trainer = TrainingPipeline(config_path="config/training.yaml")
    >>> metrics = trainer.train(dataset_dir="data/raw")
    >>> print(f"Test AUC: {metrics['test_auc']:.4f}")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger, log_execution_time
from src.utils.config import load_config, Config
from src.utils.exceptions import PipelineError

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
class TrainingConfig:
    """
    Configuration for training pipeline.

    Attributes:
        dataset_dir: Path to dataset directory.
        output_dir: Path for saving models.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        device: Compute device.
    """
    dataset_dir: str = "data/raw"
    output_dir: str = "models"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    device: Optional[str] = None


@dataclass
class TrainingResult:
    """
    Container for training results.

    Attributes:
        train_metrics: Training set metrics.
        val_metrics: Validation set metrics.
        test_metrics: Test set metrics.
        model_path: Path to saved model.
        config: Training configuration used.
    """
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    model_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "model_path": self.model_path,
        }


# =============================================================================
# Training Pipeline
# =============================================================================

class TrainingPipeline:
    """
    End-to-end training pipeline for deepfake detection.

    This class handles the complete training workflow from raw videos
    to trained models, including preprocessing, feature extraction,
    and classifier training.

    Attributes:
        config: Training configuration.
        output_dir: Directory for saving models.

    Example:
        >>> trainer = TrainingPipeline(output_dir="models/")
        >>> result = trainer.train(dataset_dir="data/raw")
        >>> print(f"Test accuracy: {result.test_metrics['accuracy']:.2%}")
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "models/",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize training pipeline.

        Args:
            config_path: Path to configuration file.
            output_dir: Directory for saving trained models.
            device: Compute device.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = self._get_default_config()

        # Initialize components
        self._init_components()

        logger.info(f"TrainingPipeline initialized: output_dir={output_dir}")

    def _get_default_config(self) -> Config:
        """Get default training configuration."""
        return Config({
            "preprocessing": {
                "video": {"sample_rate": 2, "max_frames": 150},
                "face": {"backend": "mediapipe", "output_size": [224, 224]},
                "audio": {"sample_rate": 16000, "n_mfcc": 13},
            },
            "features": {
                "visual": {"model": "resnet50"},
                "temporal": {"architecture": "lstm", "hidden_dim": 256},
                "audio": {"method": "mfcc", "include_deltas": True},
            },
            "retrieval": {"k_neighbors": 5, "aggregation": "concat"},
            "classifier": {"model_type": "mlp", "hidden_dims": [256, 128]},
            "training": {
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            },
        })

    def _init_components(self) -> None:
        """Initialize pipeline components."""
        logger.info("Initializing training components...")

        # Preprocessing
        self.video_processor = VideoProcessor(
            sample_rate=self.config.get("preprocessing.video.sample_rate", 2),
            max_frames=self.config.get("preprocessing.video.max_frames", 150),
        )

        self.face_detector = FaceDetector(
            backend=self.config.get("preprocessing.face.backend", "mediapipe"),
            output_size=tuple(self.config.get("preprocessing.face.output_size", [224, 224])),
        )

        self.audio_processor = AudioProcessor(
            sample_rate=self.config.get("preprocessing.audio.sample_rate", 16000),
            n_mfcc=self.config.get("preprocessing.audio.n_mfcc", 13),
        )

        # Feature extraction
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

        self.correlation_analyzer = CorrelationAnalyzer(fps=25.0)

        logger.info("Training components initialized")

    @log_execution_time()
    def preprocess_dataset(
        self,
        dataset_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess all videos in dataset and extract features.

        Args:
            dataset_dir: Path to dataset with real/ and fake/ subdirectories.
            cache_dir: Optional directory for caching features.

        Returns:
            Tuple of (feature_vectors, labels, video_paths).
        """
        dataset_dir = Path(dataset_dir)
        features_list = []
        labels_list = []
        paths_list = []

        for label, subdir in [(0, "real"), (1, "fake")]:
            class_dir = dataset_dir / subdir
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue

            video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
            logger.info(f"Found {len(video_files)} videos in {subdir}/")

            for video_path in video_files:
                try:
                    features = self._extract_features(video_path)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(label)
                        paths_list.append(str(video_path))
                except Exception as e:
                    logger.warning(f"Failed to process {video_path.name}: {e}")

        if not features_list:
            raise PipelineError(
                "No features extracted from dataset",
                details={"dataset_dir": str(dataset_dir)},
            )

        features = np.array(features_list)
        labels = np.array(labels_list)

        logger.info(
            f"Preprocessed {len(features)} videos: "
            f"{sum(labels == 0)} real, {sum(labels == 1)} fake"
        )

        return features, labels, paths_list

    def _extract_features(self, video_path: Path) -> Optional[np.ndarray]:
        """Extract features from a single video."""
        # Extract frames
        frames, fps = self.video_processor.extract_frames(video_path)
        if not frames:
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

        if not face_crops:
            return None

        # Extract audio features
        mfccs = self.audio_processor.extract_features(video_path, len(frames), fps)

        # Visual features
        visual_embeddings = self.visual_encoder.encode(face_crops)
        temporal_embedding = self.temporal_model.encode(visual_embeddings)

        # Audio features
        audio_embedding = self.audio_encoder.encode(mfccs)

        # Correlation features
        correlation = self.correlation_analyzer.analyze(
            face_crops=face_crops,
            mfcc_sequence=mfccs,
            landmarks_sequence=landmarks_list,
        )

        # Combine all features
        feature_vector = np.concatenate([
            temporal_embedding.flatten(),
            audio_embedding.flatten(),
            correlation.to_feature_vector(),
        ])

        return feature_vector

    @log_execution_time()
    def train(
        self,
        dataset_dir: Union[str, Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> TrainingResult:
        """
        Train the detection model.

        Args:
            dataset_dir: Path to dataset directory.
            train_ratio: Ratio of data for training.
            val_ratio: Ratio of data for validation.

        Returns:
            TrainingResult with metrics and model path.
        """
        result = TrainingResult()

        # Step 1: Preprocess dataset
        logger.info("Step 1: Preprocessing dataset...")
        features, labels, paths = self.preprocess_dataset(dataset_dir)

        # Step 2: Split data
        logger.info("Step 2: Splitting data...")
        test_ratio = 1.0 - train_ratio - val_ratio

        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels,
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=42,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=y_temp,
            random_state=42,
        )

        logger.info(
            f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        # Step 3: Build vector store for RAD
        logger.info("Step 3: Building vector store...")
        vector_store = VectorStore(
            dimension=features.shape[1],
            index_type=self.config.get("retrieval.index_type", "flat"),
        )
        vector_store.add(X_train, y_train)

        # Save vector store
        vector_store_path = self.output_dir / "vector_store"
        vector_store.save(vector_store_path)
        logger.info(f"Saved vector store to: {vector_store_path}")

        # Step 4: Create augmenter and augment features
        logger.info("Step 4: Augmenting features with RAD...")
        augmenter = FeatureAugmenter(
            vector_store,
            k=self.config.get("retrieval.k_neighbors", 5),
            aggregation=self.config.get("retrieval.aggregation", "concat"),
        )

        X_train_aug = augmenter.get_augmented_vectors(X_train)
        X_val_aug = augmenter.get_augmented_vectors(X_val)
        X_test_aug = augmenter.get_augmented_vectors(X_test)

        # Step 5: Train classifier
        logger.info("Step 5: Training classifier...")
        classifier = DeepfakeClassifier(
            input_dim=X_train_aug.shape[1],
            model_type=self.config.get("classifier.model_type", "mlp"),
            hidden_dims=self.config.get("classifier.hidden_dims", [256, 128]),
            device=self.device,
        )

        train_metrics = classifier.fit(
            X_train_aug, y_train,
            X_val_aug, y_val,
            epochs=self.config.get("training.epochs", 50),
            batch_size=self.config.get("training.batch_size", 32),
            learning_rate=self.config.get("training.learning_rate", 0.001),
        )

        result.train_metrics = train_metrics.to_dict()

        # Step 6: Evaluate on test set
        logger.info("Step 6: Evaluating on test set...")
        test_probs = classifier.predict_proba(X_test_aug)
        test_preds = classifier.predict(X_test_aug)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        result.test_metrics = {
            "accuracy": accuracy_score(y_test, test_preds),
            "precision": precision_score(y_test, test_preds, zero_division=0),
            "recall": recall_score(y_test, test_preds, zero_division=0),
            "f1": f1_score(y_test, test_preds, zero_division=0),
            "auc": roc_auc_score(y_test, test_probs),
        }

        # Step 7: Save classifier
        classifier_path = self.output_dir / "classifier"
        classifier.save(classifier_path)
        result.model_path = str(classifier_path)
        logger.info(f"Saved classifier to: {classifier_path}")

        # Save training config
        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Training complete! Test metrics: {result.test_metrics}")

        return result


# =============================================================================
# Service Factory
# =============================================================================

def create_training_pipeline(
    config: Optional[Dict[str, Any]] = None,
) -> TrainingPipeline:
    """
    Factory function to create a TrainingPipeline.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured TrainingPipeline instance.
    """
    if config is None:
        config = {}

    return TrainingPipeline(
        output_dir=config.get("output_dir", "models/"),
        device=config.get("device"),
    )
