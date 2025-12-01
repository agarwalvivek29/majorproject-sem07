"""
Pipeline Module
================

This module provides end-to-end pipelines for deepfake detection.

Components:
    - TrainingPipeline: Train detection models
    - DetectionPipeline: Inference on videos
    - DeepfakeDetector: High-level detector interface

Example Usage:
    >>> from src.pipeline import DeepfakeDetector
    >>> detector = DeepfakeDetector(model_dir="models/")
    >>> result = detector.detect("video.mp4")
    >>> print(f"Fake probability: {result.probability:.2%}")
"""

from src.pipeline.detector import DeepfakeDetector, DetectionResult
from src.pipeline.trainer import TrainingPipeline

__all__ = [
    "DeepfakeDetector",
    "DetectionResult",
    "TrainingPipeline",
]
