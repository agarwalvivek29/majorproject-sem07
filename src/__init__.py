"""
Audio-Visual Deepfake Detection System
=======================================

A modular, microservices-based system for detecting lip-sync deepfakes
and fake news content through audio-visual synchronization analysis.

This package provides:
    - Preprocessing services for video, face, and audio extraction
    - Feature extraction using CNNs, LSTMs, and audio encoders
    - Retrieval-augmented detection (RAD) with FAISS
    - Binary classification for real/fake detection
    - Real-time streaming detection with alerting
    - Pipeline orchestration for batch processing

Example Usage:
    >>> from src.pipeline.detector import DeepfakeDetector
    >>> detector = DeepfakeDetector(model_dir="models/")
    >>> result = detector.detect("video.mp4")
    >>> print(f"Fake probability: {result.fake_probability:.2%}")

Architecture:
    The system follows a microservices architecture where each component
    can run independently and communicate via REST APIs or message queues.
    Pipeline orchestration is handled by a DAG-based system similar to
    Apache Airflow.

Author: AI-Generated Implementation
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Deepfake Detection Team"

# Package-level imports for convenience
from src.utils.config import Config, load_config
from src.utils.logging import setup_logging, get_logger

__all__ = [
    "Config",
    "load_config",
    "setup_logging",
    "get_logger",
    "__version__",
]
