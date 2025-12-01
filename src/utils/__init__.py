"""
Utility modules for the Deepfake Detection System.

This package provides common utilities used across all services:
    - config: Configuration loading and management
    - logging: Structured logging with rotation
    - exceptions: Custom exception hierarchy
    - visualization: Debug and analysis visualizations
    - metrics: Evaluation metrics computation
"""

from src.utils.config import Config, load_config
from src.utils.logging import setup_logging, get_logger
from src.utils.exceptions import (
    DeepfakeDetectionError,
    PreprocessingError,
    FeatureExtractionError,
    ClassificationError,
    PipelineError,
)

__all__ = [
    "Config",
    "load_config",
    "setup_logging",
    "get_logger",
    "DeepfakeDetectionError",
    "PreprocessingError",
    "FeatureExtractionError",
    "ClassificationError",
    "PipelineError",
]
