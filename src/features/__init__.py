"""
Feature Extraction Module
==========================

This module provides feature extraction services for visual, temporal,
and audio modalities, as well as correlation analysis for lip-sync detection.

Components:
    - VisualEncoder: CNN-based visual feature extraction
    - TemporalModel: LSTM/Transformer for temporal modeling
    - AudioEncoder: MFCC and neural audio encoders
    - CorrelationAnalyzer: Lip-audio synchronization analysis

Example Usage:
    >>> from src.features import VisualEncoder, TemporalModel, CorrelationAnalyzer
    >>>
    >>> visual = VisualEncoder(model="resnet50")
    >>> frame_embeddings = visual.encode(face_crops)
    >>>
    >>> temporal = TemporalModel(architecture="lstm")
    >>> video_embedding = temporal.encode(frame_embeddings)
    >>>
    >>> analyzer = CorrelationAnalyzer()
    >>> correlation = analyzer.compute_correlation(mouth_signal, audio_energy)
"""

from src.features.visual_encoder import VisualEncoder
from src.features.temporal_model import TemporalModel
from src.features.audio_encoder import AudioEncoder
from src.features.correlation_analyzer import CorrelationAnalyzer

__all__ = [
    "VisualEncoder",
    "TemporalModel",
    "AudioEncoder",
    "CorrelationAnalyzer",
]
