"""
Classification Module
======================

This module provides classification capabilities for deepfake detection.

Components:
    - DeepfakeClassifier: Binary classifier for real/fake detection

Example Usage:
    >>> from src.classifier import DeepfakeClassifier
    >>> classifier = DeepfakeClassifier(input_dim=4096, model_type="mlp")
    >>> classifier.fit(X_train, y_train)
    >>> predictions = classifier.predict(X_test)
"""

from src.classifier.deepfake_classifier import DeepfakeClassifier

__all__ = [
    "DeepfakeClassifier",
]
