"""
Retrieval-Augmented Detection Module
=====================================

This module provides retrieval-augmented detection (RAD) capabilities
for improving deepfake detection accuracy.

Components:
    - VectorStore: FAISS-based vector storage and retrieval
    - FeatureAugmenter: Query augmentation with neighbor features

The RAD approach retrieves similar samples from a reference set
and fuses their features with the query to improve detection.

Example Usage:
    >>> from src.retrieval import VectorStore, FeatureAugmenter
    >>>
    >>> store = VectorStore(dimension=2048)
    >>> store.add(feature_vectors, labels)
    >>>
    >>> augmenter = FeatureAugmenter(store, k=5)
    >>> augmented = augmenter.augment(query_vector)
"""

from src.retrieval.vector_store import VectorStore
from src.retrieval.augmenter import FeatureAugmenter

__all__ = [
    "VectorStore",
    "FeatureAugmenter",
]
