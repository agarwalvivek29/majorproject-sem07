"""
Feature Augmenter Module
=========================

This module provides feature augmentation capabilities for the
retrieval-augmented detection pipeline.

Features:
    - Query augmentation with k-NN features
    - Multiple aggregation strategies
    - Label-based neighbor filtering
    - Distance-weighted averaging

Example Usage:
    >>> from src.retrieval.augmenter import FeatureAugmenter
    >>> augmenter = FeatureAugmenter(vector_store, k=5)
    >>> augmented = augmenter.augment(query_vector)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.retrieval.vector_store import VectorStore, SearchResult

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AggregationMethod(Enum):
    """Feature aggregation methods."""
    CONCAT = "concat"           # Concatenate query with mean of neighbors
    MEAN = "mean"               # Mean of query and neighbors
    WEIGHTED = "weighted"       # Distance-weighted mean
    ATTENTION = "attention"     # Attention-based aggregation


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AugmentationResult:
    """
    Container for augmentation results.

    Attributes:
        augmented_vector: The augmented feature vector.
        query_vector: Original query vector.
        neighbor_vectors: Retrieved neighbor vectors.
        neighbor_labels: Labels of neighbors.
        neighbor_distances: Distances to neighbors.
        aggregation_weights: Weights used in aggregation.
    """
    augmented_vector: np.ndarray
    query_vector: np.ndarray
    neighbor_vectors: np.ndarray
    neighbor_labels: np.ndarray
    neighbor_distances: np.ndarray
    aggregation_weights: Optional[np.ndarray] = None

    @property
    def neighbor_vote(self) -> float:
        """Get majority vote from neighbors (proportion of fake)."""
        if len(self.neighbor_labels) == 0:
            return 0.5
        return float(np.mean(self.neighbor_labels))


# =============================================================================
# Feature Augmenter
# =============================================================================

class FeatureAugmenter:
    """
    Feature augmentation service for RAD pipeline.

    This class augments query features with information from
    similar samples in the reference set.

    Attributes:
        vector_store: Vector store with reference samples.
        k: Number of neighbors to retrieve.
        aggregation: Aggregation method.
        include_query: Include query in aggregation.

    Example:
        >>> augmenter = FeatureAugmenter(store, k=5)
        >>> result = augmenter.augment(query_vector)
        >>> classifier.predict(result.augmented_vector)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 5,
        aggregation: str = "concat",
        include_query: bool = True,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize feature augmenter.

        Args:
            vector_store: Initialized vector store with training data.
            k: Number of neighbors to retrieve.
            aggregation: Aggregation method ("concat", "mean", "weighted").
            include_query: Include query vector in aggregation.
            temperature: Temperature for distance weighting.
        """
        self.vector_store = vector_store
        self.k = k
        self.aggregation = AggregationMethod(aggregation.lower())
        self.include_query = include_query
        self.temperature = temperature

        # Calculate output dimension
        self._calculate_output_dim()

        logger.info(
            f"FeatureAugmenter initialized: k={k}, "
            f"aggregation={aggregation}, output_dim={self.output_dim}"
        )

    def _calculate_output_dim(self) -> None:
        """Calculate output dimension based on aggregation method."""
        input_dim = self.vector_store.dimension

        if self.aggregation == AggregationMethod.CONCAT:
            # Query + aggregated neighbors
            self.output_dim = input_dim * 2
        else:
            # Same as input
            self.output_dim = input_dim

    def _compute_weights(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:
        """
        Compute aggregation weights from distances.

        Args:
            distances: Distances to neighbors (k,).

        Returns:
            Normalized weights (k,).
        """
        if len(distances) == 0:
            return np.array([])

        # Convert distances to similarities
        # Use softmax with temperature
        similarities = -distances / self.temperature
        similarities = similarities - np.max(similarities)  # Numerical stability
        weights = np.exp(similarities)
        weights = weights / (np.sum(weights) + 1e-8)

        return weights

    def _aggregate_concat(
        self,
        query: np.ndarray,
        neighbors: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Concatenate query with mean of neighbors.

        Args:
            query: Query vector (d,).
            neighbors: Neighbor vectors (k, d).
            weights: Optional weights for neighbors.

        Returns:
            Concatenated vector (2d,).
        """
        if len(neighbors) == 0:
            neighbor_mean = np.zeros_like(query)
        elif weights is not None:
            neighbor_mean = np.average(neighbors, axis=0, weights=weights)
        else:
            neighbor_mean = np.mean(neighbors, axis=0)

        return np.concatenate([query, neighbor_mean])

    def _aggregate_mean(
        self,
        query: np.ndarray,
        neighbors: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute mean of query and neighbors.

        Args:
            query: Query vector (d,).
            neighbors: Neighbor vectors (k, d).
            weights: Optional weights for neighbors.

        Returns:
            Mean vector (d,).
        """
        if len(neighbors) == 0:
            return query

        if self.include_query:
            all_vectors = np.vstack([query.reshape(1, -1), neighbors])
            if weights is not None:
                all_weights = np.concatenate([[1.0], weights])
                all_weights = all_weights / all_weights.sum()
                return np.average(all_vectors, axis=0, weights=all_weights)
            return np.mean(all_vectors, axis=0)
        else:
            if weights is not None:
                return np.average(neighbors, axis=0, weights=weights)
            return np.mean(neighbors, axis=0)

    def _aggregate_weighted(
        self,
        query: np.ndarray,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Distance-weighted aggregation.

        Args:
            query: Query vector (d,).
            neighbors: Neighbor vectors (k, d).
            weights: Distance-based weights (k,).

        Returns:
            Weighted vector (d,).
        """
        if len(neighbors) == 0:
            return query

        if self.include_query:
            # Give query weight equal to sum of neighbor weights
            query_weight = np.sum(weights)
            all_vectors = np.vstack([query.reshape(1, -1), neighbors])
            all_weights = np.concatenate([[query_weight], weights])
            all_weights = all_weights / all_weights.sum()
            return np.average(all_vectors, axis=0, weights=all_weights)
        else:
            return np.average(neighbors, axis=0, weights=weights)

    @log_execution_time()
    def augment(
        self,
        query: np.ndarray,
        filter_by_label: Optional[int] = None,
    ) -> AugmentationResult:
        """
        Augment query vector with neighbor information.

        Args:
            query: Query feature vector (d,).
            filter_by_label: Only use neighbors with this label.

        Returns:
            AugmentationResult with augmented vector.

        Example:
            >>> result = augmenter.augment(query_vector)
            >>> prediction = classifier.predict(result.augmented_vector)
        """
        # Ensure 1D
        query = query.flatten()

        # Search for neighbors
        search_result = self.vector_store.search(query, k=self.k)

        # Get neighbor vectors
        neighbor_vectors = self.vector_store.get_vectors(
            search_result.indices.tolist()
        )
        neighbor_labels = search_result.labels
        neighbor_distances = search_result.distances

        # Filter by label if specified
        if filter_by_label is not None:
            mask = neighbor_labels == filter_by_label
            neighbor_vectors = neighbor_vectors[mask]
            neighbor_labels = neighbor_labels[mask]
            neighbor_distances = neighbor_distances[mask]

        # Compute weights
        weights = self._compute_weights(neighbor_distances)

        # Aggregate based on method
        if self.aggregation == AggregationMethod.CONCAT:
            augmented = self._aggregate_concat(query, neighbor_vectors, weights)
        elif self.aggregation == AggregationMethod.MEAN:
            augmented = self._aggregate_mean(query, neighbor_vectors, weights)
        elif self.aggregation == AggregationMethod.WEIGHTED:
            augmented = self._aggregate_weighted(query, neighbor_vectors, weights)
        else:
            augmented = self._aggregate_concat(query, neighbor_vectors, weights)

        return AugmentationResult(
            augmented_vector=augmented,
            query_vector=query,
            neighbor_vectors=neighbor_vectors,
            neighbor_labels=neighbor_labels,
            neighbor_distances=neighbor_distances,
            aggregation_weights=weights,
        )

    def augment_batch(
        self,
        queries: np.ndarray,
    ) -> List[AugmentationResult]:
        """
        Augment a batch of query vectors.

        Args:
            queries: Query vectors (N, d).

        Returns:
            List of AugmentationResults.
        """
        results = []
        for query in queries:
            result = self.augment(query)
            results.append(result)
        return results

    def get_augmented_vectors(
        self,
        queries: np.ndarray,
    ) -> np.ndarray:
        """
        Get augmented vectors for a batch of queries.

        Args:
            queries: Query vectors (N, d).

        Returns:
            Augmented vectors (N, output_dim).
        """
        results = self.augment_batch(queries)
        return np.array([r.augmented_vector for r in results])

    def get_neighbor_votes(
        self,
        queries: np.ndarray,
    ) -> np.ndarray:
        """
        Get neighbor label votes for classification.

        Args:
            queries: Query vectors (N, d).

        Returns:
            Vote proportions (N,). Higher = more neighbors are fake.
        """
        results = self.augment_batch(queries)
        return np.array([r.neighbor_vote for r in results])

    def get_output_dim(self) -> int:
        """Get output dimension of augmented vectors."""
        return self.output_dim


# =============================================================================
# Ensemble Augmenter
# =============================================================================

class EnsembleAugmenter:
    """
    Ensemble of augmenters with different configurations.

    Combines multiple augmentation strategies for more robust
    feature augmentation.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k_values: List[int] = [3, 5, 10],
        aggregations: List[str] = ["concat", "weighted"],
    ) -> None:
        """
        Initialize ensemble augmenter.

        Args:
            vector_store: Vector store with reference samples.
            k_values: Different k values to use.
            aggregations: Different aggregation methods to use.
        """
        self.augmenters: List[FeatureAugmenter] = []

        for k in k_values:
            for agg in aggregations:
                augmenter = FeatureAugmenter(
                    vector_store=vector_store,
                    k=k,
                    aggregation=agg,
                )
                self.augmenters.append(augmenter)

        # Calculate total output dimension
        self.output_dim = sum(a.output_dim for a in self.augmenters)

        logger.info(
            f"EnsembleAugmenter initialized with {len(self.augmenters)} augmenters"
        )

    def augment(self, query: np.ndarray) -> np.ndarray:
        """
        Augment query with all augmenters.

        Args:
            query: Query vector (d,).

        Returns:
            Concatenated augmented vector.
        """
        augmented_parts = []
        for augmenter in self.augmenters:
            result = augmenter.augment(query)
            augmented_parts.append(result.augmented_vector)

        return np.concatenate(augmented_parts)

    def augment_batch(self, queries: np.ndarray) -> np.ndarray:
        """
        Augment batch of queries.

        Args:
            queries: Query vectors (N, d).

        Returns:
            Augmented vectors (N, output_dim).
        """
        return np.array([self.augment(q) for q in queries])


# =============================================================================
# Service Factory
# =============================================================================

def create_augmenter_service(
    vector_store: VectorStore,
    config: Optional[Dict[str, Any]] = None,
) -> FeatureAugmenter:
    """
    Factory function to create a FeatureAugmenter from configuration.

    Args:
        vector_store: Initialized vector store.
        config: Configuration dictionary.

    Returns:
        Configured FeatureAugmenter instance.
    """
    if config is None:
        config = {}

    return FeatureAugmenter(
        vector_store=vector_store,
        k=config.get("k_neighbors", 5),
        aggregation=config.get("aggregation", "concat"),
        include_query=config.get("include_query", True),
        temperature=config.get("temperature", 1.0),
    )
