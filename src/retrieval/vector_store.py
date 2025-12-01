"""
Vector Store Module
====================

This module provides FAISS-based vector storage and retrieval
for the retrieval-augmented detection pipeline.

Features:
    - Multiple index types (Flat, IVF, IVFPQ, HNSW)
    - L2 and cosine distance metrics
    - Metadata storage
    - Index persistence
    - GPU acceleration support

Microservice API:
    POST /add
        Request: {"vectors": [[...]], "labels": [...], "metadata": [...]}
        Response: {"count": int}
    POST /search
        Request: {"query": [...], "k": int}
        Response: {"indices": [...], "distances": [...], "labels": [...]}

Example Usage:
    >>> from src.retrieval.vector_store import VectorStore
    >>> store = VectorStore(dimension=2048, index_type="flat")
    >>> store.add(vectors, labels)
    >>> distances, indices, labels = store.search(query, k=5)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import IndexBuildError, SearchError

# Module logger
logger = get_logger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True

    # Check for GPU support
    try:
        faiss.get_num_gpus()
        FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
    except Exception:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    logger.warning("FAISS not available, vector search will be limited")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """
    Container for search results.

    Attributes:
        distances: Distance to each neighbor.
        indices: Index of each neighbor in the store.
        labels: Label of each neighbor.
        metadata: Metadata of each neighbor.
    """
    distances: np.ndarray
    indices: np.ndarray
    labels: np.ndarray
    metadata: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VectorStoreConfig:
    """
    Configuration for vector store.

    Attributes:
        dimension: Vector dimension.
        index_type: Index type (flat, ivf, ivfpq, hnsw).
        metric: Distance metric (l2, cosine).
        nlist: Number of clusters for IVF.
        nprobe: Number of clusters to search.
        m: Number of subquantizers for PQ.
        nbits: Bits per subquantizer.
    """
    dimension: int = 2048
    index_type: str = "flat"
    metric: str = "l2"
    nlist: int = 100
    nprobe: int = 10
    m: int = 8
    nbits: int = 8


# =============================================================================
# Vector Store
# =============================================================================

class VectorStore:
    """
    FAISS-based vector storage and retrieval service.

    This class provides efficient similarity search capabilities
    using Facebook's FAISS library.

    Attributes:
        dimension: Vector dimension.
        index_type: Index algorithm.
        metric: Distance metric.
        normalize: Whether to L2 normalize vectors.

    Example:
        >>> store = VectorStore(dimension=2048, index_type="flat")
        >>> store.add(feature_vectors, labels)
        >>> result = store.search(query_vector, k=5)
        >>> print(f"Nearest neighbor distance: {result.distances[0]:.4f}")
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "l2",
        normalize: bool = True,
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 8,
        nbits: int = 8,
        use_gpu: bool = False,
    ) -> None:
        """
        Initialize vector store.

        Args:
            dimension: Vector dimension.
            index_type: Index type ("flat", "ivf", "ivfpq", "hnsw").
            metric: Distance metric ("l2", "cosine").
            normalize: L2 normalize vectors before indexing.
            nlist: Number of clusters for IVF indices.
            nprobe: Number of clusters to search.
            m: Number of subquantizers for PQ.
            nbits: Bits per subquantizer.
            use_gpu: Use GPU acceleration if available.

        Raises:
            IndexBuildError: If initialization fails.
        """
        if not FAISS_AVAILABLE:
            raise IndexBuildError(
                "FAISS not available. Install with: pip install faiss-cpu",
                index_type=index_type,
            )

        self.dimension = dimension
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.normalize = normalize
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        self.use_gpu = use_gpu and FAISS_GPU_AVAILABLE

        # Storage for labels and metadata
        self.labels: List[int] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

        # Initialize index
        self.index: Optional[faiss.Index] = None
        self._is_trained = False

        logger.info(
            f"VectorStore initialized: dimension={dimension}, "
            f"index_type={index_type}, metric={metric}"
        )

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        try:
            # Determine metric type
            if self.metric == "cosine":
                # Use inner product after L2 normalization
                metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                metric_type = faiss.METRIC_L2

            # Create index based on type
            if self.index_type == "flat":
                if self.metric == "cosine":
                    index = faiss.IndexFlatIP(self.dimension)
                else:
                    index = faiss.IndexFlatL2(self.dimension)

            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, metric_type
                )

            elif self.index_type == "ivfpq":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFPQ(
                    quantizer, self.dimension, self.nlist, self.m, self.nbits
                )

            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)

            else:
                raise IndexBuildError(
                    f"Unsupported index type: {self.index_type}",
                    index_type=self.index_type,
                )

            # Move to GPU if requested
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU for vector search")

            return index

        except Exception as e:
            raise IndexBuildError(
                f"Failed to create index: {e}",
                index_type=self.index_type,
                dimension=self.dimension,
                cause=e,
            )

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors."""
        if not self.normalize:
            return vectors

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return vectors / norms

    @log_execution_time()
    def add(
        self,
        vectors: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add vectors to the index.

        Args:
            vectors: Feature vectors (N, dimension).
            labels: Ground truth labels (N,). 0=real, 1=fake.
            metadata: Per-vector metadata.

        Returns:
            Number of vectors added.

        Raises:
            IndexBuildError: If adding fails.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise IndexBuildError(
                f"Invalid vector shape: {vectors.shape}, expected (N, {self.dimension})",
                dimension=self.dimension,
            )

        n_vectors = vectors.shape[0]

        # Ensure float32
        vectors = vectors.astype(np.float32)

        # Normalize if needed
        vectors = self._normalize_vectors(vectors)

        try:
            # Create index if needed
            if self.index is None:
                self.index = self._create_index()

            # Train index if needed (for IVF types)
            if not self._is_trained and self.index_type in ("ivf", "ivfpq"):
                logger.info(f"Training index with {n_vectors} vectors")
                self.index.train(vectors)
                self._is_trained = True

            # Add vectors
            self.index.add(vectors)

            # Store vectors for reconstruction
            if self.vectors is None:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])

            # Store labels
            if labels is not None:
                self.labels.extend(labels.tolist())
            else:
                self.labels.extend([0] * n_vectors)

            # Store metadata
            if metadata is not None:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * n_vectors)

            # Set nprobe for IVF indices
            if self.index_type in ("ivf", "ivfpq"):
                self.index.nprobe = self.nprobe

            logger.info(f"Added {n_vectors} vectors. Total: {self.index.ntotal}")
            return n_vectors

        except Exception as e:
            raise IndexBuildError(
                f"Failed to add vectors: {e}",
                cause=e,
            )

    @log_execution_time()
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> SearchResult:
        """
        Search for nearest neighbors.

        Args:
            query: Query vector(s) (dimension,) or (N, dimension).
            k: Number of neighbors to retrieve.

        Returns:
            SearchResult with distances, indices, labels, and metadata.

        Raises:
            SearchError: If search fails.
        """
        if self.index is None or self.index.ntotal == 0:
            raise SearchError(
                "Index is empty. Add vectors before searching.",
                k=k,
            )

        # Ensure 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)
            single_query = True
        else:
            single_query = False

        # Ensure float32
        query = query.astype(np.float32)

        # Normalize
        query = self._normalize_vectors(query)

        # Limit k to index size
        k = min(k, self.index.ntotal)

        try:
            # Search
            distances, indices = self.index.search(query, k)

            # Get labels
            labels = np.array([
                [self.labels[i] if 0 <= i < len(self.labels) else -1 for i in row]
                for row in indices
            ])

            # Get metadata
            metadata_list = []
            for row in indices:
                row_metadata = [
                    self.metadata[i] if 0 <= i < len(self.metadata) else {}
                    for i in row
                ]
                metadata_list.append(row_metadata)

            if single_query:
                distances = distances[0]
                indices = indices[0]
                labels = labels[0]
                metadata_list = metadata_list[0]

            return SearchResult(
                distances=distances,
                indices=indices,
                labels=labels,
                metadata=metadata_list,
            )

        except Exception as e:
            raise SearchError(
                f"Search failed: {e}",
                k=k,
                cause=e,
            )

    def get_vector(self, index: int) -> Optional[np.ndarray]:
        """
        Get vector at specified index.

        Args:
            index: Vector index.

        Returns:
            Vector or None if index is invalid.
        """
        if self.vectors is None or index < 0 or index >= len(self.vectors):
            return None
        return self.vectors[index]

    def get_vectors(self, indices: List[int]) -> np.ndarray:
        """
        Get multiple vectors by indices.

        Args:
            indices: List of indices.

        Returns:
            Vectors array (N, dimension).
        """
        if self.vectors is None:
            return np.zeros((len(indices), self.dimension))

        valid_indices = [i for i in indices if 0 <= i < len(self.vectors)]
        if not valid_indices:
            return np.zeros((len(indices), self.dimension))

        return self.vectors[valid_indices]

    def save(self, path: Union[str, Path]) -> None:
        """
        Save index and metadata to disk.

        Args:
            path: Directory path for saving.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            index_path = path / "index.faiss"
            # Move to CPU if on GPU
            if self.use_gpu:
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))

        # Save vectors
        if self.vectors is not None:
            np.save(path / "vectors.npy", self.vectors)

        # Save labels and metadata
        meta = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize": self.normalize,
            "labels": self.labels,
            "metadata": self.metadata,
            "is_trained": self._is_trained,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f)

        logger.info(f"Saved vector store to: {path}")

    def load(self, path: Union[str, Path]) -> None:
        """
        Load index and metadata from disk.

        Args:
            path: Directory path to load from.
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            meta = json.load(f)

        self.dimension = meta["dimension"]
        self.index_type = meta["index_type"]
        self.metric = meta["metric"]
        self.normalize = meta["normalize"]
        self.labels = meta["labels"]
        self.metadata = meta["metadata"]
        self._is_trained = meta.get("is_trained", True)

        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Load vectors
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            self.vectors = np.load(vectors_path)

        logger.info(f"Loaded vector store from: {path}")

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.index = None
        self.labels = []
        self.metadata = []
        self.vectors = None
        self._is_trained = False
        logger.info("Vector store cleared")

    def __len__(self) -> int:
        """Return number of vectors in the store."""
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def ntotal(self) -> int:
        """Return total number of vectors."""
        return len(self)


# =============================================================================
# Service Factory
# =============================================================================

def create_vector_store_service(config: Optional[Dict[str, Any]] = None) -> VectorStore:
    """
    Factory function to create a VectorStore from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured VectorStore instance.
    """
    if config is None:
        config = {}

    return VectorStore(
        dimension=config.get("dimension", 2048),
        index_type=config.get("index_type", "flat"),
        metric=config.get("metric", "l2"),
        normalize=config.get("normalize", True),
        nlist=config.get("nlist", 100),
        nprobe=config.get("nprobe", 10),
        m=config.get("m", 8),
        nbits=config.get("nbits", 8),
        use_gpu=config.get("use_gpu", False),
    )
