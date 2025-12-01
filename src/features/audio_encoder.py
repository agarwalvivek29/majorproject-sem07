"""
Audio Feature Encoder Module
==============================

This module provides audio feature encoding capabilities for the
deepfake detection pipeline.

Features:
    - MFCC-based encoding with delta features
    - Neural audio encoding with wav2vec 2.0
    - Temporal aggregation methods
    - Batch processing support

Microservice API:
    POST /encode
        Request: {"mfccs": [[...], ...]}
        Response: {"embedding": [...]}

Example Usage:
    >>> from src.features.audio_encoder import AudioEncoder
    >>> encoder = AudioEncoder(method="mfcc", include_deltas=True)
    >>> embedding = encoder.encode(mfccs)  # (39,) for MFCC+deltas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import AudioEncodingError

# Module logger
logger = get_logger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import transformers for wav2vec
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False
    logger.info("Transformers not available, wav2vec encoding disabled")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AudioEncoderConfig:
    """
    Configuration for audio encoder.

    Attributes:
        method: Encoding method ("mfcc", "wav2vec").
        include_deltas: Include delta features for MFCC.
        embedding_dim: Output embedding dimension.
        aggregation: Temporal aggregation method.
    """
    method: str = "mfcc"
    include_deltas: bool = True
    embedding_dim: int = 39  # 13 MFCC + 13 delta + 13 delta-delta
    aggregation: str = "mean"


# =============================================================================
# Neural Audio Encoder
# =============================================================================

class Wav2VecEncoder(nn.Module):
    """
    Wav2Vec 2.0 based audio encoder.

    Uses self-supervised speech representations for robust
    audio feature extraction.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze: bool = True,
    ) -> None:
        """
        Initialize Wav2Vec encoder.

        Args:
            model_name: HuggingFace model identifier.
            freeze: Freeze model weights.
        """
        super().__init__()

        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.embedding_dim = 768  # Base model dimension

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            waveform: Audio waveform (batch, samples).

        Returns:
            Audio embeddings (batch, time, 768).
        """
        outputs = self.model(waveform)
        return outputs.last_hidden_state


class MFCCEncoder(nn.Module):
    """
    Simple MLP encoder for MFCC features.

    Projects MFCC features to a fixed embedding dimension.
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize MFCC encoder.

        Args:
            input_dim: Input MFCC dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.embedding_dim = output_dim

    def forward(self, mfccs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mfccs: MFCC features (batch, time, features).

        Returns:
            Encoded features (batch, time, output_dim).
        """
        return self.encoder(mfccs)


# =============================================================================
# Audio Encoder Service
# =============================================================================

class AudioEncoder:
    """
    Audio feature encoding service.

    This class provides audio feature encoding capabilities,
    supporting both traditional MFCC-based encoding and neural
    methods like wav2vec 2.0.

    Attributes:
        method: Encoding method.
        include_deltas: Include delta features.
        aggregation: Temporal aggregation method.
        embedding_dim: Output embedding dimension.

    Example:
        >>> encoder = AudioEncoder(method="mfcc", include_deltas=True)
        >>> embedding = encoder.encode(mfcc_features)
    """

    def __init__(
        self,
        method: str = "mfcc",
        include_deltas: bool = True,
        aggregation: str = "mean",
        hidden_dim: int = 128,
        output_dim: int = 256,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize audio encoder.

        Args:
            method: Encoding method ("mfcc", "wav2vec").
            include_deltas: Include delta features for MFCC.
            aggregation: Temporal aggregation ("mean", "max", "last").
            hidden_dim: Hidden dimension for neural encoder.
            output_dim: Output embedding dimension.
            device: Compute device.

        Raises:
            AudioEncodingError: If initialization fails.
        """
        self.method = method.lower()
        self.include_deltas = include_deltas
        self.aggregation = aggregation.lower()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Determine device
        if device is None:
            if TORCH_AVAILABLE:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = None
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else None

        # Calculate embedding dimension
        if self.method == "mfcc":
            # 13 MFCC coefficients, optionally with deltas
            base_dim = 13
            self.input_dim = base_dim * (3 if include_deltas else 1)
            self.embedding_dim = self.input_dim
        elif self.method == "wav2vec":
            self.input_dim = 768
            self.embedding_dim = 768
        else:
            raise AudioEncodingError(
                f"Unsupported method: {method}",
                method=method,
            )

        # Initialize neural encoder if using PyTorch
        self._init_encoder()

        logger.info(
            f"AudioEncoder initialized: method={method}, "
            f"embedding_dim={self.embedding_dim}"
        )

    def _init_encoder(self) -> None:
        """Initialize the encoder model."""
        self.model = None

        if self.method == "wav2vec":
            if not WAV2VEC_AVAILABLE:
                raise AudioEncodingError(
                    "Wav2Vec not available. Install transformers.",
                    method="wav2vec",
                )
            self.model = Wav2VecEncoder()
            self.model = self.model.to(self.device)
            self.model.eval()

        elif self.method == "mfcc" and TORCH_AVAILABLE:
            # Optional neural projection for MFCCs
            self.model = MFCCEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = self.output_dim

    def compute_deltas(self, mfccs: np.ndarray) -> np.ndarray:
        """
        Compute delta and delta-delta features.

        Args:
            mfccs: MFCC features (time, n_mfcc).

        Returns:
            MFCCs with deltas (time, n_mfcc * 3).
        """
        # Simple delta computation using finite differences
        n_frames = len(mfccs)

        if n_frames < 3:
            # Not enough frames for delta computation
            zeros = np.zeros_like(mfccs)
            return np.concatenate([mfccs, zeros, zeros], axis=1)

        # Compute first-order delta
        delta = np.zeros_like(mfccs)
        for t in range(n_frames):
            t_minus = max(0, t - 1)
            t_plus = min(n_frames - 1, t + 1)
            delta[t] = (mfccs[t_plus] - mfccs[t_minus]) / 2

        # Compute second-order delta (delta-delta)
        delta2 = np.zeros_like(mfccs)
        for t in range(n_frames):
            t_minus = max(0, t - 1)
            t_plus = min(n_frames - 1, t + 1)
            delta2[t] = (delta[t_plus] - delta[t_minus]) / 2

        return np.concatenate([mfccs, delta, delta2], axis=1)

    def _aggregate(self, features: np.ndarray) -> np.ndarray:
        """
        Aggregate temporal features.

        Args:
            features: Temporal features (time, dim).

        Returns:
            Aggregated features (dim,).
        """
        if self.aggregation == "mean":
            return np.mean(features, axis=0)
        elif self.aggregation == "max":
            return np.max(features, axis=0)
        elif self.aggregation == "last":
            return features[-1]
        elif self.aggregation == "first":
            return features[0]
        else:
            # Default to mean
            return np.mean(features, axis=0)

    @log_execution_time()
    def encode_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Encode a single audio segment.

        Args:
            segment: Audio features (time, features) or raw waveform.

        Returns:
            Feature embedding (embedding_dim,).
        """
        if self.method == "mfcc":
            # Add deltas if needed
            if self.include_deltas and segment.shape[1] == 13:
                segment = self.compute_deltas(segment)

            # Use neural encoder if available
            if self.model is not None and TORCH_AVAILABLE:
                with torch.no_grad():
                    tensor = torch.from_numpy(segment).float().unsqueeze(0)
                    tensor = tensor.to(self.device)
                    encoded = self.model(tensor)
                    segment = encoded.squeeze(0).cpu().numpy()

            # Aggregate over time
            return self._aggregate(segment)

        elif self.method == "wav2vec":
            if self.model is None:
                raise AudioEncodingError(
                    "Wav2Vec model not initialized",
                    method="wav2vec",
                )

            with torch.no_grad():
                # Assume segment is raw waveform
                tensor = torch.from_numpy(segment).float().unsqueeze(0)
                tensor = tensor.to(self.device)
                encoded = self.model(tensor)
                features = encoded.squeeze(0).cpu().numpy()

            return self._aggregate(features)

        else:
            raise AudioEncodingError(
                f"Unsupported method: {self.method}",
                method=self.method,
            )

    @log_execution_time()
    def encode(
        self,
        audio_features: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """
        Encode audio features.

        Args:
            audio_features: Audio features array or list of segments.

        Returns:
            Encoded embedding(s).

        Example:
            >>> embedding = encoder.encode(mfcc_sequence)
            >>> print(embedding.shape)  # (39,) or (256,)
        """
        if isinstance(audio_features, list):
            # Encode list of segments
            embeddings = [self.encode_segment(seg) for seg in audio_features]
            return np.array(embeddings)
        else:
            return self.encode_segment(audio_features)

    def encode_batch(
        self,
        batch: List[np.ndarray],
    ) -> np.ndarray:
        """
        Encode a batch of audio features.

        Args:
            batch: List of audio feature arrays.

        Returns:
            Batch of embeddings (batch_size, embedding_dim).
        """
        if not batch:
            return np.zeros((0, self.embedding_dim))

        embeddings = []
        for features in batch:
            embedding = self.encode_segment(features)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.embedding_dim

    def save(self, path: str) -> None:
        """Save encoder weights."""
        if self.model is not None and TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Saved audio encoder to: {path}")

    def load(self, path: str) -> None:
        """Load encoder weights."""
        if self.model is not None and TORCH_AVAILABLE:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded audio encoder from: {path}")


# =============================================================================
# Service Factory
# =============================================================================

def create_audio_encoder_service(config: Optional[Dict[str, Any]] = None) -> AudioEncoder:
    """
    Factory function to create an AudioEncoder from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured AudioEncoder instance.
    """
    if config is None:
        config = {}

    return AudioEncoder(
        method=config.get("method", "mfcc"),
        include_deltas=config.get("include_deltas", True),
        aggregation=config.get("aggregation", "mean"),
        hidden_dim=config.get("hidden_dim", 128),
        output_dim=config.get("output_dim", 256),
        device=config.get("device"),
    )
