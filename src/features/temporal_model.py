"""
Temporal Modeling Module
=========================

This module provides temporal sequence modeling for video frame embeddings
in the deepfake detection pipeline.

Features:
    - LSTM-based temporal modeling
    - Transformer-based attention modeling
    - 1D CNN temporal convolutions
    - Bidirectional processing support
    - Variable-length sequence handling

Microservice API:
    POST /encode
        Request: {"embeddings": [[...], ...], "sequence_length": int}
        Response: {"temporal_embedding": [...]}

Example Usage:
    >>> from src.features.temporal_model import TemporalModel
    >>> model = TemporalModel(architecture="lstm", hidden_dim=256)
    >>> temporal_embedding = model.encode(frame_embeddings)  # (512,)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import TemporalEncodingError

# Module logger
logger = get_logger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, temporal modeling will be limited")


# =============================================================================
# Model Architectures
# =============================================================================

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for temporal sequences.

    Captures forward and backward temporal dependencies in
    frame embedding sequences.

    Attributes:
        input_dim: Input feature dimension.
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        bidirectional: Use bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        """
        Initialize LSTM encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            bidirectional: Use bidirectional LSTM.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output dimension
        self.output_dim = hidden_dim * self.num_directions

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            lengths: Sequence lengths for packing (optional).

        Returns:
            Output tensor (batch, output_dim).
        """
        batch_size = x.size(0)

        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM forward
        _, (hidden, _) = self.lstm(x)

        # Reshape hidden state
        # hidden: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        # Apply layer norm
        output = self.layer_norm(hidden)

        return output


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for temporal sequences.

    Uses self-attention to capture long-range dependencies
    between frames.

    Attributes:
        input_dim: Input feature dimension.
        hidden_dim: Transformer hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_seq_len: int = 500,
    ) -> None:
        """
        Initialize Transformer encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Transformer hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_seq_len)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # CLS token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            mask: Attention mask (optional).

        Returns:
            Output tensor (batch, output_dim).
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_proj(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use CLS token output as sequence representation
        output = x[:, 0, :]

        # Apply layer norm
        output = self.layer_norm(output)

        return output


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv1DEncoder(nn.Module):
    """
    1D Convolutional encoder for temporal sequences.

    Uses stacked 1D convolutions with pooling for
    temporal feature extraction.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize Conv1D encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of conv layers.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
        """
        super().__init__()

        self.output_dim = hidden_dim

        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            out_channels = hidden_dim
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim).

        Returns:
            Output tensor (batch, output_dim).
        """
        # Transpose for conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutions
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x).squeeze(-1)

        return x


# =============================================================================
# Temporal Model Service
# =============================================================================

class TemporalModel:
    """
    Temporal sequence modeling service.

    This class provides temporal modeling capabilities for video frame
    embeddings, capturing temporal dynamics in lip movements.

    Attributes:
        architecture: Model architecture (lstm, transformer, conv1d).
        input_dim: Input embedding dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output embedding dimension.
        device: Compute device.

    Example:
        >>> model = TemporalModel(architecture="lstm", hidden_dim=256)
        >>> temporal_embedding = model.encode(frame_embeddings)
    """

    def __init__(
        self,
        architecture: str = "lstm",
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_heads: int = 8,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize temporal model.

        Args:
            architecture: Architecture type ("lstm", "transformer", "conv1d").
            input_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of layers.
            dropout: Dropout rate.
            bidirectional: Use bidirectional (LSTM only).
            num_heads: Number of attention heads (Transformer only).
            device: Compute device.

        Raises:
            TemporalEncodingError: If initialization fails.
        """
        if not TORCH_AVAILABLE:
            raise TemporalEncodingError(
                "PyTorch not available for temporal modeling",
                code="TORCH_NOT_AVAILABLE",
            )

        self.architecture = architecture.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self._init_model(dropout, bidirectional, num_heads)

        logger.info(
            f"TemporalModel initialized: architecture={architecture}, "
            f"input_dim={input_dim}, output_dim={self.output_dim}"
        )

    def _init_model(
        self,
        dropout: float,
        bidirectional: bool,
        num_heads: int,
    ) -> None:
        """Initialize the model architecture."""
        try:
            if self.architecture == "lstm":
                self.model = LSTMEncoder(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            elif self.architecture == "transformer":
                self.model = TransformerEncoder(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            elif self.architecture == "conv1d":
                self.model = Conv1DEncoder(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=dropout,
                )
            else:
                raise TemporalEncodingError(
                    f"Unsupported architecture: {self.architecture}",
                    code="UNSUPPORTED_ARCHITECTURE",
                )

            self.output_dim = self.model.output_dim
            self.model = self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise TemporalEncodingError(
                f"Failed to initialize temporal model: {e}",
                cause=e,
            )

    @log_execution_time()
    def encode(
        self,
        frame_embeddings: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Encode a sequence of frame embeddings.

        Args:
            frame_embeddings: Frame embeddings (T, input_dim) or (B, T, input_dim).

        Returns:
            Temporal embedding (output_dim,) or (B, output_dim).

        Example:
            >>> embeddings = model.encode(frame_embeddings)
            >>> print(embeddings.shape)  # (512,)
        """
        # Convert to tensor if needed
        if isinstance(frame_embeddings, np.ndarray):
            frame_embeddings = torch.from_numpy(frame_embeddings).float()

        # Add batch dimension if needed
        if frame_embeddings.dim() == 2:
            frame_embeddings = frame_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Move to device
        frame_embeddings = frame_embeddings.to(self.device)

        try:
            with torch.no_grad():
                output = self.model(frame_embeddings)

            # Convert to numpy
            output = output.cpu().numpy()

            if squeeze_output:
                output = output.squeeze(0)

            return output

        except Exception as e:
            raise TemporalEncodingError(
                f"Temporal encoding failed: {e}",
                sequence_length=frame_embeddings.size(1),
                cause=e,
            )

    def encode_batch(
        self,
        sequences: List[np.ndarray],
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """
        Encode a batch of variable-length sequences.

        Args:
            sequences: List of frame embedding arrays.
            pad_value: Value to use for padding.

        Returns:
            Batch of temporal embeddings (B, output_dim).
        """
        if not sequences:
            return np.zeros((0, self.output_dim))

        # Get maximum length
        max_len = max(len(seq) for seq in sequences)

        # Pad sequences
        padded = np.full(
            (len(sequences), max_len, self.input_dim),
            pad_value,
            dtype=np.float32,
        )
        lengths = []

        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
            lengths.append(len(seq))

        # Convert to tensor
        padded_tensor = torch.from_numpy(padded).to(self.device)
        lengths_tensor = torch.tensor(lengths)

        try:
            with torch.no_grad():
                if self.architecture == "lstm":
                    output = self.model(padded_tensor, lengths_tensor)
                else:
                    output = self.model(padded_tensor)

            return output.cpu().numpy()

        except Exception as e:
            raise TemporalEncodingError(
                f"Batch encoding failed: {e}",
                cause=e,
            )

    def get_output_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def save(self, path: str) -> None:
        """
        Save model weights.

        Args:
            path: Path to save weights.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved temporal model to: {path}")

    def load(self, path: str) -> None:
        """
        Load model weights.

        Args:
            path: Path to load weights from.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded temporal model from: {path}")


# =============================================================================
# Service Factory
# =============================================================================

def create_temporal_model_service(config: Optional[Dict[str, Any]] = None) -> TemporalModel:
    """
    Factory function to create a TemporalModel from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured TemporalModel instance.
    """
    if config is None:
        config = {}

    return TemporalModel(
        architecture=config.get("architecture", "lstm"),
        input_dim=config.get("input_dim", 2048),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
        bidirectional=config.get("bidirectional", True),
        num_heads=config.get("num_heads", 8),
        device=config.get("device"),
    )
