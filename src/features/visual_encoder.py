"""
Visual Feature Encoder Module
==============================

This module provides CNN-based visual feature extraction for face images
in the deepfake detection pipeline.

Features:
    - Multiple backbone support (ResNet-50, ResNet-18, MobileNetV3)
    - Pre-trained ImageNet weights
    - Batch processing for efficiency
    - GPU acceleration support
    - Fine-tuning capabilities

Microservice API:
    POST /encode
        Request: {"images": [base64_image, ...]}
        Response: {"embeddings": [[...], ...]}

Example Usage:
    >>> from src.features.visual_encoder import VisualEncoder
    >>> encoder = VisualEncoder(model="resnet50", pretrained=True)
    >>> embeddings = encoder.encode(face_crops)  # (N, 2048)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import VisualEncodingError

# Module logger
logger = get_logger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, visual encoding will be limited")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EncoderConfig:
    """
    Configuration for visual encoder.

    Attributes:
        model_name: Backbone model name.
        pretrained: Use pretrained weights.
        embedding_dim: Output embedding dimension.
        input_size: Expected input image size.
        normalize_mean: ImageNet normalization mean.
        normalize_std: ImageNet normalization std.
    """
    model_name: str = "resnet50"
    pretrained: bool = True
    embedding_dim: int = 2048
    input_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)


# =============================================================================
# Model Definitions
# =============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet-based visual encoder.

    Extracts features from the penultimate layer (before classification).
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
    ) -> None:
        """
        Initialize ResNet encoder.

        Args:
            model_name: ResNet variant ("resnet18", "resnet34", "resnet50").
            pretrained: Use ImageNet pretrained weights.
        """
        super().__init__()

        # Load pretrained model
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.embedding_dim = 2048
        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self.embedding_dim = 512
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            self.embedding_dim = 512
        else:
            raise ValueError(f"Unsupported ResNet variant: {model_name}")

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Feature tensor of shape (batch, embedding_dim).
        """
        return self.backbone(x)


class MobileNetEncoder(nn.Module):
    """
    MobileNetV3-based visual encoder.

    Lightweight encoder for resource-constrained environments.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """
        Initialize MobileNet encoder.

        Args:
            pretrained: Use ImageNet pretrained weights.
        """
        super().__init__()

        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)
        self.embedding_dim = 960

        # Remove classifier
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)


# =============================================================================
# Visual Encoder Service
# =============================================================================

class VisualEncoder:
    """
    Visual feature extraction service.

    This class provides CNN-based visual feature extraction for face images,
    supporting multiple backbone architectures and GPU acceleration.

    Attributes:
        model_name: Backbone model name.
        pretrained: Whether to use pretrained weights.
        device: Compute device (cuda/cpu).
        batch_size: Batch size for inference.
        embedding_dim: Output embedding dimension.

    Example:
        >>> encoder = VisualEncoder(model="resnet50", device="cuda")
        >>> embeddings = encoder.encode(face_crops)
        >>> print(embeddings.shape)  # (N, 2048)
    """

    # Model name to embedding dimension mapping
    MODEL_DIMS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "mobilenetv3": 960,
    }

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        fine_tune: bool = False,
    ) -> None:
        """
        Initialize the visual encoder.

        Args:
            model_name: Backbone model ("resnet50", "resnet18", "mobilenetv3").
            pretrained: Use ImageNet pretrained weights.
            device: Compute device ("cuda", "cpu", or None for auto).
            batch_size: Batch size for inference.
            fine_tune: Enable gradient computation for fine-tuning.

        Raises:
            VisualEncodingError: If initialization fails.
        """
        if not TORCH_AVAILABLE:
            raise VisualEncodingError(
                "PyTorch not available for visual encoding",
                code="TORCH_NOT_AVAILABLE",
            )

        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.fine_tune = fine_tune

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Get embedding dimension
        self.embedding_dim = self.MODEL_DIMS.get(self.model_name, 2048)

        # Initialize model
        self._init_model()

        # Initialize preprocessing transform
        self._init_transform()

        logger.info(
            f"VisualEncoder initialized: model={model_name}, "
            f"device={self.device}, embedding_dim={self.embedding_dim}"
        )

    def _init_model(self) -> None:
        """Initialize the backbone model."""
        try:
            if self.model_name.startswith("resnet"):
                self.model = ResNetEncoder(self.model_name, self.pretrained)
            elif self.model_name == "mobilenetv3":
                self.model = MobileNetEncoder(self.pretrained)
            else:
                raise VisualEncodingError(
                    f"Unsupported model: {self.model_name}",
                    model_name=self.model_name,
                )

            # Move to device
            self.model = self.model.to(self.device)

            # Set evaluation mode unless fine-tuning
            if not self.fine_tune:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

        except Exception as e:
            raise VisualEncodingError(
                f"Failed to initialize model: {e}",
                model_name=self.model_name,
                cause=e,
            )

    def _init_transform(self) -> None:
        """Initialize image preprocessing transform."""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image for the model.

        Args:
            image: BGR numpy array (H, W, 3).

        Returns:
            Preprocessed tensor (3, 224, 224).
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()

        return self.transform(image)

    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of BGR numpy arrays.

        Returns:
            Batch tensor (N, 3, 224, 224).
        """
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors)

    @log_execution_time()
    def encode(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """
        Extract embeddings from images.

        Args:
            images: Single image or list of images (BGR numpy arrays).

        Returns:
            Embeddings array of shape (N, embedding_dim).

        Example:
            >>> embeddings = encoder.encode(face_crops)
            >>> print(embeddings.shape)  # (100, 2048)
        """
        # Handle single image
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]

        if not images:
            return np.zeros((0, self.embedding_dim))

        all_embeddings = []

        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]

            try:
                # Preprocess batch
                batch_tensor = self.preprocess_batch(batch).to(self.device)

                # Forward pass
                with torch.no_grad():
                    embeddings = self.model(batch_tensor)

                # Convert to numpy
                all_embeddings.append(embeddings.cpu().numpy())

            except Exception as e:
                logger.warning(f"Batch encoding failed: {e}")
                # Return zeros for failed batch
                all_embeddings.append(np.zeros((len(batch), self.embedding_dim)))

        return np.vstack(all_embeddings)

    @log_execution_time()
    def encode_video(
        self,
        face_crops: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract embeddings for all frames of a video.

        This method handles the entire video at once, batching internally.

        Args:
            face_crops: List of face crop images.

        Returns:
            Embeddings array of shape (T, embedding_dim).

        Example:
            >>> frame_embeddings = encoder.encode_video(face_crops)
            >>> print(frame_embeddings.shape)  # (300, 2048)
        """
        return self.encode(face_crops)

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.embedding_dim

    def to(self, device: str) -> "VisualEncoder":
        """
        Move encoder to a different device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def save(self, path: str) -> None:
        """
        Save encoder weights.

        Args:
            path: Path to save weights.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved encoder weights to: {path}")

    def load(self, path: str) -> None:
        """
        Load encoder weights.

        Args:
            path: Path to load weights from.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded encoder weights from: {path}")


# =============================================================================
# Service Factory
# =============================================================================

def create_visual_encoder_service(config: Optional[Dict[str, Any]] = None) -> VisualEncoder:
    """
    Factory function to create a VisualEncoder from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured VisualEncoder instance.
    """
    if config is None:
        config = {}

    return VisualEncoder(
        model_name=config.get("model", "resnet50"),
        pretrained=config.get("pretrained", True),
        device=config.get("device"),
        batch_size=config.get("batch_size", 32),
        fine_tune=config.get("fine_tune", False),
    )
