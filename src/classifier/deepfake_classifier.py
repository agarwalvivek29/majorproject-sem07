"""
Deepfake Classifier Module
===========================

This module provides binary classification for deepfake detection.

Features:
    - Multiple model types (Logistic Regression, MLP, SVM, XGBoost)
    - Training with validation
    - Probability calibration
    - Threshold optimization
    - Model persistence

Microservice API:
    POST /train
        Request: {"features": [[...]], "labels": [...]}
        Response: {"metrics": {...}}
    POST /predict
        Request: {"features": [[...]]}
        Response: {"predictions": [...], "probabilities": [...]}

Example Usage:
    >>> from src.classifier.deepfake_classifier import DeepfakeClassifier
    >>> classifier = DeepfakeClassifier(input_dim=4096, model_type="mlp")
    >>> classifier.fit(X_train, y_train)
    >>> probs = classifier.predict_proba(X_test)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import ClassificationError, ModelLoadError, PredictionError

# Module logger
logger = get_logger(__name__)

# Try to import PyTorch for MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingMetrics:
    """
    Container for training metrics.

    Attributes:
        accuracy: Classification accuracy.
        precision: Precision for fake class.
        recall: Recall for fake class.
        f1: F1 score for fake class.
        auc: Area under ROC curve.
        confusion_matrix: Confusion matrix.
        threshold: Optimal decision threshold.
    """
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    threshold: float = 0.5
    val_accuracy: float = 0.0
    val_loss: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "threshold": self.threshold,
            "val_accuracy": self.val_accuracy,
        }


@dataclass
class PredictionResult:
    """
    Container for prediction results.

    Attributes:
        predictions: Binary predictions (0=real, 1=fake).
        probabilities: Probability of fake class.
        is_fake: Boolean flags for fake detection.
    """
    predictions: np.ndarray
    probabilities: np.ndarray
    is_fake: np.ndarray = field(init=False)

    def __post_init__(self):
        self.is_fake = self.predictions == 1


# =============================================================================
# Neural Network Model
# =============================================================================

class MLPClassifier(nn.Module):
    """
    Multi-layer Perceptron classifier.

    A simple feedforward network for binary classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        """
        Initialize MLP classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
            activation: Activation function.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


# =============================================================================
# Deepfake Classifier
# =============================================================================

class DeepfakeClassifier:
    """
    Binary classifier for deepfake detection.

    This class provides training and inference capabilities for
    classifying videos as real or fake.

    Attributes:
        input_dim: Input feature dimension.
        model_type: Classification model type.
        threshold: Decision threshold for fake classification.

    Example:
        >>> classifier = DeepfakeClassifier(input_dim=4096, model_type="mlp")
        >>> classifier.fit(X_train, y_train, X_val, y_val)
        >>> probs = classifier.predict_proba(X_test)
        >>> preds = classifier.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        model_type: str = "mlp",
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        threshold: float = 0.5,
        device: Optional[str] = None,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Initialize classifier.

        Args:
            input_dim: Input feature dimension.
            model_type: Model type ("logistic", "mlp", "svm", "xgboost").
            hidden_dims: Hidden layer dimensions (MLP only).
            dropout: Dropout rate (MLP only).
            threshold: Decision threshold.
            device: Compute device (MLP only).
            class_weights: Class weights for imbalanced data.

        Raises:
            ClassificationError: If initialization fails.
        """
        self.input_dim = input_dim
        self.model_type = model_type.lower()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.threshold = threshold
        self.class_weights = class_weights

        # Determine device for neural models
        if device is None:
            if TORCH_AVAILABLE:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = None
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else None

        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False

        # Initialize model
        self._init_model()

        logger.info(
            f"DeepfakeClassifier initialized: model_type={model_type}, "
            f"input_dim={input_dim}"
        )

    def _init_model(self) -> None:
        """Initialize the classification model."""
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced" if self.class_weights is None else self.class_weights,
            )

        elif self.model_type == "svm":
            self.model = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced" if self.class_weights is None else self.class_weights,
            )

        elif self.model_type == "mlp":
            if not TORCH_AVAILABLE:
                raise ClassificationError(
                    "PyTorch not available for MLP classifier",
                    code="TORCH_NOT_AVAILABLE",
                )
            self.model = MLPClassifier(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
            self.model = self.model.to(self.device)

        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ClassificationError(
                    "XGBoost not available",
                    code="XGBOOST_NOT_AVAILABLE",
                )
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
            )

        else:
            raise ClassificationError(
                f"Unsupported model type: {self.model_type}",
                code="UNSUPPORTED_MODEL",
            )

    @log_execution_time()
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> TrainingMetrics:
        """
        Train the classifier.

        Args:
            X: Training features (N, input_dim).
            y: Training labels (N,). 0=real, 1=fake.
            X_val: Validation features.
            y_val: Validation labels.
            epochs: Training epochs (MLP only).
            batch_size: Batch size (MLP only).
            learning_rate: Learning rate (MLP only).

        Returns:
            TrainingMetrics with evaluation results.
        """
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None

        metrics = TrainingMetrics()

        if self.model_type == "mlp":
            metrics = self._train_mlp(
                X_scaled, y, X_val_scaled, y_val,
                epochs, batch_size, learning_rate
            )
        else:
            # Sklearn models
            self.model.fit(X_scaled, y)

            # Evaluate on training set
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)[:, 1]

            metrics.accuracy = accuracy_score(y, y_pred)
            metrics.precision = precision_score(y, y_pred, zero_division=0)
            metrics.recall = recall_score(y, y_pred, zero_division=0)
            metrics.f1 = f1_score(y, y_pred, zero_division=0)
            metrics.auc = roc_auc_score(y, y_proba)
            metrics.confusion_matrix = confusion_matrix(y, y_pred)

            # Validation metrics
            if X_val_scaled is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val_scaled)
                metrics.val_accuracy = accuracy_score(y_val, y_val_pred)

        # Find optimal threshold
        if X_val_scaled is not None and y_val is not None:
            metrics.threshold = self._find_optimal_threshold(X_val_scaled, y_val)
            self.threshold = metrics.threshold

        self._is_fitted = True
        logger.info(f"Training complete: {metrics.to_dict()}")

        return metrics

    def _train_mlp(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> TrainingMetrics:
        """Train MLP model."""
        # Create data loaders
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float().unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
            y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1).to(self.device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Training loop
        self.model.train()
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                self.model.train()

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}"
                )

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Compute metrics
        self.model.eval()
        metrics = TrainingMetrics()

        with torch.no_grad():
            train_outputs = self.model(X_tensor.to(self.device))
            train_probs = torch.sigmoid(train_outputs).cpu().numpy().flatten()
            train_preds = (train_probs > 0.5).astype(int)

            metrics.accuracy = accuracy_score(y, train_preds)
            metrics.precision = precision_score(y, train_preds, zero_division=0)
            metrics.recall = recall_score(y, train_preds, zero_division=0)
            metrics.f1 = f1_score(y, train_preds, zero_division=0)
            metrics.auc = roc_auc_score(y, train_probs)

            if X_val is not None:
                val_probs = torch.sigmoid(self.model(X_val_tensor)).cpu().numpy().flatten()
                val_preds = (val_probs > 0.5).astype(int)
                metrics.val_accuracy = accuracy_score(y_val, val_preds)
                metrics.val_loss = best_val_loss

        return metrics

    def _find_optimal_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """Find optimal decision threshold."""
        probs = self._predict_proba_internal(X)

        best_threshold = 0.5
        best_score = 0

        for threshold in np.linspace(0.1, 0.9, 81):
            preds = (probs > threshold).astype(int)

            if metric == "f1":
                score = f1_score(y, preds, zero_division=0)
            elif metric == "accuracy":
                score = accuracy_score(y, preds)
            else:
                score = f1_score(y, preds, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        """Internal probability prediction."""
        if self.model_type == "mlp":
            self.model.eval()
            X_tensor = torch.from_numpy(X).float().to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            return probs
        else:
            return self.model.predict_proba(X)[:, 1]

    @log_execution_time()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of fake class.

        Args:
            X: Feature vectors (N, input_dim).

        Returns:
            Probabilities (N,). Higher = more likely fake.
        """
        if not self._is_fitted:
            raise PredictionError(
                "Classifier not fitted. Call fit() first.",
                code="NOT_FITTED",
            )

        X_scaled = self.scaler.transform(X)
        return self._predict_proba_internal(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Feature vectors (N, input_dim).

        Returns:
            Binary predictions (N,). 0=real, 1=fake.
        """
        probs = self.predict_proba(X)
        return (probs > self.threshold).astype(int)

    def predict_with_result(self, X: np.ndarray) -> PredictionResult:
        """
        Predict with detailed results.

        Args:
            X: Feature vectors (N, input_dim).

        Returns:
            PredictionResult with predictions and probabilities.
        """
        probs = self.predict_proba(X)
        preds = (probs > self.threshold).astype(int)
        return PredictionResult(predictions=preds, probabilities=probs)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save classifier to disk.

        Args:
            path: Directory path for saving.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "input_dim": self.input_dim,
            "model_type": self.model_type,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "threshold": self.threshold,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        # Save scaler
        joblib.dump(self.scaler, path / "scaler.pkl")

        # Save model
        if self.model_type == "mlp":
            torch.save(self.model.state_dict(), path / "model.pth")
        else:
            joblib.dump(self.model, path / "model.pkl")

        logger.info(f"Saved classifier to: {path}")

    def load(self, path: Union[str, Path]) -> None:
        """
        Load classifier from disk.

        Args:
            path: Directory path to load from.
        """
        path = Path(path)

        # Load configuration
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        self.input_dim = config["input_dim"]
        self.model_type = config["model_type"]
        self.hidden_dims = config["hidden_dims"]
        self.dropout = config["dropout"]
        self.threshold = config["threshold"]

        # Reinitialize model
        self._init_model()

        # Load scaler
        self.scaler = joblib.load(path / "scaler.pkl")

        # Load model
        if self.model_type == "mlp":
            state_dict = torch.load(path / "model.pth", map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            self.model = joblib.load(path / "model.pkl")

        self._is_fitted = True
        logger.info(f"Loaded classifier from: {path}")


# =============================================================================
# Service Factory
# =============================================================================

def create_classifier_service(config: Optional[Dict[str, Any]] = None) -> DeepfakeClassifier:
    """
    Factory function to create a DeepfakeClassifier from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured DeepfakeClassifier instance.
    """
    if config is None:
        config = {}

    return DeepfakeClassifier(
        input_dim=config.get("input_dim", 4096),
        model_type=config.get("model_type", "mlp"),
        hidden_dims=config.get("hidden_dims", [256, 128]),
        dropout=config.get("dropout", 0.3),
        threshold=config.get("threshold", 0.5),
        device=config.get("device"),
    )
