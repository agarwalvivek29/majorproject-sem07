#!/usr/bin/env python3
"""
Training Script
================

Command-line interface for training deepfake detection models.

Usage:
    python scripts/train.py --dataset data/raw --output models/
    python scripts/train.py --config config/training.yaml
    python scripts/train.py --dataset data/raw --epochs 100 --batch-size 64

Examples:
    # Basic training
    python scripts/train.py --dataset data/raw

    # Training with custom configuration
    python scripts/train.py --config config/training.yaml --output models/v2/

    # Training with specific parameters
    python scripts/train.py --dataset data/raw --epochs 100 --lr 0.0001 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging
from src.utils.config import load_config
from src.pipeline.trainer import TrainingPipeline, TrainingResult

# Setup logging
setup_logging(level="INFO", log_file="logs/training.log")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train deepfake detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to dataset directory (with real/ and fake/ subdirectories)",
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (YAML)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/",
        help="Output directory for trained models (default: models/)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )

    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp",
        choices=["logistic", "mlp", "svm", "xgboost"],
        help="Classifier model type (default: mlp)",
    )

    parser.add_argument(
        "--visual-encoder",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet101", "mobilenet"],
        help="Visual encoder model (default: resnet50)",
    )

    parser.add_argument(
        "--temporal-model",
        type=str,
        default="lstm",
        choices=["lstm", "transformer", "conv1d"],
        help="Temporal model architecture (default: lstm)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Compute device (default: auto-detect)",
    )

    # RAD configuration
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for RAD (default: 5)",
    )

    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "ivfpq", "hnsw"],
        help="Vector index type (default: flat)",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without training",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Build configuration from arguments."""
    # Load base config if provided
    if args.config:
        config = load_config(args.config).to_dict()
    else:
        config = {}

    # Override with command line arguments
    if args.dataset:
        config.setdefault("dataset", {})
        config["dataset"]["dir"] = args.dataset

    config.setdefault("training", {})
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["learning_rate"] = args.lr
    config["training"]["train_ratio"] = args.train_ratio
    config["training"]["val_ratio"] = args.val_ratio

    config.setdefault("classifier", {})
    config["classifier"]["model_type"] = args.model_type

    config.setdefault("features", {})
    config["features"]["visual"] = {"model": args.visual_encoder}
    config["features"]["temporal"] = {"architecture": args.temporal_model}

    config.setdefault("retrieval", {})
    config["retrieval"]["k_neighbors"] = args.k_neighbors
    config["retrieval"]["index_type"] = args.index_type

    config["output_dir"] = args.output
    config["device"] = args.device
    config["seed"] = args.seed

    return config


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set log level
    if args.verbose:
        setup_logging(level="DEBUG")

    # Build configuration
    config = build_config(args)

    # Validate required arguments
    dataset_dir = config.get("dataset", {}).get("dir")
    if not dataset_dir:
        logger.error("Dataset directory is required. Use --dataset or --config")
        return 1

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return 1

    # Print configuration
    logger.info("Training Configuration:")
    logger.info(f"  Dataset: {dataset_dir}")
    logger.info(f"  Output: {config['output_dir']}")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch Size: {config['training']['batch_size']}")
    logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"  Model Type: {config['classifier']['model_type']}")
    logger.info(f"  Device: {config.get('device', 'auto')}")

    if args.dry_run:
        logger.info("Dry run mode - printing full configuration:")
        print(json.dumps(config, indent=2))
        return 0

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    config_path = output_dir / "training_args.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to: {config_path}")

    try:
        # Initialize pipeline
        logger.info("Initializing training pipeline...")
        pipeline = TrainingPipeline(
            output_dir=config["output_dir"],
            device=config.get("device"),
        )

        # Run training
        logger.info("Starting training...")
        start_time = datetime.now()

        result = pipeline.train(
            dataset_dir=dataset_dir,
            train_ratio=config["training"]["train_ratio"],
            val_ratio=config["training"]["val_ratio"],
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Print results
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Model saved to: {result.model_path}")
        logger.info("")
        logger.info("Test Metrics:")
        for metric, value in result.test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save results
        results_path = output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved results to: {results_path}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
