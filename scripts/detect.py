#!/usr/bin/env python3
"""
Detection Script
=================

Command-line interface for deepfake detection on videos.

Usage:
    python scripts/detect.py video.mp4
    python scripts/detect.py --batch videos/*.mp4 --output results.json
    python scripts/detect.py video.mp4 --model models/ --threshold 0.7

Examples:
    # Detect single video
    python scripts/detect.py suspicious_video.mp4

    # Detect with custom model
    python scripts/detect.py video.mp4 --model models/v2/

    # Batch detection
    python scripts/detect.py --batch data/test/*.mp4 --output results.json

    # Detection with detailed output
    python scripts/detect.py video.mp4 --detailed --json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging
from src.pipeline.detector import DeepfakeDetector, DetectionResult

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect deepfakes in video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        help="Path to video file to analyze",
    )

    parser.add_argument(
        "--batch", "-b",
        type=str,
        nargs="+",
        help="Batch mode: paths or glob patterns for multiple videos",
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/",
        help="Path to model directory (default: models/)",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Compute device (default: auto-detect)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Include detailed analysis in output",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (only show result)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # Analysis options
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum frames to analyze (default: 300)",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Frame sampling rate (default: 1)",
    )

    return parser.parse_args()


def expand_paths(patterns: List[str]) -> List[Path]:
    """Expand glob patterns to file paths."""
    paths = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            matches = glob.glob(pattern, recursive=True)
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))

    # Filter to video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    paths = [p for p in paths if p.suffix.lower() in video_extensions]

    return sorted(set(paths))


def format_result(result: DetectionResult, detailed: bool = False) -> str:
    """Format detection result for display."""
    status = "FAKE" if result.is_fake else "REAL"
    prob_bar = "=" * int(result.probability * 20) + " " * (20 - int(result.probability * 20))

    lines = [
        f"Result: {status}",
        f"Probability: [{prob_bar}] {result.probability:.1%}",
        f"Confidence: {result.confidence:.1%}",
        f"Correlation: {result.correlation_score:.3f}",
        f"Processing Time: {result.processing_time:.2f}s",
        f"Frames Analyzed: {result.frame_count}",
    ]

    if detailed and result.details:
        lines.append("")
        lines.append("Details:")
        for key, value in result.details.items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def print_summary(results: List[DetectionResult]) -> None:
    """Print summary for batch results."""
    total = len(results)
    fake_count = sum(1 for r in results if r.is_fake)
    real_count = total - fake_count

    avg_prob = sum(r.probability for r in results) / total if total > 0 else 0
    avg_time = sum(r.processing_time for r in results) / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total Videos: {total}")
    print(f"Fake: {fake_count} ({fake_count/total:.1%})")
    print(f"Real: {real_count} ({real_count/total:.1%})")
    print(f"Average Probability: {avg_prob:.1%}")
    print(f"Average Processing Time: {avg_time:.2f}s")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set log level
    if args.verbose:
        setup_logging(level="DEBUG")
    elif args.quiet:
        setup_logging(level="WARNING")

    # Determine input mode
    if args.batch:
        video_paths = expand_paths(args.batch)
        if not video_paths:
            logger.error("No video files found matching the pattern(s)")
            return 1
        batch_mode = True
    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {args.video}")
            return 1
        video_paths = [video_path]
        batch_mode = False
    else:
        logger.error("No input specified. Provide a video path or use --batch")
        return 1

    # Check model directory
    model_dir = Path(args.model)
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {args.model}")
        logger.warning("Detection will use correlation-based fallback")

    try:
        # Initialize detector
        if not args.quiet:
            logger.info("Initializing detector...")

        detector = DeepfakeDetector(
            model_dir=args.model,
            config_path=args.config,
            device=args.device,
            threshold=args.threshold,
        )

        # Process videos
        results = []
        for i, video_path in enumerate(video_paths):
            if not args.quiet:
                if batch_mode:
                    logger.info(f"Processing [{i+1}/{len(video_paths)}]: {video_path.name}")
                else:
                    logger.info(f"Processing: {video_path}")

            result = detector.detect(video_path, detailed=args.detailed)
            results.append(result)

            # Print individual result
            if not args.quiet:
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    print("\n" + "-" * 40)
                    print(f"File: {video_path.name}")
                    print("-" * 40)
                    print(format_result(result, args.detailed))

        # Print batch summary
        if batch_mode and not args.quiet:
            print_summary(results)

        # Save results to file
        if args.output:
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "threshold": args.threshold,
                "model_dir": str(args.model),
                "results": [r.to_dict() for r in results],
            }

            if batch_mode:
                output_data["summary"] = {
                    "total": len(results),
                    "fake": sum(1 for r in results if r.is_fake),
                    "real": sum(1 for r in results if not r.is_fake),
                }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            if not args.quiet:
                logger.info(f"Results saved to: {args.output}")

        # Return code based on detection
        if not batch_mode:
            # Single video: return 0 for real, 1 for fake
            return 1 if results[0].is_fake else 0
        else:
            # Batch: return number of fakes detected (capped at 255)
            fake_count = sum(1 for r in results if r.is_fake)
            return min(fake_count, 255)

    except KeyboardInterrupt:
        logger.warning("Detection interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
