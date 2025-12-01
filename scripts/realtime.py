#!/usr/bin/env python3
"""
Real-time Detection Script
===========================

Command-line interface for real-time deepfake detection on video streams.

Usage:
    python scripts/realtime.py rtmp://source/stream
    python scripts/realtime.py 0  # Webcam
    python scripts/realtime.py rtsp://camera/feed --webhook https://slack.com/webhook

Examples:
    # Process webcam
    python scripts/realtime.py 0

    # Process RTMP stream
    python scripts/realtime.py rtmp://source/stream

    # Process with alerting
    python scripts/realtime.py rtmp://source/stream --webhook https://slack.com/webhook

    # Process with custom configuration
    python scripts/realtime.py rtsp://camera/feed --threshold 0.7 --window-size 90
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging
from src.realtime import StreamProcessor, StreamConfig, AlertManager, Alert, AlertLevel

# Setup logging
setup_logging(level="INFO", log_file="logs/realtime.log")
logger = get_logger(__name__)

# Global processor for signal handling
_processor = None


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal, stopping...")
    if _processor:
        _processor.stop()
    sys.exit(0)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time deepfake detection on video streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument(
        "source",
        type=str,
        help="Video source: URL (rtmp://, rtsp://, http://) or camera index (0, 1)",
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/",
        help="Path to model directory (default: models/)",
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

    # Stream processing
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=300,
        help="Frame buffer size (default: 300)",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Analysis window size in frames (default: 60)",
    )

    parser.add_argument(
        "--window-stride",
        type=int,
        default=15,
        help="Analysis window stride (default: 15)",
    )

    parser.add_argument(
        "--fps-limit",
        type=float,
        default=30.0,
        help="Maximum FPS to process (default: 30)",
    )

    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Frames to skip between captures (default: 1)",
    )

    # Alerting
    parser.add_argument(
        "--webhook",
        type=str,
        default=None,
        help="Webhook URL for alerts (Slack, Discord, etc.)",
    )

    parser.add_argument(
        "--alert-file",
        type=str,
        default="alerts.jsonl",
        help="File to log alerts (default: alerts.jsonl)",
    )

    parser.add_argument(
        "--alert-interval",
        type=int,
        default=30,
        help="Minimum seconds between alerts for same source (default: 30)",
    )

    # Duration
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum duration in seconds (default: unlimited)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for detection events (JSON)",
    )

    parser.add_argument(
        "--stats-interval",
        type=int,
        default=60,
        help="Interval for printing stats in seconds (default: 60)",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


def create_callback(args: argparse.Namespace, events: list):
    """Create detection callback function."""
    def on_detection(event):
        events.append(event)

        if not args.quiet:
            status = "FAKE" if event.is_fake else "REAL"
            logger.warning(
                f"Detection: {status} "
                f"(prob={event.probability:.1%}, conf={event.confidence:.1%}) "
                f"at frame {event.frame_index}"
            )

    return on_detection


def main() -> int:
    """Main entry point."""
    global _processor

    args = parse_args()

    # Set log level
    if args.verbose:
        setup_logging(level="DEBUG", log_file="logs/realtime.log")
    elif args.quiet:
        setup_logging(level="WARNING", log_file="logs/realtime.log")

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse source
    source = args.source
    if source.isdigit():
        source = int(source)
        logger.info(f"Using camera index: {source}")
    else:
        logger.info(f"Using stream URL: {source}")

    # Create stream configuration
    stream_config = StreamConfig(
        buffer_size=args.buffer_size,
        window_size=args.window_size,
        window_stride=args.window_stride,
        detection_threshold=args.threshold,
        fps_limit=args.fps_limit,
        skip_frames=args.skip_frames,
    )

    # Create alerter
    alerter = AlertManager(
        webhook_url=args.webhook,
        alert_file=args.alert_file,
    )
    alerter.config.min_interval = args.alert_interval

    # Print configuration
    if not args.quiet:
        logger.info("Configuration:")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Threshold: {args.threshold}")
        logger.info(f"  Window Size: {args.window_size}")
        logger.info(f"  Buffer Size: {args.buffer_size}")
        if args.webhook:
            logger.info(f"  Webhook: {args.webhook[:50]}...")
        if args.duration:
            logger.info(f"  Duration: {args.duration}s")

    try:
        # Initialize processor
        logger.info("Initializing stream processor...")
        _processor = StreamProcessor(
            model_dir=args.model,
            config=stream_config,
            device=args.device,
            alerter=alerter,
        )

        # Track events
        events = []
        callback = create_callback(args, events)

        # Start processing
        logger.info("Starting real-time detection...")
        logger.info("Press Ctrl+C to stop")
        print()

        start_time = datetime.now()
        stats = _processor.process_stream(
            source=source,
            callback=callback,
            duration=args.duration,
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Print final stats
        if not args.quiet:
            print("\n" + "=" * 60)
            print("SESSION COMPLETE")
            print("=" * 60)
            print(f"Duration: {duration:.1f}s")
            print(f"Frames Processed: {stats.total_frames}")
            print(f"Windows Analyzed: {stats.analyzed_windows}")
            print(f"Detections: {stats.detections}")
            print(f"Average FPS: {stats.avg_fps:.1f}")
            print(f"Average Latency: {stats.avg_latency*1000:.1f}ms")

        # Save events to file
        if args.output:
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "source": str(source),
                "duration": duration,
                "stats": stats.to_dict(),
                "events": [e.to_dict() for e in events],
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            if not args.quiet:
                logger.info(f"Events saved to: {args.output}")

        # Return detection count
        return min(stats.detections, 255)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
        return 0

    except Exception as e:
        logger.exception(f"Real-time detection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
