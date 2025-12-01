"""
Real-time Processing Module
============================

This module provides real-time streaming capabilities for deepfake detection.

Components:
    - StreamProcessor: Process video streams in real-time
    - AlertManager: Handle detection alerts and notifications
    - BufferManager: Manage frame buffering and batching

Example Usage:
    >>> from src.realtime import StreamProcessor, AlertManager
    >>> alerter = AlertManager(webhook_url="https://slack.com/webhook")
    >>> processor = StreamProcessor(model_dir="models/", alerter=alerter)
    >>> processor.process_stream("rtmp://source/stream")
"""

from src.realtime.stream_processor import StreamProcessor, StreamConfig
from src.realtime.alerter import AlertManager, Alert, AlertLevel

__all__ = [
    "StreamProcessor",
    "StreamConfig",
    "AlertManager",
    "Alert",
    "AlertLevel",
]
