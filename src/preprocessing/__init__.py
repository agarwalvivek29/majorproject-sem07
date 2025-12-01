"""
Preprocessing Module
=====================

This module provides preprocessing services for video, face, and audio data.
Each preprocessor can run as an independent microservice or be used directly.

Components:
    - VideoProcessor: Frame extraction and video decoding
    - FaceDetector: Face detection, alignment, and cropping
    - AudioProcessor: Audio extraction and MFCC computation

Example Usage:
    >>> from src.preprocessing import VideoProcessor, FaceDetector, AudioProcessor
    >>>
    >>> video_proc = VideoProcessor(sample_rate=2)
    >>> frames, fps = video_proc.extract_frames("video.mp4")
    >>>
    >>> face_det = FaceDetector(backend="mediapipe")
    >>> face_crops = [face_det.process_frame(f) for f in frames]
    >>>
    >>> audio_proc = AudioProcessor(sample_rate=16000)
    >>> mfccs = audio_proc.extract_features("video.mp4", len(frames))
"""

from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.audio_processor import AudioProcessor

__all__ = [
    "VideoProcessor",
    "FaceDetector",
    "AudioProcessor",
]
