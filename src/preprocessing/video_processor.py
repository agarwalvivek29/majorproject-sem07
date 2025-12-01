"""
Video Processing Module
========================

This module provides video frame extraction and processing capabilities
for the deepfake detection pipeline.

Features:
    - Efficient frame extraction using OpenCV
    - Configurable frame sampling rates
    - Video metadata extraction
    - Support for multiple video formats
    - Memory-efficient batch processing

Microservice API:
    POST /process
        Request: {"video_path": str, "sample_rate": int}
        Response: {"frames_path": str, "metadata": dict}

Example Usage:
    >>> from src.preprocessing.video_processor import VideoProcessor
    >>> processor = VideoProcessor(sample_rate=2, max_frames=300)
    >>> frames, fps = processor.extract_frames("input.mp4")
    >>> print(f"Extracted {len(frames)} frames at {fps} FPS")
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import VideoProcessingError

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VideoMetadata:
    """
    Container for video metadata.

    Attributes:
        path: Path to the video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        frame_count: Total number of frames.
        duration: Video duration in seconds.
        codec: Video codec identifier.
        file_size: File size in bytes.
    """
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    file_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "codec": self.codec,
            "file_size": self.file_size,
        }


@dataclass
class FrameData:
    """
    Container for extracted frame data.

    Attributes:
        frame: The frame image as BGR numpy array.
        index: Original frame index in the video.
        timestamp: Timestamp in seconds.
    """
    frame: np.ndarray
    index: int
    timestamp: float


@dataclass
class ProcessingResult:
    """
    Result of video processing operation.

    Attributes:
        frames: List of extracted frames.
        metadata: Video metadata.
        frame_indices: Original indices of extracted frames.
        success: Whether processing was successful.
        error: Error message if processing failed.
    """
    frames: List[np.ndarray] = field(default_factory=list)
    metadata: Optional[VideoMetadata] = None
    frame_indices: List[int] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Video Processor
# =============================================================================

class VideoProcessor:
    """
    Video frame extraction and processing service.

    This class provides efficient video processing capabilities including
    frame extraction, metadata parsing, and format conversion.

    Attributes:
        sample_rate: Extract every n-th frame.
        max_frames: Maximum frames to extract.
        target_fps: Target FPS for normalization (optional).
        supported_formats: List of supported video formats.

    Example:
        >>> processor = VideoProcessor(sample_rate=2, max_frames=300)
        >>> frames, fps = processor.extract_frames("video.mp4")
        >>> for frame in processor.stream_frames("video.mp4"):
        ...     process(frame)
    """

    # Supported video formats
    SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    def __init__(
        self,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize the video processor.

        Args:
            sample_rate: Extract every n-th frame (1 = all frames).
            max_frames: Maximum number of frames to extract.
            target_fps: Target FPS for frame rate normalization.
            resize: Optional (width, height) to resize frames.

        Raises:
            ValueError: If sample_rate < 1.
        """
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")

        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.resize = resize

        logger.info(
            f"VideoProcessor initialized: sample_rate={sample_rate}, "
            f"max_frames={max_frames}, target_fps={target_fps}"
        )

    def validate_video(self, video_path: Union[str, Path]) -> Path:
        """
        Validate that the video file exists and is a supported format.

        Args:
            video_path: Path to the video file.

        Returns:
            Validated Path object.

        Raises:
            VideoProcessingError: If validation fails.
        """
        path = Path(video_path)

        if not path.exists():
            raise VideoProcessingError(
                f"Video file not found: {path}",
                video_path=str(path),
                code="VIDEO_NOT_FOUND",
            )

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise VideoProcessingError(
                f"Unsupported video format: {path.suffix}",
                video_path=str(path),
                code="UNSUPPORTED_FORMAT",
                details={"supported_formats": list(self.SUPPORTED_FORMATS)},
            )

        return path

    @log_execution_time()
    def get_metadata(self, video_path: Union[str, Path]) -> VideoMetadata:
        """
        Extract metadata from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata object with video properties.

        Raises:
            VideoProcessingError: If metadata extraction fails.
        """
        path = self.validate_video(video_path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoProcessingError(
                f"Failed to open video: {path}",
                video_path=str(path),
                code="VIDEO_OPEN_FAILED",
            )

        try:
            # Extract properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode fourcc to string
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0.0

            # Get file size
            file_size = path.stat().st_size

            metadata = VideoMetadata(
                path=str(path),
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec,
                file_size=file_size,
            )

            logger.debug(f"Video metadata: {metadata}")
            return metadata

        finally:
            cap.release()

    @log_execution_time()
    def extract_frames(
        self,
        video_path: Union[str, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Extract frames from a video file.

        This method loads all requested frames into memory. For large videos,
        consider using stream_frames() instead.

        Args:
            video_path: Path to the video file.
            start_frame: Starting frame index.
            end_frame: Ending frame index (exclusive). None = end of video.

        Returns:
            Tuple of (frames list, fps).

        Raises:
            VideoProcessingError: If frame extraction fails.

        Example:
            >>> frames, fps = processor.extract_frames("video.mp4")
            >>> print(f"Extracted {len(frames)} frames")
        """
        path = self.validate_video(video_path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoProcessingError(
                f"Failed to open video: {path}",
                video_path=str(path),
            )

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if end_frame is None:
                end_frame = total_frames

            # Validate frame range
            start_frame = max(0, start_frame)
            end_frame = min(end_frame, total_frames)

            frames: List[np.ndarray] = []
            frame_idx = 0

            # Seek to start frame if needed
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if we should keep this frame (based on sample rate)
                if (frame_idx - start_frame) % self.sample_rate == 0:
                    # Resize if specified
                    if self.resize:
                        frame = cv2.resize(frame, self.resize)

                    frames.append(frame)

                    # Check max frames limit
                    if self.max_frames and len(frames) >= self.max_frames:
                        break

                frame_idx += 1

            logger.info(
                f"Extracted {len(frames)} frames from {path.name} "
                f"(total: {total_frames}, sampled 1/{self.sample_rate})"
            )

            return frames, fps

        except Exception as e:
            raise VideoProcessingError(
                f"Failed to extract frames: {e}",
                video_path=str(path),
                cause=e,
            )
        finally:
            cap.release()

    def stream_frames(
        self,
        video_path: Union[str, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Generator[FrameData, None, None]:
        """
        Stream frames from a video file.

        This generator yields frames one at a time, which is memory-efficient
        for processing large videos.

        Args:
            video_path: Path to the video file.
            start_frame: Starting frame index.
            end_frame: Ending frame index (exclusive).

        Yields:
            FrameData objects containing frame, index, and timestamp.

        Raises:
            VideoProcessingError: If streaming fails.

        Example:
            >>> for frame_data in processor.stream_frames("video.mp4"):
            ...     result = process(frame_data.frame)
            ...     print(f"Processed frame {frame_data.index}")
        """
        path = self.validate_video(video_path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoProcessingError(
                f"Failed to open video: {path}",
                video_path=str(path),
            )

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if end_frame is None:
                end_frame = total_frames

            start_frame = max(0, start_frame)
            end_frame = min(end_frame, total_frames)

            # Seek to start frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_idx = start_frame
            extracted_count = 0

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % self.sample_rate == 0:
                    if self.resize:
                        frame = cv2.resize(frame, self.resize)

                    timestamp = frame_idx / fps if fps > 0 else 0.0

                    yield FrameData(
                        frame=frame,
                        index=frame_idx,
                        timestamp=timestamp,
                    )

                    extracted_count += 1
                    if self.max_frames and extracted_count >= self.max_frames:
                        break

                frame_idx += 1

        finally:
            cap.release()

    def extract_frames_at_timestamps(
        self,
        video_path: Union[str, Path],
        timestamps: List[float],
    ) -> List[np.ndarray]:
        """
        Extract frames at specific timestamps.

        Args:
            video_path: Path to the video file.
            timestamps: List of timestamps in seconds.

        Returns:
            List of frames at the specified timestamps.

        Raises:
            VideoProcessingError: If extraction fails.
        """
        path = self.validate_video(video_path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoProcessingError(
                f"Failed to open video: {path}",
                video_path=str(path),
            )

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames: List[np.ndarray] = []

            for ts in sorted(timestamps):
                frame_idx = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if ret:
                    if self.resize:
                        frame = cv2.resize(frame, self.resize)
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to extract frame at timestamp {ts}")
                    frames.append(None)

            return frames

        finally:
            cap.release()

    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "frame",
        format: str = "jpg",
        quality: int = 95,
    ) -> List[str]:
        """
        Save extracted frames to disk.

        Args:
            frames: List of frames to save.
            output_dir: Directory to save frames.
            prefix: Filename prefix.
            format: Image format (jpg, png).
            quality: JPEG quality (1-100).

        Returns:
            List of saved file paths.

        Raises:
            VideoProcessingError: If saving fails.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[str] = []

        for i, frame in enumerate(frames):
            filename = f"{prefix}_{i:06d}.{format}"
            filepath = output_dir / filename

            try:
                if format.lower() == "jpg":
                    cv2.imwrite(
                        str(filepath),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality],
                    )
                else:
                    cv2.imwrite(str(filepath), frame)

                saved_paths.append(str(filepath))
            except Exception as e:
                raise VideoProcessingError(
                    f"Failed to save frame {i}: {e}",
                    frame_index=i,
                    cause=e,
                )

        logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths

    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_path: Union[str, Path],
        fps: float = 25.0,
        codec: str = "mp4v",
    ) -> str:
        """
        Create a video from a list of frames.

        Args:
            frames: List of frames (BGR numpy arrays).
            output_path: Output video path.
            fps: Output frame rate.
            codec: Video codec.

        Returns:
            Path to the created video.

        Raises:
            VideoProcessingError: If video creation fails.
        """
        if not frames:
            raise VideoProcessingError(
                "Cannot create video from empty frame list",
                code="EMPTY_FRAMES",
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise VideoProcessingError(
                f"Failed to create video writer for {output_path}",
                video_path=str(output_path),
            )

        try:
            for frame in frames:
                writer.write(frame)
            logger.info(f"Created video: {output_path}")
            return str(output_path)
        finally:
            writer.release()

    def process(
        self,
        video_path: Union[str, Path],
        save_frames: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> ProcessingResult:
        """
        Full processing pipeline for a video.

        This method extracts frames and metadata, optionally saving
        frames to disk.

        Args:
            video_path: Path to the video file.
            save_frames: Whether to save frames to disk.
            output_dir: Directory for saved frames.

        Returns:
            ProcessingResult with frames and metadata.

        Example:
            >>> result = processor.process("video.mp4", save_frames=True)
            >>> if result.success:
            ...     print(f"Processed {len(result.frames)} frames")
        """
        result = ProcessingResult()

        try:
            # Get metadata
            result.metadata = self.get_metadata(video_path)

            # Extract frames
            frames, _ = self.extract_frames(video_path)
            result.frames = frames

            # Generate frame indices
            total_extracted = len(frames)
            result.frame_indices = list(range(0, total_extracted * self.sample_rate, self.sample_rate))

            # Optionally save frames
            if save_frames:
                if output_dir is None:
                    output_dir = Path(tempfile.mkdtemp(prefix="frames_"))
                self.save_frames(frames, output_dir)

            result.success = True
            logger.info(f"Successfully processed video: {video_path}")

        except VideoProcessingError as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Video processing failed: {e}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.exception(f"Unexpected error processing video: {e}")

        return result


# =============================================================================
# Service Factory
# =============================================================================

def create_video_processor_service(config: Optional[Dict[str, Any]] = None) -> VideoProcessor:
    """
    Factory function to create a VideoProcessor from configuration.

    Args:
        config: Configuration dictionary. If None, uses defaults.

    Returns:
        Configured VideoProcessor instance.
    """
    if config is None:
        config = {}

    return VideoProcessor(
        sample_rate=config.get("sample_rate", 1),
        max_frames=config.get("max_frames"),
        target_fps=config.get("target_fps"),
        resize=config.get("resize"),
    )
