"""
Audio Processing Module
========================

This module provides audio extraction and feature computation capabilities
for the deepfake detection pipeline.

Features:
    - Audio extraction from video files using FFmpeg
    - MFCC (Mel-frequency cepstral coefficients) computation
    - Mel-spectrogram generation
    - Frame-synchronized audio segmentation
    - Audio normalization and preprocessing

Microservice API:
    POST /extract
        Request: {"video_path": str}
        Response: {"audio_path": str, "sample_rate": int}
    POST /features
        Request: {"audio_path": str, "n_frames": int}
        Response: {"mfccs": [[...]], "energy": [...]}

Example Usage:
    >>> from src.preprocessing.audio_processor import AudioProcessor
    >>> processor = AudioProcessor(sample_rate=16000, n_mfcc=13)
    >>> mfccs = processor.extract_features("video.mp4", n_frames=100)
    >>> print(f"MFCC shape: {mfccs.shape}")  # (100, 13)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import AudioProcessingError

# Module logger
logger = get_logger(__name__)

# Try to import librosa
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available, audio processing will be limited")

# Try to import scipy for basic audio processing
try:
    from scipy.io import wavfile
    from scipy.fft import fft, fftfreq
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AudioMetadata:
    """
    Container for audio metadata.

    Attributes:
        path: Path to the audio file.
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        channels: Number of audio channels.
        samples: Total number of samples.
    """
    path: str
    sample_rate: int
    duration: float
    channels: int
    samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "channels": self.channels,
            "samples": self.samples,
        }


@dataclass
class AudioFeatures:
    """
    Container for extracted audio features.

    Attributes:
        mfccs: MFCC features array (n_frames, n_mfcc).
        mel_spectrogram: Mel spectrogram (n_mels, time).
        energy: RMS energy per frame.
        zero_crossing_rate: Zero crossing rate per frame.
        spectral_centroid: Spectral centroid per frame.
    """
    mfccs: Optional[np.ndarray] = None
    mel_spectrogram: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    zero_crossing_rate: Optional[np.ndarray] = None
    spectral_centroid: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (arrays as lists)."""
        return {
            "mfccs": self.mfccs.tolist() if self.mfccs is not None else None,
            "mel_spectrogram": self.mel_spectrogram.tolist() if self.mel_spectrogram is not None else None,
            "energy": self.energy.tolist() if self.energy is not None else None,
        }


@dataclass
class ProcessingResult:
    """
    Result of audio processing operation.

    Attributes:
        success: Whether processing was successful.
        audio_path: Path to extracted audio file.
        metadata: Audio metadata.
        features: Extracted audio features.
        error: Error message if processing failed.
    """
    success: bool = True
    audio_path: Optional[str] = None
    metadata: Optional[AudioMetadata] = None
    features: Optional[AudioFeatures] = None
    error: Optional[str] = None


# =============================================================================
# Audio Processor
# =============================================================================

class AudioProcessor:
    """
    Audio extraction and feature computation service.

    This class provides comprehensive audio processing capabilities
    for the deepfake detection pipeline, including audio extraction
    from videos and acoustic feature computation.

    Attributes:
        sample_rate: Target sample rate in Hz.
        n_mfcc: Number of MFCC coefficients.
        n_mels: Number of Mel filterbank bands.
        window_ms: FFT window size in milliseconds.
        hop_ms: Hop length in milliseconds.
        pre_emphasis: Pre-emphasis coefficient.

    Example:
        >>> processor = AudioProcessor(sample_rate=16000, n_mfcc=13)
        >>> mfccs = processor.extract_features("video.mp4", n_frames=100)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 40,
        window_ms: float = 25.0,
        hop_ms: float = 10.0,
        pre_emphasis: float = 0.97,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the audio processor.

        Args:
            sample_rate: Target sample rate in Hz.
            n_mfcc: Number of MFCC coefficients.
            n_mels: Number of Mel filterbank bands.
            window_ms: FFT window size in milliseconds.
            hop_ms: Hop length in milliseconds.
            pre_emphasis: Pre-emphasis filter coefficient.
            normalize: Whether to normalize audio.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.pre_emphasis = pre_emphasis
        self.normalize = normalize

        # Calculate window and hop length in samples
        self.n_fft = int(self.sample_rate * self.window_ms / 1000)
        self.hop_length = int(self.sample_rate * self.hop_ms / 1000)

        # Verify FFmpeg is available
        self._check_ffmpeg()

        logger.info(
            f"AudioProcessor initialized: sample_rate={sample_rate}, "
            f"n_mfcc={n_mfcc}, n_mels={n_mels}"
        )

    def _check_ffmpeg(self) -> None:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning("FFmpeg not available, audio extraction may fail")
        except FileNotFoundError:
            logger.warning("FFmpeg not found in PATH")

    @log_execution_time()
    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Extract audio track from video file.

        Uses FFmpeg to extract audio and convert to mono WAV at
        the configured sample rate.

        Args:
            video_path: Path to the video file.
            output_path: Optional output path. If None, creates temp file.

        Returns:
            Path to the extracted audio file.

        Raises:
            AudioProcessingError: If extraction fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise AudioProcessingError(
                f"Video file not found: {video_path}",
                audio_path=str(video_path),
            )

        # Create output path if not provided
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
        output_path = Path(output_path)

        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                      # No video
            "-acodec", "pcm_s16le",     # PCM 16-bit
            "-ar", str(self.sample_rate),  # Sample rate
            "-ac", "1",                 # Mono
            "-y",                       # Overwrite
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise AudioProcessingError(
                    f"FFmpeg failed: {result.stderr}",
                    audio_path=str(video_path),
                    code="FFMPEG_FAILED",
                )

            logger.debug(f"Extracted audio to: {output_path}")
            return str(output_path)

        except subprocess.TimeoutExpired:
            raise AudioProcessingError(
                "FFmpeg timed out",
                audio_path=str(video_path),
                code="FFMPEG_TIMEOUT",
            )
        except Exception as e:
            raise AudioProcessingError(
                f"Audio extraction failed: {e}",
                audio_path=str(video_path),
                cause=e,
            )

    def load_audio(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file as numpy array.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple of (waveform, sample_rate).

        Raises:
            AudioProcessingError: If loading fails.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioProcessingError(
                f"Audio file not found: {audio_path}",
                audio_path=str(audio_path),
            )

        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for loading
                y, sr = librosa.load(
                    str(audio_path),
                    sr=self.sample_rate,
                    mono=True,
                )
            elif SCIPY_AVAILABLE:
                # Fallback to scipy
                sr, y = wavfile.read(str(audio_path))
                y = y.astype(np.float32) / 32768.0  # Normalize
                if len(y.shape) > 1:
                    y = y.mean(axis=1)  # Convert to mono
            else:
                raise AudioProcessingError(
                    "No audio loading library available",
                    code="NO_AUDIO_LIBRARY",
                )

            # Apply pre-emphasis
            if self.pre_emphasis > 0:
                y = np.append(y[0], y[1:] - self.pre_emphasis * y[:-1])

            # Normalize
            if self.normalize:
                max_val = np.abs(y).max()
                if max_val > 0:
                    y = y / max_val

            return y, sr

        except Exception as e:
            raise AudioProcessingError(
                f"Failed to load audio: {e}",
                audio_path=str(audio_path),
                cause=e,
            )

    def get_metadata(self, audio_path: Union[str, Path]) -> AudioMetadata:
        """
        Get metadata for an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            AudioMetadata object.
        """
        y, sr = self.load_audio(audio_path)

        return AudioMetadata(
            path=str(audio_path),
            sample_rate=sr,
            duration=len(y) / sr,
            channels=1,  # Always mono after loading
            samples=len(y),
        )

    def segment_by_frames(
        self,
        waveform: np.ndarray,
        n_frames: int,
        fps: float = 25.0,
    ) -> List[np.ndarray]:
        """
        Divide waveform into segments aligned with video frames.

        Each segment corresponds to one video frame, covering
        (1/fps) seconds of audio.

        Args:
            waveform: Audio waveform array.
            n_frames: Number of video frames.
            fps: Video frame rate.

        Returns:
            List of audio segments.

        Example:
            >>> segments = processor.segment_by_frames(waveform, 100, fps=25)
            >>> len(segments)  # 100 segments
        """
        samples_per_frame = int(self.sample_rate / fps)
        segments = []

        for i in range(n_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame

            if end <= len(waveform):
                segment = waveform[start:end]
            else:
                # Pad with zeros if audio is shorter than video
                segment = np.zeros(samples_per_frame)
                if start < len(waveform):
                    available = waveform[start:]
                    segment[:len(available)] = available

            segments.append(segment)

        return segments

    def compute_mfcc(
        self,
        waveform: np.ndarray,
        include_deltas: bool = True,
    ) -> np.ndarray:
        """
        Compute MFCC features for audio.

        Args:
            waveform: Audio waveform array.
            include_deltas: Include delta and delta-delta features.

        Returns:
            MFCC array of shape (n_frames, n_mfcc * (1 + 2*include_deltas)).
        """
        if not LIBROSA_AVAILABLE:
            # Simple fallback MFCC computation
            return self._compute_mfcc_simple(waveform)

        try:
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )

            if include_deltas:
                # Compute deltas
                delta = librosa.feature.delta(mfccs)
                delta2 = librosa.feature.delta(mfccs, order=2)
                mfccs = np.vstack([mfccs, delta, delta2])

            # Transpose to (time, features)
            return mfccs.T

        except Exception as e:
            logger.warning(f"MFCC computation failed: {e}")
            return np.zeros((1, self.n_mfcc))

    def _compute_mfcc_simple(self, waveform: np.ndarray) -> np.ndarray:
        """Simple MFCC computation without librosa."""
        # This is a simplified version; use librosa for proper MFCCs
        n_frames = len(waveform) // self.hop_length
        mfccs = np.zeros((n_frames, self.n_mfcc))

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            if end > len(waveform):
                break

            frame = waveform[start:end]
            # Apply Hamming window
            frame = frame * np.hamming(len(frame))
            # Compute FFT
            spectrum = np.abs(fft(frame)[:self.n_fft // 2])
            # Simple energy-based features
            if len(spectrum) > self.n_mfcc:
                step = len(spectrum) // self.n_mfcc
                for j in range(self.n_mfcc):
                    mfccs[i, j] = np.log(np.mean(spectrum[j*step:(j+1)*step]) + 1e-10)

        return mfccs

    def compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute log-Mel spectrogram.

        Args:
            waveform: Audio waveform array.

        Returns:
            Log-Mel spectrogram of shape (n_mels, time).
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available for Mel spectrogram")
            return np.zeros((self.n_mels, 1))

        try:
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec

        except Exception as e:
            logger.warning(f"Mel spectrogram computation failed: {e}")
            return np.zeros((self.n_mels, 1))

    def compute_rms_energy(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute RMS energy per frame.

        Args:
            waveform: Audio waveform array.

        Returns:
            RMS energy array.
        """
        if LIBROSA_AVAILABLE:
            rms = librosa.feature.rms(
                y=waveform,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
            )
            return rms.flatten()
        else:
            # Simple RMS computation
            n_frames = len(waveform) // self.hop_length
            rms = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * self.hop_length
                end = start + self.n_fft
                if end <= len(waveform):
                    frame = waveform[start:end]
                    rms[i] = np.sqrt(np.mean(frame ** 2))
            return rms

    def compute_segment_features(
        self,
        segment: np.ndarray,
    ) -> np.ndarray:
        """
        Compute features for a single audio segment.

        Args:
            segment: Audio segment array.

        Returns:
            Feature vector (mean MFCC across segment).
        """
        if len(segment) < self.n_fft:
            # Pad short segments
            segment = np.pad(segment, (0, self.n_fft - len(segment)))

        mfccs = self.compute_mfcc(segment, include_deltas=False)
        # Return mean across time
        return np.mean(mfccs, axis=0)

    @log_execution_time()
    def extract_features(
        self,
        video_path: Union[str, Path],
        n_frames: int,
        fps: float = 25.0,
    ) -> np.ndarray:
        """
        Extract MFCC features aligned with video frames.

        This is the main method for feature extraction, producing
        one MFCC vector per video frame.

        Args:
            video_path: Path to the video file.
            n_frames: Number of video frames.
            fps: Video frame rate.

        Returns:
            MFCC array of shape (n_frames, n_mfcc).

        Example:
            >>> mfccs = processor.extract_features("video.mp4", 100, fps=25)
            >>> print(mfccs.shape)  # (100, 13)
        """
        # Extract audio from video
        audio_path = self.extract_audio(video_path)

        try:
            # Load audio
            waveform, _ = self.load_audio(audio_path)

            # Segment by frames
            segments = self.segment_by_frames(waveform, n_frames, fps)

            # Compute features for each segment
            features = np.zeros((n_frames, self.n_mfcc))
            for i, segment in enumerate(segments):
                features[i] = self.compute_segment_features(segment)

            logger.debug(f"Extracted features shape: {features.shape}")
            return features

        finally:
            # Clean up temp file
            if audio_path and Path(audio_path).exists():
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def extract_all_features(
        self,
        video_path: Union[str, Path],
    ) -> AudioFeatures:
        """
        Extract all audio features from a video.

        Args:
            video_path: Path to the video file.

        Returns:
            AudioFeatures object with all computed features.
        """
        # Extract audio
        audio_path = self.extract_audio(video_path)

        try:
            # Load audio
            waveform, _ = self.load_audio(audio_path)

            # Compute all features
            features = AudioFeatures(
                mfccs=self.compute_mfcc(waveform, include_deltas=True),
                mel_spectrogram=self.compute_mel_spectrogram(waveform),
                energy=self.compute_rms_energy(waveform),
            )

            return features

        finally:
            # Clean up
            if audio_path and Path(audio_path).exists():
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def process(
        self,
        video_path: Union[str, Path],
        n_frames: Optional[int] = None,
        fps: float = 25.0,
    ) -> ProcessingResult:
        """
        Full audio processing pipeline.

        Args:
            video_path: Path to the video file.
            n_frames: Number of video frames (for frame-aligned features).
            fps: Video frame rate.

        Returns:
            ProcessingResult with extracted features.
        """
        result = ProcessingResult()

        try:
            # Extract audio
            audio_path = self.extract_audio(video_path)
            result.audio_path = audio_path

            # Get metadata
            result.metadata = self.get_metadata(audio_path)

            # Extract features
            if n_frames:
                mfccs = self.extract_features(video_path, n_frames, fps)
                result.features = AudioFeatures(mfccs=mfccs)
            else:
                result.features = self.extract_all_features(video_path)

            result.success = True
            logger.info(f"Successfully processed audio from: {video_path}")

        except AudioProcessingError as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Audio processing failed: {e}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.exception(f"Unexpected error in audio processing: {e}")

        return result


# =============================================================================
# Service Factory
# =============================================================================

def create_audio_processor_service(config: Optional[Dict[str, Any]] = None) -> AudioProcessor:
    """
    Factory function to create an AudioProcessor from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured AudioProcessor instance.
    """
    if config is None:
        config = {}

    return AudioProcessor(
        sample_rate=config.get("sample_rate", 16000),
        n_mfcc=config.get("n_mfcc", 13),
        n_mels=config.get("n_mels", 40),
        window_ms=config.get("window_ms", 25.0),
        hop_ms=config.get("hop_ms", 10.0),
        pre_emphasis=config.get("pre_emphasis", 0.97),
        normalize=config.get("normalize", True),
    )
