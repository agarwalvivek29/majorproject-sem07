"""
Lip-Audio Correlation Analyzer Module
======================================

This module provides audio-visual synchronization analysis for
lip-sync deepfake detection.

Features:
    - Mouth opening signal extraction from landmarks
    - Audio energy signal computation
    - Pearson correlation analysis
    - Cross-correlation with lag detection
    - Dynamic Time Warping (DTW) distance
    - Spectral coherence analysis

Microservice API:
    POST /analyze
        Request: {"mouth_signal": [...], "audio_signal": [...]}
        Response: {"pearson": float, "cross_corr": float, "lag": int}

Example Usage:
    >>> from src.features.correlation_analyzer import CorrelationAnalyzer
    >>> analyzer = CorrelationAnalyzer(fps=25.0)
    >>> metrics = analyzer.compute_all_metrics(mouth_signal, audio_signal)
    >>> print(f"Pearson correlation: {metrics['pearson']:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.stats import pearsonr

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import DeepfakeDetectionError

# Module logger
logger = get_logger(__name__)

# Try to import DTW library
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    logger.info("dtaidistance not available, DTW metric disabled")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CorrelationMetrics:
    """
    Container for correlation analysis results.

    Attributes:
        pearson_correlation: Pearson correlation coefficient.
        pearson_pvalue: P-value for Pearson correlation.
        cross_correlation: Maximum cross-correlation value.
        optimal_lag: Optimal lag in frames.
        optimal_lag_ms: Optimal lag in milliseconds.
        dtw_distance: Dynamic Time Warping distance.
        coherence: Spectral coherence at speech frequencies.
        is_synchronized: Whether signals appear synchronized.
        confidence: Confidence score for synchronization.
    """
    pearson_correlation: float = 0.0
    pearson_pvalue: float = 1.0
    cross_correlation: float = 0.0
    optimal_lag: int = 0
    optimal_lag_ms: float = 0.0
    dtw_distance: float = float("inf")
    coherence: float = 0.0
    is_synchronized: bool = True
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pearson_correlation": self.pearson_correlation,
            "pearson_pvalue": self.pearson_pvalue,
            "cross_correlation": self.cross_correlation,
            "optimal_lag": self.optimal_lag,
            "optimal_lag_ms": self.optimal_lag_ms,
            "dtw_distance": self.dtw_distance,
            "coherence": self.coherence,
            "is_synchronized": self.is_synchronized,
            "confidence": self.confidence,
        }

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for classification."""
        return np.array([
            self.pearson_correlation,
            self.cross_correlation,
            self.optimal_lag / 10.0,  # Normalize lag
            self.coherence,
            self.confidence,
        ])


# =============================================================================
# Correlation Analyzer
# =============================================================================

class CorrelationAnalyzer:
    """
    Lip-audio synchronization analysis service.

    This class analyzes the correlation between lip movements and
    audio energy to detect lip-sync deepfakes.

    In genuine videos, there is typically a strong positive correlation
    between mouth opening and audio energy. Deepfakes often show poor
    synchronization.

    Attributes:
        fps: Video frame rate for time calculations.
        genuine_threshold: Correlation threshold for genuine videos.
        max_lag_frames: Maximum lag to search for cross-correlation.
        speech_freq_range: Frequency range for coherence analysis.

    Example:
        >>> analyzer = CorrelationAnalyzer(fps=25.0)
        >>> metrics = analyzer.analyze(mouth_signal, audio_energy)
        >>> if metrics.pearson_correlation < 0.3:
        ...     print("Potential deepfake detected!")
    """

    def __init__(
        self,
        fps: float = 25.0,
        genuine_threshold: float = 0.5,
        max_lag_frames: int = 5,
        max_lag_ms: float = 100.0,
        speech_freq_range: Tuple[float, float] = (2.0, 8.0),
    ) -> None:
        """
        Initialize correlation analyzer.

        Args:
            fps: Video frame rate.
            genuine_threshold: Correlation threshold for genuine videos.
            max_lag_frames: Maximum lag in frames for cross-correlation.
            max_lag_ms: Maximum lag in milliseconds (alternative to frames).
            speech_freq_range: Frequency range (Hz) for coherence.
        """
        self.fps = fps
        self.genuine_threshold = genuine_threshold

        # Calculate max lag in frames from ms if needed
        if max_lag_ms:
            self.max_lag_frames = int(max_lag_ms / 1000.0 * fps)
        else:
            self.max_lag_frames = max_lag_frames

        self.speech_freq_range = speech_freq_range

        logger.info(
            f"CorrelationAnalyzer initialized: fps={fps}, "
            f"threshold={genuine_threshold}, max_lag={self.max_lag_frames} frames"
        )

    def extract_mouth_opening_signal(
        self,
        landmarks_sequence: List[Dict[str, Tuple[int, int]]],
    ) -> np.ndarray:
        """
        Extract mouth opening signal from landmark sequence.

        Computes the vertical distance between upper and lower lip
        across all frames.

        Args:
            landmarks_sequence: List of landmark dictionaries per frame.

        Returns:
            Mouth opening signal (T,).
        """
        mouth_signal = []

        for landmarks in landmarks_sequence:
            if landmarks is None:
                mouth_signal.append(0.0)
                continue

            upper_lip = landmarks.get("upper_lip")
            lower_lip = landmarks.get("lower_lip")

            if upper_lip is None or lower_lip is None:
                mouth_signal.append(0.0)
            else:
                # Vertical distance between lips
                opening = abs(lower_lip[1] - upper_lip[1])
                mouth_signal.append(float(opening))

        return np.array(mouth_signal)

    def extract_mouth_signal_from_crops(
        self,
        face_crops: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract mouth opening signal from face crops using heuristics.

        Uses pixel intensity differences in the mouth region as a proxy
        for mouth opening.

        Args:
            face_crops: List of aligned face crop images.

        Returns:
            Mouth opening signal (T,).
        """
        mouth_signal = []

        for crop in face_crops:
            if crop is None or crop.size == 0:
                mouth_signal.append(0.0)
                continue

            h, w = crop.shape[:2]

            # Mouth region is approximately in lower middle of face
            mouth_top = int(h * 0.55)
            mouth_bottom = int(h * 0.75)
            mouth_left = int(w * 0.3)
            mouth_right = int(w * 0.7)

            # Extract mouth region
            mouth_region = crop[mouth_top:mouth_bottom, mouth_left:mouth_right]

            if mouth_region.size == 0:
                mouth_signal.append(0.0)
                continue

            # Convert to grayscale if color
            if len(mouth_region.shape) == 3:
                mouth_region = np.mean(mouth_region, axis=2)

            # Use variance as proxy for mouth opening
            # Open mouth has more variation (teeth, tongue, dark interior)
            variance = np.var(mouth_region)
            mouth_signal.append(float(variance))

        return np.array(mouth_signal)

    def extract_audio_energy_signal(
        self,
        mfcc_sequence: np.ndarray,
    ) -> np.ndarray:
        """
        Extract audio energy signal from MFCC sequence.

        Uses the first MFCC coefficient (log energy) as the signal.

        Args:
            mfcc_sequence: MFCC features (T, n_mfcc).

        Returns:
            Audio energy signal (T,).
        """
        if mfcc_sequence.size == 0:
            return np.array([])

        # First MFCC coefficient represents log energy
        energy = mfcc_sequence[:, 0]

        # Normalize
        if np.std(energy) > 0:
            energy = (energy - np.mean(energy)) / np.std(energy)

        return energy

    def compute_rms_energy(self, waveform: np.ndarray, n_frames: int) -> np.ndarray:
        """
        Compute RMS energy aligned with video frames.

        Args:
            waveform: Audio waveform.
            n_frames: Number of video frames.

        Returns:
            RMS energy per frame (n_frames,).
        """
        samples_per_frame = len(waveform) // n_frames
        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            if end <= len(waveform):
                frame = waveform[start:end]
                energy[i] = np.sqrt(np.mean(frame ** 2))

        return energy

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal to zero mean and unit variance.

        Args:
            signal: Input signal.

        Returns:
            Normalized signal.
        """
        if len(signal) == 0:
            return signal

        mean = np.mean(signal)
        std = np.std(signal)

        if std > 1e-8:
            return (signal - mean) / std
        else:
            return signal - mean

    @log_execution_time()
    def compute_pearson_correlation(
        self,
        mouth_signal: np.ndarray,
        audio_signal: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient.

        Args:
            mouth_signal: Mouth opening signal.
            audio_signal: Audio energy signal.

        Returns:
            Tuple of (correlation, p-value).
        """
        # Ensure same length
        min_len = min(len(mouth_signal), len(audio_signal))
        if min_len < 3:
            return 0.0, 1.0

        mouth = mouth_signal[:min_len]
        audio = audio_signal[:min_len]

        # Normalize
        mouth = self.normalize_signal(mouth)
        audio = self.normalize_signal(audio)

        try:
            corr, pval = pearsonr(mouth, audio)
            if np.isnan(corr):
                return 0.0, 1.0
            return float(corr), float(pval)
        except Exception as e:
            logger.warning(f"Pearson correlation failed: {e}")
            return 0.0, 1.0

    @log_execution_time()
    def compute_cross_correlation(
        self,
        mouth_signal: np.ndarray,
        audio_signal: np.ndarray,
    ) -> Tuple[float, int]:
        """
        Compute cross-correlation and find optimal lag.

        Args:
            mouth_signal: Mouth opening signal.
            audio_signal: Audio energy signal.

        Returns:
            Tuple of (max correlation, optimal lag in frames).
        """
        min_len = min(len(mouth_signal), len(audio_signal))
        if min_len < 3:
            return 0.0, 0

        mouth = self.normalize_signal(mouth_signal[:min_len])
        audio = self.normalize_signal(audio_signal[:min_len])

        try:
            # Compute full cross-correlation
            cross_corr = signal.correlate(mouth, audio, mode="full")

            # Normalize
            cross_corr = cross_corr / (min_len * np.std(mouth) * np.std(audio) + 1e-8)

            # Find center (zero lag)
            center = len(cross_corr) // 2

            # Search within max lag range
            start = max(0, center - self.max_lag_frames)
            end = min(len(cross_corr), center + self.max_lag_frames + 1)

            search_range = cross_corr[start:end]

            if len(search_range) == 0:
                return 0.0, 0

            # Find maximum
            max_idx = np.argmax(search_range)
            max_corr = search_range[max_idx]

            # Calculate lag (relative to center)
            optimal_lag = max_idx - (center - start)

            return float(max_corr), int(optimal_lag)

        except Exception as e:
            logger.warning(f"Cross-correlation failed: {e}")
            return 0.0, 0

    @log_execution_time()
    def compute_dtw_distance(
        self,
        mouth_signal: np.ndarray,
        audio_signal: np.ndarray,
    ) -> float:
        """
        Compute Dynamic Time Warping distance.

        DTW measures similarity between two sequences that may vary
        in speed. Lower distance indicates better synchronization.

        Args:
            mouth_signal: Mouth opening signal.
            audio_signal: Audio energy signal.

        Returns:
            DTW distance (lower is better).
        """
        if not DTW_AVAILABLE:
            return float("inf")

        min_len = min(len(mouth_signal), len(audio_signal))
        if min_len < 3:
            return float("inf")

        mouth = self.normalize_signal(mouth_signal[:min_len])
        audio = self.normalize_signal(audio_signal[:min_len])

        try:
            distance = dtw.distance(mouth, audio)
            return float(distance)
        except Exception as e:
            logger.warning(f"DTW computation failed: {e}")
            return float("inf")

    @log_execution_time()
    def compute_coherence(
        self,
        mouth_signal: np.ndarray,
        audio_signal: np.ndarray,
    ) -> float:
        """
        Compute spectral coherence at speech frequencies.

        Coherence measures the frequency-domain correlation between
        two signals. High coherence at speech frequencies indicates
        good synchronization.

        Args:
            mouth_signal: Mouth opening signal.
            audio_signal: Audio energy signal.

        Returns:
            Mean coherence in speech frequency range [0, 1].
        """
        min_len = min(len(mouth_signal), len(audio_signal))
        if min_len < 16:  # Need sufficient samples for coherence
            return 0.0

        mouth = mouth_signal[:min_len]
        audio = audio_signal[:min_len]

        try:
            # Compute coherence
            freqs, coherence = signal.coherence(
                mouth, audio, fs=self.fps, nperseg=min(64, min_len // 4)
            )

            # Find indices in speech frequency range
            freq_mask = (
                (freqs >= self.speech_freq_range[0]) &
                (freqs <= self.speech_freq_range[1])
            )

            if not np.any(freq_mask):
                return float(np.mean(coherence))

            # Mean coherence in speech range
            speech_coherence = np.mean(coherence[freq_mask])

            return float(speech_coherence)

        except Exception as e:
            logger.warning(f"Coherence computation failed: {e}")
            return 0.0

    @log_execution_time()
    def compute_all_metrics(
        self,
        mouth_signal: np.ndarray,
        audio_signal: np.ndarray,
    ) -> CorrelationMetrics:
        """
        Compute all correlation metrics.

        Args:
            mouth_signal: Mouth opening signal.
            audio_signal: Audio energy signal.

        Returns:
            CorrelationMetrics with all computed values.
        """
        metrics = CorrelationMetrics()

        # Pearson correlation
        pearson, pval = self.compute_pearson_correlation(mouth_signal, audio_signal)
        metrics.pearson_correlation = pearson
        metrics.pearson_pvalue = pval

        # Cross-correlation
        cross_corr, lag = self.compute_cross_correlation(mouth_signal, audio_signal)
        metrics.cross_correlation = cross_corr
        metrics.optimal_lag = lag
        metrics.optimal_lag_ms = lag * (1000.0 / self.fps)

        # DTW distance
        metrics.dtw_distance = self.compute_dtw_distance(mouth_signal, audio_signal)

        # Spectral coherence
        metrics.coherence = self.compute_coherence(mouth_signal, audio_signal)

        # Determine synchronization
        metrics.is_synchronized = (
            pearson >= self.genuine_threshold and
            abs(lag) <= self.max_lag_frames / 2
        )

        # Compute confidence score
        metrics.confidence = self._compute_confidence(metrics)

        return metrics

    def _compute_confidence(self, metrics: CorrelationMetrics) -> float:
        """
        Compute overall confidence score for synchronization.

        Args:
            metrics: Computed correlation metrics.

        Returns:
            Confidence score [0, 1].
        """
        # Combine multiple metrics into confidence score
        weights = {
            "pearson": 0.3,
            "cross_corr": 0.3,
            "lag_penalty": 0.2,
            "coherence": 0.2,
        }

        # Normalize pearson to [0, 1] range
        pearson_score = (metrics.pearson_correlation + 1) / 2

        # Cross-correlation is already in [-1, 1], normalize
        cross_score = (metrics.cross_correlation + 1) / 2

        # Lag penalty: penalize large lags
        lag_score = max(0, 1 - abs(metrics.optimal_lag) / (self.max_lag_frames + 1))

        # Coherence is already in [0, 1]
        coherence_score = metrics.coherence

        confidence = (
            weights["pearson"] * pearson_score +
            weights["cross_corr"] * cross_score +
            weights["lag_penalty"] * lag_score +
            weights["coherence"] * coherence_score
        )

        return float(np.clip(confidence, 0, 1))

    def analyze(
        self,
        mouth_signal: Optional[np.ndarray] = None,
        audio_signal: Optional[np.ndarray] = None,
        face_crops: Optional[List[np.ndarray]] = None,
        mfcc_sequence: Optional[np.ndarray] = None,
        landmarks_sequence: Optional[List[Dict]] = None,
    ) -> CorrelationMetrics:
        """
        Perform complete correlation analysis.

        This method can accept either pre-computed signals or raw data
        from which signals will be extracted.

        Args:
            mouth_signal: Pre-computed mouth opening signal.
            audio_signal: Pre-computed audio energy signal.
            face_crops: Face crop images (to extract mouth signal).
            mfcc_sequence: MFCC features (to extract audio signal).
            landmarks_sequence: Facial landmarks (to extract mouth signal).

        Returns:
            CorrelationMetrics with analysis results.
        """
        # Extract mouth signal if not provided
        if mouth_signal is None:
            if landmarks_sequence is not None:
                mouth_signal = self.extract_mouth_opening_signal(landmarks_sequence)
            elif face_crops is not None:
                mouth_signal = self.extract_mouth_signal_from_crops(face_crops)
            else:
                raise ValueError(
                    "Either mouth_signal, landmarks_sequence, or face_crops required"
                )

        # Extract audio signal if not provided
        if audio_signal is None:
            if mfcc_sequence is not None:
                audio_signal = self.extract_audio_energy_signal(mfcc_sequence)
            else:
                raise ValueError("Either audio_signal or mfcc_sequence required")

        return self.compute_all_metrics(mouth_signal, audio_signal)


# =============================================================================
# Service Factory
# =============================================================================

def create_correlation_analyzer_service(
    config: Optional[Dict[str, Any]] = None
) -> CorrelationAnalyzer:
    """
    Factory function to create a CorrelationAnalyzer from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured CorrelationAnalyzer instance.
    """
    if config is None:
        config = {}

    return CorrelationAnalyzer(
        fps=config.get("fps", 25.0),
        genuine_threshold=config.get("genuine_threshold", 0.5),
        max_lag_frames=config.get("max_lag_frames", 5),
        max_lag_ms=config.get("max_lag_ms"),
        speech_freq_range=tuple(config.get("speech_freq_range", [2.0, 8.0])),
    )
