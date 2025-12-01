"""
Face Detection and Alignment Module
=====================================

This module provides face detection, landmark extraction, and face
alignment capabilities for the deepfake detection pipeline.

Features:
    - Multiple backend support (MediaPipe, OpenCV Haar)
    - Facial landmark extraction (468 points with MediaPipe)
    - Face alignment using eye positions
    - Configurable crop margins and output sizes
    - Mouth region extraction for lip-sync analysis

Microservice API:
    POST /detect
        Request: {"frame": base64_image}
        Response: {"faces": [{"bbox": [...], "landmarks": {...}}]}

Example Usage:
    >>> from src.preprocessing.face_detector import FaceDetector
    >>> detector = FaceDetector(backend="mediapipe")
    >>> result = detector.detect_and_crop(frame)
    >>> if result.success:
    ...     face_crop = result.face_crop
    ...     landmarks = result.landmarks
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.utils.logging import get_logger, log_execution_time
from src.utils.exceptions import FaceDetectionError

# Module logger
logger = get_logger(__name__)

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available, falling back to Haar cascades")


# =============================================================================
# Enums and Constants
# =============================================================================

class DetectorBackend(Enum):
    """Supported face detection backends."""
    MEDIAPIPE = "mediapipe"
    HAAR = "haar"
    OPENCV_DNN = "opencv_dnn"


# MediaPipe Face Mesh landmark indices
class LandmarkIndices:
    """MediaPipe Face Mesh landmark indices for key facial features."""
    # Eyes
    LEFT_EYE_CENTER = 468  # Approximate center
    RIGHT_EYE_CENTER = 473
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362

    # Lips
    UPPER_LIP_TOP = 13
    UPPER_LIP_BOTTOM = 14
    LOWER_LIP_TOP = 14
    LOWER_LIP_BOTTOM = 17
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_CENTER = 0
    LOWER_LIP_CENTER = 17

    # Nose
    NOSE_TIP = 4

    # Face contour
    CHIN = 152
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoundingBox:
    """Face bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) tuple."""
        return (self.x, self.y, self.width, self.height)

    def expand(self, factor: float) -> "BoundingBox":
        """
        Expand bounding box by a factor.

        Args:
            factor: Expansion factor (1.3 = 30% larger).

        Returns:
            New expanded BoundingBox.
        """
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)
        new_x = self.x - (new_width - self.width) // 2
        new_y = self.y - (new_height - self.height) // 2
        return BoundingBox(new_x, new_y, new_width, new_height)


@dataclass
class FaceLandmarks:
    """
    Container for facial landmarks.

    Attributes:
        raw_landmarks: Raw landmark coordinates from detector.
        left_eye: Left eye center (x, y).
        right_eye: Right eye center (x, y).
        nose_tip: Nose tip (x, y).
        mouth_left: Left mouth corner (x, y).
        mouth_right: Right mouth corner (x, y).
        upper_lip: Upper lip center (x, y).
        lower_lip: Lower lip center (x, y).
    """
    raw_landmarks: Optional[np.ndarray] = None
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    nose_tip: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    upper_lip: Optional[Tuple[int, int]] = None
    lower_lip: Optional[Tuple[int, int]] = None

    @property
    def mouth_opening(self) -> float:
        """Calculate vertical mouth opening distance."""
        if self.upper_lip and self.lower_lip:
            return abs(self.lower_lip[1] - self.upper_lip[1])
        return 0.0

    @property
    def mouth_width(self) -> float:
        """Calculate horizontal mouth width."""
        if self.mouth_left and self.mouth_right:
            return abs(self.mouth_right[0] - self.mouth_left[0])
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert landmarks to dictionary."""
        return {
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "nose_tip": self.nose_tip,
            "mouth_left": self.mouth_left,
            "mouth_right": self.mouth_right,
            "upper_lip": self.upper_lip,
            "lower_lip": self.lower_lip,
            "mouth_opening": self.mouth_opening,
            "mouth_width": self.mouth_width,
        }


@dataclass
class FaceDetectionResult:
    """
    Result of face detection operation.

    Attributes:
        success: Whether detection was successful.
        bbox: Detected face bounding box.
        landmarks: Facial landmarks.
        confidence: Detection confidence score.
        face_crop: Aligned and cropped face image.
        mouth_crop: Cropped mouth region.
        error: Error message if detection failed.
    """
    success: bool = False
    bbox: Optional[BoundingBox] = None
    landmarks: Optional[FaceLandmarks] = None
    confidence: float = 0.0
    face_crop: Optional[np.ndarray] = None
    mouth_crop: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class BatchDetectionResult:
    """Result of batch face detection."""
    results: List[FaceDetectionResult] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0


# =============================================================================
# Face Detector
# =============================================================================

class FaceDetector:
    """
    Face detection and alignment service.

    This class provides comprehensive face detection capabilities including
    landmark extraction, face alignment, and cropping for the deepfake
    detection pipeline.

    Attributes:
        backend: Detection backend (mediapipe, haar).
        min_confidence: Minimum detection confidence.
        output_size: Output face crop size.
        margin_factor: Bounding box expansion factor.
        enable_alignment: Whether to align faces.

    Example:
        >>> detector = FaceDetector(backend="mediapipe")
        >>> result = detector.detect_and_crop(frame)
        >>> if result.success:
        ...     cv2.imwrite("face.jpg", result.face_crop)
    """

    def __init__(
        self,
        backend: str = "mediapipe",
        min_confidence: float = 0.7,
        output_size: Tuple[int, int] = (224, 224),
        margin_factor: float = 1.3,
        enable_alignment: bool = True,
        mouth_crop_size: Tuple[int, int] = (96, 96),
    ) -> None:
        """
        Initialize the face detector.

        Args:
            backend: Detection backend ("mediapipe" or "haar").
            min_confidence: Minimum detection confidence (0-1).
            output_size: Output face crop size (width, height).
            margin_factor: Bounding box expansion factor.
            enable_alignment: Enable face alignment.
            mouth_crop_size: Size of mouth region crop.

        Raises:
            FaceDetectionError: If backend initialization fails.
        """
        self.backend = DetectorBackend(backend.lower())
        self.min_confidence = min_confidence
        self.output_size = output_size
        self.margin_factor = margin_factor
        self.enable_alignment = enable_alignment
        self.mouth_crop_size = mouth_crop_size

        # Initialize backend
        self._detector = None
        self._face_mesh = None
        self._init_backend()

        logger.info(
            f"FaceDetector initialized: backend={backend}, "
            f"min_confidence={min_confidence}, output_size={output_size}"
        )

    def _init_backend(self) -> None:
        """Initialize the detection backend."""
        if self.backend == DetectorBackend.MEDIAPIPE:
            if not MEDIAPIPE_AVAILABLE:
                logger.warning("MediaPipe not available, falling back to Haar")
                self.backend = DetectorBackend.HAAR
                self._init_haar()
            else:
                self._init_mediapipe()
        elif self.backend == DetectorBackend.HAAR:
            self._init_haar()
        else:
            raise FaceDetectionError(
                f"Unsupported backend: {self.backend}",
                code="UNSUPPORTED_BACKEND",
            )

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe face detection and mesh."""
        try:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=self.min_confidence,
                model_selection=1,  # Full range model
            )
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=0.5,
            )
            logger.debug("MediaPipe initialized successfully")
        except Exception as e:
            raise FaceDetectionError(
                f"Failed to initialize MediaPipe: {e}",
                cause=e,
            )

    def _init_haar(self) -> None:
        """Initialize OpenCV Haar cascade detector."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade_path)
            if self._detector.empty():
                raise FaceDetectionError(
                    "Failed to load Haar cascade classifier",
                    code="HAAR_LOAD_FAILED",
                )
            logger.debug("Haar cascade initialized successfully")
        except Exception as e:
            raise FaceDetectionError(
                f"Failed to initialize Haar cascade: {e}",
                cause=e,
            )

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """
        Detect all faces in a frame.

        Args:
            frame: Input frame (BGR numpy array).

        Returns:
            List of (BoundingBox, confidence) tuples.

        Raises:
            FaceDetectionError: If detection fails.
        """
        if frame is None or frame.size == 0:
            raise FaceDetectionError(
                "Invalid frame: empty or None",
                code="INVALID_FRAME",
            )

        if self.backend == DetectorBackend.MEDIAPIPE:
            return self._detect_mediapipe(frame)
        else:
            return self._detect_haar(frame)

    def _detect_mediapipe(self, frame: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """Detect faces using MediaPipe."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                confidence = detection.score[0] if detection.score else 0.0

                faces.append((
                    BoundingBox(x, y, width, height),
                    confidence,
                ))

        return faces

    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """Detect faces using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        faces = []
        for (x, y, w, h) in detections:
            # Haar doesn't provide confidence, use placeholder
            faces.append((BoundingBox(x, y, w, h), 0.9))

        return faces

    def extract_landmarks(
        self,
        frame: np.ndarray,
        bbox: Optional[BoundingBox] = None,
    ) -> Optional[FaceLandmarks]:
        """
        Extract facial landmarks from a frame.

        Args:
            frame: Input frame (BGR numpy array).
            bbox: Optional bounding box to crop before extraction.

        Returns:
            FaceLandmarks object or None if extraction fails.
        """
        if self.backend != DetectorBackend.MEDIAPIPE or self._face_mesh is None:
            # Haar doesn't support landmarks
            return None

        try:
            # Optionally crop to bounding box
            if bbox:
                x, y, w, h = bbox.to_tuple()
                x, y = max(0, x), max(0, y)
                crop = frame[y:y+h, x:x+w]
                offset = (x, y)
            else:
                crop = frame
                offset = (0, 0)

            # Convert to RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_crop)

            if not results.multi_face_landmarks:
                return None

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = crop.shape[:2]

            # Extract raw landmarks as numpy array
            raw = np.array([
                (int(lm.x * w) + offset[0], int(lm.y * h) + offset[1])
                for lm in face_landmarks.landmark
            ])

            # Extract key landmarks
            def get_point(idx: int) -> Tuple[int, int]:
                return (
                    int(face_landmarks.landmark[idx].x * w) + offset[0],
                    int(face_landmarks.landmark[idx].y * h) + offset[1],
                )

            # Calculate eye centers from multiple points
            left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [263, 362, 384, 385, 386, 387, 388, 466]

            left_eye = np.mean([get_point(i) for i in left_eye_indices], axis=0)
            right_eye = np.mean([get_point(i) for i in right_eye_indices], axis=0)

            return FaceLandmarks(
                raw_landmarks=raw,
                left_eye=(int(left_eye[0]), int(left_eye[1])),
                right_eye=(int(right_eye[0]), int(right_eye[1])),
                nose_tip=get_point(LandmarkIndices.NOSE_TIP),
                mouth_left=get_point(LandmarkIndices.MOUTH_LEFT),
                mouth_right=get_point(LandmarkIndices.MOUTH_RIGHT),
                upper_lip=get_point(LandmarkIndices.UPPER_LIP_TOP),
                lower_lip=get_point(LandmarkIndices.LOWER_LIP_BOTTOM),
            )

        except Exception as e:
            logger.warning(f"Landmark extraction failed: {e}")
            return None

    def align_face(
        self,
        frame: np.ndarray,
        landmarks: FaceLandmarks,
    ) -> np.ndarray:
        """
        Align face based on eye positions.

        Applies an affine transformation to rotate and scale the face
        so that eyes are horizontal and at a standard distance.

        Args:
            frame: Input frame.
            landmarks: Facial landmarks with eye positions.

        Returns:
            Aligned face image.
        """
        if landmarks.left_eye is None or landmarks.right_eye is None:
            return frame

        left_eye = np.array(landmarks.left_eye)
        right_eye = np.array(landmarks.right_eye)

        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))

        # Calculate center point between eyes
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # Calculate current eye distance
        current_dist = np.sqrt(dx**2 + dy**2)

        # Target eye distance (proportion of output size)
        target_dist = self.output_size[0] * 0.4
        scale = target_dist / current_dist if current_dist > 0 else 1.0

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Adjust translation to center the face
        M[0, 2] += self.output_size[0] / 2 - center[0]
        M[1, 2] += self.output_size[1] / 2 - center[1]

        # Apply transformation
        aligned = cv2.warpAffine(
            frame,
            M,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray:
        """
        Crop and resize face region.

        Args:
            frame: Input frame.
            bbox: Face bounding box.

        Returns:
            Cropped and resized face image.
        """
        # Expand bounding box
        expanded = bbox.expand(self.margin_factor)

        # Clip to image bounds
        h, w = frame.shape[:2]
        x1 = max(0, expanded.x)
        y1 = max(0, expanded.y)
        x2 = min(w, expanded.x + expanded.width)
        y2 = min(h, expanded.y + expanded.height)

        # Crop
        crop = frame[y1:y2, x1:x2]

        # Resize to output size
        if crop.size > 0:
            crop = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_LINEAR)

        return crop

    def crop_mouth(
        self,
        frame: np.ndarray,
        landmarks: FaceLandmarks,
    ) -> Optional[np.ndarray]:
        """
        Crop mouth region from frame.

        Args:
            frame: Input frame.
            landmarks: Facial landmarks.

        Returns:
            Cropped mouth region or None.
        """
        if landmarks.mouth_left is None or landmarks.mouth_right is None:
            return None

        if landmarks.upper_lip is None or landmarks.lower_lip is None:
            return None

        # Calculate mouth bounding box
        left_x = landmarks.mouth_left[0]
        right_x = landmarks.mouth_right[0]
        top_y = landmarks.upper_lip[1]
        bottom_y = landmarks.lower_lip[1]

        # Add margin
        width = right_x - left_x
        height = bottom_y - top_y
        margin_x = int(width * 0.3)
        margin_y = int(height * 0.5)

        x1 = max(0, left_x - margin_x)
        y1 = max(0, top_y - margin_y)
        x2 = min(frame.shape[1], right_x + margin_x)
        y2 = min(frame.shape[0], bottom_y + margin_y)

        # Crop and resize
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crop = cv2.resize(crop, self.mouth_crop_size, interpolation=cv2.INTER_LINEAR)
            return crop

        return None

    @log_execution_time()
    def detect_and_crop(
        self,
        frame: np.ndarray,
        extract_mouth: bool = True,
    ) -> FaceDetectionResult:
        """
        Complete face detection and cropping pipeline.

        This method performs:
            1. Face detection
            2. Landmark extraction
            3. Face alignment (optional)
            4. Face cropping
            5. Mouth region extraction (optional)

        Args:
            frame: Input frame (BGR numpy array).
            extract_mouth: Whether to extract mouth region.

        Returns:
            FaceDetectionResult with cropped face and landmarks.

        Example:
            >>> result = detector.detect_and_crop(frame)
            >>> if result.success:
            ...     process_face(result.face_crop)
        """
        result = FaceDetectionResult()

        try:
            # Detect faces
            faces = self.detect_faces(frame)

            if not faces:
                result.error = "No face detected"
                return result

            # Use the largest face (or first one)
            faces.sort(key=lambda x: x[0].area, reverse=True)
            bbox, confidence = faces[0]

            if confidence < self.min_confidence:
                result.error = f"Detection confidence too low: {confidence:.2f}"
                return result

            result.bbox = bbox
            result.confidence = confidence

            # Extract landmarks
            landmarks = self.extract_landmarks(frame, bbox)
            result.landmarks = landmarks

            # Align and crop face
            if self.enable_alignment and landmarks:
                face_crop = self.align_face(frame, landmarks)
            else:
                face_crop = self.crop_face(frame, bbox)

            result.face_crop = face_crop

            # Extract mouth region
            if extract_mouth and landmarks:
                result.mouth_crop = self.crop_mouth(frame, landmarks)

            result.success = True

        except FaceDetectionError as e:
            result.error = str(e)
            logger.warning(f"Face detection failed: {e}")

        except Exception as e:
            result.error = str(e)
            logger.exception(f"Unexpected error in face detection: {e}")

        return result

    def process_batch(
        self,
        frames: List[np.ndarray],
        extract_mouth: bool = True,
    ) -> BatchDetectionResult:
        """
        Process a batch of frames.

        Args:
            frames: List of frames to process.
            extract_mouth: Whether to extract mouth regions.

        Returns:
            BatchDetectionResult with results for each frame.
        """
        batch_result = BatchDetectionResult()

        for frame in frames:
            result = self.detect_and_crop(frame, extract_mouth)
            batch_result.results.append(result)

            if result.success:
                batch_result.success_count += 1
            else:
                batch_result.failure_count += 1

        logger.info(
            f"Batch processing complete: {batch_result.success_count} succeeded, "
            f"{batch_result.failure_count} failed"
        )

        return batch_result

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "_face_detection") and self._face_detection:
            self._face_detection.close()
        if hasattr(self, "_face_mesh") and self._face_mesh:
            self._face_mesh.close()


# =============================================================================
# Service Factory
# =============================================================================

def create_face_detector_service(config: Optional[Dict[str, Any]] = None) -> FaceDetector:
    """
    Factory function to create a FaceDetector from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured FaceDetector instance.
    """
    if config is None:
        config = {}

    return FaceDetector(
        backend=config.get("backend", "mediapipe"),
        min_confidence=config.get("min_confidence", 0.7),
        output_size=tuple(config.get("output_size", [224, 224])),
        margin_factor=config.get("margin_factor", 1.3),
        enable_alignment=config.get("enable_alignment", True),
        mouth_crop_size=tuple(config.get("mouth_crop_size", [96, 96])),
    )
