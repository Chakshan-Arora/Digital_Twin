"""Real-time body scanning and limb measurement extraction module.

Captures video via OpenCV VideoCapture, processes frames with MediaPipe Pose (complexity 2),
extracts 3D world landmarks in physical meters, and computes Euclidean limb lengths and scale factors.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Standard anthropometric reference lengths in meters for an average adult human
DEFAULT_REFERENCE_LENGTHS: dict[str, float] = {
    "left_upper_arm": 0.30,  # ~30 cm humerus length
    "left_forearm": 0.25,    # ~25 cm radius/ulna length
    "right_upper_arm": 0.30, # ~30 cm
    "right_forearm": 0.25,   # ~25 cm
    "shoulder_width": 0.38,  # ~38 cm biacromial diameter
}

# MediaPipe Pose Landmark Enum Indices
MP_LANDMARK_LEFT_SHOULDER = 11
MP_LANDMARK_RIGHT_SHOULDER = 12
MP_LANDMARK_LEFT_ELBOW = 13
MP_LANDMARK_RIGHT_ELBOW = 14
MP_LANDMARK_LEFT_WRIST = 15
MP_LANDMARK_RIGHT_WRIST = 16


@dataclass
class Point3D:
    """3D point coordinate in meters with confidence visibility."""

    x: float
    y: float
    z: float
    visibility: float = 1.0

    def distance_to(self, other: Point3D) -> float:
        """Calculate Euclidean distance in 3D meters."""
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "visibility": round(self.visibility, 3),
        }


@dataclass
class LimbMeasurements:
    """Calculated limb lengths and scale factors."""

    left_upper_arm: float
    left_forearm: float
    right_upper_arm: float
    right_forearm: float
    shoulder_width: float
    left_arm_total: float = 0.0
    right_arm_total: float = 0.0
    unit: str = "meters"

    def __post_init__(self) -> None:
        if self.left_arm_total == 0.0:
            self.left_arm_total = round(self.left_upper_arm + self.left_forearm, 4)
        if self.right_arm_total == 0.0:
            self.right_arm_total = round(self.right_upper_arm + self.right_forearm, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_upper_arm": round(self.left_upper_arm, 4),
            "left_forearm": round(self.left_forearm, 4),
            "right_upper_arm": round(self.right_upper_arm, 4),
            "right_forearm": round(self.right_forearm, 4),
            "left_arm_total": round(self.left_arm_total, 4),
            "right_arm_total": round(self.right_arm_total, 4),
            "shoulder_width": round(self.shoulder_width, 4),
            "unit": self.unit,
        }


class BodyScanner:
    """Real-time body scanning and anthropometric limb measurement extractor.

    Uses OpenCV for camera acquisition and MediaPipe Pose with model_complexity=2
    to extract 3D world landmarks (measured in meters). Calculates Euclidean
    distances for upper arms, forearms, and shoulder width, maintaining exponential
    moving average smoothing to eliminate jitter.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_factor: float = 0.35,
        reference_lengths: dict[str, float] | None = None,
        auto_start_camera: bool = True,
    ) -> None:
        """Initialize OpenCV camera capture and MediaPipe Pose model.

        Parameters
        ----------
        camera_index : int
            OpenCV video capture device index (default 0).
        model_complexity : int
            MediaPipe Pose landmark model complexity (0, 1, or 2). Complexity 2
            provides the most accurate 3D landmark prediction.
        min_detection_confidence : float
            Minimum confidence value ([0.0, 1.0]) for person detection.
        min_tracking_confidence : float
            Minimum confidence value ([0.0, 1.0]) for landmark tracking.
        smoothing_factor : float
            Exponential moving average coefficient alpha in (0.0, 1.0].
            Higher values track faster; lower values provide smoother outputs.
        reference_lengths : dict[str, float] | None
            Base reference lengths in meters to calculate scale factors.
        auto_start_camera : bool
            Whether to open the camera immediately on instantiation.
        """
        self.camera_index = camera_index
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_factor = max(0.01, min(1.0, smoothing_factor))

        self.reference_lengths = (
            dict(reference_lengths) if reference_lengths else dict(DEFAULT_REFERENCE_LENGTHS)
        )

        self._smoothed_measurements: dict[str, float] = {}
        self.cap: Any = None
        self.pose: Any = None
        self._cv2_module: Any = None
        self._mp_module: Any = None
        self._camera_initialized: bool = False
        self._pose_initialized: bool = False
        self._last_frame_time: float = time.time()
        self._frame_count: int = 0
        self._measured_fps: float = 0.0

        if auto_start_camera:
            self.initialize()

    def initialize(self) -> bool:
        """Initialize camera and MediaPipe Pose model.

        Returns
        -------
        bool
            True if camera and model initialized successfully, False otherwise.
        """
        # Initialize MediaPipe Pose
        try:
            import mediapipe as mp
            self._mp_module = mp
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._pose_initialized = True
            logger.info("MediaPipe Pose (complexity=%d) initialized", self.model_complexity)
        except Exception as exc:
            logger.warning("Could not initialize MediaPipe Pose: %s", exc)
            self._pose_initialized = False

        # Initialize OpenCV VideoCapture
        try:
            import cv2
            self._cv2_module = cv2
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                # Set reasonable camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self._camera_initialized = True
                logger.info("OpenCV VideoCapture opened on camera index %d", self.camera_index)
            else:
                logger.warning("OpenCV VideoCapture failed to open device %d", self.camera_index)
                self._camera_initialized = False
        except Exception as exc:
            logger.warning("Could not initialize OpenCV VideoCapture: %s", exc)
            self._camera_initialized = False

        return self._camera_initialized and self._pose_initialized

    def is_camera_open(self) -> bool:
        """Check if OpenCV camera resource is open."""
        return bool(self.cap is not None and getattr(self.cap, "isOpened", lambda: False)())

    def calibrate_baseline(self) -> dict[str, float] | None:
        """Calibrate baseline reference lengths using current measurements."""
        if self._smoothed_measurements:
            self.reference_lengths = dict(self._smoothed_measurements)
            return dict(self.reference_lengths)
        return None

    def calculate_scale_factors(self, measurements: dict[str, float]) -> dict[str, float]:
        """Calculate scale factors relative to anthropometric reference lengths.

        Scale factors are clamped to [0.4, 2.5] for biological stability.
        """
        scales: dict[str, float] = {}
        for key in ["left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm", "shoulder_width"]:
            meas = measurements.get(key, 0.0)
            ref = self.reference_lengths.get(key, DEFAULT_REFERENCE_LENGTHS.get(key, 0.30))
            if ref > 0 and meas > 0:
                raw_scale = meas / ref
                scales[key] = round(max(0.4, min(2.5, raw_scale)), 4)
            else:
                scales[key] = 1.0

        # Compute overall body scale factor
        arm_scales = [
            scales[k]
            for k in ["left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm"]
            if k in scales
        ]
        scales["overall"] = round(sum(arm_scales) / len(arm_scales), 4) if arm_scales else 1.0
        return scales

    def extract_landmarks_from_world(self, world_landmarks: Any) -> dict[str, Point3D] | None:
        """Extract relevant joints as 3D points in meters from MediaPipe world landmarks."""
        try:
            landmarks_list = world_landmarks.landmark
            if len(landmarks_list) < 17:
                return None

            left_shoulder = landmarks_list[MP_LANDMARK_LEFT_SHOULDER]
            right_shoulder = landmarks_list[MP_LANDMARK_RIGHT_SHOULDER]
            left_elbow = landmarks_list[MP_LANDMARK_LEFT_ELBOW]
            right_elbow = landmarks_list[MP_LANDMARK_RIGHT_ELBOW]
            left_wrist = landmarks_list[MP_LANDMARK_LEFT_WRIST]
            right_wrist = landmarks_list[MP_LANDMARK_RIGHT_WRIST]

            return {
                "left_shoulder": Point3D(
                    x=float(left_shoulder.x),
                    y=float(left_shoulder.y),
                    z=float(left_shoulder.z),
                    visibility=float(getattr(left_shoulder, "visibility", 1.0)),
                ),
                "right_shoulder": Point3D(
                    x=float(right_shoulder.x),
                    y=float(right_shoulder.y),
                    z=float(right_shoulder.z),
                    visibility=float(getattr(right_shoulder, "visibility", 1.0)),
                ),
                "left_elbow": Point3D(
                    x=float(left_elbow.x),
                    y=float(left_elbow.y),
                    z=float(left_elbow.z),
                    visibility=float(getattr(left_elbow, "visibility", 1.0)),
                ),
                "right_elbow": Point3D(
                    x=float(right_elbow.x),
                    y=float(right_elbow.y),
                    z=float(right_elbow.z),
                    visibility=float(getattr(right_elbow, "visibility", 1.0)),
                ),
                "left_wrist": Point3D(
                    x=float(left_wrist.x),
                    y=float(left_wrist.y),
                    z=float(left_wrist.z),
                    visibility=float(getattr(left_wrist, "visibility", 1.0)),
                ),
                "right_wrist": Point3D(
                    x=float(right_wrist.x),
                    y=float(right_wrist.y),
                    z=float(right_wrist.z),
                    visibility=float(getattr(right_wrist, "visibility", 1.0)),
                ),
            }
        except Exception as exc:
            logger.warning("Error extracting world landmarks: %s", exc)
            return None

    def compute_limb_distances(self, joints: dict[str, Point3D]) -> dict[str, float]:
        """Compute Euclidean 3D physical distances between joints in meters."""
        left_shoulder = joints["left_shoulder"]
        right_shoulder = joints["right_shoulder"]
        left_elbow = joints["left_elbow"]
        right_elbow = joints["right_elbow"]
        left_wrist = joints["left_wrist"]
        right_wrist = joints["right_wrist"]

        left_upper_arm = left_shoulder.distance_to(left_elbow)
        left_forearm = left_elbow.distance_to(left_wrist)
        right_upper_arm = right_shoulder.distance_to(right_elbow)
        right_forearm = right_elbow.distance_to(right_wrist)
        shoulder_width = left_shoulder.distance_to(right_shoulder)

        return {
            "left_upper_arm": left_upper_arm,
            "left_forearm": left_forearm,
            "right_upper_arm": right_upper_arm,
            "right_forearm": right_forearm,
            "shoulder_width": shoulder_width,
            "left_arm_total": left_upper_arm + left_forearm,
            "right_arm_total": right_upper_arm + right_forearm,
        }

    def _apply_temporal_smoothing(self, raw_measurements: dict[str, float]) -> dict[str, float]:
        """Apply exponential moving average to filter out high-frequency noise."""
        alpha = self.smoothing_factor
        smoothed: dict[str, float] = {}

        for key, val in raw_measurements.items():
            if key not in self._smoothed_measurements:
                smoothed[key] = val
            else:
                prev_val = self._smoothed_measurements[key]
                smoothed[key] = alpha * val + (1.0 - alpha) * prev_val

        self._smoothed_measurements = smoothed
        return smoothed

    def process_frame(self, frame_bgr: Any) -> dict[str, Any]:
        """Process a single BGR frame image and extract body measurements.

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV image in BGR format.

        Returns
        -------
        dict[str, Any]
            JSON-serializable response payload with detection status,
            measurements, scale factors, and raw landmark coordinates.
        """
        now = time.time()
        self._frame_count += 1
        elapsed = now - self._last_frame_time
        if elapsed >= 0.5:
            self._measured_fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._last_frame_time = now

        if self.pose is None:
            return {
                "status": "model_unavailable",
                "detected": False,
                "message": "MediaPipe Pose model not initialized",
                "timestamp": now,
                "fps": self._measured_fps,
            }

        cv2 = self._cv2_module
        if cv2 is None:
            import cv2 as cv_mod
            cv2 = cv_mod

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.pose.process(image_rgb)

        if not results or not results.pose_world_landmarks:
            return {
                "status": "searching",
                "detected": False,
                "message": "No person detected in frame. Please step into camera view.",
                "timestamp": now,
                "fps": self._measured_fps,
                "measurements": None,
                "scale_factors": None,
            }

        joints = self.extract_landmarks_from_world(results.pose_world_landmarks)
        if not joints:
            return {
                "status": "searching",
                "detected": False,
                "message": "Landmarks could not be extracted",
                "timestamp": now,
                "fps": self._measured_fps,
            }

        raw_distances = self.compute_limb_distances(joints)
        smoothed = self._apply_temporal_smoothing(raw_distances)
        scale_factors = self.calculate_scale_factors(smoothed)

        measurements_obj = LimbMeasurements(
            left_upper_arm=smoothed["left_upper_arm"],
            left_forearm=smoothed["left_forearm"],
            right_upper_arm=smoothed["right_upper_arm"],
            right_forearm=smoothed["right_forearm"],
            shoulder_width=smoothed["shoulder_width"],
            left_arm_total=smoothed["left_arm_total"],
            right_arm_total=smoothed["right_arm_total"],
        )

        return {
            "status": "detected",
            "detected": True,
            "message": "Subject body detected and tracked successfully",
            "timestamp": now,
            "fps": self._measured_fps,
            "measurements": measurements_obj.to_dict(),
            "scale_factors": scale_factors,
            "raw_landmarks": {k: pt.to_dict() for k, pt in joints.items()},
        }

    def get_frame_measurements(self) -> dict[str, Any]:
        """Capture one frame from webcam and extract measurements.

        Handles camera read failures and returns friendly status payloads.
        """
        now = time.time()
        if self.cap is None or not self.cap.isOpened():
            return {
                "status": "camera_unavailable",
                "detected": False,
                "message": (
                    f"Camera index {self.camera_index} is unavailable. "
                    "Ensure webcam is connected and not opened by another application."
                ),
                "timestamp": now,
                "fps": 0.0,
                "measurements": None,
                "scale_factors": None,
            }

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return {
                "status": "camera_read_failed",
                "detected": False,
                "message": "Failed to grab frame from video device",
                "timestamp": now,
                "fps": 0.0,
                "measurements": None,
                "scale_factors": None,
            }

        return self.process_frame(frame)

    def release(self) -> None:
        """Cleanly release OpenCV camera capture and MediaPipe Pose resources.

        Guarantees that hardware devices are freed immediately.
        """
        logger.info("Releasing BodyScanner resources (camera_index=%d)...", self.camera_index)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                logger.warning("Error releasing VideoCapture: %s", exc)
            self.cap = None
            self._camera_initialized = False

        if self.pose is not None:
            try:
                self.pose.close()
            except Exception as exc:
                logger.warning("Error closing MediaPipe Pose: %s", exc)
            self.pose = None
            self._pose_initialized = False

    def __enter__(self) -> BodyScanner:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
