"""Unit and integration tests for BodyScanner CV pipeline and WebSocket endpoint.

Compatible with both pytest and python -m unittest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hdt.api.body_scanner import (
    BodyScanner,
    LimbMeasurements,
    Point3D,
)


class TestBodyScannerPipeline(unittest.TestCase):
    """Test 3D Euclidean distances, scale factors, and temporal smoothing."""

    def test_point3d_distance(self) -> None:
        """Test 3D Euclidean distance in meters."""
        p1 = Point3D(x=0.0, y=0.0, z=0.0)
        p2 = Point3D(x=0.3, y=0.4, z=0.0)
        # 3-4-5 right triangle -> 0.50 m
        self.assertAlmostEqual(p1.distance_to(p2), 0.50, places=4)

        p3 = Point3D(x=1.0, y=2.0, z=3.0)
        p4 = Point3D(x=2.0, y=4.0, z=5.0)
        # dx=1, dy=2, dz=2 -> sqrt(1+4+4) = 3.0 m
        self.assertAlmostEqual(p3.distance_to(p4), 3.0, places=4)

    def test_limb_measurements_initialization(self) -> None:
        """Test LimbMeasurements auto-calculates total arm lengths."""
        meas = LimbMeasurements(
            left_upper_arm=0.32,
            left_forearm=0.26,
            right_upper_arm=0.31,
            right_forearm=0.27,
            shoulder_width=0.41,
        )
        self.assertAlmostEqual(meas.left_arm_total, 0.58, places=3)
        self.assertAlmostEqual(meas.right_arm_total, 0.58, places=3)
        d = meas.to_dict()
        self.assertEqual(d["left_upper_arm"], 0.32)
        self.assertEqual(d["unit"], "meters")

    def test_compute_limb_distances(self) -> None:
        """Test compute_limb_distances with known joint coordinates."""
        scanner = BodyScanner(auto_start_camera=False)

        joints = {
            "left_shoulder": Point3D(x=0.20, y=0.50, z=0.0),
            "right_shoulder": Point3D(x=-0.20, y=0.50, z=0.0),
            "left_elbow": Point3D(x=0.20, y=0.20, z=0.0),
            "right_elbow": Point3D(x=-0.20, y=0.20, z=0.0),
            "left_wrist": Point3D(x=0.20, y=-0.05, z=0.0),
            "right_wrist": Point3D(x=-0.20, y=-0.05, z=0.0),
        }

        distances = scanner.compute_limb_distances(joints)

        # Upper arms: 0.50 - 0.20 = 0.30 m
        self.assertAlmostEqual(distances["left_upper_arm"], 0.30, places=4)
        self.assertAlmostEqual(distances["right_upper_arm"], 0.30, places=4)

        # Forearms: 0.20 - (-0.05) = 0.25 m
        self.assertAlmostEqual(distances["left_forearm"], 0.25, places=4)
        self.assertAlmostEqual(distances["right_forearm"], 0.25, places=4)

        # Shoulder width: 0.20 - (-0.20) = 0.40 m
        self.assertAlmostEqual(distances["shoulder_width"], 0.40, places=4)

    def test_calculate_scale_factors(self) -> None:
        """Test scale factor derivation relative to standard references."""
        scanner = BodyScanner(
            auto_start_camera=False,
            reference_lengths={
                "left_upper_arm": 0.30,
                "left_forearm": 0.25,
                "right_upper_arm": 0.30,
                "right_forearm": 0.25,
                "shoulder_width": 0.40,
            },
        )

        measurements = {
            "left_upper_arm": 0.33,  # 1.10x
            "left_forearm": 0.25,    # 1.00x
            "right_upper_arm": 0.33, # 1.10x
            "right_forearm": 0.25,   # 1.00x
            "shoulder_width": 0.44,  # 1.10x
        }

        scales = scanner.calculate_scale_factors(measurements)
        self.assertAlmostEqual(scales["left_upper_arm"], 1.10, places=2)
        self.assertAlmostEqual(scales["left_forearm"], 1.00, places=2)
        self.assertAlmostEqual(scales["right_upper_arm"], 1.10, places=2)
        self.assertAlmostEqual(scales["shoulder_width"], 1.10, places=2)
        self.assertAlmostEqual(scales["overall"], 1.05, places=2)

    def test_temporal_smoothing(self) -> None:
        """Test Exponential Moving Average filter smoothing between frames."""
        scanner = BodyScanner(auto_start_camera=False, smoothing_factor=0.50)

        f1 = {"left_upper_arm": 0.30}
        s1 = scanner._apply_temporal_smoothing(f1)
        self.assertEqual(s1["left_upper_arm"], 0.30)

        # Sudden step increase to 0.40 with alpha=0.5 -> 0.5*0.40 + 0.5*0.30 = 0.35
        f2 = {"left_upper_arm": 0.40}
        s2 = scanner._apply_temporal_smoothing(f2)
        self.assertAlmostEqual(s2["left_upper_arm"], 0.35, places=4)

    def test_scanner_clean_release(self) -> None:
        """Test camera and MediaPipe resources are released cleanly."""
        scanner = BodyScanner(auto_start_camera=False)

        mock_cap = MagicMock()
        mock_pose = MagicMock()
        scanner.cap = mock_cap
        scanner.pose = mock_pose

        scanner.release()

        mock_cap.release.assert_called_once()
        mock_pose.close.assert_called_once()
        self.assertIsNone(scanner.cap)
        self.assertIsNone(scanner.pose)

    def test_camera_unavailable_payload(self) -> None:
        """Test get_frame_measurements returns camera_unavailable when cap is closed."""
        scanner = BodyScanner(auto_start_camera=False)
        scanner.cap = None

        payload = scanner.get_frame_measurements()
        self.assertEqual(payload["status"], "camera_unavailable")
        self.assertFalse(payload["detected"])
        self.assertIsNone(payload["measurements"])
        self.assertIsNone(payload["scale_factors"])

    def test_process_frame_searching_when_no_landmarks(self) -> None:
        """Test searching payload when MediaPipe detects no pose."""
        scanner = BodyScanner(auto_start_camera=False)
        mock_pose = MagicMock()
        mock_pose.process.return_value = MagicMock(pose_world_landmarks=None)
        scanner.pose = mock_pose

        mock_frame = MagicMock()
        with patch.object(scanner, "_cv2_module", MagicMock()) as mock_cv2:
            mock_cv2.cvtColor.return_value = MagicMock()
            payload = scanner.process_frame(mock_frame)

        self.assertEqual(payload["status"], "searching")
        self.assertFalse(payload["detected"])


class TestBodyScanWebSocket(unittest.TestCase):
    """Test FastAPI body scan endpoints and WebSocket streaming."""

    @classmethod
    def setUpClass(cls) -> None:
        # Mock any external scientific libraries if not installed in current environment
        if "scipy" not in sys.modules:
            try:
                import scipy  # noqa: F401
            except ImportError:
                mock_scipy = MagicMock()
                mock_scipy.__path__ = []
                sys.modules["scipy"] = mock_scipy
                for sub in ["ndimage", "spatial", "spatial.distance", "stats"]:
                    m = MagicMock()
                    setattr(mock_scipy, sub, m)
                    sys.modules[f"scipy.{sub}"] = m

        for mod in ["SimpleITK", "nibabel", "skimage", "skimage.measure", "pandas"]:
            if mod not in sys.modules:
                m = MagicMock()
                m.__path__ = []
                sys.modules[mod] = m

        from starlette.testclient import TestClient

        from hdt.api.app import app

        cls.app = app
        cls.client = TestClient(app)

    def test_reference_endpoint(self) -> None:
        """Test GET /body-scan/reference returns default values."""
        resp = self.client.get("/body-scan/reference")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("reference_lengths_meters", data)
        self.assertEqual(data["websocket_endpoint"], "/ws/body-scan")

    def test_websocket_stream_and_cleanup(self) -> None:
        """Test WebSocket client connects, streams frame, and disconnects cleanly."""
        with self.client.websocket_connect("/ws/body-scan?camera_index=99") as ws:
            msg = ws.receive_json()
            self.assertIn("status", msg)
            self.assertIn("detected", msg)
            self.assertIn("timestamp", msg)
            # Camera 99 is invalid, expect camera_unavailable status
            self.assertEqual(msg["status"], "camera_unavailable")
            self.assertFalse(msg["detected"])



if __name__ == "__main__":
    unittest.main()
