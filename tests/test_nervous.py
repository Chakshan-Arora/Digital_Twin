"""Unit tests for procedural nervous system geometry, anchors, and conduction signals."""

import math

from hdt.atlas.models import FiberType
from hdt.body.presets import PRESETS
from hdt.nervous.anchors import DERMATOME_ANCHORS, nearest_dermatome, skin_anchor
from hdt.nervous.geometry import (
    full_pathway_polyline,
    nerve_polyline,
    spinal_cord_levels,
    spinal_cord_polyline,
    total_length_mm,
)
from hdt.nervous.signal import signal_timeline


def test_straight_polyline_length_analytic():
    """Verify total_length_mm matches hand-computed analytic distance for 3D lines."""
    straight_line = [(0.0, 0.0, 0.0), (0.0, 0.0, 400.0), (0.0, 0.0, 850.0)]
    assert total_length_mm(straight_line) == 850.0

    # 3-4-5 right triangle segment
    triangle = [(0.0, 0.0, 0.0), (300.0, 400.0, 0.0)]
    assert total_length_mm(triangle) == 500.0


def test_mirrored_anchors_symmetric_across_midline():
    """Verify left and right anchors are exact mirror reflections across x = 0."""
    body = PRESETS["average"]
    for d_id in DERMATOME_ANCHORS:
        pt_l = skin_anchor(body, d_id, side="l")
        pt_r = skin_anchor(body, d_id, side="r")

        # x must be reflected across x=0
        assert math.isclose(pt_l[0], -pt_r[0], abs_tol=1e-2), f"{d_id} x-symmetry failed"
        # y and z must be identical
        assert math.isclose(pt_l[1], pt_r[1], abs_tol=1e-2), f"{d_id} y-symmetry failed"
        assert math.isclose(pt_l[2], pt_r[2], abs_tol=1e-2), f"{d_id} z-symmetry failed"


def test_nearest_dermatome_right_foot_dorsum():
    """Verify nearest_dermatome identifies L5 on the right side for a point on the right foot."""
    body = PRESETS["average"]
    # Anchor of L5 on right foot dorsum
    anchor_l5_r = skin_anchor(body, "L5", side="r")

    # Add minor query perturbation (+5 mm in anterior y)
    query_pt = (anchor_l5_r[0], anchor_l5_r[1] + 5.0, anchor_l5_r[2])
    d_id, side, dist = nearest_dermatome(body, query_pt)

    assert d_id == "L5"
    assert side == "r"
    assert math.isclose(dist, 5.0, abs_tol=1e-1)


def test_taller_body_strictly_longer_path_and_latency():
    """Verify path length and latency scale with stature, approximately matching height ratio."""
    short = PRESETS["short"]  # 1500 mm
    tall = PRESETS["tall"]  # 1950 mm

    time_short = signal_timeline(short, "L5", side="r")
    time_tall = signal_timeline(tall, "L5", side="r")

    assert tall.height_mm > short.height_mm
    assert time_tall.total_path_length_mm > time_short.total_path_length_mm

    expected_ratio = tall.height_mm / short.height_mm  # 1950 / 1500 = 1.30
    measured_ratio = time_tall.total_path_length_mm / time_short.total_path_length_mm
    # Path includes peripheral leg + spinal ascent; should be within 10% of stature ratio
    assert math.isclose(measured_ratio, expected_ratio, rel_tol=0.10)

    # Latencies must also be strictly greater
    for f_type in [FiberType.A_BETA, FiberType.A_DELTA, FiberType.C]:
        lat_short = time_short.fibers[f_type].latency_ms
        lat_tall = time_tall.fibers[f_type].latency_ms
        assert lat_tall > lat_short


def test_fiber_latencies_ordering():
    """Verify A_beta has shortest latency, A_delta intermediate, and C longest."""
    body = PRESETS["average"]
    timeline = signal_timeline(body, "L5", side="r")

    lat_ab = timeline.fibers[FiberType.A_BETA].latency_ms
    lat_ad = timeline.fibers[FiberType.A_DELTA].latency_ms
    lat_c = timeline.fibers[FiberType.C].latency_ms

    assert lat_ab < lat_ad < lat_c
    # Conduction velocities: A_beta 50 m/s, A_delta 15 m/s, C 1 m/s
    # Latency of C should be roughly 50x that of A_beta
    ratio_c_ab = lat_c / lat_ab
    assert math.isclose(ratio_c_ab, 50.0, rel_tol=0.05)


def test_spinal_cord_levels_ordering_and_polyline():
    """Verify spinal cord levels are properly distributed and monotonic in Z."""
    body = PRESETS["average"]
    levels = spinal_cord_levels(body)
    assert len(levels) == 30

    cord_line = spinal_cord_polyline(body)
    assert len(cord_line) == 30

    # Ensure strictly increasing in z from cord_S5 to cord_C1
    for i in range(len(cord_line) - 1):
        assert cord_line[i][2] < cord_line[i + 1][2]


def test_nerve_and_full_pathway():
    """Verify nerve polyline and full labelled pathway connectivity."""
    body = PRESETS["average"]
    nerve_pts = nerve_polyline(body, "C6", side="l")
    assert len(nerve_pts) >= 4  # anchor, wrist, elbow, shoulder, cord_C6

    full_path = full_pathway_polyline(body, "C6", side="l")
    labels = [label for label, _ in full_path]
    assert labels[0].startswith("skin_anchor_C6_l")
    assert labels[-1] == "cortex_S1"
    assert "cord_C6" in labels
    assert "brainstem" in labels
    assert "thalamus" in labels
