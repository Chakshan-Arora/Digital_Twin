"""Unit tests for body shape segments and signed surface distance evaluation."""

import math

from hdt.body.models import JointName
from hdt.body.presets import PRESETS
from hdt.body.shape import segments, surface_distance_mm


def test_segment_radii_scale_with_height():
    """Verify segment radii scale linearly with stature H."""
    short = PRESETS["short"]  # 1500 mm
    tall = PRESETS["tall"]  # 1950 mm

    segs_short = {s.name: s for s in segments(short)}
    segs_tall = {s.name: s for s in segments(tall)}

    assert len(segs_short) == 15
    assert len(segs_tall) == 15

    ratio = tall.height_mm / short.height_mm  # 1950 / 1500 = 1.30

    for name in segs_short:
        s1 = segs_short[name]
        s2 = segs_tall[name]

        assert math.isclose(s2.radius_mm_a / s1.radius_mm_a, ratio, rel_tol=1e-2)
        assert math.isclose(s2.radius_mm_b / s1.radius_mm_b, ratio, rel_tol=1e-2)

        if s1.half_width_mm is not None:
            assert s2.half_width_mm is not None
            assert math.isclose(s2.half_width_mm / s1.half_width_mm, ratio, rel_tol=1e-2)
            assert math.isclose(s2.half_depth_mm / s1.half_depth_mm, ratio, rel_tol=1e-2)


def test_torso_cross_section_wider_than_deep():
    """Verify torso elliptical cross section has half_width > half_depth."""
    body = PRESETS["average"]
    torso_seg = next(s for s in segments(body) if s.name == "torso")

    assert torso_seg.half_width_mm is not None
    assert torso_seg.half_depth_mm is not None
    assert torso_seg.half_width_mm > torso_seg.half_depth_mm


def test_surface_distance_signs_and_zeros():
    """Verify surface_distance_mm is negative inside, zero on surface, positive outside."""
    body = PRESETS["average"]
    head_joint = body.joints[JointName.HEAD]

    # Interior point (head joint center): should be deeply negative
    dist_inside = surface_distance_mm(body, head_joint)
    assert dist_inside < -30.0

    # Outside point far away (500 mm to the lateral side)
    query_far = (head_joint[0] + 500.0, head_joint[1], head_joint[2])
    dist_outside = surface_distance_mm(body, query_far)
    assert dist_outside > 300.0

    # Point on arm surface
    # Right shoulder: (0.115*H, 0, 0.810*H), upper arm radius ~0.028*H
    r_arm = next(s for s in segments(body) if s.name == "r_upper_arm")
    sh = body.joints[JointName.R_SHOULDER]
    surface_pt = (sh[0] + r_arm.radius_mm_a, sh[1], sh[2])
    dist_surface = surface_distance_mm(body, surface_pt)
    assert math.isclose(dist_surface, 0.0, abs_tol=5.0)
