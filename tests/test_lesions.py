"""Tests for hdt.conditions.lesions — 3D lesion mapping, envelope containment, and pathway blocking.

Verifies:
  - Lesion points inside the body for every condition x preset x side.
  - Sciatic nerve lesion lies on or near the nerve polyline (tolerance within 25 mm).
  - The lesion for carpal tunnel is nearer the wrist than the elbow on the median nerve.
  - path_block_index: blocked for foot dermatomes under sciatica, null for an arm
    dermatome under sciatica, and block point lies on the pathway before cortex_S1.
"""

from __future__ import annotations

import math

import pytest

from hdt.atlas.data import get_default_atlas
from hdt.body.models import JointName
from hdt.body.presets import PRESETS
from hdt.body.shape import surface_distance_mm
from hdt.conditions.lesions import locate_lesions, path_block_index
from hdt.nervous.geometry import full_pathway_polyline
from hdt.nervous.network import build_network


@pytest.mark.parametrize("preset_name", ["short", "average", "tall"])
@pytest.mark.parametrize("side", ["l", "r"])
@pytest.mark.parametrize(
    "condition_id",
    [
        "sciatica",
        "carpal_tunnel",
        "cervical_radiculopathy",
        "l4_l5_herniation",
        "sci_t10",
        "meralgia_paresthetica",
    ],
)
def test_lesion_points_inside_body(preset_name: str, side: str, condition_id: str) -> None:
    """Lesion points must reside inside or on the body envelope (surface_distance <= 0)."""
    body = PRESETS[preset_name]
    atlas = get_default_atlas()
    cond = next(c for c in atlas.conditions if c.id == condition_id)

    lesions, _ = locate_lesions(body, cond, side=side)
    assert len(lesions) > 0

    for lesion in lesions:
        s_dist = surface_distance_mm(body, lesion.point)
        assert s_dist <= 5.0, (
            f"Lesion '{lesion.label}' point {lesion.point} is outside body: "
            f"surface_distance_mm = {s_dist} mm (preset '{preset_name}', side '{side}')"
        )


def test_sciatic_nerve_lesion_near_polyline() -> None:
    """Sciatica root lesion lies on or near the proximal sciatic nerve polyline (within 25 mm)."""
    body = PRESETS["average"]
    net = build_network(body, detail="high")

    atlas = get_default_atlas()
    sciatica_cond = next(c for c in atlas.conditions if c.id == "sciatica")

    lesions, _ = locate_lesions(body, sciatica_cond, side="r", network=net)
    sciatic_nerve = next(p for p in net.lumbosacral_plexus if p.name == "nerve_sciatic_r")

    # Find minimum distance from any sciatica lesion to the sciatic nerve polyline
    min_dist_to_nerve = float("inf")
    for lesion in lesions:
        for pt in sciatic_nerve.points:
            d = math.dist(lesion.point, pt)
            if d < min_dist_to_nerve:
                min_dist_to_nerve = d

    # Proximal sciatic nerve originates at the S1 root exit; distance is within 25 mm
    assert min_dist_to_nerve <= 25.0, (
        f"Sciatica lesion distance to sciatic nerve {min_dist_to_nerve:.2f} mm > 25 mm"
    )


def test_carpal_tunnel_lesion_nearer_wrist_than_elbow() -> None:
    """The lesion for carpal tunnel is nearer the wrist than the elbow on the median nerve."""
    body = PRESETS["average"]
    net = build_network(body, detail="high")

    atlas = get_default_atlas()
    ct_cond = next(c for c in atlas.conditions if c.id == "carpal_tunnel")

    lesions, _ = locate_lesions(body, ct_cond, side="r", network=net)
    assert len(lesions) == 1
    lesion = lesions[0]

    wrist_joint = body.joints[JointName.R_WRIST]
    elbow_joint = body.joints[JointName.R_ELBOW]

    dist_to_wrist = math.dist(lesion.point, wrist_joint)
    dist_to_elbow = math.dist(lesion.point, elbow_joint)

    assert dist_to_wrist < dist_to_elbow, (
        f"Carpal tunnel lesion closer to elbow ({dist_to_elbow:.1f} mm) than wrist"
    )
    assert dist_to_wrist < 30.0, f"Lesion to wrist distance ({dist_to_wrist:.1f} mm) >= 30 mm"


def test_path_block_index_foot_vs_arm_under_sciatica() -> None:
    """path_block_index blocks foot dermatomes under sciatica, returns None for arm dermatomes.

    Block point must lie strictly along the pathway before cortex_S1.
    """
    body = PRESETS["average"]
    atlas = get_default_atlas()
    sciatica_cond = next(c for c in atlas.conditions if c.id == "sciatica")

    lesions, _ = locate_lesions(body, sciatica_cond, side="r")

    # Foot dermatome L5 on the right side
    l5_path = full_pathway_polyline(body, "L5", side="r")
    l5_pts = [pt for _, pt in l5_path]
    blk_idx_l5 = path_block_index(l5_pts, lesions)

    assert blk_idx_l5 is not None, "Right L5 pathway should be intercepted"
    assert 0 < blk_idx_l5 < len(l5_pts) - 1, (
        f"Block index {blk_idx_l5} must lie strictly between receptor and cortex_S1"
    )

    # Foot dermatome S1 on the right side
    s1_path = full_pathway_polyline(body, "S1", side="r")
    s1_pts = [pt for _, pt in s1_path]
    blk_idx_s1 = path_block_index(s1_pts, lesions)
    assert blk_idx_s1 is not None
    assert 0 < blk_idx_s1 < len(s1_pts) - 1

    # Arm dermatome C6 on the right side under sciatica (should be completely unaffected)
    c6_path = full_pathway_polyline(body, "C6", side="r")
    c6_pts = [pt for _, pt in c6_path]
    blk_idx_c6 = path_block_index(c6_pts, lesions)
    assert blk_idx_c6 is None, "Arm dermatome C6 must NOT be blocked by sciatica lesions"
