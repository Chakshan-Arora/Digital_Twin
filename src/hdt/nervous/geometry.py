"""Procedural 3D nervous system geometry, spinal cord reconstruction, and pathway generation.

All coordinates are in RAS millimetres.
Origin (0,0,0) on floor midway between feet.
x: person's right (+x), left (-x)
y: anterior (+y), posterior (-y)
z: superior (+z)
"""

import math
from typing import Literal

from hdt.atlas.models import SPINAL_CORD_LEVELS
from hdt.body.models import BodyParams, JointName
from hdt.nervous.anchors import skin_anchor


def total_length_mm(polyline: list[tuple[float, float, float]]) -> float:
    """Calculate the cumulative 3D Euclidean length of a polyline in millimetres.

    Parameters
    ----------
    polyline : list[tuple[float, float, float]]
        Ordered sequence of 3D points [x, y, z] in mm.

    Returns
    -------
    float
        Sum of Euclidean lengths between adjacent points in mm.
    """
    if len(polyline) < 2:
        return 0.0

    total: float = 0.0
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)

    return round(total, 2)


def spinal_cord_levels(body: BodyParams) -> dict[str, tuple[float, float, float]]:
    """Calculate 3D spatial positions for all 30 spinal cord levels (cord_S5 to cord_C1).

    Anatomical Note
    ---------------
    In living humans, the adult spinal cord terminates at the conus medullaris
    near vertebral levels L1-L2, below which the lumbar and sacral nerve roots
    descend as the cauda equina. In this educational visual model, the 30 cord
    segments are distributed along the vertebral column from the sacrum/pelvis
    up to the craniocervical junction, with the sacral and lumbar segments
    represented in anatomical caudal-to-rostral order.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal stature and joint positions.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Mapping of spinal level name (e.g. 'cord_L5', 'cord_C6') to [x, y, z] in mm.
    """
    joints = body.joints
    pelvis = joints[JointName.PELVIS]
    chest = joints[JointName.CHEST]
    neck = joints[JointName.NECK]
    head = joints[JointName.HEAD]

    # Dorsal / posterior displacement offset behind the vertebral bodies
    post_offset_y = -15.0

    z_start = pelvis[2] + 10.0
    z_end = neck[2] + 0.15 * (head[2] - neck[2])

    levels_dict: dict[str, tuple[float, float, float]] = {}
    n_levels = len(SPINAL_CORD_LEVELS)

    for i, level_name in enumerate(SPINAL_CORD_LEVELS):
        # Parametric progression t from 0.0 (cord_S5) to 1.0 (cord_C1)
        t = i / (n_levels - 1)
        z = z_start + t * (z_end - z_start)

        # Bilateral X is centered at midline (0.0)
        x = 0.0

        # Interpolate anterior-posterior Y coordinate along spine curvature
        if t < 0.5:
            # Pelvis to chest
            tau = t / 0.5
            y = pelvis[1] + tau * (chest[1] - pelvis[1]) + post_offset_y
        else:
            # Chest to neck
            tau = (t - 0.5) / 0.5
            y = chest[1] + tau * (neck[1] - chest[1]) + post_offset_y

        levels_dict[level_name] = (round(x, 2), round(y, 2), round(z, 2))

    return levels_dict


def spinal_cord_polyline(body: BodyParams) -> list[tuple[float, float, float]]:
    """Generate the full ascending 30-level spinal cord polyline from cord_S5 to cord_C1.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model.

    Returns
    -------
    list[tuple[float, float, float]]
        Ordered list of 3D coordinates [x, y, z] in mm from caudal to rostral.
    """
    levels = spinal_cord_levels(body)
    return [levels[lvl] for lvl in SPINAL_CORD_LEVELS]


def brain_center(body: BodyParams) -> tuple[float, float, float]:
    """Calculate the center of the brain vault inside the cranium in mm."""
    joints = body.joints
    neck = joints[JointName.NECK]
    head = joints[JointName.HEAD]
    dz = head[2] - neck[2]
    return (0.0, round(head[1], 2), round(neck[2] + 0.55 * dz, 2))


def brainstem_position(body: BodyParams) -> tuple[float, float, float]:
    """Calculate 3D position of the brainstem / foramen magnum relay in mm."""
    joints = body.joints
    neck = joints[JointName.NECK]
    head = joints[JointName.HEAD]
    dz = head[2] - neck[2]
    return (0.0, round(head[1] - 0.05 * dz, 2), round(neck[2] + 0.20 * dz, 2))


def thalamus_position(body: BodyParams) -> tuple[float, float, float]:
    """Calculate 3D position of the central thalamic sensory relay in mm."""
    joints = body.joints
    neck = joints[JointName.NECK]
    head = joints[JointName.HEAD]
    dz = head[2] - neck[2]
    return (0.0, round(head[1], 2), round(neck[2] + 0.55 * dz, 2))


def cortex_s1_position(
    body: BodyParams, side: Literal["l", "r", "both"] = "r"
) -> tuple[float, float, float]:
    """Calculate 3D position of the primary somatosensory cortex (postcentral gyrus).

    Note
    ----
    Ascending somatosensory pathways decussate across the neuraxis. Peripheral
    sensations on the left side ('l') project to the contralateral right
    cerebral cortex (+x), and right-side sensations project to the left (-x).
    """
    joints = body.joints
    neck = joints[JointName.NECK]
    head = joints[JointName.HEAD]
    dz = head[2] - neck[2]

    # Contralateral hemisphere projection
    target_side = "r" if side == "both" else side.lower()
    x_offset = 0.25 * dz if target_side == "l" else -0.25 * dz
    y = head[1] - 0.05 * dz
    z = neck[2] + 0.85 * dz
    return (round(x_offset, 2), round(y, 2), round(z, 2))


def nerve_polyline(
    body: BodyParams, dermatome_id: str, side: Literal["l", "r", "both"] = "r"
) -> list[tuple[float, float, float]]:
    """Construct the peripheral nerve polyline from skin anchor to spinal cord entry.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model.
    dermatome_id : str
        Dermatome identifier (e.g. 'L5', 'C6', 'T10').
    side : {'l', 'r', 'both'}
        Anatomical stimulation side.

    Returns
    -------
    list[tuple[float, float, float]]
        Polyline [x, y, z] points in mm starting at skin anchor and terminating at cord root.
    """
    key = dermatome_id.upper()
    target_side = "r" if side == "both" else side.lower()
    prefix = "l_" if target_side == "l" else "r_"

    anchor_pt = skin_anchor(body, key, side=target_side)
    cord_levels_map = spinal_cord_levels(body)
    cord_key = f"cord_{key}"
    if cord_key not in cord_levels_map:
        raise KeyError(f"Spinal cord root '{cord_key}' not found.")
    root_pt = cord_levels_map[cord_key]

    joints = body.joints
    pts: list[tuple[float, float, float]] = [anchor_pt]

    # Route through limb skeleton based on segment category
    if key.startswith("C") or key == "T1":
        # Upper limb / neck
        if key in ("C6", "C7", "C8"):
            pts.append(joints[JointName(f"{prefix}wrist")])
            pts.append(joints[JointName(f"{prefix}elbow")])
            pts.append(joints[JointName(f"{prefix}shoulder")])
        elif key in ("C5", "T1"):
            pts.append(joints[JointName(f"{prefix}elbow")])
            pts.append(joints[JointName(f"{prefix}shoulder")])
        elif key == "C4":
            pts.append(joints[JointName(f"{prefix}shoulder")])

    elif key.startswith("T"):
        # Thoracic trunk
        # Route through anterior torso to spine level
        pass

    elif key.startswith("L") or key.startswith("S"):
        # Lower limb / pelvis
        if key in ("L4", "L5", "S1"):
            pts.append(joints[JointName(f"{prefix}ankle")])
            pts.append(joints[JointName(f"{prefix}knee")])
            pts.append(joints[JointName(f"{prefix}hip")])
        elif key in ("L2", "L3", "S2"):
            pts.append(joints[JointName(f"{prefix}knee")])
            pts.append(joints[JointName(f"{prefix}hip")])
        elif key == "L1":
            pts.append(joints[JointName(f"{prefix}hip")])

    pts.append(root_pt)
    return pts


def full_pathway_polyline(
    body: BodyParams, dermatome_id: str, side: Literal["l", "r", "both"] = "r"
) -> list[tuple[str, tuple[float, float, float]]]:
    """Generate the complete ascending sensory pathway from skin to S1 cortex with node labels.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model.
    dermatome_id : str
        Dermatome identifier (e.g. 'L5', 'C6', 'T10').
    side : {'l', 'r', 'both'}
        Anatomical stimulation side.

    Returns
    -------
    list[tuple[str, tuple[float, float, float]]]
        Labelled 3D points [(label, (x, y, z))] tracing peripheral nerve, entry root,
        ascending spinal cord segments, brainstem, thalamus, and cortex_S1.
    """
    key = dermatome_id.upper()
    target_side = "r" if side == "both" else side.lower()

    # 1. Peripheral segment
    peripheral_pts = nerve_polyline(body, key, side=target_side)
    labelled_path: list[tuple[str, tuple[float, float, float]]] = [
        (f"skin_anchor_{key}_{target_side}", peripheral_pts[0])
    ]

    for pt in peripheral_pts[1:-1]:
        labelled_path.append(("peripheral_nerve", pt))

    # 2. Spinal cord ascent
    cord_map = spinal_cord_levels(body)
    root_cord_name = f"cord_{key}"
    start_idx = SPINAL_CORD_LEVELS.index(root_cord_name)

    for lvl in SPINAL_CORD_LEVELS[start_idx:]:
        labelled_path.append((lvl, cord_map[lvl]))

    # 3. Supraspinal relays
    labelled_path.append(("brainstem", brainstem_position(body)))
    labelled_path.append(("thalamus", thalamus_position(body)))
    labelled_path.append(("cortex_S1", cortex_s1_position(body, side=target_side)))

    return labelled_path
