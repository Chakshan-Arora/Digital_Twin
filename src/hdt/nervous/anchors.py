"""Dermatome skin surface anchor definitions and nearest-dermatome picker.

All coordinates are in RAS millimetres.
Origin (0,0,0) on floor midway between feet.
x: person's right (+x), left (-x)
y: anterior (+y), posterior (-y)
z: superior (+z)
"""

import math
from dataclasses import dataclass
from typing import Literal

from hdt.body.models import BodyParams, JointName


@dataclass(frozen=True)
class LimbAnchorDef:
    """Surface skin anchor defined relative to a skeletal bone segment."""

    joint_a_stem: str  # e.g. "shoulder", "elbow", "hip", "knee", "ankle"
    joint_b_stem: str
    t: float  # interpolation parameter along [joint_a, joint_b], 0.0 to 1.0
    normal_dir: tuple[float, float, float]  # unit offset direction in right-body frame
    radius_fraction_h: float  # offset distance as fraction of standing height H


@dataclass(frozen=True)
class TorsoAnchorDef:
    """Surface skin anchor defined for a horizontal torso band."""

    z_fraction_h: float  # height above floor as fraction of standing height H
    normal_dir: tuple[float, float, float]  # unit offset direction in right-body frame
    radius_fraction_h: float  # torso radius as fraction of standing height H


# Canonical dermatome anchor definition table (right side representation)
# Anchored to standard ASIA / Netter key sensory points
DERMATOME_ANCHORS: dict[str, LimbAnchorDef | TorsoAnchorDef] = {
    # -----------------------------------------------------------------------
    # Cervical (C2 - C8)
    # -----------------------------------------------------------------------
    "C2": TorsoAnchorDef(
        z_fraction_h=0.950,
        normal_dir=(0.0, -1.0, 0.2),  # Occiput / posterior scalp
        radius_fraction_h=0.065,
    ),
    "C3": TorsoAnchorDef(
        z_fraction_h=0.865,
        normal_dir=(0.5, 0.5, 0.0),  # Supraclavicular fossa / high neck
        radius_fraction_h=0.060,
    ),
    "C4": TorsoAnchorDef(
        z_fraction_h=0.825,
        normal_dir=(0.7, 0.0, 0.7),  # Acromioclavicular summit
        radius_fraction_h=0.080,
    ),
    "C5": LimbAnchorDef(
        joint_a_stem="shoulder",
        joint_b_stem="elbow",
        t=0.35,
        normal_dir=(1.0, 0.0, 0.0),  # Lateral upper arm over deltoid
        radius_fraction_h=0.028,
    ),
    "C6": LimbAnchorDef(
        joint_a_stem="elbow",
        joint_b_stem="wrist",
        t=0.85,
        normal_dir=(1.0, 0.3, 0.0),  # Lateral forearm and base of thumb
        radius_fraction_h=0.022,
    ),
    "C7": LimbAnchorDef(
        joint_a_stem="wrist",
        joint_b_stem="hand",
        t=0.90,
        normal_dir=(0.0, 0.0, -1.0),  # Middle finger
        radius_fraction_h=0.015,
    ),
    "C8": LimbAnchorDef(
        joint_a_stem="elbow",
        joint_b_stem="wrist",
        t=0.85,
        normal_dir=(-1.0, -0.2, 0.0),  # Medial forearm / hypothenar little finger
        radius_fraction_h=0.022,
    ),
    # -----------------------------------------------------------------------
    # Thoracic (T1 - T12)
    # -----------------------------------------------------------------------
    "T1": LimbAnchorDef(
        joint_a_stem="shoulder",
        joint_b_stem="elbow",
        t=0.80,
        normal_dir=(-1.0, 0.0, 0.0),  # Medial upper arm proximal to medial epicondyle
        radius_fraction_h=0.025,
    ),
    "T2": TorsoAnchorDef(
        z_fraction_h=0.815,
        normal_dir=(0.4, 0.8, 0.0),  # Upper chest / axilla
        radius_fraction_h=0.095,
    ),
    "T3": TorsoAnchorDef(
        z_fraction_h=0.790,
        normal_dir=(0.3, 0.9, 0.0),  # Third intercostal space upper thorax
        radius_fraction_h=0.095,
    ),
    "T4": TorsoAnchorDef(
        z_fraction_h=0.765,
        normal_dir=(0.3, 0.95, 0.0),  # Nipple line across fourth intercostal space
        radius_fraction_h=0.098,
    ),
    "T5": TorsoAnchorDef(
        z_fraction_h=0.735,
        normal_dir=(0.3, 0.95, 0.0),  # Inframammary chest
        radius_fraction_h=0.096,
    ),
    "T6": TorsoAnchorDef(
        z_fraction_h=0.705,
        normal_dir=(0.2, 0.98, 0.0),  # Xiphoid process level
        radius_fraction_h=0.094,
    ),
    "T7": TorsoAnchorDef(
        z_fraction_h=0.675,
        normal_dir=(0.2, 0.98, 0.0),  # Subxiphoid epigastric band
        radius_fraction_h=0.092,
    ),
    "T8": TorsoAnchorDef(
        z_fraction_h=0.645,
        normal_dir=(0.25, 0.97, 0.0),  # Costal arch upper abdomen
        radius_fraction_h=0.090,
    ),
    "T9": TorsoAnchorDef(
        z_fraction_h=0.615,
        normal_dir=(0.25, 0.97, 0.0),  # Mid-abdominal band supraumbilical
        radius_fraction_h=0.088,
    ),
    "T10": TorsoAnchorDef(
        z_fraction_h=0.585,
        normal_dir=(0.2, 0.98, 0.0),  # Umbilicus level
        radius_fraction_h=0.088,
    ),
    "T11": TorsoAnchorDef(
        z_fraction_h=0.555,
        normal_dir=(0.25, 0.97, 0.0),  # Infraumbilical lower abdomen
        radius_fraction_h=0.090,
    ),
    "T12": TorsoAnchorDef(
        z_fraction_h=0.530,
        normal_dir=(0.35, 0.93, 0.0),  # Inguinal ligament suprapubic crease
        radius_fraction_h=0.092,
    ),
    # -----------------------------------------------------------------------
    # Lumbar (L1 - L5)
    # -----------------------------------------------------------------------
    "L1": LimbAnchorDef(
        joint_a_stem="hip",
        joint_b_stem="knee",
        t=0.08,
        normal_dir=(0.2, 0.95, 0.0),  # Groin and upper anterior thigh
        radius_fraction_h=0.038,
    ),
    "L2": LimbAnchorDef(
        joint_a_stem="hip",
        joint_b_stem="knee",
        t=0.35,
        normal_dir=(0.2, 0.98, 0.0),  # Anterior mid-thigh
        radius_fraction_h=0.036,
    ),
    "L3": LimbAnchorDef(
        joint_a_stem="hip",
        joint_b_stem="knee",
        t=0.88,
        normal_dir=(0.1, 0.98, 0.0),  # Knee / medial femoral condyle
        radius_fraction_h=0.032,
    ),
    "L4": LimbAnchorDef(
        joint_a_stem="knee",
        joint_b_stem="ankle",
        t=0.75,
        normal_dir=(-0.95, 0.1, 0.0),  # Medial lower leg and medial malleolus
        radius_fraction_h=0.024,
    ),
    "L5": LimbAnchorDef(
        joint_a_stem="knee",
        joint_b_stem="ankle",
        t=0.65,
        normal_dir=(0.95, 0.2, 0.0),  # Lateral lower leg and dorsum of foot
        radius_fraction_h=0.026,
    ),
    # -----------------------------------------------------------------------
    # Sacral (S1 - S5)
    # -----------------------------------------------------------------------
    "S1": LimbAnchorDef(
        joint_a_stem="ankle",
        joint_b_stem="foot",
        t=0.70,
        normal_dir=(0.9, -0.3, -0.3),  # Lateral foot, lateral heel, sole
        radius_fraction_h=0.022,
    ),
    "S2": LimbAnchorDef(
        joint_a_stem="hip",
        joint_b_stem="knee",
        t=0.55,
        normal_dir=(0.0, -1.0, 0.0),  # Midline posterior thigh / popliteal fossa
        radius_fraction_h=0.036,
    ),
    "S3": TorsoAnchorDef(
        z_fraction_h=0.520,
        normal_dir=(0.3, -0.95, 0.0),  # Medial infragluteal crease / ischial region
        radius_fraction_h=0.090,
    ),
    "S4": TorsoAnchorDef(
        z_fraction_h=0.505,
        normal_dir=(0.1, -0.7, -0.7),  # Perineal saddle region
        radius_fraction_h=0.060,
    ),
    "S5": TorsoAnchorDef(
        z_fraction_h=0.495,
        normal_dir=(0.0, -0.5, -0.8),  # Innermost perianal verge
        radius_fraction_h=0.040,
    ),
}


def skin_anchor(
    body: BodyParams, dermatome_id: str, side: Literal["l", "r", "both"] = "r"
) -> tuple[float, float, float]:
    """Calculate the 3D surface point of a dermatome skin anchor in RAS millimetres.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal stature and joint positions.
    dermatome_id : str
        Dermatome identifier (e.g. 'L5', 'C6', 'T10').
    side : {'l', 'r', 'both'}
        Anatomical side: 'l' (person's left, x < 0) or 'r' (person's right, x > 0).

    Returns
    -------
    tuple[float, float, float]
        3D spatial coordinate [x, y, z] in mm on the skin surface.
    """
    key = dermatome_id.upper()
    if key not in DERMATOME_ANCHORS:
        raise KeyError(f"Dermatome anchor '{dermatome_id}' not found in atlas.")

    target_side = "r" if side == "both" else side.lower()
    is_left = target_side == "l"
    h = body.height_mm
    anchor_def = DERMATOME_ANCHORS[key]

    if isinstance(anchor_def, LimbAnchorDef):
        prefix = "l_" if is_left else "r_"
        j_a_name = JointName(f"{prefix}{anchor_def.joint_a_stem}")
        j_b_name = JointName(f"{prefix}{anchor_def.joint_b_stem}")
        pos_a = body.joints[j_a_name]
        pos_b = body.joints[j_b_name]

        # Interpolated skeletal axis position
        t = anchor_def.t
        axis_x = pos_a[0] + t * (pos_b[0] - pos_a[0])
        axis_y = pos_a[1] + t * (pos_b[1] - pos_a[1])
        axis_z = pos_a[2] + t * (pos_b[2] - pos_a[2])

        # Surface displacement offset
        offset_dist = anchor_def.radius_fraction_h * h
        dir_x, dir_y, dir_z = anchor_def.normal_dir
        # Normalize direction
        norm = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z) or 1.0
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm

        # Mirror lateral X component for left side
        if is_left:
            dir_x = -dir_x

        skin_x = axis_x + dir_x * offset_dist
        skin_y = axis_y + dir_y * offset_dist
        skin_z = axis_z + dir_z * offset_dist

    else:
        # TorsoAnchorDef
        z_pos = anchor_def.z_fraction_h * h
        offset_dist = anchor_def.radius_fraction_h * h
        dir_x, dir_y, dir_z = anchor_def.normal_dir
        norm = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z) or 1.0
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm

        if is_left:
            dir_x = -dir_x

        # Torso center at height z_pos
        torso_y = body.joints[JointName.CHEST][1] * 0.5 + body.joints[JointName.PELVIS][1] * 0.5
        skin_x = dir_x * offset_dist
        skin_y = torso_y + dir_y * offset_dist
        skin_z = z_pos + dir_z * offset_dist

    return (round(skin_x, 2), round(skin_y, 2), round(skin_z, 2))


def skin_samples(
    body: BodyParams,
    dermatome_id: str,
    side: Literal["l", "r", "both"] = "r",
    n: int = 6,
) -> list[tuple[float, float, float]]:
    """Generate dense surface skin samples for a dermatome's territory in RAS mm.

    The first sample is always the canonical skin_anchor point. Additional
    samples span circumferential arcs or longitudinal bands across the segment.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal stature and joint positions.
    dermatome_id : str
        Dermatome identifier (e.g. 'L5', 'C6', 'T10').
    side : {'l', 'r', 'both'}
        Anatomical side: 'l', 'r', or 'both'.
    n : int
        Number of skin samples to generate per side (minimum 2).

    Returns
    -------
    list[tuple[float, float, float]]
        List of 3D coordinates [x, y, z] in RAS millimetres.
    """
    key = dermatome_id.upper()
    if key not in DERMATOME_ANCHORS:
        raise KeyError(f"Dermatome anchor '{dermatome_id}' not found in atlas.")

    if side == "both":
        return skin_samples(body, dermatome_id, side="r", n=n) + skin_samples(
            body, dermatome_id, side="l", n=n
        )

    target_side: Literal["l", "r"] = "l" if side.lower() == "l" else "r"
    is_left = target_side == "l"
    h = body.height_mm
    anchor_def = DERMATOME_ANCHORS[key]

    # Sample 0 is always the canonical anchor
    canonical = skin_anchor(body, key, side=target_side)
    samples: list[tuple[float, float, float]] = [canonical]

    if n <= 1:
        return samples

    if isinstance(anchor_def, LimbAnchorDef):
        prefix = "l_" if is_left else "r_"
        j_a_name = JointName(f"{prefix}{anchor_def.joint_a_stem}")
        j_b_name = JointName(f"{prefix}{anchor_def.joint_b_stem}")
        pos_a = body.joints[j_a_name]
        pos_b = body.joints[j_b_name]

        t_base = anchor_def.t
        dt_list = [-0.08, -0.04, 0.04, 0.08, 0.12, -0.12, 0.16]
        dir_x, dir_y, dir_z = anchor_def.normal_dir
        norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2) or 1.0
        dir_x, dir_y, dir_z = dir_x / norm, dir_y / norm, dir_z / norm
        if is_left:
            dir_x = -dir_x
        offset_dist = anchor_def.radius_fraction_h * h

        for dt in dt_list[: n - 1]:
            t = max(0.02, min(0.98, t_base + dt))
            ax = pos_a[0] + t * (pos_b[0] - pos_a[0])
            ay = pos_a[1] + t * (pos_b[1] - pos_a[1])
            az = pos_a[2] + t * (pos_b[2] - pos_a[2])
            samples.append(
                (
                    round(ax + dir_x * offset_dist, 2),
                    round(ay + dir_y * offset_dist, 2),
                    round(az + dir_z * offset_dist, 2),
                )
            )
    else:
        # TorsoAnchorDef
        z_base = anchor_def.z_fraction_h * h
        offset_dist = anchor_def.radius_fraction_h * h
        torso_y = (
            body.joints[JointName.CHEST][1] * 0.5 + body.joints[JointName.PELVIS][1] * 0.5
        )
        dir_x, dir_y, dir_z = anchor_def.normal_dir
        norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2) or 1.0
        dir_x, dir_y, dir_z = dir_x / norm, dir_y / norm, dir_z / norm
        base_angle = math.atan2(dir_y, abs(dir_x))

        angles = []
        for step in [-0.4, -0.2, 0.2, 0.4, 0.6, -0.6, 0.8]:
            ang = base_angle + step
            ang = max(-1.3, min(1.3, ang))
            angles.append(ang)

        for ang in angles[: n - 1]:
            dx = math.cos(ang)
            dy = math.sin(ang)
            x_val = -dx * offset_dist if is_left else dx * offset_dist
            if abs(x_val) < 2.0:
                x_val = -2.0 if is_left else 2.0
            sy = torso_y + dy * offset_dist
            sz = z_base + dir_z * offset_dist
            samples.append((round(x_val, 2), round(sy, 2), round(sz, 2)))

    # Preserve canonical coordinate at index 0, clamp sign of x for auxiliary samples
    res: list[tuple[float, float, float]] = [samples[0]]
    for s in samples[1:]:
        sx, sy, sz = s
        if is_left and sx >= 0.0:
            sx = -1.0
        elif not is_left and sx <= 0.0:
            sx = 1.0
        res.append((sx, sy, sz))

    return res


def dermatome_samples_all(
    body: BodyParams, n: int = 6
) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    """Return surface sample sets for all dermatomes across both body sides.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model.
    n : int
        Number of surface samples per dermatome per side (default 6).

    Returns
    -------
    dict[str, dict[str, list[tuple[float, float, float]]]]
        Mapping of {dermatome_id: {'l': [samples], 'r': [samples]}}.
    """
    return {
        d_id: {
            "l": skin_samples(body, d_id, side="l", n=n),
            "r": skin_samples(body, d_id, side="r", n=n),
        }
        for d_id in DERMATOME_ANCHORS
    }


def nearest_dermatome(
    body: BodyParams, point_ras: tuple[float, float, float]
) -> tuple[str, str, float]:
    """Find the closest dermatome anchor to a query 3D point.

    Evaluates query distance across all dense surface samples belonging to
    the targeted hemisphere, identifying the closest matching dermatome.

    Side is determined directly by the sign of x:
      - x < 0: person's left ('l')
      - x >= 0: person's right ('r')

    Parameters
    ----------
    body : BodyParams
        Skeletal model defining anchor positions.
    point_ras : tuple[float, float, float]
        Target 3D coordinate [x, y, z] in RAS millimetres.

    Returns
    -------
    tuple[str, str, float]
        (dermatome_id, side, distance_mm) where distance_mm is Euclidean distance
        to the nearest skin anchor.
    """
    side: Literal["l", "r"] = "l" if point_ras[0] < 0.0 else "r"
    best_id: str = "L5"
    min_dist: float = float("inf")

    px, py, pz = point_ras

    for d_id in DERMATOME_ANCHORS:
        samples = skin_samples(body, d_id, side=side, n=6)
        for sx, sy, sz in samples:
            dx = px - sx
            dy = py - sy
            dz = pz - sz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < min_dist:
                min_dist = dist
                best_id = d_id

    return (best_id, side, round(min_dist, 2))
