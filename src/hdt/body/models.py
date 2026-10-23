"""Body Explorer skeletal models, validation, and anthropometric presets."""

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Joint Names
# ---------------------------------------------------------------------------

JOINT_NAMES = (
    "head",
    "neck",
    "chest",
    "pelvis",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "l_foot",
    "r_foot",
)


class JointName(str, Enum):
    """Canonical 20 skeletal joints in the human body skeleton."""

    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    PELVIS = "pelvis"
    L_SHOULDER = "l_shoulder"
    R_SHOULDER = "r_shoulder"
    L_ELBOW = "l_elbow"
    R_ELBOW = "r_elbow"
    L_WRIST = "l_wrist"
    R_WRIST = "r_wrist"
    L_HAND = "l_hand"
    R_HAND = "r_hand"
    L_HIP = "l_hip"
    R_HIP = "r_hip"
    L_KNEE = "l_knee"
    R_KNEE = "r_knee"
    L_ANKLE = "l_ankle"
    R_ANKLE = "r_ankle"
    L_FOOT = "l_foot"
    R_FOOT = "r_foot"


class BodyParams(BaseModel):
    """Body Explorer 3D human stature and skeletal joint parameters.

    Coordinate system:
      - Units: Millimetres (mm).
      - Axes: RAS (x toward person's right, y anterior, z superior).
      - Origin (0, 0, 0): Situated on the floor plane directly between the feet.
      - Person stands erect facing +y.
      - "l_" corresponds to person's own left (x < 0).
      - "r_" corresponds to person's own right (x > 0).
    """

    height_mm: float = Field(
        ..., description="Standing body stature from floor to vertex in millimetres (mm)"
    )
    joints: dict[JointName, tuple[float, float, float]] = Field(
        ..., description="3D positions [x, y, z] in mm under RAS convention for all 20 joints"
    )
    source: Literal["scan", "preset"] = Field(
        ..., description="Origin source of the body parameters: camera scan vs preset"
    )


def _euclidean_dist(
    p1: tuple[float, float, float], p2: tuple[float, float, float]
) -> float:
    """Compute 3D Euclidean distance between two points in mm."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def validate_body_params(body: BodyParams) -> list[str]:
    """Validate BodyParams against anatomical plausibility and geometric invariants.

    Parameters
    ----------
    body : BodyParams
        Skeletal model to inspect.

    Returns
    -------
    list[str]
        A list of error descriptions. An empty list signifies a valid body model.
    """
    errors: list[str] = []

    # 1. Height range check [1000, 2300] mm
    if not math.isfinite(body.height_mm):
        errors.append(f"height_mm must be finite; got {body.height_mm}")
    elif body.height_mm < 1000.0 or body.height_mm > 2300.0:
        errors.append(
            f"height_mm ({body.height_mm:.1f} mm) is outside plausible range [1000, 2300] mm."
        )

    # 2. Check all 20 joints are present with finite numbers
    missing_joints: list[str] = []
    non_finite_joints: list[str] = []
    for j_name in JointName:
        if j_name not in body.joints:
            missing_joints.append(j_name.value)
            continue
        pt = body.joints[j_name]
        if len(pt) != 3 or not all(math.isfinite(c) for c in pt):
            non_finite_joints.append(j_name.value)

    if missing_joints:
        errors.append(f"Missing required joints: {', '.join(missing_joints)}.")
    if non_finite_joints:
        errors.append(f"Non-finite coordinates for joints: {', '.join(non_finite_joints)}.")

    if errors:
        return errors

    # Extract coordinates
    joints = body.joints

    # 3. Head must be the highest joint in Z
    head_z = joints[JointName.HEAD][2]
    for j_name, pt in joints.items():
        if j_name == JointName.HEAD:
            continue
        if pt[2] >= head_z:
            errors.append(
                f"Head (z={head_z:.1f} mm) is not highest joint; "
                f"'{j_name.value}' has z={pt[2]:.1f} mm."
            )

    # 4. Feet must be the lowest joints in Z
    foot_min_z = min(joints[JointName.L_FOOT][2], joints[JointName.R_FOOT][2])
    for j_name, pt in joints.items():
        if j_name in (JointName.L_FOOT, JointName.R_FOOT):
            continue
        if pt[2] < foot_min_z:
            errors.append(
                f"Feet are not lowest joints; '{j_name.value}' "
                f"(z={pt[2]:.1f} mm) is lower than foot."
            )

    # 5. Bilateral sign convention: left is -x, right is +x for shoulders and hips
    if joints[JointName.L_SHOULDER][0] >= 0.0:
        val = joints[JointName.L_SHOULDER][0]
        errors.append(
            f"l_shoulder must have negative x (person's left); got x={val:.1f} mm."
        )
    if joints[JointName.R_SHOULDER][0] <= 0.0:
        val = joints[JointName.R_SHOULDER][0]
        errors.append(
            f"r_shoulder must have positive x (person's right); got x={val:.1f} mm."
        )
    if joints[JointName.L_HIP][0] >= 0.0:
        val = joints[JointName.L_HIP][0]
        errors.append(
            f"l_hip must have negative x (person's left); got x={val:.1f} mm."
        )
    if joints[JointName.R_HIP][0] <= 0.0:
        val = joints[JointName.R_HIP][0]
        errors.append(
            f"r_hip must have positive x (person's right); got x={val:.1f} mm."
        )

    # 6. Plausible bone segment lengths
    # Upper leg (hip to knee): 250 - 650 mm
    l_upper_leg = _euclidean_dist(joints[JointName.L_HIP], joints[JointName.L_KNEE])
    r_upper_leg = _euclidean_dist(joints[JointName.R_HIP], joints[JointName.R_KNEE])
    if l_upper_leg < 250.0 or l_upper_leg > 650.0:
        errors.append(
            f"Left upper leg length ({l_upper_leg:.1f} mm) is outside "
            "plausible range [250, 650] mm."
        )
    if r_upper_leg < 250.0 or r_upper_leg > 650.0:
        errors.append(
            f"Right upper leg length ({r_upper_leg:.1f} mm) is outside "
            "plausible range [250, 650] mm."
        )

    # Upper arm (shoulder to elbow): 200 - 450 mm
    l_upper_arm = _euclidean_dist(joints[JointName.L_SHOULDER], joints[JointName.L_ELBOW])
    r_upper_arm = _euclidean_dist(joints[JointName.R_SHOULDER], joints[JointName.R_ELBOW])
    if l_upper_arm < 200.0 or l_upper_arm > 450.0:
        errors.append(
            f"Left upper arm length ({l_upper_arm:.1f} mm) is outside "
            "plausible range [200, 450] mm."
        )
    if r_upper_arm < 200.0 or r_upper_arm > 450.0:
        errors.append(
            f"Right upper arm length ({r_upper_arm:.1f} mm) is outside "
            "plausible range [200, 450] mm."
        )

    return errors
