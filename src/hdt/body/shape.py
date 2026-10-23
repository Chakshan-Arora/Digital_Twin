"""Procedural 3D body shape model, segment radii, and surface distance evaluator.

All coordinates are in RAS millimetres.
Origin (0,0,0) on floor midway between feet.
x: person's right (+x), left (-x)
y: anterior (+y), posterior (-y)
z: superior (+z)
"""

import math

from pydantic import BaseModel, Field

from hdt.body.models import BodyParams, JointName


class BodySegment(BaseModel):
    """Geometric definition of a body segment connecting two skeletal joints."""

    name: str = Field(..., description="Segment identifier, e.g. 'torso', 'l_upper_arm'")
    joint_a: JointName = Field(..., description="Proximal skeletal joint anchor")
    joint_b: JointName = Field(..., description="Distal skeletal joint anchor")
    radius_mm_a: float = Field(..., ge=0.0, description="Cross-section radius at joint_a in mm")
    radius_mm_b: float = Field(..., ge=0.0, description="Cross-section radius at joint_b in mm")
    half_width_mm: float | None = Field(
        default=None,
        ge=0.0,
        description="Lateral half-width (x-axis) in mm for elliptical cross-sections (e.g. torso)",
    )
    half_depth_mm: float | None = Field(
        default=None,
        ge=0.0,
        description="Anteroposterior half-depth (y-axis) in mm for elliptical cross-sections",
    )


class ShapeModelResponse(BaseModel):
    """Response schema for body shape segments and dimensions."""

    height_mm: float = Field(..., description="Body stature in millimetres")
    segments: list[BodySegment] = Field(..., description="List of all articulated body segments")


# Proportional radii relative to standing stature H (Drillis & Contini; Winter)
# Scaled linearly with body height
SEGMENT_SPECS = [
    # Torso (elliptical cylinder chest <-> pelvis)
    {
        "name": "torso",
        "joint_a": JointName.CHEST,
        "joint_b": JointName.PELVIS,
        "r_frac_a": 0.110,
        "r_frac_b": 0.105,
        "half_w_frac": 0.120,
        "half_d_frac": 0.085,
    },
    # Neck (chest -> neck)
    {
        "name": "neck",
        "joint_a": JointName.CHEST,
        "joint_b": JointName.NECK,
        "r_frac_a": 0.040,
        "r_frac_b": 0.035,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    # Head (neck -> head)
    {
        "name": "head",
        "joint_a": JointName.NECK,
        "joint_b": JointName.HEAD,
        "r_frac_a": 0.055,
        "r_frac_b": 0.070,
        "half_w_frac": 0.065,
        "half_d_frac": 0.075,
    },
    # Upper extremities (left & right)
    {
        "name": "r_upper_arm",
        "joint_a": JointName.R_SHOULDER,
        "joint_b": JointName.R_ELBOW,
        "r_frac_a": 0.028,
        "r_frac_b": 0.022,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "l_upper_arm",
        "joint_a": JointName.L_SHOULDER,
        "joint_b": JointName.L_ELBOW,
        "r_frac_a": 0.028,
        "r_frac_b": 0.022,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "r_forearm",
        "joint_a": JointName.R_ELBOW,
        "joint_b": JointName.R_WRIST,
        "r_frac_a": 0.022,
        "r_frac_b": 0.016,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "l_forearm",
        "joint_a": JointName.L_ELBOW,
        "joint_b": JointName.L_WRIST,
        "r_frac_a": 0.022,
        "r_frac_b": 0.016,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "r_hand",
        "joint_a": JointName.R_WRIST,
        "joint_b": JointName.R_HAND,
        "r_frac_a": 0.016,
        "r_frac_b": 0.012,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "l_hand",
        "joint_a": JointName.L_WRIST,
        "joint_b": JointName.L_HAND,
        "r_frac_a": 0.016,
        "r_frac_b": 0.012,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    # Lower extremities (left & right)
    {
        "name": "r_thigh",
        "joint_a": JointName.R_HIP,
        "joint_b": JointName.R_KNEE,
        "r_frac_a": 0.045,
        "r_frac_b": 0.032,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "l_thigh",
        "joint_a": JointName.L_HIP,
        "joint_b": JointName.L_KNEE,
        "r_frac_a": 0.045,
        "r_frac_b": 0.032,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "r_shank",
        "joint_a": JointName.R_KNEE,
        "joint_b": JointName.R_ANKLE,
        "r_frac_a": 0.030,
        "r_frac_b": 0.020,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "l_shank",
        "joint_a": JointName.L_KNEE,
        "joint_b": JointName.L_ANKLE,
        "r_frac_a": 0.030,
        "r_frac_b": 0.020,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "r_foot",
        "joint_a": JointName.R_ANKLE,
        "joint_b": JointName.R_FOOT,
        "r_frac_a": 0.020,
        "r_frac_b": 0.018,
        "half_w_frac": None,
        "half_d_frac": None,
    },
    {
        "name": "l_foot",
        "joint_a": JointName.L_ANKLE,
        "joint_b": JointName.L_FOOT,
        "r_frac_a": 0.020,
        "r_frac_b": 0.018,
        "half_w_frac": None,
        "half_d_frac": None,
    },
]


def segments(body: BodyParams) -> list[BodySegment]:
    """Compute articulated body segments with anthropometrically scaled radii.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model containing joint positions in RAS mm.

    Returns
    -------
    list[BodySegment]
        Complete set of segments for head, neck, torso, and bilateral limbs.
    """
    h = body.height_mm
    res: list[BodySegment] = []

    for spec in SEGMENT_SPECS:
        hw = round(spec["half_w_frac"] * h, 2) if spec["half_w_frac"] is not None else None
        hd = round(spec["half_d_frac"] * h, 2) if spec["half_d_frac"] is not None else None
        res.append(
            BodySegment(
                name=spec["name"],
                joint_a=spec["joint_a"],
                joint_b=spec["joint_b"],
                radius_mm_a=round(spec["r_frac_a"] * h, 2),
                radius_mm_b=round(spec["r_frac_b"] * h, 2),
                half_width_mm=hw,
                half_depth_mm=hd,
            )
        )
    return res


def _segment_signed_distance(
    seg: BodySegment,
    body: BodyParams,
    point: tuple[float, float, float],
) -> float:
    """Compute signed distance from 3D point to a single segment surface in mm.

    Negative inside the segment volume, 0.0 on the boundary surface, positive outside.
    """
    pa = body.joints[seg.joint_a]
    pb = body.joints[seg.joint_b]
    px, py, pz = point

    vx = pb[0] - pa[0]
    vy = pb[1] - pa[1]
    vz = pb[2] - pa[2]
    len_sq = vx * vx + vy * vy + vz * vz

    if len_sq < 1e-6:
        # Degenerate segment: treat as sphere at pa
        dx = px - pa[0]
        dy = py - pa[1]
        dz = pz - pa[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz) - seg.radius_mm_a

    # Project point onto segment axis [0, 1]
    wx = px - pa[0]
    wy = py - pa[1]
    wz = pz - pa[2]
    dot = wx * vx + wy * vy + wz * vz
    t = max(0.0, min(1.0, dot / len_sq))

    proj_x = pa[0] + t * vx
    proj_y = pa[1] + t * vy
    proj_z = pa[2] + t * vz

    # Cross-sectional radius at projection parameter t
    r_t = (1.0 - t) * seg.radius_mm_a + t * seg.radius_mm_b

    # Distance in axial/normal plane
    dx = px - proj_x
    dy = py - proj_y
    dz = pz - proj_z

    if seg.half_width_mm is not None and seg.half_depth_mm is not None:
        # Elliptical cross section (e.g. torso or head)
        a = (1.0 - t) * seg.half_width_mm + t * seg.half_width_mm
        b = (1.0 - t) * seg.half_depth_mm + t * seg.half_depth_mm

        # Scaled elliptical radius rho
        rho = math.sqrt((dx / a) ** 2 + (dy / b) ** 2)
        if rho < 1e-6:
            # Right on axis
            radial_dist = -min(a, b)
        else:
            # Distance to elliptical boundary along ray
            norm_xy = math.sqrt(dx * dx + dy * dy)
            # Boundary radius at this ray angle:
            r_boundary = norm_xy / rho
            radial_dist = norm_xy - r_boundary

        # Include axial distance if point lies beyond ends of the segment
        if dot < 0.0:
            d_sq = (px - pa[0]) ** 2 + (py - pa[1]) ** 2 + (pz - pa[2]) ** 2
            axial_over = math.sqrt(d_sq) - min(a, b)
            return max(radial_dist, axial_over)
        elif dot > len_sq:
            d_sq = (px - pb[0]) ** 2 + (py - pb[1]) ** 2 + (pz - pb[2]) ** 2
            axial_over = math.sqrt(d_sq) - min(a, b)
            return max(radial_dist, axial_over)
        return radial_dist

    # Circular tapered cylinder / capsule
    axis_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return axis_dist - r_t


def surface_distance_mm(body: BodyParams, point: tuple[float, float, float]) -> float:
    """Compute signed distance from 3D query point to nearest body surface in mm.

    Negative inside the body envelope, 0.0 on the skin surface, positive outside.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model in RAS mm.
    point : tuple[float, float, float]
        Query 3D coordinate (x, y, z) in RAS mm.

    Returns
    -------
    float
        Minimum signed distance to the body envelope in millimetres.
    """
    all_segs = segments(body)
    min_dist = float("inf")

    for seg in all_segs:
        d = _segment_signed_distance(seg, body, point)
        if d < min_dist:
            min_dist = d

    return round(min_dist, 2)
