"""Anthropometric proportion table and preset generators for human skeletons.

Based on standard anthropometric reference data (Drillis & Contini; Winter)
normalized to standing stature H.
All coordinates in RAS millimetres.
"""

from hdt.body.models import BodyParams, JointName

# ---------------------------------------------------------------------------
# Anthropometric Proportions relative to Stature H [x_frac, y_frac, z_frac]
# ---------------------------------------------------------------------------
# Origin (0,0,0) is at the floor midway between feet.
# x: person's right (+x), person's left (-x)
# y: anterior (+y), posterior (-y)
# z: cranial/superior (+z), floor is 0.0
ANTHROPOMETRIC_PROPORTIONS: dict[JointName, tuple[float, float, float]] = {
    JointName.HEAD: (0.000, 0.000, 0.985),
    JointName.NECK: (0.000, 0.000, 0.860),
    JointName.CHEST: (0.000, 0.000, 0.730),
    JointName.PELVIS: (0.000, -0.010, 0.550),
    JointName.L_SHOULDER: (-0.115, 0.000, 0.810),
    JointName.R_SHOULDER: (0.115, 0.000, 0.810),
    JointName.L_ELBOW: (-0.145, 0.000, 0.630),
    JointName.R_ELBOW: (0.145, 0.000, 0.630),
    JointName.L_WRIST: (-0.155, 0.000, 0.485),
    JointName.R_WRIST: (0.155, 0.000, 0.485),
    JointName.L_HAND: (-0.160, 0.000, 0.395),
    JointName.R_HAND: (0.160, 0.000, 0.395),
    JointName.L_HIP: (-0.052, 0.000, 0.530),
    JointName.R_HIP: (0.052, 0.000, 0.530),
    JointName.L_KNEE: (-0.060, 0.000, 0.285),
    JointName.R_KNEE: (0.060, 0.000, 0.285),
    JointName.L_ANKLE: (-0.065, 0.000, 0.045),
    JointName.R_ANKLE: (0.065, 0.000, 0.045),
    JointName.L_FOOT: (-0.065, 0.040, 0.015),
    JointName.R_FOOT: (0.065, 0.040, 0.015),
}


def make_preset(height_mm: float) -> BodyParams:
    """Construct a geometrically valid BodyParams instance scaled to height_mm.

    Parameters
    ----------
    height_mm : float
        Total standing body stature in millimetres.

    Returns
    -------
    BodyParams
        Synthesized skeletal model with joints scaled linearly by height.
    """
    joints: dict[JointName, tuple[float, float, float]] = {
        j_name: (
            round(prop[0] * height_mm, 2),
            round(prop[1] * height_mm, 2),
            round(prop[2] * height_mm, 2),
        )
        for j_name, prop in ANTHROPOMETRIC_PROPORTIONS.items()
    }
    return BodyParams(
        height_mm=float(height_mm),
        joints=joints,
        source="preset",
    )


# Standard clinical and ergonomic stature presets
PRESETS: dict[str, BodyParams] = {
    "short": make_preset(1500.0),
    "average": make_preset(1750.0),
    "tall": make_preset(1950.0),
}


def get_preset(name: str) -> BodyParams | None:
    """Retrieve a pre-calculated stature preset by name ('short', 'average', 'tall')."""
    return PRESETS.get(name.lower())
