"""Synthetic phantom generator for geometric testing and clearance validation.

Generates spheres, ellipsoids, and multi-structure cases in physical space (mm)
under internal (z, y, x) NumPy array conventions.
"""

import math
from typing import Any

import numpy as np

from hdt.core.case import Case
from hdt.core.structure import Structure


def make_sphere(
    shape_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    center_mm_zyx: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    """Generate a 3D boolean spherical mask in physical coordinates.

    Voxel centres are computed as: physical_coord = voxel_index * spacing.
    This guarantees correct spherical geometry under anisotropic voxel spacing.

    Args:
        shape_zyx: Volume dimensions (z, y, x) in voxels.
        spacing_zyx: Voxel spacing (spacing_z, spacing_y, spacing_x) in mm.
        center_mm_zyx: Sphere centre (cz, cy, cx) in mm physical coordinates.
        radius_mm: Sphere radius in mm.

    Returns:
        3D boolean NumPy array in (z, y, x) order.
    """
    if radius_mm <= 0.0:
        return np.zeros(shape_zyx, dtype=bool)

    z = np.arange(shape_zyx[0], dtype=np.float64) * spacing_zyx[0]
    y = np.arange(shape_zyx[1], dtype=np.float64) * spacing_zyx[1]
    x = np.arange(shape_zyx[2], dtype=np.float64) * spacing_zyx[2]

    # Generate physical grid coordinate matrices matching (z, y, x) indexing
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")

    dist_sq = (
        (grid_z - center_mm_zyx[0]) ** 2
        + (grid_y - center_mm_zyx[1]) ** 2
        + (grid_x - center_mm_zyx[2]) ** 2
    )

    return dist_sq <= (radius_mm**2)


def make_ellipsoid(
    shape_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    center_mm_zyx: tuple[float, float, float],
    radii_mm_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Generate a 3D boolean ellipsoidal mask in physical coordinates.

    Voxel centres are computed as: physical_coord = voxel_index * spacing.

    Args:
        shape_zyx: Volume dimensions (z, y, x) in voxels.
        spacing_zyx: Voxel spacing (spacing_z, spacing_y, spacing_x) in mm.
        center_mm_zyx: Ellipsoid centre (cz, cy, cx) in mm physical coordinates.
        radii_mm_zyx: Semi-principal axes (rz, ry, rx) in mm.

    Returns:
        3D boolean NumPy array in (z, y, x) order.
    """
    if any(r <= 0.0 for r in radii_mm_zyx):
        return np.zeros(shape_zyx, dtype=bool)

    z = np.arange(shape_zyx[0], dtype=np.float64) * spacing_zyx[0]
    y = np.arange(shape_zyx[1], dtype=np.float64) * spacing_zyx[1]
    x = np.arange(shape_zyx[2], dtype=np.float64) * spacing_zyx[2]

    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")

    normalized_sq = (
        ((grid_z - center_mm_zyx[0]) / radii_mm_zyx[0]) ** 2
        + ((grid_y - center_mm_zyx[1]) / radii_mm_zyx[1]) ** 2
        + ((grid_x - center_mm_zyx[2]) / radii_mm_zyx[2]) ** 2
    )

    return normalized_sq <= 1.0


def make_phantom_case(
    case_id: str = "phantom_001",
    shape_zyx: tuple[int, int, int] = (64, 64, 64),
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: np.ndarray | None = None,
    tumor_center_mm: tuple[float, float, float] = (32.0, 32.0, 24.0),
    tumor_radius_mm: float = 6.0,
    vessel_center_mm: tuple[float, float, float] = (32.0, 32.0, 40.0),
    vessel_radius_mm: float = 4.0,
) -> tuple[Case, dict[str, Any]]:
    """Create a synthetic phantom Case with target ('tumor') and risk ('vessel') structures.

    Calculates the analytic surface clearance between the two spheres and returns it in
    a separate dictionary (the clearance is NOT stored inside the Case object).

    Args:
        case_id: Opaque non-patient identifier.
        shape_zyx: Volume dimensions (z, y, x) in voxels.
        spacing_mm: Voxel spacing (sz, sy, sx) in mm.
        origin_mm: Physical origin (oz, oy, ox) in mm.
        direction: 3x3 direction matrix in standard ITK space. Defaults to identity matrix.
        tumor_center_mm: Target sphere centre (cz, cy, cx) in mm.
        tumor_radius_mm: Target sphere radius in mm.
        vessel_center_mm: Risk sphere centre (cz, cy, cx) in mm.
        vessel_radius_mm: Risk sphere radius in mm.

    Returns:
        tuple containing:
            - case: Populated Case instance containing 'tumor' (target) and 'vessel' (risk).
            - metrics: Dictionary containing analytic geometric values:
                * 'center_distance_mm': Euclidean distance between sphere centres.
                * 'analytic_clearance_mm': Exact surface-to-surface clearance in mm
                  (0.0 if touching or overlapping).
                * 'tumor_radius_mm': Target sphere radius in mm.
                * 'vessel_radius_mm': Risk sphere radius in mm.
    """
    if direction is None:
        direction = np.eye(3, dtype=np.float64)

    # Generate spherical masks in physical space
    tumor_mask = make_sphere(
        shape_zyx=shape_zyx,
        spacing_zyx=spacing_mm,
        center_mm_zyx=tumor_center_mm,
        radius_mm=tumor_radius_mm,
    )

    vessel_mask = make_sphere(
        shape_zyx=shape_zyx,
        spacing_zyx=spacing_mm,
        center_mm_zyx=vessel_center_mm,
        radius_mm=vessel_radius_mm,
    )

    # Construct synthetic background image (float32)
    image = np.zeros(shape_zyx, dtype=np.float32)
    image[tumor_mask] = 100.0
    image[vessel_mask] = 200.0

    case = Case(
        case_id=case_id,
        images={"t1": image},
        spacing_mm=spacing_mm,
        origin_mm=origin_mm,
        direction=direction,
    )

    case.add_structure(Structure(name="tumor", mask=tumor_mask, role="target"))
    case.add_structure(Structure(name="vessel", mask=vessel_mask, role="risk"))

    # Compute analytic distance and surface clearance
    dz = tumor_center_mm[0] - vessel_center_mm[0]
    dy = tumor_center_mm[1] - vessel_center_mm[1]
    dx = tumor_center_mm[2] - vessel_center_mm[2]
    center_dist = math.sqrt(dz * dz + dy * dy + dx * dx)

    # If overlapping or touching, clearance is 0.0 mm
    analytic_clearance = max(0.0, center_dist - (tumor_radius_mm + vessel_radius_mm))

    metrics: dict[str, Any] = {
        "center_distance_mm": float(center_dist),
        "analytic_clearance_mm": float(analytic_clearance),
        "tumor_radius_mm": float(tumor_radius_mm),
        "vessel_radius_mm": float(vessel_radius_mm),
    }

    return case, metrics
