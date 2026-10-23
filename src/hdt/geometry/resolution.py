"""Voxel grid resolution and directional extent along closest voxel pairs."""

from typing import Any, Sequence

import numpy as np


def compute_resolution_block(
    spacing_mm: Sequence[float],
    point_a_zyx_mm: Sequence[float] | np.ndarray,
    point_b_zyx_mm: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Compute voxel grid resolution, diagonal, and directional extent (2t term).

    Derivation:
        For an arbitrary unit separation direction u = (pa - pb) / ||pa - pb||,
        the voxel half-extent along u is:
            t = min_{i, u_i != 0} (spacing_i / (2 * |u_i|))
        The total voxel extent along the pair (the 2t term) converts center-to-center
        separation to face-to-face clearance:
            extent_2t = 2.0 * t

    Args:
        spacing_mm: Voxel spacing (sz, sy, sx) in millimetres.
        point_a_zyx_mm: Coordinates of closest point on structure A in local volume mm.
        point_b_zyx_mm: Coordinates of closest point on structure B in local volume mm.

    Returns:
        Dictionary with:
            - spacing_mm: list[float] (sz, sy, sx) in mm.
            - voxel_diagonal_mm: float (Euclidean norm of spacing) in mm.
            - voxel_extent_along_pair_mm: float (2t term) in mm.
    """
    sz, sy, sx = (float(s) for s in spacing_mm)
    sp = (sz, sy, sx)
    voxel_diagonal_mm = float(np.linalg.norm(sp))

    diff = np.array(point_a_zyx_mm, dtype=np.float64) - np.array(point_b_zyx_mm, dtype=np.float64)
    norm = float(np.linalg.norm(diff))

    if norm < 1e-12:
        voxel_extent_along_pair_mm = float(min(sp))
    else:
        u = np.abs(diff / norm)
        ratios = [sp[i] / (2.0 * u[i]) for i in range(3) if u[i] > 1e-9]
        t = min(ratios)
        voxel_extent_along_pair_mm = float(2.0 * t)

    return {
        "spacing_mm": [round(sz, 4), round(sy, 4), round(sx, 4)],
        "voxel_diagonal_mm": round(voxel_diagonal_mm, 4),
        "voxel_extent_along_pair_mm": round(voxel_extent_along_pair_mm, 4),
    }
