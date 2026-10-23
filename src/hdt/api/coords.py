"""Coordinate conversion utilities for external wire representations.

COORDINATE CONVENTION NOTE:
This module is the SINGLE AND ONLY PLACE in the API layer that converts internal
HDT coordinates for the wire.
- Internal HDT convention: NumPy axis order (z, y, x), voxel units, and LPS physical
  space (ITK standard).
- Wire / API convention: RAS millimetres (Right, Anterior, Superior; standard for NIfTI,
  3D Slicer, and web 3D viewers).
- Conversion between LPS and RAS: Negate the x and y axes. Because two axes are negated,
  the transformation matrix T = diag(-1, -1, 1) has determinant det(T) = +1, preserving
  spatial orientation and surface mesh triangle winding.

Every API response containing spatial coordinates must include "coordinate_system": "RAS".
"""

from typing import Any, Sequence

import numpy as np

from hdt.imaging.load_scan import ImageGeometry


def voxel_to_ras_mm(
    geom: ImageGeometry,
    index_zyx: tuple[float, float, float] | Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert continuous voxel index (z, y, x) to physical RAS coordinates (x, y, z) in mm.

    Args:
        geom: ImageGeometry containing voxel spacing, origin, and LPS direction cosine matrix.
        index_zyx: Continuous or discrete voxel index (z, y, x).

    Returns:
        (x, y, z) tuple in physical millimetres under the RAS coordinate convention.
    """
    lps = geom.voxel_to_lps_mm(index_zyx)
    return ImageGeometry.lps_to_ras(lps)


def origin_to_ras_mm(geom: ImageGeometry) -> tuple[float, float, float]:
    """Compute physical RAS origin (x, y, z) in mm corresponding to voxel index (0, 0, 0).

    Args:
        geom: ImageGeometry instance.

    Returns:
        (x, y, z) origin coordinates in RAS mm.
    """
    return voxel_to_ras_mm(geom, (0.0, 0.0, 0.0))


def direction_to_ras(geom: ImageGeometry) -> list[list[float]]:
    """Convert LPS direction cosine matrix (3x3) to RAS coordinate frame.

    In LPS physical space:
      p_LPS = origin_LPS + D_LPS @ scaled_xyz
    In RAS physical space:
      p_RAS = T @ p_LPS = T @ origin_LPS + (T @ D_LPS) @ scaled_xyz
    where T = diag(-1, -1, 1). Thus D_RAS = T @ D_LPS.
    Because det(T) = +1, det(D_RAS) = det(D_LPS) = +1.

    Args:
        geom: ImageGeometry instance.

    Returns:
        3x3 list of floats representing the direction cosine matrix in RAS space.
    """
    d_lps = np.array(geom.direction, dtype=np.float64).reshape((3, 3))
    # T = diag(-1, -1, 1), negates first two rows
    d_ras = np.array([
        [-d_lps[0, 0], -d_lps[0, 1], -d_lps[0, 2]],
        [-d_lps[1, 0], -d_lps[1, 1], -d_lps[1, 2]],
        [d_lps[2, 0], d_lps[2, 1], d_lps[2, 2]],
    ], dtype=np.float64)
    return [[float(val) for val in row] for row in d_ras]


def local_point_zyx_mm_to_ras_mm(
    arg1: Any,
    arg2: Any,
) -> tuple[float, float, float]:
    """Convert local volume point (z, y, x) in mm to physical RAS coordinates (x, y, z) in mm.

    Accepts either (geom, point_zyx_mm) or (point_zyx_mm, geom).

    `closest_points_mm` returns local (z, y, x) in mm, defined as:
        point_zyx_mm = (index_z * sz, index_y * sy, index_x * sx)
    To transform to physical RAS space:
      1. Divide by spacing_mm (z, y, x) to recover continuous voxel index (z, y, x).
      2. Map continuous voxel index to LPS physical space via voxel_to_lps_mm.
      3. Map LPS physical space to RAS physical space via lps_to_ras.

    Args:
        arg1: ImageGeometry or local point (z, y, x) in millimetres.
        arg2: Local point or ImageGeometry.

    Returns:
        (x, y, z) physical point in millimetres under the RAS coordinate convention.
    """
    if isinstance(arg1, ImageGeometry):
        geom: ImageGeometry = arg1
        point_zyx_mm = arg2
    elif isinstance(arg2, ImageGeometry):
        geom = arg2
        point_zyx_mm = arg1
    else:
        raise TypeError("One of the arguments must be an ImageGeometry instance.")

    sz, sy, sx = geom.spacing_mm
    index_zyx = (
        float(point_zyx_mm[0]) / sz,
        float(point_zyx_mm[1]) / sy,
        float(point_zyx_mm[2]) / sx,
    )
    return voxel_to_ras_mm(geom, index_zyx)


def transform_mesh_vertices_to_ras(
    geom: ImageGeometry,
    voxel_indices_zyx: np.ndarray,
) -> np.ndarray:
    """Vectorized conversion of continuous voxel indices (N, 3) in (z,y,x) to RAS mm (x,y,z).

    Args:
        geom: ImageGeometry instance.
        voxel_indices_zyx: (N, 3) continuous voxel indices in internal (z, y, x) order.

    Returns:
        NumPy array of shape (N, 3) containing (x, y, z) coordinates in physical RAS mm.
    """
    idx = np.asarray(voxel_indices_zyx, dtype=np.float64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(f"voxel_indices_zyx must have shape (N, 3), got {idx.shape}.")

    sp_x = float(geom.spacing_mm[2])
    sp_y = float(geom.spacing_mm[1])
    sp_z = float(geom.spacing_mm[0])

    origin_xyz = np.array(
        [float(geom.origin_mm[2]), float(geom.origin_mm[1]), float(geom.origin_mm[0])],
        dtype=np.float64,
    )

    # ITK continuous index order (x, y, z)
    scaled_xyz = np.column_stack([idx[:, 2] * sp_x, idx[:, 1] * sp_y, idx[:, 0] * sp_z])
    pts_lps = origin_xyz + scaled_xyz @ geom.direction.T
    pts_ras = np.column_stack([-pts_lps[:, 0], -pts_lps[:, 1], pts_lps[:, 2]])
    return pts_ras


def mesh_signed_volume_ras(vertices_ras: np.ndarray, faces: np.ndarray) -> float:
    """Compute the signed volume of a triangular surface mesh in RAS coordinates.

    Uses divergence theorem / discrete tetrahedra summation:
        Volume = 1/6 * sum_{(i, j, k) in faces} (v_i . (v_j x v_k))
    A positive signed volume indicates outward-pointing normal vectors (counter-clockwise
    triangle winding from the exterior).

    Args:
        vertices_ras: (N, 3) array of vertices in RAS millimetres.
        faces: (M, 3) array of 0-based vertex index triplets.

    Returns:
        float: Signed volume in mm^3.
    """
    v = np.asarray(vertices_ras, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    cross_prod = np.cross(v1, v2)
    signed_vol = float(np.sum(np.einsum("ij,ij->i", v0, cross_prod)) / 6.0)
    return signed_vol
