"""Point-in-prism continuous structure rasteriser for VS-SEG.

Discretises continuous polygonal prism structures defined on native series planes
directly onto a target voxel grid (e.g. native T2 grid) without post-rasterisation
resampling or intermediate volume interpolation.

Applies:
- Vectorised affine transformation mapping target voxel centres to the native frame.
- Slab gating with user-configurable end-plane extent (default: half-spacing; capped: 0.0).
- In-plane point-in-polygon evaluation using skimage.measure.points_in_poly with the
  even-odd winding rule (supporting nested holes and disjoint multi-ring components).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from skimage.measure import points_in_poly

from hdt.domains.vs_seg.contours import ContourPlane, StructureContours, group_rings_by_plane
from hdt.imaging.load_scan import ImageGeometry


def polygon_shoelace_area(pts_2d: np.ndarray) -> float:
    """Compute 2D polygon area using the shoelace formula.

    Parameters
    ----------
    pts_2d : np.ndarray
        Array of shape (N, 2) containing polygon vertices.

    Returns
    -------
    float
        Non-negative area.
    """
    pts = np.asarray(pts_2d, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return 0.0

    u = pts[:, 0]
    v = pts[:, 1]
    return 0.5 * abs(float(np.dot(u, np.roll(v, 1)) - np.dot(v, np.roll(u, 1))))


def compute_plane_net_area(rings_2d: list[np.ndarray]) -> float:
    """Compute the net planar area for a list of rings using the even-odd rule.

    Nested rings (holes) are subtracted; disjoint rings are added.
    """
    if not rings_2d:
        return 0.0
    if len(rings_2d) == 1:
        return polygon_shoelace_area(rings_2d[0])

    areas = [polygon_shoelace_area(r) for r in rings_2d]
    centroids = []
    for r in rings_2d:
        pts = r[:-1] if np.allclose(r[0], r[-1]) else r
        centroids.append(np.mean(pts, axis=0))

    net_area = 0.0
    for i, (area, c) in enumerate(zip(areas, centroids, strict=False)):
        # Count how many other rings enclose this centroid
        c_pt = np.array([c])
        enclosing_count = 0
        for j, other_ring in enumerate(rings_2d):
            if i == j:
                continue
            if bool(points_in_poly(c_pt, other_ring)[0]):
                enclosing_count += 1
        # Even enclosing count (0, 2...) -> positive component; odd (1, 3...) -> hole
        sign = -1.0 if (enclosing_count % 2 == 1) else 1.0
        net_area += sign * area

    return max(0.0, net_area)


def compute_shoelace_volume(
    structure_or_planes: StructureContours | list[ContourPlane],
    slice_spacing_mm: float | None = None,
) -> float:
    """Calculate the theoretical prism volume (shoelace area * slice spacing).

    Parameters
    ----------
    structure_or_planes : StructureContours | list[ContourPlane]
        Structure contours or list of slice planes.
    slice_spacing_mm : float | None
        Slice spacing in mm. If None and StructureContours passed, uses structure's spacing.

    Returns
    -------
    float
        Theoretical volume in mm^3.
    """
    if isinstance(structure_or_planes, StructureContours):
        planes = structure_or_planes.planes
        sp = slice_spacing_mm or structure_or_planes.slice_spacing_mm
    else:
        planes = structure_or_planes
        sp = slice_spacing_mm if slice_spacing_mm is not None else 1.5

    total_area = sum(compute_plane_net_area(p.rings_2d) for p in planes)
    return float(total_area * sp)


def check_mask_touches_border(mask: np.ndarray) -> bool:
    """Check if any True voxel in a 3D mask touches the array border.

    Parameters
    ----------
    mask : np.ndarray
        3D boolean NumPy array in (z, y, x) order.

    Returns
    -------
    bool
        True if the mask touches any border face, False otherwise.
    """
    if not np.any(mask):
        return False
    return bool(
        np.any(mask[0, :, :])
        or np.any(mask[-1, :, :])
        or np.any(mask[:, 0, :])
        or np.any(mask[:, -1, :])
        or np.any(mask[:, :, 0])
        or np.any(mask[:, :, -1])
    )


def rasterise_structure(
    rings: StructureContours | list[ContourPlane] | list[list[list[float]] | np.ndarray],
    native_geometry: ImageGeometry,
    target_geometry: ImageGeometry,
    transform: Callable[[np.ndarray], np.ndarray] | Any | None = None,
    end_extent_mm: float | None = None,
    bbox_margin_voxels: int = 5,
) -> np.ndarray:
    """Rasterise continuous prism structures onto the target geometry grid.

    Samples the continuous polygonal prism solid directly at the target voxel centres
    without post-rasterisation resampling.

    Parameters
    ----------
    rings : StructureContours | list[ContourPlane] | list[list[list[float]] | np.ndarray]
        Input contour definition (rings or grouped planes).
    native_geometry : ImageGeometry
        Spatial geometry of the series where contours were natively created.
    target_geometry : ImageGeometry
        Spatial geometry of the target grid (e.g. native T2 grid) where the mask will be rasterised.
    transform : Callable[[np.ndarray], np.ndarray] | Any | None
        Vectorised transformation mapping points from target physical space (LPS mm)
        to native physical space (LPS mm). If None, identity mapping is used.
    end_extent_mm : float | None
        Extent beyond the outermost ring planes along the normal vector (mm).
        Defaults to half slice spacing (slab convention). Pass 0.0 for the capped variant.
    bbox_margin_voxels : int
        Margin around the structure bounding box in voxels to restrict evaluation.

    Returns
    -------
    np.ndarray
        3D boolean NumPy array in (z, y, x) order matching target_geometry.shape.
    """
    target_shape = tuple(target_geometry.shape)
    if isinstance(rings, StructureContours):
        planes = rings.planes
    elif isinstance(rings, list) and len(rings) > 0 and isinstance(rings[0], ContourPlane):
        planes = rings
    elif isinstance(rings, list) and len(rings) == 0:
        return np.zeros(target_shape, dtype=bool)
    else:
        # Raw rings list
        planes, _ = group_rings_by_plane(
            rings=rings,  # type: ignore[arg-type]
            normal=native_geometry.direction[:, 2],
            in_plane_u=native_geometry.direction[:, 0],
            in_plane_v=native_geometry.direction[:, 1],
            slice_spacing_mm=float(native_geometry.spacing_mm[0]),
        )

    if not planes:
        return np.zeros(target_shape, dtype=bool)

    # 1. Determine bounding box of the structure in target voxel index space
    all_verts_lps = np.concatenate([r for p in planes for r in p.rings_lps], axis=0)  # (M, 3)

    if transform is not None and hasattr(transform, "get_inverse"):
        inv_tfm = transform.get_inverse()
        verts_in_target_lps = inv_tfm(all_verts_lps)
    else:
        # If no explicit inverse, points in native frame are approximated or identity
        verts_in_target_lps = all_verts_lps

    # Convert LPS points to continuous voxel indices (z, y, x) in target geometry
    target_origin_xyz = np.array(
        [
            float(target_geometry.origin_mm[2]),
            float(target_geometry.origin_mm[1]),
            float(target_geometry.origin_mm[0]),
        ],
        dtype=np.float64,
    )
    target_sp_xyz = np.array(
        [
            float(target_geometry.spacing_mm[2]),
            float(target_geometry.spacing_mm[1]),
            float(target_geometry.spacing_mm[0]),
        ],
        dtype=np.float64,
    )
    inv_target_dir = np.linalg.inv(target_geometry.direction)

    diff_lps = verts_in_target_lps - target_origin_xyz
    scaled_xyz = diff_lps @ inv_target_dir.T
    vox_indices_xyz = scaled_xyz / target_sp_xyz  # columns are (idx_x, idx_y, idx_z)

    z_min = float(np.min(vox_indices_xyz[:, 2]))
    z_max = float(np.max(vox_indices_xyz[:, 2]))
    y_min = float(np.min(vox_indices_xyz[:, 1]))
    y_max = float(np.max(vox_indices_xyz[:, 1]))
    x_min = float(np.min(vox_indices_xyz[:, 0]))
    x_max = float(np.max(vox_indices_xyz[:, 0]))

    margin = int(bbox_margin_voxels)
    k0 = max(0, int(math.floor(z_min)) - margin)
    k1 = min(target_shape[0], int(math.ceil(z_max)) + margin + 1)
    j0 = max(0, int(math.floor(y_min)) - margin)
    j1 = min(target_shape[1], int(math.ceil(y_max)) + margin + 1)
    i0 = max(0, int(math.floor(x_min)) - margin)
    i1 = min(target_shape[2], int(math.ceil(x_max)) + margin + 1)

    if k0 >= k1 or j0 >= j1 or i0 >= i1:
        return np.zeros(target_shape, dtype=bool)

    # 2. Generate target voxel centres inside the bounding box
    grid_k, grid_j, grid_i = np.meshgrid(
        np.arange(k0, k1, dtype=np.int64),
        np.arange(j0, j1, dtype=np.int64),
        np.arange(i0, i1, dtype=np.int64),
        indexing="ij",
    )
    flat_k = grid_k.ravel()
    flat_j = grid_j.ravel()
    flat_i = grid_i.ravel()

    # Convert voxel centres to physical LPS coordinates q (x, y, z)
    scaled_centers = np.column_stack(
        [
            flat_i * target_sp_xyz[0],
            flat_j * target_sp_xyz[1],
            flat_k * target_sp_xyz[2],
        ]
    )
    q_lps = target_origin_xyz + scaled_centers @ target_geometry.direction.T

    # 3. Apply transform: p = transform(q) mapping T2-frame point to T1-frame point
    if transform is not None:
        p_lps = transform(q_lps)
    else:
        p_lps = q_lps

    # 4. Express p in native frame: (u, v, s) along native axes
    u_axis = np.asarray(native_geometry.direction[:, 0], dtype=np.float64)
    v_axis = np.asarray(native_geometry.direction[:, 1], dtype=np.float64)
    norm = np.asarray(native_geometry.direction[:, 2], dtype=np.float64)

    u_pts = p_lps @ u_axis
    v_pts = p_lps @ v_axis
    s_pts = p_lps @ norm

    # 5. Find nearest ring plane and apply slab / end-plane extent gating
    slice_sp = float(native_geometry.spacing_mm[0])
    half_sp = slice_sp / 2.0
    limit_end = half_sp if end_extent_mm is None else float(end_extent_mm)

    plane_coords = np.array([p.coordinate for p in planes], dtype=np.float64)
    # Planes are strictly ordered: s_0 < s_1 < ... < s_{K-1}
    s0 = plane_coords[0]
    s_end = plane_coords[-1]

    dists_to_planes = np.abs(s_pts[:, None] - plane_coords[None, :])
    nearest_k = np.argmin(dists_to_planes, axis=1)
    dist_nearest = dists_to_planes[np.arange(len(s_pts)), nearest_k]

    # Z-acceptance mask
    eps = 1e-7
    z_accepted = np.zeros(len(s_pts), dtype=bool)

    # (a) Between end planes [s0, s_end]
    mask_between = (s_pts >= s0) & (s_pts <= s_end)
    z_accepted[mask_between] = dist_nearest[mask_between] <= (half_sp + eps)

    # (b) Beyond lower end plane (s < s0)
    mask_lower = s_pts < s0
    z_accepted[mask_lower] = (s0 - s_pts[mask_lower]) <= (limit_end + eps)

    # (c) Beyond upper end plane (s > s_end)
    mask_upper = s_pts > s_end
    z_accepted[mask_upper] = (s_end - s_pts[mask_upper]) >= -(limit_end + eps)

    # 6. Inside accepted planes: evaluate 2D point-in-polygon with even-odd rule
    inside_mask = np.zeros(len(s_pts), dtype=bool)

    for k, pl in enumerate(planes):
        plane_candidates = np.flatnonzero(z_accepted & (nearest_k == k))
        if len(plane_candidates) == 0:
            continue

        pts_2d = np.column_stack([u_pts[plane_candidates], v_pts[plane_candidates]])
        rings_2d = pl.rings_2d

        if len(rings_2d) == 1:
            poly_in = points_in_poly(pts_2d, rings_2d[0])
        else:
            poly_counts = np.zeros(len(plane_candidates), dtype=np.int32)
            for r in rings_2d:
                poly_counts += points_in_poly(pts_2d, r).astype(np.int32)
            poly_in = poly_counts % 2 == 1

        inside_mask[plane_candidates[poly_in]] = True

    # 7. Construct final 3D boolean mask
    mask_3d = np.zeros(target_shape, dtype=bool)
    accepted_voxels = np.flatnonzero(inside_mask)
    if len(accepted_voxels) > 0:
        mask_3d[
            flat_k[accepted_voxels],
            flat_j[accepted_voxels],
            flat_i[accepted_voxels],
        ] = True

    return mask_3d
