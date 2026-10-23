"""3D mask-to-mask distance, clearance, and mesh reference computations in millimetres.

Conventions:
- Masks are 3D boolean NumPy arrays with shape (z, y, x).
- Spacing is a 3-tuple (sz, sy, sx) in millimetres in (z, y, x) order.
- Distance and coordinate units are in millimetres.
- Coordinate order for points is (z, y, x) in the local volume frame (voxel_index * spacing_mm).
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class DistanceFieldResult:
    """Precomputed signed distance field with spatial bounding box offset.

    Attributes:
        data: 3D float array containing signed distance values in mm.
        origin_offset_voxels: (z, y, x) integer voxel offset of the cropped region
            within the full reference volume. (0, 0, 0) if uncropped.
        spacing_mm: (sz, sy, sx) voxel spacing in millimetres.
        full_shape: Original (z, y, x) shape of the reference volume before cropping.
        max_distance_mm: Distance cap applied during computation, or None.
    """

    data: np.ndarray
    origin_offset_voxels: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    full_shape: tuple[int, int, int]
    max_distance_mm: float | None = None


def _validate_mask_inputs(
    a: np.ndarray,
    b: np.ndarray,
    spacing_mm: Sequence[float],
) -> tuple[float, float, float]:
    """Validate input masks and spacing.

    Raises:
        ValueError: If a or b is not a 3D boolean array, if shapes mismatch,
            if either mask is empty, or if spacing_mm is invalid.
    """
    if not isinstance(a, np.ndarray) or a.dtype != bool or a.ndim != 3:
        raise ValueError("Mask 'a' must be a 3D boolean numpy array")
    if not isinstance(b, np.ndarray) or b.dtype != bool or b.ndim != 3:
        raise ValueError("Mask 'b' must be a 3D boolean numpy array")
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: mask 'a' shape {a.shape} != mask 'b' shape {b.shape}"
        )
    if not np.any(a):
        raise ValueError("Mask 'a' is empty")
    if not np.any(b):
        raise ValueError("Mask 'b' is empty")

    if len(spacing_mm) != 3:
        raise ValueError(
            f"spacing_mm must have length 3 (z, y, x), got length {len(spacing_mm)}"
        )
    sz, sy, sx = (float(s) for s in spacing_mm)
    if sz <= 0.0 or sy <= 0.0 or sx <= 0.0:
        raise ValueError(f"spacing_mm components must be strictly positive, got {spacing_mm}")

    return sz, sy, sx


def _crop_to_union_bbox(
    a: np.ndarray,
    b: np.ndarray,
    margin_voxels: int = 5,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    """Crop masks a and b to their union bounding box expanded by margin_voxels."""
    coords = np.argwhere(a | b)
    if len(coords) == 0:
        return a, b, (slice(None), slice(None), slice(None))

    min_idx = np.maximum(coords.min(axis=0) - margin_voxels, 0)
    max_idx = np.minimum(coords.max(axis=0) + margin_voxels + 1, a.shape)

    slices = (
        slice(int(min_idx[0]), int(max_idx[0])),
        slice(int(min_idx[1]), int(max_idx[1])),
        slice(int(min_idx[2]), int(max_idx[2])),
    )
    return a[slices], b[slices], slices


def signed_distance_field_mm(
    b: np.ndarray,
    spacing_mm: Sequence[float],
    max_distance_mm: float | None = None,
) -> DistanceFieldResult:
    """Precompute the 3D signed distance field of mask `b` in millimetres.

    Signs:
      - Outside `b`: positive Euclidean distance to the outer boundary of `b`.
      - Inside `b`: negative Euclidean distance to the outer background (~b).

    Border Handling:
      To prevent artificial boundary truncation when `b` touches the edge of the array,
      masks are padded by 1 voxel before computing Euclidean distance transforms,
      then cropped back to the original or cropped shape.

    Cropping with max_distance_mm:
      If `max_distance_mm` is provided, the computation is cropped to the bounding box of `b`
      expanded by `ceil(max_distance_mm / spacing) + 1` voxels. The returned DistanceFieldResult
      stores the spatial offset, allowing `clearance_from_sdf` to extract values correctly.

    Args:
        b: 3D boolean mask of reference structure (z, y, x).
        spacing_mm: Voxel spacing (sz, sy, sx) in millimetres.
        max_distance_mm: Optional distance cap. If given, crops field to bbox of `b` plus margin.

    Returns:
        DistanceFieldResult: Precomputed distance field object.
    """
    if not isinstance(b, np.ndarray) or b.dtype != bool or b.ndim != 3:
        raise ValueError("Mask 'b' must be a 3D boolean numpy array")
    if not np.any(b):
        raise ValueError("Mask 'b' is empty")
    if len(spacing_mm) != 3:
        raise ValueError(
            f"spacing_mm must have length 3 (z, y, x), got length {len(spacing_mm)}"
        )
    sz, sy, sx = (float(s) for s in spacing_mm)
    if sz <= 0.0 or sy <= 0.0 or sx <= 0.0:
        raise ValueError(f"spacing_mm components must be strictly positive, got {spacing_mm}")
    spacing = (sz, sy, sx)

    full_shape = b.shape
    if max_distance_mm is not None:
        if max_distance_mm <= 0.0:
            raise ValueError("max_distance_mm must be positive")
        # Compute margin in voxels per axis
        margin = [int(np.ceil(max_distance_mm / s)) + 2 for s in spacing]
        coords = np.argwhere(b)
        min_idx = np.maximum(coords.min(axis=0) - margin, 0)
        max_idx = np.minimum(coords.max(axis=0) + margin + 1, full_shape)
        slices = (
            slice(int(min_idx[0]), int(max_idx[0])),
            slice(int(min_idx[1]), int(max_idx[1])),
            slice(int(min_idx[2]), int(max_idx[2])),
        )
        sub_b = b[slices]
        offset = (int(min_idx[0]), int(min_idx[1]), int(min_idx[2]))
    else:
        sub_b = b
        offset = (0, 0, 0)

    # 1-voxel zero-padding for border robustness
    pad_b = np.pad(sub_b, 1, mode="constant", constant_values=False)

    edt_outside = distance_transform_edt(~pad_b, sampling=spacing)[1:-1, 1:-1, 1:-1]
    edt_inside = distance_transform_edt(pad_b, sampling=spacing)[1:-1, 1:-1, 1:-1]

    sdf_data = edt_outside - edt_inside

    return DistanceFieldResult(
        data=sdf_data,
        origin_offset_voxels=offset,
        spacing_mm=spacing,
        full_shape=full_shape,
        max_distance_mm=max_distance_mm,
    )


def clearance_from_sdf(
    a: np.ndarray,
    sdf: np.ndarray | DistanceFieldResult,
) -> float:
    """Compute signed clearance from mask `a` given a precomputed signed distance field.

    Accepts either a raw 3D numpy array (of identical shape to `a`) or a `DistanceFieldResult`
    which may be cropped to a bounding box.

    Args:
        a: 3D boolean mask of primary structure (z, y, x).
        sdf: Precomputed signed distance field or DistanceFieldResult.

    Returns:
        float: Signed clearance in millimetres.
    """
    if not isinstance(a, np.ndarray) or a.dtype != bool or a.ndim != 3:
        raise ValueError("Mask 'a' must be a 3D boolean numpy array")
    if not np.any(a):
        raise ValueError("Mask 'a' is empty")

    if isinstance(sdf, DistanceFieldResult):
        if a.shape != sdf.full_shape:
            raise ValueError(
                f"Shape mismatch: mask 'a' shape {a.shape} != sdf full_shape {sdf.full_shape}"
            )
        oz, oy, ox = sdf.origin_offset_voxels
        sz, sy, sx = sdf.data.shape
        # Intersect a with cropped window
        sub_a = a[oz : oz + sz, oy : oy + sy, ox : ox + sx]
        if np.any(sub_a):
            return float(np.min(sdf.data[sub_a]))
        # All voxels of a lie outside the cropped bounding box
        if sdf.max_distance_mm is not None:
            return float(sdf.max_distance_mm)
        raise ValueError("Mask 'a' does not intersect the provided distance field window")

    if not isinstance(sdf, np.ndarray) or sdf.ndim != 3:
        raise ValueError("sdf must be a 3D numpy array or DistanceFieldResult")
    if a.shape != sdf.shape:
        raise ValueError(f"Shape mismatch: mask 'a' {a.shape} != sdf {sdf.shape}")

    return float(np.min(sdf[a]))


def signed_clearance_mm(
    a: np.ndarray,
    b: np.ndarray,
    spacing_mm: Sequence[float],
    crop_to_bbox: bool = False,
    bbox_margin_voxels: int = 5,
) -> float:
    """Compute the signed clearance from mask `a` to mask `b` in millimetres.

    Signs & Face-to-Face Convention:
      - Positive (> 0): Mask `a` and `b` are disjoint. Value is the minimum face-to-face
        separation between the two voxel sets.
      - Zero (== 0): Mask `a` and `b` touch (adjacent abutting voxel faces).
      - Negative (< 0): Mask `a` penetrates mask `b`. Magnitude represents the deepest
        penetration depth of voxels in `a` inside `b`.

    Derivation of Voxel-Extent Half-Voxel Correction (2t):
      In discrete voxel grids, distance_transform_edt computes distances between voxel centers.
      For two abutting voxels along axis i (separated by k=0 empty voxels), their centers are
      at distance 1.0 * spacing_i, but their continuous face-to-face distance is 0.0.
      For k empty voxels separating two cuboids along axis i, the center-to-center distance
      is (k + 1) * spacing_i, while the physical face-to-face clearance is k * spacing_i.

      Along an arbitrary unit separation direction u = (pa - pb) / ||pa - pb||, the half-extent
      from the voxel center to the voxel boundary is:
          t = min_{i, u_i != 0} (spacing_i / (2 * |u_i|))
      Subtracting 2t (one half-extent at pa, one at pb) converts center-to-center distance
      to exact face-to-face distance:
          clearance = max(0.0, ||pa - pb|| - 2t)
      Along any cardinal axis i, u_i = 1 and t = spacing_i / 2, so:
          clearance = (k + 1) * spacing_i - 2 * (spacing_i / 2) = k * spacing_i
      This passes the axis-aligned ladder test exactly (tolerance 1e-6 mm).

    Overlapping Branch:
      For overlapping voxels (a & b), the penetration depth is evaluated via edt(pad_b).
      For k overlapping voxel layers along axis i, the deepest penetration is exactly
      k * spacing_i, giving signed clearance = -k * spacing_i.

    Args:
        a: 3D boolean mask of primary structure (shape z, y, x).
        b: 3D boolean mask of reference structure (shape z, y, x).
        spacing_mm: Voxel spacing (sz, sy, sx) in millimetres.
        crop_to_bbox: If True, crops to union bounding box plus margin for acceleration.
        bbox_margin_voxels: Margin added around union bounding box when crop_to_bbox is True.

    Returns:
        float: Signed clearance in millimetres.
    """
    sz, sy, sx = _validate_mask_inputs(a, b, spacing_mm)
    spacing = (sz, sy, sx)

    if crop_to_bbox:
        sub_a, sub_b, _ = _crop_to_union_bbox(a, b, margin_voxels=bbox_margin_voxels)
    else:
        sub_a, sub_b = a, b

    # Overlap check
    overlap = sub_a & sub_b
    if not np.any(overlap):
        # Disjoint or touching: compute closest voxel pair with 1-voxel padding
        pad_b = np.pad(sub_b, 1, mode="constant", constant_values=False)
        dist_padded, indices_padded = distance_transform_edt(
            ~pad_b, sampling=spacing, return_indices=True
        )
        dist = dist_padded[1:-1, 1:-1, 1:-1]

        dist_in_a = np.where(sub_a, dist, np.inf)
        flat_idx = int(np.argmin(dist_in_a))
        za, ya, xa = np.unravel_index(flat_idx, sub_a.shape)

        zb = int(indices_padded[0][za + 1, ya + 1, xa + 1]) - 1
        yb = int(indices_padded[1][za + 1, ya + 1, xa + 1]) - 1
        xb = int(indices_padded[2][za + 1, ya + 1, xa + 1]) - 1

        pa = np.array([za * sz, ya * sy, xa * sx], dtype=np.float64)
        pb = np.array([zb * sz, yb * sy, xb * sx], dtype=np.float64)

        diff = pa - pb
        norm = float(np.linalg.norm(diff))
        if norm < 1e-12:
            return 0.0

        u = np.abs(diff / norm)
        ratios = [spacing[i] / (2.0 * u[i]) for i in range(3) if u[i] > 1e-9]
        t = min(ratios)

        corrected = norm - 2.0 * t
        return float(max(0.0, corrected))

    # Overlapping: compute negative penetration depth with 1-voxel padding
    pad_b = np.pad(sub_b, 1, mode="constant", constant_values=False)
    dist_inside_b = distance_transform_edt(pad_b, sampling=spacing)[1:-1, 1:-1, 1:-1]
    max_depth = float(np.max(dist_inside_b[overlap]))
    return -max_depth


def mesh_clearance_mm(
    a: np.ndarray,
    b: np.ndarray,
    spacing_mm: Sequence[float],
    bbox_margin_voxels: int = 2,
) -> float:
    """Compute clearance between disjoint masks using marching cubes surface meshes.

    This serves as an independent geometric reference for signed_clearance_mm.

    Surfaces are extracted using skimage.measure.marching_cubes at level=0.5 on
    zero-padded float32 masks. Minimum vertex-to-vertex Euclidean distance between
    the two meshes is computed with scipy.spatial.cKDTree.

    Args:
        a: 3D boolean mask of structure a (z, y, x).
        b: 3D boolean mask of structure b (z, y, x).
        spacing_mm: Voxel spacing (sz, sy, sx) in millimetres.
        bbox_margin_voxels: Safety margin when cropping to union bounding box.

    Raises:
        ValueError: If masks overlap or either mask is empty/invalid.

    Returns:
        float: Minimum mesh vertex separation in millimetres.
    """
    sz, sy, sx = _validate_mask_inputs(a, b, spacing_mm)
    spacing = (sz, sy, sx)

    if np.any(a & b):
        raise ValueError("mesh_clearance_mm requires disjoint masks; masks overlap")

    # Crop to union bounding box plus margin
    sub_a, sub_b, _ = _crop_to_union_bbox(a, b, margin_voxels=bbox_margin_voxels)

    # Pad float32 masks by 1 voxel to allow closed boundary extraction
    pad_a = np.pad(sub_a.astype(np.float32), 1, mode="constant", constant_values=0.0)
    pad_b = np.pad(sub_b.astype(np.float32), 1, mode="constant", constant_values=0.0)

    verts_a, _, _, _ = marching_cubes(pad_a, level=0.5, spacing=spacing)
    verts_b, _, _, _ = marching_cubes(pad_b, level=0.5, spacing=spacing)

    # Offset vertices back by the 1-voxel pad in physical coordinates
    offset = np.array(spacing, dtype=np.float64)
    verts_a = verts_a - offset
    verts_b = verts_b - offset

    # Compute minimum vertex-to-vertex Euclidean distance
    tree_b = cKDTree(verts_b)
    distances, _ = tree_b.query(verts_a)
    return float(np.min(distances))


def closest_points_mm(
    a: np.ndarray,
    b: np.ndarray,
    spacing_mm: Sequence[float],
) -> dict[str, float | tuple[float, float, float]]:
    """Find closest points between mask `a` and mask `b` in millimetres.

    Uses Euclidean distance transform with indices returned on ~b:
        edt(~b, return_indices=True)
    to find the nearest voxel center in `b` for each voxel center in `a`.

    Coordinates are given in physical millimetres in the volume local frame:
        point_zyx_mm = (index_z * sz, index_y * sy, index_x * sx)

    Args:
        a: 3D boolean mask of structure a (z, y, x).
        b: 3D boolean mask of structure b (z, y, x).
        spacing_mm: Voxel spacing (sz, sy, sx) in millimetres.

    Returns:
        dict with keys:
          - "distance_mm": float Euclidean distance between the closest pair.
          - "point_a_zyx_mm": (z, y, x) coordinates in mm of the point on a.
          - "point_b_zyx_mm": (z, y, x) coordinates in mm of the point on b.
    """
    sz, sy, sx = _validate_mask_inputs(a, b, spacing_mm)
    spacing = (sz, sy, sx)

    # Check for direct overlap
    overlap = a & b
    if np.any(overlap):
        coords = np.argwhere(overlap)
        idx_zyx = coords[0]
        pt_mm = (
            float(idx_zyx[0] * sz),
            float(idx_zyx[1] * sy),
            float(idx_zyx[2] * sx),
        )
        return {
            "distance_mm": 0.0,
            "point_a_zyx_mm": pt_mm,
            "point_b_zyx_mm": pt_mm,
        }

    # Disjoint: query closest voxel in b from voxels in a
    pad_b = np.pad(b, 1, mode="constant", constant_values=False)
    dist_padded, indices_padded = distance_transform_edt(
        ~pad_b, sampling=spacing, return_indices=True
    )
    dist = dist_padded[1:-1, 1:-1, 1:-1]
    indices_z = indices_padded[0][1:-1, 1:-1, 1:-1] - 1
    indices_y = indices_padded[1][1:-1, 1:-1, 1:-1] - 1
    indices_x = indices_padded[2][1:-1, 1:-1, 1:-1] - 1

    # Mask out voxels not in a
    dist_in_a = np.where(a, dist, np.inf)
    flat_idx = int(np.argmin(dist_in_a))
    za, ya, xa = np.unravel_index(flat_idx, a.shape)

    zb = int(indices_z[za, ya, xa])
    yb = int(indices_y[za, ya, xa])
    xb = int(indices_x[za, ya, xa])

    pt_a_mm = (float(za * sz), float(ya * sy), float(xa * sx))
    pt_b_mm = (float(zb * sz), float(yb * sy), float(xb * sx))

    dist_mm = float(
        np.sqrt(
            (pt_a_mm[0] - pt_b_mm[0]) ** 2
            + (pt_a_mm[1] - pt_b_mm[1]) ** 2
            + (pt_a_mm[2] - pt_b_mm[2]) ** 2
        )
    )

    return {
        "distance_mm": dist_mm,
        "point_a_zyx_mm": pt_a_mm,
        "point_b_zyx_mm": pt_b_mm,
    }


def _extract_surface_mask(mask: np.ndarray) -> np.ndarray:
    """Extract 1-voxel outer surface using 6-connectivity erosion with 1-voxel padding.

    surface = mask & ~binary_erosion(mask, structure=struct_6)
    Zero-padding by 1 voxel ensures boundaries at the array edge are properly identified.
    """
    struct_6 = generate_binary_structure(3, 1)
    pad_mask = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded_pad = binary_erosion(pad_mask, structure=struct_6)[1:-1, 1:-1, 1:-1]
    surface = mask & ~eroded_pad
    if not np.any(surface):
        return mask.copy()
    return surface


def surface_distance_stats(
    a: np.ndarray,
    b: np.ndarray,
    spacing_mm: Sequence[float],
) -> dict[str, float]:
    """Compute surface-to-surface distance statistics in millimetres.

    Surface voxels are extracted via 6-neighbourhood boundary erosion:
        surface_m = m & ~binary_erosion(m, structure=6_connectivity)
    with 1-voxel padding applied to ensure invariant boundary handling.

    Metrics computed:
      - "min": minimum surface distance (0.0 if surfaces touch or intersect).
      - "mean": average bidirectional surface distance over pooled surface points.
      - "hd95_a_to_b": 95th percentile distance from surface `a` to surface `b`.
      - "hd95_b_to_a": 95th percentile distance from surface `b` to surface `a`.
      - "hd95": max(hd95_a_to_b, hd95_b_to_a), standard symmetric 95th percentile
        Hausdorff distance.
      - "max": maximum bidirectional surface distance (Hausdorff distance).

    Args:
        a: 3D boolean mask (z, y, x).
        b: 3D boolean mask (z, y, x).
        spacing_mm: Voxel spacing (sz, sy, sx) in millimetres.

    Returns:
        dict with keys "min", "mean", "hd95_a_to_b", "hd95_b_to_a", "hd95", "max".
    """
    sz, sy, sx = _validate_mask_inputs(a, b, spacing_mm)
    spacing = (sz, sy, sx)

    surf_a = _extract_surface_mask(a)
    surf_b = _extract_surface_mask(b)

    # Distances from surf_a to surf_b (using 1-voxel padded EDT)
    pad_b = np.pad(surf_b, 1, mode="constant", constant_values=False)
    edt_to_surf_b = distance_transform_edt(~pad_b, sampling=spacing)[1:-1, 1:-1, 1:-1]
    d_a_to_b = edt_to_surf_b[surf_a]

    # Distances from surf_b to surf_a (using 1-voxel padded EDT)
    pad_a = np.pad(surf_a, 1, mode="constant", constant_values=False)
    edt_to_surf_a = distance_transform_edt(~pad_a, sampling=spacing)[1:-1, 1:-1, 1:-1]
    d_b_to_a = edt_to_surf_a[surf_b]

    # Directional 95th percentile Hausdorff distances
    hd95_ab = float(np.percentile(d_a_to_b, 95))
    hd95_ba = float(np.percentile(d_b_to_a, 95))
    hd95_sym = max(hd95_ab, hd95_ba)

    # Pooled distribution for symmetric min, mean, max
    all_d = np.concatenate([d_a_to_b, d_b_to_a])

    return {
        "min": float(np.min(all_d)),
        "mean": float(np.mean(all_d)),
        "hd95_a_to_b": hd95_ab,
        "hd95_b_to_a": hd95_ba,
        "hd95": hd95_sym,
        "max": float(np.max(all_d)),
    }
