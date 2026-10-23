"""Synthetic unit tests for VS-SEG continuous structure rasteriser."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from hdt.domains.vs_seg.sampler import (
    rasterise_structure,
)
from hdt.imaging.dicom_series import AffineTransformLPS
from hdt.imaging.load_scan import ImageGeometry


def create_circular_ring(
    radius: float,
    center_uv: tuple[float, float],
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    plane_coord: float,
    num_pts: int = 150,
) -> np.ndarray:
    """Create a 3D planar circular ring with given in-plane center and radius."""
    theta = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    u_vals = center_uv[0] + radius * np.cos(theta)
    v_vals = center_uv[1] + radius * np.sin(theta)

    pts = []
    for u, v in zip(u_vals, v_vals, strict=False):
        pt = u * u_axis + v * v_axis + plane_coord * normal
        pts.append(pt)
    pts.append(pts[0])
    return np.array(pts, dtype=np.float64)


def test_oblique_series_cylinder_sampling() -> None:
    """Test sampling an oblique cylinder with circular rings onto an offset target grid."""
    # Oblique native geometry: 30 degree rotation around X axis
    theta = np.deg2rad(30.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    native_dir = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t],
        ],
        dtype=np.float64,
    )
    u_axis = native_dir[:, 0]
    v_axis = native_dir[:, 1]
    normal = native_dir[:, 2]

    native_geom = ImageGeometry(
        shape=(50, 80, 80),
        spacing_mm=(1.5, 0.5, 0.5),
        origin_mm=(-2.0, -20.0, -20.0),
        direction=native_dir,
    )

    # 5 planes at 1.5 mm spacing with circular rings of radius 5.0 mm
    radius = 5.0
    plane_coords = [0.0, 1.5, 3.0, 4.5, 6.0]
    rings = [
        create_circular_ring(radius, (0.0, 0.0), normal, u_axis, v_axis, s) for s in plane_coords
    ]

    # Target grid: centered with offset of 0.4 mm in x/y and 0.7 mm in z
    target_dir = np.eye(3, dtype=np.float64)
    target_geom = ImageGeometry(
        shape=(60, 100, 100),
        spacing_mm=(0.5, 0.25, 0.25),
        origin_mm=(-5.0 + 0.7, -12.0 + 0.4, -12.0 + 0.4),
        direction=target_dir,
    )

    # Translation transform: p = q + T -> T2 point q is mapped to T1 point p = q + T
    trans_vec = np.array([0.3, 0.0, 0.0], dtype=np.float64)
    tfm = AffineTransformLPS(
        matrix=np.eye(3, dtype=np.float64),
        offset=trans_vec,
        center=np.zeros(3, dtype=np.float64),
    )

    # Rasterise slab variant
    mask_slab = rasterise_structure(
        rings=rings,
        native_geometry=native_geom,
        target_geometry=target_geom,
        transform=tfm,
        end_extent_mm=0.75,
    )

    vox_vol = float(
        target_geom.spacing_mm[0] * target_geom.spacing_mm[1] * target_geom.spacing_mm[2]
    )
    measured_vol_slab = float(np.sum(mask_slab) * vox_vol)

    area = np.pi * (radius**2)
    expected_vol = area * (len(plane_coords) * 1.5)  # pi * r^2 * (5 * 1.5) = 589.05 mm3

    vol_error_pct = abs(measured_vol_slab - expected_vol) / expected_vol
    assert vol_error_pct < 0.03, f"Volume error {vol_error_pct * 100:.2f}% exceeds 3%"

    # Rasterise capped variant (end_extent_mm = 0.0)
    mask_capped = rasterise_structure(
        rings=rings,
        native_geometry=native_geom,
        target_geometry=target_geom,
        transform=tfm,
        end_extent_mm=0.0,
    )
    measured_vol_capped = float(np.sum(mask_capped) * vox_vol)

    # Capped variant volume smaller by about 2 * 0.75 * area (tolerance one voxel layer)
    expected_diff = 2 * 0.75 * area
    actual_diff = measured_vol_slab - measured_vol_capped

    tol = area * float(target_geom.spacing_mm[0]) * 1.2
    assert abs(actual_diff - expected_diff) < tol, (
        f"Capped diff {actual_diff:.2f} mm3 vs expected {expected_diff:.2f} mm3 "
        f"exceeds tolerance {tol:.2f} mm3"
    )


def test_nested_ring_plane_gives_hole() -> None:
    """Test that a nested ring produces a hole under the even-odd rule."""
    native_dir = np.eye(3, dtype=np.float64)
    u_axis, v_axis, normal = native_dir[:, 0], native_dir[:, 1], native_dir[:, 2]

    native_geom = ImageGeometry(
        shape=(10, 80, 80),
        spacing_mm=(1.5, 0.5, 0.5),
        origin_mm=(0.0, -20.0, -20.0),
        direction=native_dir,
    )
    target_geom = ImageGeometry(
        shape=(10, 80, 80),
        spacing_mm=(1.5, 0.25, 0.25),
        origin_mm=(0.0, -10.0, -10.0),
        direction=native_dir,
    )

    outer_r = 5.0
    inner_r = 2.0
    r_outer = create_circular_ring(outer_r, (0.0, 0.0), normal, u_axis, v_axis, 0.0)
    r_inner = create_circular_ring(inner_r, (0.0, 0.0), normal, u_axis, v_axis, 0.0)

    # Single plane with both rings (nested hole)
    rings = [r_outer, r_inner]
    mask = rasterise_structure(
        rings=rings,
        native_geometry=native_geom,
        target_geometry=target_geom,
        transform=None,
    )

    vox_vol = float(
        target_geom.spacing_mm[0] * target_geom.spacing_mm[1] * target_geom.spacing_mm[2]
    )
    measured_vol = float(np.sum(mask) * vox_vol)

    expected_area = np.pi * (outer_r**2 - inner_r**2)
    expected_vol = expected_area * 1.5

    assert abs(measured_vol - expected_vol) / expected_vol < 0.03


def test_two_disjoint_rings_gives_union() -> None:
    """Test that two disjoint rings on the same plane give their union."""
    native_dir = np.eye(3, dtype=np.float64)
    u_axis, v_axis, normal = native_dir[:, 0], native_dir[:, 1], native_dir[:, 2]

    geom = ImageGeometry(
        shape=(10, 100, 100),
        spacing_mm=(1.5, 0.25, 0.25),
        origin_mm=(0.0, -12.5, -12.5),
        direction=native_dir,
    )

    r = 2.0
    r1 = create_circular_ring(r, (-5.0, 0.0), normal, u_axis, v_axis, 0.0)
    r2 = create_circular_ring(r, (+5.0, 0.0), normal, u_axis, v_axis, 0.0)

    mask = rasterise_structure(
        rings=[r1, r2],
        native_geometry=geom,
        target_geometry=geom,
        transform=None,
    )

    vox_vol = float(geom.spacing_mm[0] * geom.spacing_mm[1] * geom.spacing_mm[2])
    measured_vol = float(np.sum(mask) * vox_vol)
    expected_vol = 2 * (np.pi * r**2) * 1.5

    assert abs(measured_vol - expected_vol) / expected_vol < 0.04


def test_unsorted_rings_gives_same_mask() -> None:
    """Test that input ring ordering does not alter the rasterised mask."""
    geom = ImageGeometry(
        shape=(20, 60, 60),
        spacing_mm=(1.5, 0.5, 0.5),
        origin_mm=(0.0, -15.0, -15.0),
        direction=np.eye(3),
    )
    u_axis, v_axis, normal = geom.direction[:, 0], geom.direction[:, 1], geom.direction[:, 2]

    rings = [
        create_circular_ring(3.0, (0.0, 0.0), normal, u_axis, v_axis, s)
        for s in [0.0, 1.5, 3.0, 4.5]
    ]
    shuffled = [rings[2], rings[0], rings[3], rings[1]]

    mask1 = rasterise_structure(rings=rings, native_geometry=geom, target_geometry=geom)
    mask2 = rasterise_structure(rings=shuffled, native_geometry=geom, target_geometry=geom)

    assert np.array_equal(mask1, mask2)


def test_transform_direction_known_translation() -> None:
    """Test that a structure sampled through a known translation transform lands where expected."""
    geom = ImageGeometry(
        shape=(30, 40, 40),
        spacing_mm=(1.0, 0.5, 0.5),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3),
    )
    u_axis, v_axis, normal = geom.direction[:, 0], geom.direction[:, 1], geom.direction[:, 2]

    # Native sphere/cylinder centered at (10, 10, 10) in LPS
    native_center = (10.0, 10.0, 10.0)
    rings = [
        create_circular_ring(
            3.0, (native_center[0], native_center[1]), normal, u_axis, v_axis, native_center[2]
        )
    ]

    # Transform: p = q + T where T = (2.0, 3.0, 1.0)
    # Target voxel q = p - T, so target centroid should be at native_center - T
    T = np.array([2.0, 3.0, 1.0], dtype=np.float64)
    tfm = AffineTransformLPS(
        matrix=np.eye(3),
        offset=T,
        center=np.zeros(3),
    )

    mask = rasterise_structure(
        rings=rings,
        native_geometry=geom,
        target_geometry=geom,
        transform=tfm,
    )

    # Compute centroid of sampled voxels in LPS mm
    z_idx, y_idx, x_idx = np.where(mask)
    vox_centers_lps = np.column_stack(
        [
            x_idx * geom.spacing_mm[2] + geom.origin_mm[2],
            y_idx * geom.spacing_mm[1] + geom.origin_mm[1],
            z_idx * geom.spacing_mm[0] + geom.origin_mm[0],
        ]
    )
    measured_centroid = np.mean(vox_centers_lps, axis=0)
    expected_centroid = np.array(native_center) - T

    assert np.allclose(measured_centroid, expected_centroid, atol=geom.spacing_mm[1] * 1.5)


def test_affine_transform_against_simpleitk() -> None:
    """Test AffineTransformLPS against SimpleITK.Transform.TransformPoint to 1e-9."""
    rng = np.random.default_rng(12345)
    rot_angle = 0.42
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    offset = np.array([1.23, -4.56, 7.89], dtype=np.float64)
    center = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    sitk_tfm = sitk.AffineTransform(3)
    sitk_tfm.SetMatrix(tuple(mat.flatten()))
    sitk_tfm.SetTranslation(tuple(offset))
    sitk_tfm.SetCenter(tuple(center))

    vec_tfm = AffineTransformLPS(matrix=mat, offset=offset, center=center)

    test_points = rng.uniform(-100.0, 100.0, size=(200, 3))
    vec_results = vec_tfm(test_points)
    sitk_results = np.array([sitk_tfm.TransformPoint(tuple(pt)) for pt in test_points])

    max_err = float(np.max(np.abs(vec_results - sitk_results)))
    assert max_err < 1e-9, f"Max error {max_err} exceeds 1e-9"
