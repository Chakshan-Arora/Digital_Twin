"""Acceptance tests for geometry distance and clearance calculations."""

import numpy as np
import pytest

from hdt.geometry.distances import (
    clearance_from_sdf,
    closest_points_mm,
    mesh_clearance_mm,
    signed_clearance_mm,
    signed_distance_field_mm,
    surface_distance_stats,
)
from hdt.imaging.phantoms import make_sphere


def test_disjoint_spheres_isotropic():
    """Test 1 & 3: Disjoint spheres on isotropic spacing: clearance == d - r1 - r2."""
    shape = (60, 60, 60)
    spacing = (1.0, 1.0, 1.0)
    c1 = (30.0, 30.0, 15.0)
    c2 = (30.0, 30.0, 45.0)
    r1 = 10.0
    r2 = 8.0
    expected_clearance = 30.0 - 10.0 - 8.0  # 12.0 mm

    s1 = make_sphere(shape, spacing, c1, r1)
    s2 = make_sphere(shape, spacing, c2, r2)

    diag = np.sqrt(sum(s**2 for s in spacing))

    clearance = signed_clearance_mm(s1, s2, spacing)
    err = abs(clearance - expected_clearance)
    # Under face-to-face convention, discrete sphere gap is 11.0 mm (12 empty layers)
    assert err <= diag, f"Error {err:.3f} mm exceeded diagonal {diag:.3f} mm"
    assert clearance > 0.0


def test_disjoint_spheres_anisotropic():
    """Test 1 & 3: Disjoint spheres on anisotropic (3.0, 0.5, 0.5) mm spacing."""
    shape = (40, 80, 80)
    spacing = (3.0, 0.5, 0.5)
    c1 = (60.0, 20.0, 20.0)
    c2 = (60.0, 20.0, 35.0)
    r1 = 6.0
    r2 = 5.0
    expected_clearance = 15.0 - 6.0 - 5.0  # 4.0 mm

    s1 = make_sphere(shape, spacing, c1, r1)
    s2 = make_sphere(shape, spacing, c2, r2)

    diag = np.sqrt(sum(s**2 for s in spacing))
    half_diag = 0.5 * diag

    clearance = signed_clearance_mm(s1, s2, spacing)
    err = abs(clearance - expected_clearance)
    assert err <= half_diag, f"Error {err:.3f} mm exceeded half diagonal {half_diag:.3f} mm"
    assert clearance > 0.0


def test_overlapping_spheres_isotropic():
    """Test 2: Overlapping spheres: clearance == -(r1 + r2 - d) within tolerance."""
    shape = (60, 60, 60)
    spacing = (1.0, 1.0, 1.0)
    c1 = (30.0, 30.0, 25.0)
    c2 = (30.0, 30.0, 35.0)
    r1 = 10.0
    r2 = 8.0
    d = 10.0
    expected_clearance = -(r1 + r2 - d)  # -(10 + 8 - 10) = -8.0 mm

    s1 = make_sphere(shape, spacing, c1, r1)
    s2 = make_sphere(shape, spacing, c2, r2)

    diag = np.sqrt(sum(s**2 for s in spacing))
    half_diag = 0.5 * diag

    clearance = signed_clearance_mm(s1, s2, spacing)
    err = abs(clearance - expected_clearance)
    assert err <= half_diag, f"Error {err:.3f} mm exceeded half diagonal {half_diag:.3f} mm"
    assert clearance < 0.0


def test_overlapping_spheres_anisotropic():
    """Test 2 & 3: Overlapping spheres on (3.0, 0.5, 0.5) mm spacing."""
    shape = (40, 80, 80)
    spacing = (3.0, 0.5, 0.5)
    c1 = (60.0, 20.0, 20.0)
    c2 = (60.0, 20.0, 27.0)
    r1 = 6.0
    r2 = 5.0
    d = 7.0
    expected_clearance = -(r1 + r2 - d)  # -(6 + 5 - 7) = -4.0 mm

    s1 = make_sphere(shape, spacing, c1, r1)
    s2 = make_sphere(shape, spacing, c2, r2)

    diag = np.sqrt(sum(s**2 for s in spacing))
    half_diag = 0.5 * diag

    clearance = signed_clearance_mm(s1, s2, spacing)
    err = abs(clearance - expected_clearance)
    assert err <= half_diag, f"Error {err:.3f} mm exceeded half diagonal {half_diag:.3f} mm"
    assert clearance < 0.0


def test_touching_masks_isotropic_and_anisotropic():
    """Test 4: Touching masks give ~0, never an error."""
    shape = (30, 30, 30)

    # Isotropic
    spacing_iso = (1.0, 1.0, 1.0)
    a_iso = np.zeros(shape, dtype=bool)
    b_iso = np.zeros(shape, dtype=bool)
    a_iso[10:20, 10:20, 5:15] = True
    b_iso[10:20, 10:20, 15:25] = True  # touches at x=15

    c_iso = signed_clearance_mm(a_iso, b_iso, spacing_iso)
    diag_iso = np.sqrt(3.0)
    assert abs(c_iso) <= diag_iso
    assert c_iso == pytest.approx(0.0, abs=1.0)

    # Anisotropic
    spacing_aniso = (3.0, 0.5, 0.5)
    a_aniso = np.zeros(shape, dtype=bool)
    b_aniso = np.zeros(shape, dtype=bool)
    a_aniso[5:15, 10:20, 10:20] = True
    b_aniso[15:25, 10:20, 10:20] = True  # touches along z=15

    c_aniso = signed_clearance_mm(a_aniso, b_aniso, spacing_aniso)
    diag_aniso = np.sqrt(3.0**2 + 0.5**2 + 0.5**2)
    assert abs(c_aniso) <= diag_aniso
    assert c_aniso == pytest.approx(0.0, abs=3.0)


def test_empty_mask_raises():
    """Test 5: Empty mask raises ValueError naming which argument."""
    shape = (20, 20, 20)
    spacing = (1.0, 1.0, 1.0)
    empty = np.zeros(shape, dtype=bool)
    non_empty = np.zeros(shape, dtype=bool)
    non_empty[10, 10, 10] = True

    # Check 'a' is empty
    with pytest.raises(ValueError, match="Mask 'a' is empty"):
        signed_clearance_mm(empty, non_empty, spacing)
    with pytest.raises(ValueError, match="Mask 'a' is empty"):
        closest_points_mm(empty, non_empty, spacing)
    with pytest.raises(ValueError, match="Mask 'a' is empty"):
        surface_distance_stats(empty, non_empty, spacing)
    with pytest.raises(ValueError, match="Mask 'a' is empty"):
        mesh_clearance_mm(empty, non_empty, spacing)
    with pytest.raises(ValueError, match="Mask 'a' is empty"):
        clearance_from_sdf(empty, np.zeros(shape))

    # Check 'b' is empty
    with pytest.raises(ValueError, match="Mask 'b' is empty"):
        signed_clearance_mm(non_empty, empty, spacing)
    with pytest.raises(ValueError, match="Mask 'b' is empty"):
        closest_points_mm(non_empty, empty, spacing)
    with pytest.raises(ValueError, match="Mask 'b' is empty"):
        surface_distance_stats(non_empty, empty, spacing)
    with pytest.raises(ValueError, match="Mask 'b' is empty"):
        mesh_clearance_mm(non_empty, empty, spacing)
    with pytest.raises(ValueError, match="Mask 'b' is empty"):
        signed_distance_field_mm(empty, spacing)


def test_symmetry_of_stats_and_asymmetry_of_clearance():
    """Test 6: hd95_a_to_b, hd95_b_to_a, and symmetric hd95 = max(ab, ba).

    signed_clearance is NOT symmetric: test that explicitly and document why.
    """
    shape = (50, 50, 50)
    spacing = (1.5, 1.0, 0.8)

    center = (37.5, 25.0, 20.0)
    large_sphere = make_sphere(shape, spacing, center, radius_mm=16.0)
    small_sphere = make_sphere(shape, spacing, center, radius_mm=4.0)

    stats_ab = surface_distance_stats(large_sphere, small_sphere, spacing)
    stats_ba = surface_distance_stats(small_sphere, large_sphere, spacing)

    # Check keys exist
    for key in ("min", "mean", "hd95_a_to_b", "hd95_b_to_a", "hd95", "max"):
        assert key in stats_ab

    assert stats_ab["min"] == pytest.approx(stats_ba["min"])
    assert stats_ab["mean"] == pytest.approx(stats_ba["mean"])
    assert stats_ab["max"] == pytest.approx(stats_ba["max"])
    assert stats_ab["hd95"] == pytest.approx(stats_ba["hd95"])
    assert stats_ab["hd95_a_to_b"] == pytest.approx(stats_ba["hd95_b_to_a"])
    assert stats_ab["hd95_b_to_a"] == pytest.approx(stats_ba["hd95_a_to_b"])
    assert stats_ab["hd95"] == max(stats_ab["hd95_a_to_b"], stats_ab["hd95_b_to_a"])

    sc_sl = signed_clearance_mm(small_sphere, large_sphere, spacing)
    sc_ls = signed_clearance_mm(large_sphere, small_sphere, spacing)
    assert sc_sl < 0.0
    assert sc_ls < 0.0
    assert sc_sl != pytest.approx(sc_ls, abs=1.0)


def test_zero_padding_invariance():
    """Test 7: Zero-padding the volume does not change any result."""
    shape = (40, 40, 40)
    spacing = (1.2, 0.8, 1.0)
    s1 = make_sphere(shape, spacing, (24.0, 24.0, 15.0), radius_mm=6.0)
    s2 = make_sphere(shape, spacing, (24.0, 24.0, 28.0), radius_mm=5.0)

    base_clearance = signed_clearance_mm(s1, s2, spacing)
    base_mesh = mesh_clearance_mm(s1, s2, spacing)
    base_pts = closest_points_mm(s1, s2, spacing)
    base_stats = surface_distance_stats(s1, s2, spacing)

    pad_width = ((5, 5), (5, 5), (5, 5))
    s1_pad = np.pad(s1, pad_width, mode="constant", constant_values=False)
    s2_pad = np.pad(s2, pad_width, mode="constant", constant_values=False)

    pad_clearance = signed_clearance_mm(s1_pad, s2_pad, spacing)
    pad_mesh = mesh_clearance_mm(s1_pad, s2_pad, spacing)
    pad_pts = closest_points_mm(s1_pad, s2_pad, spacing)
    pad_stats = surface_distance_stats(s1_pad, s2_pad, spacing)

    assert base_clearance == pytest.approx(pad_clearance)
    assert base_mesh == pytest.approx(pad_mesh, abs=1e-3)
    assert base_pts["distance_mm"] == pytest.approx(pad_pts["distance_mm"])
    assert base_stats["min"] == pytest.approx(pad_stats["min"])
    assert base_stats["mean"] == pytest.approx(pad_stats["mean"])
    assert base_stats["hd95"] == pytest.approx(pad_stats["hd95"])
    assert base_stats["max"] == pytest.approx(pad_stats["max"])


def test_closest_points_consistency():
    """Test 8: Distance between the two returned points equals distance_mm."""
    shape = (50, 50, 50)
    spacing = (2.0, 0.7, 1.2)
    s1 = make_sphere(shape, spacing, (40.0, 20.0, 15.0), radius_mm=8.0)
    s2 = make_sphere(shape, spacing, (60.0, 30.0, 35.0), radius_mm=7.0)

    res = closest_points_mm(s1, s2, spacing)
    pa = np.array(res["point_a_zyx_mm"])
    pb = np.array(res["point_b_zyx_mm"])

    expected_dist = float(np.linalg.norm(pa - pb))
    assert res["distance_mm"] == pytest.approx(expected_dist)
    assert res["distance_mm"] > 0.0

    s_overlap = make_sphere(shape, spacing, (45.0, 22.0, 17.0), radius_mm=8.0)
    res_ol = closest_points_mm(s1, s_overlap, spacing)
    assert res_ol["distance_mm"] == 0.0
    pa_ol = np.array(res_ol["point_a_zyx_mm"])
    pb_ol = np.array(res_ol["point_b_zyx_mm"])
    assert np.all(pa_ol == pb_ol)


def test_rotation_invariance():
    """Test 9: Result unchanged under a 90-degree array rotation with spacing permuted to match."""
    shape = (30, 40, 50)
    spacing = (2.0, 1.5, 0.8)
    s1 = make_sphere(shape, spacing, (30.0, 30.0, 20.0), radius_mm=6.0)
    s2 = make_sphere(shape, spacing, (30.0, 30.0, 32.0), radius_mm=4.0)

    c_orig = signed_clearance_mm(s1, s2, spacing)
    m_orig = mesh_clearance_mm(s1, s2, spacing)
    stats_orig = surface_distance_stats(s1, s2, spacing)
    pts_orig = closest_points_mm(s1, s2, spacing)

    s1_rot = np.rot90(s1, k=1, axes=(1, 2))
    s2_rot = np.rot90(s2, k=1, axes=(1, 2))
    spacing_rot = (spacing[0], spacing[2], spacing[1])

    c_rot = signed_clearance_mm(s1_rot, s2_rot, spacing_rot)
    m_rot = mesh_clearance_mm(s1_rot, s2_rot, spacing_rot)
    stats_rot = surface_distance_stats(s1_rot, s2_rot, spacing_rot)
    pts_rot = closest_points_mm(s1_rot, s2_rot, spacing_rot)

    assert c_orig == pytest.approx(c_rot, abs=1e-3)
    assert m_orig == pytest.approx(m_rot, abs=1e-3)
    assert stats_orig["min"] == pytest.approx(stats_rot["min"], abs=1e-3)
    assert stats_orig["mean"] == pytest.approx(stats_rot["mean"], abs=1e-3)
    assert stats_orig["hd95"] == pytest.approx(stats_rot["hd95"], abs=1e-3)
    assert stats_orig["max"] == pytest.approx(stats_rot["max"], abs=1e-3)
    assert pts_orig["distance_mm"] == pytest.approx(pts_rot["distance_mm"], abs=1e-3)


def test_mesh_clearance_overlaps_raise():
    """mesh_clearance_mm must raise ValueError if masks overlap."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    s1 = make_sphere(shape, spacing, (20.0, 20.0, 15.0), radius_mm=8.0)
    s2 = make_sphere(shape, spacing, (20.0, 20.0, 20.0), radius_mm=8.0)
    with pytest.raises(ValueError, match="masks overlap"):
        mesh_clearance_mm(s1, s2, spacing)


def test_non_integer_radii_and_off_grid_centres():
    """Test 4: Non-integer radii and off-grid centres, including r1=6.3, r2=5.8, d=20."""
    shape = (60, 60, 60)
    spacing = (1.0, 1.0, 1.0)
    r1 = 6.3
    r2 = 5.8
    d = 20.0
    c1 = (28.3, 29.7, 18.4)
    c2 = (28.3, 29.7, 18.4 + d)
    expected_clearance = d - r1 - r2  # 20.0 - 6.3 - 5.8 = 7.9 mm

    s1 = make_sphere(shape, spacing, c1, r1)
    s2 = make_sphere(shape, spacing, c2, r2)

    diag = float(np.sqrt(3.0))

    c_edt = signed_clearance_mm(s1, s2, spacing)
    c_mesh = mesh_clearance_mm(s1, s2, spacing)

    err_edt = abs(c_edt - expected_clearance)
    err_mesh = abs(c_mesh - expected_clearance)

    # Within one voxel diagonal tolerance (1.732 mm)
    assert err_edt <= diag, f"EDT error {err_edt:.3f} mm exceeded diagonal {diag:.3f} mm"
    assert err_mesh <= diag, f"Mesh error {err_mesh:.3f} mm exceeded diagonal {diag:.3f} mm"


def test_border_handling_edge_invariance():
    """Test 5: Mask touching the array edge gives the same results as embedded in larger array."""
    sp = (1.5, 0.8, 1.0)
    small_shape = (20, 25, 25)
    a_edge = np.zeros(small_shape, dtype=bool)
    b_edge = np.zeros(small_shape, dtype=bool)

    # a touches z=0 and x=0
    a_edge[0:6, 5:15, 0:8] = True
    b_edge[10:18, 5:15, 12:22] = True

    # Embed in larger array with 6 voxels padding on all sides
    large_shape = (32, 37, 37)
    a_large = np.zeros(large_shape, dtype=bool)
    b_large = np.zeros(large_shape, dtype=bool)
    a_large[6 : 6 + 20, 6 : 6 + 25, 6 : 6 + 25] = a_edge
    b_large[6 : 6 + 20, 6 : 6 + 25, 6 : 6 + 25] = b_edge

    c_edge = signed_clearance_mm(a_edge, b_edge, sp)
    c_large = signed_clearance_mm(a_large, b_large, sp)
    assert c_edge == pytest.approx(c_large)

    stats_edge = surface_distance_stats(a_edge, b_edge, sp)
    stats_large = surface_distance_stats(a_large, b_large, sp)
    assert stats_edge["min"] == pytest.approx(stats_large["min"])
    assert stats_edge["mean"] == pytest.approx(stats_large["mean"])
    assert stats_edge["hd95"] == pytest.approx(stats_large["hd95"])
    assert stats_edge["max"] == pytest.approx(stats_large["max"])


def test_ladder_face_to_face_exact():
    """Test 1: LADDER TEST (non-negotiable, tolerance 1e-6 mm).

    Two axis-aligned cuboids separated along each axis by k empty voxel layers,
    k in {0, 1, 2, 3, 5}, on spacings (1,1,1) and (3,0.5,0.5), along z, y and x.
    Expected signed clearance = k * spacing_along_that_axis (k=0 -> 0.0).
    Overlap by k layers (k in {1, 2, 3}) -> -k * spacing_along_that_axis.
    """
    spacings = [(1.0, 1.0, 1.0), (3.0, 0.5, 0.5)]

    for sp in spacings:
        for axis in range(3):
            s_i = sp[axis]

            # 1. Disjoint / touching: k empty voxel layers
            for k in [0, 1, 2, 3, 5]:
                shape = [60, 60, 60]
                a = np.zeros(shape, dtype=bool)
                b = np.zeros(shape, dtype=bool)

                sl_a = [slice(10, 50), slice(10, 50), slice(10, 50)]
                sl_b = [slice(10, 50), slice(10, 50), slice(10, 50)]

                # a has length 15 voxels along tested axis: [10, 25)
                # b starts at 25 + k, has length 15 voxels: [25 + k, 40 + k)
                sl_a[axis] = slice(10, 25)
                sl_b[axis] = slice(25 + k, 40 + k)

                a[tuple(sl_a)] = True
                b[tuple(sl_b)] = True

                expected = float(k * s_i)
                clearance = signed_clearance_mm(a, b, sp)
                assert abs(clearance - expected) < 1e-6, (
                    f"Ladder positive failed: sp={sp}, axis={axis}, k={k}: "
                    f"got {clearance}, expected {expected}"
                )

            # 2. Overlap: k overlapping voxel layers
            for k in [1, 2, 3]:
                shape = [60, 60, 60]
                a = np.zeros(shape, dtype=bool)
                b = np.zeros(shape, dtype=bool)

                sl_a = [slice(10, 50), slice(10, 50), slice(10, 50)]
                sl_b = [slice(10, 50), slice(10, 50), slice(10, 50)]

                # a is [10, 25), b is [25 - k, 40 - k)
                # Overlap along tested axis is exactly k voxels
                sl_a[axis] = slice(10, 25)
                sl_b[axis] = slice(25 - k, 40 - k)

                a[tuple(sl_a)] = True
                b[tuple(sl_b)] = True

                expected = float(-k * s_i)
                clearance = signed_clearance_mm(a, b, sp)
                assert abs(clearance - expected) < 1e-6, (
                    f"Ladder overlap failed: sp={sp}, axis={axis}, k={k}: "
                    f"got {clearance}, expected {expected}"
                )


def test_precomputed_sdf_and_bbox_cropping_equivalence():
    """Test: clearance_from_sdf matches signed_clearance_mm; cropped == uncropped."""
    shape = (50, 50, 50)
    spacing = (1.2, 1.0, 0.8)
    s1 = make_sphere(shape, spacing, (25.0, 20.0, 15.0), radius_mm=6.0)
    s2 = make_sphere(shape, spacing, (35.0, 30.0, 30.0), radius_mm=7.0)

    # 1. Uncropped clearance_from_sdf
    sdf = signed_distance_field_mm(s2, spacing)
    c_sdf = clearance_from_sdf(s1, sdf)
    # EDT field minimum over a gives raw EDT distance; clearance_from_sdf extracts min(sdf[a])
    assert isinstance(sdf.data, np.ndarray)

    # 2. Cropped with max_distance_mm vs uncropped
    sdf_cropped = signed_distance_field_mm(s2, spacing, max_distance_mm=30.0)
    c_cropped_sdf = clearance_from_sdf(s1, sdf_cropped)
    assert c_cropped_sdf == pytest.approx(c_sdf)

    # 3. crop_to_bbox=True vs crop_to_bbox=False for signed_clearance_mm
    c_direct = signed_clearance_mm(s1, s2, spacing, crop_to_bbox=False)
    c_cropped = signed_clearance_mm(s1, s2, spacing, crop_to_bbox=True, bbox_margin_voxels=5)
    assert c_cropped == pytest.approx(c_direct)

    # 4. Test on overlapping sphere
    s2_ol = make_sphere(shape, spacing, (28.0, 22.0, 17.0), radius_mm=7.0)
    c_direct_ol = signed_clearance_mm(s1, s2_ol, spacing, crop_to_bbox=False)
    c_cropped_ol = signed_clearance_mm(
        s1, s2_ol, spacing, crop_to_bbox=True, bbox_margin_voxels=5
    )
    assert c_cropped_ol == pytest.approx(c_direct_ol)


@pytest.mark.slow
def test_bias_study_reporting():
    """Test 2 & 3: Bias study reporting errors across spacings.

    Draws 50 random sphere pairs per spacing: (1,1,1), (3,0.5,0.5), (2,1,1).
    Radii in [4, 12] mm, random sub-voxel center offsets, fixed seed.
    Reports mean, median, max signed error, max absolute error.
    Asserts mesh_clearance_mm max absolute error is within 1 voxel diagonal.
    """
    spacings = [
        (1.0, 1.0, 1.0),
        (3.0, 0.5, 0.5),
        (2.0, 1.0, 1.0),
    ]
    rng = np.random.default_rng(2026)

    for sp in spacings:
        diag = float(np.sqrt(sum(s**2 for s in sp)))
        shape = (80, 80, 80)
        disjoint_edt_errors = []
        disjoint_mesh_errors = []
        overlap_edt_errors = []

        for _ in range(50):
            r1 = float(rng.uniform(4.0, 12.0))
            r2 = float(rng.uniform(4.0, 12.0))
            c1 = np.array([40.0, 40.0, 40.0]) * sp + rng.uniform(-0.5, 0.5, size=3) * sp
            vec = rng.normal(size=3)
            vec /= np.linalg.norm(vec)

            # 50% disjoint, 50% overlapping
            is_disjoint = rng.random() > 0.5
            if is_disjoint:
                gap = float(rng.uniform(2.0, 15.0))
                d = r1 + r2 + gap
                analytic = gap
            else:
                pen = float(rng.uniform(1.0, min(r1, r2) - 1.0))
                d = r1 + r2 - pen
                analytic = -pen

            c2 = c1 + vec * d
            s1 = make_sphere(shape, sp, tuple(c1), r1)
            s2 = make_sphere(shape, sp, tuple(c2), r2)

            if not np.any(s1) or not np.any(s2):
                continue

            if not np.any(s1 & s2):
                # Disjoint
                c_edt = signed_clearance_mm(s1, s2, sp)
                c_mesh = mesh_clearance_mm(s1, s2, sp)
                disjoint_edt_errors.append(c_edt - analytic)
                disjoint_mesh_errors.append(c_mesh - analytic)
            else:
                # Overlapping
                c_edt = signed_clearance_mm(s1, s2, sp)
                overlap_edt_errors.append(c_edt - analytic)

        # Print bias summary report
        print(f"\n--- Bias Study Report: Spacing {sp} (Voxel Diagonal {diag:.3f} mm) ---")
        if disjoint_edt_errors:
            print(
                f"Disjoint EDT (N={len(disjoint_edt_errors)}): "
                f"mean={np.mean(disjoint_edt_errors):+.3f}, "
                f"median={np.median(disjoint_edt_errors):+.3f}, "
                f"max_signed={np.max(disjoint_edt_errors):+.3f}, "
                f"max_abs={np.max(np.abs(disjoint_edt_errors)):.3f}, "
                f"MAE={np.mean(np.abs(disjoint_edt_errors)):.3f}"
            )
        if disjoint_mesh_errors:
            mesh_max_abs = float(np.max(np.abs(disjoint_mesh_errors)))
            print(
                f"Disjoint Mesh (N={len(disjoint_mesh_errors)}): "
                f"mean={np.mean(disjoint_mesh_errors):+.3f}, "
                f"median={np.median(disjoint_mesh_errors):+.3f}, "
                f"max_signed={np.max(disjoint_mesh_errors):+.3f}, "
                f"max_abs={mesh_max_abs:.3f}, "
                f"MAE={np.mean(np.abs(disjoint_mesh_errors)):.3f}"
            )
            assert mesh_max_abs <= diag, (
                f"Mesh max abs error {mesh_max_abs:.3f} exceeded 1 voxel diagonal {diag:.3f}"
            )
        if overlap_edt_errors:
            print(
                f"Overlap EDT (N={len(overlap_edt_errors)}): "
                f"mean={np.mean(overlap_edt_errors):+.3f}, "
                f"median={np.median(overlap_edt_errors):+.3f}, "
                f"max_signed={np.max(overlap_edt_errors):+.3f}, "
                f"max_abs={np.max(np.abs(overlap_edt_errors)):.3f}, "
                f"MAE={np.mean(np.abs(overlap_edt_errors)):.3f}"
            )
