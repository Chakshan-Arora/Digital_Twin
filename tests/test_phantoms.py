"""Unit tests for synthetic phantoms and analytical geometry verification."""

import math

import numpy as np
import pytest

from hdt.imaging.phantoms import make_ellipsoid, make_phantom_case, make_sphere


def test_sphere_volume_isotropic_and_anisotropic() -> None:
    """Test that sphere volume is within 3% of 4/3*pi*r^3 on isotropic AND anisotropic grids.

    Tests both:
      - Isotropic spacing (1.0, 1.0, 1.0) mm
      - Anisotropic spacing (3.0, 0.5, 0.5) mm
    """
    radius_mm = 15.0
    analytic_vol = (4.0 / 3.0) * math.pi * (radius_mm**3)

    # 1. Isotropic grid
    shape_iso = (64, 64, 64)
    spacing_iso = (1.0, 1.0, 1.0)
    center_iso = (32.0, 32.0, 32.0)

    mask_iso = make_sphere(shape_iso, spacing_iso, center_iso, radius_mm)
    voxel_vol_iso = spacing_iso[0] * spacing_iso[1] * spacing_iso[2]
    discrete_vol_iso = float(np.sum(mask_iso)) * voxel_vol_iso

    rel_error_iso = abs(discrete_vol_iso - analytic_vol) / analytic_vol
    assert rel_error_iso < 0.03, (
        f"Isotropic volume error {rel_error_iso:.4%} exceeded 3% tolerance "
        f"(discrete: {discrete_vol_iso:.2f}, analytic: {analytic_vol:.2f} mm^3)"
    )

    # 2. Anisotropic grid: thick slices (3.0 mm) and high in-plane resolution (0.5 mm)
    shape_aniso = (32, 128, 128)
    spacing_aniso = (3.0, 0.5, 0.5)
    center_aniso = (16.0 * 3.0, 64.0 * 0.5, 64.0 * 0.5)  # in physical mm

    mask_aniso = make_sphere(shape_aniso, spacing_aniso, center_aniso, radius_mm)
    voxel_vol_aniso = spacing_aniso[0] * spacing_aniso[1] * spacing_aniso[2]
    discrete_vol_aniso = float(np.sum(mask_aniso)) * voxel_vol_aniso

    rel_error_aniso = abs(discrete_vol_aniso - analytic_vol) / analytic_vol
    assert rel_error_aniso < 0.03, (
        f"Anisotropic volume error {rel_error_aniso:.4%} exceeded 3% tolerance "
        f"(discrete: {discrete_vol_aniso:.2f}, analytic: {analytic_vol:.2f} mm^3)"
    )


def test_ellipsoid_volume_analytic() -> None:
    """Test that make_ellipsoid volume matches 4/3*pi*rz*ry*rx within 3% tolerance."""
    radii_mm = (12.0, 8.0, 15.0)
    analytic_vol = (4.0 / 3.0) * math.pi * radii_mm[0] * radii_mm[1] * radii_mm[2]

    shape = (48, 48, 64)
    spacing = (1.0, 1.0, 1.0)
    center = (24.0, 24.0, 32.0)

    mask = make_ellipsoid(shape, spacing, center, radii_mm)
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    discrete_vol = float(np.sum(mask)) * voxel_vol

    rel_error = abs(discrete_vol - analytic_vol) / analytic_vol
    assert rel_error < 0.03


def test_make_phantom_case_separated_structures() -> None:
    """Test make_phantom_case with known distance and clearance between target and risk."""
    case, metrics = make_phantom_case(
        case_id="phantom_test",
        shape_zyx=(64, 64, 64),
        spacing_mm=(1.0, 1.0, 1.0),
        tumor_center_mm=(32.0, 32.0, 20.0),
        tumor_radius_mm=5.0,
        vessel_center_mm=(32.0, 32.0, 40.0),
        vessel_radius_mm=3.0,
    )

    assert case.case_id == "phantom_test"
    assert case.image.dtype == np.float32
    assert set(case.names()) == {"tumor", "vessel"}
    assert case.get("tumor").role == "target"
    assert case.get("vessel").role == "risk"

    # Known centre distance: |40.0 - 20.0| = 20.0 mm
    assert metrics["center_distance_mm"] == pytest.approx(20.0, abs=1e-5)
    # Analytic clearance: 20.0 - (5.0 + 3.0) = 12.0 mm
    assert metrics["analytic_clearance_mm"] == pytest.approx(12.0, abs=1e-5)


def test_make_phantom_case_touching_or_overlapping_clearance_is_zero() -> None:
    """Test that overlapping or touching spheres produce an analytic clearance of 0.0 mm."""
    _, metrics = make_phantom_case(
        tumor_center_mm=(30.0, 30.0, 30.0),
        tumor_radius_mm=8.0,
        vessel_center_mm=(30.0, 30.0, 35.0),
        vessel_radius_mm=5.0,
    )

    # Centre distance is 5.0 mm, sum of radii is 13.0 mm (overlapping)
    assert metrics["center_distance_mm"] == pytest.approx(5.0, abs=1e-5)
    assert metrics["analytic_clearance_mm"] == 0.0


def test_empty_mask_edge_cases() -> None:
    """Test edge cases resulting in empty masks (radius <= 0 or center completely outside)."""
    shape = (20, 20, 20)
    spacing = (1.0, 1.0, 1.0)
    center = (10.0, 10.0, 10.0)

    # Zero radius
    mask_zero = make_sphere(shape, spacing, center, radius_mm=0.0)
    assert not np.any(mask_zero)

    # Negative radius
    mask_neg = make_sphere(shape, spacing, center, radius_mm=-2.0)
    assert not np.any(mask_neg)

    # Ellipsoid with zero radius along one axis
    mask_ellip = make_ellipsoid(shape, spacing, center, radii_mm_zyx=(10.0, 0.0, 10.0))
    assert not np.any(mask_ellip)
