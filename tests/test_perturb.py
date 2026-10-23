"""Unit and integration tests for mask perturbation model."""

import warnings

import numpy as np
import pytest
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label
from skimage.measure import marching_cubes, mesh_surface_area

from hdt.imaging.phantoms import make_sphere
from hdt.uncertainty.perturb_masks import perturb_mask, uniform_shift


def test_reproducibility_same_seed_identical_output():
    """Same seed produces identical output; different seeds produce different output."""
    shape = (50, 50, 50)
    spacing = (1.0, 1.0, 1.0)
    sphere = make_sphere(shape, spacing, (25.0, 25.0, 25.0), radius_mm=12.0)

    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)
    rng3 = np.random.default_rng(99999)

    pert1 = perturb_mask(sphere, spacing, sigma_mm=1.5, corr_length_mm=4.0, rng=rng1)
    pert2 = perturb_mask(sphere, spacing, sigma_mm=1.5, corr_length_mm=4.0, rng=rng2)
    pert3 = perturb_mask(sphere, spacing, sigma_mm=1.5, corr_length_mm=4.0, rng=rng3)

    assert np.array_equal(pert1, pert2), "Same seed must produce bitwise identical output"
    assert not np.array_equal(pert1, pert3), "Different seeds must produce different output"


def test_shape_dtype_and_component_count():
    """Perturbed mask has bool dtype, identical shape, and retains single component."""
    shape = (40, 50, 60)
    spacing = (2.0, 1.0, 1.0)
    sphere = make_sphere(shape, spacing, (40.0, 25.0, 30.0), radius_mm=10.0)

    _, orig_components = label(sphere)
    assert orig_components == 1

    rng = np.random.default_rng(42)
    pert = perturb_mask(sphere, spacing, sigma_mm=1.5, corr_length_mm=4.0, rng=rng)

    assert pert.dtype == bool
    assert pert.shape == shape
    _, pert_components = label(pert)
    assert pert_components == orig_components, "Component count must be preserved"


def test_sphere_surface_displacement_statistics():
    """Sphere r=30 mm, spacing 1 mm, sigma 2 mm, corr 5 mm:

    std of signed displacement at original surface voxels within 15% of 2 mm,
    mean displacement within 0.2 mm of 0 with bias 0.
    """
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)
    center = (50.0, 50.0, 50.0)
    r = 30.0
    sphere = make_sphere(shape, spacing, center, r)

    struct_6 = generate_binary_structure(3, 1)
    surf_orig = sphere & ~binary_erosion(sphere, structure=struct_6)
    coords = np.argwhere(surf_orig)
    rad_orig = np.sqrt(np.sum((coords - np.array(center)) ** 2, axis=1))

    # Evaluate for seed 123
    rng = np.random.default_rng(123)
    pert = perturb_mask(sphere, spacing, sigma_mm=2.0, corr_length_mm=5.0, rng=rng)

    pad_p = np.pad(pert, 1, mode="constant", constant_values=False)
    edt_out = distance_transform_edt(~pad_p, sampling=spacing)[1:-1, 1:-1, 1:-1]
    edt_in = distance_transform_edt(pad_p, sampling=spacing)[1:-1, 1:-1, 1:-1]
    sdf_pert = edt_out - edt_in

    # Signed displacement along outward normal: -sdf_pert - (30 - rad_orig)
    disp = -sdf_pert[surf_orig] - (30.0 - rad_orig)
    mean_disp = float(np.mean(disp))
    std_disp = float(np.std(disp))

    # Mean displacement within 0.2 mm of 0
    assert abs(mean_disp) <= 0.2, f"Mean displacement {mean_disp:.4f} exceeded 0.2 mm"

    # std of signed displacement within 15% of 2 mm: [1.70, 2.30] mm
    assert (
        1.70 <= std_disp <= 2.30
    ), f"std displacement {std_disp:.4f} not within 15% of 2.0 mm ([1.70, 2.30])"


def test_surface_area_within_thirty_percent():
    """Surface area of perturbed mask within 30% of original."""
    shape = (80, 80, 80)
    spacing = (1.0, 1.0, 1.0)
    center = (40.0, 40.0, 40.0)
    r = 20.0
    sphere = make_sphere(shape, spacing, center, r)

    v_orig, f_orig, _, _ = marching_cubes(
        np.pad(sphere.astype(np.float32), 1), level=0.5, spacing=spacing
    )
    area_orig = float(mesh_surface_area(v_orig, f_orig))

    rng = np.random.default_rng(42)
    pert = perturb_mask(sphere, spacing, sigma_mm=1.5, corr_length_mm=4.0, rng=rng)
    v_pert, f_pert, _, _ = marching_cubes(
        np.pad(pert.astype(np.float32), 1), level=0.5, spacing=spacing
    )
    area_pert = float(mesh_surface_area(v_pert, f_pert))

    ratio = area_pert / area_orig
    assert (
        0.70 <= ratio <= 1.30
    ), f"Surface area ratio {ratio:.3f} was not within 30% of original"


def test_sigma_below_half_voxel_warns():
    """sigma below half a voxel warns."""
    shape = (40, 40, 40)
    spacing = (2.0, 1.0, 1.0)  # min spacing is 1.0 mm, half is 0.5 mm
    sphere = make_sphere(shape, spacing, (40.0, 20.0, 20.0), radius_mm=8.0)

    rng = np.random.default_rng(42)
    with pytest.warns(UserWarning, match="smaller than half the minimum voxel spacing"):
        _ = perturb_mask(sphere, spacing, sigma_mm=0.3, corr_length_mm=3.0, rng=rng)

    # Above half voxel should not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = perturb_mask(sphere, spacing, sigma_mm=0.6, corr_length_mm=3.0, rng=rng)


def test_uniform_shift_sphere_radius():
    """uniform_shift(+k) changes a sphere's radius by k within one voxel."""
    shape = (60, 60, 60)
    spacing = (1.0, 1.0, 1.0)
    center = (30.0, 30.0, 30.0)
    r_orig = 15.0
    sphere = make_sphere(shape, spacing, center, r_orig)

    k_shift = 3.0
    shifted = uniform_shift(sphere, spacing, shift_mm=k_shift)

    coords = np.argwhere(shifted)
    rads = np.sqrt(np.sum((coords - np.array(center)) ** 2, axis=1))
    max_r = float(np.max(rads))

    # Radius should be close to 15.0 + 3.0 = 18.0 mm within 1 voxel spacing (1.0 mm)
    assert (
        abs(max_r - (r_orig + k_shift)) <= 1.0
    ), f"Shifted radius {max_r:.2f} differs from expected {r_orig + k_shift:.2f} by > 1 voxel"
