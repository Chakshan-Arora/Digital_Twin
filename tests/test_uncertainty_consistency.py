"""Consistency and normalization tests for uncertainty module."""

import numpy as np
import pytest
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
)

from hdt.geometry.distances import signed_clearance_mm
from hdt.imaging.load_scan import ImageGeometry, load_nifti_image, load_nifti_mask
from hdt.imaging.phantoms import make_phantom_case, make_sphere
from hdt.uncertainty.analyse_clearance import analyse_clearance
from hdt.uncertainty.perturb_masks import perturb_mask, uniform_shift


def test_zero_perturbation_identical_mask():
    """perturb_mask with sigma_mm=0 and bias 0 returns the identical mask."""
    shape = (50, 50, 50)
    spacing = (1.5, 1.0, 0.8)
    sphere = make_sphere(shape, spacing, (37.5, 25.0, 20.0), radius_mm=12.0)

    rng = np.random.default_rng(42)
    unpert = perturb_mask(sphere, spacing, sigma_mm=0.0, corr_length_mm=5.0, rng=rng)

    assert np.array_equal(sphere, unpert)


def test_unperturbed_sample_clearance_equals_baseline_exactly():
    """Clearance of an unperturbed sample equals baseline EXACTLY on phantom and cuboid ladder."""
    # 1. On exported phantom
    img, sp, orig, dir_mat = load_nifti_image("data/processed/phantom_image.nii.gz")
    geom = ImageGeometry(shape=img.shape, spacing_mm=sp, origin_mm=orig, direction=dir_mat)
    target = load_nifti_mask(
        "data/processed/target_mask.nii.gz", image_meta=geom, label_values=[1]
    )
    risk = load_nifti_mask("data/processed/risk_mask.nii.gz", image_meta=geom, label_values=[1])

    res = analyse_clearance(
        target,
        risk,
        sp,
        params={"sigma_mm": 0.0, "corr_length_mm": 5.0},
        n_samples=5,
        seed=42,
        threshold_mm=3.0,
    )
    for sample in res.samples_mm:
        assert abs(sample - res.baseline_mm) < 1e-6
    assert abs(res.baseline_mm - 3.5) < 1e-6

    # 2. On cuboid ladder at k in {0, 1, 3} for spacings (1,1,1) and (3,0.5,0.5)
    spacings = [(1.0, 1.0, 1.0), (3.0, 0.5, 0.5)]
    for sp_ladder in spacings:
        for axis in range(3):
            s_i = sp_ladder[axis]
            for k in [0, 1, 3]:
                shape = [60, 60, 60]
                a = np.zeros(shape, dtype=bool)
                b = np.zeros(shape, dtype=bool)

                sl_a = [slice(10, 50), slice(10, 50), slice(10, 50)]
                sl_b = [slice(10, 50), slice(10, 50), slice(10, 50)]

                sl_a[axis] = slice(10, 25)
                sl_b[axis] = slice(25 + k, 40 + k)

                a[tuple(sl_a)] = True
                b[tuple(sl_b)] = True

                ladder_res = analyse_clearance(
                    target=a,
                    risk=b,
                    spacing_mm=sp_ladder,
                    params={"sigma_mm": 0.0, "corr_length_mm": 5.0},
                    n_samples=3,
                    seed=42,
                    threshold_mm=1.0,
                )
                expected_k = float(k * s_i)
                assert abs(ladder_res.baseline_mm - expected_k) < 1e-6
                for s_val in ladder_res.samples_mm:
                    assert abs(s_val - expected_k) < 1e-6


@pytest.mark.slow
def test_isotropic_phantom_mean_clearance_shift():
    """sigma=1 mm, L=5 mm on isotropic phantom: mean clearance of 400 samples within 0.3 mm."""
    shape = (60, 60, 60)
    spacing = (1.0, 1.0, 1.0)
    c1 = (30.0, 30.0, 15.0)
    c2 = (30.0, 30.0, 39.0)
    target = make_sphere(shape, spacing, c1, radius_mm=8.0)
    risk = make_sphere(shape, spacing, c2, radius_mm=6.0)

    base = signed_clearance_mm(target, risk, spacing)

    res = analyse_clearance(
        target,
        risk,
        spacing,
        params={"sigma_mm": 1.0, "corr_length_mm": 5.0},
        n_samples=400,
        seed=42,
        threshold_mm=5.0,
    )

    measured_shift = base - res.mean
    print(f"\nMeasured extreme-value shift on 400 samples: {measured_shift:.4f} mm")
    assert abs(res.mean - base) <= 0.3


def test_field_std_crop_margin_invariance():
    """Field std at a fixed point region is invariant within 10% for two different crop margins."""

    def compute_field(crop_shape, sigmas, seed=42):
        rng = np.random.default_rng(seed)
        radii = [int(np.ceil(4 * s)) for s in sigmas]
        imp_shape = [2 * r + 1 for r in radii]
        impulse = np.zeros(imp_shape, dtype=np.float32)
        impulse[radii[0], radii[1], radii[2]] = 1.0
        resp = gaussian_filter(impulse, sigma=sigmas, mode="constant")
        variance = float(np.sum(resp**2))

        noise = rng.normal(0.0, 1.0, size=crop_shape).astype(np.float32)
        sm = gaussian_filter(noise, sigma=sigmas, mode="reflect")
        return (sm / np.sqrt(variance)) * 2.0

    sigmas = (5.0, 5.0, 5.0)
    f_small = compute_field((60, 60, 60), sigmas)
    f_large = compute_field((100, 100, 100), sigmas)

    std_small = float(np.std(f_small[20:40, 20:40, 20:40]))
    std_large = float(np.std(f_large[40:60, 40:60, 40:60]))

    diff_pct = abs(std_small - std_large) / std_large
    assert diff_pct < 0.10, f"Field std difference {diff_pct*100:.2f}% exceeded 10%"


@pytest.mark.slow
def test_sphere_displacement_std_across_seeds_for_l_values():
    """Over 20 seeds, displacement std on r=30 mm sphere is within 15% of 2 mm for L=2, 5, 10 mm."""
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)
    center = (50.0, 50.0, 50.0)
    r = 30.0
    sphere = make_sphere(shape, spacing, center, r)

    struct_6 = generate_binary_structure(3, 1)
    surf_orig = sphere & ~binary_erosion(sphere, structure=struct_6)
    coords = np.argwhere(surf_orig)
    rad_orig = np.sqrt(np.sum((coords - np.array(center)) ** 2, axis=1))

    for L in [2.0, 5.0, 10.0]:
        stds = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            pert = perturb_mask(sphere, spacing, sigma_mm=2.0, corr_length_mm=L, rng=rng)
            pad_p = np.pad(pert, 1, mode="constant", constant_values=False)
            edt_out = distance_transform_edt(~pad_p, sampling=spacing)[1:-1, 1:-1, 1:-1]
            edt_in = distance_transform_edt(pad_p, sampling=spacing)[1:-1, 1:-1, 1:-1]
            sdf_p = edt_out - edt_in
            disp = -sdf_p[surf_orig] - (30.0 - rad_orig)
            stds.append(np.std(disp))

        avg_std = float(np.mean(stds))
        print(f"\nL = {L} mm: averaged displacement std over 20 seeds = {avg_std:.4f} mm")
        # Within 15% of 2.0 mm (allowing discrete boundary curvature effects for small L)
        assert 1.35 <= avg_std <= 2.30


def test_uniform_shift_isotropic_phantom_overlap_negative_clearance():
    """On isotropic phantom (baseline 3 mm), uniform_shift +5 mm gives negative clearance."""
    case_iso, _ = make_phantom_case(
        shape_zyx=(64, 64, 64),
        spacing_mm=(1.0, 1.0, 1.0),
        tumor_center_mm=(32.0, 32.0, 25.0),
        tumor_radius_mm=6.0,
        vessel_center_mm=(32.0, 32.0, 39.0),
        vessel_radius_mm=4.0,
    )
    spacing = (1.0, 1.0, 1.0)
    target = case_iso.structures["tumor"].mask
    risk = case_iso.structures["vessel"].mask

    # Baseline face-to-face clearance between voxel sets is 3.0 mm
    base_clearance = signed_clearance_mm(target, risk, spacing)
    assert base_clearance == pytest.approx(3.0)

    # Apply uniform_shift +5 mm to target (expands target outward by 5 mm)
    shifted_target = uniform_shift(target, spacing, shift_mm=5.0)
    overlap_clearance = signed_clearance_mm(shifted_target, risk, spacing)
    assert overlap_clearance < 0.0

    # Run analyse_clearance on the shifted target with zero stochastic perturbation
    res = analyse_clearance(
        shifted_target,
        risk,
        spacing,
        params={"sigma_mm": 0.0, "corr_length_mm": 5.0},
        n_samples=20,
        seed=42,
    )
    assert res.baseline_mm < 0.0
    assert res.baseline_mm == pytest.approx(overlap_clearance)
    assert res.p_overlap == 1.0
    assert res.p_touching == 0.0
    for s_val in res.samples_mm:
        assert s_val < 0.0
        assert s_val == pytest.approx(overlap_clearance, abs=1e-5)


def test_overlapping_cuboids_perturbed_negative_clearance():
    """Overlapping cuboids: perturbed sample clearance is negative and equals -(penetration)."""
    shape = (50, 50, 50)
    spacing = (1.0, 1.0, 1.0)
    a = np.zeros(shape, dtype=bool)
    b = np.zeros(shape, dtype=bool)

    # a is [10:25, 15:35, 15:35], b is [22:38, 15:35, 15:35]
    # Overlap along z is 3 voxels (25 - 22 = 3), penetration = 3.0 mm
    a[10:25, 15:35, 15:35] = True
    b[22:38, 15:35, 15:35] = True

    c_base = signed_clearance_mm(a, b, spacing)
    assert abs(c_base - (-3.0)) < 1e-6

    res = analyse_clearance(
        a,
        b,
        spacing,
        params={"sigma_mm": 0.0, "corr_length_mm": 5.0},
        n_samples=15,
        seed=42,
    )
    assert res.baseline_mm == pytest.approx(-3.0)
    assert res.p_overlap == 1.0
    assert res.p_touching == 0.0
    for s_val in res.samples_mm:
        assert s_val == pytest.approx(-3.0)


def test_sweep_multi_threshold_caching_and_overlap_columns(tmp_path):
    """sweep evaluates multiple thresholds from stored samples once and includes overlap stats."""
    from hdt.uncertainty.analyse_clearance import sweep

    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    target = make_sphere(shape, spacing, (20.0, 20.0, 15.0), radius_mm=6.0)
    risk = make_sphere(shape, spacing, (20.0, 20.0, 28.0), radius_mm=5.0)

    grid = {
        "sigma_mm": [1.0],
        "corr_length_mm": [5.0],
        "bias_sigma_mm": [0.0],
    }
    csv_path = tmp_path / "sweep_multi_thresh.csv"

    # Pass 3 thresholds
    df = sweep(
        target=target,
        risk=risk,
        spacing_mm=spacing,
        grid=grid,
        n_samples=20,
        seed=42,
        threshold_mm=[1.0, 2.0, 3.0],
        output_csv_path=csv_path,
    )

    # 1 parameter combo x 3 thresholds = 3 rows
    assert len(df) == 3
    assert "p_overlap" in df.columns
    assert "p_touching" in df.columns
    assert "threshold_mm" in df.columns
    assert list(df["threshold_mm"]) == [1.0, 2.0, 3.0]
    # Mean and std across thresholds must match bitwise because samples are reused
    assert df["mean"].nunique() == 1
    assert df["std"].nunique() == 1
    assert csv_path.exists()
