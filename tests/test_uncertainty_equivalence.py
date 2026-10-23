"""Regression and numerical equivalence tests comparing optimized uncertainty to reference.

Verifies:
  1. Masks are bitwise equal to tests/_reference_uncertainty.py for seeds 0..4.
  2. Sample clearances match within 1e-6 mm under all geometric cases.
  3. Sample i is unchanged between n_samples=5 and n_samples=20.
"""

import math

import numpy as np
import pytest
from _reference_uncertainty import reference_sample_clearance

from hdt.api.repository import PhantomRepository
from hdt.geometry.fast_clearance import PrecomputedRiskClearance
from hdt.uncertainty.analyse_clearance import analyse_clearance
from hdt.uncertainty.perturb_masks import perturb_mask, prepare_perturbation_plan


@pytest.fixture
def default_params() -> dict:
    return {
        "sigma_mm": 1.0,
        "corr_length_mm": 5.0,
        "bias_sigma_mm": 0.0,
        "max_hole_mm3": 10.0,
    }


def test_equivalence_on_phantom(default_params: dict) -> None:
    """Verify bitwise mask equality and clearance within 1e-6 mm on anisotropic oblique phantom."""
    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")
    target = case.structures["tumor"].mask
    risk = case.structures["vessel"].mask
    spacing = geom.spacing_mm

    plan = prepare_perturbation_plan(
        target,
        spacing,
        default_params["sigma_mm"],
        default_params["corr_length_mm"],
        default_params["bias_sigma_mm"],
        default_params["max_hole_mm3"],
    )
    risk_eval = PrecomputedRiskClearance(risk, spacing, target_bbox_slices=plan.slices)

    for seed in range(5):
        # Reference
        rng_ref = np.random.default_rng(seed)
        ref_mask, ref_c = reference_sample_clearance(target, risk, spacing, default_params, rng_ref)

        # Optimized
        rng_fast = np.random.default_rng(seed)
        fast_mask = perturb_mask(
            target,
            spacing,
            default_params["sigma_mm"],
            default_params["corr_length_mm"],
            rng_fast,
            default_params["bias_sigma_mm"],
            default_params["max_hole_mm3"],
            _plan=plan,
        )
        fast_c = risk_eval.evaluate(fast_mask)

        # Bitwise equal masks
        assert np.array_equal(fast_mask, ref_mask), f"Phantom mask mismatch on seed {seed}"
        # Clearance tolerance 1e-6 mm
        assert math.isclose(fast_c, ref_c, abs_tol=1e-6), (
            f"Phantom clearance mismatch on seed {seed}: fast={fast_c}, ref={ref_c}"
        )


def test_equivalence_on_cuboid_ladder(default_params: dict) -> None:
    """Verify bitwise equality and clearance within 1e-6 mm on disjoint cuboid ladder case."""
    shape = (48, 48, 48)
    spacing = (1.5, 1.0, 0.8)
    target = np.zeros(shape, dtype=bool)
    risk = np.zeros(shape, dtype=bool)

    # Cuboids separated along z axis by 5 empty voxels
    target[10:20, 15:30, 15:30] = True
    risk[25:35, 15:30, 15:30] = True

    plan = prepare_perturbation_plan(
        target,
        spacing,
        default_params["sigma_mm"],
        default_params["corr_length_mm"],
        default_params["bias_sigma_mm"],
        default_params["max_hole_mm3"],
    )
    risk_eval = PrecomputedRiskClearance(risk, spacing, target_bbox_slices=plan.slices)

    for seed in range(5):
        rng_ref = np.random.default_rng(seed)
        ref_mask, ref_c = reference_sample_clearance(target, risk, spacing, default_params, rng_ref)

        rng_fast = np.random.default_rng(seed)
        fast_mask = perturb_mask(
            target,
            spacing,
            default_params["sigma_mm"],
            default_params["corr_length_mm"],
            rng_fast,
            default_params["bias_sigma_mm"],
            default_params["max_hole_mm3"],
            _plan=plan,
        )
        fast_c = risk_eval.evaluate(fast_mask)

        assert np.array_equal(fast_mask, ref_mask), f"Ladder mask mismatch on seed {seed}"
        assert math.isclose(fast_c, ref_c, abs_tol=1e-6), (
            f"Ladder clearance mismatch on seed {seed}: fast={fast_c}, ref={ref_c}"
        )


def test_equivalence_on_overlapping_case(default_params: dict) -> None:
    """Verify bitwise equality and negative clearance within 1e-6 mm on overlapping case."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.2, 0.9)
    target = np.zeros(shape, dtype=bool)
    risk = np.zeros(shape, dtype=bool)

    # Overlapping region
    target[10:25, 10:25, 10:25] = True
    risk[18:32, 18:32, 18:32] = True

    plan = prepare_perturbation_plan(
        target,
        spacing,
        default_params["sigma_mm"],
        default_params["corr_length_mm"],
        default_params["bias_sigma_mm"],
        default_params["max_hole_mm3"],
    )
    risk_eval = PrecomputedRiskClearance(risk, spacing, target_bbox_slices=plan.slices)

    for seed in range(5):
        rng_ref = np.random.default_rng(seed)
        ref_mask, ref_c = reference_sample_clearance(target, risk, spacing, default_params, rng_ref)

        rng_fast = np.random.default_rng(seed)
        fast_mask = perturb_mask(
            target,
            spacing,
            default_params["sigma_mm"],
            default_params["corr_length_mm"],
            rng_fast,
            default_params["bias_sigma_mm"],
            default_params["max_hole_mm3"],
            _plan=plan,
        )
        fast_c = risk_eval.evaluate(fast_mask)

        assert np.array_equal(fast_mask, ref_mask), f"Overlap mask mismatch on seed {seed}"
        assert math.isclose(fast_c, ref_c, abs_tol=1e-6), (
            f"Overlap clearance mismatch on seed {seed}: fast={fast_c}, ref={ref_c}"
        )


def test_sample_i_unchanged_between_n5_and_n20(default_params: dict) -> None:
    """Verify that sample i is identical regardless of whether n_samples is 5 or 20."""
    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")
    target = case.structures["tumor"].mask
    risk = case.structures["vessel"].mask
    spacing = geom.spacing_mm

    res_5 = analyse_clearance(
        target=target,
        risk=risk,
        spacing_mm=spacing,
        params=default_params,
        n_samples=5,
        seed=123,
    )

    res_20 = analyse_clearance(
        target=target,
        risk=risk,
        spacing_mm=spacing,
        params=default_params,
        n_samples=20,
        seed=123,
    )

    assert math.isclose(res_5.baseline_mm, res_20.baseline_mm, abs_tol=1e-9)
    assert len(res_5.samples_mm) == 5
    assert len(res_20.samples_mm) == 20

    # First 5 samples must be numerically identical within 1e-6 mm
    for i in range(5):
        assert math.isclose(res_5.samples_mm[i], res_20.samples_mm[i], abs_tol=1e-6), (
            f"Sample {i} mismatch: {res_5.samples_mm[i]} vs {res_20.samples_mm[i]}"
        )
