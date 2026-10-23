"""Unit and integration tests for clearance uncertainty analysis."""

import json

import numpy as np

from hdt.imaging.phantoms import make_sphere
from hdt.uncertainty.analyse_clearance import (
    analyse_clearance,
    sweep,
    wilson_score_interval,
)


def test_sample_i_identical_regardless_of_n_samples():
    """Sample i is identical for n_samples = 20 and n_samples = 50."""
    shape = (50, 50, 50)
    spacing = (1.0, 1.0, 1.0)
    target = make_sphere(shape, spacing, (25.0, 25.0, 15.0), radius_mm=8.0)
    risk = make_sphere(shape, spacing, (25.0, 25.0, 35.0), radius_mm=6.0)

    params = {"sigma_mm": 1.0, "corr_length_mm": 3.0, "bias_sigma_mm": 0.0}
    seed = 42
    threshold_mm = 5.0

    res_20 = analyse_clearance(
        target, risk, spacing, params, n_samples=20, seed=seed, threshold_mm=threshold_mm
    )
    res_50 = analyse_clearance(
        target, risk, spacing, params, n_samples=50, seed=seed, threshold_mm=threshold_mm
    )

    # First 20 samples must match bitwise exactly
    np.testing.assert_array_equal(res_20.samples_mm, res_50.samples_mm[:20])


def test_wilson_interval_narrows_with_larger_n():
    """Wilson interval narrows as n grows (n = 50 -> 400)."""
    # Fix k proportional to n, say 20%
    ci_50 = wilson_score_interval(k=10, n=50, confidence=0.95)
    width_50 = ci_50[1] - ci_50[0]

    ci_400 = wilson_score_interval(k=80, n=400, confidence=0.95)
    width_400 = ci_400[1] - ci_400[0]

    assert (
        width_400 < width_50
    ), f"Width at n=400 ({width_400:.4f}) must be smaller than at n=50 ({width_50:.4f})"


def test_to_dict_passes_json_dumps():
    """to_dict() passes json.dumps with plain JSON-serializable types."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    target = make_sphere(shape, spacing, (20.0, 20.0, 15.0), radius_mm=6.0)
    risk = make_sphere(shape, spacing, (20.0, 20.0, 28.0), radius_mm=5.0)

    params = {"sigma_mm": 1.0, "corr_length_mm": 3.0}
    res = analyse_clearance(
        target, risk, spacing, params, n_samples=10, seed=123, threshold_mm=4.0
    )

    d = res.to_dict()
    dumped = json.dumps(d)
    loaded = json.loads(dumped)

    assert loaded["n_samples"] == 10
    assert loaded["convention"] == "face_to_face_voxel_sets"
    assert isinstance(loaded["samples_mm"], list)
    assert len(loaded["samples_mm"]) == 10
    assert isinstance(loaded["p_below_ci95"], list)
    assert len(loaded["p_below_ci95"]) == 2


def test_sweep_generates_dataframe_and_saves_csv(tmp_path):
    """sweep generates expected columns and writes to CSV file."""
    shape = (40, 40, 40)
    spacing = (1.0, 1.0, 1.0)
    target = make_sphere(shape, spacing, (20.0, 20.0, 15.0), radius_mm=6.0)
    risk = make_sphere(shape, spacing, (20.0, 20.0, 28.0), radius_mm=5.0)

    grid = {
        "sigma_mm": [1.0, 2.0],
        "corr_length_mm": [3.0],
        "bias_sigma_mm": [0.0],
    }
    csv_file = tmp_path / "sweep_results.csv"

    df = sweep(
        target=target,
        risk=risk,
        spacing_mm=spacing,
        grid=grid,
        n_samples=5,
        seed=42,
        threshold_mm=3.0,
        output_csv_path=csv_file,
    )

    assert len(df) == 2
    expected_cols = [
        "sigma_mm",
        "corr_length_mm",
        "bias_sigma_mm",
        "mean",
        "std",
        "p05",
        "p95",
        "p_below_threshold",
        "ci_low",
        "ci_high",
        "censored_fraction",
    ]
    for col in expected_cols:
        assert col in df.columns

    assert csv_file.exists()
