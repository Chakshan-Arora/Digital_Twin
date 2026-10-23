"""Clearance uncertainty analysis and Monte Carlo parameter sweeps.

Conventions:
- All distances in millimetres and face-to-face between voxel sets.
- Randomness strictly controlled via a passed-in seed or Generator.
- Probability values represent "probability under the assumed perturbation model".
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from hdt.geometry.distances import signed_clearance_mm
from hdt.geometry.fast_clearance import PrecomputedRiskClearance
from hdt.uncertainty.perturb_masks import perturb_mask, prepare_perturbation_plan


@dataclass(frozen=True)
class ClearanceResult:
    """Clearance uncertainty analysis results under assumed perturbation model.

    Attributes:
        baseline_mm: Face-to-face clearance between unperturbed masks in mm.
        samples_mm: 1D array of clearance samples across Monte Carlo realizations in mm.
        mean: Mean clearance across realizations in mm.
        std: Sample standard deviation of clearance in mm.
        p05: 5th percentile clearance in mm.
        p50: 50th percentile (median) clearance in mm.
        p95: 95th percentile clearance in mm.
        p_below_threshold: Fraction of realizations with clearance below threshold_mm
            (probability under the assumed perturbation model).
        p_below_ci95: 95% Wilson score confidence interval (ci_low, ci_high) for p_below_threshold.
        n_samples: Total number of Monte Carlo realizations.
        seed: Random seed used to spawn generators.
        threshold_mm: Clearance distance threshold in mm.
        params: Perturbation parameters dict.
        censored_fraction: Fraction of realizations where clearance exceeded max_distance_mm.
        convention: Geometry distance convention ("face_to_face_voxel_sets").
    """

    baseline_mm: float
    samples_mm: np.ndarray
    mean: float
    std: float
    p05: float
    p50: float
    p95: float
    p_below_threshold: float
    p_below_ci95: tuple[float, float]
    p_overlap: float
    p_touching: float
    n_samples: int
    seed: int
    threshold_mm: float
    params: dict[str, Any]
    censored_fraction: float
    convention: str = "face_to_face_voxel_sets"

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary with plain Python types."""
        return {
            "baseline_mm": float(self.baseline_mm),
            "samples_mm": [float(x) for x in self.samples_mm],
            "mean": float(self.mean),
            "std": float(self.std),
            "p05": float(self.p05),
            "p50": float(self.p50),
            "p95": float(self.p95),
            "p_below_threshold": float(self.p_below_threshold),
            "p_below_ci95": [float(self.p_below_ci95[0]), float(self.p_below_ci95[1])],
            "p_overlap": float(self.p_overlap),
            "p_touching": float(self.p_touching),
            "n_samples": int(self.n_samples),
            "seed": int(self.seed),
            "threshold_mm": float(self.threshold_mm),
            "params": {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in self.params.items()
            },
            "censored_fraction": float(self.censored_fraction),
            "convention": str(self.convention),
        }


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes / occurrences.
        n: Total number of Bernoulli trials.
        confidence: Confidence level (default 0.95).

    Returns:
        tuple[float, float]: (ci_low, ci_high) bounded in [0.0, 1.0].
    """
    if n == 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p_hat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    radicand = (p_hat * (1.0 - p_hat) / float(n)) + (z2 / (4.0 * float(n) ** 2))
    margin = (z / denom) * np.sqrt(radicand)

    ci_low = max(0.0, float(center - margin))
    ci_high = min(1.0, float(center + margin))
    return (ci_low, ci_high)


def analyse_clearance(
    target: np.ndarray,
    risk: np.ndarray,
    spacing_mm: Sequence[float],
    params: dict[str, Any],
    n_samples: int = 300,
    seed: int = 42,
    threshold_mm: float = 2.0,
    max_distance_mm: float = 40.0,
    risk_params: dict[str, Any] | None = None,
) -> ClearanceResult:
    """Perform Monte Carlo clearance uncertainty analysis between target and risk masks.

    Both the baseline and every sample realization go through ONE unified function:
    `signed_clearance_mm(..., crop_to_bbox=True)` to guarantee exact convention consistency
    under the face-to-face voxel sets convention.

    Args:
        target: 3D boolean array of target structure (z, y, x).
        risk: 3D boolean array of risk structure (z, y, x).
        spacing_mm: (sz, sy, sx) in millimetres.
        params: Dict of perturbation parameters for target:
            - "sigma_mm": float
            - "corr_length_mm": float
            - "bias_sigma_mm": float (optional, default 0.0)
            - "max_hole_mm3": float (optional, default 10.0)
        n_samples: Number of Monte Carlo realizations (default: 300).
        seed: Master random seed.
        threshold_mm: Clearance distance threshold in mm.
        max_distance_mm: Distance cap; values exceeding this are capped and counted as censored.
        risk_params: Optional dict of perturbation parameters for risk structure.

    Returns:
        ClearanceResult: Summary of clearance uncertainty statistics.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if len(spacing_mm) != 3:
        raise ValueError(f"spacing_mm must have length 3 (z, y, x), got {len(spacing_mm)}")

    sigma_mm = float(params["sigma_mm"])
    corr_length_mm = float(params["corr_length_mm"])
    bias_sigma_mm = float(params.get("bias_sigma_mm", 0.0))
    max_hole_mm3 = float(params.get("max_hole_mm3", 10.0))

    # Baseline face-to-face clearance between unperturbed masks (unified code path)
    baseline_mm = signed_clearance_mm(target, risk, spacing_mm, crop_to_bbox=True)

    # Deterministic independent generator per sample via SeedSequence spawning
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_samples)

    perturb_risk = risk_params is not None
    if perturb_risk:
        risk_sigma = float(risk_params["sigma_mm"])
        risk_corr = float(risk_params["corr_length_mm"])
        risk_bias = float(risk_params.get("bias_sigma_mm", 0.0))
        risk_hole = float(risk_params.get("max_hole_mm3", 10.0))
        risk_plan = prepare_perturbation_plan(
            risk,
            spacing_mm,
            sigma_mm=risk_sigma,
            corr_length_mm=risk_corr,
            bias_sigma_mm=risk_bias,
            max_hole_mm3=risk_hole,
        )

    # Precompute perturbation spatial plan for target
    target_plan = prepare_perturbation_plan(
        target,
        spacing_mm,
        sigma_mm=sigma_mm,
        corr_length_mm=corr_length_mm,
        bias_sigma_mm=bias_sigma_mm,
        max_hole_mm3=max_hole_mm3,
    )

    # Precompute static risk clearance evaluator when risk is unperturbed
    if not perturb_risk:
        risk_eval = PrecomputedRiskClearance(
            risk,
            spacing_mm,
            target_bbox_slices=target_plan.slices,
            max_distance_mm=max_distance_mm,
        )

    samples: list[float] = []
    censored_count = 0

    for i in range(n_samples):
        rng = np.random.default_rng(child_seeds[i])

        t_perturbed = perturb_mask(
            target,
            spacing_mm,
            sigma_mm=sigma_mm,
            corr_length_mm=corr_length_mm,
            rng=rng,
            bias_sigma_mm=bias_sigma_mm,
            max_hole_mm3=max_hole_mm3,
            _plan=target_plan,
        )

        if not perturb_risk:
            c = risk_eval.evaluate(t_perturbed)
        else:
            r_perturbed = perturb_mask(
                risk,
                spacing_mm,
                sigma_mm=risk_sigma,
                corr_length_mm=risk_corr,
                rng=rng,
                bias_sigma_mm=risk_bias,
                max_hole_mm3=risk_hole,
                _plan=risk_plan,
            )
            c = signed_clearance_mm(t_perturbed, r_perturbed, spacing_mm, crop_to_bbox=True)

        if c >= max_distance_mm:
            c = max_distance_mm
            censored_count += 1

        samples.append(c)

    samples_arr = np.array(samples, dtype=np.float64)
    k_below = int(np.count_nonzero(samples_arr < threshold_mm))
    p_below = float(k_below) / float(n_samples)
    ci95 = wilson_score_interval(k_below, n_samples, confidence=0.95)
    k_overlap = int(np.count_nonzero(samples_arr < 0.0))
    p_overlap = float(k_overlap) / float(n_samples)
    k_touching = int(np.count_nonzero(samples_arr == 0.0))
    p_touching = float(k_touching) / float(n_samples)
    censored_frac = float(censored_count) / float(n_samples)

    return ClearanceResult(
        baseline_mm=float(baseline_mm),
        samples_mm=samples_arr,
        mean=float(np.mean(samples_arr)),
        std=float(np.std(samples_arr, ddof=1)) if n_samples > 1 else 0.0,
        p05=float(np.percentile(samples_arr, 5)),
        p50=float(np.percentile(samples_arr, 50)),
        p95=float(np.percentile(samples_arr, 95)),
        p_below_threshold=p_below,
        p_below_ci95=ci95,
        p_overlap=p_overlap,
        p_touching=p_touching,
        n_samples=n_samples,
        seed=seed,
        threshold_mm=float(threshold_mm),
        params=params,
        censored_fraction=censored_frac,
        convention="face_to_face_voxel_sets",
    )


def sweep(
    target: np.ndarray,
    risk: np.ndarray,
    spacing_mm: Sequence[float],
    grid: dict[str, Sequence[float]],
    n_samples: int = 300,
    seed: int = 42,
    threshold_mm: float | Sequence[float] = 2.0,
    output_csv_path: str | Path | None = None,
    max_distance_mm: float = 40.0,
    risk_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a parameter grid sweep for clearance uncertainty analysis.

    Monte Carlo perturbation samples are stored once per parameter configuration,
    and all requested clearance thresholds are evaluated directly from the stored
    samples without re-running the perturbation model.

    Columns in returned DataFrame:
      sigma_mm, corr_length_mm, bias_sigma_mm, mean, std, p05, p95,
      p_below_threshold, ci_low, ci_high, p_overlap, p_touching,
      censored_fraction (and threshold_mm when multiple thresholds evaluated).

    Args:
        target: 3D boolean mask of target (z, y, x).
        risk: 3D boolean mask of risk (z, y, x).
        spacing_mm: (sz, sy, sx) in millimetres.
        grid: Dict mapping param names to sequences:
            - "sigma_mm": list of floats
            - "corr_length_mm": list of floats
            - "bias_sigma_mm": list of floats (optional, default [0.0])
            - "threshold_mm": optional list of floats (overrides threshold_mm arg)
        n_samples: Realizations per configuration (default: 300).
        seed: Random seed.
        threshold_mm: Distance threshold in mm (single float or sequence of floats).
        output_csv_path: Optional CSV output path to save results.
        max_distance_mm: Censoring distance cap.
        risk_params: Optional perturbation parameters for risk structure.

    Returns:
        pd.DataFrame: Sweep summary table.
    """
    sigmas = list(grid.get("sigma_mm", [1.0]))
    corrs = list(grid.get("corr_length_mm", [5.0]))
    biases = list(grid.get("bias_sigma_mm", [0.0]))

    if "threshold_mm" in grid:
        thresholds = [float(t) for t in grid["threshold_mm"]]
    elif isinstance(threshold_mm, (int, float)):
        thresholds = [float(threshold_mm)]
    else:
        thresholds = [float(t) for t in threshold_mm]

    records = []
    for s_val in sigmas:
        for c_val in corrs:
            for b_val in biases:
                params = {
                    "sigma_mm": float(s_val),
                    "corr_length_mm": float(c_val),
                    "bias_sigma_mm": float(b_val),
                }
                # Run Monte Carlo realizations ONCE and cache samples in result
                res = analyse_clearance(
                    target=target,
                    risk=risk,
                    spacing_mm=spacing_mm,
                    params=params,
                    n_samples=n_samples,
                    seed=seed,
                    threshold_mm=thresholds[0],
                    max_distance_mm=max_distance_mm,
                    risk_params=risk_params,
                )

                # Compute all requested thresholds from stored samples without re-running
                for thresh in thresholds:
                    k_thresh = int(np.count_nonzero(res.samples_mm < thresh))
                    p_thresh = float(k_thresh) / float(n_samples)
                    ci95_thresh = wilson_score_interval(k_thresh, n_samples, confidence=0.95)

                    row: dict[str, Any] = {
                        "sigma_mm": float(s_val),
                        "corr_length_mm": float(c_val),
                        "bias_sigma_mm": float(b_val),
                        "mean": float(res.mean),
                        "std": float(res.std),
                        "p05": float(res.p05),
                        "p95": float(res.p95),
                        "p_below_threshold": float(p_thresh),
                        "ci_low": float(ci95_thresh[0]),
                        "ci_high": float(ci95_thresh[1]),
                        "p_overlap": float(res.p_overlap),
                        "p_touching": float(res.p_touching),
                        "censored_fraction": float(res.censored_fraction),
                    }
                    if len(thresholds) > 1 or "threshold_mm" in grid:
                        row["threshold_mm"] = float(thresh)
                    records.append(row)

    df = pd.DataFrame.from_records(records)
    if output_csv_path is not None:
        p = Path(output_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df
