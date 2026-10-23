"""Dermatome pick accuracy measurement benchmark across stature presets.

Evaluates nearest_dermatome() against 200 synthetic surface points per dermatome
per side generated between stored skin anchors.

Research prototype - NOT a clinical diagnostic tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from hdt.body.models import BodyParams
from hdt.body.presets import PRESETS
from hdt.nervous.anchors import DERMATOME_ANCHORS, nearest_dermatome, skin_samples

# ---------------------------------------------------------------------------
# Anatomical Spinal Level Order & Adjacency Definition
# ---------------------------------------------------------------------------
SPINAL_ROOT_ORDER: list[str] = [
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
]

# Developmental axial boundary adjacencies across embryonic limb bud frontiers
AXIAL_LINE_ADJACENCIES: dict[str, set[str]] = {
    "C4": {"T2"},
    "T2": {"C4"},
    "L1": {"S2"},
    "S2": {"L1"},
}


def are_dermatomes_adjacent(d1: str, d2: str) -> bool:
    """Determine whether two dermatomes are clinically or anatomically adjacent.

    Adjacency Rule
    --------------
    Two dermatomes d1 and d2 are considered adjacent if:
      1. They represent the identical dermatome (d1 == d2).
      2. They occupy adjacent contiguous levels in the cranio-caudal spinal column
         (|order_index(d1) - order_index(d2)| <= 1).
      3. They border each other across developmental axial lines where embryonic
         limb buds disrupted contiguous cranial-caudal innervation (e.g. C4-T2 across
         the superior anterior chest, or L1-S2 across the proximal inguinal-gluteal fold).
    """
    if d1 == d2:
        return True

    if d1 in SPINAL_ROOT_ORDER and d2 in SPINAL_ROOT_ORDER:
        idx1 = SPINAL_ROOT_ORDER.index(d1)
        idx2 = SPINAL_ROOT_ORDER.index(d2)
        if abs(idx1 - idx2) <= 1:
            return True

    if d1 in AXIAL_LINE_ADJACENCIES and d2 in AXIAL_LINE_ADJACENCIES[d1]:
        return True

    return False


def generate_surface_points_for_dermatome(
    body: BodyParams,
    dermatome_id: str,
    side: Literal["l", "r"],
    n_points: int,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    """Generate points strictly on the body surface between stored skin samples."""
    samples = skin_samples(body, dermatome_id, side=side, n=6)
    m = len(samples)

    points: list[tuple[float, float, float]] = []

    for _ in range(n_points):
        # Pick two stored samples
        i, j = rng.choice(m, size=2, replace=False) if m >= 2 else (0, 0)
        p1 = samples[i]
        p2 = samples[j]

        # Convex combination along surface between samples
        t = float(rng.uniform(0.05, 0.95))
        interp_x = (1.0 - t) * p1[0] + t * p2[0]
        interp_y = (1.0 - t) * p1[1] + t * p2[1]
        interp_z = (1.0 - t) * p1[2] + t * p2[2]

        # Add small surface tangential perturbation
        dx = float(rng.normal(0.0, 1.5))
        dy = float(rng.normal(0.0, 1.5))
        dz = float(rng.normal(0.0, 1.5))

        px = interp_x + dx
        py = interp_y + dy
        pz = interp_z + dz

        # Maintain hemispheric lateralisation sign
        if side == "l" and px >= -1.0:
            px = -1.5
        elif side == "r" and px <= 1.0:
            px = 1.5

        points.append((round(px, 2), round(py, 2), round(pz, 2)))

    return points


def measure_preset_accuracy(
    preset_name: str,
    body: BodyParams,
    dermatome_ids: list[str],
    rng: np.random.Generator,
    n_points_per_dermatome: int = 200,
) -> dict:
    """Measure exact and adjacent dermatome picking rates for a stature preset."""
    total_evals = 0
    total_exact = 0
    total_adjacent = 0

    per_dermatome_exact: dict[str, int] = {d_id: 0 for d_id in dermatome_ids}
    per_dermatome_evals: dict[str, int] = {d_id: 0 for d_id in dermatome_ids}

    for d_id in dermatome_ids:
        for s in ("l", "r"):
            pts = generate_surface_points_for_dermatome(
                body, d_id, s, n_points_per_dermatome, rng
            )

            for pt in pts:
                pred_id, pred_side, _ = nearest_dermatome(body, pt)

                is_exact = (pred_id == d_id) and (pred_side == s)
                is_adj = are_dermatomes_adjacent(pred_id, d_id) and (pred_side == s)

                total_evals += 1
                per_dermatome_evals[d_id] += 1

                if is_exact:
                    total_exact += 1
                    per_dermatome_exact[d_id] += 1
                if is_adj:
                    total_adjacent += 1

    overall_exact_rate = total_exact / total_evals if total_evals > 0 else 0.0
    overall_adjacent_rate = total_adjacent / total_evals if total_evals > 0 else 0.0

    # 10 worst dermatomes by exact match rate
    dermatome_rates = [
        (d_id, per_dermatome_exact[d_id] / per_dermatome_evals[d_id])
        for d_id in dermatome_ids
        if per_dermatome_evals[d_id] > 0
    ]
    dermatome_rates.sort(key=lambda item: item[1])
    worst_10 = dermatome_rates[:10]

    return {
        "preset": preset_name,
        "height_mm": body.height_mm,
        "total_evals": total_evals,
        "exact_rate": overall_exact_rate,
        "adjacent_rate": overall_adjacent_rate,
        "worst_10": worst_10,
    }


def main() -> None:
    """Execute pick accuracy benchmark across all presets and write report."""
    rng = np.random.default_rng(seed=42)
    dermatome_ids = list(DERMATOME_ANCHORS.keys())

    report_lines: list[str] = [
        "=" * 78,
        "BODY EXPLORER DERMATOME PICK ACCURACY BENCHMARK REPORT",
        "Evaluated with 200 synthetic surface test points per dermatome per side",
        "Adjacency Rule: identical dermatome, |order difference| <= 1 in cranial-caudal column,",
        "or axial boundary line transitions (C4-T2, L1-S2).",
        "=" * 78,
        "",
    ]

    for preset_name in ("short", "average", "tall"):
        body = PRESETS[preset_name]
        res = measure_preset_accuracy(
            preset_name=preset_name,
            body=body,
            dermatome_ids=dermatome_ids,
            rng=rng,
            n_points_per_dermatome=200,
        )

        preset_header = (
            f"Preset '{res['preset'].upper()}' (Height: {res['height_mm']:.0f} mm, "
            f"N={res['total_evals']} queries):"
        )
        exact_line = f"  Exact Intended Dermatome Match Rate: {res['exact_rate'] * 100:.2f}%"
        adj_line = f"  Intended OR Adjacent Dermatome Match Rate: {res['adjacent_rate'] * 100:.2f}%"

        worst_lines = ["  Top 10 Lowest Accuracy Dermatomes:"]
        for rank, (d_id, rate) in enumerate(res["worst_10"], 1):
            worst_lines.append(f"    {rank:2d}. Dermatome {d_id:4s}: {rate * 100:.2f}% exact")

        report_lines.append(preset_header)
        report_lines.append(exact_line)
        report_lines.append(adj_line)
        report_lines.extend(worst_lines)
        report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to logs/pick_accuracy.txt
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "pick_accuracy.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"[Done] Report saved to {out_path}")


if __name__ == "__main__":
    main()
