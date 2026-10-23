"""Benchmark script comparing uncropped, cropped per call, and precomputed cropped SDF.

Volume size: (155, 240, 240) (standard Brain MRI / BraTS dimension in z, y, x).
Queries: 100 perturbed-target clearance evaluations.
"""

import time

import numpy as np

from hdt.geometry.distances import (
    clearance_from_sdf,
    signed_clearance_mm,
    signed_distance_field_mm,
)
from hdt.imaging.phantoms import make_sphere


def run_benchmark():
    shape = (155, 240, 240)
    spacing = (1.0, 1.0, 1.0)
    total_voxels = shape[0] * shape[1] * shape[2]
    print(f"Creating phantom volume with shape {shape} (total voxels: {total_voxels:,})...")

    # Fixed risk structure (e.g. brainstem / optic chiasm, radius 10 mm)
    c_risk = (75.0, 120.0, 140.0)
    risk_mask = make_sphere(shape, spacing, c_risk, radius_mm=10.0)

    # Base target structure (e.g. tumor, radius 15 mm)
    base_target_c = np.array([75.0, 120.0, 100.0])

    # Generate 100 perturbed targets
    rng = np.random.default_rng(42)
    n_queries = 100
    perturbed_targets = []
    for _ in range(n_queries):
        offset = rng.uniform(-2.0, 2.0, size=3)
        c = tuple(base_target_c + offset)
        perturbed_targets.append(make_sphere(shape, spacing, c, radius_mm=15.0))

    print(f"Prepared {n_queries} perturbed target masks.")
    print("Warmup call...")
    _ = signed_clearance_mm(perturbed_targets[0], risk_mask, spacing, crop_to_bbox=False)

    # (a) Uncropped: 100 queries
    t0 = time.perf_counter()
    for t_mask in perturbed_targets:
        _ = signed_clearance_mm(t_mask, risk_mask, spacing, crop_to_bbox=False)
    t_a_total = time.perf_counter() - t0
    t_a_per_call = (t_a_total / n_queries) * 1000.0

    # (b) Cropped per call: 100 queries
    t0 = time.perf_counter()
    for t_mask in perturbed_targets:
        _ = signed_clearance_mm(
            t_mask, risk_mask, spacing, crop_to_bbox=True, bbox_margin_voxels=5
        )
    t_b_total = time.perf_counter() - t0
    t_b_per_call = (t_b_total / n_queries) * 1000.0

    # (c) Cropped SDF with precompute INCLUDED: precompute once + 100 queries
    t0 = time.perf_counter()
    cropped_sdf = signed_distance_field_mm(risk_mask, spacing, max_distance_mm=60.0)
    t_precompute = time.perf_counter() - t0
    t1 = time.perf_counter()
    for t_mask in perturbed_targets:
        _ = clearance_from_sdf(t_mask, cropped_sdf)
    t_queries_only = time.perf_counter() - t1
    t_c_total = time.perf_counter() - t0
    t_c_amortised = (t_c_total / n_queries) * 1000.0

    sp_b = t_a_total / max(t_b_total, 1e-6)
    sp_c = t_a_total / max(t_c_total, 1e-6)
    print("\n=== Benchmark Results (100 Perturbed-Target Calls on 155x240x240 Volume) ===")
    print(f"(a) Uncropped:            {t_a_total:.2f} s ({t_a_per_call:.2f} ms/call)")
    print(f"(b) Cropped per call:     {t_b_total:.2f} s ({t_b_per_call:.2f} ms/call) [{sp_b:.1f}x]")
    print(f"(c) Cropped SDF:        {t_c_total:.2f} s ({t_c_amortised:.2f} ms/call) [{sp_c:.1f}x]")
    print(f"    - Precompute time:    {t_precompute*1000:.2f} ms")
    print(f"    - Query time / call:  {(t_queries_only/n_queries)*1000:.3f} ms")


if __name__ == "__main__":
    run_benchmark()
