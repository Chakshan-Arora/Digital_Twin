"""Profile clearance uncertainty on real-size and phantom cases.

Runs cProfile and detailed timing breakdowns on:
  (a) 0274-1 vs copy shifted 10 mm
  (b) 0274-1 vs copy shifted 60 mm (disjoint)
  (c) Phantom (phantom_001 tumor vs vessel)
  (d) 0087-1's GTV vs copy shifted 60 mm
"""

import cProfile
import io
import pstats
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.ndimage
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from hdt.api.repository import PhantomRepository
from hdt.geometry.distances import signed_clearance_mm
from hdt.geometry.fast_clearance import PrecomputedRiskClearance
from hdt.imaging.load_scan import load_nifti_mask_array
from hdt.uncertainty.perturb_masks import perturb_mask, prepare_perturbation_plan
from tests._reference_uncertainty import reference_sample_clearance


def compute_stage_breakdown(
    target: np.ndarray,
    risk: np.ndarray,
    spacing_mm: Sequence[float],
    params: dict,
    rng: np.random.Generator,
) -> dict:
    """Run one sample instrumented with fine-grained timers to measure time splits."""
    sz, sy, sx = (float(s) for s in spacing_mm)
    spacing = (sz, sy, sx)
    max_sp = max(spacing)
    sigma_mm = float(params["sigma_mm"])
    corr_length_mm = float(params["corr_length_mm"])
    bias_sigma_mm = float(params.get("bias_sigma_mm", 0.0))
    max_hole_mm3 = float(params.get("max_hole_mm3", 10.0))

    t_start = time.perf_counter()

    # Crop calculation
    margin_mm = 4.0 * sigma_mm + 3.0 * corr_length_mm + 2.0 * max_sp
    margin_vox = [int(np.ceil(margin_mm / s)) for s in spacing]

    coords = np.argwhere(target)
    min_idx = np.maximum(coords.min(axis=0) - margin_vox, 0)
    max_idx = np.minimum(coords.max(axis=0) + margin_vox + 1, target.shape)

    slices = (
        slice(int(min_idx[0]), int(max_idx[0])),
        slice(int(min_idx[1]), int(max_idx[1])),
        slice(int(min_idx[2]), int(max_idx[2])),
    )
    crop_mask = target[slices]
    perturb_bbox_voxels = crop_mask.size
    perturb_bbox_shape = crop_mask.shape

    # 1. SDF on crop
    t_sdf_start = time.perf_counter()
    pad_crop = np.pad(crop_mask, 1, mode="constant", constant_values=False)
    edt_out = distance_transform_edt(~pad_crop, sampling=spacing)[1:-1, 1:-1, 1:-1]
    edt_in = distance_transform_edt(pad_crop, sampling=spacing)[1:-1, 1:-1, 1:-1]
    sdf = (edt_out - edt_in).astype(np.float32)
    t_sdf_end = time.perf_counter()

    # 2. Field generation
    t_field_start = time.perf_counter()
    filter_sigmas = tuple(corr_length_mm / s for s in spacing)
    radii = [int(np.ceil(4 * s)) for s in filter_sigmas]
    imp_shape = [2 * r + 1 for r in radii]
    impulse = np.zeros(imp_shape, dtype=np.float32)
    impulse[radii[0], radii[1], radii[2]] = 1.0
    resp = gaussian_filter(impulse, sigma=filter_sigmas, mode="constant")
    variance = float(np.sum(resp**2))

    white_noise = rng.normal(0.0, 1.0, size=crop_mask.shape).astype(np.float32)
    smoothed = gaussian_filter(white_noise, sigma=filter_sigmas, mode="reflect")
    field = (smoothed / np.sqrt(variance)) * float(sigma_mm)
    t_field_end = time.perf_counter()

    # 3. Thresholding
    t_thresh_start = time.perf_counter()
    bias = float(rng.normal(0.0, bias_sigma_mm)) if bias_sigma_mm > 0.0 else 0.0
    band_half_width = 4.0 * sigma_mm + 2.0 * max_sp
    band = np.abs(sdf) < band_half_width

    perturbed_crop = crop_mask.copy()
    perturbed_crop[band] = (sdf[band] + field[band] + bias) <= 0.0
    t_thresh_end = time.perf_counter()

    # 4. Post-processing: component filtering & hole filling
    t_post_start = time.perf_counter()
    labeled_crop, n_comp = label(perturbed_crop)
    if n_comp > 0:
        retained = np.zeros_like(perturbed_crop)
        for comp_idx in range(1, n_comp + 1):
            comp_mask = labeled_crop == comp_idx
            if np.any(comp_mask & crop_mask):
                retained |= comp_mask
        perturbed_crop = retained

    voxel_vol_mm3 = sz * sy * sx
    max_hole_voxels = int(np.floor(max_hole_mm3 / voxel_vol_mm3))
    if max_hole_voxels > 0:
        filled = binary_fill_holes(perturbed_crop)
        holes = filled & ~perturbed_crop
        if np.any(holes):
            labeled_holes, n_holes = label(holes)
            for h_idx in range(1, n_holes + 1):
                h_mask = labeled_holes == h_idx
                if np.count_nonzero(h_mask) <= max_hole_voxels:
                    perturbed_crop[h_mask] = True

    t_perturbed = target.copy()
    t_perturbed[slices] = perturbed_crop
    t_post_end = time.perf_counter()

    # 5. Clearance step
    # Determine clearance bounding box
    coords_a = np.argwhere(t_perturbed)
    coords_b = np.argwhere(risk)
    if len(coords_a) > 0 and len(coords_b) > 0:
        min_c = np.minimum(coords_a.min(axis=0), coords_b.min(axis=0))
        max_c = np.maximum(coords_a.max(axis=0), coords_b.max(axis=0))
        clearance_bbox_shape = tuple((max_c - min_c + 1).tolist())
        clearance_bbox_voxels = int(np.prod(clearance_bbox_shape))
    else:
        clearance_bbox_shape = target.shape
        clearance_bbox_voxels = target.size

    t_clearance_start = time.perf_counter()
    c = signed_clearance_mm(t_perturbed, risk, spacing, crop_to_bbox=True)
    t_clearance_end = time.perf_counter()

    t_total = time.perf_counter() - t_start

    return {
        "t_perturbed": t_perturbed,
        "clearance_mm": c,
        "perturb_bbox_shape": perturb_bbox_shape,
        "perturb_bbox_voxels": perturb_bbox_voxels,
        "clearance_bbox_shape": clearance_bbox_shape,
        "clearance_bbox_voxels": clearance_bbox_voxels,
        "sdf_time_ms": (t_sdf_end - t_sdf_start) * 1000.0,
        "field_time_ms": (t_field_end - t_field_start) * 1000.0,
        "thresh_time_ms": (t_thresh_end - t_thresh_start) * 1000.0,
        "post_time_ms": (t_post_end - t_post_start) * 1000.0,
        "clearance_time_ms": (t_clearance_end - t_clearance_start) * 1000.0,
        "total_time_ms": t_total * 1000.0,
    }


def profile_case(name: str, target: np.ndarray, risk: np.ndarray, spacing_mm: tuple, params: dict):
    """Run cProfile and timing breakdown for one case."""
    print("=" * 78)
    print(f"CASE: {name}")
    print(f"Full volume shape: {target.shape}, Spacing: {spacing_mm}")
    print(
        f"Target voxels: {int(np.count_nonzero(target))}, "
        f"Risk voxels: {int(np.count_nonzero(risk))}"
    )
    print("-" * 78)

    # 1. cProfile run
    seed = 42
    rng = np.random.default_rng(seed)

    pr = cProfile.Profile()
    pr.enable()
    _ = reference_sample_clearance(target, risk, spacing_mm, params, rng)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)

    print("TOP 15 FUNCTIONS BY CUMULATIVE TIME (cProfile):")
    for line in s.getvalue().splitlines()[:20]:
        print(line)

    # 2. Detailed stage breakdown & bounding box voxel counts
    rng_breakdown = np.random.default_rng(seed)
    breakdown = compute_stage_breakdown(target, risk, spacing_mm, params, rng_breakdown)

    print("-" * 78)
    print("BOUNDING BOX VOXEL COUNTS:")
    print(f"  perturb_mask crop shape:  {breakdown['perturb_bbox_shape']}")
    print(
        f"  perturb_mask crop voxels: {breakdown['perturb_bbox_voxels']:,} voxels "
        f"({breakdown['perturb_bbox_voxels'] / target.size * 100:.1f}% of volume)"
    )
    print(f"  clearance bbox shape:     {breakdown['clearance_bbox_shape']}")
    print(
        f"  clearance bbox voxels:    {breakdown['clearance_bbox_voxels']:,} voxels "
        f"({breakdown['clearance_bbox_voxels'] / target.size * 100:.1f}% of volume)"
    )

    print("-" * 78)
    print("TIME SPLIT:")
    total = breakdown["total_time_ms"]
    print(
        f"  SDF on crop:               {breakdown['sdf_time_ms']:8.2f} ms "
        f"({breakdown['sdf_time_ms'] / total * 100:5.1f}%)"
    )
    print(
        f"  Field generation:          {breakdown['field_time_ms']:8.2f} ms "
        f"({breakdown['field_time_ms'] / total * 100:5.1f}%)"
    )
    print(
        f"  Thresholding:              {breakdown['thresh_time_ms']:8.2f} ms "
        f"({breakdown['thresh_time_ms'] / total * 100:5.1f}%)"
    )
    print(
        f"  Component/hole post-proc:  {breakdown['post_time_ms']:8.2f} ms "
        f"({breakdown['post_time_ms'] / total * 100:5.1f}%)"
    )
    print(
        f"  Clearance call:            {breakdown['clearance_time_ms']:8.2f} ms "
        f"({breakdown['clearance_time_ms'] / total * 100:5.1f}%)"
    )
    print(
        f"  TOTAL REFERENCE SAMPLE TIME: {breakdown['total_time_ms']:8.2f} ms "
        f"(clearance={breakdown['clearance_mm']:.4f} mm)"
    )

    # 3. Optimized per-sample time measurement
    plan = prepare_perturbation_plan(
        target,
        spacing_mm,
        params["sigma_mm"],
        params["corr_length_mm"],
        params.get("bias_sigma_mm", 0.0),
        params.get("max_hole_mm3", 10.0),
    )
    risk_eval = PrecomputedRiskClearance(risk, spacing_mm, target_bbox_slices=plan.slices)
    rng_opt = np.random.default_rng(seed)

    t_opt_0 = time.perf_counter()
    opt_mask = perturb_mask(
        target,
        spacing_mm,
        params["sigma_mm"],
        params["corr_length_mm"],
        rng_opt,
        params.get("bias_sigma_mm", 0.0),
        params.get("max_hole_mm3", 10.0),
        _plan=plan,
    )
    opt_c = risk_eval.evaluate(opt_mask)
    t_opt_1 = time.perf_counter()
    opt_time_ms = (t_opt_1 - t_opt_0) * 1000.0

    print("-" * 78)
    print(
        f"OPTIMIZED PER-SAMPLE TIME:   {opt_time_ms:8.2f} ms "
        f"(speedup: {breakdown['total_time_ms'] / opt_time_ms:.1f}x)"
    )
    print(f"  Mask bitwise equal:        {np.array_equal(opt_mask, breakdown['t_perturbed'])}")
    print(f"  Clearance difference:      {abs(opt_c - breakdown['clearance_mm']):.2e} mm")
    print()


def main():
    params = {
        "sigma_mm": 1.0,
        "corr_length_mm": 5.0,
        "bias_sigma_mm": 0.0,
        "max_hole_mm3": 10.0,
    }

    # Case (a) & (b): BraTS-MEN-RT-0274-1
    case_0274_path = Path("data/processed/cases/BraTS-MEN-RT-0274-1/structures/gtv.nii.gz")
    if not case_0274_path.is_file():
        case_0274_path = Path("data/raw/brats/BraTS-MEN-RT-0274-1/BraTS-MEN-RT-0274-1_gtv.nii.gz")
    gtv_0274, geom_0274, _ = load_nifti_mask_array(case_0274_path)
    gtv_0274_bool = np.array(gtv_0274 > 0, dtype=bool)
    sp_0274 = geom_0274.spacing_mm

    # Shift 10 mm along x (axis 2)
    shift_10_vox = int(round(10.0 / sp_0274[2]))
    risk_0274_10mm = scipy.ndimage.shift(gtv_0274_bool, (0, 0, shift_10_vox), order=0, cval=False)

    # Shift 60 mm along x (axis 2)
    shift_60_vox = int(round(60.0 / sp_0274[2]))
    risk_0274_60mm = scipy.ndimage.shift(gtv_0274_bool, (0, 0, shift_60_vox), order=0, cval=False)

    # (a) 0274-1 vs copy shifted 10 mm
    profile_case(
        "(a) 0274-1 vs copy shifted 10 mm",
        gtv_0274_bool,
        risk_0274_10mm,
        sp_0274,
        params,
    )

    # (b) 0274-1 vs copy shifted 60 mm (disjoint)
    profile_case(
        "(b) 0274-1 vs copy shifted 60 mm (disjoint)",
        gtv_0274_bool,
        risk_0274_60mm,
        sp_0274,
        params,
    )

    # (c) The phantom
    phantom_repo = PhantomRepository()
    phantom_case, phantom_geom = phantom_repo.get_case("phantom_001")
    tumor = phantom_case.structures["tumor"].mask
    vessel = phantom_case.structures["vessel"].mask
    sp_phantom = phantom_geom.spacing_mm
    profile_case("(c) Phantom (tumor vs vessel)", tumor, vessel, sp_phantom, params)

    # (d) 0087-1's GTV vs copy shifted 60 mm
    case_0087_path = Path("data/raw/brats/BraTS-MEN-RT-0087-1/BraTS-MEN-RT-0087-1_gtv.nii.gz")
    gtv_0087, geom_0087, _ = load_nifti_mask_array(case_0087_path)
    gtv_0087_bool = np.array(gtv_0087 > 0, dtype=bool)
    sp_0087 = geom_0087.spacing_mm
    shift_60_vox_0087 = int(round(60.0 / sp_0087[2]))
    risk_0087_60mm = scipy.ndimage.shift(
        gtv_0087_bool, (0, 0, shift_60_vox_0087), order=0, cval=False
    )

    profile_case(
        "(d) 0087-1 GTV vs copy shifted 60 mm",
        gtv_0087_bool,
        risk_0087_60mm,
        sp_0087,
        params,
    )


if __name__ == "__main__":
    main()
