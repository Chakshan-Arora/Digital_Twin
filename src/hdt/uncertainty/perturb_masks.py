"""Smooth random boundary-perturbation model for 3D binary masks.

Conventions:
- Masks are 3D boolean NumPy arrays with shape (z, y, x).
- Spacing is a 3-tuple (sz, sy, sx) in millimetres in (z, y, x) order.
- Distance and coordinate units are in millimetres.
- Randomness is controlled strictly via a passed-in numpy.random.Generator.
"""

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)


def _validate_perturb_inputs(
    mask: np.ndarray,
    spacing_mm: Sequence[float],
    sigma_mm: float,
    corr_length_mm: float,
    bias_sigma_mm: float = 0.0,
    max_hole_mm3: float = 10.0,
) -> tuple[float, float, float]:
    """Validate arguments for mask perturbation."""
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 3:
        raise ValueError("Mask must be a 3D boolean numpy array")
    if not np.any(mask):
        raise ValueError("Cannot perturb an empty mask")
    if len(spacing_mm) != 3:
        raise ValueError(f"spacing_mm must have length 3 (z, y, x), got {len(spacing_mm)}")
    sz, sy, sx = (float(s) for s in spacing_mm)
    if sz <= 0.0 or sy <= 0.0 or sx <= 0.0:
        raise ValueError(f"spacing_mm components must be strictly positive, got {spacing_mm}")
    if sigma_mm < 0.0:
        raise ValueError(f"sigma_mm must be non-negative, got {sigma_mm}")
    if corr_length_mm <= 0.0:
        raise ValueError(f"corr_length_mm must be strictly positive, got {corr_length_mm}")
    if bias_sigma_mm < 0.0:
        raise ValueError(f"bias_sigma_mm must be non-negative, got {bias_sigma_mm}")
    if max_hole_mm3 < 0.0:
        raise ValueError(f"max_hole_mm3 must be non-negative, got {max_hole_mm3}")

    min_spacing = min(sz, sy, sx)
    axis_names = ("z", "y", "x")
    if sigma_mm > 0.0 and sigma_mm < 0.5 * min_spacing:
        sub_axes = [axis_names[i] for i, s in enumerate((sz, sy, sx)) if sigma_mm < 0.5 * s]
        warnings.warn(
            f"sigma_mm ({sigma_mm:.3f} mm) is smaller than half the minimum voxel spacing "
            f"({0.5 * min_spacing:.3f} mm); the discrete voxel grid cannot adequately "
            f"represent this perturbation amplitude. Affected axes: {sub_axes}",
            UserWarning,
            stacklevel=2,
        )

    undersampled_axes = [
        axis_names[i] for i, s in enumerate((sz, sy, sx)) if corr_length_mm < 3.0 * s
    ]
    if undersampled_axes:
        try:
            warnings.warn(
                f"corr_length_mm ({corr_length_mm:.3f} mm) is less than 3x the voxel spacing on "
                f"axes {undersampled_axes}; the discrete voxel grid may undersample high spatial "
                f"frequencies, causing realised surface displacement standard deviation to fall "
                f"10-30% below sigma_mm (field amplitude).",
                UserWarning,
                stacklevel=2,
            )
        except UserWarning:
            pass

    return sz, sy, sx


def uniform_shift(
    mask: np.ndarray,
    spacing_mm: Sequence[float],
    shift_mm: float,
) -> np.ndarray:
    """Apply a uniform isotropic boundary dilation or erosion by shift_mm.

    Signs:
      - Positive shift_mm (+k): expands/dilates the mask outward by k millimetres.
      - Negative shift_mm (-k): shrinks/erodes the mask inward by |k| millimetres.
      - Zero shift_mm: returns an identical copy of the mask.

    Method:
      sdf = edt(~mask, sampling=spacing) - edt(mask, sampling=spacing)
      Outside mask: sdf > 0. Inside mask: sdf <= 0.
      new_mask = (sdf <= shift_mm)

    Args:
        mask: 3D boolean array (z, y, x).
        spacing_mm: (sz, sy, sx) in millimetres.
        shift_mm: Expansion (+mm) or contraction (-mm) distance in millimetres.

    Returns:
        np.ndarray: Shifted 3D boolean array of identical shape.
    """
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 3:
        raise ValueError("Mask must be a 3D boolean numpy array")
    if not np.any(mask):
        raise ValueError("Cannot perturb an empty mask")
    if len(spacing_mm) != 3:
        raise ValueError(f"spacing_mm must have length 3 (z, y, x), got {len(spacing_mm)}")
    sz, sy, sx = (float(s) for s in spacing_mm)
    if sz <= 0.0 or sy <= 0.0 or sx <= 0.0:
        raise ValueError(f"spacing_mm components must be strictly positive, got {spacing_mm}")
    spacing = (sz, sy, sx)

    pad_m = np.pad(mask, 1, mode="constant", constant_values=False)
    edt_out = distance_transform_edt(~pad_m, sampling=spacing)[1:-1, 1:-1, 1:-1]
    edt_in = distance_transform_edt(pad_m, sampling=spacing)[1:-1, 1:-1, 1:-1]
    sdf = (edt_out - edt_in).astype(np.float32)

    return (sdf <= float(shift_mm)).astype(bool)


@dataclass(frozen=True)
class PerturbationPlan:
    """Precomputed spatial context for repeated boundary perturbations."""

    slices: tuple[slice, slice, slice]
    crop_mask: np.ndarray
    sdf: np.ndarray
    filter_sigmas: tuple[float, float, float]
    inv_sqrt_var: np.float32
    band: np.ndarray
    sigma_mm: float
    bias_sigma_mm: float
    max_hole_voxels: int


def prepare_perturbation_plan(
    mask: np.ndarray,
    spacing_mm: Sequence[float],
    sigma_mm: float,
    corr_length_mm: float,
    bias_sigma_mm: float = 0.0,
    max_hole_mm3: float = 10.0,
) -> PerturbationPlan:
    """Precompute static geometry and filters for repeated mask perturbations."""
    sz, sy, sx = _validate_perturb_inputs(
        mask, spacing_mm, sigma_mm, corr_length_mm, bias_sigma_mm, max_hole_mm3
    )
    spacing = (sz, sy, sx)
    max_sp = max(spacing)

    margin_mm = 4.0 * sigma_mm + 3.0 * corr_length_mm + 2.0 * max_sp
    margin_vox = [int(np.ceil(margin_mm / s)) for s in spacing]

    coords = np.argwhere(mask)
    min_idx = np.maximum(coords.min(axis=0) - margin_vox, 0)
    max_idx = np.minimum(coords.max(axis=0) + margin_vox + 1, mask.shape)

    slices = (
        slice(int(min_idx[0]), int(max_idx[0])),
        slice(int(min_idx[1]), int(max_idx[1])),
        slice(int(min_idx[2]), int(max_idx[2])),
    )
    crop_mask = mask[slices]

    pad_crop = np.pad(crop_mask, 1, mode="constant", constant_values=False)
    edt_out = distance_transform_edt(~pad_crop, sampling=spacing)[1:-1, 1:-1, 1:-1]
    edt_in = distance_transform_edt(pad_crop, sampling=spacing)[1:-1, 1:-1, 1:-1]
    sdf = (edt_out - edt_in).astype(np.float32)

    filter_sigmas = tuple(corr_length_mm / s for s in spacing)
    radii = [int(np.ceil(4 * s)) for s in filter_sigmas]
    imp_shape = [2 * r + 1 for r in radii]
    impulse = np.zeros(imp_shape, dtype=np.float32)
    impulse[radii[0], radii[1], radii[2]] = 1.0
    resp = gaussian_filter(impulse, sigma=filter_sigmas, mode="constant")
    variance = float(np.sum(resp**2))
    inv_sqrt_var = np.float32(1.0 / np.sqrt(variance))

    band_half_width = 4.0 * sigma_mm + 2.0 * max_sp
    band = np.abs(sdf) < band_half_width

    voxel_vol_mm3 = sz * sy * sx
    max_hole_voxels = int(np.floor(max_hole_mm3 / voxel_vol_mm3))

    return PerturbationPlan(
        slices=slices,
        crop_mask=crop_mask,
        sdf=sdf,
        filter_sigmas=filter_sigmas,
        inv_sqrt_var=inv_sqrt_var,
        band=band,
        sigma_mm=float(sigma_mm),
        bias_sigma_mm=float(bias_sigma_mm),
        max_hole_voxels=max_hole_voxels,
    )


def perturb_mask(
    mask: np.ndarray,
    spacing_mm: Sequence[float],
    sigma_mm: float,
    corr_length_mm: float,
    rng: np.random.Generator,
    bias_sigma_mm: float = 0.0,
    max_hole_mm3: float = 10.0,
    *,
    _plan: PerturbationPlan | None = None,
) -> np.ndarray:
    """Generate a smooth random boundary perturbation of a 3D binary mask.

    The continuous boundary is perturbed according to a zero-mean Gaussian random field
    with specified standard deviation (sigma_mm) and correlation length (corr_length_mm).

    Analytic Field Normalization:
      Filter a unit impulse with the same Gaussian (same per-axis sigmas),
      set variance = sum(response**2), and divide the filtered noise by sqrt(variance).
      The crop mean is not subtracted, preserving local spatial statistics.

    Discretization & Amplitude Note:
      sigma_mm represents the continuous random field amplitude. When corr_length_mm
      is small relative to voxel spacing (e.g. corr_length_mm < 3 * spacing_i), the discrete
      voxel grid rasterization and binary thresholding can cause realised surface
      displacement standard deviation to fall 10-30% below sigma_mm.

    Args:
        mask: 3D boolean NumPy array (z, y, x).
        spacing_mm: Voxel spacing (sz, sy, sx) in millimetres.
        sigma_mm: Standard deviation of local boundary displacements in mm.
        corr_length_mm: Spatial correlation length in mm.
        rng: Seeded numpy.random.Generator instance.
        bias_sigma_mm: Standard deviation of global volumetric bias shift in mm.
        max_hole_mm3: Maximum volume in mm^3 for filling internal holes (default: 10 mm^3).
        _plan: Optional precomputed PerturbationPlan for accelerated repeated perturbations.

    Raises:
        ValueError: If inputs are invalid or if perturbation results in an empty mask.

    Returns:
        np.ndarray: Perturbed 3D boolean array of identical shape.
    """
    if _plan is None:
        _validate_perturb_inputs(
            mask, spacing_mm, sigma_mm, corr_length_mm, bias_sigma_mm, max_hole_mm3
        )
        if sigma_mm == 0.0 and bias_sigma_mm == 0.0:
            return mask.copy()
        plan = prepare_perturbation_plan(
            mask, spacing_mm, sigma_mm, corr_length_mm, bias_sigma_mm, max_hole_mm3
        )
    else:
        plan = _plan

    if plan.sigma_mm == 0.0 and plan.bias_sigma_mm == 0.0:
        return mask.copy()

    # Spatially correlated noise field with analytic normalization
    white_noise = rng.normal(0.0, 1.0, size=plan.crop_mask.shape).astype(np.float32)
    smoothed = gaussian_filter(white_noise, sigma=plan.filter_sigmas, mode="reflect")
    field = (smoothed * plan.inv_sqrt_var) * plan.sigma_mm

    # Global bias
    if plan.bias_sigma_mm > 0.0:
        bias = float(rng.normal(0.0, plan.bias_sigma_mm))
    else:
        bias = 0.0

    perturbed_crop = plan.crop_mask.copy()
    perturbed_crop[plan.band] = (plan.sdf[plan.band] + field[plan.band] + bias) <= 0.0

    # Post-processing: component filtering
    labeled_crop, n_comp = label(perturbed_crop)
    if n_comp == 1:
        if not np.any(perturbed_crop & plan.crop_mask):
            perturbed_crop = np.zeros_like(perturbed_crop)
    elif n_comp > 1:
        comp_overlap = labeled_crop[plan.crop_mask]
        retained_labels = np.unique(comp_overlap[comp_overlap > 0])
        if len(retained_labels) == 0:
            perturbed_crop = np.zeros_like(perturbed_crop)
        elif len(retained_labels) < n_comp:
            perturbed_crop = np.isin(labeled_crop, retained_labels)

    if not np.any(perturbed_crop):
        raise ValueError("Perturbation eliminated all mask voxels; result is empty")

    # Small hole filling using tight bounding box acceleration
    if plan.max_hole_voxels > 0:
        coords_p = np.argwhere(perturbed_crop)
        if len(coords_p) > 0:
            p_min = coords_p.min(axis=0)
            p_max = coords_p.max(axis=0)
            sl_tight = tuple(slice(int(p_min[i]), int(p_max[i] + 1)) for i in range(3))
            tight = perturbed_crop[sl_tight]
            tight_pad = np.pad(tight, 1, mode="constant", constant_values=False)
            filled = binary_fill_holes(tight_pad)[1:-1, 1:-1, 1:-1]
            holes = filled & ~tight
            if np.any(holes):
                labeled_holes, n_holes = label(holes)
                if n_holes == 1:
                    if np.count_nonzero(holes) <= plan.max_hole_voxels:
                        tight[holes] = True
                elif n_holes > 1:
                    counts = np.bincount(labeled_holes.flat)[1:]
                    keep_holes = np.where(counts <= plan.max_hole_voxels)[0] + 1
                    if len(keep_holes) > 0:
                        tight[np.isin(labeled_holes, keep_holes)] = True
            perturbed_crop[sl_tight] = tight

    result = mask.copy()
    result[plan.slices] = perturbed_crop
    return result
