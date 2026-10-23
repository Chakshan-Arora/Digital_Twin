"""Frozen reference implementation of perturb_mask and per-sample clearance calculation.

DO NOT OPTIMISE OR MODIFY THIS FILE.
Used for exact bitwise and numerical regression testing against optimized implementations.
"""

import warnings
from typing import Sequence

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from hdt.geometry.distances import signed_clearance_mm


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


def reference_perturb_mask(
    mask: np.ndarray,
    spacing_mm: Sequence[float],
    sigma_mm: float,
    corr_length_mm: float,
    rng: np.random.Generator,
    bias_sigma_mm: float = 0.0,
    max_hole_mm3: float = 10.0,
) -> np.ndarray:
    """Frozen reference implementation of perturb_mask."""
    sz, sy, sx = _validate_perturb_inputs(
        mask, spacing_mm, sigma_mm, corr_length_mm, bias_sigma_mm, max_hole_mm3
    )
    spacing = (sz, sy, sx)
    max_sp = max(spacing)

    # Fast-path: sigma_mm == 0 and bias_sigma_mm == 0 returns the identical mask
    if sigma_mm == 0.0 and bias_sigma_mm == 0.0:
        return mask.copy()

    # 1. Bounding box crop with margin
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

    # 2. Signed distance field on crop
    pad_crop = np.pad(crop_mask, 1, mode="constant", constant_values=False)
    edt_out = distance_transform_edt(~pad_crop, sampling=spacing)[1:-1, 1:-1, 1:-1]
    edt_in = distance_transform_edt(pad_crop, sampling=spacing)[1:-1, 1:-1, 1:-1]
    sdf = (edt_out - edt_in).astype(np.float32)

    # 3. Spatially correlated noise field with analytic normalization
    filter_sigmas = tuple(corr_length_mm / s for s in spacing)

    # Unit impulse response to determine exact continuous/discrete filter variance
    radii = [int(np.ceil(4 * s)) for s in filter_sigmas]
    imp_shape = [2 * r + 1 for r in radii]
    impulse = np.zeros(imp_shape, dtype=np.float32)
    impulse[radii[0], radii[1], radii[2]] = 1.0
    resp = gaussian_filter(impulse, sigma=filter_sigmas, mode="constant")
    variance = float(np.sum(resp**2))

    white_noise = rng.normal(0.0, 1.0, size=crop_mask.shape).astype(np.float32)
    smoothed = gaussian_filter(white_noise, sigma=filter_sigmas, mode="reflect")

    # Divide by sqrt(variance) and scale by sigma_mm (do NOT subtract crop mean)
    field = (smoothed / np.sqrt(variance)) * float(sigma_mm)

    # 4. Global bias
    if bias_sigma_mm > 0.0:
        bias = float(rng.normal(0.0, bias_sigma_mm))
    else:
        bias = 0.0

    # 5. Narrow-band perturbation
    band_half_width = 4.0 * sigma_mm + 2.0 * max_sp
    band = np.abs(sdf) < band_half_width

    perturbed_crop = crop_mask.copy()
    perturbed_crop[band] = (sdf[band] + field[band] + bias) <= 0.0

    # 6. Post-processing: component filtering and small hole filling
    labeled_crop, n_comp = label(perturbed_crop)
    if n_comp > 0:
        retained = np.zeros_like(perturbed_crop)
        for comp_idx in range(1, n_comp + 1):
            comp_mask = labeled_crop == comp_idx
            if np.any(comp_mask & crop_mask):
                retained |= comp_mask
        perturbed_crop = retained

    if not np.any(perturbed_crop):
        raise ValueError("Perturbation eliminated all mask voxels; result is empty")

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

    # 7. Paste back into full volume
    result = mask.copy()
    result[slices] = perturbed_crop

    return result


def reference_sample_clearance(
    target: np.ndarray,
    risk: np.ndarray,
    spacing_mm: Sequence[float],
    params: dict,
    rng: np.random.Generator,
    max_distance_mm: float = 40.0,
) -> tuple[np.ndarray, float]:
    """Frozen reference per-sample perturbation and clearance computation."""
    sigma_mm = float(params["sigma_mm"])
    corr_length_mm = float(params["corr_length_mm"])
    bias_sigma_mm = float(params.get("bias_sigma_mm", 0.0))
    max_hole_mm3 = float(params.get("max_hole_mm3", 10.0))

    t_perturbed = reference_perturb_mask(
        target,
        spacing_mm,
        sigma_mm=sigma_mm,
        corr_length_mm=corr_length_mm,
        rng=rng,
        bias_sigma_mm=bias_sigma_mm,
        max_hole_mm3=max_hole_mm3,
    )

    c = signed_clearance_mm(t_perturbed, risk, spacing_mm, crop_to_bbox=True)
    if c >= max_distance_mm:
        c = max_distance_mm

    return t_perturbed, float(c)
