"""Export 10 perturbed target masks and +/- 1 mm uniform shifts to data/processed/perturbed/."""

from pathlib import Path

import numpy as np

from hdt.imaging.load_scan import ImageGeometry, load_nifti_image, load_nifti_mask, save_nifti_mask
from hdt.uncertainty.perturb_masks import perturb_mask, uniform_shift


def export_perturbed() -> None:
    img, sp, orig, dir_mat = load_nifti_image("data/processed/phantom_image.nii.gz")
    geom = ImageGeometry(shape=img.shape, spacing_mm=sp, origin_mm=orig, direction=dir_mat)
    target = load_nifti_mask("data/processed/target_mask.nii.gz", image_meta=geom, label_values=[1])

    out_dir = Path("data/processed/perturbed")
    out_dir.mkdir(parents=True, exist_ok=True)

    voxel_vol = float(np.prod(sp))

    # 1. 10 perturbed target masks (sigma = 1.0 mm, L = 5.0 mm, fixed seed)
    ss = np.random.SeedSequence(2026)
    child_seeds = ss.spawn(10)

    print("=== Exporting Perturbed Target Masks ===")
    for i in range(10):
        rng = np.random.default_rng(child_seeds[i])
        t_pert = perturb_mask(target, sp, sigma_mm=1.0, corr_length_mm=5.0, rng=rng)
        out_path = out_dir / f"target_perturbed_{i:02d}.nii.gz"
        save_nifti_mask(t_pert, geom, out_path)
        vol_mm3 = float(np.count_nonzero(t_pert)) * voxel_vol
        print(f"Sample {i:02d}: {out_path.name:<25} Volume = {vol_mm3:10.2f} mm^3")

    # 2. Uniform shift +1 mm and -1 mm
    shift_pos = uniform_shift(target, sp, shift_mm=1.0)
    out_pos = out_dir / "target_uniform_shift_plus_1mm.nii.gz"
    save_nifti_mask(shift_pos, geom, out_pos)
    vol_pos = float(np.count_nonzero(shift_pos)) * voxel_vol
    print(f"Shift +1mm: {out_pos.name:<25} Volume = {vol_pos:10.2f} mm^3")

    shift_neg = uniform_shift(target, sp, shift_mm=-1.0)
    out_neg = out_dir / "target_uniform_shift_minus_1mm.nii.gz"
    save_nifti_mask(shift_neg, geom, out_neg)
    vol_neg = float(np.count_nonzero(shift_neg)) * voxel_vol
    print(f"Shift -1mm: {out_neg.name:<25} Volume = {vol_neg:10.2f} mm^3")


if __name__ == "__main__":
    export_perturbed()
