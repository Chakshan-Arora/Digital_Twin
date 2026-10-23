#!/usr/bin/env python3
"""Prepare a brain MRI case for HDT processing and write standardized artifacts."""

import argparse
import json
from pathlib import Path

from hdt.domains.brain_mri.adapter import load_case
from hdt.imaging.load_scan import inspect_nifti_info, save_nifti_image, save_nifti_mask


def prepare_case(
    root: str | Path,
    case_id: str,
    out: str | Path,
) -> Path:
    """Load, standardize, and write a case to the target output directory.

    Writes:
        - <out>/t1c.nii.gz
        - <out>/structures/gtv.nii.gz
        - <out>/case.json
    """
    root_path = Path(root).resolve()
    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case = load_case(root=root_path, case_id=case_id)

    # Determine original dtype from source modality file
    case_dir = root_path / case_id if (root_path / case_id).is_dir() else root_path
    t1c_files = list(case_dir.glob("*_t1c.nii*"))
    orig_dtype = "unknown"
    if t1c_files:
        info = inspect_nifti_info(t1c_files[0])
        orig_dtype = info["dtype"]

    # Write t1c modality NIfTI
    t1c_out = out_dir / "t1c.nii.gz"
    save_nifti_image(case.images["t1c"], case, t1c_out)

    # Write structures
    struct_dir = out_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    gtv_out = struct_dir / "gtv.nii.gz"
    save_nifti_mask(case.structures["gtv"].mask, case, gtv_out)

    # Write case.json metadata
    metadata = {
        "case_id": case.case_id,
        "spacing_mm_zyx": [float(s) for s in case.spacing_mm],
        "shape_zyx": [int(d) for d in case.shape],
        "roles": {name: s.role for name, s in case.structures.items()},
        "label_map": "BRATS_MEN_RT",
        "source": "BraTS2024-MEN-RT",
        "original_dtype": orig_dtype,
    }

    case_json_path = out_dir / "case.json"
    with case_json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a BraTS2024-MEN-RT case for HDT.")
    parser.add_argument("--root", required=True, help="Root path of raw BraTS dataset.")
    parser.add_argument("--case", required=True, help="Case identifier to prepare.")
    parser.add_argument("--out", required=True, help="Output directory path.")

    args = parser.parse_args()
    out_path = prepare_case(root=args.root, case_id=args.case, out=args.out)
    print(f"Prepared case '{args.case}' written to '{out_path}'.")


if __name__ == "__main__":
    main()
