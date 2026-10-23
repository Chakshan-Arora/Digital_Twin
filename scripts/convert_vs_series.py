"""Convert TCIA VS-SEG DICOM image series to processed NIfTI files.

Extracts each patient's native T1 and T2 DICOM series (no resampling, no intensity
changes) using `load_series()` from `hdt.imaging.dicom_series` and writes:
- data/processed/vs_series/<patient>/t1.nii.gz
- data/processed/vs_series/<patient>/t2.nii.gz
- data/processed/vs_series/<patient>/series.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pydicom

from hdt.imaging.dicom_series import load_series
from hdt.imaging.load_scan import save_nifti_image


def convert_all_series() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    raw_dcm_dir = root_dir / "data" / "raw" / "vs_seg" / "vestibular_schwannoma_seg"
    out_base_dir = root_dir / "data" / "processed" / "vs_series"

    patient_dirs = sorted([d for d in raw_dcm_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patients to convert in {raw_dcm_dir}\n")

    for p_dir in patient_dirs:
        pat_id = p_dir.name
        pat_out_dir = out_base_dir / pat_id
        pat_out_dir.mkdir(parents=True, exist_ok=True)

        t1_dirs = [d for d in p_dir.glob("*/*t1*") if d.is_dir()]
        t2_dirs = [d for d in p_dir.glob("*/*t2*") if d.is_dir()]

        if not t1_dirs or not t2_dirs:
            print(f"[!] Skipping {pat_id}: missing T1 or T2 directory")
            continue

        t1_dir = t1_dirs[0]
        t2_dir = t2_dirs[0]

        # Extract series descriptions from first DICOM file
        t1_first = next(t1_dir.glob("*.dcm"))
        t2_first = next(t2_dir.glob("*.dcm"))
        t1_ds0 = pydicom.dcmread(str(t1_first), stop_before_pixels=True)
        t2_ds0 = pydicom.dcmread(str(t2_first), stop_before_pixels=True)
        t1_desc = str(getattr(t1_ds0, "SeriesDescription", "T1"))
        t2_desc = str(getattr(t2_ds0, "SeriesDescription", "T2"))

        print(f"Converting {pat_id}:")
        print(f"  T1 source: {t1_dir.name} ({t1_desc})")
        t1_arr, t1_geom = load_series(t1_dir)
        t1_out_file = pat_out_dir / "t1.nii.gz"
        save_nifti_image(t1_arr, t1_geom, t1_out_file)
        print(
            f"  -> Written {t1_out_file.name}: shape={t1_geom.shape}, spacing={t1_geom.spacing_mm}"
        )

        print(f"  T2 source: {t2_dir.name} ({t2_desc})")
        t2_arr, t2_geom = load_series(t2_dir)
        t2_out_file = pat_out_dir / "t2.nii.gz"
        save_nifti_image(t2_arr, t2_geom, t2_out_file)
        print(
            f"  -> Written {t2_out_file.name}: shape={t2_geom.shape}, spacing={t2_geom.spacing_mm}"
        )

        # Write metadata series.json
        meta = {
            "patient_id": pat_id,
            "t1": {
                "series_description": t1_desc,
                "shape_zyx": list(t1_geom.shape),
                "spacing_zyx_mm": list(t1_geom.spacing_mm),
                "origin_zyx_mm": list(t1_geom.origin_mm),
            },
            "t2": {
                "series_description": t2_desc,
                "shape_zyx": list(t2_geom.shape),
                "spacing_zyx_mm": list(t2_geom.spacing_mm),
                "origin_zyx_mm": list(t2_geom.origin_mm),
            },
        }

        json_out_file = pat_out_dir / "series.json"
        with open(json_out_file, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)
        print(f"  -> Written {json_out_file.name}\n")

    print("All patient series converted successfully.")


if __name__ == "__main__":
    convert_all_series()
