#!/usr/bin/env python3
"""Dataset summary script for streaming brain MRI cases and computing metrics."""

import argparse
import csv
import gc
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi

from hdt.domains.brain_mri.label_maps import BRATS_MEN_RT, validate_labels
from hdt.imaging.load_scan import (
    inspect_nifti_header,
    load_nifti_mask_array,
)


def find_cases(root_dir: Path) -> list[str]:
    """Discover all valid case directory IDs in root directory."""
    subdirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
    )
    if len(subdirs) == 1 and not list(subdirs[0].glob("*.nii*")):
        subdirs = sorted(
            [d for d in subdirs[0].iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
        )

    cases = []
    for d in subdirs:
        nii_files = list(d.glob("*.nii*"))
        if any(f.name.endswith(("_t1c.nii.gz", "_t1c.nii")) for f in nii_files) and any(
            f.name.endswith(("_gtv.nii.gz", "_gtv.nii")) for f in nii_files
        ):
            cases.append(d.name)
    return cases


def analyze_case(root: Path, case_id: str) -> dict[str, Any]:
    """Stream and analyze a single case, releasing arrays after extraction.

    Reads modality metadata from header only (never holding large image volumes
    in memory), and loads only the lightweight segmentation mask.
    """
    case_dir = root / case_id if (root / case_id).is_dir() else root
    t1c_files = sorted(
        [
            f
            for f in case_dir.iterdir()
            if f.is_file() and (f.name.endswith("_t1c.nii.gz") or f.name.endswith("_t1c.nii"))
        ]
    )
    gtv_files = sorted(
        [
            f
            for f in case_dir.iterdir()
            if f.is_file() and (f.name.endswith("_gtv.nii.gz") or f.name.endswith("_gtv.nii"))
        ]
    )

    warnings_list = []

    if not t1c_files:
        raise FileNotFoundError(f"Missing t1c file for case '{case_id}' in '{case_dir}'.")
    if not gtv_files:
        raise FileNotFoundError(f"Missing gtv file for case '{case_id}' in '{case_dir}'.")

    # Read modality header without loading 3D volume into memory
    t1c_hdr = inspect_nifti_header(t1c_files[0])
    t1c_dtype = t1c_hdr["dtype"]

    # Load segmentation mask
    mask_arr, mask_geom, gtv_dtype = load_nifti_mask_array(gtv_files[0])
    orig_dtypes_str = f"t1c:{t1c_dtype}, gtv:{gtv_dtype}"

    # Validate mask values
    try:
        validate_labels(mask_arr, BRATS_MEN_RT)
    except Exception as e:
        warnings_list.append(f"invalid_labels: {e}")

    spacing_zyx = mask_geom.spacing_mm
    shape_zyx = mask_geom.shape

    gtv_mask = mask_arr == 1
    gtv_voxels = int(np.sum(gtv_mask))
    voxel_vol = spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2]
    gtv_volume_mm3 = float(gtv_voxels * voxel_vol)

    if gtv_voxels == 0:
        warnings_list.append("empty_mask")
        bbox_extent_str = "(0.00, 0.00, 0.00)"
        touches_border = False
        num_components = 0
    else:
        # Bounding box physical extent
        indices = np.argwhere(gtv_mask)
        min_c = indices.min(axis=0)
        max_c = indices.max(axis=0)
        extents_mm = tuple(float((max_c[i] - min_c[i] + 1) * spacing_zyx[i]) for i in range(3))
        bbox_extent_str = f"({extents_mm[0]:.2f}, {extents_mm[1]:.2f}, {extents_mm[2]:.2f})"

        # Touches array border
        touches_border = bool(
            np.any(gtv_mask[0, :, :])
            or np.any(gtv_mask[-1, :, :])
            or np.any(gtv_mask[:, 0, :])
            or np.any(gtv_mask[:, -1, :])
            or np.any(gtv_mask[:, :, 0])
            or np.any(gtv_mask[:, :, -1])
        )

        # 3D connected components (26-connectivity)
        struct_26 = np.ones((3, 3, 3), dtype=bool)
        _, num_components = ndi.label(gtv_mask, structure=struct_26)

    if touches_border:
        warnings_list.append("touches_border")
    if num_components > 1:
        warnings_list.append("multi_component")

    warnings_str = ";".join(warnings_list) if warnings_list else "none"

    row_data = {
        "case_id": case_id,
        "shape": f"({shape_zyx[0]}, {shape_zyx[1]}, {shape_zyx[2]})",
        "spacing": f"({spacing_zyx[0]:.4f}, {spacing_zyx[1]:.4f}, {spacing_zyx[2]:.4f})",
        "original_dtypes": orig_dtypes_str,
        "gtv_voxels": gtv_voxels,
        "gtv_volume_mm3": round(gtv_volume_mm3, 2),
        "gtv_bbox_extent_mm": bbox_extent_str,
        "touches_array_border": touches_border,
        "gtv_connected_components": num_components,
        "warnings": warnings_str,
    }

    # Free arrays
    del mask_arr
    del gtv_mask
    gc.collect()

    return row_data


def generate_summary(root_path: str | Path, out_csv: str | Path) -> None:
    root = Path(root_path).resolve()
    out_file = Path(out_csv).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    case_ids = find_cases(root)
    if not case_ids:
        print(f"ERROR: No valid cases found in '{root}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(case_ids)} cases. Streaming analysis...")
    sys.stdout.flush()

    fieldnames = [
        "case_id",
        "shape",
        "spacing",
        "original_dtypes",
        "gtv_voxels",
        "gtv_volume_mm3",
        "gtv_bbox_extent_mm",
        "touches_array_border",
        "gtv_connected_components",
        "warnings",
    ]

    records: list[dict[str, Any]] = []

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, cid in enumerate(case_ids, start=1):
            row = analyze_case(root, cid)
            writer.writerow(row)
            # Store only lightweight metadata for summary metrics
            records.append({
                "case_id": row["case_id"],
                "spacing": row["spacing"],
                "volume": row["gtv_volume_mm3"],
                "touches_border": row["touches_array_border"],
                "components": row["gtv_connected_components"],
            })
            if idx % 100 == 0 or idx == len(case_ids):
                print(f"  Processed {idx}/{len(case_ids)} cases...")
                sys.stdout.flush()

    print(f"\nSummary successfully written to: {out_file}\n")
    sys.stdout.flush()

    # Metrics computations
    volumes = np.array([r["volume"] for r in records], dtype=np.float64)
    spacings = [r["spacing"] for r in records]
    spacing_counts = Counter(spacings)
    multi_component_count = sum(1 for r in records if r["components"] > 1)
    touches_border_count = sum(1 for r in records if r["touches_border"])

    # Find pairs of cases with identical volume within 0.1%
    # Tolerance: abs(v1 - v2) / max(v1, v2) <= 0.001
    sorted_records = sorted(records, key=lambda x: x["volume"])
    close_pairs: list[tuple[str, str, float, float]] = []
    n = len(sorted_records)
    for i in range(n):
        v_i = sorted_records[i]["volume"]
        cid_i = sorted_records[i]["case_id"]
        for j in range(i + 1, n):
            v_j = sorted_records[j]["volume"]
            cid_j = sorted_records[j]["case_id"]
            if v_j > 0 and (v_j - v_i) / v_j <= 0.001:
                close_pairs.append((cid_i, cid_j, v_i, v_j))
            else:
                break

    print("================================================================================")
    print("                         DATASET SUMMARY REPORT                                 ")
    print("================================================================================")
    print(f"Total Cases: {len(records)}")
    print("\nDistinct Spacing Counts (z, y, x):")
    for sp_str, count in spacing_counts.most_common():
        print(f"  {sp_str}: {count} cases ({count / len(records) * 100:.1f}%)")

    print("\nGTV Volume Metrics (mm3):")
    print(f"  Minimum volume: {np.min(volumes):.2f} mm3")
    print(f"  Median volume:  {np.median(volumes):.2f} mm3")
    print(f"  Maximum volume: {np.max(volumes):.2f} mm3")
    print(f"  Mean volume:    {np.mean(volumes):.2f} mm3")

    print(f"\nCases with multi-component GTV (> 1 component): {multi_component_count}")
    print(f"Cases touching array border:                    {touches_border_count}")

    print(f"\nPairs of cases with identical volume within 0.1% (found {len(close_pairs)} pairs):")
    if close_pairs:
        # Display first 25 pairs for readability
        for p in close_pairs[:25]:
            diff_pct = abs(p[2] - p[3]) / max(p[2], p[3]) * 100
            print(f"  {p[0]} ({p[2]:.2f} mm3) <-> {p[1]} ({p[3]:.2f} mm3) [diff: {diff_pct:.4f}%]")
        if len(close_pairs) > 25:
            print(f"  ... and {len(close_pairs) - 25} more pairs.")
    else:
        print("  None found.")
    print("================================================================================")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summary CSV and report for brain MRI dataset."
    )
    parser.add_argument("--root", default="data/raw/brats", help="Path to raw dataset.")
    parser.add_argument(
        "--out", default="data/processed/summary.csv", help="Path to output summary CSV."
    )
    args = parser.parse_args()

    generate_summary(root_path=args.root, out_csv=args.out)


if __name__ == "__main__":
    main()
