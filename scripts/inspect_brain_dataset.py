#!/usr/bin/env python3
"""Dataset inspector for brain MRI datasets (msd4d vs brats_folder)."""

import json
import sys
from pathlib import Path

from hdt.imaging.load_scan import inspect_nifti_info


def detect_layout(root_dir: Path) -> tuple[str, list[dict], dict | None]:
    """Detect whether dataset follows 'msd4d' or 'brats_folder' layout.

    Returns:
        (layout_name, case_specs, dataset_json_dict_or_none)
    """
    dataset_json_path = root_dir / "dataset.json"
    dataset_meta = None
    if dataset_json_path.is_file():
        try:
            with dataset_json_path.open("r", encoding="utf-8") as f:
                dataset_meta = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to parse dataset.json: {e}", file=sys.stderr)

    # Check for msd4d layout
    images_tr = root_dir / "imagesTr"
    labels_tr = root_dir / "labelsTr"
    if images_tr.is_dir() and labels_tr.is_dir():
        image_files = sorted(
            [p for p in images_tr.iterdir() if p.name.endswith((".nii", ".nii.gz"))]
        )
        if image_files:
            # Check first image dimension
            info = inspect_nifti_info(image_files[0])
            if info["dimension"] == 4:
                cases = []
                for img_path in image_files:
                    stem = img_path.name.split(".")[0]
                    lbl_path = labels_tr / img_path.name
                    if not lbl_path.exists():
                        # Try without extension differences
                        lbl_candidates = list(labels_tr.glob(f"{stem}*.nii*"))
                        lbl_path = lbl_candidates[0] if lbl_candidates else None

                    cases.append({
                        "case_id": stem,
                        "modalities": [img_path],
                        "seg": lbl_path,
                        "layout": "msd4d",
                    })
                return "msd4d", cases, dataset_meta

    # Check for brats_folder layout
    # First check subdirectories in root_dir
    candidate_subdirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]
    )

    # If there is a single directory that contains case directories, unwrap it
    if len(candidate_subdirs) == 1 and not list(candidate_subdirs[0].glob("*.nii*")):
        candidate_subdirs = sorted(
            [
                d
                for d in candidate_subdirs[0].iterdir()
                if d.is_dir() and not d.name.startswith((".", "_"))
            ]
        )

    cases = []
    for d in candidate_subdirs:
        nii_files = sorted([f for f in d.iterdir() if f.name.endswith((".nii", ".nii.gz"))])
        if not nii_files:
            continue

        seg_files = []
        mod_files = []
        for f in nii_files:
            low = f.name.lower()
            if any(term in low for term in ("_seg.", "_gtv.", "_mask.", "_label.")):
                seg_files.append(f)
            else:
                mod_files.append(f)

        if seg_files and mod_files:
            cases.append({
                "case_id": d.name,
                "modalities": mod_files,
                "seg": seg_files[0],
                "layout": "brats_folder",
            })

    if cases:
        return "brats_folder", cases, dataset_meta

    return "neither", [], dataset_meta


def inspect_dataset(root_path: str | Path) -> None:
    """Inspect dataset and print metadata for the first 5 cases."""
    root = Path(root_path).resolve()
    if not root.exists():
        print(f"ERROR: Dataset root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    layout, cases, dataset_meta = detect_layout(root)

    print("================================================================================")
    print("                      BRAIN TUMOUR DATASET INSPECTION REPORT                     ")
    print("================================================================================")
    print(f"Dataset root: {root}")
    print(f"Detected layout: {layout}")
    print(f"Total cases found: {len(cases)}")
    print(f"dataset.json present: {'YES' if dataset_meta is not None else 'NO'}")
    print("================================================================================\n")

    if layout == "neither":
        print("STOP: The dataset layout is neither 'msd4d' nor 'brats_folder'.")
        print("Please check directory structure with the human before proceeding.")
        sys.exit(1)

    # Print dataset.json mappings if present
    if dataset_meta is not None:
        print("--- dataset.json Mappings (verbatim) ---")
        if "modality" in dataset_meta:
            print("Modalities:")
            print(json.dumps(dataset_meta["modality"], indent=2))
        elif "channel_names" in dataset_meta:
            print("Channel names:")
            print(json.dumps(dataset_meta["channel_names"], indent=2))

        if "labels" in dataset_meta:
            print("Labels:")
            print(json.dumps(dataset_meta["labels"], indent=2))
        print("----------------------------------------\n")
    else:
        print("--- dataset.json Mappings ---")
        print("None (dataset.json is NOT present in the dataset root).")
        print("-----------------------------\n")

    num_cases_to_inspect = min(5, len(cases))
    print(f"Inspecting first {num_cases_to_inspect} cases:\n")

    for idx, case in enumerate(cases[:num_cases_to_inspect], start=1):
        case_id = case["case_id"]
        mod_paths: list[Path] = case["modalities"]
        seg_path: Path | None = case["seg"]

        print(f"Case {idx}/{num_cases_to_inspect}: {case_id}")
        print(f"  Layout: {case['layout']}")
        print(f"  Modality files: {[p.name for p in mod_paths]}")
        print(f"  Segmentation file: {seg_path.name if seg_path else 'None'}")

        # Inspect first modality
        first_mod_info = inspect_nifti_info(mod_paths[0])
        print(f"  Shape (z, y, x): {first_mod_info['shape_zyx']}")
        sp = first_mod_info["spacing_zyx"]
        print(f"  Spacing (z, y, x) [mm]: ({sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f})")
        orig = first_mod_info["origin_zyx"]
        print(f"  Origin (z, y, x) [mm]: ({orig[0]:.4f}, {orig[1]:.4f}, {orig[2]:.4f})")
        print("  Direction (3x3 LPS):")
        for row in first_mod_info["direction"]:
            print(f"    [{row[0]:9.4f}, {row[1]:9.4f}, {row[2]:9.4f}]")
        print(f"  Modality dtype: {first_mod_info['dtype']}")

        if seg_path is not None and seg_path.exists():
            seg_info = inspect_nifti_info(seg_path)
            print(f"  Segmentation dtype: {seg_info['dtype']}")
            print("  Unique label values (voxel counts & volume):")
            voxel_volume_mm3 = sp[0] * sp[1] * sp[2]
            for label, count in sorted(seg_info["unique_labels"].items()):
                vol_mm3 = count * voxel_volume_mm3
                print(f"    Label {label:2d}: {count:10d} voxels ({vol_mm3:14.2f} mm3)")
        else:
            print("  Segmentation: No valid segmentation file found.")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_brain_dataset.py <root_path>")
        sys.exit(1)
    inspect_dataset(sys.argv[1])
