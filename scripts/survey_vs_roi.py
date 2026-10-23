"""Survey ROIs across all patients in TCIA Vestibular-Schwannoma-SEG contours.

JSON-only survey across the contours archive:
- Structure names in t1 and t2 folders per patient
- Name of first item in native/t1 folder
- Cochlea-like names (case-insensitive prefix 'cochlea')
- Presence of Brainstem or Modiolus
- Number of cochlea-like items
- Patients whose first-item name is not in {TV, tv, AN, tumour}

Writes data/processed/vs_roi_survey.csv and prints summary counts.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

CONTOURS_DIR = Path("data/raw/vs_seg/Vestibular-Schwannoma-SEG-contours-Mar-2021/contours")
OUTPUT_CSV = Path("data/processed/vs_roi_survey.csv")
STANDARD_TUMOUR_NAMES = {"TV", "tv", "AN", "tumour"}


def run_survey(
    contours_dir: Path = CONTOURS_DIR,
    output_csv: Path = OUTPUT_CSV,
) -> dict[str, int]:
    """Run survey across all patients in contours_dir and write CSV report."""
    if not contours_dir.is_dir():
        raise FileNotFoundError(f"Contours directory not found: {contours_dir}")

    folders = [f.name for f in contours_dir.iterdir() if f.is_dir()]
    patient_ids = sorted(
        list(
            {
                int(re.match(r"vs_gk_(\d+)_(t1|t2)", f).group(1))
                for f in folders
                if re.match(r"vs_gk_(\d+)_(t1|t2)", f)
            }
        )
    )

    rows: list[dict[str, str | int | bool]] = []
    non_standard_first_items: list[tuple[int, str | None]] = []
    all_cochlea_variants: set[str] = set()
    brainstem_patients: list[int] = []
    modiolus_patients: list[int] = []

    for pid in patient_ids:
        t1_path = contours_dir / f"vs_gk_{pid}_t1" / "contours.json"
        t2_path = contours_dir / f"vs_gk_{pid}_t2" / "contours.json"

        t1_names: list[str] = []
        if t1_path.is_file():
            with open(t1_path, "r", encoding="utf-8") as f:
                t1_data = json.load(f)
                t1_names = [
                    item.get("structure_name", "")
                    for item in t1_data
                    if item.get("structure_name") is not None
                ]

        t2_names: list[str] = []
        if t2_path.is_file():
            with open(t2_path, "r", encoding="utf-8") as f:
                t2_data = json.load(f)
                t2_names = [
                    item.get("structure_name", "")
                    for item in t2_data
                    if item.get("structure_name") is not None
                ]

        first_item = t1_names[0] if t1_names else None
        if first_item not in STANDARD_TUMOUR_NAMES:
            non_standard_first_items.append((pid, first_item))

        # Cochlea variants
        patient_cochlea: list[str] = []
        has_brainstem = False
        has_modiolus = False

        for name in t1_names + t2_names:
            lower = name.lower()
            if lower.startswith("cochlea"):
                if name not in patient_cochlea:
                    patient_cochlea.append(name)
                all_cochlea_variants.add(name)
            if "brainstem" in lower:
                has_brainstem = True
            if "modiolus" in lower:
                has_modiolus = True

        if has_brainstem:
            brainstem_patients.append(pid)
        if has_modiolus:
            modiolus_patients.append(pid)

        rows.append(
            {
                "patient_id": f"VS-SEG-{pid:03d}",
                "gk_id": f"vs_gk_{pid}",
                "first_item_name": first_item or "",
                "first_item_is_candidate": first_item in STANDARD_TUMOUR_NAMES,
                "t1_structures": ";".join(t1_names),
                "t2_structures": ";".join(t2_names),
                "cochlea_like_names": ";".join(patient_cochlea),
                "cochlea_like_count": len(patient_cochlea),
                "has_brainstem": has_brainstem,
                "has_modiolus": has_modiolus,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "gk_id",
        "first_item_name",
        "first_item_is_candidate",
        "t1_structures",
        "t2_structures",
        "cochlea_like_names",
        "cochlea_like_count",
        "has_brainstem",
        "has_modiolus",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=== VS-SEG ROI SURVEY SUMMARY ===")
    print(f"Total patients surveyed: {len(patient_ids)}")
    print(f"All cochlea-like variants found: {sorted(all_cochlea_variants)}")
    print(
        f"Patients with first item NOT in {sorted(STANDARD_TUMOUR_NAMES)}: "
        f"{len(non_standard_first_items)}"
    )
    for pid, name in non_standard_first_items:
        print(f"  VS-SEG-{pid:03d} (vs_gk_{pid}): first_item='{name}'")
    print(f"Patients with Brainstem: {len(brainstem_patients)} -> {brainstem_patients}")
    print(f"Patients with Modiolus: {len(modiolus_patients)} -> {modiolus_patients}")
    print(f"Survey written to: {output_csv}")

    return {
        "total_patients": len(patient_ids),
        "non_standard_first_items": len(non_standard_first_items),
        "cochlea_variants": len(all_cochlea_variants),
        "brainstem_patients": len(brainstem_patients),
        "modiolus_patients": len(modiolus_patients),
    }


if __name__ == "__main__":
    run_survey()
