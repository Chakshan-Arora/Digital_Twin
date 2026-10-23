"""Explicit label map definitions and validations for brain MRI segmentation datasets."""

import json
from pathlib import Path
from typing import Any

import numpy as np

# Explicit, named standard BraTS label maps
BRATS_MEN_RT: dict[int, str] = {
    1: "gtv",
}

BRATS_2020_21: dict[int, str] = {
    1: "necrotic_core",
    2: "edema",
    4: "enhancing",
}

BRATS_2023: dict[int, str] = {
    1: "necrotic_core",
    2: "edema",
    3: "enhancing",
}


def get_msd_label_map(dataset_json_path: str | Path) -> dict[int, str]:
    """Parse label mapping dynamically from an MSD dataset.json file.

    Never hard-coded; strictly parsed from dataset metadata.

    Args:
        dataset_json_path: Path to dataset.json.

    Returns:
        Dictionary mapping integer label values (excluding background 0) to structure names.

    Raises:
        FileNotFoundError: If dataset.json does not exist.
        ValueError: If labels field is missing or invalid.
    """
    path = Path(dataset_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset.json not found at '{path}'.")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "labels" not in data or not isinstance(data["labels"], dict):
        raise ValueError(f"Missing or invalid 'labels' dictionary in '{path}'.")

    label_map: dict[int, str] = {}
    for key, val in data["labels"].items():
        int_key = int(key)
        val_str = str(val).strip().lower().replace(" ", "_")
        # Exclude background
        if int_key != 0 and val_str not in ("background", "bg"):
            label_map[int_key] = val_str

    return label_map


def validate_labels(
    labels_or_mask: Any,
    label_map: dict[int, str],
) -> None:
    """Verify that observed unique labels are a subset of label_map keys plus 0.

    Also enforces that if a floating-point array mask is provided, all values
    must be exactly integer-valued (`np.all(m == np.round(m))`).

    Args:
        labels_or_mask: Unique integer labels or a NumPy mask array.
        label_map: Expected mapping of integer label -> structure name.

    Raises:
        ValueError: If float mask is not integer-valued, or if observed labels
            contain values outside `label_map.keys() | {0}`, including both sets.
    """
    if isinstance(labels_or_mask, np.ndarray):
        if np.issubdtype(labels_or_mask.dtype, np.floating):
            if not np.all(np.isfinite(labels_or_mask)):
                raise ValueError("Float mask contains non-finite values (NaN or Inf).")
            if not np.all(labels_or_mask == np.round(labels_or_mask)):
                raise ValueError(
                    "Float mask values must be exactly integer-valued, "
                    "but fractional values were detected."
                )
        unique_vals = np.unique(labels_or_mask)
        observed_set = {int(x) for x in unique_vals}
    else:
        observed_set = {int(x) for x in labels_or_mask}

    allowed_set = set(label_map.keys()) | {0}

    if not observed_set.issubset(allowed_set):
        raise ValueError(
            f"Observed labels {sorted(observed_set)} are not a subset of allowed labels "
            f"{sorted(allowed_set)} (label map keys + background 0)."
        )

