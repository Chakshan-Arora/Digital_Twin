"""Brain MRI domain adapter for BraTS2024-MEN-RT and related datasets."""

from pathlib import Path

import numpy as np

from hdt.core.case import Case
from hdt.core.structure import Structure
from hdt.domains.brain_mri.label_maps import BRATS_MEN_RT, validate_labels
from hdt.imaging.load_scan import (
    ImageGeometry,
    assert_same_geometry,
    load_nifti_image,
    load_nifti_mask_array,
)


def load_case(
    root: str | Path,
    case_id: str,
    layout: str = "brats_folder",
    label_map: dict[int, str] | None = None,
) -> Case:
    """Load a brain MRI case from disk into HDT Case data model.

    Adheres strictly to HDT conventions:
    - Arrays are NumPy in (z, y, x) order.
    - Images are cast to float32 with NO rescaling or normalization.
    - Physical coordinates and direction cosine matrices are LPS (SimpleITK).
    - Structures are boolean masks in (z, y, x) order with assigned roles.

    Args:
        root: Root directory of dataset or directory containing case folders.
        case_id: Identifier of the case (directory name or prefix).
        layout: Layout format (currently "brats_folder").
        label_map: Label dictionary mapping integer labels to structure names.
            Defaults to BRATS_MEN_RT ({1: "gtv"}).

    Returns:
        Case object initialized with image modalities and segmented structures.

    Raises:
        ValueError: If layout is unsupported, files are duplicated, float mask is
            fractional, observed labels are invalid, or GTV mask is empty.
        FileNotFoundError: If case folder, modality file, or mask file is missing.
        GeometryMismatchError: If image and mask geometries do not match.
    """
    if layout != "brats_folder":
        raise ValueError(
            f"Unsupported layout '{layout}'. Only 'brats_folder' is supported by this adapter."
        )

    if label_map is None:
        label_map = BRATS_MEN_RT

    root_path = Path(root).resolve()
    case_dir = root_path / case_id if (root_path / case_id).is_dir() else root_path
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: '{case_dir}'.")

    # Locate modality file (_t1c.nii.gz or _t1c.nii)
    t1c_candidates = sorted(
        [
            f
            for f in case_dir.iterdir()
            if f.is_file()
            and (f.name.endswith("_t1c.nii.gz") or f.name.endswith("_t1c.nii"))
        ]
    )
    if len(t1c_candidates) == 0:
        raise FileNotFoundError(
            f"Missing modality file ending with '_t1c.nii.gz' for case '{case_id}' in '{case_dir}'."
        )
    if len(t1c_candidates) > 1:
        raise ValueError(
            f"Duplicate modality files ending with '_t1c.nii.gz' found for case '{case_id}': "
            f"{[f.name for f in t1c_candidates]}."
        )
    t1c_path = t1c_candidates[0]

    # Locate segmentation mask file (_gtv.nii.gz or _gtv.nii)
    gtv_candidates = sorted(
        [
            f
            for f in case_dir.iterdir()
            if f.is_file()
            and (f.name.endswith("_gtv.nii.gz") or f.name.endswith("_gtv.nii"))
        ]
    )
    if len(gtv_candidates) == 0:
        raise FileNotFoundError(
            f"Missing segmentation file ending with '_gtv.nii.gz' for case '{case_id}' "
            f"in '{case_dir}'."
        )
    if len(gtv_candidates) > 1:
        raise ValueError(
            f"Duplicate segmentation files ending with '_gtv.nii.gz' found for case '{case_id}': "
            f"{[f.name for f in gtv_candidates]}."
        )
    gtv_path = gtv_candidates[0]

    # Load modality (cast to float32 with NO rescaling)
    img_arr, sp_zyx, orig_zyx, direction = load_nifti_image(t1c_path)
    img_geom = ImageGeometry(
        shape=img_arr.shape,
        spacing_mm=sp_zyx,
        origin_mm=orig_zyx,
        direction=direction,
    )

    # Load segmentation mask
    mask_arr, mask_geom, _ = load_nifti_mask_array(gtv_path, image_meta=img_geom)
    assert_same_geometry(img_geom, mask_geom)

    # Validate label values and ensure float masks are strictly integer-valued
    validate_labels(mask_arr, label_map)

    # Extract GTV boolean mask (label 1)
    gtv_mask_bool = np.array(mask_arr == 1, dtype=bool)

    # Enforce non-empty GTV requirement
    if not np.any(gtv_mask_bool):
        raise ValueError(
            f"Empty GTV mask detected for case '{case_id}'. "
            f"GTV segmentation must contain at least one positive voxel."
        )

    gtv_struct = Structure(name="gtv", mask=gtv_mask_bool, role="target")

    return Case(
        case_id=case_id,
        images={"t1c": img_arr},
        spacing_mm=sp_zyx,
        origin_mm=orig_zyx,
        direction=direction,
        structures={"gtv": gtv_struct},
        primary="t1c",
    )
