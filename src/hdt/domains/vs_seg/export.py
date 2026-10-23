"""Case export and report generation for VS-SEG cases.

Converts and packages VS-SEG patients into standard HDT case directory structure:
data/processed/cases/VS-SEG-00N/
├── t2.nii.gz
├── structures/
│   ├── tumour.nii.gz
│   └── cochlea.nii.gz
├── structures_alt/
│   ├── tumour_capped.nii.gz
│   └── cochlea_capped.nii.gz
└── case.json

Computes metrics (volume, shoelace ratio, connected components, bounding boxes,
clearances) and logs the verification report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from hdt.domains.vs_seg.contours import StructureContours, load_patient_contours
from hdt.domains.vs_seg.sampler import (
    check_mask_touches_border,
    compute_shoelace_volume,
    rasterise_structure,
)
from hdt.geometry import mesh_clearance_mm, signed_clearance_mm
from hdt.imaging.dicom_series import load_affine_transform
from hdt.imaging.load_scan import ImageGeometry, load_nifti_image, save_nifti_image, save_nifti_mask


@dataclass
class StructureReport:
    """Report metrics for an individual anatomical mask."""

    name: str
    role: str
    voxel_count: int
    volume_mm3: float
    shoelace_volume_mm3: float
    shoelace_ratio: float
    connected_components: int
    bbox_ras_mm: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    multi_ring_planes: list[dict[str, Any]]


@dataclass
class CaseExportReport:
    """Full export report for a single case."""

    case_id: str
    tumour_slab: StructureReport
    tumour_capped: StructureReport
    cochlea_slab: StructureReport
    cochlea_capped: StructureReport
    slab_signed_clearance_mm: float
    slab_mesh_clearance_mm: float
    capped_signed_clearance_mm: float
    capped_mesh_clearance_mm: float
    signed_clearance_diff_mm: float
    mesh_clearance_diff_mm: float


def compute_structure_report(
    mask: np.ndarray,
    contours: StructureContours,
    target_geom: ImageGeometry,
    role: str,
) -> StructureReport:
    """Compute verification metrics for a single segmented structure."""
    vox_count = int(np.sum(mask))
    voxel_vol = float(
        target_geom.spacing_mm[0] * target_geom.spacing_mm[1] * target_geom.spacing_mm[2]
    )
    vol_mm3 = float(vox_count * voxel_vol)
    shoelace_vol = float(compute_shoelace_volume(contours))
    ratio = float(vol_mm3 / shoelace_vol) if shoelace_vol > 0 else 0.0

    # 6-connectivity connected components
    struct_6 = generate_binary_structure(3, 1)
    _, num_components = label(mask, structure=struct_6)

    # Multi-ring planes
    multi_rings = []
    for p in contours.planes:
        if len(p.rings_2d) > 1:
            multi_rings.append(
                {
                    "plane_coordinate_mm": float(p.coordinate),
                    "ring_count": len(p.rings_2d),
                }
            )

    # Bounding box in RAS mm
    if vox_count > 0:
        z_idx, y_idx, x_idx = np.where(mask)
        min_idx = (int(np.min(z_idx)), int(np.min(y_idx)), int(np.min(x_idx)))
        max_idx = (int(np.max(z_idx)), int(np.max(y_idx)), int(np.max(x_idx)))

        # Convert corners to LPS and then to RAS
        corners_vox = [
            (min_idx[0], min_idx[1], min_idx[2]),
            (min_idx[0], min_idx[1], max_idx[2]),
            (min_idx[0], max_idx[1], min_idx[2]),
            (min_idx[0], max_idx[1], max_idx[2]),
            (max_idx[0], min_idx[1], min_idx[2]),
            (max_idx[0], min_idx[1], max_idx[2]),
            (max_idx[0], max_idx[1], min_idx[2]),
            (max_idx[0], max_idx[1], max_idx[2]),
        ]
        ras_pts = []
        for cv in corners_vox:
            lps_pt = target_geom.voxel_to_lps_mm(cv)
            ras_pt = (-lps_pt[0], -lps_pt[1], lps_pt[2])
            ras_pts.append(ras_pt)

        ras_arr = np.array(ras_pts)
        bbox_ras = (
            (float(np.min(ras_arr[:, 0])), float(np.max(ras_arr[:, 0]))),
            (float(np.min(ras_arr[:, 1])), float(np.max(ras_arr[:, 1]))),
            (float(np.min(ras_arr[:, 2])), float(np.max(ras_arr[:, 2]))),
        )
    else:
        bbox_ras = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    return StructureReport(
        name=contours.structure_name,
        role=role,
        voxel_count=vox_count,
        volume_mm3=vol_mm3,
        shoelace_volume_mm3=shoelace_vol,
        shoelace_ratio=ratio,
        connected_components=int(num_components),
        bbox_ras_mm=bbox_ras,
        multi_ring_planes=multi_rings,
    )


def export_case(
    patient_id: str,
    vs_series_dir: Path | str,
    contours_dir: Path | str,
    tfm_matrices_dir: Path | str,
    output_cases_dir: Path | str,
) -> CaseExportReport:
    """Export a single patient into a processed case folder.

    Parameters
    ----------
    patient_id : str
        e.g. 'VS-SEG-001'
    vs_series_dir : Path | str
        Path to processed vs_series directory.
    contours_dir : Path | str
        Path to raw contours directory.
    tfm_matrices_dir : Path | str
        Path to raw registration matrices directory.
    output_cases_dir : Path | str
        Output cases destination directory.

    Returns
    -------
    CaseExportReport
        Detailed verification metrics for the exported case.
    """
    pid_int = int(patient_id.replace("VS-SEG-", ""))
    gk_id = f"vs_gk_{pid_int}"

    series_path = Path(vs_series_dir) / patient_id
    t1_file = series_path / "t1.nii.gz"
    t2_file = series_path / "t2.nii.gz"

    if not t1_file.is_file() or not t2_file.is_file():
        raise FileNotFoundError(f"Missing T1 or T2 NIfTI series for {patient_id} in {series_path}")

    t1_arr, t1_sp, t1_orig, t1_dir = load_nifti_image(t1_file)
    t2_arr, t2_sp, t2_orig, t2_dir = load_nifti_image(t2_file)

    t1_geom = ImageGeometry(
        shape=t1_arr.shape, spacing_mm=t1_sp, origin_mm=t1_orig, direction=t1_dir
    )
    t2_geom = ImageGeometry(
        shape=t2_arr.shape, spacing_mm=t2_sp, origin_mm=t2_orig, direction=t2_dir
    )

    # Load contours
    tumour_contours, cochlea_contours = load_patient_contours(
        patient_gk_id=gk_id,
        contours_dir=contours_dir,
        t1_geom=t1_geom,
        t2_geom=t2_geom,
    )

    # Load transform mapping T2 LPS -> T1 LPS
    tfm_file = Path(tfm_matrices_dir) / f"{gk_id}_t1" / "inv_T1_LPS_to_T2_LPS.tfm"
    if not tfm_file.is_file():
        raise FileNotFoundError(f"Registration transform not found: {tfm_file}")

    tfm = load_affine_transform(tfm_file)

    # 1. Rasterise Tumour (slab and capped) on native T2 grid
    half_spacing_t1 = float(t1_geom.spacing_mm[0]) / 2.0
    tumour_slab = rasterise_structure(
        rings=tumour_contours,
        native_geometry=t1_geom,
        target_geometry=t2_geom,
        transform=tfm,
        end_extent_mm=half_spacing_t1,
    )
    tumour_capped = rasterise_structure(
        rings=tumour_contours,
        native_geometry=t1_geom,
        target_geometry=t2_geom,
        transform=tfm,
        end_extent_mm=0.0,
    )

    # 2. Rasterise Cochlea (slab and capped) on native T2 grid (no transform needed)
    half_spacing_t2 = float(t2_geom.spacing_mm[0]) / 2.0
    cochlea_slab = rasterise_structure(
        rings=cochlea_contours,
        native_geometry=t2_geom,
        target_geometry=t2_geom,
        transform=None,
        end_extent_mm=half_spacing_t2,
    )
    cochlea_capped = rasterise_structure(
        rings=cochlea_contours,
        native_geometry=t2_geom,
        target_geometry=t2_geom,
        transform=None,
        end_extent_mm=0.0,
    )

    # 3. Assert structures are completely inside the T2 grid (never touch border)
    if check_mask_touches_border(tumour_slab):
        raise ValueError(f"Tumour slab mask touches grid border in {patient_id}!")
    if check_mask_touches_border(cochlea_slab):
        raise ValueError(f"Cochlea slab mask touches grid border in {patient_id}!")

    # 4. Write destination files
    case_out_dir = Path(output_cases_dir) / patient_id
    struct_dir = case_out_dir / "structures"
    struct_alt_dir = case_out_dir / "structures_alt"
    struct_dir.mkdir(parents=True, exist_ok=True)
    struct_alt_dir.mkdir(parents=True, exist_ok=True)

    # Native T2 image
    t2_out_file = case_out_dir / "t2.nii.gz"
    save_nifti_image(t2_arr, t2_geom, t2_out_file)

    # Masks
    save_nifti_mask(tumour_slab, t2_geom, struct_dir / "tumour.nii.gz")
    save_nifti_mask(cochlea_slab, t2_geom, struct_dir / "cochlea.nii.gz")
    save_nifti_mask(tumour_capped, t2_geom, struct_alt_dir / "tumour_capped.nii.gz")
    save_nifti_mask(cochlea_capped, t2_geom, struct_alt_dir / "cochlea_capped.nii.gz")

    # Metadata case.json
    case_json_data = {
        "case_id": patient_id,
        "images": ["t2"],
        "spacing_mm_zyx": [float(v) for v in t2_geom.spacing_mm],
        "shape_zyx": [int(v) for v in t2_geom.shape],
        "roles": {
            "target": "structures/tumour.nii.gz",
            "risk": "structures/cochlea.nii.gz",
        },
        "roi_names_used": {
            "tumour": tumour_contours.structure_name,
            "cochlea": cochlea_contours.structure_name,
        },
        "native_frame_per_structure": {
            "tumour": "t1",
            "cochlea": "t2",
        },
        "tfm_file": Path(
            tfm_file.relative_to(Path(tfm_matrices_dir).parent.parent)
        ).as_posix(),
        "end_extent_mm": half_spacing_t1,
        "source": "TCIA Vestibular-Schwannoma-SEG",
    }
    with open(case_out_dir / "case.json", "w", encoding="utf-8") as fp:
        json.dump(case_json_data, fp, indent=2)

    # 5. Clearances
    sp_zyx = t2_geom.spacing_mm
    slab_signed_clr = float(
        signed_clearance_mm(tumour_slab, cochlea_slab, sp_zyx, crop_to_bbox=True)
    )
    slab_mesh_clr = float(mesh_clearance_mm(tumour_slab, cochlea_slab, sp_zyx))
    capped_signed_clr = float(
        signed_clearance_mm(tumour_capped, cochlea_capped, sp_zyx, crop_to_bbox=True)
    )
    capped_mesh_clr = float(mesh_clearance_mm(tumour_capped, cochlea_capped, sp_zyx))

    # Reports
    tumour_slab_rep = compute_structure_report(tumour_slab, tumour_contours, t2_geom, "target")
    tumour_capped_rep = compute_structure_report(tumour_capped, tumour_contours, t2_geom, "target")
    cochlea_slab_rep = compute_structure_report(cochlea_slab, cochlea_contours, t2_geom, "risk")
    cochlea_capped_rep = compute_structure_report(cochlea_capped, cochlea_contours, t2_geom, "risk")

    return CaseExportReport(
        case_id=patient_id,
        tumour_slab=tumour_slab_rep,
        tumour_capped=tumour_capped_rep,
        cochlea_slab=cochlea_slab_rep,
        cochlea_capped=cochlea_capped_rep,
        slab_signed_clearance_mm=slab_signed_clr,
        slab_mesh_clearance_mm=slab_mesh_clr,
        capped_signed_clearance_mm=capped_signed_clr,
        capped_mesh_clearance_mm=capped_mesh_clr,
        signed_clearance_diff_mm=slab_signed_clr - capped_signed_clr,
        mesh_clearance_diff_mm=slab_mesh_clr - capped_mesh_clr,
    )
