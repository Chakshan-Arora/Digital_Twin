"""DICOM inspection utilities for research medical imaging datasets.

This module is the ONLY module in src/ permitted to import and use pydicom.
It provides pure, read-only inspection functions to evaluate DICOM completeness,
spatial geometry (slice spacing derived strictly from ImagePositionPatient projected
onto the slice normal), RTSTRUCT contours, and registration/modality mappings.
Units: distances in millimetres (mm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


def compute_slice_normal(
    image_orientation_patient: tuple[float, ...] | list[float] | np.ndarray,
) -> np.ndarray:
    """Compute unit slice normal vector from ImageOrientationPatient direction cosines.

    Parameters
    ----------
    image_orientation_patient : tuple[float, ...] | list[float] | np.ndarray
        Direction cosines [rx, ry, rz, cx, cy, cz] where r is the row vector
        and c is the column vector.

    Returns
    -------
    np.ndarray
        Unit normal vector of shape (3,).
    """
    iop = np.asarray(image_orientation_patient, dtype=np.float64)
    if iop.shape != (6,):
        raise ValueError(f"ImageOrientationPatient must have 6 elements, got {iop.shape}")
    row = iop[:3]
    col = iop[3:]
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        raise ValueError("Degenerate ImageOrientationPatient: row and column vectors are collinear")
    return normal / norm


def compute_slice_spacing_and_gaps(
    normal_projections: list[float],
    tolerance: float = 1e-4,
) -> tuple[float | None, list[dict[str, Any]], list[int]]:
    """Compute physical slice spacing, position gaps, and duplicate slice indices.

    Projects ImagePositionPatient onto the slice normal, sorts projections,
    and computes slice-to-slice increments.

    Parameters
    ----------
    normal_projections : list[float]
        Signed distance of each slice position along the unit normal (mm).
    tolerance : float
        Tolerance in mm to consider two slices as occupying the same position.

    Returns
    -------
    tuple[float | None, list[dict[str, Any]], list[int]]
        nominal_spacing: median of positive step increments (mm), or None if < 2 unique positions.
        gaps: list of detected gaps with index, step_mm, and expected_mm.
        duplicate_indices: list of sorted slice indices that duplicate preceding position.
    """
    if len(normal_projections) < 2:
        return None, [], []

    sorted_indices = np.argsort(normal_projections)
    sorted_pos = np.array([normal_projections[i] for i in sorted_indices], dtype=np.float64)
    diffs = np.diff(sorted_pos)

    # Detect duplicates
    duplicate_indices: list[int] = []
    positive_diffs: list[float] = []
    for idx, d in enumerate(diffs):
        if abs(d) <= tolerance:
            duplicate_indices.append(int(sorted_indices[idx + 1]))
        elif d > tolerance:
            positive_diffs.append(float(d))

    if not positive_diffs:
        return None, [], duplicate_indices

    nominal_spacing = float(np.median(positive_diffs))
    gaps: list[dict[str, Any]] = []

    # A gap is any step significantly larger than nominal spacing (e.g. >= 1.5x)
    for idx, d in enumerate(diffs):
        if d > 1.5 * nominal_spacing:
            gaps.append(
                {
                    "from_index": int(sorted_indices[idx]),
                    "to_index": int(sorted_indices[idx + 1]),
                    "step_mm": float(d),
                    "nominal_spacing_mm": nominal_spacing,
                    "missing_slices_approx": round(d / nominal_spacing) - 1,
                }
            )

    return nominal_spacing, gaps, duplicate_indices


@dataclass(frozen=True)
class SliceMetadata:
    """Metadata for a single slice file."""

    file_path: Path
    sop_instance_uid: str
    instance_number: int | None
    position_patient: tuple[float, float, float] | None
    normal_projection_mm: float | None


@dataclass(frozen=True)
class SeriesInspection:
    """Inspection result for a DICOM series."""

    series_instance_uid: str
    series_description: str
    modality: str
    folder_path: Path
    file_count: int
    distinct_positions_count: int
    rows: int | None
    columns: int | None
    pixel_spacing_mm: tuple[float, float] | None
    slice_thickness_mm: float | None
    slice_spacing_from_positions_mm: float | None
    image_orientation_patient: tuple[float, ...] | None
    frame_of_reference_uid: str | None
    is_truncated: bool
    position_gaps: list[dict[str, Any]] = field(default_factory=list)
    duplicate_files: list[str] = field(default_factory=list)
    slices: list[SliceMetadata] = field(default_factory=list)


def inspect_series_files(files: list[Path | str]) -> SeriesInspection:
    """Inspect a group of DICOM files belonging to a single series.

    Parameters
    ----------
    files : list[Path | str]
        Paths to DICOM slice files.

    Returns
    -------
    SeriesInspection
        Extracted metadata and completeness/geometry metrics.
    """
    if not files:
        raise ValueError("Cannot inspect empty list of files")

    slices: list[SliceMetadata] = []
    first_ds: pydicom.Dataset | None = None
    first_file = Path(files[0])
    folder_path = first_file.parent

    # Read each header (stop_before_pixels=True avoids loading heavy imaging data)
    normal: np.ndarray | None = None

    for f in files:
        path = Path(f)
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        if first_ds is None:
            first_ds = ds
            if hasattr(ds, "ImageOrientationPatient"):
                try:
                    normal = compute_slice_normal(ds.ImageOrientationPatient)
                except Exception:
                    normal = None

        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        inst_val = getattr(ds, "InstanceNumber", None)
        inst_num: int | None = None
        if inst_val is not None:
            try:
                inst_num = int(inst_val)
            except (ValueError, TypeError):
                inst_num = None

        ipp: tuple[float, float, float] | None = None
        proj_mm: float | None = None
        if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
            pos_arr = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
            ipp = (float(pos_arr[0]), float(pos_arr[1]), float(pos_arr[2]))
            if normal is not None:
                proj_mm = float(np.dot(pos_arr, normal))

        slices.append(
            SliceMetadata(
                file_path=path,
                sop_instance_uid=sop_uid,
                instance_number=inst_num,
                position_patient=ipp,
                normal_projection_mm=proj_mm,
            )
        )

    assert first_ds is not None
    suid = str(getattr(first_ds, "SeriesInstanceUID", "UNKNOWN"))
    desc = str(getattr(first_ds, "SeriesDescription", "No SeriesDescription"))
    modality = str(getattr(first_ds, "Modality", "UNKNOWN"))

    rows_val = getattr(first_ds, "Rows", None)
    rows = int(rows_val) if rows_val is not None else None
    cols_val = getattr(first_ds, "Columns", None)
    cols = int(cols_val) if cols_val is not None else None

    pixel_spacing: tuple[float, float] | None = None
    if hasattr(first_ds, "PixelSpacing") and first_ds.PixelSpacing is not None:
        try:
            pixel_spacing = (float(first_ds.PixelSpacing[0]), float(first_ds.PixelSpacing[1]))
        except (ValueError, TypeError, IndexError):
            pixel_spacing = None

    thick_val = getattr(first_ds, "SliceThickness", None)
    slice_thickness: float | None = None
    if thick_val is not None:
        try:
            slice_thickness = float(thick_val)
        except (ValueError, TypeError):
            slice_thickness = None

    iop: tuple[float, ...] | None = None
    if hasattr(first_ds, "ImageOrientationPatient") and first_ds.ImageOrientationPatient:
        iop = tuple(float(x) for x in first_ds.ImageOrientationPatient)

    for_uid = str(getattr(first_ds, "FrameOfReferenceUID", "")) or None

    # Geometry from positions
    projections = [s.normal_projection_mm for s in slices if s.normal_projection_mm is not None]
    distinct_positions_count = len({round(p, 4) for p in projections}) if projections else 0

    duplicate_files: list[str] = []
    seen_sops: set[str] = set()
    for s in slices:
        if s.sop_instance_uid in seen_sops:
            duplicate_files.append(s.file_path.name)
        seen_sops.add(s.sop_instance_uid)

    nominal_spacing, gaps, dup_indices = compute_slice_spacing_and_gaps(projections)
    for idx in dup_indices:
        fname = slices[idx].file_path.name
        if fname not in duplicate_files:
            duplicate_files.append(fname)

    is_truncated = len(gaps) > 0

    return SeriesInspection(
        series_instance_uid=suid,
        series_description=desc,
        modality=modality,
        folder_path=folder_path,
        file_count=len(slices),
        distinct_positions_count=distinct_positions_count,
        rows=rows,
        columns=cols,
        pixel_spacing_mm=pixel_spacing,
        slice_thickness_mm=slice_thickness,
        slice_spacing_from_positions_mm=nominal_spacing,
        image_orientation_patient=iop,
        frame_of_reference_uid=for_uid,
        is_truncated=is_truncated,
        position_gaps=gaps,
        duplicate_files=duplicate_files,
        slices=slices,
    )


@dataclass(frozen=True)
class ROIInspection:
    """Inspection result for an ROI in an RTSTRUCT file."""

    roi_number: int
    roi_name: str
    contour_count: int
    geometric_types: list[str]
    total_point_count: int
    referenced_sop_uids: list[str]
    matching_image_series_desc: str | None


@dataclass(frozen=True)
class RTStructInspection:
    """Inspection result for an RTSTRUCT file."""

    file_path: Path
    sop_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str | None
    referenced_series_uids: list[str]
    rois: list[ROIInspection]


def inspect_rtstruct(
    file_path: Path | str,
    image_series_list: list[SeriesInspection] | None = None,
) -> RTStructInspection:
    """Inspect an RTSTRUCT file and cross-reference ROIs with image series planes.

    Parameters
    ----------
    file_path : Path | str
        Path to RTSTRUCT DICOM file.
    image_series_list : list[SeriesInspection] | None
        List of inspected image series from the same study/patient.

    Returns
    -------
    RTStructInspection
        Parsed ROI details, contour counts, geometric types, and referenced series.
    """
    path = Path(file_path)
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    if getattr(ds, "Modality", None) != "RTSTRUCT":
        mod = getattr(ds, "Modality", None)
        raise ValueError(f"File {path} is not an RTSTRUCT (Modality={mod})")

    sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
    suid = str(getattr(ds, "SeriesInstanceUID", ""))
    for_uid = str(getattr(ds, "FrameOfReferenceUID", "")) or None

    # Find referenced series in ReferencedFrameOfReferenceSequence
    ref_series_uids: list[str] = []
    if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
        for for_item in ds.ReferencedFrameOfReferenceSequence:
            if hasattr(for_item, "RTReferencedStudySequence"):
                for study_item in for_item.RTReferencedStudySequence:
                    if hasattr(study_item, "RTReferencedSeriesSequence"):
                        for series_item in study_item.RTReferencedSeriesSequence:
                            if hasattr(series_item, "SeriesInstanceUID"):
                                ref_series_uids.append(str(series_item.SeriesInstanceUID))

    # Build map of ROINumber to ROIName
    roi_names: dict[int, str] = {}
    if hasattr(ds, "StructureSetROISequence"):
        for item in ds.StructureSetROISequence:
            roi_names[int(item.ROINumber)] = str(item.ROIName)

    # Inspect ROIContourSequence
    rois: list[ROIInspection] = []
    if hasattr(ds, "ROIContourSequence"):
        for roi_contour in ds.ROIContourSequence:
            roi_num = int(roi_contour.ReferencedROINumber)
            roi_name = roi_names.get(roi_num, f"ROI_{roi_num}")

            contour_seq = getattr(roi_contour, "ContourSequence", [])
            contour_count = len(contour_seq)

            geom_types: set[str] = set()
            total_pts = 0
            ref_sops: list[str] = []

            for contour in contour_seq:
                gtype = str(getattr(contour, "ContourGeometricType", "UNKNOWN"))
                geom_types.add(gtype)
                n_pts = int(getattr(contour, "NumberOfContourPoints", 0))
                total_pts += n_pts

                if hasattr(contour, "ContourImageSequence"):
                    for ci in contour.ContourImageSequence:
                        if hasattr(ci, "ReferencedSOPInstanceUID"):
                            ref_sops.append(str(ci.ReferencedSOPInstanceUID))

            # Determine matching image series
            matching_desc: str | None = None
            if image_series_list:
                # Check referenced SOP match first
                matched_series = []
                for s in image_series_list:
                    s_sop_set = {sl.sop_instance_uid for sl in s.slices}
                    overlap = set(ref_sops) & s_sop_set
                    if overlap:
                        matched_series.append((s.series_description, len(overlap)))

                if matched_series:
                    matched_series.sort(key=lambda x: x[1], reverse=True)
                    matching_desc = matched_series[0][0]
                elif ref_series_uids:
                    # Fallback to ReferencedSeriesSequence UID
                    for s in image_series_list:
                        if s.series_instance_uid in ref_series_uids:
                            matching_desc = s.series_description
                            break

            rois.append(
                ROIInspection(
                    roi_number=roi_num,
                    roi_name=roi_name,
                    contour_count=contour_count,
                    geometric_types=sorted(list(geom_types)),
                    total_point_count=total_pts,
                    referenced_sop_uids=ref_sops,
                    matching_image_series_desc=matching_desc,
                )
            )

    return RTStructInspection(
        file_path=path,
        sop_instance_uid=sop_uid,
        series_instance_uid=suid,
        frame_of_reference_uid=for_uid,
        referenced_series_uids=ref_series_uids,
        rois=rois,
    )


@dataclass(frozen=True)
class PatientInspection:
    """Inspection result across all series for a patient."""

    patient_id: str
    series: list[SeriesInspection]
    rtstructs: list[RTStructInspection]
    has_contrast_t1: bool
    has_t2: bool
    has_rtstruct: bool
    has_rtdose: bool
    has_rtplan: bool
    is_complete: bool
    missing_items: list[str]


def inspect_patient_dicom_tree(patient_dir: Path | str) -> PatientInspection:
    """Inspect all DICOM series and RT objects under a patient directory.

    Groups files by SeriesInstanceUID, determines MR vs RT objects,
    computes slice geometry and verifies presence of T1, T2, RTSTRUCT, RTDOSE, RTPLAN.

    Parameters
    ----------
    patient_dir : Path | str
        Path to patient directory (e.g. data/raw/vs_seg/vestibular_schwannoma_seg/VS-SEG-001).

    Returns
    -------
    PatientInspection
        Summary of series, RT objects, completeness, and geometry.
    """
    p_dir = Path(patient_dir)
    patient_id = p_dir.name

    # Discover all DICOM files
    all_dcm_files: list[Path] = [
        f for f in p_dir.rglob("*") if f.is_file() and not f.name.startswith(".")
    ]

    # Group files by SeriesInstanceUID
    series_groups: dict[str, list[Path]] = {}
    for f in all_dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            suid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
            series_groups.setdefault(suid, []).append(f)
        except Exception:
            continue

    inspected_series: list[SeriesInspection] = []
    for _suid, flist in series_groups.items():
        inspected = inspect_series_files(flist)
        inspected_series.append(inspected)

    # Separate image series from RTSTRUCTs
    image_series = [s for s in inspected_series if s.modality in ("MR", "CT")]

    rtstructs: list[RTStructInspection] = []
    for s in inspected_series:
        if s.modality == "RTSTRUCT":
            for sl in s.slices:
                rtstructs.append(inspect_rtstruct(sl.file_path, image_series))

    # Check required modalities
    # T1: check SeriesDescription or modality mapping
    has_t1 = any(
        s.modality == "MR"
        and ("t1" in s.series_description.lower() or "mpr" in s.series_description.lower())
        for s in inspected_series
    )
    has_t2 = any(
        s.modality == "MR" and any(k in s.series_description.lower() for k in ("t2", "ci3d", "tse"))
        for s in inspected_series
    )
    has_rtstruct = any(s.modality == "RTSTRUCT" for s in inspected_series)
    has_rtdose = any(s.modality == "RTDOSE" for s in inspected_series)
    has_rtplan = any(s.modality == "RTPLAN" for s in inspected_series)

    missing: list[str] = []
    if not has_t1:
        missing.append("T1 image")
    if not has_t2:
        missing.append("T2 image")
    if not has_rtstruct:
        missing.append("RTSTRUCT")
    if not has_rtdose:
        missing.append("RTDOSE")
    if not has_rtplan:
        missing.append("RTPLAN")

    is_complete = len(missing) == 0 and not any(s.is_truncated for s in image_series)

    return PatientInspection(
        patient_id=patient_id,
        series=inspected_series,
        rtstructs=rtstructs,
        has_contrast_t1=has_t1,
        has_t2=has_t2,
        has_rtstruct=has_rtstruct,
        has_rtdose=has_rtdose,
        has_rtplan=has_rtplan,
        is_complete=is_complete,
        missing_items=missing,
    )
