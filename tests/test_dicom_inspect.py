"""Tests for DICOM inspection functions with synthetic DICOM datasets.

Verifies:
- Oblique 3D slice normal derivation from ImageOrientationPatient.
- Exact slice spacing derived from ImagePositionPatient projected onto slice normal.
- Detection of slice position gaps (truncated series).
- Detection of duplicate slice files.
- Parsing and plane matching of RTSTRUCT contours.
"""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from hdt.imaging.dicom_inspect import (
    compute_slice_normal,
    compute_slice_spacing_and_gaps,
    inspect_rtstruct,
    inspect_series_files,
)


def _write_synthetic_slice(
    path: Path,
    iop: list[float],
    ipp: list[float],
    series_uid: str,
    sop_uid: str,
    instance_number: int,
    spacing_xy: tuple[float, float] = (0.5, 0.5),
    thickness: float = 1.5,
) -> None:
    """Helper to write a valid synthetic DICOM slice with pydicom."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.8.1000"
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesDescription = "Synthetic Oblique T1"
    ds.Rows = 32
    ds.Columns = 32
    ds.PixelSpacing = [spacing_xy[0], spacing_xy[1]]
    ds.SliceThickness = thickness
    ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.ImagePositionPatient = [float(x) for x in ipp]
    ds.InstanceNumber = instance_number
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.2000"
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))


def test_compute_slice_normal_analytic() -> None:
    """Verify analytic slice normal on standard and oblique orientations."""
    # Axial orientation: row=(1, 0, 0), col=(0, 1, 0) -> normal=(0, 0, 1)
    axial_iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    n_axial = compute_slice_normal(axial_iop)
    assert np.allclose(n_axial, [0.0, 0.0, 1.0])

    # Oblique orientation:
    # row = [1/3, 2/3, 2/3], col = [2/3, 1/3, -2/3]
    # normal = row x col = [-2/3, 2/3, -1/3]
    oblique_iop = [1 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3, -2 / 3]
    n_oblique = compute_slice_normal(oblique_iop)
    expected_normal = np.array([-2 / 3, 2 / 3, -1 / 3])
    assert np.allclose(n_oblique, expected_normal)
    assert np.isclose(np.linalg.norm(n_oblique), 1.0)


def test_compute_slice_normal_invalid() -> None:
    """Collinear or invalid length vectors raise ValueError."""
    with pytest.raises(ValueError, match="must have 6 elements"):
        compute_slice_normal([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="collinear"):
        compute_slice_normal([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])


def test_compute_slice_spacing_and_gaps() -> None:
    """Verify nominal spacing, gap detection, and duplicate detection."""
    # Projections at 0, 1.5, 3.0, 6.0 (gap at 4.5 missing), 6.0 (duplicate)
    projs = [0.0, 1.5, 3.0, 6.0, 6.0]
    nominal, gaps, dups = compute_slice_spacing_and_gaps(projs)
    assert nominal == pytest.approx(1.5)
    assert len(gaps) == 1
    assert gaps[0]["step_mm"] == pytest.approx(3.0)
    assert gaps[0]["missing_slices_approx"] == 1
    assert len(dups) == 1


def test_synthetic_dicom_series_oblique_gap_and_duplicate(tmp_path: Path) -> None:
    """Synthetic DICOM series in tmp_path with oblique orientation, 1.5 mm slices,

    one missing slice gap, and one duplicate file.
    Inspector must report the gap, the duplicate, and the right spacing from positions.
    """
    series_dir = tmp_path / "oblique_series"
    series_dir.mkdir()

    series_uid = "1.2.826.0.1.3680043.8.1001"
    iop = [1 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3, -2 / 3]
    normal = compute_slice_normal(iop)
    origin = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    nominal_spacing = 1.5  # mm

    # Slices with indices [0, 1, 2, 4, 5] -> slice 3 is missing (gap of 3.0 mm between 2 and 4)
    slice_indices = [0, 1, 2, 4, 5]
    written_files: list[Path] = []

    for k in slice_indices:
        pos_k = origin + k * nominal_spacing * normal
        file_path = series_dir / f"slice_{k:03d}.dcm"
        sop_uid = f"1.2.826.0.1.3680043.8.3000.{k}"
        _write_synthetic_slice(
            path=file_path,
            iop=iop,
            ipp=list(pos_k),
            series_uid=series_uid,
            sop_uid=sop_uid,
            instance_number=k,
            thickness=nominal_spacing,
        )
        written_files.append(file_path)

    # Add duplicate file: duplicate position of slice index 2
    dup_pos = origin + 2 * nominal_spacing * normal
    dup_file = series_dir / "slice_002_duplicate.dcm"
    _write_synthetic_slice(
        path=dup_file,
        iop=iop,
        ipp=list(dup_pos),
        series_uid=series_uid,
        sop_uid="1.2.826.0.1.3680043.8.3000.999",  # different SOP UID, same physical position
        instance_number=2,
        thickness=nominal_spacing,
    )
    written_files.append(dup_file)

    # Run inspector
    inspection = inspect_series_files(written_files)

    # 1. Spacing from positions must match nominal physical spacing (1.5 mm)
    assert inspection.slice_spacing_from_positions_mm == pytest.approx(1.5, abs=1e-4)

    # 2. Must detect gap (truncated series)
    assert inspection.is_truncated is True
    assert len(inspection.position_gaps) == 1
    gap = inspection.position_gaps[0]
    assert gap["step_mm"] == pytest.approx(3.0, abs=1e-4)
    assert gap["nominal_spacing_mm"] == pytest.approx(1.5, abs=1e-4)
    assert gap["missing_slices_approx"] == 1

    # 3. Must detect duplicate file
    assert len(inspection.duplicate_files) == 1
    assert "slice_002_duplicate.dcm" in inspection.duplicate_files

    # 4. Total files vs distinct positions
    assert inspection.file_count == 6
    assert inspection.distinct_positions_count == 5


def test_synthetic_rtstruct_inspection(tmp_path: Path) -> None:
    """Verify inspection of an RTSTRUCT file cross-referenced with image series."""
    # Write one image slice
    img_dir = tmp_path / "img_series"
    img_dir.mkdir()
    series_uid = "1.2.826.0.1.3680043.8.5001"
    sop_uid = "1.2.826.0.1.3680043.8.5002"
    img_file = img_dir / "img_001.dcm"
    _write_synthetic_slice(
        path=img_file,
        iop=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ipp=[0.0, 0.0, 10.0],
        series_uid=series_uid,
        sop_uid=sop_uid,
        instance_number=1,
    )
    image_inspection = inspect_series_files([img_file])

    # Write RTSTRUCT
    rt_file = tmp_path / "test_rtstruct.dcm"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.6001"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(rt_file), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    ds.SOPInstanceUID = "1.2.826.0.1.3680043.8.6001"
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.8.1000"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.8.6002"
    ds.Modality = "RTSTRUCT"
    ds.SeriesDescription = "Test RTSTRUCT"

    # StructureSetROISequence
    roi_item = pydicom.Dataset()
    roi_item.ROINumber = 1
    roi_item.ROIName = "Tumor"
    roi_item.ReferencedFrameOfReferenceUID = "1.2.826.0.1.3680043.8.2000"
    ds.StructureSetROISequence = Sequence([roi_item])

    # ROIContourSequence
    contour_ds = pydicom.Dataset()
    contour_ds.ReferencedROINumber = 1

    c_item = pydicom.Dataset()
    c_item.ContourGeometricType = "CLOSED_PLANAR"
    c_item.NumberOfContourPoints = 3
    c_item.ContourData = [0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 0.0, 1.0, 10.0]

    ci = pydicom.Dataset()
    ci.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ci.ReferencedSOPInstanceUID = sop_uid
    c_item.ContourImageSequence = Sequence([ci])

    contour_ds.ContourSequence = Sequence([c_item])
    ds.ROIContourSequence = Sequence([contour_ds])

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(rt_file))

    # Inspect RTSTRUCT
    rt_insp = inspect_rtstruct(rt_file, [image_inspection])
    assert len(rt_insp.rois) == 1
    roi = rt_insp.rois[0]
    assert roi.roi_name == "Tumor"
    assert roi.contour_count == 1
    assert roi.geometric_types == ["CLOSED_PLANAR"]
    assert roi.total_point_count == 3
    assert roi.matching_image_series_desc == "Synthetic Oblique T1"
