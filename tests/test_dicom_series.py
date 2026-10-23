"""Tests for DICOM series loader and geometry verification.

Tests:
- Small oblique synthetic DICOM series created with pydicom in tmp_path.
- Strict array order (z, y, x), spacing (z, y, x), origin (z, y, x), and direction.
- Independent nibabel affine check on exported NIfTI.
- Position gap raises ValueError.
- Duplicate slice position raises ValueError.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from hdt.imaging.dicom_series import load_series
from hdt.imaging.load_scan import ImageGeometry, save_nifti_image


def _write_synthetic_slice_with_pixels(
    path: Path,
    iop: list[float],
    ipp: list[float],
    series_uid: str,
    sop_uid: str,
    instance_number: int,
    spacing_xy: tuple[float, float] = (0.8, 0.6),  # (row_spacing, col_spacing)
    thickness: float = 1.5,
    rows: int = 16,
    cols: int = 16,
    slice_val: int = 10,
) -> None:
    """Helper to write a synthetic DICOM slice with pixel data."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.8.7001"
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesDescription = "Synthetic Oblique Series"
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [spacing_xy[0], spacing_xy[1]]
    ds.SliceThickness = thickness
    ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.ImagePositionPatient = [float(x) for x in ipp]
    ds.InstanceNumber = instance_number
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.7002"

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    pixels = (np.ones((rows, cols), dtype=np.uint16) * slice_val).tobytes()
    ds.PixelData = pixels

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))


def test_load_series_oblique_and_nibabel_affine(tmp_path: Path) -> None:
    """Verify loading synthetic oblique DICOM series, array order, geometry,

    and independent nibabel affine agreement on the written NIfTI.
    """
    series_dir = tmp_path / "valid_oblique"
    series_dir.mkdir()

    series_uid = "1.2.826.0.1.3680043.8.7003"
    # Oblique direction cosines:
    # row = [1/3, 2/3, 2/3], col = [2/3, 1/3, -2/3], normal = [-2/3, 2/3, -1/3]
    iop = [1 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3, -2 / 3]
    normal = np.array([-2 / 3, 2 / 3, -1 / 3], dtype=np.float64)
    origin_lps_xyz = np.array([12.0, -24.0, 36.0], dtype=np.float64)
    spacing_zyx = (1.5, 0.8, 0.6)  # (sz, sy, sx)

    n_slices = 4
    # Intentionally write files out of order to ensure position-based sorting works
    slice_order = [2, 0, 3, 1]
    for k in slice_order:
        pos_k = origin_lps_xyz + k * spacing_zyx[0] * normal
        slice_path = series_dir / f"slice_{k:02d}.dcm"
        _write_synthetic_slice_with_pixels(
            path=slice_path,
            iop=iop,
            ipp=list(pos_k),
            series_uid=series_uid,
            sop_uid=f"1.2.826.0.1.3680043.8.7004.{k}",
            instance_number=k + 1,
            spacing_xy=(spacing_zyx[1], spacing_zyx[2]),
            thickness=spacing_zyx[0],
            slice_val=(k + 1) * 10,
        )

    # Load series
    arr, geom = load_series(series_dir)

    # 1. Assert shape and array order in (z, y, x)
    assert arr.shape == (n_slices, 16, 16)
    assert arr.dtype == np.float32
    for k in range(n_slices):
        expected_val = (k + 1) * 10.0
        assert np.allclose(arr[k], expected_val)

    # 2. Assert spacing in (z, y, x)
    np.testing.assert_allclose(geom.spacing_mm, spacing_zyx, atol=1e-5)

    # 3. Assert origin in (z, y, x)
    origin_zyx_expected = (origin_lps_xyz[2], origin_lps_xyz[1], origin_lps_xyz[0])
    np.testing.assert_allclose(geom.origin_mm, origin_zyx_expected, atol=1e-5)

    # 4. Assert direction matrix (3x3 ITK LPS space)
    # Column 0 is row vector, column 1 is col vector, column 2 is slice normal
    expected_dir = np.column_stack([iop[:3], iop[3:], normal])
    np.testing.assert_allclose(geom.direction, expected_dir, atol=1e-5)

    # 5. Independent nibabel affine check on exported NIfTI
    nii_file = tmp_path / "exported_oblique.nii.gz"
    save_nifti_image(arr, geom, nii_file)

    nib_img = nib.load(str(nii_file))
    nib_affine = nib_img.affine

    rng = np.random.default_rng(1234)
    for _ in range(10):
        vz = float(rng.uniform(0.0, geom.shape[0] - 1.0))
        vy = float(rng.uniform(0.0, geom.shape[1] - 1.0))
        vx = float(rng.uniform(0.0, geom.shape[2] - 1.0))

        # HDT forward mapping: continuous voxel (z, y, x) -> LPS -> RAS
        lps_pt = geom.voxel_to_lps_mm((vz, vy, vx))
        ras_pt = ImageGeometry.lps_to_ras(lps_pt)

        # Nibabel affine applied to (x, y, z, 1)
        nib_homog = nib_affine @ np.array([vx, vy, vz, 1.0], dtype=np.float64)
        nib_ras = (float(nib_homog[0]), float(nib_homog[1]), float(nib_homog[2]))

        np.testing.assert_allclose(ras_pt, nib_ras, atol=1e-4)


def test_load_series_gap_raises(tmp_path: Path) -> None:
    """Missing slice along normal creates non-constant spacing and must raise ValueError."""
    series_dir = tmp_path / "gap_series"
    series_dir.mkdir()

    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.array([0.0, 0.0, 0.0])
    spacing_z = 1.5

    # Write slices 0, 1, and 3 (slice 2 missing -> gap = 3.0 mm)
    for k in [0, 1, 3]:
        pos_k = origin + k * spacing_z * normal
        _write_synthetic_slice_with_pixels(
            path=series_dir / f"slice_{k:02d}.dcm",
            iop=iop,
            ipp=list(pos_k),
            series_uid="1.2.826.0.1.3680043.8.7005",
            sop_uid=f"1.2.826.0.1.3680043.8.7005.{k}",
            instance_number=k + 1,
            thickness=spacing_z,
        )

    with pytest.raises(ValueError, match="Non-constant slice spacing"):
        load_series(series_dir)


def test_load_series_duplicate_raises(tmp_path: Path) -> None:
    """Duplicate slice position along normal must raise ValueError."""
    series_dir = tmp_path / "dup_series"
    series_dir.mkdir()

    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    normal = np.array([0.0, 0.0, 1.0])
    origin = np.array([0.0, 0.0, 0.0])
    spacing_z = 1.5

    # Write slices 0, 1, 1 (duplicate), 2
    for idx, k in enumerate([0, 1, 1, 2]):
        pos_k = origin + k * spacing_z * normal
        _write_synthetic_slice_with_pixels(
            path=series_dir / f"slice_{idx:02d}.dcm",
            iop=iop,
            ipp=list(pos_k),
            series_uid="1.2.826.0.1.3680043.8.7006",
            sop_uid=f"1.2.826.0.1.3680043.8.7006.{idx}",
            instance_number=idx + 1,
            thickness=spacing_z,
        )

    with pytest.raises(ValueError, match="Duplicate slice position"):
        load_series(series_dir)
