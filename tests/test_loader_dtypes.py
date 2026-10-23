"""Tests for NIfTI header inspection, pixel data types, and memory-safe metadata loading."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from hdt.imaging.load_scan import (
    _SITK_TO_NUMPY_DTYPE,
    inspect_nifti_header,
    load_nifti_mask_array,
)


def _make_nifti_with_pixel_type(
    tmp_path: Path,
    sitk_pixel_type: int,
    filename: str,
    shape_zyx: tuple[int, int, int] = (4, 5, 6),
) -> Path:
    """Helper to create a synthetic NIfTI volume with a specific SimpleITK pixel type."""
    sz, sy, sx = shape_zyx
    # SimpleITK image size is (sx, sy, sz)
    img = sitk.Image(sx, sy, sz, sitk_pixel_type)
    img.SetSpacing((1.0, 1.5, 2.0))
    img.SetOrigin((10.0, 20.0, 30.0))
    file_path = tmp_path / filename
    sitk.WriteImage(img, str(file_path))
    return file_path


@pytest.mark.parametrize(
    ("sitk_type", "expected_dtype_str"),
    [
        (sitk.sitkInt8, "int8"),
        (sitk.sitkUInt8, "uint8"),
        (sitk.sitkInt16, "int16"),
        (sitk.sitkUInt16, "uint16"),
        (sitk.sitkInt32, "int32"),
        (sitk.sitkUInt32, "uint32"),
        (sitk.sitkFloat32, "float32"),
        (sitk.sitkFloat64, "float64"),
    ],
)
def test_all_supported_pixel_types(
    tmp_path: Path,
    sitk_type: int,
    expected_dtype_str: str,
) -> None:
    """Verify inspect_nifti_header correctly recognizes every supported pixel type."""
    file_path = _make_nifti_with_pixel_type(
        tmp_path,
        sitk_type,
        f"test_{expected_dtype_str}.nii.gz",
        shape_zyx=(4, 5, 6),
    )

    info = inspect_nifti_header(file_path)
    assert info["dimension"] == 3
    assert info["shape_zyx"] == (4, 5, 6)
    assert info["dtype"] == expected_dtype_str
    assert pytest.approx(info["spacing_zyx"]) == (2.0, 1.5, 1.0)
    assert pytest.approx(info["origin_zyx"]) == (30.0, 20.0, 10.0)
    assert isinstance(info["direction"], np.ndarray)
    assert info["direction"].shape == (3, 3)

    # Also verify load_nifti_mask_array reads native dtype string correctly
    _, _, orig_dtype = load_nifti_mask_array(file_path)
    assert orig_dtype == expected_dtype_str


def test_sitk_to_numpy_dtype_completeness() -> None:
    """Ensure all declared pixel types in _SITK_TO_NUMPY_DTYPE are tested and non-empty."""
    assert len(_SITK_TO_NUMPY_DTYPE) == 8
    expected_keys = {
        sitk.sitkInt8,
        sitk.sitkUInt8,
        sitk.sitkInt16,
        sitk.sitkUInt16,
        sitk.sitkInt32,
        sitk.sitkUInt32,
        sitk.sitkFloat32,
        sitk.sitkFloat64,
    }
    assert set(_SITK_TO_NUMPY_DTYPE.keys()) == expected_keys


def test_large_volume_header_inspection_reproduction(tmp_path: Path) -> None:
    """Synthetic reproduction of BraTS-MEN-RT-0087-1 memory exhaustion vs inspect_nifti_header.

    BraTS-MEN-RT-0087-1 has dimensions (215, 704, 704) = 106.5 million voxels (~426 MB in float32).
    Attempting to decompress and hold multiple full volumes in memory simultaneously triggers
    bad_allocation / out-of-memory errors on constrained test runners.
    inspect_nifti_header uses ReadImageInformation() to parse the NIfTI header in milliseconds
    without decompressing or allocating voxel buffers.
    """
    large_file = tmp_path / "large_header_0087_sim.nii.gz"

    # Write a sparse/empty NIfTI with dimensions identical to BraTS-MEN-RT-0087-1: (215, 704, 704)
    # SimpleITK size is (x=704, y=704, z=215)
    img = sitk.Image(704, 704, 215, sitk.sitkInt16)
    img.SetSpacing((0.5, 0.5, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(img, str(large_file))

    # Inspecting header should succeed fast and allocate negligible RAM
    info = inspect_nifti_header(large_file)
    assert info["shape_zyx"] == (215, 704, 704)
    assert info["spacing_zyx"] == (1.0, 0.5, 0.5)
    assert info["dtype"] == "int16"
    assert info["dimension"] == 3
