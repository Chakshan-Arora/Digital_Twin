"""Unit tests for geometry-safe NIfTI I/O, coordinate conversions, and axis preservation."""

import ast
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from hdt.imaging.load_scan import (
    GeometryMismatchError,
    ImageGeometry,
    assert_same_geometry,
    load_nifti_image,
    load_nifti_mask,
    lps_to_ras,
    save_nifti_image,
    save_nifti_mask,
)
from hdt.imaging.phantoms import make_sphere


def test_round_trip_preserves_anisotropic_geometry_and_direction(tmp_path: Path) -> None:
    """Round trip test: write to NIfTI with non-trivial spacing, origin, and non-identity direction.

    Assert that loading back reproduces spacing, origin, direction, and array equality exactly.
    This test catches any axis-order swap bugs between SimpleITK (x, y, z) and HDT (z, y, x).
    """
    file_path = tmp_path / "test_phantom.nii.gz"

    shape_zyx = (16, 24, 32)
    spacing_zyx = (3.5, 1.25, 0.75)
    origin_zyx = (15.5, -25.0, 42.1)

    theta = np.deg2rad(30.0)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    direction = np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ref_geom = ImageGeometry(
        shape=shape_zyx,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction,
    )

    z, y, x = np.meshgrid(
        np.arange(shape_zyx[0]),
        np.arange(shape_zyx[1]),
        np.arange(shape_zyx[2]),
        indexing="ij",
    )
    test_image = (100.0 * z + 10.0 * y + 1.0 * x).astype(np.float32)

    save_nifti_image(test_image, ref_geom, file_path)

    loaded_img, loaded_spacing, loaded_origin, loaded_direction = load_nifti_image(file_path)

    assert loaded_img.dtype == np.float32
    assert loaded_img.shape == shape_zyx
    np.testing.assert_allclose(loaded_img, test_image, atol=1e-5)

    np.testing.assert_allclose(loaded_spacing, spacing_zyx, atol=1e-5)
    np.testing.assert_allclose(loaded_origin, origin_zyx, atol=1e-5)
    np.testing.assert_allclose(loaded_direction, direction, atol=1e-5)


def test_round_trip_mask(tmp_path: Path) -> None:
    """Test round-trip saving and loading of boolean segmentation masks."""
    mask_path = tmp_path / "test_mask.nii.gz"

    shape_zyx = (20, 25, 30)
    spacing_zyx = (2.0, 1.5, 0.8)
    origin_zyx = (5.0, 10.0, 15.0)
    direction = np.eye(3, dtype=np.float64)

    ref_geom = ImageGeometry(
        shape=shape_zyx,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction,
    )

    center_mm = (
        (shape_zyx[0] // 2) * spacing_zyx[0],
        (shape_zyx[1] // 2) * spacing_zyx[1],
        (shape_zyx[2] // 2) * spacing_zyx[2],
    )
    sphere_mask = make_sphere(shape_zyx, spacing_zyx, center_mm, radius_mm=8.0)

    save_nifti_mask(sphere_mask, ref_geom, mask_path)

    loaded_mask = load_nifti_mask(mask_path, image_meta=ref_geom, label_values=[1])
    assert isinstance(loaded_mask, np.ndarray)
    assert loaded_mask.dtype == bool
    assert loaded_mask.shape == shape_zyx
    assert np.array_equal(loaded_mask, sphere_mask)


def test_load_mask_with_none_label_values_returns_unique_labels(tmp_path: Path) -> None:
    """Test that loading a mask with label_values=None returns the set of unique labels."""
    mask_path = tmp_path / "multi_label.nii.gz"

    shape_zyx = (10, 10, 10)
    ref_geom = ImageGeometry(
        shape=shape_zyx,
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    mask_data = np.zeros(shape_zyx, dtype=np.uint8)
    mask_data[2:5, 2:5, 2:5] = 1
    mask_data[6:8, 6:8, 6:8] = 4

    save_nifti_mask(mask_data, ref_geom, mask_path)

    unique_labels = load_nifti_mask(mask_path, image_meta=ref_geom, label_values=None)
    assert isinstance(unique_labels, set)
    assert unique_labels == {0, 1, 4}


def test_load_mask_multiple_and_missing_label_values(tmp_path: Path) -> None:
    """Test loading multiple label values (union) and handling absent labels (all-False)."""
    mask_path = tmp_path / "multi_labels_test.nii.gz"

    shape_zyx = (12, 12, 12)
    ref_geom = ImageGeometry(
        shape=shape_zyx,
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    data = np.zeros(shape_zyx, dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1
    data[4:6, 4:6, 4:6] = 2
    data[7:9, 7:9, 7:9] = 4
    data[10:12, 10:12, 10:12] = 5

    save_nifti_mask(data, ref_geom, mask_path)

    # Union of labels [1, 2, 4]
    union_mask = load_nifti_mask(mask_path, image_meta=ref_geom, label_values=[1, 2, 4])
    assert isinstance(union_mask, np.ndarray)
    assert union_mask.dtype == bool
    assert union_mask.shape == shape_zyx
    expected_union = (data == 1) | (data == 2) | (data == 4)
    assert np.array_equal(union_mask, expected_union)

    # Querying label 3 which is absent returns an all-False boolean mask
    missing_mask = load_nifti_mask(mask_path, image_meta=ref_geom, label_values=[3])
    assert isinstance(missing_mask, np.ndarray)
    assert missing_mask.dtype == bool
    assert missing_mask.shape == shape_zyx
    assert not np.any(missing_mask)


def test_assert_same_geometry_raises_on_spacing_difference() -> None:
    """Test that assert_same_geometry raises GeometryMismatchError when spacing differs."""
    base_geom = ImageGeometry(
        shape=(20, 20, 20),
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    differing_spacing_geom = ImageGeometry(
        shape=(20, 20, 20),
        spacing_mm=(1.0, 1.1, 1.0),  # 0.1 mm difference along y-axis
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    with pytest.raises(GeometryMismatchError, match="Voxel spacing mismatch"):
        assert_same_geometry(base_geom, differing_spacing_geom, tol=1e-4)


def test_assert_same_geometry_raises_on_shape_and_origin_mismatch() -> None:
    """Test that assert_same_geometry detects shape, origin, and direction mismatches."""
    base_geom = ImageGeometry(
        shape=(20, 20, 20),
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    shape_mismatch = ImageGeometry(
        shape=(20, 21, 20),
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )
    with pytest.raises(GeometryMismatchError, match="Array shape mismatch"):
        assert_same_geometry(base_geom, shape_mismatch)

    origin_mismatch = ImageGeometry(
        shape=(20, 20, 20),
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.5, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )
    with pytest.raises(GeometryMismatchError, match="Physical origin mismatch"):
        assert_same_geometry(base_geom, origin_mismatch)


def test_voxel_to_lps_mm_matches_simpleitk() -> None:
    """Test that voxel_to_lps_mm equals SimpleITK's TransformContinuousIndexToPhysicalPoint.

    Evaluated on an oblique volume with anisotropic spacing and non-zero origin.
    """
    shape_zyx = (16, 24, 32)
    spacing_zyx = (3.5, 1.25, 0.75)
    origin_zyx = (15.5, -25.0, 42.1)

    theta = np.deg2rad(35.0)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    direction = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    geom = ImageGeometry(
        shape=shape_zyx,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction,
    )

    # Corresponding SimpleITK image with (x, y, z) metadata
    sitk_img = sitk.Image(shape_zyx[2], shape_zyx[1], shape_zyx[0], sitk.sitkFloat32)
    sitk_img.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
    sitk_img.SetOrigin((origin_zyx[2], origin_zyx[1], origin_zyx[0]))
    sitk_img.SetDirection(tuple(direction.flatten()))

    rng = np.random.default_rng(42)
    for _ in range(10):
        # Sample continuous float index (z, y, x)
        z = float(rng.uniform(0.0, shape_zyx[0] - 1.0))
        y = float(rng.uniform(0.0, shape_zyx[1] - 1.0))
        x = float(rng.uniform(0.0, shape_zyx[2] - 1.0))
        idx_zyx = (z, y, x)

        # SimpleITK expects continuous index in (x, y, z) order
        sitk_pt = sitk_img.TransformContinuousIndexToPhysicalPoint((x, y, z))

        hdt_pt = geom.voxel_to_lps_mm(idx_zyx)
        np.testing.assert_allclose(hdt_pt, sitk_pt, atol=1e-5)

        # Invert back to continuous voxel index
        recovered_idx_zyx = geom.lps_to_voxel(hdt_pt)
        np.testing.assert_allclose(recovered_idx_zyx, idx_zyx, atol=1e-5)


def test_independent_check_lps_to_ras_matches_nibabel_affine(tmp_path: Path) -> None:
    """Independent check: save with SimpleITK, load with nibabel, verify affine matches lps_to_ras.

    Nibabel applies an affine matrix from its voxel coordinates (x, y, z) to RAS mm.
    HDT computes: lps_to_ras(voxel_to_lps_mm(index_zyx)).
    """
    file_path = tmp_path / "affine_check.nii.gz"

    shape_zyx = (16, 24, 32)
    spacing_zyx = (3.5, 1.25, 0.75)
    origin_zyx = (15.5, -25.0, 42.1)

    theta = np.deg2rad(30.0)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    direction = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    geom = ImageGeometry(
        shape=shape_zyx,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction,
    )

    image_arr = np.zeros(shape_zyx, dtype=np.float32)
    save_nifti_image(image_arr, geom, file_path)

    nib_img = nib.load(str(file_path))
    affine = nib_img.affine

    rng = np.random.default_rng(123)
    for _ in range(10):
        z = float(rng.uniform(0.0, shape_zyx[0] - 1.0))
        y = float(rng.uniform(0.0, shape_zyx[1] - 1.0))
        x = float(rng.uniform(0.0, shape_zyx[2] - 1.0))
        idx_zyx = (z, y, x)

        # HDT LPS -> RAS calculation
        lps_pt = geom.voxel_to_lps_mm(idx_zyx)
        ras_pt = lps_to_ras(lps_pt)

        # Nibabel affine applied to (x, y, z) homogeneous voxel coordinate
        nib_voxel_homog = np.array([x, y, z, 1.0], dtype=np.float64)
        nib_ras_homog = affine @ nib_voxel_homog
        nib_ras = (float(nib_ras_homog[0]), float(nib_ras_homog[1]), float(nib_ras_homog[2]))

        np.testing.assert_allclose(ras_pt, nib_ras, atol=1e-4)


def test_disallowed_imports_in_src() -> None:
    """Verify SimpleITK is imported ONLY in src/hdt/imaging/ and nibabel is never in src/."""
    src_dir = Path(__file__).resolve().parent.parent / "src"

    for py_file in src_dir.rglob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(py_file))

        is_in_imaging = "imaging" in py_file.parts

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in ("nibabel", "nib") or name.startswith("nibabel."):
                        pytest.fail(f"Disallowed import of nibabel in {py_file}")
                    if (name in ("SimpleITK", "sitk") or name.startswith("SimpleITK.")) and (
                        not is_in_imaging
                    ):
                        pytest.fail(f"Disallowed import of SimpleITK outside imaging: {py_file}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "nibabel" or mod.startswith("nibabel."):
                    pytest.fail(f"Disallowed from-import of nibabel in {py_file}")
                if (mod == "SimpleITK" or mod.startswith("SimpleITK.")) and not is_in_imaging:
                    pytest.fail(f"Disallowed from-import of SimpleITK outside imaging: {py_file}")
