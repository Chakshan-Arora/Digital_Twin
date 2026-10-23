"""Tests for Brain MRI domain: adapter, label maps, prepare_case, and dataset_summary."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from hdt.domains.brain_mri.adapter import load_case
from hdt.imaging.load_scan import (
    ImageGeometry,
    assert_same_geometry,
    load_nifti_image,
    load_nifti_mask_array,
    lps_to_ras,
    save_nifti_image,
    save_nifti_mask,
)


def _create_synthetic_case(
    case_dir: Path,
    case_id: str,
    img_dtype: np.dtype = np.int16,
    shape_zyx: tuple[int, int, int] = (16, 24, 24),
    spacing_zyx: tuple[float, float, float] = (1.5, 1.0156, 1.0156),
    mask_vals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, ImageGeometry]:
    case_dir.mkdir(parents=True, exist_ok=True)

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
    origin_zyx = (10.0, -20.0, 30.0)

    geom = ImageGeometry(
        shape=shape_zyx,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction,
    )

    # Synthetic image
    if np.issubdtype(img_dtype, np.integer):
        img_arr = np.arange(np.prod(shape_zyx), dtype=img_dtype).reshape(shape_zyx)
    else:
        img_arr = np.linspace(10.5, 99.5, np.prod(shape_zyx), dtype=img_dtype).reshape(shape_zyx)

    # Synthetic mask: small central sphere if not provided
    if mask_vals is None:
        mask_arr = np.zeros(shape_zyx, dtype=np.int16)
        mask_arr[6:10, 10:14, 10:14] = 1
        save_nifti_mask(mask_arr, geom, case_dir / f"{case_id}_gtv.nii.gz")
    else:
        mask_arr = mask_vals
        if np.issubdtype(mask_arr.dtype, np.floating):
            save_nifti_image(mask_arr.astype(np.float32), geom, case_dir / f"{case_id}_gtv.nii.gz")
        else:
            save_nifti_mask(mask_arr, geom, case_dir / f"{case_id}_gtv.nii.gz")

    save_nifti_image(img_arr.astype(np.float32), geom, case_dir / f"{case_id}_t1c.nii.gz")

    return img_arr, mask_arr, geom


def test_brats_folder_loading_int16_and_float32(tmp_path: Path) -> None:
    """Test brats_folder case loading with int16 and float32 dtypes and oblique direction."""
    case1_dir = tmp_path / "case_int16"
    img1, mask1, geom1 = _create_synthetic_case(
        case1_dir, "case_int16", img_dtype=np.int16, spacing_zyx=(1.5, 1.0156, 1.0156)
    )

    loaded_case1 = load_case(tmp_path, "case_int16")
    assert loaded_case1.case_id == "case_int16"
    assert "t1c" in loaded_case1.images
    assert loaded_case1.images["t1c"].dtype == np.float32

    # Verify no intensity rescaling occurred
    np.testing.assert_array_equal(loaded_case1.images["t1c"], img1.astype(np.float32))
    np.testing.assert_allclose(loaded_case1.spacing_mm, (1.5, 1.0156, 1.0156), atol=1e-4)
    assert "gtv" in loaded_case1.structures
    assert loaded_case1.structures["gtv"].role == "target"
    np.testing.assert_array_equal(loaded_case1.structures["gtv"].mask, mask1 == 1)

    # Float32 image
    case2_dir = tmp_path / "case_float32"
    img2, _, _ = _create_synthetic_case(
        case2_dir, "case_float32", img_dtype=np.float32, spacing_zyx=(1.5, 1.0156, 1.0156)
    )
    loaded_case2 = load_case(tmp_path, "case_float32")
    assert loaded_case2.images["t1c"].dtype == np.float32
    np.testing.assert_array_equal(loaded_case2.images["t1c"], img2)


def test_non_integer_float_mask_raises(tmp_path: Path) -> None:
    """Float mask with fractional values must raise ValueError."""
    case_dir = tmp_path / "case_fractional"
    float_mask = np.zeros((16, 24, 24), dtype=np.float32)
    float_mask[6:10, 10:14, 10:14] = 0.5  # Fractional value
    _create_synthetic_case(case_dir, "case_fractional", mask_vals=float_mask)

    with pytest.raises(ValueError, match="Float mask values must be exactly integer-valued"):
        load_case(tmp_path, "case_fractional")


def test_missing_and_duplicate_files_raise(tmp_path: Path) -> None:
    """Missing or duplicate t1c or gtv files must raise clear errors."""
    # Missing gtv
    case_no_gtv = tmp_path / "case_no_gtv"
    _create_synthetic_case(case_no_gtv, "case_no_gtv")
    (case_no_gtv / "case_no_gtv_gtv.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match="Missing segmentation file"):
        load_case(tmp_path, "case_no_gtv")

    # Missing t1c
    case_no_t1c = tmp_path / "case_no_t1c"
    _create_synthetic_case(case_no_t1c, "case_no_t1c")
    (case_no_t1c / "case_no_t1c_t1c.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match="Missing modality file"):
        load_case(tmp_path, "case_no_t1c")

    # Duplicate t1c
    case_dup_t1c = tmp_path / "case_dup_t1c"
    _create_synthetic_case(case_dup_t1c, "case_dup_t1c")
    dup_file = case_dup_t1c / "extra_t1c.nii.gz"
    dup_file.write_bytes((case_dup_t1c / "case_dup_t1c_t1c.nii.gz").read_bytes())
    with pytest.raises(ValueError, match="Duplicate modality files"):
        load_case(tmp_path, "case_dup_t1c")

    # Duplicate gtv
    case_dup_gtv = tmp_path / "case_dup_gtv"
    _create_synthetic_case(case_dup_gtv, "case_dup_gtv")
    dup_gtv_file = case_dup_gtv / "extra_gtv.nii.gz"
    dup_gtv_file.write_bytes((case_dup_gtv / "case_dup_gtv_gtv.nii.gz").read_bytes())
    with pytest.raises(ValueError, match="Duplicate segmentation files"):
        load_case(tmp_path, "case_dup_gtv")


def test_empty_mask_raises(tmp_path: Path) -> None:
    """Empty GTV mask must raise ValueError."""
    case_dir = tmp_path / "case_empty"
    empty_mask = np.zeros((16, 24, 24), dtype=np.int16)
    _create_synthetic_case(case_dir, "case_empty", mask_vals=empty_mask)

    with pytest.raises(ValueError, match="Empty GTV mask detected"):
        load_case(tmp_path, "case_empty")


def test_invalid_label_value_raises(tmp_path: Path) -> None:
    """Mask containing labels outside label map keys plus 0 must raise."""
    case_dir = tmp_path / "case_invalid_label"
    bad_mask = np.zeros((16, 24, 24), dtype=np.int16)
    bad_mask[5, 5, 5] = 2  # Not in BRATS_MEN_RT ({1: 'gtv'})
    _create_synthetic_case(case_dir, "case_invalid_label", mask_vals=bad_mask)

    with pytest.raises(ValueError, match="Observed labels .* are not a subset of allowed labels"):
        load_case(tmp_path, "case_invalid_label")


def test_prepare_case_round_trip(tmp_path: Path) -> None:
    """prepare_case must output NIfTI files and case.json matching loader output exactly."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.prepare_case import prepare_case

    raw_dir = tmp_path / "raw"
    case_dir = raw_dir / "case_prep"
    img, mask, geom = _create_synthetic_case(case_dir, "case_prep")

    out_dir = tmp_path / "processed" / "case_prep"
    res_path = prepare_case(root=raw_dir, case_id="case_prep", out=out_dir)
    assert res_path == out_dir

    t1c_path = out_dir / "t1c.nii.gz"
    gtv_path = out_dir / "structures" / "gtv.nii.gz"
    json_path = out_dir / "case.json"

    assert t1c_path.is_file()
    assert gtv_path.is_file()
    assert json_path.is_file()

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["case_id"] == "case_prep"
    assert meta["label_map"] == "BRATS_MEN_RT"
    assert meta["roles"] == {"gtv": "target"}
    assert meta["source"] == "BraTS2024-MEN-RT"

    # Reload image and mask and assert geometry and arrays match loader output
    loaded_img, sp, orig, dir_m = load_nifti_image(t1c_path)
    assert_same_geometry(geom, (loaded_img.shape, sp, orig, dir_m))
    np.testing.assert_array_equal(loaded_img, img.astype(np.float32))

    loaded_mask, mask_geom, _ = load_nifti_mask_array(gtv_path)
    assert_same_geometry(geom, mask_geom)
    np.testing.assert_array_equal(loaded_mask, mask)


def test_summary_script_on_synthetic_cases(tmp_path: Path) -> None:
    """Summary script must compute exact analytical volumes and flags across synthetic cases."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.dataset_summary import analyze_case, generate_summary

    raw_dir = tmp_path / "raw_dataset"

    # Case 1: 4x4x4 cube = 64 voxels. Spacing (1.5, 1.0, 1.0) -> volume = 64 * 1.5 = 96.0 mm3.
    # Single component, not touching border.
    c1_dir = raw_dir / "c1"
    m1 = np.zeros((16, 16, 16), dtype=np.int16)
    m1[4:8, 4:8, 4:8] = 1
    _create_synthetic_case(
        c1_dir, "c1", shape_zyx=(16, 16, 16), spacing_zyx=(1.5, 1.0, 1.0), mask_vals=m1
    )

    # Case 2: 2 disjoint voxels touching border (multi-component, touches border)
    c2_dir = raw_dir / "c2"
    m2 = np.zeros((16, 16, 16), dtype=np.int16)
    m2[0, 0, 0] = 1  # Touches border
    m2[10, 10, 10] = 1
    _create_synthetic_case(
        c2_dir, "c2", shape_zyx=(16, 16, 16), spacing_zyx=(1.5, 1.0, 1.0), mask_vals=m2
    )

    # Case 3: Exactly identical volume to Case 1 (within 0.1%)
    c3_dir = raw_dir / "c3"
    m3 = np.zeros((16, 16, 16), dtype=np.int16)
    m3[2:6, 2:6, 2:6] = 1
    _create_synthetic_case(
        c3_dir, "c3", shape_zyx=(16, 16, 16), spacing_zyx=(1.5, 1.0, 1.0), mask_vals=m3
    )

    summary_csv = tmp_path / "summary.csv"
    generate_summary(raw_dir, summary_csv)
    assert summary_csv.is_file()

    # Check analyze_case directly
    res1 = analyze_case(raw_dir, "c1")
    assert res1["gtv_voxels"] == 64
    assert res1["gtv_volume_mm3"] == 96.0
    assert res1["gtv_connected_components"] == 1
    assert res1["touches_array_border"] is False

    res2 = analyze_case(raw_dir, "c2")
    assert res2["gtv_voxels"] == 2
    assert res2["gtv_connected_components"] == 2
    assert res2["touches_array_border"] is True


def test_independent_nibabel_affine_check(tmp_path: Path) -> None:
    """Independent check: nibabel affine must match HDT voxel-to-RAS calculation."""
    case_dir = tmp_path / "case_affine"
    spacing_zyx = (1.5, 1.0156, 1.0156)
    _, _, geom = _create_synthetic_case(case_dir, "case_affine", spacing_zyx=spacing_zyx)

    t1c_file = case_dir / "case_affine_t1c.nii.gz"
    nib_img = nib.load(str(t1c_file))
    affine = nib_img.affine

    rng = np.random.default_rng(42)
    for _ in range(10):
        z = float(rng.uniform(0.0, geom.shape[0] - 1.0))
        y = float(rng.uniform(0.0, geom.shape[1] - 1.0))
        x = float(rng.uniform(0.0, geom.shape[2] - 1.0))

        # HDT calculation: continuous voxel (z, y, x) -> LPS -> RAS
        lps_pt = geom.voxel_to_lps_mm((z, y, x))
        ras_pt = lps_to_ras(lps_pt)

        # Nibabel affine applied to (x, y, z, 1)
        nib_homog = affine @ np.array([x, y, z, 1.0], dtype=np.float64)
        nib_ras = (float(nib_homog[0]), float(nib_homog[1]), float(nib_homog[2]))

        np.testing.assert_allclose(ras_pt, nib_ras, atol=1e-4)
