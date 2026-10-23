"""Comprehensive NIfTI spatial geometry and affine audit regression tests.

Verifies:
1. NIfTI sform and qform codes, values, and consistency.
2. Exact mathematical agreement between backend coords.py (voxel_to_ras_mm)
   and NIfTI header affine multiplication across known voxel points.
3. 1-voxel physical displacement test: physical distance traveled by incrementing
   one voxel along each axis matches the declared voxel spacing (sz, sy, sx)
   regardless of oblique rotation angles.
4. Structure alignment: Volume, tumor mask, and vessel mask share identical spatial
   metadata (shape, spacing, origin, direction, sform, and qform).
5. 3D surface mesh correspondence: marching cubes mesh vertices in physical RAS
   are centered at the expected structure coordinates and agree with mask voxels
   within half-voxel isosurface discretization bounds.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import skimage.measure
from fastapi.testclient import TestClient

from hdt.api.app import app
from hdt.api.coords import (
    transform_mesh_vertices_to_ras,
    voxel_to_ras_mm,
)
from hdt.api.repository import PhantomRepository
from hdt.imaging.load_scan import save_nifti_image, save_nifti_mask

client = TestClient(app)


def test_nifti_sform_qform_consistency_and_codes(tmp_path: Path) -> None:
    """Verify sform and qform codes, consistency, and alignment across volume and masks."""
    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")  # type: ignore[misc]

    vol_path = tmp_path / "vol.nii.gz"
    tumor_path = tmp_path / "tumor.nii.gz"
    vessel_path = tmp_path / "vessel.nii.gz"

    save_nifti_image(case.images[case.primary], geom, vol_path)
    save_nifti_mask(case.structures["tumor"].mask, geom, tumor_path)
    save_nifti_mask(case.structures["vessel"].mask, geom, vessel_path)

    nii_vol = nib.load(str(vol_path))
    nii_tumor = nib.load(str(tumor_path))
    nii_vessel = nib.load(str(vessel_path))

    # 1. Check sform and qform codes are set to 1 (NIFTI_XFORM_SCANNER_ANAT)
    s_code = nii_vol.header.get_sform(coded=True)[1]
    q_code = nii_vol.header.get_qform(coded=True)[1]
    assert s_code == 1, f"Expected sform_code=1 (scanner anat), got {s_code}"
    assert q_code == 1, f"Expected qform_code=1 (scanner anat), got {q_code}"

    # 2. Check sform and qform match within numerical tolerance (no precedence ambiguity)
    sform = nii_vol.get_sform()
    qform = nii_vol.get_qform()
    np.testing.assert_allclose(sform, qform, atol=1e-5)

    # 3. Check masks have identical spatial metadata to volume
    assert nii_tumor.shape == nii_vol.shape
    assert nii_vessel.shape == nii_vol.shape
    np.testing.assert_allclose(nii_tumor.affine, nii_vol.affine, atol=1e-6)
    np.testing.assert_allclose(nii_vessel.affine, nii_vol.affine, atol=1e-6)


def test_voxel_to_ras_mathematical_equivalence(tmp_path: Path) -> None:
    """Verify that backend coords.py matches NIfTI affine multiplication exactly."""
    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")  # type: ignore[misc]

    vol_path = tmp_path / "vol.nii.gz"
    save_nifti_image(case.images[case.primary], geom, vol_path)
    nii_vol = nib.load(str(vol_path))
    affine = nii_vol.affine

    test_points_zyx = [
        (0.0, 0.0, 0.0),
        (63.0, 63.0, 63.0),
        (32.0, 32.0, 32.0),
        (10.5, 20.3, 45.7),
        (0.0, 32.0, 63.0),
        (63.0, 0.0, 32.0),
    ]

    for zyx in test_points_zyx:
        z, y, x = zyx
        # Backend RAS conversion: (z, y, x) -> LPS -> RAS
        ras_backend = np.array(voxel_to_ras_mm(geom, zyx), dtype=np.float64)

        # NIfTI affine applied to homogeneous (x, y, z, 1.0)
        nifti_homog = np.array([x, y, z, 1.0], dtype=np.float64)
        ras_nifti = (affine @ nifti_homog)[:3]

        # Difference must be strictly within floating point precision (< 1e-5 mm)
        np.testing.assert_allclose(ras_backend, ras_nifti, atol=1e-5)

        # Invert RAS coordinate back to voxel index and assert round-trip recovery
        pt_lps = (-ras_backend[0], -ras_backend[1], ras_backend[2])
        recovered_zyx = geom.lps_to_voxel(pt_lps)
        np.testing.assert_allclose(recovered_zyx, zyx, atol=1e-5)


def test_physical_one_voxel_displacement_agrees_with_spacing() -> None:
    """Verify that a 1-voxel step along each axis produces the declared physical spacing.

    Displacement along z must be sz (2.0 mm).
    Displacement along y must be sy (1.0 mm).
    Displacement along x must be sx (1.0 mm).
    """
    repo = PhantomRepository()
    _, geom = repo.get_case("phantom_001")  # type: ignore[misc]

    p0 = np.array(voxel_to_ras_mm(geom, (0.0, 0.0, 0.0)))
    pz = np.array(voxel_to_ras_mm(geom, (1.0, 0.0, 0.0)))
    py = np.array(voxel_to_ras_mm(geom, (0.0, 1.0, 0.0)))
    px = np.array(voxel_to_ras_mm(geom, (0.0, 0.0, 1.0)))

    disp_z = float(np.linalg.norm(pz - p0))
    disp_y = float(np.linalg.norm(py - p0))
    disp_x = float(np.linalg.norm(px - p0))

    assert abs(disp_z - geom.spacing_mm[0]) < 1e-6, f"Expected {geom.spacing_mm[0]}, got {disp_z}"
    assert abs(disp_y - geom.spacing_mm[1]) < 1e-6, f"Expected {geom.spacing_mm[1]}, got {disp_y}"
    assert abs(disp_x - geom.spacing_mm[2]) < 1e-6, f"Expected {geom.spacing_mm[2]}, got {disp_x}"


def test_structure_alignment_and_mesh_correspondence() -> None:
    """Verify that structure segmentation voxels and 3D OBJ mesh align in RAS physical space."""
    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")  # type: ignore[misc]

    tumor_mask = case.structures["tumor"].mask
    tumor_indices_zyx = np.argwhere(tumor_mask)
    assert len(tumor_indices_zyx) > 0

    # 1. Expected tumor center in local mm is (64, 32, 24) -> voxel index (32, 32, 24)
    expected_center_zyx = (32.0, 32.0, 24.0)
    expected_center_ras = voxel_to_ras_mm(geom, expected_center_zyx)

    tumor_com_zyx = np.mean(tumor_indices_zyx, axis=0)
    tumor_com_ras = voxel_to_ras_mm(geom, tumor_com_zyx)
    np.testing.assert_allclose(tumor_com_zyx, expected_center_zyx, atol=1e-5)
    np.testing.assert_allclose(tumor_com_ras, expected_center_ras, atol=1e-5)

    # 2. Check marching cubes mesh vertices in physical RAS space
    padded_tumor = np.pad(tumor_mask, pad_width=1, mode="constant", constant_values=False)
    verts_padded, faces, _, _ = skimage.measure.marching_cubes(
        padded_tumor.astype(np.float32),
        level=0.5,
        spacing=geom.spacing_mm,
    )
    voxel_indices_zyx = (verts_padded / np.array(geom.spacing_mm)) - 1.0
    mesh_verts_ras = transform_mesh_vertices_to_ras(geom, voxel_indices_zyx)

    mesh_center_ras = np.mean(mesh_verts_ras, axis=0)
    np.testing.assert_allclose(mesh_center_ras, expected_center_ras, atol=1e-4)

    # 3. Check that the voxel bounding box is enclosed by the marching cubes mesh bounding box
    tumor_voxels_ras = np.array([voxel_to_ras_mm(geom, idx) for idx in tumor_indices_zyx])
    mask_min = np.min(tumor_voxels_ras, axis=0)
    mask_max = np.max(tumor_voxels_ras, axis=0)
    mesh_min = np.min(mesh_verts_ras, axis=0)
    mesh_max = np.max(mesh_verts_ras, axis=0)

    # Mesh bounding box should be within 1 voxel of mask bounding box (due to 0.5-level isosurface)
    max_spacing = max(geom.spacing_mm)
    assert np.all(mesh_min <= mask_min + 1e-4)
    assert np.all(mesh_max >= mask_max - 1e-4)
    assert np.all(abs(mesh_min - mask_min) <= max_spacing)
    assert np.all(abs(mesh_max - mask_max) <= max_spacing)


def test_closest_points_clearance_synchronization_in_ras() -> None:
    """Verify that closest points returned by API correspond to actual surface boundary voxels."""
    payload = {"target_name": "tumor", "risk_name": "vessel"}
    response = client.post("/cases/phantom_001/clearance", json=payload)
    assert response.status_code == 200
    data = response.json()

    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")  # type: ignore[misc]

    closest = data["closest_points_mm"]
    pt_a_ras = closest["point_target_ras_mm"]
    pt_b_ras = closest["point_risk_ras_mm"]

    # Convert RAS points back to voxel coordinates
    pt_a_lps = (-pt_a_ras[0], -pt_a_ras[1], pt_a_ras[2])
    vox_a = geom.lps_to_voxel(pt_a_lps)
    idx_a = tuple(int(round(c)) for c in vox_a)
    assert bool(case.structures["tumor"].mask[idx_a]) is True

    pt_b_lps = (-pt_b_ras[0], -pt_b_ras[1], pt_b_ras[2])
    vox_b = geom.lps_to_voxel(pt_b_lps)
    idx_b = tuple(int(round(c)) for c in vox_b)
    assert bool(case.structures["vessel"].mask[idx_b]) is True
