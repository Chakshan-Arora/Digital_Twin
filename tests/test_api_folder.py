"""Tests for FolderRepository serving real prepared cases and updated API contracts."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient

from hdt.api.app import app


def _assert_no_paths_in_obj(obj: Any, root_str: str) -> None:
    """Recursively ensure no absolute or filesystem path strings leak into response objects."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert root_str.lower() not in str(k).lower(), f"Path leaked in key: {k}"
            _assert_no_paths_in_obj(v, root_str)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_paths_in_obj(item, root_str)
    elif isinstance(obj, str):
        assert root_str.lower() not in obj.lower(), f"Path leaked in value: {obj}"
        assert ":\\" not in obj, f"Windows drive path leaked in value: {obj}"
        assert not (obj.startswith("/") and len(obj) > 1 and "/" in obj[1:]), (
            f"Unix path leaked in value: {obj}"
        )


@pytest.fixture
def folder_cases_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create synthetic folder repository under tmp_path with oblique geometry and 2 components."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main test case: case_synth_01 with 2 components in GTV and an OAR risk structure
    case_dir = cases_dir / "case_synth_01"
    structs_dir = case_dir / "structures"
    structs_dir.mkdir(parents=True, exist_ok=True)

    # Oblique direction matrix: 15-degree rotation around Y axis in LPS
    theta = float(np.deg2rad(15.0))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    direction = (
        cos_t, 0.0, sin_t,
        0.0, 1.0, 0.0,
        -sin_t, 0.0, cos_t,
    )

    # Spacing (z, y, x) = (1.5, 1.0156, 1.0156) -> SimpleITK spacing is (sx, sy, sz)
    sitk_spacing = (1.0156, 1.0156, 1.5)
    sitk_origin = (10.0, -20.0, 5.0)
    shape_zyx = (32, 32, 32)
    sitk_size = (32, 32, 32)

    # t1c image
    img = sitk.Image(sitk_size, sitk.sitkFloat32)
    img.SetSpacing(sitk_spacing)
    img.SetOrigin(sitk_origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, str(case_dir / "t1c.nii.gz"))

    # gtv mask with 2 separate connected components (e.g. two 3x3x3 cubes separated by gap)
    gtv_arr = np.zeros(shape_zyx, dtype=np.uint8)
    gtv_arr[5:8, 5:8, 5:8] = 1       # Component 1: 27 voxels
    gtv_arr[15:18, 15:18, 15:18] = 1 # Component 2: 27 voxels
    # Total voxel count = 54 voxels

    gtv_img = sitk.GetImageFromArray(gtv_arr)
    gtv_img.SetSpacing(sitk_spacing)
    gtv_img.SetOrigin(sitk_origin)
    gtv_img.SetDirection(direction)
    sitk.WriteImage(gtv_img, str(structs_dir / "gtv.nii.gz"))

    # risk_oar mask: 1 component at distance
    risk_arr = np.zeros(shape_zyx, dtype=np.uint8)
    risk_arr[25:28, 25:28, 25:28] = 1
    risk_img = sitk.GetImageFromArray(risk_arr)
    risk_img.SetSpacing(sitk_spacing)
    risk_img.SetOrigin(sitk_origin)
    risk_img.SetDirection(direction)
    sitk.WriteImage(risk_img, str(structs_dir / "risk_oar.nii.gz"))

    # case.json
    case_meta = {
        "case_id": "case_synth_01",
        "spacing_mm_zyx": [1.5, 1.0156, 1.0156],
        "shape_zyx": list(shape_zyx),
        "roles": {"gtv": "target", "risk_oar": "risk"},
        "label_map": "BRATS_MEN_RT",
        "source": "BraTS2024-MEN-RT",
        "original_dtype": "int16",
    }
    with (case_dir / "case.json").open("w", encoding="utf-8") as f:
        json.dump(case_meta, f)

    # 2. Single-structure case: case_single_01 (only gtv, no risk structure)
    single_dir = cases_dir / "case_single_01"
    single_structs = single_dir / "structures"
    single_structs.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(img, str(single_dir / "t1c.nii.gz"))
    sitk.WriteImage(gtv_img, str(single_structs / "gtv.nii.gz"))
    single_meta = {
        "case_id": "case_single_01",
        "spacing_mm_zyx": [1.5, 1.0156, 1.0156],
        "shape_zyx": list(shape_zyx),
        "roles": {"gtv": "target"},
        "label_map": "BRATS_MEN_RT",
        "source": "BraTS2024-MEN-RT",
        "original_dtype": "int16",
    }
    with (single_dir / "case.json").open("w", encoding="utf-8") as f:
        json.dump(single_meta, f)

    monkeypatch.setenv("HDT_DATA_DIR", str(tmp_path))
    return tmp_path


def test_path_traversal_ids_return_404(folder_cases_setup: Path) -> None:
    """Ids containing '..', '/', or '\\' must return 404 without querying filesystem paths."""
    client = TestClient(app)

    malicious_ids = [
        "..",
        "%2E%2E",
        "../case_synth_01",
        "..\\case_synth_01",
        "foo/bar",
        "foo\\bar",
        "case..id",
    ]
    for cid in malicious_ids:
        resp = client.get(f"/cases/{cid}")
        assert resp.status_code == 404, f"Expected 404 for id '{cid}', got {resp.status_code}"

        resp_stats = client.get(f"/cases/{cid}/structures/gtv/stats")
        assert resp_stats.status_code == 404

        resp_mesh = client.get(f"/cases/{cid}/structures/gtv/mesh.obj")
        assert resp_mesh.status_code == 404


def test_structure_stats(folder_cases_setup: Path) -> None:
    """Validate GET /structures/{name}/stats: volume, voxel count, 6-connectivity n_components."""
    client = TestClient(app)
    resp = client.get("/cases/case_synth_01/structures/gtv/stats")
    assert resp.status_code == 200
    data = resp.json()

    voxel_vol = 1.5 * 1.0156 * 1.0156
    expected_count = 54
    expected_vol = expected_count * voxel_vol

    # 1. stats volume equals voxel_count times voxel volume
    assert data["voxel_count"] == expected_count
    assert math.isclose(data["volume_mm3"], expected_vol, rel_tol=1e-5)

    # 2. n_components is 2 (6-connectivity)
    assert data["n_components"] == 2
    assert len(data["component_volumes_mm3"]) == 2
    for comp_vol in data["component_volumes_mm3"]:
        assert math.isclose(comp_vol, 27 * voxel_vol, rel_tol=1e-5)

    # 3. Bounding box and centroid present in RAS
    assert data["coordinate_system"] == "RAS"
    assert "min" in data["bbox_ras_mm"] and "max" in data["bbox_ras_mm"]
    assert len(data["centroid_ras_mm"]) == 3

    _assert_no_paths_in_obj(data, str(folder_cases_setup))


def test_mesh_cap_and_step_size_headers(folder_cases_setup: Path) -> None:
    """Mesh triangles must be under max_triangles cap and X-Mesh-Step-Size header present."""
    client = TestClient(app)

    # Request small cap to force decimation (step_size > 1)
    resp = client.get("/cases/case_synth_01/structures/gtv/mesh.obj?max_triangles=60")
    assert resp.status_code == 200
    assert "X-Mesh-Step-Size" in resp.headers
    assert "X-Mesh-Triangles" in resp.headers

    step_size = int(resp.headers["X-Mesh-Step-Size"])
    triangles = int(resp.headers["X-Mesh-Triangles"])

    assert triangles <= 60
    assert step_size >= 1

    # Text verification
    text = resp.text
    assert text.startswith("# HDT Generated OBJ Mesh")
    assert "f " in text


def test_single_structure_case_returns_422(folder_cases_setup: Path) -> None:
    """Clearance with fewer than two structures in the case returns 422 and a clear message."""
    client = TestClient(app)
    payload = {"target_name": "gtv", "risk_name": "gtv"}
    resp = client.post("/cases/case_single_01/clearance", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "at least two" in str(data).lower() or "requires at least 2" in str(data).lower()
    _assert_no_paths_in_obj(data, str(folder_cases_setup))


def test_response_renames_and_removals(folder_cases_setup: Path) -> None:
    """Verify renamed fields exist and removed fields do not in responses."""
    client = TestClient(app)

    # 1. /clearance
    payload = {"target_name": "gtv", "risk_name": "risk_oar"}
    resp = client.post("/cases/case_synth_01/clearance", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # clearance_mm exists as headline number
    assert "clearance_mm" in data
    assert isinstance(data["clearance_mm"], float)

    # Removed fields: signed_clearance_mm must NOT be in response
    assert "signed_clearance_mm" not in data, "signed_clearance_mm must be removed"

    # Closest points renames & removals
    closest = data["closest_points_mm"]
    assert "centre_distance_mm" in closest
    assert "distance_mm" not in closest, "closest_points_mm.distance_mm must be renamed"
    assert "point_a_ras_mm" not in closest, "point_a_ras_mm must be removed"
    assert "point_b_ras_mm" not in closest, "point_b_ras_mm must be removed"

    # Surface distance stats renames & removals
    s_stats = data["surface_distance_stats"]
    assert "centre_min_mm" in s_stats
    assert "min" not in s_stats, "surface_distance_stats.min must be renamed"

    # Resolution block
    assert "resolution" in data
    res = data["resolution"]
    assert "spacing_mm" in res
    assert "voxel_diagonal_mm" in res
    assert "voxel_extent_along_pair_mm" in res

    _assert_no_paths_in_obj(data, str(folder_cases_setup))

    # 2. /uncertainty
    unc_payload = {
        "target_name": "gtv",
        "risk_name": "risk_oar",
        "sigma_mm": 1.0,
        "corr_length_mm": 5.0,
        "n_samples": 5,
        "seed": 42,
    }
    resp_unc = client.post("/cases/case_synth_01/uncertainty", json=unc_payload)
    assert resp_unc.status_code == 200
    unc_data = resp_unc.json()

    # Resolution block must be present in uncertainty response
    assert "resolution" in unc_data
    unc_res = unc_data["resolution"]
    assert "spacing_mm" in unc_res
    assert "voxel_diagonal_mm" in unc_res
    assert "voxel_extent_along_pair_mm" in unc_res

    _assert_no_paths_in_obj(unc_data, str(folder_cases_setup))
