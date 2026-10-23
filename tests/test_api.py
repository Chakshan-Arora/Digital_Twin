"""Integration and acceptance tests for the HDT FastAPI server."""

import math
import re
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from hdt.api.app import app
from hdt.api.coords import local_point_zyx_mm_to_ras_mm, mesh_signed_volume_ras
from hdt.api.repository import PhantomRepository
from hdt.geometry.distances import signed_clearance_mm

client = TestClient(app)


def assert_no_paths_in_json(obj: Any) -> None:
    """Recursively verify that no string in the JSON payload matches a filesystem path."""
    # Matches Windows drive paths (e.g., C:\...) or POSIX root paths (e.g., /home/..., /tmp/...)
    path_pattern = re.compile(r"^[a-zA-Z]:[\\/]|^(?:/|\\)[a-zA-Z0-9_\.\-]+[\\/]")
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_paths_in_json(k)
            assert_no_paths_in_json(v)
    elif isinstance(obj, list):
        for item in obj:
            assert_no_paths_in_json(item)
    elif isinstance(obj, str):
        msg = f"Leaked filesystem path found in JSON response: '{obj}'"
        assert not path_pattern.search(obj), msg


def test_health_check() -> None:
    """Acceptance test 1: GET /health -> 200 {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}
    assert_no_paths_in_json(data)


def test_list_cases() -> None:
    """Acceptance test 2: GET /cases -> list containing the phantom case."""
    response = client.get("/cases")
    assert response.status_code == 200
    cases = response.json()
    assert isinstance(cases, list)
    assert len(cases) == 1
    phantom = cases[0]
    assert phantom["id"] == "phantom_001"
    struct_names = {s["name"] for s in phantom["structures"]}
    assert struct_names == {"tumor", "vessel"}
    assert_no_paths_in_json(cases)


def test_get_case_detail() -> None:
    """Test GET /cases/{id} spatial metadata in RAS mm."""
    response = client.get("/cases/phantom_001")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "phantom_001"
    assert data["coordinate_system"] == "RAS"
    assert len(data["origin_ras_mm"]) == 3
    assert len(data["direction_ras"]) == 3
    assert len(data["spacing_mm"]) == 3
    assert len(data["shape"]) == 3
    assert data["primary_image_name"] == "t1"
    assert_no_paths_in_json(data)


def test_stream_volume_and_mask() -> None:
    """Test NIfTI downloads for volume and mask."""
    vol_resp = client.get("/cases/phantom_001/volume.nii.gz")
    assert vol_resp.status_code == 200
    assert vol_resp.headers["content-type"] == "application/gzip"
    assert len(vol_resp.content) > 100

    mask_resp = client.get("/cases/phantom_001/structures/tumor/mask.nii.gz")
    assert mask_resp.status_code == 200
    assert mask_resp.headers["content-type"] == "application/gzip"
    assert len(mask_resp.content) > 100


def test_clearance_calculation() -> None:
    """Acceptance test 3: POST /clearance matches signed_clearance_mm on structures."""
    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")  # type: ignore[misc]
    direct_clearance = signed_clearance_mm(
        case.structures["tumor"].mask,
        case.structures["vessel"].mask,
        geom.spacing_mm,
    )

    payload = {"target_name": "tumor", "risk_name": "vessel"}
    response = client.post("/cases/phantom_001/clearance", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["coordinate_system"] == "RAS"
    assert data["convention"] == "face_to_face_voxel_sets"
    assert math.isclose(data["signed_clearance_mm"], direct_clearance, abs_tol=1e-5)
    assert math.isclose(data["clearance_mm"], direct_clearance, abs_tol=1e-5)
    assert_no_paths_in_json(data)


def test_closest_points_in_ras() -> None:
    """Acceptance test 4: Closest points in RAS lie within one voxel of surfaces."""
    repo = PhantomRepository()
    case, geom = repo.get_case("phantom_001")  # type: ignore[misc]

    payload = {"target": "tumor", "risk": "vessel"}
    response = client.post("/cases/phantom_001/clearance", json=payload)
    assert response.status_code == 200
    data = response.json()

    closest = data["closest_points_mm"]
    assert closest["coordinate_system"] == "RAS"

    pt_target_ras = np.array(closest["point_target_ras_mm"], dtype=np.float64)
    pt_risk_ras = np.array(closest["point_risk_ras_mm"], dtype=np.float64)

    # 1. Distance between closest points matches clearance within one voxel diagonal
    pt_distance = float(np.linalg.norm(pt_target_ras - pt_risk_ras))
    voxel_diagonal = float(np.linalg.norm(geom.spacing_mm))
    clearance = float(data["signed_clearance_mm"])
    assert abs(pt_distance - clearance) <= voxel_diagonal

    # 2. Convert points back to voxel indices and verify they are on/near structure masks
    # Using ImageGeometry: RAS -> LPS -> voxel
    pt_target_lps = (-pt_target_ras[0], -pt_target_ras[1], pt_target_ras[2])
    vox_target = geom.lps_to_voxel(pt_target_lps)
    idx_target = tuple(int(round(x)) for x in vox_target)
    assert case.structures["tumor"].mask[idx_target] is True or case.structures["tumor"].mask[
        max(0, min(geom.shape[0] - 1, idx_target[0])),
        max(0, min(geom.shape[1] - 1, idx_target[1])),
        max(0, min(geom.shape[2] - 1, idx_target[2])),
    ]

    pt_risk_lps = (-pt_risk_ras[0], -pt_risk_ras[1], pt_risk_ras[2])
    vox_risk = geom.lps_to_voxel(pt_risk_lps)
    idx_risk = tuple(int(round(x)) for x in vox_risk)
    assert case.structures["vessel"].mask[idx_risk] is True or case.structures["vessel"].mask[
        max(0, min(geom.shape[0] - 1, idx_risk[0])),
        max(0, min(geom.shape[1] - 1, idx_risk[1])),
        max(0, min(geom.shape[2] - 1, idx_risk[2])),
    ]


def test_mesh_obj_export() -> None:
    """Acceptance test 5: Valid OBJ text, correct bounding box in RAS, positive signed volume."""
    repo = PhantomRepository()
    _, geom = repo.get_case("phantom_001")  # type: ignore[misc]

    response = client.get("/cases/phantom_001/structures/tumor/mesh.obj")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text

    # Parse vertices and faces
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("v "):
            parts = [float(x) for x in line.split()[1:]]
            assert len(parts) == 3
            vertices.append(parts)
        elif line.startswith("f "):
            # 1-indexed to 0-indexed
            parts = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            assert len(parts) == 3
            faces.append(parts)

    verts_arr = np.array(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int64)

    assert len(verts_arr) > 0
    assert len(faces_arr) > 0

    # 1. Signed volume is strictly positive
    signed_vol = mesh_signed_volume_ras(verts_arr, faces_arr)
    assert signed_vol > 0.0

    # Theoretical volume of sphere with r=6 is 4/3 * pi * 6^3 ~ 904.78 mm^3
    assert math.isclose(signed_vol, 904.78, rel_tol=0.1)

    # 2. Bounding box in RAS matches sphere extents within one voxel
    # Target center in local mm: (64.0, 32.0, 24.0), radius = 6.0 mm
    tumor_center_ras = np.array(
        local_point_zyx_mm_to_ras_mm(geom, (64.0, 32.0, 24.0)),
        dtype=np.float64,
    )
    radius_mm = 6.0
    max_voxel_spacing = max(geom.spacing_mm)

    mesh_min = np.min(verts_arr, axis=0)
    mesh_max = np.max(verts_arr, axis=0)
    expected_min = tumor_center_ras - radius_mm
    expected_max = tumor_center_ras + radius_mm

    # In any rotated coordinate system, a sphere of radius R has extent center +/- R
    for i in range(3):
        assert abs(mesh_min[i] - expected_min[i]) <= max_voxel_spacing
        assert abs(mesh_max[i] - expected_max[i]) <= max_voxel_spacing


def test_uncertainty_repeatability() -> None:
    """Acceptance test 6: POST /uncertainty with fixed seed returns identical results."""
    payload = {
        "target_name": "tumor",
        "risk_name": "vessel",
        "sigma_mm": 1.0,
        "corr_length_mm": 5.0,
        "n_samples": 20,
        "seed": 12345,
    }
    resp1 = client.post("/cases/phantom_001/uncertainty", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()

    resp2 = client.post("/cases/phantom_001/uncertainty", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()

    assert data1 == data2
    assert data1["coordinate_system"] == "RAS"
    assert len(data1["samples_mm"]) == 20
    assert_no_paths_in_json(data1)


def test_n_samples_cap_validation() -> None:
    """Acceptance test 7: n_samples=301 returns 422 explaining the cap."""
    payload = {
        "target_name": "tumor",
        "risk_name": "vessel",
        "sigma_mm": 1.0,
        "corr_length_mm": 5.0,
        "n_samples": 301,
        "seed": 42,
    }
    response = client.post("/cases/phantom_001/uncertainty", json=payload)
    assert response.status_code == 422
    data = response.json()
    detail_str = str(data)
    assert "300" in detail_str
    assert_no_paths_in_json(data)


def test_404_not_found_errors() -> None:
    """Acceptance test 8: Non-existent case or structure returns 404 with helpful message."""
    resp1 = client.get("/cases/non_existent_case_id")
    assert resp1.status_code == 404
    assert "non_existent_case_id" in resp1.json()["detail"]
    assert_no_paths_in_json(resp1.json())

    resp2 = client.get("/cases/phantom_001/structures/unknown_organ/mesh.obj")
    assert resp2.status_code == 404
    assert "unknown_organ" in resp2.json()["detail"]
    assert_no_paths_in_json(resp2.json())

    resp3 = client.post(
        "/cases/phantom_001/clearance",
        json={"target_name": "tumor", "risk_name": "non_existent_risk"},
    )
    assert resp3.status_code == 404
    assert "non_existent_risk" in resp3.json()["detail"]
    assert_no_paths_in_json(resp3.json())


def test_json_bodies_contain_no_filesystem_paths() -> None:
    """Acceptance test 9: Comprehensive scan across endpoints to ensure no file paths are leaked."""
    # /health
    assert_no_paths_in_json(client.get("/health").json())
    # /cases
    assert_no_paths_in_json(client.get("/cases").json())
    # /cases/phantom_001
    assert_no_paths_in_json(client.get("/cases/phantom_001").json())
    # /cases/phantom_001/clearance
    assert_no_paths_in_json(
        client.post(
            "/cases/phantom_001/clearance",
            json={"target_name": "tumor", "risk_name": "vessel"},
        ).json()
    )
