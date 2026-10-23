"""API integration tests for /conditions endpoints.

Verifies:
  - GET /conditions: lists all conditions as {id, name}.
  - POST /conditions/simulate: happy paths across presets and severities.
  - POST /conditions/pick: click-to-nerve with pathway blocking and affected timeline.
  - 404 for unknown condition ID.
  - 422 for invalid body parameters (validator error list).
  - 422 for unsupported side argument.
  - Response payload sizes strictly under 1 MB.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hdt.api.app import app
from hdt.body.presets import PRESETS

client = TestClient(app)


def test_get_conditions_endpoint() -> None:
    """GET /conditions returns list of {id, name}."""
    resp = client.get("/conditions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 6

    ids = [item["id"] for item in data]
    assert "sciatica" in ids
    assert "carpal_tunnel" in ids
    assert "sci_t10" in ids

    for item in data:
        assert "id" in item
        assert "name" in item


def test_post_simulate_condition_happy_path() -> None:
    """POST /conditions/simulate executes successfully and returns ConditionResult."""
    body_dict = PRESETS["average"].model_dump()
    payload = {
        "body": body_dict,
        "condition_id": "sciatica",
        "side": "r",
        "severity": "mild",
    }
    resp = client.post("/conditions/simulate", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["condition_id"] == "sciatica"
    assert "dermatome_effects" in data
    assert "nerve_effects" in data
    assert "lesions" in data
    assert "unaffected_dermatomes" in data
    assert len(data["dermatome_effects"]) > 0
    assert len(data["lesions"]) > 0


def test_post_pick_condition_happy_path_foot_blocked() -> None:
    """POST /conditions/pick on foot dorsum under severe sciatica blocks signal."""
    body = PRESETS["average"]
    # Query point on right lateral lower leg / foot
    pt_ras = [155.0, 9.0, 225.0]

    payload = {
        "body": body.model_dump(),
        "point_ras": pt_ras,
        "condition_id": "sciatica",
        "side": "r",
        "severity": "severe",
    }
    resp = client.post("/conditions/pick", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["dermatome_id"] in ("L5", "S1")
    assert data["side"] == "r"
    assert data["blocked"] is True
    assert data["block_index"] is not None
    assert data["block_point"] is not None
    assert len(data["block_point"]) == 3
    assert "pathway" in data
    assert "affected_timeline" in data
    assert data["affected_timeline"]["status"] == "blocked"


def test_post_pick_condition_unaffected_arm() -> None:
    """POST /conditions/pick on arm under right sciatica is not blocked."""
    body = PRESETS["average"]
    # Point on right hand / thumb (C6 region)
    pt_ras = [270.0, 0.0, 850.0]

    payload = {
        "body": body.model_dump(),
        "point_ras": pt_ras,
        "condition_id": "sciatica",
        "side": "r",
        "severity": "severe",
    }
    resp = client.post("/conditions/pick", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["blocked"] is False
    assert data["block_index"] is None
    assert data["block_point"] is None
    assert data["affected_timeline"]["status"] == "normal"


def test_unknown_condition_returns_404() -> None:
    """Non-existent condition ID returns 404."""
    body_dict = PRESETS["average"].model_dump()

    # Simulate endpoint
    resp1 = client.post(
        "/conditions/simulate",
        json={"body": body_dict, "condition_id": "nonexistent_condition", "side": "r"},
    )
    assert resp1.status_code == 404
    assert "not found" in resp1.json()["detail"].lower()

    # Pick endpoint
    resp2 = client.post(
        "/conditions/pick",
        json={
            "body": body_dict,
            "point_ras": [100.0, 0.0, 500.0],
            "condition_id": "nonexistent_condition",
            "side": "r",
        },
    )
    assert resp2.status_code == 404


def test_invalid_body_returns_422() -> None:
    """Invalid body params return 422 with validation error messages."""
    bad_body = PRESETS["average"].model_dump()
    bad_body["height_mm"] = 400.0  # Outside allowed range (1000 - 2300 mm)

    resp = client.post(
        "/conditions/simulate",
        json={"body": bad_body, "condition_id": "sciatica", "side": "r"},
    )
    assert resp.status_code == 422
    err_detail = resp.json()["detail"]
    assert "errors" in err_detail
    assert len(err_detail["errors"]) > 0


def test_unsupported_side_returns_422() -> None:
    """Unsupported side returns 422."""
    body_dict = PRESETS["average"].model_dump()

    resp = client.post(
        "/conditions/simulate",
        json={"body": body_dict, "condition_id": "sciatica", "side": "anterior"},
    )
    assert resp.status_code == 422


def test_payload_size_under_1mb() -> None:
    """Condition simulation response payload size is well under 1 MB."""
    body_dict = PRESETS["average"].model_dump()
    payload = {
        "body": body_dict,
        "condition_id": "sciatica",
        "side": "both",
        "severity": "moderate",
    }
    resp = client.post("/conditions/simulate", json=payload)
    assert resp.status_code == 200

    raw_bytes = len(resp.content)
    assert raw_bytes < 1_000_000, f"Payload {raw_bytes} bytes exceeds 1 MB"
