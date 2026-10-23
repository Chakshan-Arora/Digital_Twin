"""Integration tests for Body Explorer preset and dermatome pick endpoints."""

import copy

from fastapi.testclient import TestClient

from hdt.api.app import app
from hdt.body.models import JointName
from hdt.body.presets import PRESETS
from hdt.nervous.anchors import skin_anchor

client = TestClient(app)


def test_get_body_presets_list():
    """Verify GET /body/presets returns 200 and summary list of presets."""
    response = client.get("/body/presets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    names = [p["name"] for p in data]
    assert "short" in names
    assert "average" in names
    assert "tall" in names


def test_get_body_preset_by_name():
    """Verify GET /body/presets/{name} returns 200 and complete BodyParams."""
    response = client.get("/body/presets/average")
    assert response.status_code == 200
    body = response.json()
    assert body["height_mm"] == 1750.0
    assert body["source"] == "preset"
    assert "joints" in body
    assert len(body["joints"]) == 20


def test_get_body_preset_not_found():
    """Verify GET /body/presets/{name} returns 404 for unknown preset."""
    response = client.get("/body/presets/nonexistent_giant")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"]


def test_post_body_pick_happy_path():
    """Verify POST /body/pick returns dermatome, pathway polyline, and timeline."""
    avg_body = PRESETS["average"]
    # Query point at right foot dorsum
    pt_anchor = skin_anchor(avg_body, "L5", side="r")
    query_pt = [pt_anchor[0], pt_anchor[1] + 2.0, pt_anchor[2]]

    payload = {
        "body": avg_body.model_dump(),
        "point_ras": query_pt,
    }
    response = client.post("/body/pick", json=payload)
    assert response.status_code == 200
    result = response.json()

    assert result["dermatome_id"] == "L5"
    assert result["side"] == "r"
    assert "lower leg" in result["region"].lower() or "dorsum of foot" in result["region"].lower()
    assert result["distance_to_anchor_mm"] >= 0.0

    # Pathway verification
    pathway = result["pathway"]
    assert len(pathway) >= 5
    assert pathway[0]["label"].startswith("skin_anchor_L5_r")
    assert pathway[-1]["label"] == "cortex_S1"

    # Timeline verification
    timeline = result["timeline"]
    assert "fibers" in timeline
    assert "A_beta" in timeline["fibers"]
    assert "A_delta" in timeline["fibers"]
    assert "C" in timeline["fibers"]

    lat_ab = timeline["fibers"]["A_beta"]["latency_ms"]
    lat_ad = timeline["fibers"]["A_delta"]["latency_ms"]
    lat_c = timeline["fibers"]["C"]["latency_ms"]
    assert lat_ab < lat_ad < lat_c

    # Disclaimer check
    assert "disclaimer" in result
    assert "NOT a clinical" in result["disclaimer"]


def test_post_body_pick_invalid_body_returns_422():
    """Verify POST /body/pick returns 422 with validator messages on malformed body."""
    bad_body = copy.deepcopy(PRESETS["average"]).model_dump()
    bad_body["height_mm"] = 450.0  # Outside [1000, 2300] mm range

    payload = {
        "body": bad_body,
        "point_ras": [0.0, 0.0, 1000.0],
    }
    response = client.post("/body/pick", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

    # Test missing joint triggers 422
    bad_body_missing = copy.deepcopy(PRESETS["average"]).model_dump()
    del bad_body_missing["joints"][JointName.L_KNEE.value]
    res_missing = client.post(
        "/body/pick", json={"body": bad_body_missing, "point_ras": [0.0, 0.0, 1000.0]}
    )
    assert res_missing.status_code == 422


def test_post_body_pick_far_point():
    """Verify POST /body/pick accepts distant points and calculates distance appropriately."""
    avg_body = PRESETS["average"]
    # Point located 500 mm in front of the head
    far_pt = [0.0, 500.0, 1700.0]

    payload = {
        "body": avg_body.model_dump(),
        "point_ras": far_pt,
    }
    response = client.post("/body/pick", json=payload)
    assert response.status_code == 200
    result = response.json()

    assert result["distance_to_anchor_mm"] > 400.0
    assert result["dermatome_id"] in ["T2", "T3", "T4", "C3", "C4"]
