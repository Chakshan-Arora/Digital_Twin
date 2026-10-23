"""Tests for /body/systems endpoints via FastAPI TestClient.

Verifies:
  - GET /body/systems returns system registry list.
  - POST /body/systems/{system_id} returns 200 and items for all available systems
    across all three stature presets (short, average, tall).
  - POST /body/systems/{system_id} returns 404 for unknown or unavailable systems.
  - POST /body/systems/{system_id} returns 422 with validation errors for invalid body models.
  - Response payload sizes remain strictly under 1.5 MB for the tall preset.
  - Endpoint latency is well under 1000 ms.
"""

import time

import pytest
from fastapi.testclient import TestClient

from hdt.api.app import app
from hdt.body.presets import PRESETS

client = TestClient(app)


def test_get_systems_registry() -> None:
    """GET /body/systems returns registry list containing skeleton, vascular, organs."""
    resp = client.get("/body/systems")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    system_ids = {entry["id"] for entry in data}

    assert "skeleton" in system_ids
    assert "vascular" in system_ids
    assert "organs" in system_ids
    assert "nervous" in system_ids
    assert "muscular" in system_ids

    # Check required fields
    skeleton_entry = next(e for e in data if e["id"] == "skeleton")
    assert skeleton_entry["available"] is True
    assert skeleton_entry["default_visible"] is True
    assert "name" in skeleton_entry
    assert "color_hint" in skeleton_entry


@pytest.mark.parametrize("system_id", ["skeleton", "vascular", "organs"])
@pytest.mark.parametrize("preset_name", ["short", "average", "tall"])
def test_post_system_all_presets(system_id: str, preset_name: str) -> None:
    """POST /body/systems/{system_id} returns valid items for all systems and presets."""
    body = PRESETS[preset_name]
    resp = client.post(
        f"/body/systems/{system_id}",
        json=body.model_dump(),
    )
    assert resp.status_code == 200, f"Failed for {system_id} with preset {preset_name}: {resp.text}"

    data = resp.json()
    assert data["system_id"] == system_id
    assert "items" in data
    assert data["item_count"] == len(data["items"])
    assert data["item_count"] > 0

    # Validate structure of each item
    for item in data["items"]:
        assert "id" in item
        assert "name" in item
        assert "side" in item
        assert "primitive" in item
        assert "color_hint" in item
        assert "label" in item
        assert item["primitive"]["kind"] in ("tube", "ellipsoid", "mesh_hint")


def test_post_unknown_system_returns_404() -> None:
    """POST /body/systems/{system_id} returns 404 for nonexistent systems."""
    body = PRESETS["average"]
    resp = client.post(
        "/body/systems/unknown_system_xyz",
        json=body.model_dump(),
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_post_unavailable_system_returns_404() -> None:
    """POST /body/systems/{system_id} returns 404 for systems flagged available=False."""
    body = PRESETS["average"]
    # 'nervous' and 'muscular' have available=False in registry
    resp = client.post(
        "/body/systems/nervous",
        json=body.model_dump(),
    )
    assert resp.status_code == 404


def test_post_invalid_body_returns_422() -> None:
    """POST with an anatomically implausible BodyParams returns 422 with validator messages."""
    # Implausible height: 50,000 mm
    invalid_body = {
        "height_mm": 50000.0,
        "joints": {j.value: [0.0, 0.0, 0.0] for j in PRESETS["average"].joints},
        "source": "preset",
    }
    resp = client.post(
        "/body/systems/skeleton",
        json=invalid_body,
    )
    assert resp.status_code == 422
    err_detail = resp.json()["detail"]
    assert "errors" in err_detail
    assert any("outside plausible range" in msg for msg in err_detail["errors"])


def test_response_size_under_1500kb() -> None:
    """All system responses for the tall preset must remain strictly under 1.5 MB."""
    tall_body = PRESETS["tall"]

    for system_id in ("skeleton", "vascular", "organs"):
        resp = client.post(
            f"/body/systems/{system_id}",
            json=tall_body.model_dump(),
        )
        assert resp.status_code == 200
        size_bytes = len(resp.content)
        size_kb = size_bytes / 1024.0

        assert size_bytes < 1_500_000, (
            f"Response for {system_id} ({size_kb:.1f} KB) exceeds 1.5 MB limit"
        )


def test_endpoint_timings() -> None:
    """Each system generation endpoint should respond well under 1000 ms."""
    body = PRESETS["average"]

    for system_id in ("skeleton", "vascular", "organs"):
        t0 = time.perf_counter()
        resp = client.post(
            f"/body/systems/{system_id}",
            json=body.model_dump(),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        assert dt_ms < 1000.0, f"Endpoint {system_id} took {dt_ms:.1f} ms (expected < 1000 ms)"
