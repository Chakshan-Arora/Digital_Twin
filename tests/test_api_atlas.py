"""Integration tests for the anatomical neural atlas endpoints."""

from fastapi.testclient import TestClient

from hdt.api.app import app

client = TestClient(app)


def test_get_atlas_endpoint():
    """Verify GET /atlas returns 200 with full atlas payload."""
    response = client.get("/atlas")
    assert response.status_code == 200
    data = response.json()

    assert "disclaimer" in data
    assert "dermatomes" in data
    assert "conditions" in data
    assert len(data["dermatomes"]) == 29
    assert len(data["conditions"]) == 6

    # Verify first and last dermatomes
    ids = [d["id"] for d in data["dermatomes"]]
    assert ids[0] == "C2"
    assert ids[-1] == "S5"
    assert "L5" in ids
    assert "T10" in ids


def test_get_dermatome_success():
    """Verify GET /atlas/dermatomes/{id} returns 200 and expected single dermatome."""
    # Test L5
    response = client.get("/atlas/dermatomes/L5")
    assert response.status_code == 200
    dermatome = response.json()

    assert dermatome["id"] == "L5"
    assert dermatome["root"] == "L5"
    assert "lateral lower leg" in dermatome["region"].lower()
    assert dermatome["body_zone"]["kind"] == "leg"
    assert dermatome["path"][-1] == "cortex_S1"
    assert "cord_L5" in dermatome["path"]

    # Test case-insensitivity: c6 vs C6
    res_c6 = client.get("/atlas/dermatomes/c6")
    assert res_c6.status_code == 200
    assert res_c6.json()["id"] == "C6"


def test_get_dermatome_not_found():
    """Verify GET /atlas/dermatomes/{id} returns 404 for non-existent IDs."""
    response = client.get("/atlas/dermatomes/nonexistent_xyz")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found in atlas" in data["detail"]


def test_get_conditions_endpoint():
    """Verify GET /atlas/conditions returns list of all 6 conditions."""
    response = client.get("/atlas/conditions")
    assert response.status_code == 200
    conditions = response.json()

    assert isinstance(conditions, list)
    assert len(conditions) == 6

    cond_ids = {c["id"] for c in conditions}
    expected_ids = {
        "sciatica",
        "carpal_tunnel",
        "cervical_radiculopathy",
        "l4_l5_herniation",
        "sci_t10",
        "meralgia_paresthetica",
    }
    assert cond_ids == expected_ids

    # Check Sciatica details
    sciatica = next(c for c in conditions if c["id"] == "sciatica")
    assert "L5" in sciatica["compressed_roots"]
    assert "S1" in sciatica["compressed_roots"]
    assert len(sciatica["symptoms"]) >= 3
