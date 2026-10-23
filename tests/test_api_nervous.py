"""Unit and API integration tests for body shape, nervous network, and picking accuracy."""

from fastapi.testclient import TestClient

from hdt.api.app import app
from hdt.body.presets import PRESETS
from hdt.nervous.anchors import DERMATOME_ANCHORS, nearest_dermatome, skin_samples

client = TestClient(app)


def test_api_body_shape_presets():
    """Verify POST /body/shape returns correct segments for presets."""
    for p_name in ["short", "average", "tall"]:
        body = PRESETS[p_name]
        resp = client.post("/body/shape", json={"body": body.model_dump()})
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["height_mm"] == body.height_mm
        assert len(data["segments"]) == 15

        # Also test direct flattened body payload
        resp_flat = client.post("/body/shape", json=body.model_dump())
        assert resp_flat.status_code == 200


def test_api_body_nervous_payload_size_and_detail():
    """Verify POST /body/nervous produces valid response under 2MB for high and low detail."""
    for p_name in ["short", "average", "tall"]:
        body = PRESETS[p_name]

        # High detail
        resp_high = client.post(
            "/body/nervous", json={"body": body.model_dump(), "detail": "high"}
        )
        assert resp_high.status_code == 200, resp_high.text
        data_high = resp_high.json()

        # Payload size must be strictly under 2 MB (2,097,152 bytes)
        raw_bytes = len(resp_high.content)
        assert raw_bytes < 2 * 1024 * 1024, f"Payload {raw_bytes} exceeds 2MB limit"

        assert data_high["detail"] == "high"
        assert len(data_high["spinal_cord"]) > 50
        assert len(data_high["brain_structures"]) == 6
        assert len(data_high["spinal_roots"]) == 62
        assert len(data_high["intercostal_nerves"]) == 22
        assert len(data_high["cranial_nerves"]) == 2

        # Low detail
        resp_low = client.post(
            "/body/nervous", json={"body": body.model_dump(), "detail": "low"}
        )
        assert resp_low.status_code == 200
        data_low = resp_low.json()

        assert data_low["detail"] == "low"
        assert data_high["total_vertex_count"] > data_low["total_vertex_count"]


def test_api_body_nervous_validation_errors():
    """Verify POST /body/shape and /body/nervous return 422 on invalid bodies."""
    # Empty body
    resp_empty = client.post("/body/shape", json={})
    assert resp_empty.status_code == 422

    # Negative height
    resp_invalid = client.post(
        "/body/nervous",
        json={"body": {"height_mm": -100.0, "joints": {}}},
    )
    assert resp_invalid.status_code == 422


def test_dermatome_picking_accuracy_ge_90_percent():
    """Verify nearest_dermatome achieves >= 90% accuracy on sample self-retrieval."""
    body = PRESETS["average"]
    correct = 0
    total = 0
    per_dermatome_results: dict[str, dict[str, int]] = {}

    for d_id in DERMATOME_ANCHORS:
        per_dermatome_results[d_id] = {"correct": 0, "total": 0}
        for side in ["r", "l"]:
            samples = skin_samples(body, d_id, side=side, n=6)
            for pt in samples:
                picked_id, picked_side, dist = nearest_dermatome(body, pt)
                total += 1
                per_dermatome_results[d_id]["total"] += 1
                if picked_id == d_id and picked_side == side:
                    correct += 1
                    per_dermatome_results[d_id]["correct"] += 1

    overall_accuracy = correct / total
    assert overall_accuracy >= 0.90, f"Overall accuracy {overall_accuracy*100:.1f}% below 90%"

    # Check for any dermatome with < 80% accuracy
    below_80 = []
    for d_id, res in per_dermatome_results.items():
        acc = res["correct"] / res["total"]
        if acc < 0.80:
            below_80.append((d_id, acc))

    assert len(below_80) == 0, f"Dermatomes below 80% accuracy: {below_80}"
