"""Unit and analytic tests for hdt.conditions.simulate.

Verifies:
  - Sciatica right: L5 and S1 right affected, L5 left normal; mild affected latency
    = normal latency x factor.
  - Severe blocks all affected fibers with null latency.
  - T10 injury: all dermatomes below T10 blocked, T9 and above normal.
  - Taller body: longer normal latency for the same dermatome, ratio near height ratio;
    lesion positions scale with the body.
  - Every atlas condition x all presets x all severities simulates without error; determinism.
"""

from __future__ import annotations

import math

import pytest

from hdt.atlas.models import SPINAL_CORD_LEVELS, FiberType
from hdt.body.presets import PRESETS
from hdt.conditions.models import Severity, SignalStatus
from hdt.conditions.simulate import FIBER_SEVERITY_FACTORS, simulate_condition


def test_sciatica_right_affected_and_mild_latency() -> None:
    """Sciatica (right): L5 and S1 right are affected, L5 left is normal.

    In mild severity, affected latency = round(normal_latency * factor, 2).
    """
    body = PRESETS["average"]
    res = simulate_condition(body, "sciatica", side="r", severity=Severity.MILD)

    # Check affected dermatomes
    affected_keys = {(de.dermatome_id, de.side) for de in res.dermatome_effects}
    assert ("L5", "r") in affected_keys, "Right L5 should be affected under right sciatica"
    assert ("S1", "r") in affected_keys, "Right S1 should be affected under right sciatica"
    assert ("L5", "l") not in affected_keys, "Left L5 must NOT be affected"
    assert ("S1", "l") not in affected_keys, "Left S1 must NOT be affected"

    # L5 must not be in unaffected_dermatomes, while unaffected ones like C6 are
    assert "L5" not in res.unaffected_dermatomes
    assert "C6" in res.unaffected_dermatomes

    # Check latency calculation for mild severity
    l5_r = next(de for de in res.dermatome_effects if de.dermatome_id == "L5" and de.side == "r")
    assert l5_r.status == SignalStatus.SLOWED

    for f_type in [FiberType.A_BETA, FiberType.A_DELTA, FiberType.C]:
        factor = FIBER_SEVERITY_FACTORS[Severity.MILD][f_type]
        fe = l5_r.fibers[f_type]
        assert fe.normal_latency_ms > 0.0

        if factor == 1.0:
            assert fe.status == SignalStatus.NORMAL
            assert fe.affected_latency_ms == fe.normal_latency_ms
        else:
            assert fe.status == SignalStatus.SLOWED
            assert fe.affected_latency_ms is not None
            expected_lat = round(fe.normal_latency_ms * factor, 2)
            assert math.isclose(fe.affected_latency_ms, expected_lat, abs_tol=0.05)


def test_severe_blocks_all_affected_fibers() -> None:
    """Severe condition blocks all affected fibers, resulting in null affected latencies."""
    body = PRESETS["average"]
    res = simulate_condition(body, "sciatica", side="r", severity=Severity.SEVERE)

    l5_r = next(de for de in res.dermatome_effects if de.dermatome_id == "L5" and de.side == "r")
    assert l5_r.status == SignalStatus.BLOCKED

    for _f_type, fe in l5_r.fibers.items():
        assert fe.status == SignalStatus.BLOCKED
        assert fe.affected_latency_ms is None
        assert fe.affected_velocity_mps is None


def test_t10_injury_blocks_all_dermatomes_below_t10() -> None:
    """T10 spinal cord injury blocks all dermatomes below T10 bilateral; T9 and above normal."""
    body = PRESETS["average"]
    # Even if called with side='r', sci_t10 is a bilateral transection
    res = simulate_condition(body, "sci_t10", side="r", severity=Severity.MODERATE)

    cord_order = list(SPINAL_CORD_LEVELS)
    t10_idx = cord_order.index("cord_T10")
    caudal_to_t10 = {lvl.replace("cord_", "") for lvl in cord_order[: t10_idx + 1]}
    rostral_to_t10 = {lvl.replace("cord_", "") for lvl in cord_order[t10_idx + 1 :]}

    affected_map = {(de.dermatome_id, de.side): de for de in res.dermatome_effects}

    # Every caudal dermatome must be present on both sides and BLOCKED
    for d_id in ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1", "S2"]:
        if d_id in caudal_to_t10:
            assert (d_id, "l") in affected_map, f"{d_id}_l should be affected"
            assert (d_id, "r") in affected_map, f"{d_id}_r should be affected"
            assert affected_map[(d_id, "l")].status == SignalStatus.BLOCKED
            assert affected_map[(d_id, "r")].status == SignalStatus.BLOCKED

    # Rostral levels (T9, T8, ..., C2) must NOT be affected
    for d_id in ["T9", "T8", "T7", "T4", "C6", "C7"]:
        if d_id in rostral_to_t10:
            assert (d_id, "l") not in affected_map
            assert (d_id, "r") not in affected_map
            assert d_id in res.unaffected_dermatomes


def test_taller_body_latency_and_scaling() -> None:
    """Taller body has longer normal latency; ratio matches height ratio; lesion scales."""
    short_body = PRESETS["short"]
    tall_body = PRESETS["tall"]

    h_short = short_body.height_mm
    h_tall = tall_body.height_mm
    h_ratio = h_tall / h_short

    res_short = simulate_condition(short_body, "sciatica", side="r", severity=Severity.MILD)
    res_tall = simulate_condition(tall_body, "sciatica", side="r", severity=Severity.MILD)

    l5_short = next(de for de in res_short.dermatome_effects if de.dermatome_id == "L5")
    l5_tall = next(de for de in res_tall.dermatome_effects if de.dermatome_id == "L5")

    lat_short = l5_short.fibers[FiberType.A_BETA].normal_latency_ms
    lat_tall = l5_tall.fibers[FiberType.A_BETA].normal_latency_ms

    assert lat_tall > lat_short
    lat_ratio = lat_tall / lat_short
    assert math.isclose(lat_ratio, h_ratio, rel_tol=0.08), (
        f"Latency ratio {lat_ratio:.3f} deviates from stature ratio {h_ratio:.3f}"
    )

    # Lesion z positions scale with body stature
    lesion_short_z = res_short.lesions[0].point[2]
    lesion_tall_z = res_tall.lesions[0].point[2]
    z_ratio = lesion_tall_z / lesion_short_z
    assert math.isclose(z_ratio, h_ratio, rel_tol=0.05), (
        f"Lesion z ratio {z_ratio:.3f} deviates from stature ratio {h_ratio:.3f}"
    )


@pytest.mark.parametrize(
    "condition_id",
    [
        "sciatica",
        "carpal_tunnel",
        "sci_t10",
        "meralgia_paresthetica",
        "l4_l5_herniation",
        "cervical_radiculopathy",
    ],
)
@pytest.mark.parametrize("preset_name", ["short", "average", "tall"])
@pytest.mark.parametrize("severity", [Severity.MILD, Severity.MODERATE, Severity.SEVERE])
def test_every_condition_preset_severity_runs_and_deterministic(
    condition_id: str, preset_name: str, severity: Severity
) -> None:
    """Every atlas condition executes cleanly without error across presets and is deterministic."""
    body = PRESETS[preset_name]
    res1 = simulate_condition(body, condition_id, side="r", severity=severity)
    res2 = simulate_condition(body, condition_id, side="r", severity=severity)

    assert res1.condition_id == condition_id
    assert len(res1.dermatome_effects) > 0
    assert len(res1.lesions) > 0
    assert res1 == res2
