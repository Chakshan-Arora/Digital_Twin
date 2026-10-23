"""Tests for hdt.systems.vascular — structural, envelope containment, and vascular rules.

Verifies:
  - All vascular tube polyline points reside strictly inside the body envelope.
  - Aorta radius is larger at the arch than at the abdominal bifurcation.
  - Arterial and venous items have distinct color hints.
  - Determinism across multiple executions.
  - Execution runtime under 100 ms.
"""

import time

from hdt.body.presets import PRESETS
from hdt.systems.common import inside_envelope
from hdt.systems.vascular import build_vascular


def test_all_points_inside_envelope() -> None:
    """Every point in every vascular tube must be inside the body envelope across presets."""
    for preset_name in ("short", "average", "tall"):
        body = PRESETS[preset_name]
        items = build_vascular(body)
        assert len(items) > 0

        for item in items:
            prim = item["primitive"]
            assert prim["kind"] == "tube", f"Vessel '{item['name']}' is not a tube"
            for idx, pt in enumerate(prim["polyline"]):
                pt_tuple = (float(pt[0]), float(pt[1]), float(pt[2]))
                assert inside_envelope(body, pt_tuple), (
                    f"Vessel '{item['name']}' point #{idx} {pt_tuple} "
                    f"is outside body envelope for preset '{preset_name}'"
                )


def test_aorta_radius_decreasing() -> None:
    """Aorta radius at the arch must be larger than at the abdominal bifurcation."""
    items = build_vascular(PRESETS["average"])

    arch_item = next((it for it in items if it["name"] == "aortic_arch"), None)
    abd_item = next((it for it in items if it["name"] == "abdominal_aorta"), None)

    assert arch_item is not None, "Aortic arch vessel not found"
    assert abd_item is not None, "Abdominal aorta vessel not found"

    arch_radius = arch_item["primitive"]["radius_mm"]
    abd_end_radius = (
        abd_item["primitive"].get("radius_end_mm")
        or abd_item["primitive"]["radius_mm"]
    )

    assert arch_radius > abd_end_radius, (
        f"Aortic arch radius ({arch_radius} mm) not larger than "
        f"bifurcation radius ({abd_end_radius} mm)"
    )


def test_arterial_venous_different_colors() -> None:
    """Arterial and venous vessels must have distinct color hints."""
    items = build_vascular(PRESETS["average"])

    art_colors = {it["color_hint"] for it in items if it["system"] == "vascular_arterial"}
    ven_colors = {it["color_hint"] for it in items if it["system"] == "vascular_venous"}

    assert len(art_colors) == 1, f"Expected 1 arterial color, found {art_colors}"
    assert len(ven_colors) == 1, f"Expected 1 venous color, found {ven_colors}"
    assert art_colors != ven_colors, f"Colors should differ: {art_colors} vs {ven_colors}"


def test_determinism() -> None:
    """Calling build_vascular twice returns identical output."""
    v1 = build_vascular(PRESETS["average"])
    v2 = build_vascular(PRESETS["average"])
    assert v1 == v2


def test_execution_time() -> None:
    """build_vascular should run well under 100 ms."""
    t0 = time.perf_counter()
    _ = build_vascular(PRESETS["average"])
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 100.0, f"build_vascular took {dt_ms:.2f} ms (expected < 100 ms)"


def test_no_lateral_vessels_on_body_axis() -> None:
    """No waypoint of a lateral vessel (side != 'mid') may lie on or near body axis (x≈0, y≈0)."""
    for preset_name in ("short", "average", "tall"):
        body = PRESETS[preset_name]
        items = build_vascular(body)
        for item in items:
            if item["side"] in ("l", "r"):
                for idx, pt in enumerate(item["primitive"]["polyline"]):
                    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                    assert not (abs(x) < 2.0 and abs(y) < 2.0), (
                        f"Preset '{preset_name}', lateral vessel '{item['name']}' "
                        f"(side='{item['side']}') point #{idx} ({x:.2f}, {y:.2f}, {z:.2f}) "
                        "lies on the body axis (x≈0, y≈0)."
                    )


