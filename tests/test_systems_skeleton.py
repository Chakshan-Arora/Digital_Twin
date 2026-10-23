"""Tests for hdt.systems.skeleton — structural, analytic, and geometric invariants.

Verifies:
  - Vertebra count (24 vertebrae: 7 cervical, 12 thoracic, 5 lumbar),
    labelled C1..L5 in descending z order.
  - Rib count (24 ribs = 12 bilateral pairs).
  - Clavicle count (2) and femur count (2).
  - Femur length scales linearly with height (within 2% of height ratio).
  - Bilateral symmetry across x=0 for paired items (positions and radii).
  - Determinism across multiple executions.
"""

import math
import re
import time

from hdt.body.presets import PRESETS
from hdt.systems.common import inside_envelope
from hdt.systems.skeleton import build_skeleton

_VERTEBRA_RE = re.compile(r"^(C[1-7]|T([1-9]|1[0-2])|L[1-5])$")


def test_vertebrae_counts_labels_and_descending_z() -> None:
    """Vertebrae must be exactly 24, labelled C1..L5, strictly descending in z."""
    items = build_skeleton(PRESETS["average"])
    vert_items = [
        it
        for it in items
        if _VERTEBRA_RE.match(it["label"]) and it["primitive"]["kind"] == "ellipsoid"
    ]

    assert len(vert_items) == 24, f"Expected 24 vertebrae, found {len(vert_items)}"

    expected_labels = (
        [f"C{i}" for i in range(1, 8)]
        + [f"T{i}" for i in range(1, 13)]
        + [f"L{i}" for i in range(1, 6)]
    )
    observed_labels = [it["label"] for it in vert_items]
    assert observed_labels == expected_labels, (
        f"Vertebra labels mismatch: {observed_labels} != {expected_labels}"
    )

    # Strictly descending in z
    zs = [it["primitive"]["center"][2] for it in vert_items]
    for i in range(len(zs) - 1):
        assert zs[i] > zs[i + 1], (
            f"Vertebrae not strictly descending in z: {observed_labels[i]} (z={zs[i]:.2f}) "
            f"vs {observed_labels[i+1]} (z={zs[i+1]:.2f})"
        )


def test_ribs_count() -> None:
    """Ribcage must contain exactly 24 ribs (12 pairs) as tubes."""
    items = build_skeleton(PRESETS["average"])
    ribs = [
        it
        for it in items
        if "rib" in it["name"].lower() and it["primitive"]["kind"] == "tube"
    ]
    assert len(ribs) == 24, f"Expected 24 ribs, found {len(ribs)}"

    left_ribs = [r for r in ribs if r["side"] == "l"]
    right_ribs = [r for r in ribs if r["side"] == "r"]
    assert len(left_ribs) == 12
    assert len(right_ribs) == 12


def test_clavicles_scapulae_and_femurs_counts() -> None:
    """Must contain exactly 2 clavicles, 2 scapulae, and 2 femurs."""
    items = build_skeleton(PRESETS["average"])
    clavicles = [it for it in items if "clavicle" in it["name"].lower()]
    scapulae = [it for it in items if "scapula" in it["name"].lower()]
    femurs = [it for it in items if "femur" in it["name"].lower()]

    assert len(clavicles) == 2, f"Expected 2 clavicles, found {len(clavicles)}"
    assert len(scapulae) == 2, f"Expected 2 scapulae, found {len(scapulae)}"
    assert len(femurs) == 2, f"Expected 2 femurs, found {len(femurs)}"


def _get_femur_length(items: list[dict], side: str = "r") -> float:
    for it in items:
        if it["name"] == f"{side}_femur":
            poly = it["primitive"]["polyline"]
            p0 = poly[0]
            p1 = poly[-1]
            return math.sqrt(
                (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
            )
    raise KeyError(f"Femur '{side}_femur' not found in skeleton items")


def test_femur_scales_linearly_with_height() -> None:
    """Femur length ratio must be within 2% of height ratio across short, average, tall."""
    short_items = build_skeleton(PRESETS["short"])
    avg_items = build_skeleton(PRESETS["average"])
    tall_items = build_skeleton(PRESETS["tall"])

    h_short = PRESETS["short"].height_mm
    h_avg = PRESETS["average"].height_mm
    h_tall = PRESETS["tall"].height_mm

    l_short = _get_femur_length(short_items, "r")
    l_avg = _get_femur_length(avg_items, "r")
    l_tall = _get_femur_length(tall_items, "r")

    # Short vs Average
    ratio_sa = l_short / l_avg
    height_ratio_sa = h_short / h_avg
    assert math.isclose(ratio_sa, height_ratio_sa, rel_tol=0.02), (
        f"Femur ratio short/avg ({ratio_sa:.4f}) deviates from height ratio ({height_ratio_sa:.4f})"
    )

    # Tall vs Average
    ratio_ta = l_tall / l_avg
    height_ratio_ta = h_tall / h_avg
    assert math.isclose(ratio_ta, height_ratio_ta, rel_tol=0.02), (
        f"Femur ratio tall/avg ({ratio_ta:.4f}) deviates from height ratio ({height_ratio_ta:.4f})"
    )


def test_bilateral_mirror_symmetry() -> None:
    """Paired left and right skeletal items must mirror across x=0."""
    items = build_skeleton(PRESETS["average"])

    l_items = {
        it["name"][2:]: it
        for it in items
        if it["side"] == "l" and it["name"].startswith("l_")
    }
    r_items = {
        it["name"][2:]: it
        for it in items
        if it["side"] == "r" and it["name"].startswith("r_")
    }

    common_names = set(l_items.keys()) & set(r_items.keys())
    assert len(common_names) >= 15, f"Expected >= 15 paired bone types, got {len(common_names)}"

    for name in common_names:
        l_prim = l_items[name]["primitive"]
        r_prim = r_items[name]["primitive"]

        assert l_prim["kind"] == r_prim["kind"], f"Primitive kind mismatch for {name}"

        if l_prim["kind"] == "tube":
            l_poly = l_prim["polyline"]
            r_poly = r_prim["polyline"]
            assert len(l_poly) == len(r_poly)
            for lp, rp in zip(l_poly, r_poly, strict=True):
                # x must be negated, y and z identical within 1 mm
                assert math.isclose(lp[0], -rp[0], abs_tol=1.0), f"x symmetry failed for {name}"
                assert math.isclose(lp[1], rp[1], abs_tol=1.0), f"y symmetry failed for {name}"
                assert math.isclose(lp[2], rp[2], abs_tol=1.0), f"z symmetry failed for {name}"
            # Radii must be equal
            assert math.isclose(l_prim["radius_mm"], r_prim["radius_mm"], rel_tol=1e-3)
            if l_prim.get("radius_end_mm") is not None:
                assert math.isclose(l_prim["radius_end_mm"], r_prim["radius_end_mm"], rel_tol=1e-3)

        elif l_prim["kind"] == "ellipsoid":
            lc = l_prim["center"]
            rc = r_prim["center"]
            assert math.isclose(lc[0], -rc[0], abs_tol=1.0), f"x symmetry failed for {name}"
            assert math.isclose(lc[1], rc[1], abs_tol=1.0), f"y symmetry failed for {name}"
            assert math.isclose(lc[2], rc[2], abs_tol=1.0), f"z symmetry failed for {name}"
            for lr, rr in zip(l_prim["radii_mm"], r_prim["radii_mm"], strict=True):
                assert math.isclose(lr, rr, rel_tol=1e-3)


def test_determinism() -> None:
    """Generating the skeleton twice for the same body params must yield identical dicts."""
    s1 = build_skeleton(PRESETS["average"])
    s2 = build_skeleton(PRESETS["average"])
    assert s1 == s2


def test_execution_time() -> None:
    """build_skeleton should run well under 100 ms."""
    t0 = time.perf_counter()
    _ = build_skeleton(PRESETS["average"])
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 100.0, f"build_skeleton took {dt_ms:.2f} ms (expected < 100 ms)"


def test_every_skeleton_item_inside_envelope() -> None:
    """Every skeleton item point/surface lies inside inside_envelope across 3 presets."""
    for p_name in ("short", "average", "tall"):
        body = PRESETS[p_name]
        items = build_skeleton(body)
        for it in items:
            p = it["primitive"]
            kind = p["kind"]
            name = it["name"]
            if kind == "tube":
                for idx, pt in enumerate(p["polyline"]):
                    pt_tup = (float(pt[0]), float(pt[1]), float(pt[2]))
                    assert inside_envelope(body, pt_tup), (
                        f"Preset {p_name}: {name} polyline[{idx}]={pt_tup} outside envelope"
                    )
            elif kind == "ellipsoid":
                c = p["center"]
                r = p["radii_mm"]
                # Center + 6 axis surface extremities
                pts = [c]
                for k in range(3):
                    pp = list(c)
                    pp[k] += r[k]
                    pm = list(c)
                    pm[k] -= r[k]
                    pts.extend([pp, pm])
                for idx, pt in enumerate(pts):
                    pt_tup = (float(pt[0]), float(pt[1]), float(pt[2]))
                    assert inside_envelope(body, pt_tup), (
                        f"Preset {p_name}: {name} sample[{idx}]={pt_tup} outside envelope"
                    )
            elif kind == "mesh_hint":
                c = p["bounding_ellipsoid_center"]
                pt_tup = (float(c[0]), float(c[1]), float(c[2]))
                assert inside_envelope(body, pt_tup), (
                    f"Preset {p_name}: {name} mesh_hint center {pt_tup} outside envelope"
                )


def test_ribs_full_length_inside_envelope() -> None:
    """Ribs stay strictly inside the envelope along their full interpolated length."""
    for p_name in ("short", "average", "tall"):
        body = PRESETS[p_name]
        items = build_skeleton(body)
        ribs = [it for it in items if "rib" in it["name"].lower()]
        for rib in ribs:
            poly = rib["primitive"]["polyline"]
            for i in range(len(poly) - 1):
                p0, p1 = poly[i], poly[i + 1]
                for step in range(11):
                    t = step / 10.0
                    pt = (
                        p0[0] + t * (p1[0] - p0[0]),
                        p0[1] + t * (p1[1] - p0[1]),
                        p0[2] + t * (p1[2] - p0[2]),
                    )
                    assert inside_envelope(body, pt), (
                        f"Preset {p_name}: {rib['name']} segment {i} t={t:.1f} "
                        f"pt={pt} outside envelope"
                    )


def test_scapula_offset_scales_with_height() -> None:
    """Scapula posterior y offset scales strictly as a fraction of body height."""
    short_items = build_skeleton(PRESETS["short"])
    avg_items = build_skeleton(PRESETS["average"])
    tall_items = build_skeleton(PRESETS["tall"])

    def _get_scapula_y(items: list[dict]) -> float:
        scap = next(it for it in items if it["name"] == "r_scapula")
        return scap["primitive"]["center"][1]

    y_s = _get_scapula_y(short_items)
    y_a = _get_scapula_y(avg_items)
    y_t = _get_scapula_y(tall_items)

    h_s = PRESETS["short"].height_mm
    h_a = PRESETS["average"].height_mm
    h_t = PRESETS["tall"].height_mm

    assert math.isclose(y_s / y_a, h_s / h_a, rel_tol=1e-4)
    assert math.isclose(y_t / y_a, h_t / h_a, rel_tol=1e-4)
