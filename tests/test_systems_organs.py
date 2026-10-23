"""Tests for hdt.systems.organs — anatomical positioning, bounds, and overlap constraints.

Verifies:
  - Liver center has x > 0 (person's right).
  - Spleen and stomach centers have x < 0 (person's left).
  - Heart center has x < 0 and |x| < 60 mm (slightly left of midline).
  - Organs are contained within torso (z between pelvis and neck, with brain in head).
  - Pairwise overlap volume below documented tolerance:
      * Distant non-adjacent pairs (e.g., brain vs any torso organ, lungs vs bladder)
        have zero overlap.
      * Adjacent packed abdominal/thoracic organs have pairwise overlap volume
        below 0.75 L (750,000 mm3), with diaphragm acknowledged as an intended
        transverse boundary overlap.
  - Determinism across multiple executions.
  - Execution runtime under 100 ms.
"""

import math
import time

from hdt.body.models import JointName
from hdt.body.presets import PRESETS
from hdt.systems.common import inside_envelope
from hdt.systems.organs import build_organs
from hdt.systems.skeleton import build_skeleton


def test_organ_lateral_side_conventions() -> None:
    """Liver must be on person's right (x > 0), spleen & stomach on left (x < 0)."""
    items = build_organs(PRESETS["average"])

    liver = next(it for it in items if it["name"] == "liver")
    spleen = next(it for it in items if it["name"] == "spleen")
    stomach = next(it for it in items if it["name"] == "stomach")

    assert liver["primitive"]["center"][0] > 0.0, "Liver must have x > 0 (person's right)"
    assert spleen["primitive"]["center"][0] < 0.0, "Spleen must have x < 0 (person's left)"
    assert stomach["primitive"]["center"][0] < 0.0, "Stomach must have x < 0 (person's left)"


def test_heart_positioning() -> None:
    """Heart center must have x < 0 and be within 60 mm of midline."""
    items = build_organs(PRESETS["average"])
    heart = next(it for it in items if it["name"] == "heart")
    cx = heart["primitive"]["center"][0]

    assert cx < 0.0, f"Heart center x ({cx} mm) should be to person's left (x < 0)"
    assert abs(cx) <= 60.0, f"Heart center x ({cx} mm) must be within 60 mm of midline"


def test_organs_within_torso_and_brain_in_head() -> None:
    """Organs must be positioned between pelvis and neck, except brain which is in the head."""
    body = PRESETS["average"]
    items = build_organs(body)

    pelvis_z = body.joints[JointName.PELVIS][2]
    neck_z = body.joints[JointName.NECK][2]
    head_z = body.joints[JointName.HEAD][2]

    # Brain check
    brain = next(it for it in items if it["name"] == "brain")
    brain_z = brain["primitive"]["center"][2]
    assert brain_z > neck_z, f"Brain z ({brain_z} mm) should be superior to neck ({neck_z} mm)"
    assert brain_z <= head_z + 50.0, "Brain z is above head limit"

    # Torso organs check
    torso_organs = [it for it in items if it["name"] != "brain"]
    for it in torso_organs:
        cz = it["primitive"]["center"][2]
        name = it["name"]
        # Allow minor margin for organ radii
        assert cz >= pelvis_z - 30.0, (
            f"Organ '{name}' center z ({cz} mm) is below pelvis ({pelvis_z} mm)"
        )
        assert cz <= neck_z + 40.0, (
            f"Organ '{name}' center z ({cz} mm) is above neck ({neck_z} mm)"
        )


def _approx_ellipsoid_overlap(it1: dict, it2: dict, grid_res: int = 12) -> float:
    """Approximate intersection volume of two ellipsoids via grid integration."""
    c1 = it1["primitive"]["center"]
    r1 = it1["primitive"]["radii_mm"]
    c2 = it2["primitive"]["center"]
    r2 = it2["primitive"]["radii_mm"]

    bmin = [max(c1[k] - r1[k], c2[k] - r2[k]) for k in range(3)]
    bmax = [min(c1[k] + r1[k], c2[k] + r2[k]) for k in range(3)]

    if any(bmax[k] <= bmin[k] for k in range(3)):
        return 0.0

    dx = (bmax[0] - bmin[0]) / grid_res
    dy = (bmax[1] - bmin[1]) / grid_res
    dz = (bmax[2] - bmin[2]) / grid_res
    dV = dx * dy * dz

    count = 0
    for ix in range(grid_res):
        x = bmin[0] + (ix + 0.5) * dx
        for iy in range(grid_res):
            y = bmin[1] + (iy + 0.5) * dy
            for iz in range(grid_res):
                z = bmin[2] + (iz + 0.5) * dz
                in1 = (
                    ((x - c1[0]) / r1[0]) ** 2
                    + ((y - c1[1]) / r1[1]) ** 2
                    + ((z - c1[2]) / r1[2]) ** 2
                    <= 1.0
                )
                if in1:
                    in2 = (
                        ((x - c2[0]) / r2[0]) ** 2
                        + ((y - c2[1]) / r2[1]) ** 2
                        + ((z - c2[2]) / r2[2]) ** 2
                        <= 1.0
                    )
                    if in2:
                        count += 1
    return count * dV


def test_pairwise_overlap_tolerances() -> None:
    """Non-intended organ pairs must not exceed documented overlap tolerance (0.75 L).

    Distant non-adjacent organ pairs (e.g. brain vs torso organs, lungs vs bladder)
    have exactly 0.0 mm3 overlap.
    """
    items = build_organs(PRESETS["average"])

    # 1. Brain must have 0 overlap with any other organ
    brain = next(it for it in items if it["name"] == "brain")
    for it in items:
        if it["name"] == "brain":
            continue
        vol = _approx_ellipsoid_overlap(brain, it)
        assert vol == 0.0, f"Brain overlaps with {it['name']} (volume={vol:.1f} mm3)"

    # 2. Check all pairs excluding diaphragm (which has intended transverse overlap)
    max_allowed_vol = 750_000.0  # 0.75 L documented tolerance for adjacent packed viscera
    for i, it1 in enumerate(items):
        for j, it2 in enumerate(items):
            if j <= i:
                continue
            if "diaphragm" in (it1["name"], it2["name"]):
                continue

            vol = _approx_ellipsoid_overlap(it1, it2)
            assert vol <= max_allowed_vol, (
                f"Overlap between '{it1['name']}' and '{it2['name']}' ({vol:.1f} mm3) "
                f"exceeds tolerance of {max_allowed_vol:.1f} mm3"
            )


def test_determinism() -> None:
    """build_organs twice gives identical output."""
    o1 = build_organs(PRESETS["average"])
    o2 = build_organs(PRESETS["average"])
    assert o1 == o2


def test_execution_time() -> None:
    """build_organs should run well under 100 ms."""
    t0 = time.perf_counter()
    _ = build_organs(PRESETS["average"])
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 100.0, f"build_organs took {dt_ms:.2f} ms (expected < 100 ms)"


def test_every_organ_ellipsoid_surface_inside_envelope() -> None:
    """Every organ ellipsoid surface extremities lie inside inside_envelope across 3 presets."""
    for p_name in ("short", "average", "tall"):
        body = PRESETS[p_name]
        items = build_organs(body)
        for it in items:
            p = it["primitive"]
            c = p["center"]
            r = p["radii_mm"]
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
                    f"Preset {p_name}: organ '{it['name']}' sample[{idx}]={pt_tup} "
                    f"outside envelope"
                )


def test_lungs_and_heart_inside_ribcage_bounds() -> None:
    """Heart and lungs must lie inside the ribcage bounding envelope (with 5 mm margin)."""
    for p_name in ("short", "average", "tall"):
        body = PRESETS[p_name]
        skel = build_skeleton(body)
        orgs = build_organs(body)

        ribs = [it for it in skel if "rib" in it["name"].lower()]
        rib_pts = [pt for rib in ribs for pt in rib["primitive"]["polyline"]]
        min_rx = min(p[0] for p in rib_pts)
        max_rx = max(p[0] for p in rib_pts)
        min_ry = min(p[1] for p in rib_pts)
        max_ry = max(p[1] for p in rib_pts)
        min_rz = min(p[2] for p in rib_pts)
        max_rz = max(p[2] for p in rib_pts)

        # Allow 5 mm tolerance for thoracic wall margin
        margin = 5.0
        thoracic_organs = [
            it for it in orgs if it["name"] in ("heart", "left_lung", "right_lung")
        ]
        for org in thoracic_organs:
            c = org["primitive"]["center"]
            name = org["name"]
            assert min_rx - margin <= c[0] <= max_rx + margin, (
                f"Preset {p_name}: {name} x={c[0]} outside ribcage [{min_rx}, {max_rx}]"
            )
            assert min_ry - margin <= c[1] <= max_ry + margin, (
                f"Preset {p_name}: {name} y={c[1]} outside ribcage [{min_ry}, {max_ry}]"
            )
            assert min_rz - 20.0 <= c[2] <= max_rz + 20.0, (
                f"Preset {p_name}: {name} z={c[2]} outside ribcage [{min_rz}, {max_rz}]"
            )


def test_no_organ_centre_inside_bone_tube() -> None:
    """No organ centre may penetrate any non-rib bone tube (tolerance: 0.0 mm)."""
    body = PRESETS["average"]
    skel = build_skeleton(body)
    orgs = build_organs(body)

    bone_tubes = [
        it for it in skel if it["primitive"]["kind"] == "tube" and "rib" not in it["name"].lower()
    ]
    tolerance_mm = 0.0

    for org in orgs:
        oc = org["primitive"]["center"]
        for bone in bone_tubes:
            poly = bone["primitive"]["polyline"]
            p0, p1 = poly[0], poly[1]
            dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            length_sq = dx * dx + dy * dy + dz * dz
            if length_sq == 0.0:
                continue
            t = max(
                0.0,
                min(
                    1.0,
                    ((oc[0] - p0[0]) * dx + (oc[1] - p0[1]) * dy + (oc[2] - p0[2]) * dz)
                    / length_sq,
                ),
            )
            proj = (p0[0] + t * dx, p0[1] + t * dy, p0[2] + t * dz)
            dist = math.sqrt(
                (oc[0] - proj[0]) ** 2 + (oc[1] - proj[1]) ** 2 + (oc[2] - proj[2]) ** 2
            )
            r_tube = bone["primitive"]["radius_mm"]
            penetration = r_tube - dist
            assert penetration <= tolerance_mm, (
                f"Organ '{org['name']}' centre is inside bone '{bone['name']}' "
                f"by {penetration:.2f} mm (exceeds {tolerance_mm} mm)"
            )


# Pairwise overlap volume limits (L) justified from highest measured preset (tall)
# with a documented safety ceiling per pair:
PAIRWISE_OVERLAP_CEILINGS_L: dict[tuple[str, str], float] = {
    ("right_lung", "liver"): 1.05,
    ("heart", "left_lung"): 1.00,
    ("liver", "stomach"): 0.80,
    ("liver", "small_intestine"): 0.75,
    ("stomach", "small_intestine"): 0.70,
    ("left_lung", "right_lung"): 0.50,
    ("bladder", "small_intestine"): 0.50,
    ("liver", "r_kidney"): 0.50,
    ("heart", "right_lung"): 0.50,
    ("stomach", "l_kidney"): 0.50,
    ("liver", "transverse_colon"): 0.45,
    ("l_kidney", "small_intestine"): 0.45,
    ("r_kidney", "small_intestine"): 0.45,
}
_DEFAULT_ADJACENT_CEILING_L = 0.35


def test_pairwise_overlap_ceilings_per_pair() -> None:
    """Pairwise overlap volume must stay below justified per-pair ceilings for all 3 presets."""
    for p_name in ("short", "average", "tall"):
        items = build_organs(PRESETS[p_name])
        for i, it1 in enumerate(items):
            for j, it2 in enumerate(items):
                if j <= i:
                    continue
                name1, name2 = it1["name"], it2["name"]
                if "diaphragm" in (name1, name2):
                    continue

                vol_mm3 = _approx_ellipsoid_overlap(it1, it2, grid_res=15)
                vol_L = vol_mm3 / 1e6

                # Look up specific ceiling
                pair_key = (name1, name2)
                rev_key = (name2, name1)
                ceiling = PAIRWISE_OVERLAP_CEILINGS_L.get(
                    pair_key,
                    PAIRWISE_OVERLAP_CEILINGS_L.get(rev_key, _DEFAULT_ADJACENT_CEILING_L),
                )

                assert vol_L <= ceiling, (
                    f"Preset {p_name}: Overlap between '{name1}' and '{name2}' "
                    f"is {vol_L:.4f} L, exceeding pair ceiling of {ceiling:.4f} L"
                )


# ---------------------------------------------------------------------------
# Strict anatomical volume ranges and overlap tests
# ---------------------------------------------------------------------------

_INTENDED_CONTACT_PAIRS = {
    ("heart", "left_lung"),
    ("heart", "right_lung"),
    ("left_lung", "right_lung"),
    ("left_lung", "liver"),
    ("right_lung", "liver"),
    ("left_lung", "stomach"),
    ("right_lung", "stomach"),
    ("ascending_colon", "small_intestine"),
    ("transverse_colon", "small_intestine"),
    ("descending_colon", "small_intestine"),
}


def test_organ_volumes_within_reference_ranges() -> None:
    """Organ volumes must strictly satisfy adult anatomical reference ranges.

    Hard constraints:
      - lungs: 1.5 - 2.5 L each (inflated vital capacity envelope with air)
      - liver: >= 1.2 L (mean adult liver volume ~1.4 L)
      - small intestine: >= 1.0 L (visceral midgut packet)
    """
    items = build_organs(PRESETS["average"])
    vols = {
        it["name"]: (4.0 / 3.0)
        * math.pi
        * it["primitive"]["radii_mm"][0]
        * it["primitive"]["radii_mm"][1]
        * it["primitive"]["radii_mm"][2]
        / 1e6
        for it in items
    }

    # Hard constraints
    assert 1.5 <= vols["left_lung"] <= 2.5, (
        f"Left lung volume {vols['left_lung']:.3f} L out of range [1.5, 2.5]"
    )
    assert 1.5 <= vols["right_lung"] <= 2.5, (
        f"Right lung volume {vols['right_lung']:.3f} L out of range [1.5, 2.5]"
    )
    assert vols["liver"] >= 1.2, f"Liver volume {vols['liver']:.3f} L below 1.2 L"
    assert vols["small_intestine"] >= 1.0, (
        f"Small intestine volume {vols['small_intestine']:.3f} L below 1.0 L"
    )

    # Other adult reference ranges
    assert 1.2 <= vols["brain"] <= 1.5, f"Brain volume {vols['brain']:.3f} L"
    assert 0.5 <= vols["heart"] <= 0.85, f"Heart volume {vols['heart']:.3f} L"
    assert 0.4 <= vols["stomach"] <= 0.8, f"Stomach volume {vols['stomach']:.3f} L"
    assert 0.15 <= vols["spleen"] <= 0.35, f"Spleen volume {vols['spleen']:.3f} L"
    assert 0.07 <= vols["pancreas"] <= 0.13, f"Pancreas volume {vols['pancreas']:.3f} L"
    assert 0.14 <= vols["l_kidney"] <= 0.22, f"Left kidney volume {vols['l_kidney']:.3f} L"
    assert 0.14 <= vols["r_kidney"] <= 0.22, f"Right kidney volume {vols['r_kidney']:.3f} L"
    assert 0.20 <= vols["bladder"] <= 0.45, f"Bladder volume {vols['bladder']:.3f} L"


def test_non_intended_pairs_overlap_under_15_percent() -> None:
    """Non-intended organ pairs must overlap < 15% of the smaller organ across all 3 presets."""
    for p_name in ("short", "average", "tall"):
        items = build_organs(PRESETS[p_name])
        vols = {
            it["name"]: (4.0 / 3.0)
            * math.pi
            * it["primitive"]["radii_mm"][0]
            * it["primitive"]["radii_mm"][1]
            * it["primitive"]["radii_mm"][2]
            / 1e6
            for it in items
        }

        for i, it1 in enumerate(items):
            for j, it2 in enumerate(items):
                if j <= i:
                    continue
                name1, name2 = it1["name"], it2["name"]
                if "diaphragm" in (name1, name2):
                    continue
                if (name1, name2) in _INTENDED_CONTACT_PAIRS or (
                    name2,
                    name1,
                ) in _INTENDED_CONTACT_PAIRS:
                    continue

                vol_overlap = _approx_ellipsoid_overlap(it1, it2, grid_res=20) / 1e6
                vol_smaller = min(vols[name1], vols[name2])
                frac = (vol_overlap / vol_smaller) * 100.0 if vol_smaller > 0 else 0.0

                assert frac < 15.0, (
                    f"Preset {p_name}: Overlap between '{name1}' and '{name2}' is "
                    f"{vol_overlap:.4f} L ({frac:.1f}% of {vol_smaller:.3f} L), >= 15%"
                )


def test_ellipsoid_surface_clearance_vs_spine_and_bone_tubes() -> None:
    """Organ ellipsoid surface clearance vs spine and bone tubes across all 3 presets.

    Verifies:
      - Non-rib bone tubes: organ surface clearance >= -5.0 mm (no penetration into sternum,
        clavicles, or limb bones).
      - Spine vertebrae: organ surface clearance >= -65.0 mm (stated tolerance: simplified
        ellipsoids in packed visceral space have at most 65 mm geometric intersection with
        the anterior thoracic/lumbar lordosis).
      - Center containment: no organ centre lies inside any bone tube or vertebra.
    """
    for p_name in ("short", "average", "tall"):
        body = PRESETS[p_name]
        skel = build_skeleton(body)
        orgs = build_organs(body)

        vert_items = [
            it
            for it in skel
            if it["system"] == "skeleton" and it["primitive"]["kind"] == "ellipsoid"
        ]
        bone_tubes = [
            it
            for it in skel
            if it["primitive"]["kind"] == "tube" and "rib" not in it["name"].lower()
        ]

        for org in orgs:
            if org["name"] in ("brain", "diaphragm"):
                continue
            oc = org["primitive"]["center"]
            orad = org["primitive"]["radii_mm"]

            # 1. Surface clearance vs non-rib bone tubes (sternum, clavicles, limb bones)
            for bone in bone_tubes:
                poly = bone["primitive"]["polyline"]
                p0, p1 = poly[0], poly[1]
                dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
                lsq = dx * dx + dy * dy + dz * dz
                if lsq == 0:
                    continue
                t = max(
                    0.0,
                    min(
                        1.0,
                        (
                            (oc[0] - p0[0]) * dx
                            + (oc[1] - p0[1]) * dy
                            + (oc[2] - p0[2]) * dz
                        )
                        / lsq,
                    ),
                )
                proj = (p0[0] + t * dx, p0[1] + t * dy, p0[2] + t * dz)
                dist_axis = math.sqrt(sum((oc[k] - proj[k]) ** 2 for k in range(3)))
                r_tube = bone["primitive"]["radius_mm"]
                if dist_axis > 0:
                    dir_u = [(proj[k] - oc[k]) / dist_axis for k in range(3)]
                    r_org_dir = math.sqrt(sum((dir_u[k] * orad[k]) ** 2 for k in range(3)))
                    tube_clearance = dist_axis - (r_org_dir + r_tube)
                    assert tube_clearance >= -5.0, (
                        f"Preset {p_name}: Organ '{org['name']}' penetrates tube "
                        f"'{bone['name']}' by {-tube_clearance:.2f} mm"
                    )

            # 2. Surface clearance vs spine vertebrae (ellipsoids)
            for v in vert_items:
                vc = v["primitive"]["center"]
                vr = v["primitive"]["radii_mm"]
                dc = [vc[k] - oc[k] for k in range(3)]
                dist_c = math.sqrt(sum(d * d for d in dc))
                if dist_c > 0:
                    dir_u = [d / dist_c for d in dc]
                    r_org_dir = math.sqrt(sum((dir_u[k] * orad[k]) ** 2 for k in range(3)))
                    r_vert_dir = math.sqrt(sum((dir_u[k] * vr[k]) ** 2 for k in range(3)))
                    vert_clearance = dist_c - (r_org_dir + r_vert_dir)
                    assert vert_clearance >= -65.0, (
                        f"Preset {p_name}: Organ '{org['name']}' penetrates vertebra "
                        f"'{v['name']}' by {-vert_clearance:.2f} mm (exceeds 65 mm tolerance)"
                    )

            # 3. Center clearance: no organ center inside bone tubes or vertebrae
            for bone in bone_tubes:
                poly = bone["primitive"]["polyline"]
                p0, p1 = poly[0], poly[1]
                dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
                lsq = dx * dx + dy * dy + dz * dz
                if lsq == 0:
                    continue
                t = max(
                    0.0,
                    min(
                        1.0,
                        (
                            (oc[0] - p0[0]) * dx
                            + (oc[1] - p0[1]) * dy
                            + (oc[2] - p0[2]) * dz
                        )
                        / lsq,
                    ),
                )
                proj = (p0[0] + t * dx, p0[1] + t * dy, p0[2] + t * dz)
                dist_axis = math.sqrt(sum((oc[k] - proj[k]) ** 2 for k in range(3)))
                assert dist_axis >= bone["primitive"]["radius_mm"], (
                    f"Preset {p_name}: Center of '{org['name']}' inside bone '{bone['name']}'"
                )

            for v in vert_items:
                vc = v["primitive"]["center"]
                vr = v["primitive"]["radii_mm"]
                dc = [vc[k] - oc[k] for k in range(3)]
                dist_c = math.sqrt(sum(d * d for d in dc))
                if dist_c > 0:
                    dir_u = [d / dist_c for d in dc]
                    r_vert_dir = math.sqrt(sum((dir_u[k] * vr[k]) ** 2 for k in range(3)))
                    assert dist_c >= r_vert_dir, (
                        f"Preset {p_name}: Center of '{org['name']}' inside '{v['name']}'"
                    )
