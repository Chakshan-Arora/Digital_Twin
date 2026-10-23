"""3D neural lesion localization and ascending pathway interruption.

Maps clinical conditions to focal 3D lesion sites on the real nerve network,
computes downstream nerve effects, and evaluates signal blockage along
ascending somatosensory trajectories.

Research prototype - NOT a clinical diagnostic tool.
"""

from __future__ import annotations

import math
from typing import Literal

from hdt.atlas.data import get_default_atlas
from hdt.atlas.models import SPINAL_CORD_LEVELS, Condition
from hdt.body.models import BodyParams, JointName
from hdt.body.shape import surface_distance_mm
from hdt.conditions.models import Lesion, NerveEffect, SignalStatus
from hdt.nervous.network import NerveNetworkResponse, build_network


def _distance(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Euclidean distance in mm."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _point_to_segment_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    p: tuple[float, float, float],
) -> float:
    """Minimum Euclidean distance from point p to segment [a, b] in mm."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    len_sq = dx * dx + dy * dy + dz * dz
    if len_sq < 1e-9:
        return _distance(a, p)

    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy + (p[2] - a[2]) * dz) / len_sq
    t_clamped = max(0.0, min(1.0, t))
    proj = (a[0] + t_clamped * dx, a[1] + t_clamped * dy, a[2] + t_clamped * dz)
    return _distance(proj, p)


def locate_lesions(
    body: BodyParams,
    condition: Condition | str,
    side: Literal["l", "r", "both"] = "r",
    network: NerveNetworkResponse | None = None,
) -> tuple[list[Lesion], list[NerveEffect]]:
    """Locate 3D lesion sites and determine affected downstream nerves on the network.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model in RAS mm.
    condition : Condition or str
        Condition instance or condition identifier (e.g. 'sciatica', 'carpal_tunnel').
    side : {'l', 'r', 'both'}
        Anatomical lesion lateralisation.
    network : NerveNetworkResponse | None
        Pre-built high-detail neural network (instantiated if None).

    Returns
    -------
    tuple[list[Lesion], list[NerveEffect]]
        (lesions, nerve_effects) where each lesion is guaranteed to lie inside
        the body envelope, and nerve_effects list affected downstream nerves.
    """
    if isinstance(condition, str):
        atlas = get_default_atlas()
        cond_obj = next((c for c in atlas.conditions if c.id == condition), None)
        if cond_obj is None:
            raise ValueError(f"Unknown condition '{condition}'")
        condition = cond_obj

    if network is None:
        network = build_network(body, detail="high")

    lesions: list[Lesion] = []
    nerve_effects: list[NerveEffect] = []

    target_sides: list[Literal["l", "r"]] = (
        ["l", "r"] if side == "both" or condition.id == "sci_t10" else [side]
    )

    cond_id = condition.id

    # -----------------------------------------------------------------------
    # 1. Complete Spinal Cord Injury (e.g. sci_t10)
    # -----------------------------------------------------------------------
    if cond_id == "sci_t10":
        # Bilateral cord transection at T10
        # Find T10 root origin coordinate on spinal cord polyline
        r_t10 = next((p for p in network.spinal_roots if p.name == "root_T10_r"), None)
        if r_t10 and r_t10.points:
            t10_origin = r_t10.points[0]
        else:
            s_scale = body.height_mm / 1750.0
            t10_origin = (0.0, -45.0 * s_scale, 1067.0 * s_scale)

        # Closest point on smooth cord polyline
        best_cord_pt = min(network.spinal_cord, key=lambda pt: _distance(pt, t10_origin))
        lesion_pt = (
            round(best_cord_pt[0], 2),
            round(best_cord_pt[1], 2),
            round(best_cord_pt[2], 2),
        )

        lesions.append(
            Lesion(
                kind="cord_transection",
                label="T10 Spinal Cord Transection",
                point=lesion_pt,
                cord_level="cord_T10",
                side="both",
                radius_mm=16.0,
            )
        )

        # Downstream nerves: all roots and lumbosacral plexus nerves below T10
        cord_order = list(SPINAL_CORD_LEVELS)  # S5 -> C1
        t10_idx = cord_order.index("cord_T10")
        below_t10_levels = {lvl.replace("cord_", "") for lvl in cord_order[: t10_idx + 1]}

        for root_poly in network.spinal_roots:
            root_id = root_poly.name.split("_")[1]
            if root_id in below_t10_levels:
                nerve_effects.append(
                    NerveEffect(
                        nerve_id=root_poly.name,
                        side=root_poly.side,
                        status=SignalStatus.BLOCKED,
                        lesion_index=0,
                        lesion_point=root_poly.points[0] if root_poly.points else lesion_pt,
                    )
                )

        # All lumbosacral plexus nerves are caudal to T10
        for lumb_poly in network.lumbosacral_plexus:
            nerve_effects.append(
                NerveEffect(
                    nerve_id=lumb_poly.name,
                    side=lumb_poly.side,
                    status=SignalStatus.BLOCKED,
                    lesion_index=0,
                    lesion_point=lumb_poly.points[0] if lumb_poly.points else lesion_pt,
                )
            )

    # -----------------------------------------------------------------------
    # 2. Peripheral Entrapment: Carpal Tunnel Syndrome
    # -----------------------------------------------------------------------
    elif cond_id == "carpal_tunnel":
        for s in target_sides:
            median_nerve = next(
                (p for p in network.brachial_plexus if p.name == f"nerve_median_{s}"), None
            )
            wrist_joint = body.joints[JointName(f"{s}_wrist")]

            if median_nerve and len(median_nerve.points) > 1:
                # Find index along median nerve closest to the wrist joint
                wrist_idx = min(
                    range(len(median_nerve.points)),
                    key=lambda i: _distance(median_nerve.points[i], wrist_joint),
                )
                lesion_pt = median_nerve.points[wrist_idx]
            else:
                wrist_idx = 0
                lesion_pt = wrist_joint

            lesions.append(
                Lesion(
                    kind="peripheral_entrapment",
                    label=f"Carpal Tunnel Median Nerve Entrapment ({s.upper()})",
                    point=lesion_pt,
                    nerve_id=f"nerve_median_{s}",
                    side=s,
                    radius_mm=10.0,
                )
            )

            nerve_effects.append(
                NerveEffect(
                    nerve_id=f"nerve_median_{s}",
                    side=s,
                    status=SignalStatus.SLOWED,
                    lesion_index=wrist_idx,
                    lesion_point=lesion_pt,
                )
            )

    # -----------------------------------------------------------------------
    # 3. Peripheral Entrapment: Meralgia Paresthetica
    # -----------------------------------------------------------------------
    elif cond_id == "meralgia_paresthetica":
        for s in target_sides:
            nerve_name = f"nerve_lat_femoral_cutaneous_{s}"
            lfcn = next((p for p in network.lumbosacral_plexus if p.name == nerve_name), None)
            hip_joint = body.joints[JointName(f"{s}_hip")]

            if lfcn and len(lfcn.points) > 1:
                # Inguinal entrapment near ASIS / hip level
                ing_idx = min(
                    range(len(lfcn.points)),
                    key=lambda i: abs(lfcn.points[i][2] - hip_joint[2]),
                )
                lesion_pt = lfcn.points[ing_idx]
            else:
                ing_idx = 0
                lesion_pt = hip_joint

            lesions.append(
                Lesion(
                    kind="peripheral_entrapment",
                    label=f"Lateral Femoral Cutaneous Nerve Entrapment ({s.upper()})",
                    point=lesion_pt,
                    nerve_id=f"nerve_lat_femoral_cutaneous_{s}",
                    side=s,
                    radius_mm=10.0,
                )
            )

            nerve_effects.append(
                NerveEffect(
                    nerve_id=f"nerve_lat_femoral_cutaneous_{s}",
                    side=s,
                    status=SignalStatus.SLOWED,
                    lesion_index=ing_idx,
                    lesion_point=lesion_pt,
                )
            )

    # -----------------------------------------------------------------------
    # 4. Spinal Root Compression: Sciatica, L4-L5 Herniation, Cervical Radiculopathy
    # -----------------------------------------------------------------------
    else:
        # Radiculopathies: lesion sits on spinal root polyline at intervertebral foramen
        for s in target_sides:
            for root_id in condition.compressed_roots:
                root_poly = next(
                    (p for p in network.spinal_roots if p.name == f"root_{root_id}_{s}"), None
                )

                if root_poly and len(root_poly.points) > 1:
                    # Index 1 is at the exit foramen close to the cord
                    foramen_idx = 1 if len(root_poly.points) >= 2 else 0
                    lesion_pt = root_poly.points[foramen_idx]
                else:
                    foramen_idx = 0
                    cord_map = network.spinal_cord
                    lesion_pt = cord_map[0] if cord_map else (0.0, 0.0, 0.0)

                lesions.append(
                    Lesion(
                        kind="root_compression",
                        label=f"{root_id} Spinal Nerve Root Compression ({s.upper()})",
                        point=lesion_pt,
                        cord_level=f"cord_{root_id}",
                        side=s,
                        radius_mm=12.0,
                    )
                )

                nerve_effects.append(
                    NerveEffect(
                        nerve_id=f"root_{root_id}_{s}",
                        side=s,
                        status=SignalStatus.SLOWED,
                        lesion_index=foramen_idx,
                        lesion_point=lesion_pt,
                    )
                )

            # Map downstream nerves for radiculopathies
            if cond_id in ("sciatica", "l4_l5_herniation"):
                downstream_names = [
                    f"nerve_sciatic_{s}",
                    f"nerve_tibial_{s}",
                    f"nerve_common_fibular_{s}",
                    f"nerve_deep_fibular_{s}",
                    f"nerve_superficial_fibular_{s}",
                ]
                for d_name in downstream_names:
                    d_poly = next(
                        (p for p in network.lumbosacral_plexus if p.name == d_name), None
                    )
                    nerve_effects.append(
                        NerveEffect(
                            nerve_id=d_name,
                            side=s,
                            status=SignalStatus.SLOWED,
                            lesion_index=0,  # Entire nerve is distal to root lesion
                            lesion_point=d_poly.points[0] if (d_poly and d_poly.points) else None,
                        )
                    )

            elif cond_id == "cervical_radiculopathy":
                downstream_arm_names = [
                    f"brachial_superior_trunk_{s}",
                    f"brachial_middle_trunk_{s}",
                    f"nerve_median_{s}",
                    f"nerve_radial_{s}",
                    f"nerve_musculocutaneous_{s}",
                ]
                for d_name in downstream_arm_names:
                    d_poly = next((p for p in network.brachial_plexus if p.name == d_name), None)
                    nerve_effects.append(
                        NerveEffect(
                            nerve_id=d_name,
                            side=s,
                            status=SignalStatus.SLOWED,
                            lesion_index=0,
                            lesion_point=d_poly.points[0] if (d_poly and d_poly.points) else None,
                        )
                    )

    # Ensure all lesion points are strictly inside or on the body envelope
    for les in lesions:
        s_dist = surface_distance_mm(body, les.point)
        if s_dist > 0.0:
            # Shift inward toward body axis if near skin border
            px, py, pz = les.point
            les.point = (round(px * 0.90, 2), round(py * 0.90, 2), pz)

    return lesions, nerve_effects


def path_block_index(
    pathway_points: list[tuple[float, float, float]]
    | list[tuple[str, tuple[float, float, float]]]
    | list[object],
    lesions: list[Lesion],
    tolerance_mm: float = 85.0,
) -> int | None:
    """Find the index of the first pathway waypoint at or beyond the first lesion.

    Evaluates the ascending sensory transmission path (from skin receptor through
    peripheral nerve, root, cord ascent, to cortex_S1) against active lesions.

    Parameters
    ----------
    pathway_points : list
        Sequence of 3D coordinates, labelled tuples `(label, (x, y, z))`, or
        `PathwayPoint` instances along the ascending trajectory.
    lesions : list[Lesion]
        Active focal lesions for the condition.
    tolerance_mm : float
        Spatial intersection threshold in mm (default 85 mm).

    Returns
    -------
    int | None
        Index of the first pathway point at or beyond the lesion along the
        ascending trajectory, or None if the pathway is unaffected.
    """
    if not pathway_points or not lesions:
        return None

    # Normalise pathway coordinates and labels
    pts: list[tuple[float, float, float]] = []
    labels: list[str | None] = []

    for item in pathway_points:
        if hasattr(item, "point_ras_mm"):
            # PathwayPoint model
            pts.append(item.point_ras_mm)
            labels.append(getattr(item, "label", None))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            labels.append(item[0])
            pts.append(item[1])
        elif (
            isinstance(item, (tuple, list))
            and len(item) == 3
            and isinstance(item[0], (int, float))
        ):
            pts.append((float(item[0]), float(item[1]), float(item[2])))
            labels.append(None)
        else:
            continue

    n_pts = len(pts)
    if n_pts < 2:
        return None

    # Ascending path orientation: pts[0] is receptor, pts[-1] is cortex_S1
    # Determine anatomical side of pathway from peripheral points
    path_side: Literal["l", "r"] = "l" if pts[0][0] < 0.0 else "r"

    candidate_indices: list[int] = []

    for lesion in lesions:
        # Check lateralisation
        if lesion.side != "both" and lesion.side != path_side:
            continue

        l_pt = lesion.point

        # 1. Label-based fast match for spinal cord transection
        if lesion.kind == "cord_transection" and lesion.cord_level:
            transection_lvl = lesion.cord_level  # 'cord_T10'
            if transection_lvl in SPINAL_CORD_LEVELS:
                transection_order_idx = SPINAL_CORD_LEVELS.index(transection_lvl)
                # Check if pathway contains a cord segment at or below the transection
                for idx, lbl in enumerate(labels):
                    if lbl and lbl in SPINAL_CORD_LEVELS:
                        lvl_order_idx = SPINAL_CORD_LEVELS.index(lbl)
                        if lvl_order_idx <= transection_order_idx:
                            # Reaches transection at cord_T10
                            # Block point is where it hits the transection level
                            block_idx = next(
                                (
                                    i
                                    for i, lbl_name in enumerate(labels)
                                    if lbl_name == transection_lvl
                                ),
                                idx,
                            )
                            candidate_indices.append(block_idx)
                            break

        # 2. Geometric intersection across polyline segments
        tol = max(lesion.radius_mm, tolerance_mm)
        for i in range(n_pts - 1):
            p_a = pts[i]
            p_b = pts[i + 1]
            seg_dist = _point_to_segment_distance(p_a, p_b, l_pt)

            if seg_dist <= tol:
                # If point a itself is already close, block at point a (if > 0) or b
                if i > 0 and _distance(p_a, l_pt) <= tol:
                    candidate_indices.append(i)
                else:
                    candidate_indices.append(i + 1)

    if not candidate_indices:
        return None

    # Earliest lesion encounter on the ascending path
    # Clamp so it is always strictly before the cortex_S1 terminal point
    earliest_idx = min(candidate_indices)
    earliest_idx = max(1, min(earliest_idx, n_pts - 2))

    return earliest_idx
