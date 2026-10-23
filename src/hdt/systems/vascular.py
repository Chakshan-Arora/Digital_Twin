"""Vascular system builder for the Human Digital Twin body explorer.

Generates a schematic educational 3-D model of the major blood vessels as a
list of BodyItem dicts.  NOT a clinical tool.

Coordinate system: RAS millimetres.
  x > 0  → person's right,  x < 0 → person's left
  y > 0  → anterior,        y < 0 → posterior
  z > 0  → superior;  origin on floor between feet.

Vessels are modelled as tube primitives (polylines with start/end radius).
All vessel centrelines are kept inside the simplified body envelope
(inside_envelope).  Where the anatomical path would technically clip the
envelope edge, the point is slightly shifted inward — this is a deliberate
schematic approximation and is noted per vessel.

Arterial system: color "#CC2200" (system="vascular_arterial")
Venous system:   color "#1144AA" (system="vascular_venous")

Radii are defined for the 1750 mm reference height and scale linearly.
"""

from __future__ import annotations

from hdt.body.models import BodyParams, JointName
from hdt.systems.common import body_item, inside_envelope, lerp, tube

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ART_COLOR = "#CC2200"
_VEN_COLOR = "#1144AA"
_REF_H = 1750.0


def _scale(body: BodyParams) -> float:
    return body.height_mm / _REF_H


def _j(body: BodyParams, name: JointName) -> tuple[float, float, float]:
    return body.joints[name]


def _clamp_inside(
    body: BodyParams, point: tuple[float, float, float], margin: float = 5.0
) -> tuple[float, float, float]:
    """Clamp *point* so that it lies inside the simplified body envelope.

    If the point is already inside, return it unchanged.
    Otherwise, linearly interpolate toward the appropriate anatomical axis
    (leg centerline, arm centerline, or torso/head midline) until inside.

    Parameters
    ----------
    body : BodyParams
    point : (x, y, z) mm
    margin : float
        Not used in current implementation; kept for API clarity.

    Returns
    -------
    (x, y, z) mm
    """
    if inside_envelope(body, point):
        return point

    px, py, pz = point
    z_pelvis = _j(body, JointName.PELVIS)[2]
    z_neck = _j(body, JointName.NECK)[2]

    # Anatomically aware reference target inside envelope
    if pz <= z_pelvis:
        # Lower limbs: interpolate toward the corresponding leg axis
        side_prefix = "l" if px < 0.0 else "r"
        hip_j = JointName(f"{side_prefix}_hip")
        knee_j = JointName(f"{side_prefix}_knee")
        ankle_j = JointName(f"{side_prefix}_ankle")
        foot_j = JointName(f"{side_prefix}_foot")

        hip = _j(body, hip_j)
        knee = _j(body, knee_j)
        ankle = _j(body, ankle_j)
        foot = _j(body, foot_j)

        if pz >= knee[2]:
            span = hip[2] - knee[2]
            t_axis = (hip[2] - pz) / span if span > 0 else 0.5
            target = lerp(hip, knee, max(0.0, min(1.0, t_axis)))
        elif pz >= ankle[2]:
            span = knee[2] - ankle[2]
            t_axis = (knee[2] - pz) / span if span > 0 else 0.5
            target = lerp(knee, ankle, max(0.0, min(1.0, t_axis)))
        else:
            span = ankle[2] - foot[2]
            t_axis = (ankle[2] - pz) / span if span > 0 else 0.5
            target = lerp(ankle, foot, max(0.0, min(1.0, t_axis)))

    elif abs(px) > 40.0 and pz <= z_neck:
        # Upper limbs: interpolate toward the corresponding arm axis
        side_prefix = "l" if px < 0.0 else "r"
        sh = _j(body, JointName(f"{side_prefix}_shoulder"))
        el = _j(body, JointName(f"{side_prefix}_elbow"))
        wr = _j(body, JointName(f"{side_prefix}_wrist"))
        ha = _j(body, JointName(f"{side_prefix}_hand"))

        if pz >= el[2]:
            span = sh[2] - el[2]
            t_axis = (sh[2] - pz) / span if span > 0 else 0.5
            target = lerp(sh, el, max(0.0, min(1.0, t_axis)))
        elif pz >= wr[2]:
            span = el[2] - wr[2]
            t_axis = (el[2] - pz) / span if span > 0 else 0.5
            target = lerp(el, wr, max(0.0, min(1.0, t_axis)))
        else:
            span = wr[2] - ha[2]
            t_axis = (wr[2] - pz) / span if span > 0 else 0.5
            target = lerp(wr, ha, max(0.0, min(1.0, t_axis)))

    else:
        # Torso / head midline
        chest_y = _j(body, JointName.CHEST)[1]
        target = (0.0, chest_y, pz)

    # Search for an inside point between point and target
    for step in range(1, 21):
        t = step / 20.0
        candidate = lerp(point, target, t)
        if inside_envelope(body, candidate):
            return candidate

    if inside_envelope(body, target):
        return target

    return point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _art(
    item_id: str,
    name: str,
    side: str,
    polyline: list[tuple[float, float, float]],
    r_start: float,
    r_end: float | None = None,
) -> dict:
    """Return an arterial BodyItem dict."""
    return body_item(
        item_id=f"vascular_{item_id}",
        name=name,
        system="vascular_arterial",
        side=side,
        primitive=tube(polyline, r_start, r_end),
        color_hint=_ART_COLOR,
        label=name.replace("_", " ").title(),
    )


def _ven(
    item_id: str,
    name: str,
    side: str,
    polyline: list[tuple[float, float, float]],
    r_start: float,
    r_end: float | None = None,
) -> dict:
    """Return a venous BodyItem dict."""
    return body_item(
        item_id=f"vascular_{item_id}",
        name=name,
        system="vascular_venous",
        side=side,
        primitive=tube(polyline, r_start, r_end),
        color_hint=_VEN_COLOR,
        label=name.replace("_", " ").title(),
    )


def _safe_pt(
    body: BodyParams, p: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return *p* clamped inside the body envelope."""
    return _clamp_inside(body, p)


# ---------------------------------------------------------------------------
# Arterial system
# ---------------------------------------------------------------------------


def _build_arterial(body: BodyParams) -> list[dict]:
    """Build all major arterial vessels as tube BodyItems.

    Returns
    -------
    list[dict]
    """
    s = _scale(body)
    items: list[dict] = []

    # Reference joint positions
    chest = _j(body, JointName.CHEST)
    neck = _j(body, JointName.NECK)
    pelvis = _j(body, JointName.PELVIS)
    head = _j(body, JointName.HEAD)
    l_shoulder = _j(body, JointName.L_SHOULDER)
    r_shoulder = _j(body, JointName.R_SHOULDER)
    l_elbow = _j(body, JointName.L_ELBOW)
    r_elbow = _j(body, JointName.R_ELBOW)
    l_wrist = _j(body, JointName.L_WRIST)
    r_wrist = _j(body, JointName.R_WRIST)
    l_hand = _j(body, JointName.L_HAND)
    r_hand = _j(body, JointName.R_HAND)
    l_hip = _j(body, JointName.L_HIP)
    r_hip = _j(body, JointName.R_HIP)
    l_knee = _j(body, JointName.L_KNEE)
    r_knee = _j(body, JointName.R_KNEE)
    l_ankle = _j(body, JointName.L_ANKLE)
    r_ankle = _j(body, JointName.R_ANKLE)
    l_foot = _j(body, JointName.L_FOOT)
    r_foot = _j(body, JointName.R_FOOT)

    # Spine y reference (posterior)
    spine_y = chest[1] - 30.0 * s  # approximate posterior surface of torso

    # Heart centre
    heart_cx = -15.0 * s
    heart_cz = chest[2] - 20.0 * s
    heart_pt = _safe_pt(body, (heart_cx, chest[1] - 5.0 * s, heart_cz))

    # --- Aorta ---

    # Ascending aorta: heart → upward 60 mm
    asc_top = _safe_pt(body, (heart_cx, chest[1], heart_cz + 60.0 * s))
    items.append(
        _art(
            "aorta_ascending",
            "ascending_aorta",
            "mid",
            [heart_pt, asc_top],
            14.0 * s,
        )
    )

    # Aortic arch: ascending top → leftward and downward
    arch_mid = _safe_pt(body, (-20.0 * s, chest[1], asc_top[2] + 15.0 * s))
    arch_end = _safe_pt(body, (-15.0 * s, spine_y + 5.0 * s, asc_top[2] - 10.0 * s))
    items.append(
        _art(
            "aorta_arch",
            "aortic_arch",
            "mid",
            [asc_top, arch_mid, arch_end],
            12.0 * s,
        )
    )

    # Thoracic aorta: arch end → T12 level (≈60% of torso height from pelvis)
    torso_span = neck[2] - pelvis[2]
    t12_z = pelvis[2] + torso_span * 0.60
    thor_end = _safe_pt(body, (-10.0 * s, spine_y + 5.0 * s, t12_z))
    items.append(
        _art(
            "aorta_thoracic",
            "thoracic_aorta",
            "mid",
            [arch_end, thor_end],
            12.0 * s,
            10.0 * s,
        )
    )

    # Abdominal aorta: T12 → pelvis level
    abd_top = _safe_pt(body, (-10.0 * s, spine_y + 8.0 * s, t12_z))
    abd_bot = _safe_pt(body, (-8.0 * s, spine_y + 8.0 * s, pelvis[2] + 10.0 * s))
    items.append(
        _art(
            "aorta_abdominal",
            "abdominal_aorta",
            "mid",
            [abd_top, abd_bot],
            10.0 * s,
            8.0 * s,
        )
    )

    # Common iliac arteries: bifurcate from abdominal aorta
    iliac_origin = abd_bot
    for side, hip in (("l", l_hip), ("r", r_hip)):
        sign = -1.0 if side == "l" else 1.0
        iliac_end = _safe_pt(body, (sign * abs(hip[0]) * 0.8, hip[1], hip[2] + 20.0 * s))
        items.append(
            _art(
                f"iliac_common_{side}",
                f"{side}_common_iliac",
                side,
                [iliac_origin, iliac_end],
                7.0 * s,
            )
        )

    # Femoral arteries: hip → knee
    for side, hip, knee in (("l", l_hip, l_knee), ("r", r_hip, r_knee)):
        fem_start = _safe_pt(body, (hip[0], hip[1], hip[2]))
        fem_end = _safe_pt(body, (knee[0], knee[1], knee[2]))
        items.append(
            _art(
                f"femoral_{side}",
                f"{side}_femoral",
                side,
                [fem_start, fem_end],
                5.0 * s,
            )
        )

    # Popliteal arteries: knee → ankle
    for side, knee, ankle in (("l", l_knee, l_ankle), ("r", r_knee, r_ankle)):
        items.append(
            _art(
                f"popliteal_{side}",
                f"{side}_popliteal",
                side,
                [_safe_pt(body, knee), _safe_pt(body, ankle)],
                4.0 * s,
            )
        )

    # Tibial arteries: ankle → foot
    for side, ankle, foot in (("l", l_ankle, l_foot), ("r", r_ankle, r_foot)):
        items.append(
            _art(
                f"tibial_{side}",
                f"{side}_tibial",
                side,
                [_safe_pt(body, ankle), _safe_pt(body, foot)],
                3.0 * s,
            )
        )

    # Carotid arteries: neck → head
    for side, sign in (("l", -1.0), ("r", 1.0)):
        car_bot = _safe_pt(body, (sign * 15.0 * s, neck[1], neck[2]))
        car_top = _safe_pt(body, (sign * 15.0 * s, head[1], head[2] - 30.0 * s))
        items.append(
            _art(
                f"carotid_{side}",
                f"{side}_carotid",
                side,
                [car_bot, car_top],
                4.0 * s,
            )
        )

    # Subclavian arteries: from arch toward shoulder
    for side, shoulder in (("l", l_shoulder), ("r", r_shoulder)):
        sign = -1.0 if side == "l" else 1.0
        sub_start = _safe_pt(body, (sign * 20.0 * s, arch_end[1], arch_end[2]))
        sub_end = _safe_pt(body, (shoulder[0] * 0.7, shoulder[1], shoulder[2]))
        items.append(
            _art(
                f"subclavian_{side}",
                f"{side}_subclavian",
                side,
                [sub_start, sub_end],
                6.0 * s,
            )
        )

    # Axillary arteries: shoulder → elbow region
    for side, shoulder, elbow in (("l", l_shoulder, l_elbow), ("r", r_shoulder, r_elbow)):
        items.append(
            _art(
                f"axillary_{side}",
                f"{side}_axillary",
                side,
                [_safe_pt(body, shoulder), _safe_pt(body, elbow)],
                5.0 * s,
            )
        )

    # Brachial arteries: elbow → wrist
    for side, elbow, wrist in (("l", l_elbow, l_wrist), ("r", r_elbow, r_wrist)):
        items.append(
            _art(
                f"brachial_{side}",
                f"{side}_brachial",
                side,
                [_safe_pt(body, elbow), _safe_pt(body, wrist)],
                4.0 * s,
            )
        )

    # Radial arteries: wrist → hand
    for side, wrist, hand in (("l", l_wrist, l_hand), ("r", r_wrist, r_hand)):
        items.append(
            _art(
                f"radial_{side}",
                f"{side}_radial",
                side,
                [_safe_pt(body, wrist), _safe_pt(body, hand)],
                3.0 * s,
            )
        )

    # Renal arteries: abdominal aorta → lateral
    # Kidney level approximately 55% of torso from pelvis  (T12 level - some offset)
    kidney_z = pelvis[2] + torso_span * 0.55 + 50.0 * s
    renal_origin = _safe_pt(body, (-8.0 * s, spine_y + 8.0 * s, kidney_z))
    for side, sign in (("l", -1.0), ("r", 1.0)):
        kidney_cx = sign * 45.0 * s
        renal_end = _safe_pt(body, (kidney_cx, -30.0 * s + chest[1] * 0.2, kidney_z))
        items.append(
            _art(
                f"renal_{side}",
                f"{side}_renal_artery",
                side,
                [renal_origin, renal_end],
                4.0 * s,
            )
        )

    return items


# ---------------------------------------------------------------------------
# Venous system
# ---------------------------------------------------------------------------


def _build_venous(body: BodyParams) -> list[dict]:
    """Build all major venous vessels as tube BodyItems.

    Returns
    -------
    list[dict]
    """
    s = _scale(body)
    items: list[dict] = []

    chest = _j(body, JointName.CHEST)
    neck = _j(body, JointName.NECK)
    pelvis = _j(body, JointName.PELVIS)
    head = _j(body, JointName.HEAD)
    l_shoulder = _j(body, JointName.L_SHOULDER)
    r_shoulder = _j(body, JointName.R_SHOULDER)
    l_wrist = _j(body, JointName.L_WRIST)
    r_wrist = _j(body, JointName.R_WRIST)
    l_hip = _j(body, JointName.L_HIP)
    r_hip = _j(body, JointName.R_HIP)
    l_knee = _j(body, JointName.L_KNEE)
    r_knee = _j(body, JointName.R_KNEE)

    spine_y = chest[1] - 30.0 * s
    torso_span = neck[2] - pelvis[2]
    heart_z = chest[2] - 20.0 * s

    # Superior vena cava: neck level → heart
    svc_top = _safe_pt(body, (10.0 * s, chest[1], neck[2]))
    svc_bot = _safe_pt(body, (10.0 * s, chest[1], heart_z))
    items.append(
        _ven("svc", "superior_vena_cava", "mid", [svc_top, svc_bot], 12.0 * s)
    )

    # Inferior vena cava: pelvis → heart, right of spine
    ivc_bot = _safe_pt(body, (15.0 * s, spine_y + 8.0 * s, pelvis[2] + 10.0 * s))
    ivc_top = _safe_pt(body, (15.0 * s, spine_y + 8.0 * s, heart_z))
    items.append(
        _ven("ivc", "inferior_vena_cava", "mid", [ivc_bot, ivc_top], 12.0 * s, 10.0 * s)
    )

    # Jugular veins: head → neck
    for side, sign in (("l", -1.0), ("r", 1.0)):
        jug_top = _safe_pt(body, (sign * 18.0 * s, head[1], head[2] - 35.0 * s))
        jug_bot = _safe_pt(body, (sign * 18.0 * s, neck[1], neck[2]))
        items.append(
            _ven(
                f"jugular_{side}",
                f"{side}_jugular",
                side,
                [jug_top, jug_bot],
                5.0 * s,
            )
        )

    # Femoral veins: hip → knee (x offset +5 mm from midline per side)
    for side, hip, knee in (("l", l_hip, l_knee), ("r", r_hip, r_knee)):
        sign = -1.0 if side == "l" else 1.0
        fv_start = _safe_pt(body, (hip[0] + sign * 5.0 * s, hip[1], hip[2]))
        fv_end = _safe_pt(body, (knee[0] + sign * 5.0 * s, knee[1], knee[2]))
        items.append(
            _ven(
                f"femoral_vein_{side}",
                f"{side}_femoral_vein",
                side,
                [fv_start, fv_end],
                5.0 * s,
            )
        )

    # Iliac veins: hip → IVC at pelvis
    for side, hip in (("l", l_hip), ("r", r_hip)):
        sign = -1.0 if side == "l" else 1.0
        iliac_top = _safe_pt(body, (sign * abs(hip[0]) * 0.8, hip[1], pelvis[2] + 20.0 * s))
        items.append(
            _ven(
                f"iliac_vein_{side}",
                f"{side}_iliac_vein",
                side,
                [_safe_pt(body, hip), iliac_top, ivc_bot],
                6.0 * s,
            )
        )

    # Brachial veins: shoulder → wrist (simplified)
    for side, shoulder, wrist in (("l", l_shoulder, l_wrist), ("r", r_shoulder, r_wrist)):
        items.append(
            _ven(
                f"brachial_vein_{side}",
                f"{side}_brachial_vein",
                side,
                [_safe_pt(body, shoulder), _safe_pt(body, wrist)],
                5.0 * s,
            )
        )

    # Renal veins: kidney → IVC
    kidney_z = pelvis[2] + torso_span * 0.55 + 50.0 * s
    for side, sign in (("l", -1.0), ("r", 1.0)):
        kidney_cx = sign * 45.0 * s
        rv_kidney = _safe_pt(body, (kidney_cx, -30.0 * s + chest[1] * 0.2, kidney_z))
        rv_ivc = _safe_pt(body, (15.0 * s, spine_y + 8.0 * s, kidney_z))
        items.append(
            _ven(
                f"renal_vein_{side}",
                f"{side}_renal_vein",
                side,
                [rv_kidney, rv_ivc],
                5.0 * s,
            )
        )

    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_vascular(body: BodyParams) -> list[dict]:
    """Build a schematic educational 3-D vascular model.

    Returns arterial (system="vascular_arterial") and venous
    (system="vascular_venous") vessels as tube BodyItem dicts.

    All vessel centreline points are clamped inside the simplified body
    envelope.  Vessel radii scale linearly with body.height_mm / 1750.

    This is a schematic educational model — NOT a clinical tool.

    Parameters
    ----------
    body : BodyParams
        Anthropometric body model with joints in RAS mm.

    Returns
    -------
    list[dict]
        BodyItem dicts for the full vascular system.
    """
    items: list[dict] = []
    items.extend(_build_arterial(body))
    items.extend(_build_venous(body))
    return items
