"""Skeletal system builder for the Human Digital Twin body explorer.

Generates a schematic educational 3-D model of the human skeleton as a list
of BodyItem dicts (plain Python, JSON-serialisable).  NOT a clinical tool.

Coordinate system: RAS millimetres.
  x > 0  → person's right,  x < 0 → person's left
  y > 0  → anterior,        y < 0 → posterior
  z > 0  → superior;  origin on floor between feet.

Anatomy modelled
----------------
- Long bones (tapered tubes): humerus, forearm, femur, shank, hand, foot,
  clavicle (left + right each).
- Scapulae (ellipsoids, bilateral).
- Spine: 24 vertebrae C1-C7, T1-T12, L1-L5 (ellipsoids) + sacrum + coccyx.
  Simplified 3-segment curved spine: cervical lordosis, thoracic kyphosis,
  lumbar lordosis modelled as linear y-offsets within each region.
  Vertebrae are guaranteed to be in strictly descending z order.
- Ribcage: 12 rib pairs (4-point polyline tubes) + sternum (tube).
- Skull (mesh_hint).
- Pelvis (mesh_hint).

Size scaling: linear with height_mm / 1750 (reference height).
"""

from __future__ import annotations

from hdt.body.models import BodyParams, JointName
from hdt.systems.common import (
    body_item,
    ellipsoid,
    mesh_hint,
    tube,
)

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
_BONE_COLOR = "#E8E0D0"
_CART_COLOR = "#D0E8D0"  # cartilage / costal connections

# ---------------------------------------------------------------------------
# Reference height for scaling
# ---------------------------------------------------------------------------
_REF_HEIGHT = 1750.0  # mm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scale(body: BodyParams) -> float:
    """Return the dimensionless height-scale factor relative to reference height.

    Returns
    -------
    float
        body.height_mm / 1750.
    """
    return body.height_mm / _REF_HEIGHT


def _j(body: BodyParams, name: JointName) -> tuple[float, float, float]:
    """Return the (x, y, z) position of a joint in RAS mm."""
    return body.joints[name]


# ---------------------------------------------------------------------------
# Long bones
# ---------------------------------------------------------------------------


def _build_long_bones(body: BodyParams) -> list[dict]:
    """Build bilateral long bones as tapered-tube BodyItems.

    Returns
    -------
    list[dict]
        BodyItem dicts for humerus, forearm, femur, shank, hand-bone,
        foot-bone, and clavicle — left and right sides.
    """
    s = _scale(body)
    items: list[dict] = []

    bones: list[tuple[str, JointName, JointName, float, float]] = [
        # (name, proximal_joint, distal_joint, r_start_ref, r_end_ref)
        ("humerus", JointName.L_SHOULDER, JointName.L_ELBOW, 15.0, 10.0),
        ("forearm", JointName.L_ELBOW, JointName.L_WRIST, 12.0, 8.0),
        ("femur", JointName.L_HIP, JointName.L_KNEE, 22.0, 18.0),
        ("shank", JointName.L_KNEE, JointName.L_ANKLE, 18.0, 12.0),
        ("hand_bone", JointName.L_WRIST, JointName.L_HAND, 10.0, 8.0),
        ("foot_bone", JointName.L_ANKLE, JointName.L_FOOT, 30.0, 20.0),
    ]

    for bone_name, prox_j, dist_j in [
        (n, p, d) for (n, p, d, _, __) in bones
    ]:
        r_start, r_end = next((r0, r1) for (n, p, d, r0, r1) in bones if n == bone_name)
        prox = _j(body, prox_j)
        dist = _j(body, dist_j)
        prim = tube([prox, dist], r_start * s, r_end * s)
        # Left side
        items.append(
            body_item(
                item_id=f"skeleton_{bone_name}_l",
                name=f"l_{bone_name}",
                system="skeleton",
                side="l",
                primitive=prim,
                color_hint=_BONE_COLOR,
                label=f"L {bone_name.replace('_', ' ').title()}",
            )
        )
        # Right side by mirroring and re-setting IDs
        r_prox_j_name = prox_j.value.replace("l_", "r_")
        r_dist_j_name = dist_j.value.replace("l_", "r_")
        r_prox_j = JointName(r_prox_j_name)
        r_dist_j = JointName(r_dist_j_name)
        r_prox = _j(body, r_prox_j)
        r_dist = _j(body, r_dist_j)
        r_prim = tube([r_prox, r_dist], r_start * s, r_end * s)
        items.append(
            body_item(
                item_id=f"skeleton_{bone_name}_r",
                name=f"r_{bone_name}",
                system="skeleton",
                side="r",
                primitive=r_prim,
                color_hint=_BONE_COLOR,
                label=f"R {bone_name.replace('_', ' ').title()}",
            )
        )

    # Clavicles: neck (offset ±20mm) → shoulder
    neck = _j(body, JointName.NECK)
    for side, shoulder_j, sign in [
        ("l", JointName.L_SHOULDER, -1),
        ("r", JointName.R_SHOULDER, 1),
    ]:
        neck_offset = (sign * 20.0 * s, neck[1], neck[2])
        shoulder = _j(body, shoulder_j)
        prim = tube([neck_offset, shoulder], 8.0 * s, 6.0 * s)
        items.append(
            body_item(
                item_id=f"skeleton_clavicle_{side}",
                name=f"{side}_clavicle",
                system="skeleton",
                side=side,
                primitive=prim,
                color_hint=_BONE_COLOR,
                label=f"{'L' if side == 'l' else 'R'} Clavicle",
            )
        )

    return items


# ---------------------------------------------------------------------------
# Scapulae
# ---------------------------------------------------------------------------


# Anthropometric stature fraction for posterior scapular offset:
# -30 mm at reference height 1750 mm = -0.01714 of stature
_SCAPULA_Y_STATURE_FRAC = -30.0 / _REF_HEIGHT


def _build_scapulae(body: BodyParams) -> list[dict]:
    """Build bilateral scapula ellipsoid BodyItems.

    Scapulae are positioned posteriorly (y < 0) near each shoulder,
    with posterior offset scaling as a documented fraction of standing stature.
    """
    s = _scale(body)
    items: list[dict] = []
    cy = _SCAPULA_Y_STATURE_FRAC * body.height_mm
    for side, shoulder_j, sign in [
        ("l", JointName.L_SHOULDER, -1),
        ("r", JointName.R_SHOULDER, 1),
    ]:
        sh = _j(body, shoulder_j)
        cx = sign * 90.0 * s
        cz = sh[2]
        prim = ellipsoid((cx, cy, cz), (55.0 * s, 40.0 * s, 8.0 * s))
        items.append(
            body_item(
                item_id=f"skeleton_scapula_{side}",
                name=f"{side}_scapula",
                system="skeleton",
                side=side,
                primitive=prim,
                color_hint=_BONE_COLOR,
                label=f"{'L' if side == 'l' else 'R'} Scapula",
            )
        )
    return items


# ---------------------------------------------------------------------------
# Spine
# ---------------------------------------------------------------------------


def _spine_curve(
    z: float,
    z_pelvis: float,
    z_neck: float,
    neck_y: float,
) -> float:
    """Return the anterior (y) offset at vertebral height *z*.

    Simplified 3-segment model (linear interpolation within each segment):
      - Lumbar (pelvis_z → T12 ≈ 60 % of torso): +10 mm lordosis (anterior)
      - Thoracic (T12 → T1 ≈ 85 % of torso): -15 mm kyphosis (posterior)
      - Cervical (T1 → neck): +8 mm lordosis (anterior)

    Parameters
    ----------
    z : float
        Height of the vertebra in RAS mm.
    z_pelvis, z_neck : float
        Limits of the vertebral column in mm.
    neck_y : float
        y-coordinate of the neck joint (used as the base y of the spine).

    Returns
    -------
    float
        Anterior y offset in mm at height z.
    """
    torso_span = z_neck - z_pelvis
    if torso_span <= 0:
        return neck_y

    frac = (z - z_pelvis) / torso_span  # 0 = pelvis, 1 = neck

    # Segment boundaries (fraction of torso height)
    f_t12 = 0.60  # T12 level
    f_t1 = 0.85  # T1 level

    if frac <= f_t12:
        # Lumbar lordosis: 0 → +10 mm at T12
        t = frac / f_t12
        y_offset = 10.0 * t
    elif frac <= f_t1:
        # Thoracic kyphosis: +10 mm at T12 → -15 mm at T1
        t = (frac - f_t12) / (f_t1 - f_t12)
        y_offset = 10.0 + t * (-15.0 - 10.0)
    else:
        # Cervical lordosis: -15 mm at T1 → +8 mm at neck
        t = (frac - f_t1) / (1.0 - f_t1)
        y_offset = -15.0 + t * (8.0 - (-15.0))

    return neck_y + y_offset


def _build_spine(body: BodyParams) -> list[dict]:
    """Build 24 vertebrae + sacrum + coccyx as ellipsoid BodyItems.

    Vertebrae are placed from C1 (uppermost) to L5 (lowermost) along a
    3-segment curved spine (cervical lordosis, thoracic kyphosis, lumbar
    lordosis).  Sacrum and coccyx are added below L5.

    Vertebrae are asserted to be in strictly descending z order.

    Returns
    -------
    list[dict]
        26 BodyItem dicts (24 vertebrae + sacrum + coccyx), ordered C1→coccyx.
    """
    s = _scale(body)
    items: list[dict] = []

    z_neck = _j(body, JointName.NECK)[2]
    z_pelvis = _j(body, JointName.PELVIS)[2]
    neck_y = _j(body, JointName.NECK)[1]

    # Reference sizes (rx, ry, rz) in mm at reference height
    # Sizes increase downward within each region.
    vertebra_specs: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []

    # Cervical C1-C7 (7): (9,12,6) → (11,14,7)
    for i in range(7):
        t = i / 6.0
        label = f"C{i + 1}"
        rx = 9.0 + t * 2.0
        ry = 12.0 + t * 2.0
        rz = 6.0 + t * 1.0
        vertebra_specs.append((label, (rx, ry, rz), (0.0, 0.0, 0.0)))

    # Thoracic T1-T12 (12): (12,15,8) → (16,19,10)
    for i in range(12):
        t = i / 11.0
        label = f"T{i + 1}"
        rx = 12.0 + t * 4.0
        ry = 15.0 + t * 4.0
        rz = 8.0 + t * 2.0
        vertebra_specs.append((label, (rx, ry, rz), (0.0, 0.0, 0.0)))

    # Lumbar L1-L5 (5): (17,22,12) → (22,28,15)
    for i in range(5):
        t = i / 4.0
        label = f"L{i + 1}"
        rx = 17.0 + t * 5.0
        ry = 22.0 + t * 6.0
        rz = 12.0 + t * 3.0
        vertebra_specs.append((label, (rx, ry, rz), (0.0, 0.0, 0.0)))

    # Total vertebrae: 24 (C1-C7 + T1-T12 + L1-L5)
    n_verts = len(vertebra_specs)  # should be 24

    # Distribute vertebra z positions from z_neck (C1) down to z_pelvis + margin (L5)
    # Small spacing between vertebrae; C1 is just below neck, L5 just above pelvis.
    z_top = z_neck - 10.0 * s  # C1 just below neck joint
    z_bot = z_pelvis + 20.0 * s  # L5 just above pelvis

    # Heights of each vertebra from top (C1) to bottom (L5) — strictly descending
    vert_heights = [z_top - i * (z_top - z_bot) / (n_verts - 1) for i in range(n_verts)]

    for (label, (rx, ry, rz), _), z_v in zip(
        vertebra_specs, vert_heights, strict=True
    ):
        y_v = _spine_curve(z_v, z_pelvis, z_neck, neck_y)
        prim = ellipsoid(
            (0.0, y_v, z_v),
            (rx * s, ry * s, rz * s),
        )
        items.append(
            body_item(
                item_id=f"skeleton_vertebra_{label}",
                name=f"vertebra_{label}",
                system="skeleton",
                side="mid",
                primitive=prim,
                color_hint=_BONE_COLOR,
                label=label,
            )
        )

    # Sacrum: below L5
    z_sacrum = z_bot - 30.0 * s
    y_sacrum = _spine_curve(z_sacrum, z_pelvis, z_neck, neck_y)
    items.append(
        body_item(
            item_id="skeleton_sacrum",
            name="sacrum",
            system="skeleton",
            side="mid",
            primitive=ellipsoid((0.0, y_sacrum, z_sacrum), (25.0 * s, 20.0 * s, 30.0 * s)),
            color_hint=_BONE_COLOR,
            label="Sacrum",
        )
    )

    # Coccyx: below sacrum
    z_coccyx = z_sacrum - 25.0 * s
    y_coccyx = _spine_curve(z_coccyx, z_pelvis, z_neck, neck_y)
    items.append(
        body_item(
            item_id="skeleton_coccyx",
            name="coccyx",
            system="skeleton",
            side="mid",
            primitive=ellipsoid((0.0, y_coccyx, z_coccyx), (12.0 * s, 10.0 * s, 15.0 * s)),
            color_hint=_BONE_COLOR,
            label="Coccyx",
        )
    )

    # Internal integrity check: vertebrae must be strictly descending in z
    vert_items = [it for it in items if it["label"].startswith(("C", "T", "L"))]
    zs = [it["primitive"]["center"][2] for it in vert_items]
    for i in range(len(zs) - 1):
        assert zs[i] > zs[i + 1], (
            f"Vertebra z not strictly descending at index {i}: "
            f"{zs[i]:.2f} vs {zs[i + 1]:.2f}"
        )

    return items


# ---------------------------------------------------------------------------
# Ribcage
# ---------------------------------------------------------------------------


def _rib_polyline(
    vert_pos: tuple[float, float, float],
    x_half: float,
    y_half: float,
    side: str,  # "l" or "r"
    floating: bool,
) -> list[tuple[float, float, float]]:
    """Generate a 4-point rib polyline in RAS mm.

    The rib wraps from the thoracic vertebra posteriorly, curves outward
    in the x-direction, then comes anteriorly toward the sternum or free end.

    Parameters
    ----------
    vert_pos : (x, y, z)
        Position of the corresponding thoracic vertebra centre in RAS mm.
    x_half : float
        Half-width of the torso ellipse at this rib level, in mm.
    y_half : float
        Half-depth of the torso ellipse at this rib level, in mm.
    side : "l" | "r"
        Which side of the body.
    floating : bool
        Whether this is a floating rib (ribs 11-12); shorter, no sternum attachment.

    Returns
    -------
    list of (x, y, z) mm
        4-point polyline: [vertebra, curve outward, anterior arc, sternum/free].
    """
    sign = -1.0 if side == "l" else 1.0
    vx, vy, vz = vert_pos

    # Point 0: vertebra attachment (at posterior of spine, y slight posterior)
    p0 = (sign * 15.0, vy - 10.0, vz)
    # Point 1: lateral outward
    p1 = (sign * x_half * 0.95, vy, vz + 5.0)
    # Point 2: anterior curve
    p2 = (sign * x_half * 0.5, vy + y_half * 0.8, vz + 3.0)
    # Point 3: sternum or free end
    if floating:
        p3 = (sign * x_half * 0.3, vy + y_half * 0.6, vz + 2.0)
    else:
        p3 = (sign * 15.0, vy + y_half * 0.85, vz + 2.0)

    return [p0, p1, p2, p3]


def _build_ribcage(
    body: BodyParams,
    spine_items: list[dict],
) -> list[dict]:
    """Build 12 rib pairs and the sternum as tube BodyItems.

    Thoracic vertebra positions are taken directly from *spine_items*.

    Returns
    -------
    list[dict]
        24 rib BodyItem dicts + 1 sternum BodyItem.
    """
    s = _scale(body)
    items: list[dict] = []

    # Extract thoracic vertebra positions (T1-T12)
    thoracic: dict[int, tuple[float, float, float]] = {}
    for it in spine_items:
        lbl = it["label"]
        if lbl.startswith("T") and lbl[1:].isdigit():
            n = int(lbl[1:])
            thoracic[n] = tuple(it["primitive"]["center"])  # type: ignore[assignment]

    # Torso half-widths at mid-thorax
    x_half, y_half, _, _ = _build_torso_half_widths(body)

    for rib_n in range(1, 13):
        if rib_n not in thoracic:
            continue
        vert_pos = thoracic[rib_n]
        floating = rib_n >= 11

        # Torso narrows slightly at lower ribs; scale proportionally
        rib_x_half = x_half * (1.0 - 0.02 * (rib_n - 1))
        rib_y_half = y_half * (1.0 - 0.01 * (rib_n - 1))

        for side in ("l", "r"):
            polyline = _rib_polyline(vert_pos, rib_x_half, rib_y_half, side, floating)
            color = _CART_COLOR if not floating and rib_n <= 10 else _BONE_COLOR
            prim = tube(polyline, 6.0 * s, 5.0 * s)
            items.append(
                body_item(
                    item_id=f"skeleton_rib_{rib_n}_{side}",
                    name=f"{side}_rib_{rib_n}",
                    system="skeleton",
                    side=side,
                    primitive=prim,
                    color_hint=color,
                    label=f"{'L' if side == 'l' else 'R'} Rib {rib_n}",
                )
            )

    # Sternum: vertical tube from T1 level to T9 level, anterior to spine
    if 1 in thoracic and 9 in thoracic:
        t1_z = thoracic[1][2]
        t9_z = thoracic[9][2]
        neck_y = _j(body, JointName.NECK)[1]
        sternum_y = neck_y + y_half * 0.75
        sternum_top = (0.0, sternum_y, t1_z)
        sternum_bot = (0.0, sternum_y, t9_z)
        items.append(
            body_item(
                item_id="skeleton_sternum",
                name="sternum",
                system="skeleton",
                side="mid",
                primitive=tube([sternum_top, sternum_bot], 12.0 * s, 10.0 * s),
                color_hint=_BONE_COLOR,
                label="Sternum",
            )
        )

    return items


def _build_torso_half_widths(body: BodyParams) -> tuple[float, float, float, float]:
    """Return (x_half, y_half, z_pelvis, z_neck) of the torso at mid-chest.

    Used by rib geometry.
    """
    s = _scale(body)
    r_sh = _j(body, JointName.R_SHOULDER)
    neck = _j(body, JointName.NECK)
    pelvis = _j(body, JointName.PELVIS)
    x_half = abs(r_sh[0]) + 10.0 * s
    y_half = x_half * 0.55 + 20.0 * s
    return x_half, y_half, pelvis[2], neck[2]


# ---------------------------------------------------------------------------
# Skull & Pelvis mesh hints
# ---------------------------------------------------------------------------


def _build_skull_pelvis(body: BodyParams) -> list[dict]:
    """Build skull and pelvis mesh-hint BodyItems."""
    s = _scale(body)
    items: list[dict] = []

    # Skull: centred slightly above head joint
    head = _j(body, JointName.HEAD)
    skull_center = (head[0], head[1], head[2] + 50.0 * s)
    items.append(
        body_item(
            item_id="skeleton_skull",
            name="skull",
            system="skeleton",
            side="mid",
            primitive=mesh_hint(
                "skull_approximation",
                skull_center,
                (95.0 * s, 115.0 * s, 80.0 * s),
            ),
            color_hint=_BONE_COLOR,
            label="Skull",
        )
    )

    # Pelvis
    pelvis = _j(body, JointName.PELVIS)
    items.append(
        body_item(
            item_id="skeleton_pelvis",
            name="pelvis",
            system="skeleton",
            side="mid",
            primitive=mesh_hint(
                "pelvis_approximation",
                pelvis,
                (100.0 * s, 90.0 * s, 60.0 * s),
            ),
            color_hint=_BONE_COLOR,
            label="Pelvis",
        )
    )

    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_skeleton(body: BodyParams) -> list[dict]:
    """Build a schematic educational 3-D skeleton model.

    Returns a list of BodyItem dicts (plain Python dicts, JSON-serialisable).
    Each dict has keys: id, name, system, side, primitive, color_hint, label.

    Primitive kinds: "tube" (long bones, ribs, clavicles, sternum) and
    "ellipsoid" (vertebrae, scapulae) and "mesh_hint" (skull, pelvis).

    Item counts (exact):
      - 24 vertebrae (C1-C7, T1-T12, L1-L5)
      - 24 ribs (12 × 2 sides)
      - 2 clavicles
      - 2 scapulae
      - 2 femurs (as "femur" in long bones)
      plus: sacrum, coccyx, sternum, skull, pelvis, humerus×2, forearm×2,
      shank×2, hand_bone×2, foot_bone×2.

    Size scaling: all radii and offsets scaled linearly by
    body.height_mm / 1750.  Lengths scale naturally from joint positions.

    This is a schematic educational model — NOT a clinical tool.

    Parameters
    ----------
    body : BodyParams
        Anthropometric body model with joints in RAS mm.

    Returns
    -------
    list[dict]
        BodyItem dicts for the full skeleton.
    """
    items: list[dict] = []

    long_bones = _build_long_bones(body)
    items.extend(long_bones)

    scapulae = _build_scapulae(body)
    items.extend(scapulae)

    spine = _build_spine(body)
    items.extend(spine)

    # Extract spine items for ribcage
    spine_only = [it for it in spine if it["label"].startswith("T")]
    ribs_sternum = _build_ribcage(body, spine_only)
    items.extend(ribs_sternum)

    skull_pelvis = _build_skull_pelvis(body)
    items.extend(skull_pelvis)

    return items
