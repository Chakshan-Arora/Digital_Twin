"""Shared types, primitives, and pure helper functions for hdt.systems.

All geometry is a schematic educational model — NOT a clinical tool.

Coordinate system: RAS millimetres.
  x > 0  → person's right
  x < 0  → person's left
  y > 0  → anterior
  z > 0  → superior; origin on floor between feet.
"""

from __future__ import annotations

import math
from typing import Literal

from hdt.body.models import BodyParams, JointName

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Polyline = list[tuple[float, float, float]]
"""Ordered sequence of 3-D points (x, y, z) in RAS mm."""


# ---------------------------------------------------------------------------
# Primitive dicts
# ---------------------------------------------------------------------------
# All primitives are plain Python dicts (no dataclasses) so that the output
# is directly JSON-serialisable without dataclasses.asdict().
#
# TubePrimitive keys:
#   kind          : "tube"
#   polyline      : Polyline
#   radius_mm     : float   – radius at start
#   radius_end_mm : float | None  – radius at end (None = constant radius)
#
# EllipsoidPrimitive keys:
#   kind            : "ellipsoid"
#   center          : (x, y, z) mm
#   radii_mm        : (rx, ry, rz) mm  – semi-axes
#   rotation_deg    : (rx, ry, rz) deg | None
#
# MeshHintPrimitive keys:
#   kind                       : "mesh_hint"
#   hint_text                  : str
#   bounding_ellipsoid_center  : (x, y, z) mm
#   bounding_ellipsoid_radii_mm: (rx, ry, rz) mm
#
# BodyItem keys:
#   id          : str  – unique item identifier
#   name        : str  – human-readable name
#   system      : str  – "skeleton" | "vascular_arterial" | "vascular_venous" | "organs"
#   side        : "l" | "r" | "mid"
#   primitive   : dict – one of the primitive dicts above
#   color_hint  : str  – CSS hex colour
#   label       : str  – short display label


def tube(
    polyline: Polyline,
    radius_mm: float,
    radius_end_mm: float | None = None,
) -> dict:
    """Return a TubePrimitive dict.

    Parameters
    ----------
    polyline : Polyline
        Ordered 3-D waypoints in RAS mm.
    radius_mm : float
        Tube radius at the first point, in mm.
    radius_end_mm : float | None
        Tube radius at the last point (None = same as start).
    """
    return {
        "kind": "tube",
        "polyline": [list(p) for p in polyline],
        "radius_mm": float(radius_mm),
        "radius_end_mm": float(radius_end_mm) if radius_end_mm is not None else None,
    }


def ellipsoid(
    center: tuple[float, float, float],
    radii_mm: tuple[float, float, float],
    rotation_deg: tuple[float, float, float] | None = None,
) -> dict:
    """Return an EllipsoidPrimitive dict.

    Parameters
    ----------
    center : (x, y, z)
        Centre in RAS mm.
    radii_mm : (rx, ry, rz)
        Semi-axis lengths in mm.
    rotation_deg : (rx, ry, rz) | None
        Euler rotation angles in degrees (optional).
    """
    return {
        "kind": "ellipsoid",
        "center": list(center),
        "radii_mm": list(radii_mm),
        "rotation_deg": list(rotation_deg) if rotation_deg is not None else None,
    }


def mesh_hint(
    hint_text: str,
    center: tuple[float, float, float],
    radii_mm: tuple[float, float, float],
) -> dict:
    """Return a MeshHintPrimitive dict.

    Parameters
    ----------
    hint_text : str
        Key describing which mesh asset the renderer should load.
    center : (x, y, z)
        Bounding ellipsoid centre in RAS mm.
    radii_mm : (rx, ry, rz)
        Bounding ellipsoid semi-axes in mm.
    """
    return {
        "kind": "mesh_hint",
        "hint_text": hint_text,
        "bounding_ellipsoid_center": list(center),
        "bounding_ellipsoid_radii_mm": list(radii_mm),
    }


def body_item(
    item_id: str,
    name: str,
    system: str,
    side: Literal["l", "r", "mid"],
    primitive: dict,
    color_hint: str,
    label: str,
) -> dict:
    """Return a BodyItem as a plain dict."""
    return {
        "id": item_id,
        "name": name,
        "system": system,
        "side": side,
        "primitive": primitive,
        "color_hint": color_hint,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------


def lerp(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """Linear interpolation between two 3-D points in RAS mm.

    Parameters
    ----------
    p1, p2 : (x, y, z) mm
    t : float
        0.0 → p1, 1.0 → p2.

    Returns
    -------
    (x, y, z) mm
    """
    return (
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
        p1[2] + t * (p2[2] - p1[2]),
    )


def mirror_x(item: dict) -> dict:
    """Return a copy of *item* mirrored across the x = 0 plane.

    Flips the sign of every x-coordinate in the primitive and swaps
    side 'l' ↔ 'r'.  Does NOT mutate the original.

    Parameters
    ----------
    item : dict
        A BodyItem dict (as returned by :func:`body_item`).

    Returns
    -------
    dict
        A new BodyItem dict.
    """

    def flip_point(p: list[float]) -> list[float]:
        return [-p[0], p[1], p[2]]

    prim = item["primitive"]
    kind = prim["kind"]

    if kind == "tube":
        new_prim = {
            "kind": "tube",
            "polyline": [flip_point(p) for p in prim["polyline"]],
            "radius_mm": prim["radius_mm"],
            "radius_end_mm": prim["radius_end_mm"],
        }
    elif kind == "ellipsoid":
        new_prim = {
            "kind": "ellipsoid",
            "center": flip_point(list(prim["center"])),
            "radii_mm": list(prim["radii_mm"]),
            "rotation_deg": (
                list(prim["rotation_deg"]) if prim.get("rotation_deg") is not None else None
            ),
        }
    elif kind == "mesh_hint":
        new_prim = {
            "kind": "mesh_hint",
            "hint_text": prim["hint_text"],
            "bounding_ellipsoid_center": flip_point(list(prim["bounding_ellipsoid_center"])),
            "bounding_ellipsoid_radii_mm": list(prim["bounding_ellipsoid_radii_mm"]),
        }
    else:
        new_prim = dict(prim)

    side_map: dict[str, str] = {"l": "r", "r": "l", "mid": "mid"}
    new_side = side_map.get(item["side"], item["side"])

    return {
        "id": item["id"],
        "name": item["name"],
        "system": item["system"],
        "side": new_side,
        "primitive": new_prim,
        "color_hint": item["color_hint"],
        "label": item["label"],
    }


def offset_z(p: tuple[float, float, float], dz: float) -> tuple[float, float, float]:
    """Add *dz* to the z-coordinate of point *p* (RAS mm).

    Parameters
    ----------
    p : (x, y, z) mm
    dz : float
        Offset in mm along the z (superior) axis.

    Returns
    -------
    (x, y, z) mm
    """
    return (p[0], p[1], p[2] + dz)


def torso_bounds(body: BodyParams) -> tuple[float, float, float, float]:
    """Return a simplified rectangular torso envelope from joint positions.

    Parameters
    ----------
    body : BodyParams
        Body model (joints in RAS mm).

    Returns
    -------
    (x_half, y_half, z_pelvis, z_neck) all in mm
        x_half   – half-width (shoulder_x + 20 mm)
        y_half   – half-depth (chest_y + 30 mm; measured from x=0 anteriorly)
        z_pelvis – z of pelvis joint
        z_neck   – z of neck joint
    """
    j = body.joints
    r_shoulder_x = j[JointName.R_SHOULDER][0]
    chest_y = j[JointName.CHEST][1]
    z_pelvis = j[JointName.PELVIS][2]
    z_neck = j[JointName.NECK][2]
    # chest_y is near 0 for the average preset; add a fixed anterior depth
    x_half = abs(r_shoulder_x) + 20.0
    # torso depth: 30% of shoulder width is a rough estimate
    y_half = abs(chest_y) + abs(r_shoulder_x) * 0.55 + 30.0
    return x_half, y_half, z_pelvis, z_neck


def inside_envelope(body: BodyParams, point: tuple[float, float, float]) -> bool:
    """Return True if *point* lies inside the simplified body envelope.

    The body is modelled as five axis-aligned zones:
      1. Head: sphere of radius 100 mm around the head joint.
      2. Torso: ellipsoid between neck and pelvis joints.
      3. Upper limbs: cylinders of radius 60 mm from shoulder to wrist (both sides).
      4. Lower limbs: cylinders of radius 65 mm from hip to foot joint (both sides).

    This is a schematic educational approximation — NOT anatomically precise.

    Parameters
    ----------
    body : BodyParams
        Body model (joints in RAS mm).
    point : (x, y, z) mm
        Query point in RAS mm.

    Returns
    -------
    bool
    """
    j = body.joints
    px, py, pz = point

    scale = body.height_mm / 1750.0

    # 1. Head sphere
    hx, hy, hz = j[JointName.HEAD]
    if math.sqrt((px - hx) ** 2 + (py - hy) ** 2 + (pz - hz) ** 2) <= 120.0 * scale:
        return True

    # 2. Torso ellipsoid (spans from neck down past sacrum/coccyx to pelvic floor)
    x_half, y_half, z_pelvis, z_neck = torso_bounds(body)
    z_bot = z_pelvis - 55.0 * scale
    z_top = z_neck + 15.0 * scale
    z_mid = (z_bot + z_top) / 2.0
    z_half = (z_top - z_bot) / 2.0
    nx, ny, nz = j[JointName.NECK]
    torso_cx, torso_cy = 0.0, ny  # torso centred on midline at neck y
    ex = (px - torso_cx) / (x_half + 15.0 * scale)
    ey = (py - torso_cy) / (y_half + 15.0 * scale)
    ez = (pz - z_mid) / z_half
    if ex * ex + ey * ey + ez * ez <= 1.0:
        return True

    # 3. Upper limb cylinders (shoulder → elbow, elbow → wrist, wrist → hand)
    for sh_j, el_j, wr_j, ha_j in [
        (JointName.L_SHOULDER, JointName.L_ELBOW, JointName.L_WRIST, JointName.L_HAND),
        (JointName.R_SHOULDER, JointName.R_ELBOW, JointName.R_WRIST, JointName.R_HAND),
    ]:
        s, e, w, h = j[sh_j], j[el_j], j[wr_j], j[ha_j]
        if (
            _point_in_cylinder(point, s, e, radius=70.0 * scale)
            or _point_in_cylinder(point, e, w, radius=65.0 * scale)
            or _point_in_cylinder(point, w, h, radius=60.0 * scale)
        ):
            return True

    # 4. Lower limb cylinders (hip → knee, knee → ankle, ankle → foot)
    for hip_j, knee_j, ankle_j, foot_j in [
        (JointName.L_HIP, JointName.L_KNEE, JointName.L_ANKLE, JointName.L_FOOT),
        (JointName.R_HIP, JointName.R_KNEE, JointName.R_ANKLE, JointName.R_FOOT),
    ]:
        hi, kn, an, fo = j[hip_j], j[knee_j], j[ankle_j], j[foot_j]
        if (
            _point_in_cylinder(point, hi, kn, radius=80.0 * scale)
            or _point_in_cylinder(point, kn, an, radius=70.0 * scale)
            or _point_in_cylinder(point, an, fo, radius=65.0 * scale)
        ):
            return True

    return False


def _point_in_cylinder(
    point: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    radius: float,
) -> bool:
    """Test whether *point* is inside a cylinder of given *radius* along axis a→b.

    Allows a 5% margin beyond endpoints to seamlessly cover joints.

    Parameters
    ----------
    point : (x, y, z) mm
    a, b  : axis endpoints in mm
    radius : float mm

    Returns
    -------
    bool
    """
    ax, ay, az = a
    bx, by, bz = b
    px, py, pz = point

    # Axis direction vector
    dx, dy, dz = bx - ax, by - ay, bz - az
    length_sq = dx * dx + dy * dy + dz * dz
    if length_sq == 0.0:
        return False

    # Projection of point onto axis with small extension at joints
    t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / length_sq
    if t < -0.05 or t > 1.05:
        return False

    # Closest point on axis
    cx = ax + t * dx
    cy = ay + t * dy
    cz = az + t * dz

    dist_sq = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
    return dist_sq <= radius * radius
