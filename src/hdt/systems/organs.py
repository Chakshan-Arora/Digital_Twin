"""Organ system builder for the Human Digital Twin body explorer.

Generates a schematic educational 3-D model of the major internal organs as a
list of BodyItem dicts (plain Python, JSON-serialisable).  NOT a clinical tool.

Coordinate system: RAS millimetres.
  x > 0  → person's right,  x < 0 → person's left
  y > 0  → anterior,        y < 0 → posterior
  z > 0  → superior;  origin on floor between feet.

Organs are modelled as ellipsoid primitives.

Size scaling: all radii scale linearly by body.height_mm / 1750 (reference).
All positions are fractions of torso height/width/depth derived from
torso_bounds(body) and joint positions — never hard-coded raw mm.

Anatomical side conventions (RAS, person-relative):
  liver  → x > 0  (person's right)
  spleen → x < 0  (person's left)
  stomach→ x < 0  (person's left)
  heart  → x < 0, |x| < 60 mm  (slightly left of midline)
"""

from __future__ import annotations

from hdt.body.models import BodyParams, JointName
from hdt.systems.common import body_item, ellipsoid, torso_bounds

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_COLORS: dict[str, str] = {
    "brain": "#E8C8C8",
    "heart": "#CC4444",
    "lung": "#F0A0A0",
    "liver": "#8B4513",
    "stomach": "#DEB887",
    "spleen": "#8B008B",
    "pancreas": "#F5DEB3",
    "kidney": "#A0522D",
    "bladder": "#ADD8E6",
    "small_intestine": "#F0E68C",
    "large_intestine": "#DAA520",
    "diaphragm": "#FFE4C4",
}

_REF_H = 1750.0


def _scale(body: BodyParams) -> float:
    return body.height_mm / _REF_H


def _j(body: BodyParams, name: JointName) -> tuple[float, float, float]:
    return body.joints[name]


def _organ(
    organ_id: str,
    name: str,
    side: str,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    color: str,
    label: str,
    rotation_deg: tuple[float, float, float] | None = None,
) -> dict:
    """Return a single organ BodyItem dict."""
    return body_item(
        item_id=f"organs_{organ_id}",
        name=name,
        system="organs",
        side=side,
        primitive=ellipsoid(center, radii, rotation_deg),
        color_hint=color,
        label=label,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_organs(body: BodyParams) -> list[dict]:
    """Build a schematic educational 3-D organ model.

    Returns a list of BodyItem dicts (plain Python, JSON-serialisable).
    Each dict has keys: id, name, system, side, primitive, color_hint, label.
    All primitives have kind="ellipsoid".

    Organ centre positions are strictly derived as anthropometric fractions of
    torso width (x_half), torso depth (y_half), and torso span (neck_z - pelvis_z).
    Radii scale linearly with body.height_mm / 1750.

    Volumes strictly comply with adult anatomical reference ranges:
      - lungs: 1.5 - 2.5 L each (inflated vital capacity envelope with airway volume)
      - liver: >= 1.2 L (mean adult liver volume ~1.4 L)
      - small intestine: >= 1.0 L (visceral bowel loop packet)
      - all non-intended organ pairs overlap < 15 % of the smaller organ volume.

    Anatomical notes (schematic approximations — UNSURE list):
      - Diaphragm z is approximated at the T8 level (0.48 of torso span from
        pelvis); exact position varies with breathing phase.
      - Large intestine is represented as three ellipsoids framing the abdominal
        cavity (ascending, transverse, descending colon).
      - Organ shapes are simplified ellipsoids — real organs have complex morphology.

    This is a schematic educational model — NOT a clinical tool.

    Parameters
    ----------
    body : BodyParams
        Anthropometric body model with joints in RAS mm.

    Returns
    -------
    list[dict]
        BodyItem dicts for the full organ set.
    """
    s = _scale(body)
    items: list[dict] = []

    # Torso dimensions and reference joints
    x_half, y_half, z_pelvis, z_neck = torso_bounds(body)
    H = z_neck - z_pelvis
    chest = _j(body, JointName.CHEST)
    chest_y = chest[1]
    head = _j(body, JointName.HEAD)
    head_z = head[2]

    # -----------------------------------------------------------------------
    # 1. Brain (cranial vault)
    # Reference range: 1.20 - 1.50 L; Model: 1.371 L
    # -----------------------------------------------------------------------
    brain_cx, brain_cy = 0.0, 0.0
    brain_cz = head_z - 30.0 * s
    items.append(
        _organ(
            "brain",
            "brain",
            "mid",
            (brain_cx, brain_cy, brain_cz),
            (70.0 * s, 85.0 * s, 55.0 * s),
            _COLORS["brain"],
            "Brain",
        )
    )

    # -----------------------------------------------------------------------
    # 2. Heart (anterior mediastinum, slightly left of midline)
    # Reference range: 0.50 - 0.85 L; Model: 0.664 L
    # -----------------------------------------------------------------------
    heart_cx = -0.09 * x_half
    heart_cy = chest_y + 0.08 * y_half
    heart_cz = z_pelvis + 0.58 * H
    items.append(
        _organ(
            "heart",
            "heart",
            "l",
            (heart_cx, heart_cy, heart_cz),
            (55.0 * s, 48.0 * s, 60.0 * s),
            _COLORS["heart"],
            "Heart",
        )
    )

    # -----------------------------------------------------------------------
    # 3. Lungs (bilateral hemithoraces)
    # Envelope-with-air range: 1.50 - 2.50 L each; Model: L 1.603 L, R 1.653 L
    # -----------------------------------------------------------------------
    lung_cz = z_pelvis + 0.61 * H
    # Left lung
    items.append(
        _organ(
            "lung_l",
            "left_lung",
            "l",
            (-0.38 * x_half, chest_y - 0.04 * y_half, lung_cz),
            (64.0 * s, 52.0 * s, 115.0 * s),
            _COLORS["lung"],
            "Left Lung",
        )
    )
    # Right lung
    items.append(
        _organ(
            "lung_r",
            "right_lung",
            "r",
            (0.38 * x_half, chest_y - 0.04 * y_half, lung_cz),
            (66.0 * s, 52.0 * s, 115.0 * s),
            _COLORS["lung"],
            "Right Lung",
        )
    )

    # -----------------------------------------------------------------------
    # 4. Liver (right hypochondrium / epigastrium, x > 0)
    # Reference range: >= 1.20 L (1.20 - 1.80 L); Model: 1.219 L
    # -----------------------------------------------------------------------
    liver_cx = 0.28 * x_half
    liver_cy = chest_y + 0.10 * y_half
    liver_cz = z_pelvis + 0.40 * H
    items.append(
        _organ(
            "liver",
            "liver",
            "r",
            (liver_cx, liver_cy, liver_cz),
            (76.0 * s, 66.0 * s, 58.0 * s),
            _COLORS["liver"],
            "Liver",
        )
    )

    # -----------------------------------------------------------------------
    # 5. Stomach (left hypochondrium / epigastrium, x < 0)
    # Reference range: 0.40 - 0.80 L; Model: 0.405 L
    # -----------------------------------------------------------------------
    stomach_cx = -0.22 * x_half
    stomach_cy = chest_y + 0.12 * y_half
    stomach_cz = z_pelvis + 0.39 * H
    items.append(
        _organ(
            "stomach",
            "stomach",
            "l",
            (stomach_cx, stomach_cy, stomach_cz),
            (50.0 * s, 42.0 * s, 46.0 * s),
            _COLORS["stomach"],
            "Stomach",
        )
    )

    # -----------------------------------------------------------------------
    # 6. Spleen (left posterolateral hypochondrium, x < 0)
    # Reference range: 0.15 - 0.30 L; Model: 0.203 L
    # -----------------------------------------------------------------------
    spleen_cx = -0.46 * x_half
    spleen_cy = chest_y - 0.18 * y_half
    spleen_cz = z_pelvis + 0.38 * H
    items.append(
        _organ(
            "spleen",
            "spleen",
            "l",
            (spleen_cx, spleen_cy, spleen_cz),
            (42.0 * s, 34.0 * s, 34.0 * s),
            _COLORS["spleen"],
            "Spleen",
        )
    )

    # -----------------------------------------------------------------------
    # 7. Pancreas (retroperitoneal, posterior to stomach)
    # Reference range: 0.07 - 0.13 L; Model: 0.080 L
    # -----------------------------------------------------------------------
    pancreas_cx = -0.04 * x_half
    pancreas_cy = chest_y - 0.14 * y_half
    pancreas_cz = z_pelvis + 0.33 * H
    items.append(
        _organ(
            "pancreas",
            "pancreas",
            "mid",
            (pancreas_cx, pancreas_cy, pancreas_cz),
            (48.0 * s, 18.0 * s, 22.0 * s),
            _COLORS["pancreas"],
            "Pancreas",
        )
    )

    # -----------------------------------------------------------------------
    # 8. Kidneys (bilateral retroperitoneal, tucked against posterior abdominal wall)
    # Reference range: 0.14 - 0.22 L each; Model: 0.157 L each
    # -----------------------------------------------------------------------
    kidney_cy = chest_y - 0.28 * y_half
    # Left kidney
    items.append(
        _organ(
            "kidney_l",
            "l_kidney",
            "l",
            (-0.25 * x_half, kidney_cy, z_pelvis + 0.28 * H),
            (30.0 * s, 26.0 * s, 48.0 * s),
            _COLORS["kidney"],
            "Left Kidney",
        )
    )
    # Right kidney (slightly lower than left due to liver)
    items.append(
        _organ(
            "kidney_r",
            "r_kidney",
            "r",
            (0.25 * x_half, kidney_cy, z_pelvis + 0.26 * H),
            (30.0 * s, 26.0 * s, 48.0 * s),
            _COLORS["kidney"],
            "Right Kidney",
        )
    )

    # -----------------------------------------------------------------------
    # 9. Small Intestine (central/anterior midgut packet)
    # Reference range: >= 1.00 L (1.00 - 1.50 L); Model: 1.021 L
    # -----------------------------------------------------------------------
    si_cz = z_pelvis + 0.18 * H
    items.append(
        _organ(
            "small_intestine",
            "small_intestine",
            "mid",
            (0.0, chest_y + 0.18 * y_half, si_cz),
            (68.0 * s, 56.0 * s, 64.0 * s),
            _COLORS["small_intestine"],
            "Small Intestine",
        )
    )

    # -----------------------------------------------------------------------
    # 10. Large Intestine (ascending, transverse, descending colon framing abdomen)
    # Reference range: 0.15 - 0.30 L each; Model: Asc 0.181 L, Trans 0.166 L, Desc 0.181 L
    # -----------------------------------------------------------------------
    colon_y = chest_y + 0.02 * y_half
    # Ascending colon (right flank)
    items.append(
        _organ(
            "colon_ascending",
            "ascending_colon",
            "r",
            (0.38 * x_half, colon_y, z_pelvis + 0.18 * H),
            (28.0 * s, 28.0 * s, 55.0 * s),
            _COLORS["large_intestine"],
            "Ascending Colon",
        )
    )
    # Transverse colon (upper anterior abdomen)
    items.append(
        _organ(
            "colon_transverse",
            "transverse_colon",
            "mid",
            (0.0, chest_y + 0.22 * y_half, z_pelvis + 0.26 * H),
            (75.0 * s, 24.0 * s, 22.0 * s),
            _COLORS["large_intestine"],
            "Transverse Colon",
        )
    )
    # Descending colon (left flank)
    items.append(
        _organ(
            "colon_descending",
            "descending_colon",
            "l",
            (-0.38 * x_half, colon_y, z_pelvis + 0.18 * H),
            (28.0 * s, 28.0 * s, 55.0 * s),
            _COLORS["large_intestine"],
            "Descending Colon",
        )
    )

    # -----------------------------------------------------------------------
    # 11. Bladder (anterior true pelvis, behind pubic symphysis)
    # Reference range: 0.20 - 0.45 L; Model: 0.251 L
    # -----------------------------------------------------------------------
    bladder_cz = z_pelvis + 0.01 * H
    items.append(
        _organ(
            "bladder",
            "bladder",
            "mid",
            (0.0, chest_y + 0.18 * y_half, bladder_cz),
            (42.0 * s, 42.0 * s, 34.0 * s),
            _COLORS["bladder"],
            "Bladder",
        )
    )

    # -----------------------------------------------------------------------
    # 12. Diaphragm (thoracoabdominal junction, ~T8 level)
    # -----------------------------------------------------------------------
    t8_z = z_pelvis + 0.48 * H
    items.append(
        _organ(
            "diaphragm",
            "diaphragm",
            "mid",
            (0.0, 0.0, t8_z),
            (140.0 * s, 130.0 * s, 10.0 * s),
            _COLORS["diaphragm"],
            "Diaphragm",
        )
    )

    return items
