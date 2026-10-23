"""System registry and dispatcher for hdt.systems.

Maps system IDs to metadata and builder functions.
NOT a clinical tool.
"""

from __future__ import annotations

from hdt.body.models import BodyParams
from hdt.systems.organs import build_organs
from hdt.systems.skeleton import build_skeleton
from hdt.systems.vascular import build_vascular

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SYSTEMS: dict[str, dict] = {
    "skeleton": {
        "id": "skeleton",
        "name": "Skeletal System",
        "description": (
            "Schematic educational model of the human skeleton: "
            "long bones, vertebrae (C1-C7, T1-T12, L1-L5), ribcage, skull, pelvis."
        ),
        "default_visible": True,
        "default_opacity": 1.0,
        "color_hint": "#E8E0D0",
        "item_count_note": "~90 items",
        "available": True,
    },
    "vascular": {
        "id": "vascular",
        "name": "Vascular System",
        "description": (
            "Schematic educational model of the major blood vessels: "
            "arteries (red) and veins (blue)."
        ),
        "default_visible": True,
        "default_opacity": 0.85,
        "color_hint": "#CC2200",
        "item_count_note": "~50 items",
        "available": True,
    },
    "organs": {
        "id": "organs",
        "name": "Organ System",
        "description": (
            "Schematic educational ellipsoid approximations of major internal organs."
        ),
        "default_visible": True,
        "default_opacity": 0.75,
        "color_hint": "#CC4444",
        "item_count_note": "~20 items",
        "available": True,
    },
    "nervous": {
        "id": "nervous",
        "name": "Nervous System",
        "description": (
            "Provided by a separate /body/pathway endpoint; "
            "build on nervous branch."
        ),
        "default_visible": False,
        "default_opacity": 0.9,
        "color_hint": "#FFFF88",
        "item_count_note": "N/A",
        "available": False,
    },
    "muscular": {
        "id": "muscular",
        "name": "Muscular System",
        "description": "Muscular system — not yet implemented.",
        "default_visible": False,
        "default_opacity": 0.8,
        "color_hint": "#CC8866",
        "item_count_note": "N/A",
        "available": False,
    },
}

_BUILDERS = {
    "skeleton": build_skeleton,
    "vascular": build_vascular,
    "organs": build_organs,
}


def build_system(body: BodyParams, system_id: str) -> list[dict]:
    """Build and return BodyItem dicts for the named system.

    Parameters
    ----------
    body : BodyParams
        Anthropometric body model with joints in RAS mm.
    system_id : str
        One of the available system IDs (see :data:`SYSTEMS`).

    Returns
    -------
    list[dict]
        BodyItem dicts for the requested system.

    Raises
    ------
    KeyError
        If *system_id* is not in :data:`SYSTEMS` or the system is not available.
    """
    if system_id not in _BUILDERS:
        raise KeyError(f"System '{system_id}' is not available or does not exist.")
    return _BUILDERS[system_id](body)
