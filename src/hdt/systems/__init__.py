"""hdt.systems — schematic educational body-systems package.

Builds 3-D primitive descriptions (tubes, ellipsoids, mesh hints) for the
skeletal, vascular, and organ systems.  All geometry is a schematic
educational model and is NOT a clinical tool.

Public API
----------
build_skeleton  : build_skeleton(body) -> list[dict]
build_vascular  : build_vascular(body) -> list[dict]
build_organs    : build_organs(body)   -> list[dict]
build_system    : build_system(body, system_id) -> list[dict]
SYSTEMS         : registry dict
"""

from hdt.systems.organs import build_organs
from hdt.systems.registry import SYSTEMS, build_system
from hdt.systems.skeleton import build_skeleton
from hdt.systems.vascular import build_vascular

__all__ = [
    "SYSTEMS",
    "build_organs",
    "build_skeleton",
    "build_system",
    "build_vascular",
]
