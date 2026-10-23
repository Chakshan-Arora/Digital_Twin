"""Geometry and distance transform modules."""

from hdt.geometry.distances import (
    DistanceFieldResult,
    clearance_from_sdf,
    closest_points_mm,
    mesh_clearance_mm,
    signed_clearance_mm,
    signed_distance_field_mm,
    surface_distance_stats,
)
from hdt.geometry.resolution import compute_resolution_block

__all__ = [
    "DistanceFieldResult",
    "clearance_from_sdf",
    "closest_points_mm",
    "compute_resolution_block",
    "mesh_clearance_mm",
    "signed_clearance_mm",
    "signed_distance_field_mm",
    "surface_distance_stats",
]
