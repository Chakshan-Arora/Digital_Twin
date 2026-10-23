"""Electrophysiological nerve conduction modeling and latency timelines.

Calculates pure axonal conduction time along peripheral and spinal sensory pathways
across primary fiber types (A-beta, A-delta, C).
"""

from typing import Literal

from pydantic import BaseModel, Field

from hdt.atlas.models import FIBER_PROPERTIES_TABLE, FiberType
from hdt.body.models import BodyParams
from hdt.nervous.geometry import full_pathway_polyline, total_length_mm


class FiberTimelineItem(BaseModel):
    """Conduction parameters and estimated transmission latency for a fiber type."""

    fiber_type: FiberType
    name: str
    total_path_length_mm: float = Field(
        ..., description="Total 3D pathway distance from skin receptor to cortex_S1 in mm"
    )
    conduction_velocity_mps: float = Field(
        ..., description="Axonal conduction velocity used for standard latency in m/s"
    )
    latency_ms: float = Field(
        ..., description="Conduction latency in milliseconds (length_mm / velocity_mps)"
    )
    latency_range_ms: tuple[float, float] = Field(
        ...,
        description=(
            "Plausible physiological latency range [fastest, slowest] in ms "
            "based on velocity bounds"
        ),
    )


class SignalTimeline(BaseModel):
    """Complete multi-fiber conduction latency timeline for a cutaneous stimulus."""

    dermatome_id: str
    side: str
    total_path_length_mm: float
    fibers: dict[FiberType, FiberTimelineItem]
    notes: str = Field(
        default=(
            "Axonal conduction time only. Ignores central chemical synaptic "
            "delays (approx 0.5 - 2.0 ms per synaptic relay)."
        )
    )


def signal_timeline(
    body: BodyParams,
    dermatome_id: str,
    side: Literal["l", "r", "both"] = "r",
) -> SignalTimeline:
    """Calculate multi-fiber conduction velocities and arrival latencies to cortex_S1.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model.
    dermatome_id : str
        Stimulated dermatome identifier (e.g. 'L5', 'C6', 'T10').
    side : {'l', 'r', 'both'}
        Anatomical stimulation side.

    Returns
    -------
    SignalTimeline
        Conduction timeline broken down by A-beta, A-delta, and C fibers.

    Notes
    -----
    Models pure axonal conduction time (length_mm / velocity_mps) along the
    reconstructed 3D peripheral and spinal trajectory. Synaptic neurotransmission
    delays at the dorsal horn, thalamus, and cortex are excluded as an educational
    simplification.
    """
    key = dermatome_id.upper()
    target_side = "r" if side == "both" else side.lower()

    # Generate complete labelled pathway
    labelled_path = full_pathway_polyline(body, key, side=target_side)
    path_pts = [pt for _, pt in labelled_path]
    path_len_mm = total_length_mm(path_pts)

    fibers_dict: dict[FiberType, FiberTimelineItem] = {}

    for f_type in [FiberType.A_BETA, FiberType.A_DELTA, FiberType.C]:
        prop = FIBER_PROPERTIES_TABLE[f_type]
        v_def = prop.default_velocity_mps
        v_min, v_max = prop.velocity_range_mps

        # latency (ms) = (length (m) / velocity (m/s)) * 1000 = length_mm / velocity_mps
        lat_ms = round(path_len_mm / v_def, 2)
        fastest_ms = round(path_len_mm / v_max, 2)
        slowest_ms = round(path_len_mm / v_min, 2)

        fibers_dict[f_type] = FiberTimelineItem(
            fiber_type=f_type,
            name=prop.name,
            total_path_length_mm=path_len_mm,
            conduction_velocity_mps=v_def,
            latency_ms=lat_ms,
            latency_range_ms=(fastest_ms, slowest_ms),
        )

    return SignalTimeline(
        dermatome_id=key,
        side=target_side,
        total_path_length_mm=path_len_mm,
        fibers=fibers_dict,
    )
