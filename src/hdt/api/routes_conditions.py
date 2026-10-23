"""API routes for clinical condition simulation and dermatome lesion picking.

Research prototype - NOT a clinical diagnostic tool.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from hdt.atlas.data import get_default_atlas
from hdt.atlas.models import FiberType
from hdt.body.models import BodyParams, validate_body_params
from hdt.conditions.lesions import locate_lesions, path_block_index
from hdt.conditions.models import (
    ConditionRequest,
    ConditionResult,
    Severity,
    SignalStatus,
)
from hdt.conditions.simulate import simulate_condition
from hdt.nervous.anchors import nearest_dermatome
from hdt.nervous.geometry import full_pathway_polyline
from hdt.nervous.signal import SignalTimeline, signal_timeline

router = APIRouter(tags=["conditions"])


# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------


class ConditionSummary(BaseModel):
    """Brief summary of a clinical condition."""

    id: str = Field(..., description="Condition identifier (e.g. 'sciatica')")
    name: str = Field(..., description="Clinical diagnostic name")


class PathwayPoint(BaseModel):
    """Labelled anatomical coordinate along the ascending transmission path."""

    label: str = Field(..., description="Anatomical node name")
    point_ras_mm: tuple[float, float, float] = Field(
        ..., description="3D spatial coordinate [x, y, z] in RAS millimetres"
    )


class PickConditionRequest(BaseModel):
    """Request payload for dermatome point picking under an active condition."""

    body: BodyParams = Field(..., description="Subject skeletal model")
    point_ras: tuple[float, float, float] = Field(
        ..., description="Query surface coordinate [x, y, z] in RAS mm"
    )
    condition_id: str = Field(..., description="Clinical condition ID, e.g. 'sciatica'")
    side: Literal["l", "r", "both"] = Field(
        default="r", description="Condition lateralisation ('l', 'r', 'both')"
    )
    severity: Severity = Field(
        default=Severity.MODERATE, description="Clinical severity grading"
    )


class AffectedFiberTimelineItem(BaseModel):
    """Per-fiber conduction latency under condition impairment."""

    fiber_type: FiberType
    conduction_velocity_mps: float
    normal_latency_ms: float
    affected_latency_ms: float | None = Field(
        default=None, description="Altered latency in ms (null if blocked)"
    )
    status: SignalStatus


class AffectedSignalTimeline(BaseModel):
    """Multi-fiber conduction timeline reflecting pathological degradation."""

    dermatome_id: str
    side: str
    total_path_length_mm: float
    status: SignalStatus
    fibers: dict[FiberType, AffectedFiberTimelineItem]
    notes: str = Field(
        default="Pathologically altered axonal conduction timeline. Educational simulation only."
    )


class PickConditionResponse(BaseModel):
    """Result of dermatome pick with lesion interruption and affected conduction timeline."""

    dermatome_id: str
    region: str
    side: Literal["l", "r"]
    distance_to_anchor_mm: float
    pathway: list[PathwayPoint]
    timeline: SignalTimeline
    blocked: bool = Field(
        ..., description="Whether ascending sensory transmission is interrupted"
    )
    block_index: int | None = Field(
        default=None,
        description="Index along pathway polyline where transmission stops (null if unblocked)",
    )
    block_point: tuple[float, float, float] | None = Field(
        default=None,
        description="3D coordinate in RAS mm where signal pulse terminates (null if unblocked)",
    )
    affected_timeline: AffectedSignalTimeline = Field(
        ..., description="Multi-fiber conduction timeline with affected latencies"
    )
    disclaimer: str = Field(
        ..., description="Educational non-clinical research prototype disclaimer"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/conditions",
    response_model=list[ConditionSummary],
    summary="List available conditions",
    description="Returns list of clinical condition identifiers and names.",
)
def list_conditions() -> list[ConditionSummary]:
    """Return lightweight list of all clinical conditions."""
    atlas = get_default_atlas()
    return [ConditionSummary(id=c.id, name=c.name) for c in atlas.conditions]


@router.post(
    "/conditions/simulate",
    response_model=ConditionResult,
    summary="Simulate condition signal loss and 3D lesions",
    description=(
        "Simulates neuro-anatomical conduction slowing, fiber drop-out, and 3D focal "
        "lesion sites for an anthropometric body model."
    ),
)
def post_simulate_condition(request: ConditionRequest) -> ConditionResult:
    """Simulate functional signal loss and lesion mapping."""
    # 1. Validate body parameters
    errors = validate_body_params(request.body)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors},
        )

    # 2. Validate condition
    atlas = get_default_atlas()
    cond_obj = next((c for c in atlas.conditions if c.id == request.condition_id), None)
    if cond_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Condition '{request.condition_id}' not found in atlas.",
        )

    # 3. Validate side
    if request.side not in ("l", "r", "both"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported side '{request.side}'. Must be 'l', 'r', or 'both'.",
        )

    return simulate_condition(
        body=request.body,
        condition=cond_obj,
        side=request.side,
        severity=request.severity,
    )


@router.post(
    "/conditions/pick",
    response_model=PickConditionResponse,
    summary="Pick dermatome and compute condition pathway interruption",
    description=(
        "Identifies stimulated dermatome from surface click, computes ascending pathway, "
        "and checks for lesion interruption, stopping the pulse at the block point."
    ),
)
def pick_condition_pathway(request: PickConditionRequest) -> PickConditionResponse:
    """Process click-to-pathway with condition lesion interruption."""
    # 1. Validate body parameters
    errors = validate_body_params(request.body)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors},
        )

    # 2. Validate condition
    atlas = get_default_atlas()
    cond_obj = next((c for c in atlas.conditions if c.id == request.condition_id), None)
    if cond_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Condition '{request.condition_id}' not found in atlas.",
        )

    # 3. Validate side
    if request.side not in ("l", "r", "both"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported side '{request.side}'. Must be 'l', 'r', or 'both'.",
        )

    # 4. Identify stimulated dermatome
    dermatome_id, side, distance_mm = nearest_dermatome(request.body, request.point_ras)
    dermatome = next((d for d in atlas.dermatomes if d.id == dermatome_id), None)
    region_text = dermatome.region if dermatome else f"Dermatome {dermatome_id}"

    # 5. Build ascending labelled 3D pathway and baseline timeline
    path_labelled = full_pathway_polyline(request.body, dermatome_id, side=side)
    pathway_points = [
        PathwayPoint(label=label, point_ras_mm=pt) for label, pt in path_labelled
    ]
    base_timeline = signal_timeline(request.body, dermatome_id, side=side)

    # 6. Locate lesions on network for this condition
    lesions, _ = locate_lesions(request.body, cond_obj, side=request.side)

    # 7. Check pathway blockage
    # Is this dermatome affected by the condition?
    sim_res = simulate_condition(
        request.body, cond_obj, side=request.side, severity=request.severity
    )
    affected_effect = next(
        (
            de
            for de in sim_res.dermatome_effects
            if de.dermatome_id == dermatome_id and de.side == side
        ),
        None,
    )

    is_dermatome_affected = affected_effect is not None
    is_fully_blocked = (
        is_dermatome_affected
        and (request.severity == Severity.SEVERE or cond_obj.id == "sci_t10")
    )

    blk_idx: int | None = None
    blk_pt: tuple[float, float, float] | None = None

    if is_dermatome_affected:
        blk_idx = path_block_index(pathway_points, lesions)
        if blk_idx is not None and 0 <= blk_idx < len(pathway_points):
            blk_pt = pathway_points[blk_idx].point_ras_mm
        elif is_fully_blocked and len(pathway_points) > 1:
            blk_idx = 1
            blk_pt = pathway_points[1].point_ras_mm

    # 8. Compute affected timeline
    affected_fibers_dict: dict[FiberType, AffectedFiberTimelineItem] = {}
    overall_status = SignalStatus.NORMAL

    for f_type in [FiberType.A_BETA, FiberType.A_DELTA, FiberType.C]:
        b_item = base_timeline.fibers[f_type]
        if not is_dermatome_affected:
            affected_fibers_dict[f_type] = AffectedFiberTimelineItem(
                fiber_type=f_type,
                conduction_velocity_mps=b_item.conduction_velocity_mps,
                normal_latency_ms=b_item.latency_ms,
                affected_latency_ms=b_item.latency_ms,
                status=SignalStatus.NORMAL,
            )
        else:
            f_eff = affected_effect.fibers[f_type]
            if f_eff.status == SignalStatus.BLOCKED:
                overall_status = SignalStatus.BLOCKED
            elif f_eff.status == SignalStatus.SLOWED and overall_status != SignalStatus.BLOCKED:
                overall_status = SignalStatus.SLOWED

            aff_vel = (
                f_eff.affected_velocity_mps
                if f_eff.affected_velocity_mps is not None
                else b_item.conduction_velocity_mps
            )
            affected_fibers_dict[f_type] = AffectedFiberTimelineItem(
                fiber_type=f_type,
                conduction_velocity_mps=aff_vel,
                normal_latency_ms=b_item.latency_ms,
                affected_latency_ms=f_eff.affected_latency_ms,
                status=f_eff.status,
            )

    affected_tl = AffectedSignalTimeline(
        dermatome_id=dermatome_id,
        side=side,
        total_path_length_mm=base_timeline.total_path_length_mm,
        status=overall_status,
        fibers=affected_fibers_dict,
    )

    return PickConditionResponse(
        dermatome_id=dermatome_id,
        region=region_text,
        side=side,
        distance_to_anchor_mm=distance_mm,
        pathway=pathway_points,
        timeline=base_timeline,
        blocked=(blk_idx is not None and is_fully_blocked),
        block_index=blk_idx if is_fully_blocked else None,
        block_point=blk_pt if is_fully_blocked else None,
        affected_timeline=affected_tl,
        disclaimer=atlas.disclaimer,
    )
