"""Pydantic models for clinical conditions, nerve lesions, and signal loss simulation.

Research prototype - NOT a clinical diagnostic tool.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from hdt.atlas.models import FiberType
from hdt.body.models import BodyParams


class SignalStatus(str, Enum):
    """Signal propagation status across an axon or neural pathway."""

    NORMAL = "normal"
    SLOWED = "slowed"
    BLOCKED = "blocked"


class Severity(str, Enum):
    """Clinical severity grading for neuropathic or compressive lesions."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class FiberEffect(BaseModel):
    """Conduction latency and velocity alterations for an individual fiber type."""

    fiber_type: FiberType = Field(..., description="Fiber classification (A_beta, A_delta, C)")
    normal_latency_ms: float = Field(
        ..., description="Baseline physiological conduction latency in ms"
    )
    affected_latency_ms: float | None = Field(
        default=None,
        description="Pathologically altered conduction latency in ms (null if blocked)",
    )
    conduction_velocity_mps: float = Field(
        ..., description="Baseline conduction velocity in m/s"
    )
    affected_velocity_mps: float | None = Field(
        default=None,
        description="Pathologically altered conduction velocity in m/s (null if blocked)",
    )
    status: SignalStatus = Field(..., description="Signal transmission status for this fiber")


class DermatomeEffect(BaseModel):
    """Sensory impairment profile for a single dermatome."""

    dermatome_id: str = Field(..., description="Dermatome identifier (e.g. 'L5')")
    side: Literal["l", "r"] = Field(..., description="Anatomical side ('l' or 'r')")
    region: str = Field(..., description="Plain-English cutaneous skin territory")
    status: SignalStatus = Field(
        ..., description="Overall sensory transmission status for this dermatome"
    )
    fibers: dict[FiberType, FiberEffect] = Field(
        ..., description="Per-fiber electrophysiological effects and latencies"
    )


class NerveEffect(BaseModel):
    """Downstream functional impairment for a specific nerve tract."""

    nerve_id: str = Field(..., description="Anatomical nerve or branch name")
    side: Literal["l", "r", "midline"] = Field(..., description="Anatomical lateralisation")
    status: SignalStatus = Field(..., description="Functional signal transmission status")
    lesion_index: int | None = Field(
        default=None,
        description="Index along the nerve polyline where lesion occurs (null if unaffected)",
    )
    lesion_point: tuple[float, float, float] | None = Field(
        default=None,
        description="3D coordinates [x, y, z] of lesion in RAS mm (null if unaffected)",
    )


class Lesion(BaseModel):
    """Structural pathology location on the 3D nervous system network."""

    kind: Literal["root_compression", "peripheral_entrapment", "cord_transection"] = Field(
        ..., description="Pathological classification of the lesion"
    )
    label: str = Field(..., description="Descriptive pathology label")
    point: tuple[float, float, float] = Field(
        ..., description="3D focal lesion center [x, y, z] in RAS mm"
    )
    nerve_id: str | None = Field(
        default=None, description="Affected peripheral nerve identifier if applicable"
    )
    cord_level: str | None = Field(
        default=None, description="Spinal cord segment level if applicable (e.g. 'cord_T10')"
    )
    side: Literal["l", "r", "both"] = Field(..., description="Anatomical lateralisation")
    radius_mm: float = Field(
        default=12.0, description="Approximate spatial extent radius in mm for 3D visual rendering"
    )


class ConditionRequest(BaseModel):
    """Input parameters for condition simulation."""

    body: BodyParams = Field(..., description="Subject skeletal model")
    condition_id: str = Field(..., description="Condition identifier from atlas, e.g. 'sciatica'")
    side: Literal["l", "r", "both"] = Field(
        default="r", description="Anatomical stimulation / lesion lateralisation"
    )
    severity: Severity = Field(
        default=Severity.MODERATE, description="Clinical severity grading"
    )


class ConditionResult(BaseModel):
    """Complete condition simulation result payload."""

    condition_id: str = Field(..., description="Condition identifier from atlas")
    name: str = Field(..., description="Clinical diagnostic name")
    description: str = Field(..., description="Pathophysiological overview")
    symptoms: list[str] = Field(..., description="Characteristic clinical symptoms")
    compressed_roots: list[str] = Field(..., description="Affected spinal roots")
    dermatome_effects: list[DermatomeEffect] = Field(
        ..., description="Dermatomes experiencing signal slowing or blockade"
    )
    nerve_effects: list[NerveEffect] = Field(
        ..., description="Peripheral nerves and spinal roots affected by the condition"
    )
    lesions: list[Lesion] = Field(
        ..., description="3D focal lesion sites on the nerve network"
    )
    unaffected_dermatomes: list[str] = Field(
        ..., description="Dermatome IDs maintaining normal signal transmission"
    )
    disclaimer: str = Field(
        default=(
            "Educational simulation of neuropathic signal loss. Research prototype only; "
            "NOT a clinical diagnostic tool and never to be used for patient prognosis."
        ),
        description="Clinical and research disclaimer",
    )
