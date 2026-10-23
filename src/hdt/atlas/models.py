"""Pydantic data models for the anatomical atlas in Body Explorer."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Node Names
# ---------------------------------------------------------------------------

JOINTS = (
    "head",
    "neck",
    "chest",
    "pelvis",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "l_foot",
    "r_foot",
)

SPINAL_CORD_LEVELS = (
    # Caudal to Rostral order
    "cord_S5",
    "cord_S4",
    "cord_S3",
    "cord_S2",
    "cord_S1",
    "cord_L5",
    "cord_L4",
    "cord_L3",
    "cord_L2",
    "cord_L1",
    "cord_T12",
    "cord_T11",
    "cord_T10",
    "cord_T9",
    "cord_T8",
    "cord_T7",
    "cord_T6",
    "cord_T5",
    "cord_T4",
    "cord_T3",
    "cord_T2",
    "cord_T1",
    "cord_C8",
    "cord_C7",
    "cord_C6",
    "cord_C5",
    "cord_C4",
    "cord_C3",
    "cord_C2",
    "cord_C1",
)

CENTRAL_NODES = ("brainstem", "thalamus", "cortex_S1")

ALL_NODES = (*JOINTS, *SPINAL_CORD_LEVELS, *CENTRAL_NODES)


class NodeName(str, Enum):
    """Skeletal joints, spinal cord segments, and central sensory relay nodes."""

    # Skeletal Joints
    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    PELVIS = "pelvis"
    L_SHOULDER = "l_shoulder"
    R_SHOULDER = "r_shoulder"
    L_ELBOW = "l_elbow"
    R_ELBOW = "r_elbow"
    L_WRIST = "l_wrist"
    R_WRIST = "r_wrist"
    L_HAND = "l_hand"
    R_HAND = "r_hand"
    L_HIP = "l_hip"
    R_HIP = "r_hip"
    L_KNEE = "l_knee"
    R_KNEE = "r_knee"
    L_ANKLE = "l_ankle"
    R_ANKLE = "r_ankle"
    L_FOOT = "l_foot"
    R_FOOT = "r_foot"

    # Spinal Cord Levels (Caudal to Rostral)
    CORD_S5 = "cord_S5"
    CORD_S4 = "cord_S4"
    CORD_S3 = "cord_S3"
    CORD_S2 = "cord_S2"
    CORD_S1 = "cord_S1"
    CORD_L5 = "cord_L5"
    CORD_L4 = "cord_L4"
    CORD_L3 = "cord_L3"
    CORD_L2 = "cord_L2"
    CORD_L1 = "cord_L1"
    CORD_T12 = "cord_T12"
    CORD_T11 = "cord_T11"
    CORD_T10 = "cord_T10"
    CORD_T9 = "cord_T9"
    CORD_T8 = "cord_T8"
    CORD_T7 = "cord_T7"
    CORD_T6 = "cord_T6"
    CORD_T5 = "cord_T5"
    CORD_T4 = "cord_T4"
    CORD_T3 = "cord_T3"
    CORD_T2 = "cord_T2"
    CORD_T1 = "cord_T1"
    CORD_C8 = "cord_C8"
    CORD_C7 = "cord_C7"
    CORD_C6 = "cord_C6"
    CORD_C5 = "cord_C5"
    CORD_C4 = "cord_C4"
    CORD_C3 = "cord_C3"
    CORD_C2 = "cord_C2"
    CORD_C1 = "cord_C1"

    # Central Supraspinal Nodes
    BRAINSTEM = "brainstem"
    THALAMUS = "thalamus"
    CORTEX_S1 = "cortex_S1"


# ---------------------------------------------------------------------------
# Fiber Types & Properties
# ---------------------------------------------------------------------------


class FiberType(str, Enum):
    """Primary sensory nerve fiber classifications."""

    A_BETA = "A_beta"
    A_DELTA = "A_delta"
    C = "C"


class FiberProperties(BaseModel):
    """Electrophysiological conduction properties for a fiber type."""

    type: FiberType
    name: str
    default_velocity_mps: float = Field(
        ..., description="Default conduction velocity in metres per second (m/s)"
    )
    velocity_range_mps: tuple[float, float] = Field(
        ..., description="Plausible physiological conduction velocity range (min, max) in m/s"
    )
    modalities: list[str] = Field(
        ..., description="Primary sensory modalities mediated by this fiber"
    )


FIBER_PROPERTIES_TABLE: dict[FiberType, FiberProperties] = {
    FiberType.A_BETA: FiberProperties(
        type=FiberType.A_BETA,
        name="A-beta (Aβ)",
        default_velocity_mps=50.0,
        velocity_range_mps=(30.0, 70.0),
        modalities=["touch", "vibration", "pressure", "proprioception"],
    ),
    FiberType.A_DELTA: FiberProperties(
        type=FiberType.A_DELTA,
        name="A-delta (Aδ)",
        default_velocity_mps=15.0,
        velocity_range_mps=(5.0, 30.0),
        modalities=["fast pain", "cold temperature", "pinprick"],
    ),
    FiberType.C: FiberProperties(
        type=FiberType.C,
        name="C fiber",
        default_velocity_mps=1.0,
        velocity_range_mps=(0.5, 2.0),
        modalities=["slow pain", "warm temperature", "itch"],
    ),
}


# ---------------------------------------------------------------------------
# Dermatomes & Conditions
# ---------------------------------------------------------------------------


class BodyZone(BaseModel):
    """Anatomical body zone categorization for a dermatome."""

    kind: Literal["torso_band", "arm", "leg", "head_neck"] = Field(
        ..., description="General morphological zone"
    )
    description: str = Field(..., description="Anatomical location description")
    y_fraction: tuple[float, float] | None = Field(
        default=None,
        description="Fraction of standing height [min, max] in [0, 1] for torso bands only",
    )
    limb_part: str | None = Field(
        default=None,
        description="Specific limb region (e.g. 'lateral forearm', 'lateral calf')",
    )


class Dermatome(BaseModel):
    """Sensory cutaneous dermatome and its ascending neural transmission path."""

    id: str = Field(..., description="Unique dermatome identifier, e.g. 'L5'")
    root: str = Field(..., description="Spinal root level, e.g. 'L5'")
    region: str = Field(..., description="Plain-English cutaneous skin area")
    body_zone: BodyZone = Field(..., description="Zone category and boundary metadata")
    peripheral_nerve: str = Field(
        ..., description="Major peripheral nerve innervating this territory"
    )
    path: list[NodeName] = Field(
        ...,
        description="Ascending neural pathway from skin/joint into cord and up to cortex_S1",
    )
    tract: str = Field(
        ...,
        description="Ascending spinal tract (e.g. 'spinothalamic', 'dorsal column')",
    )
    notes: str | None = Field(default=None, description="Optional clinical/anatomical notes")


class Condition(BaseModel):
    """Common peripheral nerve or radicular clinical condition."""

    id: str = Field(..., description="Unique condition identifier, e.g. 'sciatica'")
    name: str = Field(..., description="Clinical diagnostic name")
    description: str = Field(..., description="Pathophysiological overview")
    compressed_roots: list[str] = Field(
        ..., description="Affected spinal nerve roots or cord levels"
    )
    affected_dermatomes: list[str] = Field(
        ..., description="Dermatomes experiencing sensory alteration"
    )
    symptoms: list[str] = Field(..., description="Characteristic clinical symptoms")


class Atlas(BaseModel):
    """Complete educational anatomical neural atlas."""

    disclaimer: str = Field(
        ..., description="Educational disclaimer regarding research/prototype status"
    )
    dermatomes: list[Dermatome] = Field(
        ..., description="Cutaneous dermatomes ordered anatomically"
    )
    conditions: list[Condition] = Field(
        ..., description="Representative clinical radicular/neuropathic conditions"
    )
