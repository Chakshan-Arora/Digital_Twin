"""Pydantic models for request and response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hdt.atlas.models import Atlas, Condition, Dermatome
from hdt.body.models import BodyParams, JointName
from hdt.nervous.signal import SignalTimeline


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class StructureSummary(BaseModel):
    """Summary of a segmented structure within a case."""

    name: str = Field(..., description="Unique structure name (e.g. 'tumor', 'vessel')")
    role: Literal["target", "risk", "context"] = Field(
        ..., description="Functional role of the structure"
    )


class CaseSummary(BaseModel):
    """Summary representation of an imaging case."""

    id: str = Field(..., description="Unique opaque case identifier")
    name: str = Field(..., description="Human-readable case name")
    structures: list[StructureSummary] = Field(
        ..., description="List of segmented structures and roles"
    )


class CaseDetail(BaseModel):
    """Detailed spatial geometry and structure definitions for a case."""

    id: str = Field(..., description="Unique opaque case identifier")
    name: str = Field(..., description="Human-readable case name")
    primary_image_name: str = Field(
        default="t1", description="Name of the default/primary image modality"
    )
    spacing_mm: list[float] = Field(
        ..., description="Voxel spacing (sx, sy, sz) in millimetres under RAS convention"
    )
    spacing_zyx_mm: list[float] = Field(
        ..., description="Internal voxel spacing in NumPy (z, y, x) order in mm"
    )
    shape: list[int] = Field(
        ..., description="Volume dimensions in NumPy (z, y, x) order"
    )
    shape_zyx: list[int] = Field(
        ..., description="Internal volume shape in NumPy (z, y, x) order"
    )
    size: list[int] = Field(
        ..., description="Volume dimensions (size_x, size_y, size_z) in voxels"
    )
    origin_ras_mm: list[float] = Field(
        ..., description="Physical origin (x, y, z) in millimetres under RAS convention"
    )
    direction_ras: list[list[float]] = Field(
        ..., description="3x3 direction cosine matrix in RAS space"
    )
    structures: list[StructureSummary] = Field(
        ..., description="List of segmented structures and roles"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary case-level metadata"
    )
    coordinate_system: str = Field(
        default="RAS", description="Spatial coordinate convention (always 'RAS')"
    )


class ClearanceRequest(BaseModel):
    """Request payload for pairwise mask-to-mask clearance computation."""

    target_name: str | None = Field(default=None, description="Target structure name")
    risk_name: str | None = Field(default=None, description="Risk structure name")
    target: str | None = Field(default=None, description="Alternative alias for target_name")
    risk: str | None = Field(default=None, description="Alternative alias for risk_name")

    @model_validator(mode="after")
    def validate_names(self) -> "ClearanceRequest":
        """Ensure either target_name/target and risk_name/risk are provided."""
        effective_target = self.target_name or self.target
        effective_risk = self.risk_name or self.risk
        if not effective_target:
            raise ValueError("target_name or target must be provided")
        if not effective_risk:
            raise ValueError("risk_name or risk must be provided")
        self.target_name = effective_target
        self.risk_name = effective_risk
        return self


class BBoxRAS(BaseModel):
    """Axis-aligned bounding box coordinates in physical RAS space."""

    min: list[float] = Field(..., description="Minimum bounding coordinates [x, y, z] in RAS mm")
    max: list[float] = Field(..., description="Maximum bounding coordinates [x, y, z] in RAS mm")


class StructureStats(BaseModel):
    """Geometric and volumetric statistics for a segmented structure."""

    volume_mm3: float = Field(..., description="Structure volume in cubic millimetres")
    voxel_count: int = Field(..., description="Total number of foreground voxels")
    bbox_ras_mm: BBoxRAS = Field(..., description="Axis-aligned bounding box in RAS millimetres")
    centroid_ras_mm: list[float] = Field(
        ..., description="Center of mass coordinates [x, y, z] in RAS mm"
    )
    n_components: int = Field(
        ..., description="Number of connected components using 6-connectivity"
    )
    component_volumes_mm3: list[float] = Field(
        ..., description="Volumes of the largest 10 connected components in mm^3, sorted descending"
    )
    coordinate_system: str = Field(
        default="RAS", description="Spatial coordinate convention (always 'RAS')"
    )


class ResolutionBlock(BaseModel):
    """Voxel grid resolution and directional extent along closest voxel pair."""

    spacing_mm: list[float] = Field(..., description="Voxel spacing (sz, sy, sx) in millimetres")
    voxel_diagonal_mm: float = Field(..., description="Voxel diagonal length in millimetres")
    voxel_extent_along_pair_mm: float = Field(
        ...,
        description=(
            "Voxel extent along closest voxel pair direction (2t term) in millimetres"
        ),
    )


class ClearanceResponse(BaseModel):
    """Response payload for pairwise mask-to-mask clearance."""

    model_config = ConfigDict(extra="allow")

    clearance_mm: float = Field(
        ...,
        description=(
            "Headline face-to-face clearance in mm. "
            "Positive = gap, 0 = touch, negative = penetration"
        ),
    )
    convention: str = Field(
        default="face_to_face_voxel_sets",
        description="Geometric distance convention applied",
    )
    closest_points_mm: dict[str, Any] = Field(
        ...,
        description="Closest point coordinates in RAS physical millimetres and centre distance",
    )
    surface_distance_stats: dict[str, float] = Field(
        ...,
        description="Symmetric surface distance metrics (centre_min_mm, mean, hd95, max) in mm",
    )
    resolution: ResolutionBlock = Field(
        ...,
        description="Voxel grid resolution and directional extent along closest voxel pair",
    )
    coordinate_system: str = Field(
        default="RAS", description="Spatial coordinate convention (always 'RAS')"
    )


class UncertaintyRequest(BaseModel):
    """Request payload for clearance uncertainty analysis under smooth boundary perturbation."""

    target_name: str | None = Field(default=None, description="Target structure name to perturb")
    risk_name: str | None = Field(default=None, description="Risk structure name")
    target: str | None = Field(default=None, description="Alternative alias for target_name")
    risk: str | None = Field(default=None, description="Alternative alias for risk_name")
    sigma_mm: float = Field(
        ..., ge=0.0, description="Boundary perturbation amplitude standard deviation in mm"
    )
    corr_length_mm: float = Field(
        ..., gt=0.0, description="Spatial correlation length in mm (must be positive)"
    )
    bias_sigma_mm: float = Field(
        default=0.0, ge=0.0, description="Global volumetric bias standard deviation in mm"
    )
    n_samples: int = Field(
        default=300,
        ge=1,
        le=300,
        description="Number of Monte Carlo realizations (between 1 and 300; cap is 300)",
    )
    seed: int | None = Field(
        default=None, description="Optional random seed for deterministic realizations"
    )
    threshold_mm: float = Field(
        default=2.0, description="Clearance threshold in mm for probability estimation"
    )

    @model_validator(mode="after")
    def validate_names(self) -> "UncertaintyRequest":
        """Ensure target and risk names are set."""
        effective_target = self.target_name or self.target
        effective_risk = self.risk_name or self.risk
        if not effective_target:
            raise ValueError("target_name or target must be provided")
        if not effective_risk:
            raise ValueError("risk_name or risk must be provided")
        self.target_name = effective_target
        self.risk_name = effective_risk
        return self


class UncertaintyResponse(BaseModel):
    """Response payload for clearance uncertainty analysis."""

    baseline_mm: float
    samples_mm: list[float]
    mean: float
    std: float
    p05: float
    p50: float
    p95: float
    p_below_threshold: float
    p_below_ci95: list[float]
    p_overlap: float
    p_touching: float
    n_samples: int
    seed: int
    threshold_mm: float
    params: dict[str, Any]
    censored_fraction: float
    resolution: ResolutionBlock
    convention: str = "face_to_face_voxel_sets"
    coordinate_system: str = "RAS"


class AtlasResponse(Atlas):
    """Response schema for the complete anatomical neural atlas."""


class DermatomeResponse(Dermatome):
    """Response schema for a single cutaneous dermatome."""


class ConditionResponse(Condition):
    """Response schema for a clinical nerve condition."""


# ---------------------------------------------------------------------------
# Body Explorer & Dermatome Picking Models
# ---------------------------------------------------------------------------


class BodyPresetSummary(BaseModel):
    """Summary metadata for a stature preset."""

    name: str = Field(..., description="Preset identifier ('short', 'average', 'tall')")
    height_mm: float = Field(..., description="Standing stature in millimetres")


class PickRequest(BaseModel):
    """Request payload for dermatome point picking."""

    body: BodyParams = Field(..., description="Subject skeletal model")
    point_ras: tuple[float, float, float] = Field(
        ..., description="Query surface point [x, y, z] in RAS millimetres"
    )


class PathwayPoint(BaseModel):
    """Labelled anatomical coordinate along the ascending transmission path."""

    label: str = Field(
        ..., description="Anatomical node name (e.g. skin_anchor, cord_L5, cortex_S1)"
    )
    point_ras_mm: tuple[float, float, float] = Field(
        ..., description="3D spatial coordinate [x, y, z] in RAS millimetres"
    )


class PickResponse(BaseModel):
    """Result of dermatome pick with ascending pathway polyline and conduction timeline."""

    dermatome_id: str = Field(..., description="Identified dermatome ID (e.g. 'L5')")
    region: str = Field(..., description="Plain-English anatomical skin territory from atlas")
    side: Literal["l", "r"] = Field(..., description="Anatomical stimulation side ('l' or 'r')")
    distance_to_anchor_mm: float = Field(
        ..., description="3D Euclidean distance to nearest skin anchor in mm"
    )
    pathway: list[PathwayPoint] = Field(
        ..., description="Labelled polyline tracing peripheral nerve, cord ascent, and cortex_S1"
    )
    timeline: SignalTimeline = Field(
        ..., description="Multi-fiber conduction latency calculations"
    )
    disclaimer: str = Field(
        ..., description="Educational non-clinical research prototype disclaimer"
    )


class BodyShapeRequest(BaseModel):
    """Request payload for body shape segments."""

    height_mm: float | None = Field(default=None, description="Standing stature in mm")
    joints: dict[JointName, tuple[float, float, float]] | None = Field(
        default=None, description="Joint positions in RAS mm"
    )
    source: Literal["scan", "preset"] = Field(
        default="preset", description="Origin source of the body parameters"
    )
    body: BodyParams | None = Field(
        default=None, description="Nested BodyParams model if supplied"
    )

    def extract_body_params(self) -> BodyParams:
        """Extract or construct the canonical BodyParams model."""
        if self.body is not None:
            return self.body
        if self.height_mm is not None and self.joints is not None:
            return BodyParams(
                height_mm=self.height_mm, joints=self.joints, source=self.source
            )
        raise ValueError("Must provide either 'body' or ('height_mm' and 'joints').")


class BodyNervousRequest(BaseModel):
    """Request payload for nervous network generation."""

    height_mm: float | None = Field(default=None, description="Standing stature in mm")
    joints: dict[JointName, tuple[float, float, float]] | None = Field(
        default=None, description="Joint positions in RAS mm"
    )
    source: Literal["scan", "preset"] = Field(
        default="preset", description="Origin source of the body parameters"
    )
    body: BodyParams | None = Field(
        default=None, description="Nested BodyParams model if supplied"
    )
    detail: Literal["low", "high"] = Field(
        default="high", description="Mesh/polyline resolution preset"
    )

    def extract_body_params(self) -> BodyParams:
        """Extract or construct the canonical BodyParams model."""
        if self.body is not None:
            return self.body
        if self.height_mm is not None and self.joints is not None:
            return BodyParams(
                height_mm=self.height_mm, joints=self.joints, source=self.source
            )
        raise ValueError("Must provide either 'body' or ('height_mm' and 'joints').")


# ---------------------------------------------------------------------------
# Real-Time Camera Body Scanner Models
# ---------------------------------------------------------------------------


class BodyLimbMeasurements(BaseModel):
    """Calculated 3D limb lengths in meters."""

    left_upper_arm: float = Field(..., description="Left humerus length in meters")
    left_forearm: float = Field(..., description="Left radius/ulna length in meters")
    right_upper_arm: float = Field(..., description="Right humerus length in meters")
    right_forearm: float = Field(..., description="Right radius/ulna length in meters")
    left_arm_total: float = Field(..., description="Total left arm length in meters")
    right_arm_total: float = Field(..., description="Total right arm length in meters")
    shoulder_width: float = Field(..., description="Biacromial shoulder width in meters")
    unit: str = Field(default="meters", description="Measurement unit")


class BodyScaleFactors(BaseModel):
    """Proportional scale factors relative to average adult reference."""

    left_upper_arm: float = Field(..., description="Scale factor for left upper arm")
    left_forearm: float = Field(..., description="Scale factor for left forearm")
    right_upper_arm: float = Field(..., description="Scale factor for right upper arm")
    right_forearm: float = Field(..., description="Scale factor for right forearm")
    left_arm_total: float = Field(..., description="Scale factor for left arm")
    right_arm_total: float = Field(..., description="Scale factor for right arm")
    shoulder_width: float = Field(..., description="Scale factor for shoulder width")


class BodyScanResponse(BaseModel):
    """Real-time body scanner frame result broadcast via WebSocket or polling."""

    status: str = Field(..., description="'active', 'calibrated', 'error', 'no_person'")
    person_detected: bool = Field(
        ..., description="Whether a person is currently detected in frame"
    )
    message: str = Field(..., description="Status message or error description")
    timestamp: float = Field(..., description="UNIX epoch timestamp in seconds")
    fps: float = Field(default=0.0, description="Processing frames per second")
    measurements: BodyLimbMeasurements | None = Field(
        default=None, description="Calculated limb lengths in physical meters"
    )
    scale_factors: BodyScaleFactors | None = Field(
        default=None, description="Calculated limb scale factors"
    )
    raw_landmarks: dict[str, dict[str, float]] | None = Field(
        default=None, description="Raw 3D joint landmarks in meters"
    )
