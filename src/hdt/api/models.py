"""Pydantic models for request and response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class BodyLimbMeasurements(BaseModel):
    """Limb lengths in physical meters."""

    left_upper_arm: float = Field(..., description="Left upper arm length in meters")
    left_forearm: float = Field(..., description="Left forearm length in meters")
    right_upper_arm: float = Field(..., description="Right upper arm length in meters")
    right_forearm: float = Field(..., description="Right forearm length in meters")
    shoulder_width: float = Field(..., description="Shoulder biacromial width in meters")
    left_arm_total: float = Field(default=0.0, description="Total left arm length in meters")
    right_arm_total: float = Field(default=0.0, description="Total right arm length in meters")
    unit: str = Field(default="meters", description="Measurement units")


class BodyScaleFactors(BaseModel):
    """Dynamic scale factors relative to baseline/reference anatomy."""

    left_upper_arm: float = Field(..., description="Left upper arm scale factor")
    left_forearm: float = Field(..., description="Left forearm scale factor")
    right_upper_arm: float = Field(..., description="Right upper arm scale factor")
    right_forearm: float = Field(..., description="Right forearm scale factor")
    shoulder_width: float = Field(..., description="Shoulder width scale factor")
    overall: float = Field(..., description="Overall normalized body scale factor")


class BodyScanPayload(BaseModel):
    """Real-time body scan WebSocket transmission payload."""

    status: str = Field(
        ..., description="Tracking status ('detected', 'searching', 'camera_unavailable', etc.)"
    )
    detected: bool = Field(..., description="Whether a person is currently detected in frame")
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
