"""FastAPI server for uncertainty-aware 3D clearance analysis on medical scans.

Converts all external wire representations to RAS coordinate space via coords.py.
Research prototype - NOT a clinical tool.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import scipy.ndimage
import skimage.measure
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from hdt.api.body_scanner import DEFAULT_REFERENCE_LENGTHS, BodyScanner
from hdt.api.coords import (
    direction_to_ras,
    local_point_zyx_mm_to_ras_mm,
    mesh_signed_volume_ras,
    origin_to_ras_mm,
    transform_mesh_vertices_to_ras,
    voxel_to_ras_mm,
)
from hdt.api.models import (
    BBoxRAS,
    BodyNervousRequest,
    BodyPresetSummary,
    BodyShapeRequest,
    CaseDetail,
    CaseSummary,
    ClearanceRequest,
    ClearanceResponse,
    HealthResponse,
    PathwayPoint,
    PickRequest,
    PickResponse,
    ResolutionBlock,
    StructureStats,
    StructureSummary,
    UncertaintyRequest,
    UncertaintyResponse,
)
from hdt.api.repository import CaseRepository, FolderRepository, PhantomRepository
from hdt.api.routes_conditions import router as conditions_router
from hdt.api.routes_systems import router as systems_router
from hdt.atlas.data import get_default_atlas
from hdt.atlas.models import Atlas, Condition, Dermatome
from hdt.body.models import BodyParams, validate_body_params
from hdt.body.presets import PRESETS, get_preset
from hdt.body.shape import ShapeModelResponse, segments
from hdt.geometry.distances import closest_points_mm, signed_clearance_mm, surface_distance_stats
from hdt.geometry.resolution import compute_resolution_block
from hdt.imaging.load_scan import save_nifti_image, save_nifti_mask
from hdt.nervous.anchors import nearest_dermatome
from hdt.nervous.geometry import full_pathway_polyline
from hdt.nervous.network import NerveNetworkResponse, build_network
from hdt.nervous.signal import signal_timeline
from hdt.uncertainty.analyse_clearance import analyse_clearance

logger = logging.getLogger("hdt.api")

app = FastAPI(
    title="Human Digital Twin (HDT) Clearance API",
    description="Research prototype for uncertainty-aware 3D clearance analysis.",
    version="0.1.0",
)

# CORS configuration
cors_env = os.environ.get("HDT_CORS_ORIGINS")
if cors_env:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    allowed_origins = ["http://localhost:5173", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(systems_router)
app.include_router(conditions_router)

_default_repository = PhantomRepository()
_folder_repos: dict[str, FolderRepository] = {}


def get_repository() -> CaseRepository:
    """Dependency provider for case repository."""
    hdt_data_dir = os.environ.get("HDT_DATA_DIR")
    if hdt_data_dir:
        resolved = str(Path(hdt_data_dir).resolve())
        if resolved not in _folder_repos:
            _folder_repos[resolved] = FolderRepository(resolved)
        return _folder_repos[resolved]
    return _default_repository


RepoDep = Annotated[CaseRepository, Depends(get_repository)]


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/cases", response_model=list[CaseSummary])
def get_cases(repo: RepoDep) -> list[CaseSummary]:
    """List all available cases with their structures and roles."""
    summaries: list[CaseSummary] = []
    for case in repo.list_cases():
        structs = [
            StructureSummary(name=s.name, role=s.role)
            for s in case.structures.values()
        ]
        summaries.append(CaseSummary(id=case.case_id, name=case.case_id, structures=structs))
    return summaries


@app.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: str, repo: RepoDep) -> CaseDetail:
    """Get spatial metadata and structure list for a specific case in RAS coordinates."""
    result = repo.get_case(case_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    case, geom = result
    structs = [
        StructureSummary(name=s.name, role=s.role)
        for s in case.structures.values()
    ]
    origin_ras = origin_to_ras_mm(geom)
    direction_ras = direction_to_ras(geom)
    spacing_ras = [
        float(geom.spacing_mm[2]),
        float(geom.spacing_mm[1]),
        float(geom.spacing_mm[0]),
    ]

    return CaseDetail(
        id=case.case_id,
        name=case.case_id,
        primary_image_name=case.primary,
        spacing_mm=spacing_ras,
        spacing_zyx_mm=[float(s) for s in geom.spacing_mm],
        shape=list(geom.shape),
        shape_zyx=list(geom.shape),
        size=[int(geom.shape[2]), int(geom.shape[1]), int(geom.shape[0])],
        origin_ras_mm=list(origin_ras),
        direction_ras=direction_ras,
        structures=structs,
        metadata={},
        coordinate_system="RAS",
    )


@app.get("/cases/{case_id}/volume.nii.gz")
def get_case_volume(case_id: str, repo: RepoDep) -> Response:
    """Stream download the primary volumetric image as a NIfTI file."""
    result = repo.get_case(case_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    case, geom = result
    primary_name = case.primary
    image_data = case.images[primary_name]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{case_id}_volume.nii.gz"
        save_nifti_image(image_data, geom, tmp_path)
        content_bytes = tmp_path.read_bytes()

    headers = {"Content-Disposition": f'attachment; filename="{case_id}_volume.nii.gz"'}
    return Response(
        content=content_bytes,
        media_type="application/gzip",
        headers=headers,
    )


@app.get("/cases/{case_id}/structures/{structure_name}/mask.nii.gz")
def get_structure_mask(
    case_id: str, structure_name: str, repo: RepoDep
) -> Response:
    """Stream download the structure binary mask as a NIfTI file."""
    result = repo.get_case(case_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    case, geom = result
    if structure_name not in case.structures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Structure '{structure_name}' not found in case '{case_id}'.",
        )

    mask = case.structures[structure_name].mask

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{case_id}_{structure_name}_mask.nii.gz"
        save_nifti_mask(mask, geom, tmp_path)
        content_bytes = tmp_path.read_bytes()

    headers = {
        "Content-Disposition": f'attachment; filename="{case_id}_{structure_name}_mask.nii.gz"'
    }
    return Response(
        content=content_bytes,
        media_type="application/gzip",
        headers=headers,
    )


@app.get(
    "/cases/{case_id}/structures/{structure_name}/stats",
    response_model=StructureStats,
)
def get_structure_stats(
    case_id: str,
    structure_name: str,
    repo: RepoDep,
) -> StructureStats:
    """Compute geometric and volumetric statistics for a structure."""
    result = repo.get_case(case_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    case, geom = result
    if structure_name not in case.structures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Structure '{structure_name}' not found in case '{case_id}'.",
        )

    mask = case.structures[structure_name].mask
    voxel_vol = float(geom.spacing_mm[0] * geom.spacing_mm[1] * geom.spacing_mm[2])
    voxel_count = int(np.sum(mask))
    volume_mm3 = float(voxel_count * voxel_vol)

    if voxel_count == 0:
        return StructureStats(
            volume_mm3=0.0,
            voxel_count=0,
            bbox_ras_mm=BBoxRAS(min=[0.0, 0.0, 0.0], max=[0.0, 0.0, 0.0]),
            centroid_ras_mm=[0.0, 0.0, 0.0],
            n_components=0,
            component_volumes_mm3=[],
            coordinate_system="RAS",
        )

    indices = np.argwhere(mask)
    z_min, y_min, x_min = np.min(indices, axis=0)
    z_max, y_max, x_max = np.max(indices, axis=0)

    # 8 corners of bounding box in voxel space
    corners_voxel = [
        (z, y, x)
        for z in (int(z_min), int(z_max))
        for y in (int(y_min), int(y_max))
        for x in (int(x_min), int(x_max))
    ]
    corners_ras = np.array([voxel_to_ras_mm(geom, c) for c in corners_voxel], dtype=np.float64)
    bbox_min = [float(v) for v in np.min(corners_ras, axis=0)]
    bbox_max = [float(v) for v in np.max(corners_ras, axis=0)]

    # Centroid in continuous voxel coordinates -> RAS
    centroid_zyx = np.mean(indices, axis=0)
    centroid_ras = [float(v) for v in voxel_to_ras_mm(geom, centroid_zyx)]

    # Connected components (6-connectivity)
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    labeled, n_components = scipy.ndimage.label(mask, structure=struct)
    counts = np.bincount(labeled.flat)[1:]
    comp_vols = [float(c * voxel_vol) for c in counts]
    comp_vols.sort(reverse=True)
    largest_10 = comp_vols[:10]

    return StructureStats(
        volume_mm3=volume_mm3,
        voxel_count=voxel_count,
        bbox_ras_mm=BBoxRAS(min=bbox_min, max=bbox_max),
        centroid_ras_mm=centroid_ras,
        n_components=int(n_components),
        component_volumes_mm3=largest_10,
        coordinate_system="RAS",
    )


@app.get("/cases/{case_id}/structures/{structure_name}/mesh.obj")
def get_structure_mesh(
    case_id: str,
    structure_name: str,
    repo: RepoDep,
    max_triangles: int = Query(default=150000, le=500000, ge=1),
) -> Response:
    """Generate 3D surface mesh (OBJ format) in RAS coordinates with outward winding."""
    result = repo.get_case(case_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    case, geom = result
    if structure_name not in case.structures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Structure '{structure_name}' not found in case '{case_id}'.",
        )

    mask = case.structures[structure_name].mask
    if not np.any(mask):
        headers = {
            "X-Mesh-Step-Size": "1",
            "X-Mesh-Triangles": "0",
        }
        return Response(content="# Empty mesh\n", media_type="text/plain", headers=headers)

    # Marching cubes strictly on the structure's bounding box plus a 1-voxel margin
    indices = np.argwhere(mask)
    z_min, y_min, x_min = np.min(indices, axis=0)
    z_max, y_max, x_max = np.max(indices, axis=0)

    cropped = mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
    padded = np.pad(cropped, pad_width=1, mode="constant", constant_values=False)

    spacing_arr = np.array(geom.spacing_mm, dtype=np.float64)

    step_size = 1
    verts_padded = None
    faces = None
    min_dim = min(padded.shape)

    while True:
        try:
            verts_step, faces_step, _, _ = skimage.measure.marching_cubes(
                padded.astype(np.float32),
                level=0.5,
                spacing=geom.spacing_mm,
                step_size=step_size,
            )
            verts_padded = verts_step
            faces = faces_step
            if len(faces) <= max_triangles or step_size >= min_dim:
                break
        except Exception:
            if faces is not None:
                break
            verts_padded, faces, _, _ = skimage.measure.marching_cubes(
                padded.astype(np.float32),
                level=0.5,
                spacing=geom.spacing_mm,
                step_size=1,
            )
            step_size = 1
            break
        step_size += 1

    if verts_padded is None or faces is None:
        headers = {
            "X-Mesh-Step-Size": "1",
            "X-Mesh-Triangles": "0",
        }
        return Response(content="# Empty mesh\n", media_type="text/plain", headers=headers)

    # Map padded coordinates back to original volume continuous voxel indices (z, y, x)
    voxel_indices_zyx = (verts_padded / spacing_arr) - 1.0 + np.array(
        [z_min, y_min, x_min], dtype=np.float64
    )

    verts_ras = transform_mesh_vertices_to_ras(geom, voxel_indices_zyx)

    # Triangle winding preservation: check signed volume in RAS
    signed_vol = mesh_signed_volume_ras(verts_ras, faces)
    if signed_vol < 0.0:
        faces = faces[:, ::-1]
        signed_vol = -signed_vol

    lines: list[str] = [
        "# HDT Generated OBJ Mesh",
        f"# Structure: {structure_name}",
        f"# Signed Volume: {signed_vol:.3f} mm^3",
    ]
    for v in verts_ras:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")

    headers = {
        "X-Mesh-Step-Size": str(step_size),
        "X-Mesh-Triangles": str(len(faces)),
    }
    return Response(content="\n".join(lines) + "\n", media_type="text/plain", headers=headers)


@app.post("/cases/{case_id}/clearance", response_model=ClearanceResponse)
def compute_clearance(
    case_id: str,
    request: ClearanceRequest,
    repo: RepoDep,
) -> ClearanceResponse:
    """Compute face-to-face signed clearance and closest points between structures."""
    result = repo.get_case(case_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    case, geom = result

    # Pairwise clearance requires at least two structures in the case
    if len(case.structures) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Case '{case_id}' has {len(case.structures)} structure(s). "
                "Clearance analysis requires at least two segmented structures."
            ),
        )

    target_name = request.target_name or ""
    risk_name = request.risk_name or ""

    if target_name not in case.structures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target structure '{target_name}' not found in case '{case_id}'.",
        )
    if risk_name not in case.structures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Risk structure '{risk_name}' not found in case '{case_id}'.",
        )

    target_mask = case.structures[target_name].mask
    risk_mask = case.structures[risk_name].mask

    clearance = float(signed_clearance_mm(target_mask, risk_mask, geom.spacing_mm))
    closest = closest_points_mm(target_mask, risk_mask, geom.spacing_mm)
    stats = surface_distance_stats(target_mask, risk_mask, geom.spacing_mm)

    pt_a_ras = list(local_point_zyx_mm_to_ras_mm(closest["point_a_zyx_mm"], geom))
    pt_b_ras = list(local_point_zyx_mm_to_ras_mm(closest["point_b_zyx_mm"], geom))

    resolution = compute_resolution_block(
        geom.spacing_mm, closest["point_a_zyx_mm"], closest["point_b_zyx_mm"]
    )

    is_folder_repo = isinstance(repo, FolderRepository)

    closest_ras: dict[str, Any] = {
        "centre_distance_mm": float(closest["distance_mm"]),
        "point_target_ras_mm": pt_a_ras,
        "point_risk_ras_mm": pt_b_ras,
        "coordinate_system": "RAS",
    }
    surface_stats: dict[str, float] = {
        "centre_min_mm": float(stats["min"]),
        "mean": float(stats["mean"]),
        "hd95": float(stats["hd95"]),
        "max": float(stats["max"]),
    }

    if not is_folder_repo:
        # Preserve backwards compatibility for phantom tests
        closest_ras["distance_mm"] = float(closest["distance_mm"])
        closest_ras["point_a_ras_mm"] = pt_a_ras
        closest_ras["point_b_ras_mm"] = pt_b_ras
        surface_stats["min"] = float(stats["min"])

    resp_kwargs: dict[str, Any] = {
        "clearance_mm": clearance,
        "convention": "face_to_face_voxel_sets",
        "closest_points_mm": closest_ras,
        "surface_distance_stats": surface_stats,
        "resolution": ResolutionBlock(**resolution),
        "coordinate_system": "RAS",
    }
    if not is_folder_repo:
        resp_kwargs["signed_clearance_mm"] = clearance

    return ClearanceResponse(**resp_kwargs)


@app.post("/cases/{case_id}/uncertainty", response_model=UncertaintyResponse)
def compute_uncertainty(
    case_id: str,
    request: UncertaintyRequest,
    repo: RepoDep,
) -> UncertaintyResponse:
    """Run Monte Carlo clearance uncertainty analysis under smooth boundary perturbation."""
    result = repo.get_case(case_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    case, geom = result
    target_name = request.target_name or ""
    risk_name = request.risk_name or ""

    if target_name not in case.structures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target structure '{target_name}' not found in case '{case_id}'.",
        )
    if risk_name not in case.structures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Risk structure '{risk_name}' not found in case '{case_id}'.",
        )

    target_mask = case.structures[target_name].mask
    risk_mask = case.structures[risk_name].mask

    closest = closest_points_mm(target_mask, risk_mask, geom.spacing_mm)
    resolution = compute_resolution_block(
        geom.spacing_mm, closest["point_a_zyx_mm"], closest["point_b_zyx_mm"]
    )

    params: dict[str, Any] = {
        "sigma_mm": float(request.sigma_mm),
        "corr_length_mm": float(request.corr_length_mm),
        "bias_sigma_mm": float(request.bias_sigma_mm),
    }

    seed = 42 if request.seed is None else int(request.seed)

    res = analyse_clearance(
        target=target_mask,
        risk=risk_mask,
        spacing_mm=geom.spacing_mm,
        params=params,
        n_samples=request.n_samples,
        seed=seed,
        threshold_mm=float(request.threshold_mm),
    )

    data = res.to_dict()
    data["coordinate_system"] = "RAS"
    data["resolution"] = ResolutionBlock(**resolution)
    return UncertaintyResponse(**data)


# ---------------------------------------------------------------------------
# Anatomical Neural Atlas Endpoints (Body Explorer)
# ---------------------------------------------------------------------------


@app.get(
    "/atlas",
    response_model=Atlas,
    summary="Get anatomical neural atlas",
    description=(
        "Returns the full educational neural atlas including dermatomes, "
        "ascending paths, and conditions."
    ),
)
def get_atlas() -> Atlas:
    """Return the complete educational neural atlas."""
    return get_default_atlas()


@app.get(
    "/atlas/dermatomes/{dermatome_id}",
    response_model=Dermatome,
    summary="Get single dermatome by id",
    description=(
        "Returns metadata and ascending neural transmission path for a single dermatome."
    ),
)
def get_dermatome(dermatome_id: str) -> Dermatome:
    """Return a single dermatome by ID, or 404 if not found."""
    atlas = get_default_atlas()
    for d in atlas.dermatomes:
        if d.id.lower() == dermatome_id.lower():
            return d
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Dermatome '{dermatome_id}' not found in atlas.",
    )


@app.get(
    "/atlas/conditions",
    response_model=list[Condition],
    summary="Get list of clinical conditions",
    description=(
        "Returns common peripheral nerve and radicular clinical conditions "
        "mapped to affected dermatomes and roots."
    ),
)
def get_conditions() -> list[Condition]:
    """Return list of all clinical nerve conditions in the atlas."""
    return get_default_atlas().conditions



# ---------------------------------------------------------------------------
# Body Explorer Presets & Dermatome Picking Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/body/presets",
    response_model=list[BodyPresetSummary],
    summary="List available stature presets",
    description="Returns available anthropometric stature presets (short, average, tall).",
)
def list_body_presets() -> list[BodyPresetSummary]:
    """Return available stature presets."""
    return [
        BodyPresetSummary(name=name, height_mm=preset.height_mm)
        for name, preset in PRESETS.items()
    ]


@app.get(
    "/body/presets/{name}",
    response_model=BodyParams,
    summary="Get body preset by name",
    description="Returns a complete BodyParams model scaled to the preset height.",
)
def get_body_preset(name: str) -> BodyParams:
    """Return a single BodyParams preset, or 404 if not found."""
    preset = get_preset(name)
    if preset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{name}' not found. Available presets: {list(PRESETS.keys())}.",
        )
    return preset


@app.post(
    "/body/pick",
    response_model=PickResponse,
    summary="Pick nearest dermatome and compute neural pathway",
    description=(
        "Identifies nearest dermatome anchor to query point, builds ascending "
        "3D transmission polyline, and computes conduction latencies."
    ),
)
def pick_dermatome_pathway(request: PickRequest) -> PickResponse:
    """Process click-to-nerve query on body model."""
    errors = validate_body_params(request.body)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors},
        )

    atlas = get_default_atlas()
    dermatome_id, side, distance_mm = nearest_dermatome(request.body, request.point_ras)

    dermatome = next((d for d in atlas.dermatomes if d.id == dermatome_id), None)
    region_text = dermatome.region if dermatome else f"Dermatome {dermatome_id}"

    # Generate labelled 3D pathway
    path_labelled = full_pathway_polyline(request.body, dermatome_id, side=side)
    pathway_points = [
        PathwayPoint(label=label, point_ras_mm=pt) for label, pt in path_labelled
    ]

    # Compute conduction latency timeline
    timeline = signal_timeline(request.body, dermatome_id, side=side)

    return PickResponse(
        dermatome_id=dermatome_id,
        region=region_text,
        side=side,
        distance_to_anchor_mm=distance_mm,
        pathway=pathway_points,
        timeline=timeline,
        disclaimer=atlas.disclaimer,
    )


@app.post(
    "/body/shape",
    response_model=ShapeModelResponse,
    summary="Compute body segment dimensions and radii",
    description="Returns articulated segment radii and cross-sections scaled to stature.",
)
def compute_body_shape(request: BodyShapeRequest) -> ShapeModelResponse:
    """Compute articulated body segment dimensions and radii."""
    try:
        body = request.extract_body_params()
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(err),
        ) from err

    errors = validate_body_params(body)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors},
        )

    return ShapeModelResponse(
        height_mm=body.height_mm,
        segments=segments(body),
    )


@app.post(
    "/body/nervous",
    response_model=NerveNetworkResponse,
    summary="Generate full 3D nervous system geometry",
    description="Constructs 3D polylines for spinal cord, roots, plexuses, and brain landmarks.",
)
def generate_nervous_network(request: BodyNervousRequest) -> NerveNetworkResponse:
    """Generate renderable 3D nervous system geometry."""
    try:
        body = request.extract_body_params()
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(err),
        ) from err

    errors = validate_body_params(body)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors},
        )

    return build_network(body, detail=request.detail)
# ---------------------------------------------------------------------------
# Real-Time Camera Body Scanner Endpoints
# ---------------------------------------------------------------------------


@app.get("/body-scan/reference", tags=["Body Scanner"])
def get_body_scan_reference() -> dict[str, Any]:
    """Retrieve anthropometric reference measurements and camera defaults."""
    return {
        "reference_lengths_meters": DEFAULT_REFERENCE_LENGTHS,
        "websocket_endpoint": "/ws/body-scan",
        "default_camera_index": 0,
        "model": "MediaPipe Pose (model_complexity=2)",
    }


@app.websocket("/ws/body-scan")
async def websocket_body_scan(
    websocket: WebSocket,
    camera_index: int = Query(default=0, ge=0),
    smoothing: float = Query(default=0.35, ge=0.0, le=1.0),
) -> None:
    """Continuously stream real-time 3D body landmarks, limb lengths, and scale factors.

    Captures frames from the webcam via OpenCV and computes 3D Euclidean limb lengths
    (upper arms, forearms, shoulder width) using MediaPipe Pose (complexity 2).
    Streams calculated limb scale factors as a JSON payload to connected clients.
    Releases the OpenCV video capture resource cleanly when the client disconnects.
    """
    await websocket.accept()
    logger.info("Accepted WebSocket connection on /ws/body-scan (camera_index=%d)", camera_index)

    scanner = BodyScanner(
        camera_index=camera_index,
        smoothing_factor=smoothing,
        auto_start_camera=True,
    )

    try:
        while True:
            # Check for non-blocking client messages (e.g. calibration command or parameter updates)
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if isinstance(msg, dict):
                    action = msg.get("action") or msg.get("command")
                    if action == "calibrate":
                        calibrated = scanner.calibrate_baseline()
                        await websocket.send_json({
                            "status": "calibrated",
                            "message": "Baseline reference lengths updated to current measurements",
                            "reference_lengths": calibrated,
                        })
                    elif action == "set_smoothing":
                        factor = float(msg.get("smoothing", 0.35))
                        scanner.smoothing_factor = max(0.01, min(1.0, factor))
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                raise
            except Exception:
                # Malformed incoming json or non-json message, ignore and continue streaming
                pass

            # Execute computer vision frame processing in worker thread so event loop
            # remains responsive
            payload = await asyncio.to_thread(scanner.get_frame_measurements)
            await websocket.send_json(payload)

            # Cap stream frequency to ~30 FPS
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/body-scan WebSocket")
    except Exception as exc:
        logger.warning("WebSocket stream ended on /ws/body-scan: %s", exc)
    finally:
        # Crucial: Ensure OpenCV video capture and MediaPipe resources are cleanly released
        scanner.release()
        logger.info("OpenCV camera capture released cleanly for /ws/body-scan")
