"""FastAPI router for the /body/systems endpoints.

Provides system registry listing and per-system item generation for the
Body Explorer educational 3-D body model.  NOT a clinical tool.

Endpoints
---------
GET  /body/systems               → list[SystemRegistryEntry]
POST /body/systems/{system_id}   → SystemItemsResponse
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from hdt.body.models import BodyParams, validate_body_params
from hdt.systems.registry import SYSTEMS, build_system

router = APIRouter(prefix="/body/systems", tags=["Body Systems"])


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class SystemRegistryEntry(BaseModel):
    """Metadata for a single body system."""

    id: str
    name: str
    description: str
    default_visible: bool
    default_opacity: float
    color_hint: str
    item_count_note: str
    available: bool


class SystemItemsResponse(BaseModel):
    """Response payload for /body/systems/{system_id}."""

    system_id: str
    item_count: int
    items: list[dict]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("", response_model=list[SystemRegistryEntry])
def list_systems() -> list[SystemRegistryEntry]:
    """Return the registry of all body systems (available and planned)."""
    return [SystemRegistryEntry(**entry) for entry in SYSTEMS.values()]


@router.post("/{system_id}", response_model=SystemItemsResponse)
def get_system_items(system_id: str, body: BodyParams) -> SystemItemsResponse:
    """Build and return the 3-D primitive items for *system_id*.

    Parameters
    ----------
    system_id : str
        One of the available system IDs (skeleton, vascular, organs).
    body : BodyParams
        Anthropometric body model supplied in the request body.

    Returns
    -------
    SystemItemsResponse
        JSON-serialisable list of BodyItem dicts.

    Raises
    ------
    404 : system_id not found or not available
    422 : body params fail anatomical validation
    """
    # 1. Anatomical validation
    errors = validate_body_params(body)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors},
        )

    # 2. System lookup
    if system_id not in SYSTEMS or not SYSTEMS[system_id]["available"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System '{system_id}' not found or not yet available.",
        )

    # 3. Build items
    try:
        items = build_system(body, system_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return SystemItemsResponse(
        system_id=system_id,
        item_count=len(items),
        items=items,
    )
