"""Case repository protocol and in-memory phantom repository implementation.

Provides domain-agnostic access to imaging cases and geometry for API endpoints.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from hdt.core.case import Case
from hdt.core.structure import Structure
from hdt.imaging.load_scan import (
    ImageGeometry,
    assert_same_geometry,
    load_nifti_image,
    load_nifti_mask_array,
)
from hdt.imaging.phantoms import make_phantom_case


@runtime_checkable
class CaseRepository(Protocol):
    """Protocol defining case retrieval and discovery."""

    def list_cases(self) -> list[Case]:
        """Return all available cases."""
        ...

    def get_case(self, case_id: str) -> tuple[Case, ImageGeometry] | None:
        """Retrieve a case and its ImageGeometry by ID, or None if not found."""
        ...


class FolderRepository:
    """Repository serving real prepared cases from disk with an LRU in-memory cache.

    Security & Safety:
    - Case IDs are strictly validated against scanned directory names.
    - User input is NEVER joined into a filesystem path.
    - Keeps at most 2 cases in memory via an LRU cache.
    """

    def __init__(self, root: str | Path) -> None:
        p = Path(root).resolve()
        if (p / "cases").is_dir():
            self._cases_dir = p / "cases"
        elif (p / "data" / "processed" / "cases").is_dir():
            self._cases_dir = p / "data" / "processed" / "cases"
        elif p.name == "cases":
            self._cases_dir = p
        else:
            self._cases_dir = p

        self._cache: OrderedDict[str, tuple[Case, ImageGeometry]] = OrderedDict()

    def _scan_case_dirs(self) -> dict[str, Path]:
        """Scan sub-folder names under cases directory that contain case.json."""
        if not self._cases_dir.is_dir():
            return {}
        cases: dict[str, Path] = {}
        for entry in self._cases_dir.iterdir():
            if entry.is_dir() and (entry / "case.json").is_file():
                cases[entry.name] = entry
        return cases

    def list_cases(self) -> list[Case]:
        """List all available cases without holding multiple large volumes in memory."""
        valid_cases = self._scan_case_dirs()
        summaries: list[Case] = []

        for cid, folder in valid_cases.items():
            # If already cached in LRU, use cached Case
            if cid in self._cache:
                summaries.append(self._cache[cid][0])
                continue

            # Otherwise construct lightweight Case metadata from case.json
            case_json = folder / "case.json"
            try:
                with case_json.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                roles = meta.get("roles", {})
                sp = tuple(float(s) for s in meta.get("spacing_mm_zyx", (1.0, 1.0, 1.0)))
                # Lightweight structures representing metadata without voxel memory
                structures = {
                    name: Structure(
                        name=name,
                        mask=np.zeros((1, 1, 1), dtype=bool),
                        role=role if role in ("target", "risk", "context") else "context",
                    )
                    for name, role in roles.items()
                }
                c = Case(
                    case_id=cid,
                    images={"t1c": np.zeros((1, 1, 1), dtype=np.float32)},
                    spacing_mm=sp,  # type: ignore[arg-type]
                    origin_mm=(0.0, 0.0, 0.0),
                    direction=np.eye(3, dtype=np.float64),
                    structures=structures,
                    primary="t1c",
                )
                summaries.append(c)
            except Exception:
                continue

        return summaries

    def get_case(self, case_id: str) -> tuple[Case, ImageGeometry] | None:
        """Retrieve a case and its ImageGeometry by ID with LRU caching (max 2 cases).

        Never joins user input into a filesystem path.
        """
        # Check LRU cache first
        if case_id in self._cache:
            self._cache.move_to_end(case_id)
            return self._cache[case_id]

        valid_cases = self._scan_case_dirs()
        if case_id not in valid_cases:
            return None

        # Safe directory reference from directory listing (never string-joined)
        folder = valid_cases[case_id]

        case_json_path = folder / "case.json"
        roles: dict[str, str] = {}
        if case_json_path.is_file():
            with case_json_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                roles = meta.get("roles", {})

        # Load primary modality t1c.nii.gz
        t1c_path = folder / "t1c.nii.gz"
        if not t1c_path.is_file():
            return None

        img_arr, sp, orig, dir_m = load_nifti_image(t1c_path)
        img_geom = ImageGeometry(
            shape=img_arr.shape,
            spacing_mm=sp,
            origin_mm=orig,
            direction=dir_m,
        )

        # Load all structures in structures/
        structs_dir = folder / "structures"
        structures: dict[str, Structure] = {}
        if structs_dir.is_dir():
            for s_file in sorted(structs_dir.iterdir()):
                if s_file.name.endswith((".nii.gz", ".nii")):
                    s_name = s_file.name.replace(".nii.gz", "").replace(".nii", "")
                    mask_arr, mask_geom, _ = load_nifti_mask_array(s_file, image_meta=img_geom)
                    assert_same_geometry(img_geom, mask_geom)
                    assigned_role = roles.get(s_name, "context")
                    role_val = (
                        assigned_role
                        if assigned_role in ("target", "risk", "context")
                        else "context"
                    )
                    structures[s_name] = Structure(
                        name=s_name,
                        mask=np.array(mask_arr > 0, dtype=bool),
                        role=role_val,  # type: ignore[arg-type]
                    )

        case = Case(
            case_id=case_id,
            images={"t1c": img_arr},
            spacing_mm=sp,
            origin_mm=orig,
            direction=dir_m,
            structures=structures,
            primary="t1c",
        )

        # LRU cache update: keep at most 2 cases
        result = (case, img_geom)
        self._cache[case_id] = result
        if len(self._cache) > 2:
            self._cache.popitem(last=False)

        return result


class PhantomRepository:
    """In-memory repository providing synthetic phantom cases with anisotropic oblique geometry."""

    def __init__(self) -> None:
        self._cases: dict[str, tuple[Case, ImageGeometry]] = {}

        # 15 degree rotation around Y axis in ITK (x, y, z) physical space
        theta = float(np.deg2rad(15.0))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        direction = np.array(
            [
                [cos_t, 0.0, sin_t],
                [0.0, 1.0, 0.0],
                [-sin_t, 0.0, cos_t],
            ],
            dtype=np.float64,
        )

        spacing_mm = (2.0, 1.0, 1.0)
        origin_mm = (10.0, -20.0, 5.0)
        shape_zyx = (64, 64, 64)

        phantom_case, _ = make_phantom_case(
            case_id="phantom_001",
            shape_zyx=shape_zyx,
            spacing_mm=spacing_mm,
            origin_mm=origin_mm,
            direction=direction,
            tumor_center_mm=(64.0, 32.0, 24.0),
            tumor_radius_mm=6.0,
            vessel_center_mm=(64.0, 32.0, 40.0),
            vessel_radius_mm=4.0,
        )

        geom = ImageGeometry(
            shape=shape_zyx,
            spacing_mm=spacing_mm,
            origin_mm=origin_mm,
            direction=direction,
        )

        self._cases["phantom_001"] = (phantom_case, geom)

    def list_cases(self) -> list[Case]:
        """List all available cases."""
        return [case for case, _ in self._cases.values()]

    def get_case(self, case_id: str) -> tuple[Case, ImageGeometry] | None:
        """Get a case and its geometry by ID, or None if not found."""
        return self._cases.get(case_id)
