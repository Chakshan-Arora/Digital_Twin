"""Contour parsing and validation for TCIA Vestibular-Schwannoma-SEG.

Enforces dataset rules:
- Tumour is at index 0 of the native folder; warns if name not in {TV, tv, AN, tumour}.
- Cochlea is the item whose lower-cased name is exactly "cochlea"; raises if none or several.
- Native folder rule: folder where original_contour_created_on_image_name equals
  reference_geometry_image_name for that item. Exactly one folder must match.
- Plane normal and in-plane axes come from native series ImageGeometry.
  Every ring's points must share one plane coordinate along the normal within 0.01 mm.
- Rings are grouped by plane (tolerance 0.01 mm) without requiring sorted input.
  Planes must be equally spaced within 0.05 mm of the series slice spacing;
  missing interior planes are reported rather than filled.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hdt.imaging.load_scan import ImageGeometry

STANDARD_TUMOUR_NAMES = {"TV", "tv", "AN", "tumour"}


@dataclass(frozen=True)
class ContourPlane:
    """A collection of closed polygon rings lying on a single physical slice plane.

    Attributes
    ----------
    coordinate : float
        Scalar position along the slice normal (mm in LPS frame: p . n).
    rings_lps : list[np.ndarray]
        List of ring polygons, each an (N, 3) float64 array of 3D LPS vertices.
    rings_2d : list[np.ndarray]
        List of ring polygons, each an (N, 2) float64 array of in-plane (u, v) coordinates.
    """

    coordinate: float
    rings_lps: list[np.ndarray]
    rings_2d: list[np.ndarray]


@dataclass
class StructureContours:
    """Validated contour representation for a single anatomical structure.

    Attributes
    ----------
    structure_name : str
        Name of the structure as recorded in the contour file.
    is_standard_tumour : bool | None
        Whether tumour name is in {"TV", "tv", "AN", "tumour"} (None for non-tumour).
    native_image_name : str
        Name of the native image on which the contour was drawn.
    native_folder : Path | None
        Directory path of the native contour folder.
    slice_spacing_mm : float
        Nominal slice spacing of the native series (mm).
    planes : list[ContourPlane]
        List of slice planes, strictly ordered by normal coordinate.
    missing_interior_planes : list[float]
        Coordinates of any detected missing interior planes along the normal.
    """

    structure_name: str
    is_standard_tumour: bool | None = None
    native_image_name: str = ""
    native_folder: Path | None = None
    slice_spacing_mm: float = 1.5
    planes: list[ContourPlane] = field(default_factory=list)
    missing_interior_planes: list[float] = field(default_factory=list)


def group_rings_by_plane(
    rings: list[list[list[float]] | np.ndarray],
    normal: np.ndarray,
    in_plane_u: np.ndarray,
    in_plane_v: np.ndarray,
    slice_spacing_mm: float,
    plane_tol_mm: float = 0.01,
    spacing_tol_mm: float = 0.05,
) -> tuple[list[ContourPlane], list[float]]:
    """Validate ring planarity and group rings by slice plane.

    Parameters
    ----------
    rings : list[list[list[float]] | np.ndarray]
        List of closed rings, each containing Nx3 vertex coordinates in LPS mm.
    normal : np.ndarray
        Unit normal vector of the native slice planes (3,).
    in_plane_u : np.ndarray
        First in-plane unit basis vector (3,).
    in_plane_v : np.ndarray
        Second in-plane unit basis vector (3,).
    slice_spacing_mm : float
        Expected slice spacing along the normal (mm).
    plane_tol_mm : float
        Tolerance for checking if points within a ring lie on the same plane (mm).
    spacing_tol_mm : float
        Tolerance for verifying constant slice spacing between planes (mm).

    Returns
    -------
    tuple[list[ContourPlane], list[float]]
        - Ordered list of ContourPlane objects sorted by normal coordinate.
        - List of missing interior plane coordinates along the normal.

    Raises
    ------
    ValueError
        If any ring's points have normal coordinate spread exceeding plane_tol_mm,
        or if plane spacing deviates from expected slice spacing without being
        a valid integer missing-step multiple.
    """
    if not rings:
        return [], []

    # 1. Validate each ring and project coordinates
    ring_entries: list[tuple[float, np.ndarray, np.ndarray]] = []
    norm = np.asarray(normal, dtype=np.float64)
    u_axis = np.asarray(in_plane_u, dtype=np.float64)
    v_axis = np.asarray(in_plane_v, dtype=np.float64)

    for i, ring in enumerate(rings):
        r_arr = np.asarray(ring, dtype=np.float64)
        if r_arr.ndim != 2 or r_arr.shape[1] != 3:
            raise ValueError(f"Ring {i} must have shape (N, 3), got {r_arr.shape}")
        if len(r_arr) < 3:
            raise ValueError(f"Ring {i} has fewer than 3 vertices ({len(r_arr)})")

        s_coords = r_arr @ norm
        spread = float(np.ptp(s_coords))
        if spread > plane_tol_mm:
            raise ValueError(
                f"Ring {i} points do not share a single plane coordinate along normal: "
                f"spread {spread:.5f} mm exceeds tolerance {plane_tol_mm:.5f} mm."
            )

        mean_s = float(np.mean(s_coords))
        u_coords = r_arr @ u_axis
        v_coords = r_arr @ v_axis
        ring_2d = np.column_stack([u_coords, v_coords])
        ring_entries.append((mean_s, r_arr, ring_2d))

    # 2. Group rings by plane (tolerance plane_tol_mm)
    # Sort entries by plane coordinate first
    ring_entries.sort(key=lambda item: item[0])

    plane_groups: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
    for entry in ring_entries:
        if not plane_groups:
            plane_groups.append([entry])
        else:
            group_mean = float(np.mean([e[0] for e in plane_groups[-1]]))
            if abs(entry[0] - group_mean) <= plane_tol_mm:
                plane_groups[-1].append(entry)
            else:
                plane_groups.append([entry])

    planes: list[ContourPlane] = []
    for grp in plane_groups:
        coord = float(np.mean([e[0] for e in grp]))
        lps_rings = [e[1] for e in grp]
        rings_2d = [e[2] for e in grp]
        planes.append(
            ContourPlane(
                coordinate=coord,
                rings_lps=lps_rings,
                rings_2d=rings_2d,
            )
        )

    # 3. Check plane spacing and detect missing interior planes
    missing_planes: list[float] = []
    if len(planes) > 1:
        coords = [p.coordinate for p in planes]
        for k in range(len(coords) - 1):
            diff = coords[k + 1] - coords[k]
            if diff <= 0:
                raise ValueError(
                    f"Plane coordinates must be strictly increasing: {coords[k]} >= {coords[k + 1]}"
                )

            spacing_error = abs(diff - slice_spacing_mm)
            if spacing_error > spacing_tol_mm:
                # Check if it represents missing interior planes
                step_ratio = diff / slice_spacing_mm
                m = int(round(step_ratio))
                if m > 1 and abs(diff - m * slice_spacing_mm) <= m * spacing_tol_mm:
                    for step in range(1, m):
                        missing_coord = coords[k] + step * slice_spacing_mm
                        missing_planes.append(float(missing_coord))
                else:
                    raise ValueError(
                        f"Non-constant plane spacing between planes {k} and {k + 1}: "
                        f"diff={diff:.4f} mm, expected {slice_spacing_mm:.4f} mm "
                        f"(deviation {spacing_error:.4f} mm > {spacing_tol_mm:.4f} mm)."
                    )

    return planes, missing_planes


def find_native_folder(
    candidate_folders: list[Path | str],
    structure_selector: Any,
) -> tuple[Path, dict[str, Any]]:
    """Apply native folder rule.

    Checks: original_contour_created_on_image_name == reference_geometry_image_name.

    Parameters
    ----------
    candidate_folders : list[Path | str]
        List of folders to search (e.g. [vs_gk_N_t1, vs_gk_N_t2]).
    structure_selector : callable
        Function taking parsed JSON items list and returning the structure item dict, or None.

    Returns
    -------
    tuple[Path, dict[str, Any]]
        The matching native folder path and the matched ROI dictionary item.

    Raises
    ------
    ValueError
        If zero or more than one native folder matches the rule.
    """
    matches: list[tuple[Path, dict[str, Any]]] = []

    for folder in candidate_folders:
        f_path = Path(folder)
        json_file = f_path / "contours.json"
        if not json_file.is_file():
            continue

        with open(json_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        item = structure_selector(data)
        if item is None:
            continue

        orig_name = item.get("original_contour_created_on_image_name")
        ref_name = item.get("reference_geometry_image_name")
        if orig_name and ref_name and orig_name == ref_name:
            matches.append((f_path, item))

    if len(matches) == 0:
        raise ValueError(
            f"Native folder rule failed: no folder where original == reference found among "
            f"{[str(p) for p in candidate_folders]}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Native folder rule failed: multiple folders matched original == reference: "
            f"{[str(m[0]) for m in matches]}."
        )

    return matches[0]


def extract_tumour_contour(
    candidate_folders: list[Path | str],
    native_geometry: ImageGeometry,
) -> StructureContours:
    """Extract and validate tumour contours from candidate folders.

    Tumour is the item at index 0 of the native folder.
    Warns if structure name is not in {"TV", "tv", "AN", "tumour"}.
    """

    def tumour_selector(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not items:
            return None
        return items[0]

    native_folder, item = find_native_folder(candidate_folders, tumour_selector)
    sname = str(item.get("structure_name", ""))
    is_standard = sname in STANDARD_TUMOUR_NAMES
    if not is_standard:
        warnings.warn(
            f"Tumour structure name '{sname}' is not in standard set {STANDARD_TUMOUR_NAMES}",
            UserWarning,
            stacklevel=2,
        )

    rings = item.get("LPS_contour_points", [])
    normal = native_geometry.direction[:, 2]
    u_axis = native_geometry.direction[:, 0]
    v_axis = native_geometry.direction[:, 1]
    slice_sp = float(native_geometry.spacing_mm[0])

    planes, missing = group_rings_by_plane(
        rings=rings,
        normal=normal,
        in_plane_u=u_axis,
        in_plane_v=v_axis,
        slice_spacing_mm=slice_sp,
    )

    return StructureContours(
        structure_name=sname,
        is_standard_tumour=is_standard,
        native_image_name=str(item.get("original_contour_created_on_image_name", "")),
        native_folder=native_folder,
        slice_spacing_mm=slice_sp,
        planes=planes,
        missing_interior_planes=missing,
    )


def extract_cochlea_contour(
    candidate_folders: list[Path | str],
    native_geometry: ImageGeometry,
) -> StructureContours:
    """Extract and validate cochlea contours from candidate folders.

    Cochlea is the item whose lower-cased name is exactly "cochlea".
    Raises ValueError if none or several matching items exist.
    """

    def cochlea_selector(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        exact_matches = [
            it for it in items if str(it.get("structure_name", "")).strip().lower() == "cochlea"
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            names = [it.get("structure_name") for it in exact_matches]
            raise ValueError(
                f"Multiple items with exact name 'cochlea' (case-insensitive) found: {names}"
            )
        return None

    # Check across folders for any ambiguity
    all_cochlea_names: set[str] = set()
    for f in candidate_folders:
        json_file = Path(f) / "contours.json"
        if json_file.is_file():
            with open(json_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            for it in data:
                n = str(it.get("structure_name", ""))
                if "cochlea" in n.lower():
                    all_cochlea_names.add(n)

    exact_cochleas = [n for n in all_cochlea_names if n.lower() == "cochlea"]
    if len(exact_cochleas) == 0:
        raise ValueError(
            f"No cochlea structure found with exact name 'cochlea' (case-insensitive). "
            f"Found cochlea-like names: {sorted(all_cochlea_names)}"
        )

    native_folder, item = find_native_folder(candidate_folders, cochlea_selector)
    sname = str(item.get("structure_name", ""))

    rings = item.get("LPS_contour_points", [])
    normal = native_geometry.direction[:, 2]
    u_axis = native_geometry.direction[:, 0]
    v_axis = native_geometry.direction[:, 1]
    slice_sp = float(native_geometry.spacing_mm[0])

    planes, missing = group_rings_by_plane(
        rings=rings,
        normal=normal,
        in_plane_u=u_axis,
        in_plane_v=v_axis,
        slice_spacing_mm=slice_sp,
    )

    return StructureContours(
        structure_name=sname,
        is_standard_tumour=None,
        native_image_name=str(item.get("original_contour_created_on_image_name", "")),
        native_folder=native_folder,
        slice_spacing_mm=slice_sp,
        planes=planes,
        missing_interior_planes=missing,
    )


def load_patient_contours(
    patient_gk_id: str,
    contours_dir: Path | str,
    t1_geom: ImageGeometry,
    t2_geom: ImageGeometry,
) -> tuple[StructureContours, StructureContours]:
    """Load both tumour and cochlea contours for a patient from the contours archive.

    Parameters
    ----------
    patient_gk_id : str
        e.g. 'vs_gk_1'
    contours_dir : Path | str
        Path to directory containing vs_gk_* folders.
    t1_geom : ImageGeometry
        Geometry of the patient's native T1 series.
    t2_geom : ImageGeometry
        Geometry of the patient's native T2 series.

    Returns
    -------
    tuple[StructureContours, StructureContours]
        (tumour_contours, cochlea_contours)
    """
    c_base = Path(contours_dir)
    candidate_folders = [
        c_base / f"{patient_gk_id}_t1",
        c_base / f"{patient_gk_id}_t2",
    ]
    tumour = extract_tumour_contour(candidate_folders, t1_geom)
    cochlea = extract_cochlea_contour(candidate_folders, t2_geom)
    return tumour, cochlea
