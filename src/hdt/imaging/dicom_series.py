"""DICOM series loading and registration transform utilities using SimpleITK.

This module is the ONLY new place SimpleITK reads DICOM series in the repository.
It ensures that DICOM slice files are strictly ordered by physical position along
the slice normal, validates constant slice spacing (within 1e-3 mm), rejects duplicate
slice positions, and returns the volume as a float32 NumPy array in (z, y, x) order
with associated ImageGeometry in standard HDT conventions (LPS physical space).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk

from hdt.imaging.dicom_inspect import compute_slice_normal
from hdt.imaging.load_scan import ImageGeometry


def load_series(folder: Path | str) -> tuple[np.ndarray, ImageGeometry]:
    """Load a 3D DICOM series from a folder using SimpleITK.ImageSeriesReader.

    Files are ordered strictly by physical position along the slice normal vector.
    Raises ValueError if:
    - Slice spacing from positions is not constant within 1e-3 mm.
    - Any slice position along the normal is duplicated (within 1e-4 mm).
    - No DICOM files are found in the directory.

    Parameters
    ----------
    folder : Path | str
        Path to directory containing DICOM series slices.

    Returns
    -------
    tuple[np.ndarray, ImageGeometry]
        - float32 NumPy array in internal (z, y, x) order.
        - ImageGeometry with shape (z, y, x), spacing (z, y, x) in mm,
          origin (z, y, x) in mm, and 3x3 direction matrix in standard ITK LPS space.
    """
    f_path = Path(folder)
    if not f_path.is_dir():
        raise ValueError(f"DICOM series folder does not exist or is not a directory: {folder}")

    # Gather DICOM files
    all_files = [f for f in f_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    dcm_slices: list[tuple[float, Path, str]] = []

    normal: np.ndarray | None = None
    target_series_uid: str | None = None

    for f in all_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue

        if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
            continue

        suid = str(getattr(ds, "SeriesInstanceUID", ""))
        if target_series_uid is None:
            target_series_uid = suid
        elif suid != target_series_uid:
            # Different series in same folder, skip non-primary series
            continue

        if normal is None:
            normal = compute_slice_normal(ds.ImageOrientationPatient)

        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        proj = float(np.dot(ipp, normal))
        sop_uid = str(getattr(ds, "SOPInstanceUID", f.name))
        dcm_slices.append((proj, f, sop_uid))

    if not dcm_slices:
        raise ValueError(f"No valid DICOM image slices with geometry found in folder: {folder}")

    # Sort files strictly by physical position along slice normal
    dcm_slices.sort(key=lambda item: item[0])
    projections = [item[0] for item in dcm_slices]
    ordered_files = [str(item[1]) for item in dcm_slices]

    if len(projections) < 2:
        raise ValueError(f"DICOM series must have at least 2 slices, found {len(projections)}")

    diffs = np.diff(projections)

    # Check for duplicate positions
    for d in diffs:
        if abs(d) < 1e-4:
            raise ValueError(
                f"Duplicate slice position detected in DICOM series '{folder}' (spacing < 1e-4 mm)."
            )

    # Check for constant slice spacing within 1e-3 mm
    nominal_spacing = float(np.median(diffs))
    max_spacing_dev = float(np.max(np.abs(diffs - nominal_spacing)))
    if max_spacing_dev > 1e-3:
        raise ValueError(
            f"Non-constant slice spacing detected in DICOM series '{folder}': "
            f"nominal {nominal_spacing:.6f} mm, max deviation {max_spacing_dev:.6f} mm > 1e-3 mm."
        )

    # Read using SimpleITK ImageSeriesReader with ordered file paths
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ordered_files)
    sitk_img = reader.Execute()

    # Convert to float32 NumPy array in (z, y, x) order
    arr = np.array(sitk.GetArrayFromImage(sitk_img), dtype=np.float32, copy=True)

    # SimpleITK provides (x, y, z) metadata -> convert to HDT (z, y, x)
    sitk_sp = sitk_img.GetSpacing()
    spacing_zyx = (float(sitk_sp[2]), float(sitk_sp[1]), float(sitk_sp[0]))

    sitk_orig = sitk_img.GetOrigin()
    origin_zyx = (float(sitk_orig[2]), float(sitk_orig[1]), float(sitk_orig[0]))

    direction_3x3 = np.array(sitk_img.GetDirection(), dtype=np.float64).reshape((3, 3))

    geom = ImageGeometry(
        shape=arr.shape,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction_3x3,
    )

    return arr, geom


def read_lps_transform(tfm_path: Path | str) -> sitk.Transform:
    """Read a SimpleITK registration transform (.tfm) in LPS space.

    Parameters
    ----------
    tfm_path : Path | str
        Path to .tfm file.

    Returns
    -------
    sitk.Transform
        Loaded transform.
    """
    path = Path(tfm_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transform file not found: {tfm_path}")
    return sitk.ReadTransform(str(path))


@dataclass(frozen=True)
class AffineTransformLPS:
    """Vectorised 3D affine transformation in LPS physical coordinates.

    Applies the transformation p = M * (q - C) + T + C for points in LPS space.
    """

    matrix: np.ndarray  # 3x3 float64
    offset: np.ndarray  # (3,) float64
    center: np.ndarray  # (3,) float64

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3):
            raise ValueError(f"matrix must have shape (3, 3), got {self.matrix.shape}")
        if self.offset.shape != (3,):
            raise ValueError(f"offset must have shape (3,), got {self.offset.shape}")
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform an (N, 3) or (3,) array of points from input space to output space.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 3) or (3,) containing (x, y, z) coordinates in mm.

        Returns
        -------
        np.ndarray
            Transformed coordinates with the same leading dimension(s).
        """
        pts = np.asarray(points, dtype=np.float64)
        is_single = pts.ndim == 1
        if is_single:
            pts = pts.reshape(1, 3)

        transformed = (pts - self.center) @ self.matrix.T + self.offset + self.center
        if is_single:
            return transformed[0]
        return transformed

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.transform_points(points)

    def get_inverse(self) -> AffineTransformLPS:
        """Compute analytical inverse transform."""
        inv_m = np.linalg.inv(self.matrix)
        inv_offset = -inv_m @ self.offset
        return AffineTransformLPS(
            matrix=inv_m,
            offset=inv_offset,
            center=np.array(self.center, copy=True),
        )


def load_affine_transform(tfm_path: Path | str) -> AffineTransformLPS:
    """Load a SimpleITK affine transform (.tfm) and return an AffineTransformLPS.

    Converts the internal SimpleITK parameters into a vectorised NumPy structure
    so that downstream geometry / sampling code does not need to import SimpleITK.

    Parameters
    ----------
    tfm_path : Path | str
        Path to .tfm file.

    Returns
    -------
    AffineTransformLPS
        Vectorised transform operating on NumPy arrays.
    """
    sitk_tfm = read_lps_transform(tfm_path)
    params = np.array(sitk_tfm.GetParameters(), dtype=np.float64)
    fixed = np.array(sitk_tfm.GetFixedParameters(), dtype=np.float64)

    matrix = params[:9].reshape((3, 3))
    offset = params[9:12]
    center = fixed if len(fixed) == 3 else np.zeros(3, dtype=np.float64)

    return AffineTransformLPS(matrix=matrix, offset=offset, center=center)


def evaluate_transform_hypotheses(
    tfm_path: Path | str,
    t1_points: list[list[float]] | np.ndarray,
    t2_points: list[list[float]] | np.ndarray,
) -> dict[str, Any]:
    """Test both hypotheses on point transformation between T1 and T2 spaces.

    Hypothesis A: t2_points = tfm(t1_points)
    Hypothesis B: t2_points = inverse(tfm)(t1_points)

    Parameters
    ----------
    tfm_path : Path | str
        Path to .tfm registration matrix file.
    t1_points : list[list[float]] | np.ndarray
        Nx3 points in T1 LPS space.
    t2_points : list[list[float]] | np.ndarray
        Nx3 points in T2 LPS space.

    Returns
    -------
    dict[str, Any]
        Max point errors (mm), mean point errors (mm), and supported hypothesis.
    """
    tfm = read_lps_transform(tfm_path)
    tfm_inv = tfm.GetInverse()

    p1_arr = np.asarray(t1_points, dtype=np.float64)
    p2_arr = np.asarray(t2_points, dtype=np.float64)

    if p1_arr.shape != p2_arr.shape or p1_arr.shape[1] != 3:
        raise ValueError(f"Point shape mismatch: t1 has {p1_arr.shape}, t2 has {p2_arr.shape}.")

    # Hypothesis A: t2 = tfm(t1)
    p1_tfm_A = np.array([tfm.TransformPoint(tuple(p)) for p in p1_arr], dtype=np.float64)
    err_A = np.linalg.norm(p2_arr - p1_tfm_A, axis=1)
    max_err_A = float(np.max(err_A))
    mean_err_A = float(np.mean(err_A))

    # Hypothesis B: t2 = inv(tfm)(t1)
    p1_tfm_B = np.array([tfm_inv.TransformPoint(tuple(p)) for p in p1_arr], dtype=np.float64)
    err_B = np.linalg.norm(p2_arr - p1_tfm_B, axis=1)
    max_err_B = float(np.max(err_B))
    mean_err_B = float(np.mean(err_B))

    supported: str | None = None
    if max_err_B <= 0.01 and max_err_B < max_err_A:
        supported = "B"
    elif max_err_A <= 0.01 and max_err_A < max_err_B:
        supported = "A"

    return {
        "hypothesis_A_max_error_mm": max_err_A,
        "hypothesis_A_mean_error_mm": mean_err_A,
        "hypothesis_B_max_error_mm": max_err_B,
        "hypothesis_B_mean_error_mm": mean_err_B,
        "supported_hypothesis": supported,
    }
