"""Geometry-safe NIfTI image and mask loading and saving using SimpleITK.

AXIS AND COORDINATE CONVENTION NOTE:
This file is the SINGLE AND ONLY PLACE in the entire repository where axis order
conversions between SimpleITK (x, y, z) and HDT's internal NumPy order (z, y, x)
are performed. SimpleITK reports spacing, origin, and coordinates as (x, y, z),
whereas internal HDT arrays and metadata represent axes in (z, y, x) order.

LPS vs. RAS:
- LPS (Left, Posterior, Superior) is the standard coordinate space of ITK and SimpleITK.
- RAS (Right, Anterior, Superior) is the standard space used by NIfTI and 3D Slicer.
- Conversion between LPS and RAS is performed by negating the x and y axes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


class GeometryMismatchError(ValueError):
    """Raised when spatial geometries (shape, spacing, origin, or direction) mismatch."""

    pass


@dataclass(frozen=True)
class ImageGeometry:
    """Standardized spatial geometry representation in HDT (z, y, x) convention.

    Attributes:
        shape: Volume dimensions (z, y, x) voxels.
        spacing_mm: Physical voxel spacing (spacing_z, spacing_y, spacing_x) in mm.
        origin_mm: Physical origin coordinate (origin_z, origin_y, origin_x) in mm.
        direction: 3x3 direction cosine matrix in standard ITK/LPS (x, y, z) physical space.
    """

    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    direction: np.ndarray

    def __post_init__(self) -> None:
        """Validate geometry properties."""
        if len(self.shape) != 3 or any(d <= 0 for d in self.shape):
            raise ValueError(f"shape must be 3 positive integers (z, y, x), got {self.shape}.")
        if len(self.spacing_mm) != 3 or any(s <= 0 for s in self.spacing_mm):
            raise ValueError(
                f"spacing_mm must be 3 positive floats (z, y, x), got {self.spacing_mm}."
            )
        if len(self.origin_mm) != 3:
            raise ValueError(f"origin_mm must be 3 floats (z, y, x), got {self.origin_mm}.")
        if not isinstance(self.direction, np.ndarray) or self.direction.shape != (3, 3):
            shape_str = getattr(self.direction, "shape", None)
            raise ValueError(f"direction must be a 3x3 numpy array, got {shape_str}.")

    def voxel_to_lps_mm(
        self, index_zyx: tuple[float, float, float] | np.ndarray
    ) -> tuple[float, float, float]:
        """Convert a continuous or discrete voxel index (z, y, x) to physical LPS mm (x, y, z).

        LPS (Left, Posterior, Superior) is the standard ITK / SimpleITK coordinate space.

        Args:
            index_zyx: Continuous voxel index tuple (z, y, x).

        Returns:
            (x, y, z) physical coordinate tuple in mm under LPS convention.
        """
        idx_z = float(index_zyx[0])
        idx_y = float(index_zyx[1])
        idx_x = float(index_zyx[2])

        # Spacing in ITK order (x, y, z)
        sp_x = float(self.spacing_mm[2])
        sp_y = float(self.spacing_mm[1])
        sp_z = float(self.spacing_mm[0])

        scaled_xyz = np.array([idx_x * sp_x, idx_y * sp_y, idx_z * sp_z], dtype=np.float64)
        origin_xyz = np.array(
            [float(self.origin_mm[2]), float(self.origin_mm[1]), float(self.origin_mm[0])],
            dtype=np.float64,
        )

        pt_xyz = origin_xyz + self.direction @ scaled_xyz
        return (float(pt_xyz[0]), float(pt_xyz[1]), float(pt_xyz[2]))

    def lps_to_voxel(
        self, point_xyz: tuple[float, float, float] | np.ndarray
    ) -> tuple[float, float, float]:
        """Convert physical coordinates (x, y, z) in LPS mm to continuous voxel indices (z, y, x).

        Args:
            point_xyz: Physical point (x, y, z) in mm under LPS convention.

        Returns:
            Continuous voxel index tuple (z, y, x) in internal NumPy axis order.
        """
        pt = np.array(
            [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])],
            dtype=np.float64,
        )
        origin_xyz = np.array(
            [float(self.origin_mm[2]), float(self.origin_mm[1]), float(self.origin_mm[0])],
            dtype=np.float64,
        )
        sp_x = float(self.spacing_mm[2])
        sp_y = float(self.spacing_mm[1])
        sp_z = float(self.spacing_mm[0])

        diff = pt - origin_xyz
        inv_dir = np.linalg.inv(self.direction)
        scaled_xyz = inv_dir @ diff

        idx_x = float(scaled_xyz[0] / sp_x)
        idx_y = float(scaled_xyz[1] / sp_y)
        idx_z = float(scaled_xyz[2] / sp_z)

        return (idx_z, idx_y, idx_x)

    @staticmethod
    def lps_to_ras(
        point_xyz: tuple[float, float, float] | np.ndarray,
    ) -> tuple[float, float, float]:
        """Convert a physical coordinate from ITK's LPS convention to NIfTI/Slicer's RAS convention.

        LPS (Left, Posterior, Superior) is standard for ITK/SimpleITK.
        RAS (Right, Anterior, Superior) is standard for NIfTI and 3D Slicer.
        Transformation: negate x and y coordinates.

        Args:
            point_xyz: (x, y, z) coordinate tuple in LPS mm.

        Returns:
            (x, y, z) coordinate tuple in RAS mm.
        """
        return (-float(point_xyz[0]), -float(point_xyz[1]), float(point_xyz[2]))


def lps_to_ras(
    point_xyz: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Convert (x, y, z) from LPS mm to RAS mm by negating x and y coordinates.

    Documented conventions:
        - LPS: ITK / SimpleITK convention (+x = Left, +y = Posterior, +z = Superior)
        - RAS: NIfTI / 3D Slicer convention (+x = Right, +y = Anterior, +z = Superior)
    """
    return ImageGeometry.lps_to_ras(point_xyz)


def _extract_geometry(meta: Any) -> ImageGeometry:
    """Extract an ImageGeometry instance from an ImageGeometry, Case, dict, or tuple."""
    if isinstance(meta, ImageGeometry):
        return meta

    # Handle Case with images dictionary
    if hasattr(meta, "images") and hasattr(meta, "spacing_mm") and hasattr(meta, "origin_mm"):
        ref_img = next(iter(meta.images.values()))
        return ImageGeometry(
            shape=ref_img.shape,
            spacing_mm=tuple(float(s) for s in meta.spacing_mm),  # type: ignore[arg-type]
            origin_mm=tuple(float(o) for o in meta.origin_mm),    # type: ignore[arg-type]
            direction=np.array(meta.direction, dtype=np.float64),
        )

    # Handle Case or objects with single image attribute (for backwards compatibility)
    if hasattr(meta, "image") and hasattr(meta, "spacing_mm") and hasattr(meta, "origin_mm"):
        return ImageGeometry(
            shape=meta.image.shape,
            spacing_mm=tuple(float(s) for s in meta.spacing_mm),  # type: ignore[arg-type]
            origin_mm=tuple(float(o) for o in meta.origin_mm),    # type: ignore[arg-type]
            direction=np.array(meta.direction, dtype=np.float64),
        )

    # Handle dictionary representation
    if isinstance(meta, dict):
        shape = meta.get("shape") or meta.get("size")
        spacing = meta.get("spacing_mm") or meta.get("spacing")
        origin = meta.get("origin_mm") or meta.get("origin")
        direction = meta.get("direction")
        if shape is None or spacing is None or origin is None or direction is None:
            raise ValueError(
                f"Dictionary missing required geometry keys: {list(meta.keys())}"
            )
        return ImageGeometry(
            shape=tuple(int(s) for s in shape),  # type: ignore[arg-type]
            spacing_mm=tuple(float(s) for s in spacing),  # type: ignore[arg-type]
            origin_mm=tuple(float(o) for o in origin),  # type: ignore[arg-type]
            direction=np.array(direction, dtype=np.float64),
        )

    # Handle tuple/list: (shape, spacing, origin, direction)
    # or (image_array, spacing, origin, direction)
    if isinstance(meta, (tuple, list)) and len(meta) == 4:
        shape_or_img, sp, orig, direct = meta
        shape = shape_or_img.shape if isinstance(shape_or_img, np.ndarray) else tuple(shape_or_img)
        return ImageGeometry(
            shape=tuple(int(s) for s in shape),
            spacing_mm=tuple(float(s) for s in sp),
            origin_mm=tuple(float(o) for o in orig),
            direction=np.array(direct, dtype=np.float64),
        )

    raise TypeError(f"Cannot extract geometry from object of type {type(meta).__name__}.")


def assert_same_geometry(meta_a: Any, meta_b: Any, tol: float = 1e-4) -> None:
    """Verify that two image/mask geometries match within tolerance.

    Args:
        meta_a: First geometry specification (ImageGeometry, Case, dict, or tuple).
        meta_b: Second geometry specification.
        tol: Tolerance for floating point comparisons of spacing, origin, and direction.

    Raises:
        GeometryMismatchError: If array shape, spacing, origin, or direction differ.
    """
    geom_a = _extract_geometry(meta_a)
    geom_b = _extract_geometry(meta_b)

    if geom_a.shape != geom_b.shape:
        raise GeometryMismatchError(
            f"Array shape mismatch: {geom_a.shape} vs {geom_b.shape} (z, y, x)."
        )

    sp_a = np.array(geom_a.spacing_mm, dtype=np.float64)
    sp_b = np.array(geom_b.spacing_mm, dtype=np.float64)
    spacing_diff = float(np.max(np.abs(sp_a - sp_b)))
    if spacing_diff > tol:
        raise GeometryMismatchError(
            f"Voxel spacing mismatch: {geom_a.spacing_mm} vs {geom_b.spacing_mm} mm "
            f"(max difference: {spacing_diff:.6f} mm > tol: {tol:.6f} mm)."
        )

    orig_a = np.array(geom_a.origin_mm, dtype=np.float64)
    orig_b = np.array(geom_b.origin_mm, dtype=np.float64)
    origin_diff = float(np.max(np.abs(orig_a - orig_b)))
    if origin_diff > tol:
        raise GeometryMismatchError(
            f"Physical origin mismatch: {geom_a.origin_mm} vs {geom_b.origin_mm} mm "
            f"(max difference: {origin_diff:.6f} mm > tol: {tol:.6f} mm)."
        )

    dir_diff = float(np.max(np.abs(geom_a.direction - geom_b.direction)))
    if dir_diff > tol:
        raise GeometryMismatchError(
            f"Direction cosine matrix mismatch (max difference: {dir_diff:.6f} > tol: {tol:.6f})."
        )


def load_nifti_modalities(
    path: str | Path,
    names: list[str],
) -> tuple[dict[str, np.ndarray], ImageGeometry]:
    """Load a 4D NIfTI file containing multiple image modalities (e.g. MSD format).

    Extracts each 4D volume along the 4th dimension as a 3D float32 (z, y, x) array,
    takes the spatial 3x3 direction block, and converts SimpleITK (x, y, z) metadata
    to HDT's internal (z, y, x) convention.

    Args:
        path: Path to the 4D .nii or .nii.gz file.
        names: List of modality name strings corresponding to 4th-dimension channels.

    Returns:
        tuple containing:
            - dict mapping each name in `names` to a 3D float32 NumPy array in (z, y, x) order.
            - ImageGeometry representing the shared 3D spatial geometry.

    Raises:
        ValueError: If image is not 4D, or if 4th dimension size does not match len(names).
    """
    path_str = str(path)
    sitk_img = sitk.ReadImage(path_str)

    dim = sitk_img.GetDimension()
    if dim != 4:
        raise ValueError(
            f"Expected a 4D NIfTI image for multi-modality loading, but file '{path_str}' "
            f"has dimension {dim}."
        )

    sitk_size = sitk_img.GetSize()  # (x, y, z, channels)
    n_channels = sitk_size[3]
    if n_channels != len(names):
        raise ValueError(
            f"4th dimension size ({n_channels}) of image '{path_str}' does not match "
            f"number of requested modality names ({len(names)}: {names})."
        )

    arr_4d = sitk.GetArrayFromImage(sitk_img)  # sitk (x, y, z, c) -> numpy (c, z, y, x)

    # Convert spatial SimpleITK (x, y, z) spacing to HDT (z, y, x)
    sitk_spacing = sitk_img.GetSpacing()
    spacing_zyx = (float(sitk_spacing[2]), float(sitk_spacing[1]), float(sitk_spacing[0]))

    # Convert spatial SimpleITK (x, y, z) origin to HDT (z, y, x)
    sitk_origin = sitk_img.GetOrigin()
    origin_zyx = (float(sitk_origin[2]), float(sitk_origin[1]), float(sitk_origin[0]))

    # Direction cosine matrix: extract 3x3 spatial sub-block from 4x4 matrix
    dir_flat = sitk_img.GetDirection()
    dir_4x4 = np.array(dir_flat, dtype=np.float64).reshape((4, 4))
    dir_3x3 = np.array(dir_4x4[:3, :3], dtype=np.float64, copy=True)

    geom = ImageGeometry(
        shape=(int(arr_4d.shape[1]), int(arr_4d.shape[2]), int(arr_4d.shape[3])),
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=dir_3x3,
    )

    modalities: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        modalities[name] = np.array(arr_4d[i], dtype=np.float32, copy=True)

    return modalities, geom


def inspect_nifti_info(path: str | Path) -> dict[str, Any]:
    """Inspect NIfTI file properties without exposing SimpleITK outside load_scan.py.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Dictionary with keys:
            - dimension: int (3 or 4)
            - shape_zyx: tuple[int, ...]
            - spacing_zyx: tuple[float, ...]
            - origin_zyx: tuple[float, ...]
            - direction: np.ndarray (3x3 spatial block)
            - dtype: str
            - unique_labels: dict[int, int] (if integer/label file, mapping label -> voxel count)
    """
    path_str = str(path)
    sitk_img = sitk.ReadImage(path_str)
    dim = sitk_img.GetDimension()
    arr = sitk.GetArrayFromImage(sitk_img)

    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    dir_flat = sitk_img.GetDirection()

    if dim == 4:
        spacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        origin_zyx = (float(origin[2]), float(origin[1]), float(origin[0]))
        dir_4x4 = np.array(dir_flat, dtype=np.float64).reshape((4, 4))
        dir_3x3 = np.array(dir_4x4[:3, :3], dtype=np.float64, copy=True)
        shape_zyx = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
    else:
        spacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        origin_zyx = (float(origin[2]), float(origin[1]), float(origin[0]))
        dir_3x3 = np.array(dir_flat, dtype=np.float64).reshape((3, 3))
        shape_zyx = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))

    unique_labels: dict[int, int] = {}
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, counts, strict=True):
            unique_labels[int(v)] = int(c)
    elif np.all(np.equal(np.mod(arr, 1), 0)) and arr.max() <= 100 and arr.min() >= 0:
        # Float array containing discrete integer values
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, counts, strict=True):
            unique_labels[int(v)] = int(c)

    return {
        "dimension": dim,
        "shape_zyx": shape_zyx,
        "spacing_zyx": spacing_zyx,
        "origin_zyx": origin_zyx,
        "direction": dir_3x3,
        "dtype": str(arr.dtype),
        "unique_labels": unique_labels,
    }


def load_nifti_image(
    path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float], np.ndarray]:
    """Load a 3D NIfTI image file, converting SimpleITK (x, y, z) metadata to (z, y, x).

    Does NOT resample, reorient, crop, or silently alter image content.

    Args:
        path: Path to the .nii or .nii.gz image file.

    Returns:
        tuple containing:
            - image: 3D float32 NumPy array in (z, y, x) axis order.
            - spacing_zyx: Voxel spacing (sz, sy, sx) in mm.
            - origin_zyx: Origin coordinates (oz, oy, ox) in mm.
            - direction: 3x3 direction cosine matrix in standard ITK space.
    """
    path_str = str(path)
    sitk_img = sitk.ReadImage(path_str)

    # SimpleITK GetArrayFromImage converts (x, y, z) sitk index order to (z, y, x) numpy order
    arr = np.array(sitk.GetArrayFromImage(sitk_img), dtype=np.float32, copy=True)

    # Convert SimpleITK (x, y, z) spacing to HDT (z, y, x)
    sitk_spacing = sitk_img.GetSpacing()
    spacing_zyx = (float(sitk_spacing[2]), float(sitk_spacing[1]), float(sitk_spacing[0]))

    # Convert SimpleITK (x, y, z) origin to HDT (z, y, x)
    sitk_origin = sitk_img.GetOrigin()
    origin_zyx = (float(sitk_origin[2]), float(sitk_origin[1]), float(sitk_origin[0]))

    # Direction cosine matrix (3x3)
    direction = np.array(sitk_img.GetDirection(), dtype=np.float64).reshape((3, 3))

    return arr, spacing_zyx, origin_zyx, direction


_SITK_TO_NUMPY_DTYPE: dict[int, str] = {
    sitk.sitkInt8: "int8",
    sitk.sitkUInt8: "uint8",
    sitk.sitkInt16: "int16",
    sitk.sitkUInt16: "uint16",
    sitk.sitkInt32: "int32",
    sitk.sitkUInt32: "uint32",
    sitk.sitkFloat32: "float32",
    sitk.sitkFloat64: "float64",
}


def inspect_nifti_header(path: str | Path) -> dict[str, Any]:
    """Read NIfTI header metadata without loading large volumetric voxel arrays.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Dictionary containing dimension, shape_zyx, spacing_zyx, origin_zyx,
        direction (3x3 spatial), and dtype string.
    """
    path_str = str(path)
    reader = sitk.ImageFileReader()
    reader.SetFileName(path_str)
    reader.ReadImageInformation()

    dim = reader.GetDimension()
    size = reader.GetSize()
    spacing = reader.GetSpacing()
    origin = reader.GetOrigin()
    dir_flat = reader.GetDirection()
    pid = reader.GetPixelIDValue()
    dtype_str = _SITK_TO_NUMPY_DTYPE.get(pid, sitk.GetPixelIDValueAsString(pid))

    if dim == 3:
        shape_zyx = (size[2], size[1], size[0])
        spacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        origin_zyx = (float(origin[2]), float(origin[1]), float(origin[0]))
        dir_3x3 = np.array(dir_flat, dtype=np.float64).reshape((3, 3))
    elif dim == 4:
        shape_zyx = (size[3], size[2], size[1], size[0])
        spacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        origin_zyx = (float(origin[2]), float(origin[1]), float(origin[0]))
        dir_4x4 = np.array(dir_flat, dtype=np.float64).reshape((4, 4))
        dir_3x3 = np.array(dir_4x4[:3, :3], dtype=np.float64, copy=True)
    else:
        shape_zyx = tuple(reversed(size))
        spacing_zyx = tuple(float(s) for s in reversed(spacing))
        origin_zyx = tuple(float(o) for o in reversed(origin))
        dir_3x3 = np.eye(3, dtype=np.float64)

    return {
        "dimension": dim,
        "shape_zyx": shape_zyx,
        "spacing_zyx": spacing_zyx,
        "origin_zyx": origin_zyx,
        "direction": dir_3x3,
        "dtype": dtype_str,
    }


def load_nifti_mask_array(
    path: str | Path,
    image_meta: Any = None,
) -> tuple[np.ndarray, ImageGeometry, str]:
    """Load a 3D NIfTI segmentation mask preserving its native array values and dtype.

    Args:
        path: Path to .nii or .nii.gz file.
        image_meta: Reference geometry to assert against if provided.

    Returns:
        tuple containing:
            - mask_arr: 3D NumPy array in (z, y, x) order.
            - mask_geom: ImageGeometry of the mask file.
            - original_dtype: String representation of the native array dtype.

    Raises:
        GeometryMismatchError: If image_meta is provided and geometry does not match.
    """
    path_str = str(path)
    sitk_img = sitk.ReadImage(path_str)

    mask_arr = sitk.GetArrayFromImage(sitk_img)
    orig_dtype = str(mask_arr.dtype)

    sitk_spacing = sitk_img.GetSpacing()
    spacing_zyx = (float(sitk_spacing[2]), float(sitk_spacing[1]), float(sitk_spacing[0]))

    sitk_origin = sitk_img.GetOrigin()
    origin_zyx = (float(sitk_origin[2]), float(sitk_origin[1]), float(sitk_origin[0]))

    direction = np.array(sitk_img.GetDirection(), dtype=np.float64).reshape((3, 3))

    mask_geom = ImageGeometry(
        shape=mask_arr.shape,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction,
    )

    if image_meta is not None:
        assert_same_geometry(mask_geom, image_meta)

    return mask_arr, mask_geom, orig_dtype


def load_nifti_mask(
    path: str | Path,
    image_meta: Any = None,
    label_values: list[int] | int | None = None,
) -> np.ndarray | set[int]:
    """Load a 3D NIfTI segmentation mask and validate geometry against image_meta.

    Does NOT guess label values. If `label_values` is None, returns the set of unique
    integer labels discovered in the file.

    When `label_values` is provided:
        - If multiple labels are given (e.g. [1, 2, 4]), returns the boolean union mask
          where any of those labels are present.
        - If specified labels are absent from the file (e.g. [3]), returns an all-False
          boolean mask of shape (z, y, x). This explicitly models the empty-mask edge case.

    Args:
        path: Path to the .nii or .nii.gz mask file.
        image_meta: Reference geometry (ImageGeometry, Case, dict, or tuple) to assert against.
        label_values: List of integer label values to extract as True in the boolean mask.
            If None, returns unique label values found in the file instead of returning a mask.

    Returns:
        - If label_values is None: set[int] of unique labels present in the file.
        - If label_values is provided: 3D boolean NumPy array in (z, y, x) order.

    Raises:
        GeometryMismatchError: If image_meta is provided and geometry does not match within tol.
    """
    path_str = str(path)
    sitk_img = sitk.ReadImage(path_str)

    # sitk index (x, y, z) -> numpy array (z, y, x)
    mask_arr = sitk.GetArrayFromImage(sitk_img)

    sitk_spacing = sitk_img.GetSpacing()
    spacing_zyx = (float(sitk_spacing[2]), float(sitk_spacing[1]), float(sitk_spacing[0]))

    sitk_origin = sitk_img.GetOrigin()
    origin_zyx = (float(sitk_origin[2]), float(sitk_origin[1]), float(sitk_origin[0]))

    direction = np.array(sitk_img.GetDirection(), dtype=np.float64).reshape((3, 3))

    mask_geom = ImageGeometry(
        shape=mask_arr.shape,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        direction=direction,
    )

    if image_meta is not None:
        assert_same_geometry(mask_geom, image_meta)

    if label_values is None:
        return set(int(val) for val in np.unique(mask_arr))

    labels_list = [label_values] if isinstance(label_values, int) else list(label_values)
    bool_mask = np.isin(mask_arr, labels_list)
    return bool_mask


def save_nifti_mask(
    mask: np.ndarray,
    reference_meta: Any,
    path: str | Path,
) -> None:
    """Save a 3D boolean (or integer) mask to a NIfTI file preserving reference geometry.

    Converts internal HDT (z, y, x) metadata to SimpleITK (x, y, z) metadata.

    Args:
        mask: 3D boolean or integer NumPy array in (z, y, x) order.
        reference_meta: Reference geometry providing shape, spacing_mm, origin_mm, and direction.
        path: Output destination path (.nii or .nii.gz).
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 3:
        raise ValueError(
            f"mask must be a 3D numpy array (z, y, x), got ndim={getattr(mask, 'ndim', None)}."
        )

    ref = _extract_geometry(reference_meta)
    if mask.shape != ref.shape:
        raise GeometryMismatchError(
            f"Mask shape {mask.shape} does not match reference shape {ref.shape} (z, y, x)."
        )

    # GetImageFromArray accepts (z, y, x) array and creates SimpleITK image with size (x, y, z)
    sitk_img = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))

    # Convert HDT (z, y, x) back to SimpleITK (x, y, z)
    sitk_img.SetSpacing(
        (float(ref.spacing_mm[2]), float(ref.spacing_mm[1]), float(ref.spacing_mm[0]))
    )
    sitk_img.SetOrigin(
        (float(ref.origin_mm[2]), float(ref.origin_mm[1]), float(ref.origin_mm[0]))
    )
    sitk_img.SetDirection(tuple(float(v) for v in ref.direction.flatten()))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_img, str(path))


def save_nifti_image(
    image: np.ndarray,
    reference_meta: Any,
    path: str | Path,
) -> None:
    """Save a 3D float32 image to a NIfTI file preserving reference geometry.

    Converts internal HDT (z, y, x) metadata to SimpleITK (x, y, z) metadata.

    Args:
        image: 3D float32 NumPy array in (z, y, x) order.
        reference_meta: Reference geometry providing shape, spacing_mm, origin_mm, and direction.
        path: Output destination path (.nii or .nii.gz).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError(
            f"image must be a 3D numpy array (z, y, x), got ndim={getattr(image, 'ndim', None)}."
        )

    ref = _extract_geometry(reference_meta)
    if image.shape != ref.shape:
        raise GeometryMismatchError(
            f"Image shape {image.shape} does not match reference shape {ref.shape} (z, y, x)."
        )

    sitk_img = sitk.GetImageFromArray(image.astype(np.float32, copy=False))

    sitk_img.SetSpacing(
        (float(ref.spacing_mm[2]), float(ref.spacing_mm[1]), float(ref.spacing_mm[0]))
    )
    sitk_img.SetOrigin(
        (float(ref.origin_mm[2]), float(ref.origin_mm[1]), float(ref.origin_mm[0]))
    )
    sitk_img.SetDirection(tuple(float(v) for v in ref.direction.flatten()))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_img, str(path))
