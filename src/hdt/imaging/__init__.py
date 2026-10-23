"""Imaging module: NIfTI I/O and synthetic phantoms."""

from hdt.imaging.load_scan import (
    GeometryMismatchError,
    ImageGeometry,
    assert_same_geometry,
    inspect_nifti_header,
    inspect_nifti_info,
    load_nifti_image,
    load_nifti_mask,
    load_nifti_mask_array,
    load_nifti_modalities,
    lps_to_ras,
    save_nifti_image,
    save_nifti_mask,
)
from hdt.imaging.phantoms import make_ellipsoid, make_phantom_case, make_sphere

__all__ = [
    "GeometryMismatchError",
    "ImageGeometry",
    "assert_same_geometry",
    "inspect_nifti_header",
    "inspect_nifti_info",
    "load_nifti_image",
    "load_nifti_mask",
    "load_nifti_mask_array",
    "load_nifti_modalities",
    "lps_to_ras",
    "save_nifti_image",
    "save_nifti_mask",
    "make_ellipsoid",
    "make_phantom_case",
    "make_sphere",
]
