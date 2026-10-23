"""Core data model representing a patient/subject imaging case and its segmented structures."""

from dataclasses import dataclass, field

import numpy as np

from hdt.core.structure import Structure


@dataclass
class Case:
    """Volumetric image modalities and associated segmented structures for a subject.

    Axis and Coordinate System Conventions:
        - `images`: Dictionary mapping modality identifiers (e.g. "t1", "t1ce", "t2", "flair")
          to 3D NumPy arrays indexed in `(z, y, x)` order with `float32` intensities.
          All images must share the exact same spatial dimensions. At least one image is required.
        - `primary`: Property returning the name of the default/primary image modality.
        - `spacing_mm`: Shared voxel dimensions `(spacing_z, spacing_y, spacing_x)` in millimeters.
        - `origin_mm`: Shared physical coordinate of the (0, 0, 0) voxel index, ordered as
          `(origin_z, origin_y, origin_x)` in millimeters.
        - `direction`: 3x3 NumPy array representing the direction cosine matrix in standard
          ITK/LPS (x, y, z) physical space, matching `sitk.Image.GetDirection()`.
          CRITICAL ARCHITECTURAL RULE: Nothing outside `src/hdt/imaging/load_scan.py` reads
          `direction`. Spatial transformations and coordinates are managed strictly via
          geometry utilities in `load_scan.py`.
        - `structures`: Dictionary mapping structure names to `Structure` instances.
          Every structure mask MUST match the shared image shape in `(z, y, x)` order.

    Coordinate Conventions:
        - LPS (Left, Posterior, Superior): The standard coordinate convention of SimpleITK / ITK.
        - RAS (Right, Anterior, Superior): The standard convention used by NIfTI and 3D Slicer.

    Privacy & Clinical Note:
        `case_id` MUST be an opaque identifier (e.g. 'phantom_001', 'case_42a').
        Never use patient names, MRNs, or sensitive clinical identifiers.
        This data model is part of a research prototype and is NOT a clinical tool.
    """

    case_id: str
    images: dict[str, np.ndarray]
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    direction: np.ndarray
    structures: dict[str, Structure] = field(default_factory=dict)
    _primary: str | None = None

    def __init__(
        self,
        case_id: str,
        images: dict[str, np.ndarray],
        spacing_mm: tuple[float, float, float],
        origin_mm: tuple[float, float, float],
        direction: np.ndarray,
        structures: dict[str, Structure] | None = None,
        primary: str | None = None,
    ) -> None:
        self.case_id = case_id
        self.images = images
        self.spacing_mm = spacing_mm
        self.origin_mm = origin_mm
        self.direction = direction
        self.structures = {} if structures is None else structures
        self._primary = primary
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate case attributes without implicit conversions."""
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError(f"case_id must be a non-empty string, got {self.case_id!r}.")

        if not isinstance(self.images, dict) or len(self.images) == 0:
            raise ValueError("images must be a non-empty dictionary of float32 3D numpy arrays.")

        ref_shape: tuple[int, ...] | None = None
        for name, img in self.images.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Image name key must be a non-empty string, got {name!r}.")
            if not isinstance(img, np.ndarray):
                raise ValueError(
                    f"Image '{name}' must be a numpy.ndarray, got {type(img).__name__}."
                )
            if img.dtype != np.float32:
                raise ValueError(
                    f"Image '{name}' must have float32 dtype, got {img.dtype}."
                )
            if img.ndim != 3:
                raise ValueError(
                    f"Image '{name}' must be 3-dimensional (z, y, x), got ndim={img.ndim}."
                )
            if ref_shape is None:
                ref_shape = img.shape
            elif img.shape != ref_shape:
                raise ValueError(
                    f"Image '{name}' shape {img.shape} does not match reference shape {ref_shape}."
                )

        if self._primary is not None:
            if self._primary not in self.images:
                raise ValueError(
                    f"Primary image '{self._primary}' not found in images: "
                    f"{list(self.images.keys())}."
                )
        else:
            self._primary = next(iter(self.images.keys()))

        if len(self.spacing_mm) != 3 or any(s <= 0 for s in self.spacing_mm):
            raise ValueError(
                f"spacing_mm must be a 3-tuple of positive floats (z, y, x), got {self.spacing_mm}."
            )

        if len(self.origin_mm) != 3:
            raise ValueError(
                f"origin_mm must be a 3-tuple of floats (z, y, x), got {self.origin_mm}."
            )

        if not isinstance(self.direction, np.ndarray) or self.direction.shape != (3, 3):
            shape_str = getattr(self.direction, "shape", None)
            raise ValueError(f"direction must be a 3x3 numpy array, got {shape_str}.")

        # Validate initial structures if provided
        for name, struct in self.structures.items():
            if struct.name != name:
                raise ValueError(
                    f"Structure key '{name}' does not match structure.name '{struct.name}'."
                )
            if struct.mask.shape != ref_shape:
                raise ValueError(
                    f"Structure '{name}' mask shape {struct.mask.shape} does not match "
                    f"images shape {ref_shape}."
                )

    @property
    def primary(self) -> str:
        """Name of the default / primary image modality."""
        if self._primary is not None and self._primary in self.images:
            return self._primary
        return next(iter(self.images.keys()))

    @primary.setter
    def primary(self, name: str) -> None:
        """Set the default / primary image modality."""
        if name not in self.images:
            raise KeyError(
                f"Image '{name}' not found in images: {list(self.images.keys())}."
            )
        self._primary = name

    @property
    def shape(self) -> tuple[int, int, int]:
        """Spatial shape of the images (z, y, x)."""
        ref_img = self.images[self.primary]
        return (ref_img.shape[0], ref_img.shape[1], ref_img.shape[2])

    @property
    def image(self) -> np.ndarray:
        """Convenience property returning the primary image array (float32, (z, y, x))."""
        return self.images[self.primary]

    def add_structure(self, s: Structure) -> None:
        """Add a structure to the case.

        Args:
            s: Structure instance to add.

        Raises:
            ValueError: If structure shape does not match images shape or if name already exists.
        """
        if s.name in self.structures:
            raise ValueError(
                f"Structure with name '{s.name}' already exists in Case '{self.case_id}'."
            )

        if s.mask.shape != self.shape:
            raise ValueError(
                f"Structure '{s.name}' mask shape {s.mask.shape} does not match "
                f"images shape {self.shape} (z, y, x)."
            )

        self.structures[s.name] = s

    def get(self, name: str) -> Structure:
        """Retrieve a structure by name.

        Args:
            name: Structure name.

        Returns:
            The corresponding Structure instance.

        Raises:
            KeyError: If the structure name does not exist.
        """
        if name not in self.structures:
            raise KeyError(
                f"Structure '{name}' not found in Case '{self.case_id}'. "
                f"Available structures: {list(self.structures.keys())}"
            )
        return self.structures[name]

    def names(self) -> list[str]:
        """Return list of names of all structures present in the case."""
        return list(self.structures.keys())
