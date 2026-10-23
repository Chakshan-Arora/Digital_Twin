"""Domain-agnostic representation of an anatomical or pathological structure."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Structure:
    """Represents a segmented 3D volumetric structure.

    Attributes:
        name: Unique identifier/name of the structure (e.g., 'tumor', 'vessel').
        mask: 3D boolean NumPy array in (z, y, x) axis order indicating structure occupancy.
        role: Functional role of the structure:
            - 'target': structure intended for resection/targeting.
            - 'risk': organ/structure at risk to avoid.
            - 'context': background anatomy or anatomical reference.

    Axes:
        NumPy indexing order: (z, y, x)
    """

    name: str
    mask: np.ndarray
    role: Literal["target", "risk", "context"]

    def __post_init__(self) -> None:
        """Validate structure attributes strictly without implicit conversions."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"Structure name must be a non-empty string, got {self.name!r}.")

        if not isinstance(self.mask, np.ndarray):
            raise ValueError(
                f"Structure mask must be a numpy.ndarray, got {type(self.mask).__name__}."
            )

        if self.mask.dtype != bool:
            raise ValueError(
                f"Structure mask must have boolean dtype (bool), got dtype {self.mask.dtype}."
            )

        if self.mask.ndim != 3:
            raise ValueError(
                f"Structure mask must be 3-dimensional with axis order (z, y, x), "
                f"got {self.mask.ndim}D array with shape {self.mask.shape}."
            )

        allowed_roles = {"target", "risk", "context"}
        if self.role not in allowed_roles:
            raise ValueError(
                f"Structure role must be one of {sorted(allowed_roles)}, got {self.role!r}."
            )
