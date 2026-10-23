"""Precomputed signed distance field evaluator for accelerated pairwise clearance.

Pure geometry helper designed for Monte Carlo sweeps where the reference/risk structure
remains static while the primary/target structure is repeatedly perturbed.

Guarantees exact mathematical equivalence to `signed_clearance_mm(a, b, spacing, crop_to_bbox=True)`
under the face-to-face voxel sets convention.
"""

from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from hdt.geometry.distances import _validate_mask_inputs


class PrecomputedRiskClearance:
    """Precomputes signed distance transforms on a static reference mask.

    Allows repeated face-to-face clearance queries against perturbed target masks
    in O(bounding box) lookup time without repeating Euclidean distance transforms.
    """

    def __init__(
        self,
        risk: np.ndarray,
        spacing_mm: Sequence[float],
        target_bbox_slices: tuple[slice, slice, slice] | None = None,
        max_distance_mm: float = 40.0,
        margin_voxels: int = 5,
    ) -> None:
        """Initialize precomputed distance fields for the risk structure.

        Args:
            risk: 3D boolean array of the reference risk structure (z, y, x).
            spacing_mm: (sz, sy, sx) voxel spacing in millimetres.
            target_bbox_slices: Slices bounding all possible target voxels.
            max_distance_mm: Maximum distance threshold for censoring in mm.
            margin_voxels: Safety padding margin in voxels.
        """
        sz, sy, sx = _validate_mask_inputs(risk, risk, spacing_mm)
        self.spacing = (sz, sy, sx)
        self.max_distance_mm = float(max_distance_mm)
        self.full_shape = risk.shape

        coords_r = np.argwhere(risk)
        if len(coords_r) == 0:
            raise ValueError("Cannot evaluate clearance against an empty risk mask")

        if target_bbox_slices is not None:
            min_t = np.array([s.start for s in target_bbox_slices], dtype=np.int64)
            max_t = np.array([s.stop for s in target_bbox_slices], dtype=np.int64)
        else:
            min_t = coords_r.min(axis=0)
            max_t = coords_r.max(axis=0) + 1

        min_u = np.maximum(np.minimum(min_t, coords_r.min(axis=0)) - margin_voxels, 0)
        max_u = np.minimum(np.maximum(max_t, coords_r.max(axis=0) + 1) + margin_voxels, risk.shape)
        self.slices_u = tuple(slice(int(min_u[i]), int(max_u[i])) for i in range(3))

        sub_r = risk[self.slices_u]
        pad_r = np.pad(sub_r, 1, mode="constant", constant_values=False)
        self.sub_r = sub_r

        # Inside EDT for penetration depth on overlap
        self.dist_inside = distance_transform_edt(pad_r, sampling=self.spacing)[1:-1, 1:-1, 1:-1]

        # Outside EDT for disjoint clearance and closest voxel indices
        dist_pad, idx_pad = distance_transform_edt(
            ~pad_r, sampling=self.spacing, return_indices=True
        )
        self.dist_outside = dist_pad[1:-1, 1:-1, 1:-1]
        self.idx_pad = idx_pad

    def evaluate(self, target: np.ndarray) -> float:
        """Evaluate exact face-to-face signed clearance from target to risk in mm.

        Args:
            target: 3D boolean mask of primary target structure (z, y, x).

        Returns:
            float: Signed clearance in millimetres. Positive = disjoint gap,
                   0 = touching, negative = penetration depth.
        """
        if not np.any(target):
            raise ValueError("Target mask cannot be empty")

        sub_a = target[self.slices_u]
        overlap = sub_a & self.sub_r

        if np.any(overlap):
            # Overlapping: compute negative penetration depth
            max_depth = float(np.max(self.dist_inside[overlap]))
            return -max_depth

        # Disjoint: compute closest voxel pair with 2t face-to-face correction
        dist_in_a = np.where(sub_a, self.dist_outside, np.inf)
        flat_idx = int(np.argmin(dist_in_a))
        za, ya, xa = np.unravel_index(flat_idx, sub_a.shape)

        zb = int(self.idx_pad[0][za + 1, ya + 1, xa + 1]) - 1
        yb = int(self.idx_pad[1][za + 1, ya + 1, xa + 1]) - 1
        xb = int(self.idx_pad[2][za + 1, ya + 1, xa + 1]) - 1

        sz, sy, sx = self.spacing
        pa = np.array([za * sz, ya * sy, xa * sx], dtype=np.float64)
        pb = np.array([zb * sz, yb * sy, xb * sx], dtype=np.float64)

        diff = pa - pb
        norm = float(np.linalg.norm(diff))
        if norm < 1e-12:
            return 0.0

        u = np.abs(diff / norm)
        ratios = [self.spacing[i] / (2.0 * u[i]) for i in range(3) if u[i] > 1e-9]
        t = min(ratios)

        corrected = norm - 2.0 * t
        return float(max(0.0, corrected))
