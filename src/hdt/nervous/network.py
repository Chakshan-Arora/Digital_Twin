"""Procedural 3D nervous system network generator for whole-body visualisation.

Constructs full 3D polyline pathways including an anatomically curved spinal
cord (Catmull-Rom spline), brain landmarks, 31 pairs of spinal roots, brachial
plexus, intercostal nerves, lumbosacral plexus, and cranial nerve X (vagus).

All coordinates in RAS millimetres.
Origin (0,0,0) is floor midway between feet.
x: person's right (+x), left (-x)
y: anterior (+y), posterior (-y)
z: superior (+z)

Research prototype - NOT a clinical tool.
"""

import math
from typing import Literal

from pydantic import BaseModel, Field

from hdt.body.models import BodyParams


class BrainStructure(BaseModel):
    """Anatomical representation of an intracranial brain structure."""

    name: str = Field(
        ...,
        description="Structure name, e.g. 'cerebrum', 'cerebellum', 'brainstem', 'thalamus'",
    )
    centroid_ras_mm: tuple[float, float, float] = Field(
        ..., description="Center of mass [x, y, z] in RAS millimetres"
    )
    dimensions_mm: tuple[float, float, float] = Field(
        ..., description="Bounding box extent [dx, dy, dz] in millimetres"
    )
    boundary_points: list[tuple[float, float, float]] = Field(
        default_factory=list,
        description="Key boundary landmarks or contour points in RAS millimetres",
    )


class NervePolyline(BaseModel):
    """Renderable 3D polyline representing a nerve tract or branch."""

    name: str = Field(..., description="Anatomical nerve or branch identifier")
    category: str = Field(
        ...,
        description="Plexus category (spinal_root, brachial, intercostal, lumbosacral, cranial)",
    )
    side: Literal["l", "r", "midline"] = Field(
        ..., description="Anatomical lateralisation: 'l', 'r', or 'midline'"
    )
    points: list[tuple[float, float, float]] = Field(
        ..., description="Sequence of 3D vertices [x, y, z] in RAS millimetres"
    )


class NerveNetworkResponse(BaseModel):
    """Complete 3D nervous system geometry payload for rendering."""

    detail: Literal["low", "high"] = Field(
        ..., description="Mesh / polyline resolution level"
    )
    total_vertex_count: int = Field(
        ..., description="Total count of 3D vertices across all nerve polylines"
    )
    spinal_cord: list[tuple[float, float, float]] = Field(
        ..., description="Smooth Catmull-Rom spinal cord polyline from brainstem to conus"
    )
    brain_structures: list[BrainStructure] = Field(
        ..., description="Intracranial brain regions and somatosensory cortical landmarks"
    )
    spinal_roots: list[NervePolyline] = Field(
        ..., description="31 bilateral pairs of spinal nerve roots (C1-Co1)"
    )
    brachial_plexus: list[NervePolyline] = Field(
        ...,
        description="Trunks, cords, and terminal arm nerves (axillary, radial, median, ulnar)",
    )
    intercostal_nerves: list[NervePolyline] = Field(
        ..., description="Thoracic intercostal nerves wrapping the ribcage (T1-T11)"
    )
    lumbosacral_plexus: list[NervePolyline] = Field(
        ...,
        description="Lower extremity nerves (femoral, obturator, saphenous, sciatic, fibular)",
    )
    cranial_nerves: list[NervePolyline] = Field(
        ..., description="Cranial nerves including bilateral vagus (CN X)"
    )
    disclaimer: str = Field(
        default=(
            "Research prototype for 3D educational anatomical visualisation. "
            "NOT a clinical tool."
        ),
        description="Clinical disclaimer",
    )


def _catmull_rom_resample(
    control_points: list[tuple[float, float, float]],
    step_mm: float = 10.0,
) -> list[tuple[float, float, float]]:
    """Resample a 3D Catmull-Rom spline passing through control points at regular intervals.

    Parameters
    ----------
    control_points : list[tuple[float, float, float]]
        Ordered anchor points defining the spline trajectory.
    step_mm : float
        Nominal sampling arc-length in millimetres.

    Returns
    -------
    list[tuple[float, float, float]]
        Evenly spaced 3D polyline vertices in RAS mm.
    """
    if len(control_points) < 2:
        return [(round(x, 2), round(y, 2), round(z, 2)) for x, y, z in control_points]

    # Duplicate endpoints to form natural cubic Catmull-Rom boundary conditions
    pts = [control_points[0]] + list(control_points) + [control_points[-1]]
    res: list[tuple[float, float, float]] = []

    m = len(pts)
    for i in range(1, m - 2):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2]

        seg_dist = math.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
        )
        steps = max(1, round(seg_dist / step_mm))

        for s in range(steps):
            u = s / float(steps)
            u2 = u * u
            u3 = u2 * u

            # Catmull-Rom basis matrix evaluation
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * u
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * u2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * u3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * u
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * u2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * u3
            )
            z = 0.5 * (
                (2 * p1[2])
                + (-p0[2] + p2[2]) * u
                + (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * u2
                + (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * u3
            )
            res.append((round(x, 2), round(y, 2), round(z, 2)))

    # Add final endpoint
    last = control_points[-1]
    res.append((round(last[0], 2), round(last[1], 2), round(last[2], 2)))
    return res


def _make_box_boundary(
    cx: float, cy: float, cz: float, dx: float, dy: float, dz: float
) -> list[tuple[float, float, float]]:
    """Generate 8 bounding vertices for an anatomical structure."""
    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    return [
        (round(cx + sx * hx, 2), round(cy + sy * hy, 2), round(cz + sz * hz, 2))
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]


def build_network(
    body: BodyParams,
    detail: Literal["low", "high"] = "high",
) -> NerveNetworkResponse:
    """Generate the complete 3D nervous system geometry for a given body skeleton.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal stature and joint positions in RAS mm.
    detail : {'low', 'high'}
        Resolution preset. 'high' uses ~10 mm polyline spacing; 'low' uses ~25 mm.

    Returns
    -------
    NerveNetworkResponse
        Complete renderable nervous system including brain landmarks and nerve tracts.
    """
    h = body.height_mm
    step_mm = 10.0 if detail == "high" else 25.0

    # -------------------------------------------------------------------------
    # 1. Brain structures
    # -------------------------------------------------------------------------
    brain_structs: list[BrainStructure] = [
        BrainStructure(
            name="cerebrum",
            centroid_ras_mm=(0.0, round(0.005 * h, 2), round(0.940 * h, 2)),
            dimensions_mm=(round(0.080 * h, 2), round(0.095 * h, 2), round(0.065 * h, 2)),
            boundary_points=_make_box_boundary(
                0.0, 0.005 * h, 0.940 * h, 0.080 * h, 0.095 * h, 0.065 * h
            ),
        ),
        BrainStructure(
            name="cerebellum",
            centroid_ras_mm=(0.0, round(-0.025 * h, 2), round(0.895 * h, 2)),
            dimensions_mm=(round(0.060 * h, 2), round(0.040 * h, 2), round(0.035 * h, 2)),
            boundary_points=_make_box_boundary(
                0.0, -0.025 * h, 0.895 * h, 0.060 * h, 0.040 * h, 0.035 * h
            ),
        ),
        BrainStructure(
            name="brainstem",
            centroid_ras_mm=(0.0, round(-0.010 * h, 2), round(0.885 * h, 2)),
            dimensions_mm=(round(0.025 * h, 2), round(0.025 * h, 2), round(0.045 * h, 2)),
            boundary_points=_make_box_boundary(
                0.0, -0.010 * h, 0.885 * h, 0.025 * h, 0.025 * h, 0.045 * h
            ),
        ),
        BrainStructure(
            name="thalamus",
            centroid_ras_mm=(0.0, 0.0, round(0.915 * h, 2)),
            dimensions_mm=(round(0.030 * h, 2), round(0.025 * h, 2), round(0.025 * h, 2)),
            boundary_points=_make_box_boundary(
                0.0, 0.0, 0.915 * h, 0.030 * h, 0.025 * h, 0.025 * h
            ),
        ),
        BrainStructure(
            name="cortex_S1_r",
            centroid_ras_mm=(round(0.030 * h, 2), round(-0.005 * h, 2), round(0.955 * h, 2)),
            dimensions_mm=(round(0.020 * h, 2), round(0.015 * h, 2), round(0.020 * h, 2)),
            boundary_points=_make_box_boundary(
                0.030 * h, -0.005 * h, 0.955 * h, 0.020 * h, 0.015 * h, 0.020 * h
            ),
        ),
        BrainStructure(
            name="cortex_S1_l",
            centroid_ras_mm=(round(-0.030 * h, 2), round(-0.005 * h, 2), round(0.955 * h, 2)),
            dimensions_mm=(round(0.020 * h, 2), round(0.015 * h, 2), round(0.020 * h, 2)),
            boundary_points=_make_box_boundary(
                -0.030 * h, -0.005 * h, 0.955 * h, 0.020 * h, 0.015 * h, 0.020 * h
            ),
        ),
    ]

    # -------------------------------------------------------------------------
    # 2. Spinal cord: Catmull-Rom spline with anatomical curvatures
    # -------------------------------------------------------------------------
    # Sagittal curvature:
    # - Cervical lordosis (anterior bow around C4-C5)
    # - Thoracic kyphosis (posterior bow around T6-T7)
    # - Lumbar lordosis (anterior bow around L3-L4)
    # - Sacral curvature (conus medullaris and cauda equina descent)
    cord_ctrl_pts: list[tuple[float, float, float]] = [
        (0.0, -0.012 * h, 0.880 * h),  # Foramen magnum / brainstem
        (0.0, -0.015 * h, 0.840 * h),  # High cervical C2
        (0.0, -0.006 * h, 0.795 * h),  # Mid cervical lordosis apex (C4-C5)
        (0.0, -0.018 * h, 0.745 * h),  # C7-T1 transition
        (0.0, -0.030 * h, 0.650 * h),  # Mid-thoracic kyphosis apex (T6-T7)
        (0.0, -0.020 * h, 0.580 * h),  # T12-L1 transition
        (0.0, -0.012 * h, 0.550 * h),  # Lumbar lordosis apex (L3-L4)
        (0.0, -0.016 * h, 0.525 * h),  # Conus medullaris
        (0.0, -0.020 * h, 0.510 * h),  # Filum terminale / coccygeal apex
    ]
    spinal_cord = _catmull_rom_resample(cord_ctrl_pts, step_mm=step_mm)

    # -------------------------------------------------------------------------
    # 3. 31 pairs of spinal roots (C1-C8, T1-T12, L1-L5, S1-S5, Co1)
    # -------------------------------------------------------------------------
    # Z fractions for all 31 pairs positioned anatomically along spinal column
    root_fractions: list[tuple[str, float]] = [
        ("C1", 0.855), ("C2", 0.842), ("C3", 0.828), ("C4", 0.815),
        ("C5", 0.802), ("C6", 0.788), ("C7", 0.775), ("C8", 0.760),
        ("T1", 0.745), ("T2", 0.730), ("T3", 0.715), ("T4", 0.700),
        ("T5", 0.685), ("T6", 0.670), ("T7", 0.655), ("T8", 0.640),
        ("T9", 0.625), ("T10", 0.610), ("T11", 0.595), ("T12", 0.580),
        ("L1", 0.565), ("L2", 0.552), ("L3", 0.542), ("L4", 0.535),
        ("L5", 0.528), ("S1", 0.522), ("S2", 0.518), ("S3", 0.514),
        ("S4", 0.511), ("S5", 0.509), ("Co1", 0.507),
    ]

    def _get_cord_y(z_target: float) -> float:
        """Find the interpolated y coordinate on the spinal cord at height z_target."""
        best_pt = min(spinal_cord, key=lambda pt: abs(pt[2] - z_target))
        return best_pt[1]

    spinal_roots: list[NervePolyline] = []
    root_foramen_pos: dict[str, dict[str, tuple[float, float, float]]] = {}

    for name, z_frac in root_fractions:
        z_pos = z_frac * h
        cord_y = _get_cord_y(z_pos)
        root_foramen_pos[name] = {}

        # Lateral exit width through foramen
        if name.startswith("C"):
            lat_w = 0.022 * h
        elif name.startswith("T"):
            lat_w = 0.026 * h
        else:
            lat_w = 0.020 * h

        for s_char, sign in [("r", 1.0), ("l", -1.0)]:
            foramen_pt = (
                round(sign * lat_w, 2),
                round(cord_y - 0.003 * h, 2),
                round(z_pos - 0.004 * h, 2),
            )
            root_foramen_pos[name][s_char] = foramen_pt

            # Root polyline
            mid_pt = (
                round(sign * lat_w * 0.5, 2),
                round(cord_y - 0.001 * h, 2),
                round(z_pos - 0.002 * h, 2),
            )
            root_pts = [
                (0.0, round(cord_y, 2), round(z_pos, 2)),
                mid_pt,
                foramen_pt,
            ]
            spinal_roots.append(
                NervePolyline(
                    name=f"root_{name}_{s_char}",
                    category="spinal_root",
                    side=s_char,  # type: ignore[arg-type]
                    points=root_pts,
                )
            )

    # -------------------------------------------------------------------------
    # 4. Brachial Plexus (C5 - T1) into trunks, cords, and terminal arm nerves
    # -------------------------------------------------------------------------
    brachial_nerves: list[NervePolyline] = []

    for s_char, sign in [("r", 1.0), ("l", -1.0)]:
        # Trunks
        trunk_sup = (round(sign * 0.055 * h, 2), round(-0.004 * h, 2), round(0.790 * h, 2))
        trunk_mid = (round(sign * 0.058 * h, 2), round(-0.006 * h, 2), round(0.775 * h, 2))
        trunk_inf = (round(sign * 0.060 * h, 2), round(-0.008 * h, 2), round(0.755 * h, 2))

        # Trunks polylines
        brachial_nerves.append(
            NervePolyline(
                name=f"brachial_superior_trunk_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=[root_foramen_pos["C5"][s_char], trunk_sup],
            )
        )
        brachial_nerves.append(
            NervePolyline(
                name=f"brachial_middle_trunk_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=[root_foramen_pos["C7"][s_char], trunk_mid],
            )
        )
        brachial_nerves.append(
            NervePolyline(
                name=f"brachial_inferior_trunk_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=[root_foramen_pos["T1"][s_char], trunk_inf],
            )
        )

        # Axillary cords
        cord_lateral = (round(sign * 0.105 * h, 2), round(-0.002 * h, 2), round(0.785 * h, 2))
        cord_posterior = (round(sign * 0.108 * h, 2), round(-0.006 * h, 2), round(0.780 * h, 2))
        cord_medial = (round(sign * 0.105 * h, 2), round(-0.004 * h, 2), round(0.765 * h, 2))

        # Cords polylines
        brachial_nerves.append(
            NervePolyline(
                name=f"brachial_lateral_cord_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=[trunk_sup, cord_lateral],
            )
        )
        brachial_nerves.append(
            NervePolyline(
                name=f"brachial_posterior_cord_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=[trunk_mid, cord_posterior],
            )
        )
        brachial_nerves.append(
            NervePolyline(
                name=f"brachial_medial_cord_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=[trunk_inf, cord_medial],
            )
        )

        # Terminal Nerves down the arm:
        # a) Axillary nerve
        brachial_nerves.append(
            NervePolyline(
                name=f"nerve_axillary_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        cord_posterior,
                        (sign * 0.114 * h, -0.010 * h, 0.795 * h),
                        (sign * 0.125 * h, -0.005 * h, 0.775 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # b) Musculocutaneous nerve
        brachial_nerves.append(
            NervePolyline(
                name=f"nerve_musculocutaneous_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        cord_lateral,
                        (sign * 0.116 * h, 0.006 * h, 0.730 * h),
                        (sign * 0.136 * h, 0.005 * h, 0.640 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # c) Radial nerve (posterior arm -> lateral elbow -> dorsal forearm/hand)
        brachial_nerves.append(
            NervePolyline(
                name=f"nerve_radial_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        cord_posterior,
                        (sign * 0.118 * h, -0.008 * h, 0.720 * h),
                        (sign * 0.138 * h, -0.004 * h, 0.630 * h),
                        (sign * 0.148 * h, -0.004 * h, 0.540 * h),
                        (sign * 0.154 * h, -0.002 * h, 0.485 * h),
                        (sign * 0.158 * h, -0.002 * h, 0.405 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # d) Median nerve (cubital fossa -> carpal tunnel -> palmar hand)
        brachial_nerves.append(
            NervePolyline(
                name=f"nerve_median_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        cord_medial,
                        (sign * 0.118 * h, 0.002 * h, 0.720 * h),
                        (sign * 0.136 * h, 0.005 * h, 0.630 * h),
                        (sign * 0.146 * h, 0.004 * h, 0.540 * h),
                        (sign * 0.152 * h, 0.003 * h, 0.485 * h),
                        (sign * 0.156 * h, 0.003 * h, 0.400 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # e) Ulnar nerve (medial epicondyle -> Guyon's canal -> hypothenar hand)
        brachial_nerves.append(
            NervePolyline(
                name=f"nerve_ulnar_{s_char}",
                category="brachial",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        cord_medial,
                        (sign * 0.116 * h, -0.005 * h, 0.720 * h),
                        (sign * 0.132 * h, -0.006 * h, 0.625 * h),
                        (sign * 0.142 * h, -0.004 * h, 0.540 * h),
                        (sign * 0.148 * h, 0.000, 0.485 * h),
                        (sign * 0.150 * h, 0.000, 0.400 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

    # -------------------------------------------------------------------------
    # 5. Intercostal nerves (T1 - T11 pairs wrapping around ribcage)
    # -------------------------------------------------------------------------
    intercostal_nerves: list[NervePolyline] = []

    for t_idx in range(1, 12):
        t_name = f"T{t_idx}"
        z_frac = 0.735 - (t_idx - 1) * 0.0165
        z_level = z_frac * h
        cord_y = _get_cord_y(z_level)

        for s_char, sign in [("r", 1.0), ("l", -1.0)]:
            pts = _catmull_rom_resample(
                [
                    (sign * 0.024 * h, cord_y, z_level),
                    (sign * 0.075 * h, -0.015 * h, z_level - 0.004 * h),
                    (sign * 0.100 * h, 0.000, z_level - 0.008 * h),
                    (sign * 0.075 * h, 0.045 * h, z_level - 0.012 * h),
                    (sign * 0.025 * h, 0.065 * h, z_level - 0.015 * h),
                ],
                step_mm=step_mm,
            )
            intercostal_nerves.append(
                NervePolyline(
                    name=f"nerve_intercostal_{t_name}_{s_char}",
                    category="intercostal",
                    side=s_char,  # type: ignore[arg-type]
                    points=pts,
                )
            )

    # -------------------------------------------------------------------------
    # 6. Lumbosacral Plexus (femoral, obturator, saphenous, sciatic -> tibial & fibular)
    # -------------------------------------------------------------------------
    lumbosacral_nerves: list[NervePolyline] = []

    for s_char, sign in [("r", 1.0), ("l", -1.0)]:
        # a) Lateral femoral cutaneous nerve
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_lat_femoral_cutaneous_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        root_foramen_pos["L2"][s_char],
                        (sign * 0.055 * h, -0.005 * h, 0.535 * h),
                        (sign * 0.068 * h, 0.005 * h, 0.460 * h),
                        (sign * 0.065 * h, 0.005 * h, 0.380 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # b) Femoral nerve
        femoral_pts = _catmull_rom_resample(
            [
                root_foramen_pos["L3"][s_char],
                (sign * 0.040 * h, 0.012 * h, 0.515 * h),
                (sign * 0.046 * h, 0.012 * h, 0.440 * h),
                (sign * 0.052 * h, 0.010 * h, 0.350 * h),
                (sign * 0.056 * h, 0.008 * h, 0.290 * h),
            ],
            step_mm=step_mm,
        )
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_femoral_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=femoral_pts,
            )
        )

        # c) Saphenous nerve (continuation of femoral down medial shank)
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_saphenous_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        (sign * 0.056 * h, 0.008 * h, 0.290 * h),
                        (sign * 0.050 * h, 0.004 * h, 0.230 * h),
                        (sign * 0.053 * h, 0.004 * h, 0.140 * h),
                        (sign * 0.057 * h, 0.004 * h, 0.050 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # d) Obturator nerve
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_obturator_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        root_foramen_pos["L3"][s_char],
                        (sign * 0.026 * h, 0.004 * h, 0.505 * h),
                        (sign * 0.034 * h, 0.000, 0.440 * h),
                        (sign * 0.040 * h, 0.000, 0.350 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # e) Sciatic nerve (posterior thigh down to bifurcation above knee)
        sciatic_bifurcation = (sign * 0.056 * h, -0.010 * h, 0.320 * h)
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_sciatic_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        root_foramen_pos["S1"][s_char],
                        (sign * 0.038 * h, -0.020 * h, 0.490 * h),
                        (sign * 0.048 * h, -0.015 * h, 0.430 * h),
                        sciatic_bifurcation,
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # f) Tibial nerve (popliteal fossa -> posterior calf -> plantar foot)
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_tibial_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        sciatic_bifurcation,
                        (sign * 0.058 * h, -0.008 * h, 0.280 * h),
                        (sign * 0.060 * h, -0.010 * h, 0.180 * h),
                        (sign * 0.058 * h, -0.004 * h, 0.050 * h),
                        (sign * 0.060 * h, 0.015 * h, 0.018 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # g) Common fibular (peroneal) nerve -> deep & superficial branches
        fibular_neck = (sign * 0.068 * h, 0.000, 0.265 * h)
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_common_fibular_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        sciatic_bifurcation,
                        (sign * 0.064 * h, -0.006 * h, 0.290 * h),
                        fibular_neck,
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # Deep fibular (anterior compartment -> dorsum of foot)
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_deep_fibular_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        fibular_neck,
                        (sign * 0.063 * h, 0.010 * h, 0.180 * h),
                        (sign * 0.064 * h, 0.012 * h, 0.050 * h),
                        (sign * 0.065 * h, 0.025 * h, 0.020 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

        # Superficial fibular (lateral compartment)
        lumbosacral_nerves.append(
            NervePolyline(
                name=f"nerve_superficial_fibular_{s_char}",
                category="lumbosacral",
                side=s_char,  # type: ignore[arg-type]
                points=_catmull_rom_resample(
                    [
                        fibular_neck,
                        (sign * 0.068 * h, 0.004 * h, 0.170 * h),
                        (sign * 0.067 * h, 0.008 * h, 0.050 * h),
                    ],
                    step_mm=step_mm,
                ),
            )
        )

    # -------------------------------------------------------------------------
    # 7. Cranial nerves: Vagus (CN X)
    # -------------------------------------------------------------------------
    cranial_nerves: list[NervePolyline] = []

    for s_char, sign in [("r", 1.0), ("l", -1.0)]:
        vagus_pts = _catmull_rom_resample(
            [
                (sign * 0.012 * h, -0.012 * h, 0.880 * h),  # Jugular foramen
                (sign * 0.018 * h, 0.003 * h, 0.810 * h),   # Carotid sheath in neck
                (sign * 0.015 * h, 0.004 * h, 0.730 * h),   # Superior mediastinum
                (sign * 0.012 * h, -0.008 * h, 0.640 * h),  # Cardiac / esophageal plexus
            ],
            step_mm=step_mm,
        )
        cranial_nerves.append(
            NervePolyline(
                name=f"nerve_vagus_{s_char}",
                category="cranial",
                side=s_char,  # type: ignore[arg-type]
                points=vagus_pts,
            )
        )

    # Calculate total vertex count
    all_polylines = (
        [spinal_cord]
        + [p.points for p in spinal_roots]
        + [p.points for p in brachial_nerves]
        + [p.points for p in intercostal_nerves]
        + [p.points for p in lumbosacral_nerves]
        + [p.points for p in cranial_nerves]
    )
    total_vertices = sum(len(pts) for pts in all_polylines)

    return NerveNetworkResponse(
        detail=detail,
        total_vertex_count=total_vertices,
        spinal_cord=spinal_cord,
        brain_structures=brain_structs,
        spinal_roots=spinal_roots,
        brachial_plexus=brachial_nerves,
        intercostal_nerves=intercostal_nerves,
        lumbosacral_plexus=lumbosacral_nerves,
        cranial_nerves=cranial_nerves,
    )
