"""Condition simulation engine for neuro-anatomical signal loss and latency degradation.

Models electrophysiological conduction slowing, fiber drop-out, and complete
pathway blockades under clinical radiculopathies and entrapment neuropathies.

Research prototype - NOT a clinical diagnostic tool.
Conduction velocity and latency calculations are educational approximations.
"""

from __future__ import annotations

import math
from typing import Literal

from hdt.atlas.data import get_default_atlas
from hdt.atlas.models import (
    SPINAL_CORD_LEVELS,
    Condition,
    FiberType,
)
from hdt.body.models import BodyParams
from hdt.conditions.lesions import locate_lesions
from hdt.conditions.models import (
    ConditionResult,
    DermatomeEffect,
    FiberEffect,
    Severity,
    SignalStatus,
)
from hdt.nervous.signal import signal_timeline

# ---------------------------------------------------------------------------
# Constants Table: Conduction Slowing and Blocking Factors per Severity
# ---------------------------------------------------------------------------
# Values represent educational simulation parameters, not physiological claims.
# - Factor > 1.0 indicates conduction velocity degradation: v_affected = v_normal / factor
#   and latency_affected = latency_normal * factor.
# - Factor is None indicates total conduction failure (blockade; latency is null).
FIBER_SEVERITY_FACTORS: dict[Severity, dict[FiberType, float | None]] = {
    Severity.MILD: {
        # Mild compression: large myelinated A-beta and A-delta fibers exhibit focal demyelinating
        # conduction slowing; unmyelinated C fibers remain physiologically resistant and unaffected.
        FiberType.A_BETA: 1.40,
        FiberType.A_DELTA: 1.30,
        FiberType.C: 1.00,
    },
    Severity.MODERATE: {
        # Moderate compression: worsened slowing across all fibers with severe A-beta disruption.
        FiberType.A_BETA: 2.20,
        FiberType.A_DELTA: 1.80,
        FiberType.C: 1.40,
    },
    Severity.SEVERE: {
        # Severe compression / transection: complete axonal conduction blockade.
        FiberType.A_BETA: None,
        FiberType.A_DELTA: None,
        FiberType.C: None,
    },
}


def simulate_condition(
    body: BodyParams,
    condition: Condition | str,
    side: Literal["l", "r", "both"] = "r",
    severity: Severity = Severity.MODERATE,
) -> ConditionResult:
    """Simulate neural conduction alterations and signal loss for a clinical condition.

    Parameters
    ----------
    body : BodyParams
        Subject skeletal model in RAS mm.
    condition : Condition or str
        Condition instance or identifier from atlas (e.g. 'sciatica', 'carpal_tunnel').
    side : {'l', 'r', 'both'}
        Anatomical stimulation / lesion lateralisation (default 'r').
    severity : Severity
        Clinical severity grading ('mild', 'moderate', 'severe').

    Returns
    -------
    ConditionResult
        Detailed simulation output including affected dermatomes, per-fiber latencies,
        nerve effects, and 3D lesion sites.

    Notes
    -----
    Educational simulation model. All slowing factors and blockade rules reflect
    simplified illustrative heuristics for anatomical learning, not patient prognosis.
    """
    atlas = get_default_atlas()

    if isinstance(condition, str):
        cond_obj = next((c for c in atlas.conditions if c.id == condition), None)
        if cond_obj is None:
            raise ValueError(f"Unknown condition '{condition}'")
        condition = cond_obj

    # 1. Determine affected dermatome identifiers and sides
    cond_id = condition.id
    target_sides: list[Literal["l", "r"]] = (
        ["l", "r"] if side == "both" or cond_id == "sci_t10" else [side]
    )

    # Cord levels caudal to T10 in caudal-to-rostral order
    t10_idx = SPINAL_CORD_LEVELS.index("cord_T10")
    caudal_to_t10_roots = {lvl.replace("cord_", "") for lvl in SPINAL_CORD_LEVELS[: t10_idx + 1]}

    affected_dermatome_effects: list[DermatomeEffect] = []
    affected_d_ids: set[str] = set()

    for d in atlas.dermatomes:
        d_id = d.id
        root_id = d.root.upper()

        is_affected = False

        if cond_id == "sci_t10":
            # Complete cord injury at T10: every dermatome caudal to T10 on both sides
            if root_id in caudal_to_t10_roots:
                is_affected = True
        else:
            # Unilateral radiculopathy / peripheral entrapment
            if root_id in condition.compressed_roots or d_id in condition.affected_dermatomes:
                is_affected = True

        if not is_affected:
            continue

        affected_d_ids.add(d_id)

        # For cord transection, both sides are always affected; otherwise target_sides
        d_sides: list[Literal["l", "r"]] = ["l", "r"] if cond_id == "sci_t10" else target_sides

        for s in d_sides:
            # Baseline conduction timeline
            base_timeline = signal_timeline(body, d_id, side=s)
            fibers_effect: dict[FiberType, FiberEffect] = {}

            dermatome_status = SignalStatus.NORMAL

            for f_type in [FiberType.A_BETA, FiberType.A_DELTA, FiberType.C]:
                base_item = base_timeline.fibers[f_type]
                norm_lat = base_item.latency_ms
                norm_vel = base_item.conduction_velocity_mps

                if cond_id == "sci_t10" or severity == Severity.SEVERE:
                    # Complete blockade
                    f_status = SignalStatus.BLOCKED
                    aff_lat = None
                    aff_vel = None
                    dermatome_status = SignalStatus.BLOCKED
                else:
                    factor = FIBER_SEVERITY_FACTORS[severity][f_type]
                    if factor is None:
                        f_status = SignalStatus.BLOCKED
                        aff_lat = None
                        aff_vel = None
                        dermatome_status = SignalStatus.BLOCKED
                    elif math.isclose(factor, 1.0, rel_tol=1e-4):
                        f_status = SignalStatus.NORMAL
                        aff_lat = norm_lat
                        aff_vel = norm_vel
                    else:
                        f_status = SignalStatus.SLOWED
                        aff_lat = round(norm_lat * factor, 2)
                        aff_vel = round(norm_vel / factor, 2)
                        if dermatome_status != SignalStatus.BLOCKED:
                            dermatome_status = SignalStatus.SLOWED

                fibers_effect[f_type] = FiberEffect(
                    fiber_type=f_type,
                    normal_latency_ms=norm_lat,
                    affected_latency_ms=aff_lat,
                    conduction_velocity_mps=norm_vel,
                    affected_velocity_mps=aff_vel,
                    status=f_status,
                )

            affected_dermatome_effects.append(
                DermatomeEffect(
                    dermatome_id=d_id,
                    side=s,
                    region=d.region,
                    status=dermatome_status,
                    fibers=fibers_effect,
                )
            )

    # 2. Locate focal 3D lesions and affected downstream nerves on the network
    lesions, nerve_effects = locate_lesions(body, condition, side=side)

    # Update nerve effect statuses based on condition severity
    for ne in nerve_effects:
        if cond_id == "sci_t10" or severity == Severity.SEVERE:
            ne.status = SignalStatus.BLOCKED
        else:
            ne.status = SignalStatus.SLOWED

    # 3. Compile unaffected dermatomes
    all_d_ids = [d.id for d in atlas.dermatomes]
    unaffected_dermatomes = [did for did in all_d_ids if did not in affected_d_ids]

    return ConditionResult(
        condition_id=condition.id,
        name=condition.name,
        description=condition.description,
        symptoms=condition.symptoms,
        compressed_roots=condition.compressed_roots,
        dermatome_effects=affected_dermatome_effects,
        nerve_effects=nerve_effects,
        lesions=lesions,
        unaffected_dermatomes=unaffected_dermatomes,
        disclaimer=atlas.disclaimer,
    )
