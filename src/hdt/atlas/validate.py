"""Validation functions for the anatomical neural atlas."""

from hdt.atlas.models import (
    CENTRAL_NODES,
    SPINAL_CORD_LEVELS,
    Atlas,
    NodeName,
)

_SPINAL_CORD_ORDER = {level: i for i, level in enumerate(SPINAL_CORD_LEVELS)}
_CENTRAL_ORDER = {node: i for i, node in enumerate(CENTRAL_NODES)}


def validate_atlas(atlas: Atlas) -> list[str]:
    """Validate an Atlas instance against anatomical consistency and schema invariants.

    Parameters
    ----------
    atlas : Atlas
        The complete neural atlas data model to inspect.

    Returns
    -------
    list[str]
        A list of human-readable problem descriptions. An empty list signifies
        a completely valid atlas.
    """
    problems: list[str] = []

    # 1. Unique dermatome IDs
    seen_dermatome_ids: set[str] = set()
    dermatome_roots: set[str] = set()
    for d in atlas.dermatomes:
        if d.id in seen_dermatome_ids:
            problems.append(f"Duplicate dermatome id: '{d.id}'.")
        seen_dermatome_ids.add(d.id)
        dermatome_roots.add(d.root)

    # 2. Path nodes validation & ascending order check
    for d in atlas.dermatomes:
        if not d.path:
            problems.append(f"Dermatome '{d.id}' has an empty path.")
            continue

        # Check path ends at cortex_S1
        last_node = d.path[-1].value if isinstance(d.path[-1], NodeName) else str(d.path[-1])
        if last_node != NodeName.CORTEX_S1.value:
            problems.append(
                f"Dermatome '{d.id}' path must end at '{NodeName.CORTEX_S1.value}'; "
                f"got '{last_node}'."
            )

        # Check root spinal cord level presence
        expected_cord_level = (
            d.root if d.root.startswith("cord_") else f"cord_{d.root}"
        )
        path_str_nodes = [
            node.value if isinstance(node, NodeName) else str(node) for node in d.path
        ]

        if expected_cord_level not in path_str_nodes:
            problems.append(
                f"Dermatome '{d.id}' path does not pass through matching root level "
                f"'{expected_cord_level}'."
            )

        # Check spinal cord levels ascend in caudal-to-rostral order
        cord_indices: list[int] = []
        for node_val in path_str_nodes:
            if node_val in _SPINAL_CORD_ORDER:
                cord_indices.append(_SPINAL_CORD_ORDER[node_val])

        for k in range(len(cord_indices) - 1):
            if cord_indices[k] >= cord_indices[k + 1]:
                problems.append(
                    f"Dermatome '{d.id}' spinal cord levels do not ascend in "
                    f"caudal-to-rostral order: level index {cord_indices[k]} >= "
                    f"{cord_indices[k + 1]}."
                )
                break

        # Check central nodes order if present
        central_indices: list[int] = []
        for node_val in path_str_nodes:
            if node_val in _CENTRAL_ORDER:
                central_indices.append(_CENTRAL_ORDER[node_val])

        for k in range(len(central_indices) - 1):
            if central_indices[k] >= central_indices[k + 1]:
                problems.append(
                    f"Dermatome '{d.id}' central relay nodes do not proceed in forward order: "
                    f"index {central_indices[k]} >= {central_indices[k + 1]}."
                )
                break

    # 3. Conditions referential integrity
    seen_condition_ids: set[str] = set()
    for cond in atlas.conditions:
        if cond.id in seen_condition_ids:
            problems.append(f"Duplicate condition id: '{cond.id}'.")
        seen_condition_ids.add(cond.id)

        for aff_id in cond.affected_dermatomes:
            if aff_id not in seen_dermatome_ids:
                problems.append(
                    f"Condition '{cond.id}' references non-existent dermatome '{aff_id}'."
                )

        for comp_root in cond.compressed_roots:
            # Match either the root field or the dermatome id
            if comp_root not in dermatome_roots and comp_root not in seen_dermatome_ids:
                problems.append(
                    f"Condition '{cond.id}' references non-existent compressed root '{comp_root}'."
                )

    # 4. Torso y_fractions bounds and non-overlapping check
    torso_bands: list[tuple[float, float, str]] = []
    for d in atlas.dermatomes:
        if d.body_zone.kind == "torso_band":
            if d.body_zone.y_fraction is None:
                problems.append(
                    f"Torso band dermatome '{d.id}' is missing required y_fraction."
                )
                continue

            y_min, y_max = d.body_zone.y_fraction
            if not (0.0 <= y_min < y_max <= 1.0):
                problems.append(
                    f"Torso band dermatome '{d.id}' has invalid y_fraction [{y_min}, {y_max}]: "
                    "must satisfy 0.0 <= y_min < y_max <= 1.0."
                )
            else:
                torso_bands.append((y_min, y_max, d.id))
        else:
            if d.body_zone.y_fraction is not None:
                problems.append(
                    f"Non-torso dermatome '{d.id}' (kind='{d.body_zone.kind}') "
                    "must not define y_fraction."
                )

    # Check for overlapping torso bands
    torso_bands.sort(key=lambda b: b[0])
    for i in range(len(torso_bands) - 1):
        curr_min, curr_max, curr_id = torso_bands[i]
        next_min, next_max, next_id = torso_bands[i + 1]
        if curr_max > next_min:
            problems.append(
                f"Torso bands overlap: '{curr_id}' [{curr_min:.3f}, {curr_max:.3f}] "
                f"overlaps with '{next_id}' [{next_min:.3f}, {next_max:.3f}]."
            )

    return problems
