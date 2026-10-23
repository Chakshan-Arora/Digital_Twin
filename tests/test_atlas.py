"""Unit tests for the anatomical neural atlas data and validation rules."""

import copy

from hdt.atlas.data import get_default_atlas
from hdt.atlas.models import (
    FIBER_PROPERTIES_TABLE,
    Condition,
    FiberType,
    NodeName,
)
from hdt.atlas.validate import validate_atlas


def test_default_atlas_is_valid():
    """Verify that the shipped canonical atlas passes all validation rules without error."""
    atlas = get_default_atlas()
    problems = validate_atlas(atlas)
    assert problems == []
    assert len(atlas.dermatomes) == 29
    assert len(atlas.conditions) == 6


def test_fiber_properties_table():
    """Verify default velocities and plausibility ranges for primary fiber types."""
    assert FiberType.A_BETA in FIBER_PROPERTIES_TABLE
    assert FiberType.A_DELTA in FIBER_PROPERTIES_TABLE
    assert FiberType.C in FIBER_PROPERTIES_TABLE

    ab = FIBER_PROPERTIES_TABLE[FiberType.A_BETA]
    assert ab.default_velocity_mps == 50.0
    assert ab.velocity_range_mps == (30.0, 70.0)

    ad = FIBER_PROPERTIES_TABLE[FiberType.A_DELTA]
    assert ad.default_velocity_mps == 15.0
    assert ad.velocity_range_mps == (5.0, 30.0)

    c = FIBER_PROPERTIES_TABLE[FiberType.C]
    assert c.default_velocity_mps == 1.0
    assert c.velocity_range_mps == (0.5, 2.0)


def test_broken_duplicate_dermatome_id():
    """Reject atlases containing duplicate dermatome identifiers."""
    atlas = copy.deepcopy(get_default_atlas())
    # Duplicate the first dermatome with the same ID
    dup = atlas.dermatomes[0].model_copy()
    atlas.dermatomes.append(dup)

    problems = validate_atlas(atlas)
    assert any("Duplicate dermatome id" in p for p in problems)


def test_broken_path_not_ending_at_cortex():
    """Reject paths that fail to terminate at cortex_S1."""
    atlas = copy.deepcopy(get_default_atlas())
    # Truncate path of L5 so it ends at thalamus
    l5 = next(d for d in atlas.dermatomes if d.id == "L5")
    l5.path = [n for n in l5.path if n != NodeName.CORTEX_S1]

    problems = validate_atlas(atlas)
    assert any("path must end at 'cortex_S1'" in p and "'L5'" in p for p in problems)


def test_broken_path_missing_root_level():
    """Reject paths that do not pass through the matching spinal cord root level."""
    atlas = copy.deepcopy(get_default_atlas())
    l5 = next(d for d in atlas.dermatomes if d.id == "L5")
    # Remove cord_L5 from path
    l5.path = [n for n in l5.path if n != NodeName.CORD_L5]

    problems = validate_atlas(atlas)
    assert any("does not pass through matching root level 'cord_L5'" in p for p in problems)


def test_broken_cord_levels_descending_order():
    """Reject paths where spinal cord levels descend or are scrambled."""
    atlas = copy.deepcopy(get_default_atlas())
    l5 = next(d for d in atlas.dermatomes if d.id == "L5")
    # Swap cord_L5 and cord_L1 so it descends
    idx_l5 = l5.path.index(NodeName.CORD_L5)
    idx_l1 = l5.path.index(NodeName.CORD_L1)
    l5.path[idx_l5], l5.path[idx_l1] = l5.path[idx_l1], l5.path[idx_l5]

    problems = validate_atlas(atlas)
    assert any("spinal cord levels do not ascend" in p and "'L5'" in p for p in problems)


def test_broken_condition_nonexistent_dermatome():
    """Reject conditions referencing non-existent dermatomes."""
    atlas = copy.deepcopy(get_default_atlas())
    atlas.conditions.append(
        Condition(
            id="fake_cond",
            name="Fake Neuropathy",
            description="Testing missing dermatome reference",
            compressed_roots=["L5"],
            affected_dermatomes=["X99"],
            symptoms=["Hypothetical symptom"],
        )
    )
    problems = validate_atlas(atlas)
    assert any("references non-existent dermatome 'X99'" in p for p in problems)


def test_broken_condition_nonexistent_root():
    """Reject conditions referencing non-existent spinal roots."""
    atlas = copy.deepcopy(get_default_atlas())
    atlas.conditions.append(
        Condition(
            id="fake_cond_root",
            name="Fake Radiculopathy",
            description="Testing missing root reference",
            compressed_roots=["Z99"],
            affected_dermatomes=["L5"],
            symptoms=["Hypothetical symptom"],
        )
    )
    problems = validate_atlas(atlas)
    assert any("references non-existent compressed root 'Z99'" in p for p in problems)


def test_broken_torso_band_invalid_y_fraction():
    """Reject torso bands with invalid y_fraction values."""
    atlas = copy.deepcopy(get_default_atlas())
    t4 = next(d for d in atlas.dermatomes if d.id == "T4")
    # Inverted range (0.80 > 0.70)
    t4.body_zone.y_fraction = (0.80, 0.70)

    problems = validate_atlas(atlas)
    assert any("has invalid y_fraction" in p and "'T4'" in p for p in problems)


def test_broken_overlapping_torso_bands():
    """Reject torso bands whose height fractions overlap."""
    atlas = copy.deepcopy(get_default_atlas())
    t4 = next(d for d in atlas.dermatomes if d.id == "T4")
    t5 = next(d for d in atlas.dermatomes if d.id == "T5")
    # Deliberately overlap T5 [0.72, 0.75] with T4 [0.74, 0.78]
    t4.body_zone.y_fraction = (0.74, 0.78)
    t5.body_zone.y_fraction = (0.72, 0.76)

    problems = validate_atlas(atlas)
    assert any("Torso bands overlap" in p for p in problems)


def test_broken_non_torso_with_y_fraction():
    """Reject non-torso dermatomes that define a y_fraction."""
    atlas = copy.deepcopy(get_default_atlas())
    c6 = next(d for d in atlas.dermatomes if d.id == "C6")
    c6.body_zone.y_fraction = (0.5, 0.6)

    problems = validate_atlas(atlas)
    assert any("must not define y_fraction" in p and "'C6'" in p for p in problems)
