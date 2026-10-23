"""Unit tests for BodyParams models, validator, and anthropometric presets."""

import copy
import math

from hdt.body.models import JointName, validate_body_params
from hdt.body.presets import PRESETS, get_preset, make_preset


def test_presets_are_valid():
    """Verify that all predefined presets ('short', 'average', 'tall') pass validation."""
    for name in ["short", "average", "tall"]:
        preset = get_preset(name)
        assert preset is not None
        errors = validate_body_params(preset)
        assert errors == [], f"Preset '{name}' failed validation: {errors}"


def test_presets_ordered_by_height():
    """Verify presets have strictly increasing heights and scale linearly."""
    short = PRESETS["short"]
    avg = PRESETS["average"]
    tall = PRESETS["tall"]

    assert short.height_mm < avg.height_mm < tall.height_mm
    assert math.isclose(short.height_mm, 1500.0)
    assert math.isclose(avg.height_mm, 1750.0)
    assert math.isclose(tall.height_mm, 1950.0)

    # Linear scaling of limb lengths
    l_knee_z_short = short.joints[JointName.L_KNEE][2]
    l_knee_z_avg = avg.joints[JointName.L_KNEE][2]
    ratio_height = avg.height_mm / short.height_mm
    ratio_knee = l_knee_z_avg / l_knee_z_short
    assert math.isclose(ratio_height, ratio_knee, rel_tol=1e-3)


def test_validator_rejects_missing_joints():
    """Reject bodies missing required joint definitions."""
    body = copy.deepcopy(PRESETS["average"])
    del body.joints[JointName.L_KNEE]
    errors = validate_body_params(body)
    assert any("Missing required joints" in e and "l_knee" in e for e in errors)


def test_validator_rejects_non_finite_coordinates():
    """Reject bodies with NaN or Inf coordinates."""
    body = copy.deepcopy(PRESETS["average"])
    body.joints[JointName.R_SHOULDER] = (float("nan"), 0.0, 1400.0)
    errors = validate_body_params(body)
    assert any("Non-finite coordinates" in e and "r_shoulder" in e for e in errors)


def test_validator_rejects_stature_out_of_bounds():
    """Reject heights outside [1000, 2300] mm."""
    too_short = make_preset(800.0)
    errs_short = validate_body_params(too_short)
    assert any("outside plausible range" in e for e in errs_short)

    too_tall = make_preset(2500.0)
    errs_tall = validate_body_params(too_tall)
    assert any("outside plausible range" in e for e in errs_tall)


def test_validator_rejects_head_not_highest():
    """Reject bodies where head is lower than another joint."""
    body = copy.deepcopy(PRESETS["average"])
    # Move neck above head
    neck_pt = body.joints[JointName.NECK]
    body.joints[JointName.NECK] = (neck_pt[0], neck_pt[1], 2000.0)
    errors = validate_body_params(body)
    assert any("Head" in e and "not highest joint" in e for e in errors)


def test_validator_rejects_feet_not_lowest():
    """Reject bodies where a non-foot joint is lower than the feet."""
    body = copy.deepcopy(PRESETS["average"])
    knee_pt = body.joints[JointName.L_KNEE]
    body.joints[JointName.L_KNEE] = (knee_pt[0], knee_pt[1], 5.0)  # below foot at 26 mm
    errors = validate_body_params(body)
    assert any("Feet are not lowest joints" in e and "l_knee" in e for e in errors)


def test_validator_rejects_wrong_lateral_sides():
    """Reject bodies where left joints have positive x or right joints have negative x."""
    body = copy.deepcopy(PRESETS["average"])
    # Put left shoulder on +x
    sh_pt = body.joints[JointName.L_SHOULDER]
    body.joints[JointName.L_SHOULDER] = (150.0, sh_pt[1], sh_pt[2])
    errors = validate_body_params(body)
    assert any("l_shoulder must have negative x" in e for e in errors)


def test_validator_rejects_implausible_bone_lengths():
    """Reject upper leg or upper arm lengths outside anatomical limits."""
    body = copy.deepcopy(PRESETS["average"])
    # Distort hip to knee distance to 750 mm (limit is 650 mm)
    body.joints[JointName.L_KNEE] = (-100.0, 0.0, 100.0)  # hip is at ~927 mm
    errors = validate_body_params(body)
    assert any("Left upper leg length" in e and "outside plausible range" in e for e in errors)
