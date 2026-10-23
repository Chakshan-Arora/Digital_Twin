"""Unit tests for domain-agnostic core data models (Structure and Case)."""

import numpy as np
import pytest

from hdt.core.case import Case
from hdt.core.structure import Structure


def test_structure_valid_creation() -> None:
    """Test valid instantiation of Structure for target, risk, and context."""
    mask = np.zeros((10, 20, 30), dtype=bool)
    mask[2:5, 3:7, 4:8] = True

    s_target = Structure(name="tumor", mask=mask, role="target")
    assert s_target.name == "tumor"
    assert s_target.role == "target"
    assert s_target.mask.shape == (10, 20, 30)
    assert s_target.mask.dtype == bool

    s_risk = Structure(name="optic_chiasm", mask=mask, role="risk")
    assert s_risk.role == "risk"

    s_context = Structure(name="brainstem", mask=mask, role="context")
    assert s_context.role == "context"


def test_structure_rejects_non_bool_array() -> None:
    """Test that Structure raises ValueError when mask dtype is not bool."""
    int_mask = np.zeros((10, 10, 10), dtype=np.int32)
    with pytest.raises(ValueError, match="boolean dtype"):
        Structure(name="tumor", mask=int_mask, role="target")

    uint8_mask = np.ones((10, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="boolean dtype"):
        Structure(name="tumor", mask=uint8_mask, role="target")

    float_mask = np.zeros((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="boolean dtype"):
        Structure(name="tumor", mask=float_mask, role="target")


def test_structure_rejects_non_3d_array() -> None:
    """Test that Structure raises ValueError when mask is not 3-dimensional."""
    mask_2d = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="3-dimensional"):
        Structure(name="tumor", mask=mask_2d, role="target")

    mask_4d = np.zeros((2, 10, 10, 10), dtype=bool)
    with pytest.raises(ValueError, match="3-dimensional"):
        Structure(name="tumor", mask=mask_4d, role="target")


def test_structure_rejects_invalid_inputs() -> None:
    """Test that Structure raises ValueError for invalid names, roles, or mask types."""
    mask = np.zeros((10, 10, 10), dtype=bool)

    with pytest.raises(ValueError, match="numpy.ndarray"):
        Structure(name="tumor", mask=[[[False]]], role="target")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty string"):
        Structure(name="   ", mask=mask, role="target")

    with pytest.raises(ValueError, match="role must be one of"):
        Structure(name="tumor", mask=mask, role="invalid_role")  # type: ignore[arg-type]


def test_case_valid_creation() -> None:
    """Test valid Case instantiation with multiple images."""
    t1 = np.zeros((16, 32, 48), dtype=np.float32)
    t2 = np.ones((16, 32, 48), dtype=np.float32)
    spacing = (2.0, 1.0, 1.0)
    origin = (10.0, 20.0, 30.0)
    direction = np.eye(3, dtype=np.float64)

    case = Case(
        case_id="case_001",
        images={"t1": t1, "t2": t2},
        spacing_mm=spacing,
        origin_mm=origin,
        direction=direction,
    )

    assert case.case_id == "case_001"
    assert case.primary == "t1"
    assert case.shape == (16, 32, 48)
    assert case.images["t1"].shape == (16, 32, 48)
    assert case.spacing_mm == (2.0, 1.0, 1.0)
    assert case.origin_mm == (10.0, 20.0, 30.0)
    assert np.array_equal(case.direction, direction)
    assert case.names() == []


def test_case_primary_property_and_setter() -> None:
    """Test primary property default, explicit initialization, and setter."""
    img1 = np.zeros((10, 10, 10), dtype=np.float32)
    img2 = np.ones((10, 10, 10), dtype=np.float32)

    # Defaults to first image in dict
    case1 = Case(
        case_id="c1",
        images={"flair": img1, "t1ce": img2},
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )
    assert case1.primary == "flair"

    # Explicit primary
    case2 = Case(
        case_id="c2",
        images={"flair": img1, "t1ce": img2},
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
        primary="t1ce",
    )
    assert case2.primary == "t1ce"

    # Mutate primary via setter
    case2.primary = "flair"
    assert case2.primary == "flair"

    # Setting unknown primary raises KeyError
    with pytest.raises(KeyError, match="not found in images"):
        case2.primary = "nonexistent"


def test_case_rejects_invalid_images() -> None:
    """Test Case validation for empty images, wrong dtypes, shapes, and shape mismatch."""
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3, dtype=np.float64)

    # Empty images dictionary
    with pytest.raises(ValueError, match="non-empty dictionary"):
        Case("c1", {}, spacing, origin, direction)

    # Float64 instead of float32
    img_f64 = np.zeros((10, 10, 10), dtype=np.float64)
    with pytest.raises(ValueError, match="float32 dtype"):
        Case("c1", {"t1": img_f64}, spacing, origin, direction)

    # 2D instead of 3D
    img_2d = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="3-dimensional"):
        Case("c1", {"t1": img_2d}, spacing, origin, direction)

    # Non-ndarray
    with pytest.raises(ValueError, match="numpy.ndarray"):
        Case("c1", {"t1": [[1.0]]}, spacing, origin, direction)  # type: ignore[arg-type]

    # Shape mismatch across modalities
    img_a = np.zeros((10, 10, 10), dtype=np.float32)
    img_b = np.zeros((10, 10, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match reference shape"):
        Case("c1", {"t1": img_a, "t2": img_b}, spacing, origin, direction)


def test_case_rejects_shape_mismatched_structure() -> None:
    """Test that Case rejects structures whose mask shape does not match the images shape."""
    case = Case(
        case_id="case_001",
        images={"t1": np.zeros((20, 30, 40), dtype=np.float32)},
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    mismatched_mask = np.zeros((20, 30, 41), dtype=bool)
    struct = Structure(name="mismatched", mask=mismatched_mask, role="target")

    with pytest.raises(ValueError, match="mask shape .* does not match"):
        case.add_structure(struct)


def test_case_rejects_duplicate_structures() -> None:
    """Test that Case rejects adding duplicate structure names."""
    case = Case(
        case_id="case_001",
        images={"t1": np.zeros((20, 20, 20), dtype=np.float32)},
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    mask = np.zeros((20, 20, 20), dtype=bool)
    s1 = Structure(name="lesion", mask=mask, role="target")
    s2 = Structure(name="lesion", mask=mask, role="risk")

    case.add_structure(s1)
    with pytest.raises(ValueError, match="already exists"):
        case.add_structure(s2)


def test_case_get_and_names() -> None:
    """Test retrieving structures and checking registered names."""
    case = Case(
        case_id="case_001",
        images={"t1": np.zeros((15, 15, 15), dtype=np.float32)},
        spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3, dtype=np.float64),
    )

    mask = np.zeros((15, 15, 15), dtype=bool)
    target = Structure(name="tumor", mask=mask, role="target")
    risk = Structure(name="artery", mask=mask, role="risk")

    case.add_structure(target)
    case.add_structure(risk)

    assert set(case.names()) == {"tumor", "artery"}
    assert case.get("tumor").role == "target"
    assert case.get("artery").role == "risk"

    with pytest.raises(KeyError, match="not found"):
        case.get("nonexistent")
