"""Unit tests for VS-SEG contour parsing and validation rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdt.domains.vs_seg.contours import (
    extract_cochlea_contour,
    extract_tumour_contour,
    find_native_folder,
    group_rings_by_plane,
)
from hdt.imaging.load_scan import ImageGeometry


def test_native_folder_rule(tmp_path: Path) -> None:
    """Test that native folder rule picks the folder where original == reference."""
    t1_dir = tmp_path / "vs_gk_1_t1"
    t2_dir = tmp_path / "vs_gk_1_t2"
    t1_dir.mkdir()
    t2_dir.mkdir()

    # In T1 folder, tumour has original == reference
    t1_data = [
        {
            "structure_name": "TV",
            "original_contour_created_on_image_name": "MR T1 Gd",
            "reference_geometry_image_name": "MR T1 Gd",
            "LPS_contour_points": [],
        }
    ]
    with open(t1_dir / "contours.json", "w") as f:
        json.dump(t1_data, f)

    # In T2 folder, tumour has original != reference
    t2_data = [
        {
            "structure_name": "TV",
            "original_contour_created_on_image_name": "MR T1 Gd",
            "reference_geometry_image_name": "MR T2 CISS",
            "LPS_contour_points": [],
        }
    ]
    with open(t2_dir / "contours.json", "w") as f:
        json.dump(t2_data, f)

    folder, item = find_native_folder([t1_dir, t2_dir], lambda items: items[0])
    assert folder == t1_dir
    assert item["structure_name"] == "TV"


def test_native_folder_rule_failures(tmp_path: Path) -> None:
    """Test native folder rule raises ValueError if 0 or >1 matches."""
    f1 = tmp_path / "f1"
    f2 = tmp_path / "f2"
    f1.mkdir()
    f2.mkdir()

    # Zero matches
    with open(f1 / "contours.json", "w") as fp:
        json.dump(
            [{"original_contour_created_on_image_name": "A", "reference_geometry_image_name": "B"}],
            fp,
        )
    with open(f2 / "contours.json", "w") as fp:
        json.dump(
            [{"original_contour_created_on_image_name": "A", "reference_geometry_image_name": "C"}],
            fp,
        )

    with pytest.raises(ValueError, match="no folder where original == reference"):
        find_native_folder([f1, f2], lambda items: items[0])

    # Multiple matches
    with open(f2 / "contours.json", "w") as fp:
        json.dump(
            [{"original_contour_created_on_image_name": "A", "reference_geometry_image_name": "A"}],
            fp,
        )
    with open(f1 / "contours.json", "w") as fp:
        json.dump(
            [{"original_contour_created_on_image_name": "A", "reference_geometry_image_name": "A"}],
            fp,
        )

    with pytest.raises(ValueError, match="multiple folders matched"):
        find_native_folder([f1, f2], lambda items: items[0])


def test_cochlea_selection(tmp_path: Path) -> None:
    """Test cochlea exact matching and failure cases."""
    geom = ImageGeometry(
        shape=(10, 10, 10),
        spacing_mm=(1.5, 0.5, 0.5),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3),
    )

    t2_dir = tmp_path / "vs_gk_1_t2"
    t2_dir.mkdir()

    # Valid cochlea
    ring = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    data = [
        {
            "structure_name": "Cochlea",
            "original_contour_created_on_image_name": "T2",
            "reference_geometry_image_name": "T2",
            "LPS_contour_points": [ring],
        }
    ]
    with open(t2_dir / "contours.json", "w") as fp:
        json.dump(data, fp)

    sc = extract_cochlea_contour([t2_dir], geom)
    assert sc.structure_name == "Cochlea"
    assert len(sc.planes) == 1

    # Missing cochlea
    with open(t2_dir / "contours.json", "w") as fp:
        json.dump([{"structure_name": "Brainstem"}], fp)
    with pytest.raises(ValueError, match="No cochlea structure found"):
        extract_cochlea_contour([t2_dir], geom)

    # Multiple exact matches
    with open(t2_dir / "contours.json", "w") as fp:
        json.dump(
            [
                {
                    "structure_name": "Cochlea",
                    "original_contour_created_on_image_name": "T2",
                    "reference_geometry_image_name": "T2",
                    "LPS_contour_points": [ring],
                },
                {
                    "structure_name": "cochlea",
                    "original_contour_created_on_image_name": "T2",
                    "reference_geometry_image_name": "T2",
                    "LPS_contour_points": [ring],
                },
            ],
            fp,
        )
    with pytest.raises(ValueError, match="Multiple items with exact name 'cochlea'"):
        extract_cochlea_contour([t2_dir], geom)


def test_tumour_warning_on_nonstandard_name(tmp_path: Path) -> None:
    """Test warning when tumour name is not in standard set."""
    geom = ImageGeometry(
        shape=(10, 10, 10),
        spacing_mm=(1.5, 0.5, 0.5),
        origin_mm=(0.0, 0.0, 0.0),
        direction=np.eye(3),
    )
    t1_dir = tmp_path / "t1"
    t1_dir.mkdir()
    ring = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    data = [
        {
            "structure_name": "Men",  # Non-standard
            "original_contour_created_on_image_name": "T1",
            "reference_geometry_image_name": "T1",
            "LPS_contour_points": [ring],
        }
    ]
    with open(t1_dir / "contours.json", "w") as fp:
        json.dump(data, fp)

    with pytest.warns(UserWarning, match="Tumour structure name 'Men' is not in standard set"):
        sc = extract_tumour_contour([t1_dir], geom)
    assert sc.is_standard_tumour is False


def test_ring_off_plane_raises() -> None:
    """Test that a ring whose vertices do not share a normal coordinate raises ValueError."""
    normal = np.array([0.0, 0.0, 1.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0, 0.0])

    # Ring with 0.05 mm spread along normal (> 0.01 mm tolerance)
    bad_ring = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.05],
    ]
    with pytest.raises(ValueError, match="do not share a single plane coordinate"):
        group_rings_by_plane(
            rings=[bad_ring],
            normal=normal,
            in_plane_u=u_axis,
            in_plane_v=v_axis,
            slice_spacing_mm=1.5,
        )


def test_unsorted_rings_and_missing_interior_plane() -> None:
    """Test that unsorted ring input groups properly and missing interior planes are reported."""
    normal = np.array([0.0, 0.0, 1.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0, 0.0])

    r0 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    r1 = [[0.0, 0.0, 1.5], [1.0, 0.0, 1.5], [0.0, 1.0, 1.5]]
    # Skip z = 3.0 (missing interior plane)
    r2 = [[0.0, 0.0, 4.5], [1.0, 0.0, 4.5], [0.0, 1.0, 4.5]]

    # Pass in shuffled order
    shuffled_rings = [r2, r0, r1]
    planes, missing = group_rings_by_plane(
        rings=shuffled_rings,
        normal=normal,
        in_plane_u=u_axis,
        in_plane_v=v_axis,
        slice_spacing_mm=1.5,
    )

    assert len(planes) == 3
    assert [p.coordinate for p in planes] == [0.0, 1.5, 4.5]
    assert len(missing) == 1
    assert abs(missing[0] - 3.0) < 1e-4
