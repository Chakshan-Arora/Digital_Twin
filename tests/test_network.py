"""Unit tests for procedural nervous system 3D network generation."""

import time

from hdt.body.presets import PRESETS
from hdt.body.shape import surface_distance_mm
from hdt.nervous.network import build_network


def test_network_structure_completeness():
    """Verify generated network contains all required anatomical subsystems."""
    body = PRESETS["average"]
    net = build_network(body, detail="high")

    # Spinal cord present and has vertices
    assert len(net.spinal_cord) > 50

    # 31 bilateral pairs of spinal roots = 62 root polylines
    assert len(net.spinal_roots) == 62

    # Brain structures present
    brain_names = {b.name for b in net.brain_structures}
    expected_brains = {
        "cerebrum",
        "cerebellum",
        "brainstem",
        "thalamus",
        "cortex_S1_r",
        "cortex_S1_l",
    }
    assert expected_brains.issubset(brain_names)

    # Brachial plexus
    brachial_names = {p.name for p in net.brachial_plexus}
    for branch in ["axillary", "musculocutaneous", "radial", "median", "ulnar"]:
        assert f"nerve_{branch}_r" in brachial_names
        assert f"nerve_{branch}_l" in brachial_names

    # Intercostal nerves (T1 - T11 bilateral = 22 polylines)
    assert len(net.intercostal_nerves) == 22

    # Lumbosacral plexus (femoral, saphenous, obturator, sciatic, tibial, fibular)
    lumbo_names = {p.name for p in net.lumbosacral_plexus}
    for branch in ["femoral", "saphenous", "obturator", "sciatic", "tibial", "common_fibular"]:
        assert f"nerve_{branch}_r" in lumbo_names
        assert f"nerve_{branch}_l" in lumbo_names

    # Cranial nerves (vagus bilateral)
    cranial_names = {p.name for p in net.cranial_nerves}
    assert "nerve_vagus_r" in cranial_names
    assert "nerve_vagus_l" in cranial_names


def test_detail_resolution_scaling():
    """Verify detail='low' generates fewer vertices than detail='high'."""
    body = PRESETS["average"]
    net_high = build_network(body, detail="high")
    net_low = build_network(body, detail="low")

    assert net_high.total_vertex_count > net_low.total_vertex_count
    assert len(net_high.spinal_cord) > len(net_low.spinal_cord)


def test_all_vertices_inside_body_envelope():
    """Verify every nerve vertex is inside the body envelope or within 5mm tolerance."""
    for preset_name in ["short", "average", "tall"]:
        body = PRESETS[preset_name]
        net = build_network(body, detail="high")

        all_polys = [net.spinal_cord]
        for poly in (
            net.spinal_roots
            + net.brachial_plexus
            + net.intercostal_nerves
            + net.lumbosacral_plexus
            + net.cranial_nerves
        ):
            all_polys.append(poly.points)

        for poly_pts in all_polys:
            for pt in poly_pts:
                dist = surface_distance_mm(body, pt)
                # Must be strictly within body envelope (< 5.0 mm tolerance for skin branches)
                assert dist <= 5.0, f"Vertex {pt} in {preset_name} exceeds envelope: {dist} mm"


def test_network_execution_performance():
    """Verify build_network runs in well under 200 ms."""
    body = PRESETS["average"]

    # Warmup
    build_network(body, detail="high")

    t0 = time.perf_counter()
    net = build_network(body, detail="high")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 200.0, f"Execution too slow: {elapsed_ms:.1f} ms"
    assert net.total_vertex_count > 1000
