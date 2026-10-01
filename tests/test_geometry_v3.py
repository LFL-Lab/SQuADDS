"""Contract tests for the experimental capacitance-operator-v3 signature."""

from __future__ import annotations

import numpy as np
import pytest

from squadds.layouts.geometry_v3 import (
    V3_BLOCK_SLICES,
    V3_DIMENSIONS,
    V3_ELECTROSTATIC_BLOCK_SIZE,
    V3_ENVIRONMENT_BLOCK_SIZE,
    V3_FIXED_STACK_DIMENSIONS,
    V3_INTERACTION_BLOCK_SIZE,
    V3_METRIC_BLOCK_SIZE,
    V3_OPERATOR_BLOCK_SIZE,
    V3_TOPOLOGY_BLOCK_SIZE,
    capacitance_operator_v3_schema,
    encode_v3,
)


def test_v3_block_slices_are_contiguous_and_cover_schema():
    ordered = ["metrics", "interactions", "operator", "topology", "electrostatic", "environment"]
    assert V3_BLOCK_SLICES[ordered[0]].start == 0
    for left, right in zip(ordered[:-1], ordered[1:]):
        assert V3_BLOCK_SLICES[left].stop == V3_BLOCK_SLICES[right].start
    assert V3_BLOCK_SLICES["environment"].stop == V3_DIMENSIONS
    assert V3_BLOCK_SLICES["fixed_stack_core"].stop == V3_FIXED_STACK_DIMENSIONS

ENVIRONMENT = {
    "relative_permittivity": 11.45,
    "substrate_thickness_um": 350.0,
    "metal_thickness_um": 0.25,
    "vacuum_height_um": 1000.0,
}


def build_capacitor(
    path,
    *,
    finger_gap=3.0,
    ground_gap=8.0,
    ground_factor=4.0,
    scale=1.0,
    shift=(0.0, 0.0),
):
    kdb = pytest.importorskip("klayout.db")
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    conductor_layer = layout.layer(1, 10)
    ground_layer = layout.layer(1, 0)
    port_layers = (layout.layer(2, 0), layout.layer(3, 0))

    def box(x0, y0, x1, y1):
        shift_x, shift_y = shift
        return kdb.Box(
            int(round((x0 * scale + shift_x) * 1000)),
            int(round((y0 * scale + shift_y) * 1000)),
            int(round((x1 * scale + shift_x) * 1000)),
            int(round((y1 * scale + shift_y) * 1000)),
        )

    length = 42.0
    height = 24.0
    half_gap = finger_gap / 2
    conductors = kdb.Region()
    conductors.insert(box(-length - half_gap, 0, -half_gap, height))
    conductors.insert(box(half_gap, 0, length + half_gap, height))

    left, bottom, right, top_y = -length - half_gap, 0, length + half_gap, height
    span = max(right - left, top_y - bottom)
    margin = ground_factor * span
    outer = kdb.Region(box(left - margin, bottom - margin, right + margin, top_y + margin))
    moat = conductors.sized(int(round(ground_gap * scale * 1000)))
    top.shapes(ground_layer).insert(outer - moat)
    top.shapes(conductor_layer).insert(conductors)
    top.shapes(port_layers[0]).insert(box(left - ground_gap, 10, left, 12))
    top.shapes(port_layers[1]).insert(box(right, 12, right + ground_gap, 14))
    layout.write(str(path))
    return path


def transformed_copy(source, destination, *, angle=0.0, mirror=False, scale=1.0, shift=(0.0, 0.0)):
    kdb = pytest.importorskip("klayout.db")
    layout = kdb.Layout()
    layout.read(str(source))
    transform = kdb.DCplxTrans(scale, angle, mirror, shift[0], shift[1])
    for cell in layout.each_cell():
        for layer_index in layout.layer_indices():
            cell.shapes(layer_index).transform(transform)
    layout.write(str(destination))
    return destination


@pytest.fixture
def baseline_path(tmp_path):
    return build_capacitor(tmp_path / "baseline.gds")


@pytest.fixture
def baseline(baseline_path):
    return encode_v3(baseline_path, environment=ENVIRONMENT)


def test_v3_schema_is_additive_fixed_and_catalogue_free():
    schema = capacitance_operator_v3_schema()
    blocks = list(schema["blocks"].values())

    assert schema["dimensions"] == V3_DIMENSIONS == 256
    assert schema["experimental"] is True
    assert schema["fitted_on_catalogue"] is False
    assert schema["simulation_results_used"] is False
    assert sum(block["dimensions"] for block in blocks) == V3_DIMENSIONS
    assert [block["offset"] for block in blocks] == sorted(block["offset"] for block in blocks)


def test_v3_is_deterministic_and_ignores_pose_bearing_design_options(baseline_path, baseline):
    repeated = encode_v3(
        baseline_path,
        {"pos_x": "9000um", "pos_y": "-3000um", "orientation": "137deg", "family_secret": 12},
        environment=ENVIRONMENT,
    )

    assert baseline.shape == (V3_DIMENSIONS,)
    assert np.array_equal(baseline, repeated)
    assert np.isfinite(baseline).all()


def test_v3_is_exact_for_grid_aligned_translation(tmp_path, baseline_path, baseline):
    translated = encode_v3(
        transformed_copy(baseline_path, tmp_path / "translated.gds", shift=(4321.0, -8765.0)),
        environment=ENVIRONMENT,
    )

    assert np.array_equal(baseline, translated)


@pytest.mark.parametrize("angle", [90.0, 180.0, 270.0])
def test_v3_is_rigid_invariant_for_right_angle_rotations(tmp_path, baseline_path, baseline, angle):
    rotated = encode_v3(
        transformed_copy(baseline_path, tmp_path / f"rotated-{angle}.gds", angle=angle),
        environment=ENVIRONMENT,
    )

    assert np.allclose(baseline, rotated, rtol=2e-4, atol=2e-4)


def test_v3_is_numerically_stable_for_arbitrary_rotation(tmp_path, baseline_path, baseline):
    rotated = encode_v3(
        transformed_copy(baseline_path, tmp_path / "rotated-37.gds", angle=37.0),
        environment=ENVIRONMENT,
    )

    relative = np.linalg.norm(baseline - rotated) / max(np.linalg.norm(baseline), 1e-12)
    assert relative < 2e-3


def test_v3_is_reflection_invariant(tmp_path, baseline_path, baseline):
    reflected = encode_v3(
        transformed_copy(baseline_path, tmp_path / "reflected.gds", mirror=True),
        environment=ENVIRONMENT,
    )

    assert np.allclose(baseline, reflected, rtol=2e-4, atol=2e-4)


def test_shape_operator_and_topology_factor_scale_from_shape(tmp_path, baseline_path, baseline):
    scaled_environment = {
        **ENVIRONMENT,
        "substrate_thickness_um": 1050.0,
        "metal_thickness_um": 0.75,
        "vacuum_height_um": 3000.0,
    }
    scaled = encode_v3(
        transformed_copy(baseline_path, tmp_path / "scaled.gds", scale=3.0),
        environment=scaled_environment,
    )
    interaction_start = V3_METRIC_BLOCK_SIZE
    interaction_stop = interaction_start + V3_INTERACTION_BLOCK_SIZE
    operator_stop = interaction_stop + V3_OPERATOR_BLOCK_SIZE
    topology_stop = operator_stop + V3_TOPOLOGY_BLOCK_SIZE

    assert np.allclose(baseline[interaction_start:topology_stop], scaled[interaction_start:topology_stop], rtol=1e-3, atol=1e-3)
    assert baseline[0] != scaled[0], "absolute scale remains explicit"
    electrostatic_start = topology_stop
    assert not np.allclose(
        baseline[electrostatic_start : electrostatic_start + V3_ELECTROSTATIC_BLOCK_SIZE],
        scaled[electrostatic_start : electrostatic_start + V3_ELECTROSTATIC_BLOCK_SIZE],
    )


def test_outer_ground_crop_is_not_a_scientific_feature(tmp_path):
    compact = encode_v3(build_capacitor(tmp_path / "compact.gds", ground_factor=3.0), environment=ENVIRONMENT)
    enormous = encode_v3(build_capacitor(tmp_path / "enormous.gds", ground_factor=30.0), environment=ENVIRONMENT)

    assert np.allclose(compact, enormous, rtol=1e-5, atol=1e-5)


def test_environment_is_explicit_and_geometry_blocks_do_not_move(baseline_path):
    silicon = encode_v3(baseline_path, environment=ENVIRONMENT)
    sapphire = encode_v3(
        baseline_path,
        environment={**ENVIRONMENT, "relative_permittivity": 9.4, "substrate_thickness_um": 500.0},
    )
    geometry_stop = V3_DIMENSIONS - V3_ELECTROSTATIC_BLOCK_SIZE - V3_ENVIRONMENT_BLOCK_SIZE

    assert np.array_equal(silicon[:geometry_stop], sapphire[:geometry_stop])
    assert not np.array_equal(silicon, sapphire)


def test_grounded_proxy_is_a_passive_maxwell_matrix(baseline_path):
    _, metadata = encode_v3(baseline_path, environment=ENVIRONMENT, return_metadata=True)
    matrix = metadata["electrostatic_proxy"]["matrix_scaled_fF"]

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(matrix.sum(axis=1), 0.0, atol=1e-9)
    assert np.all(matrix[np.triu_indices(3, 1)] <= 0.0)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-8


def test_v3_tracks_a_real_gap_change(tmp_path):
    narrow = encode_v3(build_capacitor(tmp_path / "narrow.gds", finger_gap=2.0), environment=ENVIRONMENT)
    wide = encode_v3(build_capacitor(tmp_path / "wide.gds", finger_gap=12.0), environment=ENVIRONMENT)

    assert wide[18] > narrow[18]
    interaction_start = V3_METRIC_BLOCK_SIZE
    assert not np.array_equal(
        narrow[interaction_start : interaction_start + V3_INTERACTION_BLOCK_SIZE],
        wide[interaction_start : interaction_start + V3_INTERACTION_BLOCK_SIZE],
    )
