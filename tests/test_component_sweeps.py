"""Every declared sweep family must generate GDS the v2 encoder accepts."""

from __future__ import annotations

import numpy as np
import pytest

from squadds.layouts.component_sweeps import (
    PORT_LAYERS,
    SWEEPS,
    SweepSpec,
    build_component,
    ground_clearance_um,
    port_markers,
    terminal_count,
    write_design_point,
)

pytest.importorskip("qiskit_metal")
pytest.importorskip("klayout.db")

from squadds.layouts.geometry_v2 import (  # noqa: E402
    COUPLING_BINS,
    GROUND_BINS,
    MAX_TERMINALS,
    METRIC_BLOCK_SIZE,
    METRIC_NAMES,
    TERMINAL_PAIRS,
    V2_DIMENSIONS,
    encode,
)

FAMILIES = sorted(SWEEPS)


@pytest.mark.parametrize("family", FAMILIES)
def test_first_point_generates_and_encodes(family, tmp_path):
    spec = SWEEPS[family]
    point = next(spec.points())
    destination = tmp_path / f"{family}.gds"
    row = write_design_point(spec, point, destination, name=f"c_{family}")

    assert destination.is_file()
    assert row["component_name"] == family
    assert row["port_count"] == len(spec.port_pins)
    assert row["ground_clearance_um"] > 0

    vector = encode(destination, row["design_options"])
    assert vector.shape == (V2_DIMENSIONS,)
    assert np.isfinite(vector).all()
    # terminal_count is recorded untruncated even when it exceeds MAX_TERMINALS.
    assert vector[METRIC_NAMES.index("terminal_count")] == row["terminal_count"]


@pytest.mark.parametrize("family", FAMILIES)
def test_grid_is_non_degenerate(family):
    spec = SWEEPS[family]
    assert spec.size >= 40, "a sweep too small to train on is not worth simulating"
    assert len(list(spec.points())) == spec.size


def test_transmon_pocket_populates_every_coupling_slot(tmp_path):
    """Four isolated islands must exercise all six pair slots and all four grounds.

    Every design published so far is two-terminal, so five of the six pair slots
    and two of the four ground slots have never been populated by any row in the
    catalogue.  This is the regression that proves those coordinates work.
    """
    spec = SWEEPS["TransmonPocket"]
    destination = tmp_path / "pocket.gds"
    row = write_design_point(spec, next(spec.points()), destination, name="pocket")
    assert row["terminal_count"] == 4

    vector = encode(destination, row["design_options"])
    for slot in range(len(TERMINAL_PAIRS)):
        start = METRIC_BLOCK_SIZE + slot * COUPLING_BINS
        assert np.any(vector[start : start + COUPLING_BINS] != 0), f"pair slot {slot} is empty"
    offset = METRIC_BLOCK_SIZE + len(TERMINAL_PAIRS) * COUPLING_BINS
    for index in range(MAX_TERMINALS):
        start = offset + index * GROUND_BINS
        assert np.any(vector[start : start + GROUND_BINS] != 0), f"ground slot {index} is empty"


def test_star_qubit_truncates_but_records_true_terminal_count(tmp_path):
    spec = SWEEPS["StarQubit"]
    destination = tmp_path / "star.gds"
    row = write_design_point(spec, next(spec.points()), destination, name="star")
    assert row["terminal_count"] > MAX_TERMINALS
    vector = encode(destination, row["design_options"])
    assert vector[METRIC_NAMES.index("terminal_count")] == row["terminal_count"]


def test_generation_is_deterministic(tmp_path):
    spec = SWEEPS["Cap3Interdigital"]
    point = next(spec.points())
    first = tmp_path / "a.gds"
    second = tmp_path / "b.gds"
    write_design_point(spec, point, first, name="c")
    write_design_point(spec, point, second, name="c")
    assert first.read_bytes() == second.read_bytes()


def test_port_markers_use_ordered_layers():
    spec = SWEEPS["StarQubit"]
    component = build_component(spec, next(spec.points()), name="star")
    markers = port_markers(component, spec)
    assert [marker.layer for marker in markers] == list(PORT_LAYERS[: len(markers)])
    assert terminal_count(component) >= len(markers)


def test_clearance_falls_back_to_gap_options():
    """A spec that declares no clearance path still resolves from the option tree."""
    spec = SWEEPS["CoupledLineTee"]
    generic = SweepSpec(
        component_name=spec.component_name,
        import_path=spec.import_path,
        port_pins=spec.port_pins,
        grid=spec.grid,
        base_options=spec.base_options,
    )
    component = build_component(generic, next(generic.points()), name="c")
    assert ground_clearance_um(component, generic) > 0


def test_unknown_pin_is_rejected():
    spec = SWEEPS["Cap3Interdigital"]
    broken = SweepSpec(
        component_name=spec.component_name,
        import_path=spec.import_path,
        port_pins=("does_not_exist",),
        grid=spec.grid,
    )
    component = build_component(broken, next(broken.points()), name="c")
    with pytest.raises(ValueError, match="has no pin"):
        port_markers(component, broken)
