"""Regression checks for the executed three-family v2 similarity tutorial."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-22_Geometry_Similarity_and_Capacitance_with_v2.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _prose():
    text = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    return " ".join(text.split())


def test_tutorial_22_is_executed_and_deeply_visual():
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in code_cells for output in cell["outputs"]]
    hidden = [cell for cell in code_cells if "hide-input" in cell["metadata"].get("tags", [])]

    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "SQuADDS Tutorial (uv)",
        "language": "python",
        "name": "squadds-tutorial",
    }
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not [output for output in outputs if output["output_type"] == "error"]
    assert json.dumps(outputs).count("Plotly.newPlot") >= 12
    assert len(hidden) >= 10
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden)


def test_tutorial_22_uses_three_families_and_balances_them():
    source = _prose()

    assert "GeneralizedCapNInterdigital" in source
    assert "CapNInterdigitalTee" in source
    assert "TransmonCross" in source
    assert "BALANCED_PER_CLASS = 894" in source
    assert "assert len(data) == BALANCED_PER_CLASS * len(CLASSES)" in source


def test_tutorial_22_audits_similarity_in_geometry_and_physics():
    source = _prose()

    assert "terminal-ground spectrum" in source
    assert "coupling spectrum" in source
    assert "physics proxy" in source
    assert "standardized_unit" in source
    assert "mean_cosine_matrix" in source
    assert "Capacitance calibration" in source
    assert "cross-family high-cosine false friend" in source


def test_tutorial_22_replays_the_encoder_for_every_family():
    source = _prose()

    assert "The whole v2 algorithm, now on all three component families" in source
    assert "FAMILY_ANATOMY_ORDER" in source
    assert "PIPELINE_STEPS" in source
    assert "_raster_frame" in source
    assert "_boundary_samples" in source
    assert "VACUUM_PERMITTIVITY" in source
    assert "assert len(terminals) == 2" in source
    assert "encoder step" in source


def test_tutorial_22_reports_r2_rmse_and_median_absolute_error():
    source = _prose()

    assert "r2_log" in source
    assert "rmse_fF" in source
    assert "median_absolute_error_fF" in source
    assert "within-family" in source
    assert "cross-family" in source


def test_tutorial_22_states_limits_and_practical_rule():
    source = _prose()

    assert "Not established" in source
    assert "High full-vector cosine does **not** establish electrostatic equivalence" in source
    assert "Standardizing on this balanced cohort" in source
    assert "Practical rule" in source
    assert "roughly ten" in source
