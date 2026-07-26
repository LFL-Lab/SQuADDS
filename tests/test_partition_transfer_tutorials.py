"""Regression checks for the executed partition transfer tutorials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TUTORIALS = {
    16: Path("tutorials/Tutorial-16_Geometry_Domain_Transfer_Learning.ipynb"),
    17: Path("tutorials/Tutorial-17_Cross_Component_Transfer_Learning.ipynb"),
}


@pytest.mark.parametrize(("number", "minimum_plots"), [(16, 5), (17, 5)])
def test_partition_transfer_tutorial_is_executed_and_visual(number, minimum_plots):
    notebook = json.loads(TUTORIALS[number].read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in code_cells for output in cell["outputs"]]
    hidden_plot_cells = [cell for cell in code_cells if "hide-input" in cell["metadata"].get("tags", [])]
    rendered = json.dumps(outputs)

    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "SQuADDS Tutorial (uv)",
        "language": "python",
        "name": "squadds-tutorial",
    }
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not [output for output in outputs if output["output_type"] == "error"]
    assert rendered.count("Plotly.newPlot") >= minimum_plots
    assert len(hidden_plot_cells) >= minimum_plots
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden_plot_cells)


def test_tutorial_16_uses_balanced_geometry_domains_and_fractional_api():
    notebook = TUTORIALS[16].read_text(encoding="utf-8")

    assert "2-4 fingers" in notebook
    assert "5-7 fingers" in notebook
    assert "8-10 fingers" in notebook
    assert "V0PartitionTransferStudy" in notebook
    assert "study.learning_curves" in notebook
    assert "study.dedicated_benchmarks" in notebook
    assert "same_budget_gain" in notebook
    assert "specialist_gap" in notebook
    assert "Claim audit" in notebook


def test_tutorial_17_compares_schema_baseline_with_cross_class_v0_transfer():
    notebook = TUTORIALS[17].read_text(encoding="utf-8")

    assert "GeneralizedCapNInterdigital" in notebook
    assert "CapNInterdigitalTee" in notebook
    assert "PartitionTransferStudy" in notebook
    assert "V0PartitionTransferStudy" in notebook
    assert "SourceFeatureProjector" in notebook
    assert "shared parameters" in notebook
    assert "v0 embeddings" in notebook
    assert "log_mutual_capacitance" in notebook
    assert "Five percent is too small here" in notebook
    assert "Claim audit and scope" in notebook
