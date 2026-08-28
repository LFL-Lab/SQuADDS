"""Regression checks for the executed partition transfer tutorials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TUTORIALS = {
    16: Path("tutorials/Tutorial-16_Geometry_Domain_Transfer_Learning.ipynb"),
    "16b": Path("tutorials/Tutorial-16b_Balanced_Geometry_Domain_Transfer_Learning.ipynb"),
    17: Path("tutorials/Tutorial-17_Cross_Component_Transfer_Learning.ipynb"),
}


@pytest.mark.parametrize(
    ("number", "minimum_plots"),
    [(16, 5), ("16b", 5), (17, 5)],
)
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


def test_tutorial_16_uses_exact_domains_two_generalists_and_foundation_sweep():
    notebook = TUTORIALS[16].read_text(encoding="utf-8")

    assert "13 exact finger-count domains" in notebook
    assert "coverage-matched generalist" in notebook
    assert "budget-matched generalist" in notebook
    assert "partition_splits" in notebook
    assert "source-prior adaptation" in notebook
    assert "applicability score" in notebook
    assert "Sweep every finger count as the foundation" in notebook
    assert "fingerprinted checkpoint" in notebook
    assert "Final interactive foundation snapshot" in notebook


def test_tutorial_16b_uses_one_balanced_cohort_for_every_analysis():
    notebook = TUTORIALS["16b"].read_text(encoding="utf-8")

    assert "BALANCED_ROWS_PER_DOMAIN = 1_260" in notebook
    assert "EXPECTED_BALANCED_ROWS = 13 * BALANCED_ROWS_PER_DOMAIN" in notebook
    assert "balanced_design_ids" in notebook
    assert "design_id in balanced_design_ids" in notebook
    assert "design_id not in balanced_design_ids" in notebook
    assert "balanced_counts" in notebook
    assert "same 945-row training pool" in notebook
    assert "only scientific variable changed is domain size" in notebook


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
