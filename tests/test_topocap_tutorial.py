"""Regression contract for the executed TopoCap research tutorial."""

from __future__ import annotations

from pathlib import Path

import nbformat

REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "tutorials/Tutorial-18_Topology_General_Transfer_Learning.ipynb"


def test_topocap_tutorial_is_executed_and_uses_the_research_kernel():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"
    ]

    assert notebook.metadata.kernelspec.name == "squadds-tutorial"
    assert notebook.metadata.topocap_report.contract_version == "topocap-report-results-v1"
    assert len(code_cells) >= 20
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert all(cell.get("outputs") for cell in code_cells)
    assert not errors


def test_topocap_plot_cells_are_hidden_but_keep_visible_outputs():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    hidden_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code" and "hide-input" in cell.metadata.get("tags", [])
    ]

    assert len(hidden_cells) >= 15
    for cell in hidden_cells:
        assert cell.source.startswith("# %% hide input")
        assert cell.metadata.jupyter.source_hidden is True
        assert cell.outputs


def test_topocap_tutorial_records_complete_exploratory_evidence_and_handoff():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    rendered_text = "\n".join(
        str(output.get("text", "")) + str(output.get("data", {}).get("text/markdown", ""))
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )

    assert "Part II: Dataset collection plan for Saikat Das" in source
    assert "source_retrieval_2048_ebra" in source
    assert "retrieved source labels - matched shuffled labels" in source
    assert "QUICK_SMOKE_NON_CLAIM_BEARING" not in rendered_text
    assert "Study status: EXPLORATORY_COMPLETE" in rendered_text
    assert "must **not** be described as evidence" in rendered_text
