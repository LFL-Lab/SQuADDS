"""Regression checks for the executed balanced-domain v2 tutorial."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-21_Balanced_Geometry_Domains_with_v2.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _prose():
    """Join every cell source so assertions survive JSON line wrapping."""
    text = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    return " ".join(text.split())


def test_tutorial_21_is_executed_and_visual():
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
    assert json.dumps(outputs).count("Plotly.newPlot") >= 6
    assert len(hidden) >= 6
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden)


def test_tutorial_21_keeps_the_16b_protocol():
    """Only the embedding may differ from Tutorial 16b's experiment."""
    source = _prose()

    assert "BALANCED_PER_DOMAIN" in source
    assert "EXPECTED_BALANCED_ROWS" in source
    assert "coverage-matched generalist" in source
    assert "budget-matched generalist" in source
    assert "Benjamini-Hochberg" in source
    assert "only variable is the embedding" in source
    assert "BASE_FINGER_COUNT = 8" in source


def test_tutorial_21_verifies_its_fast_solver():
    """The atlas reuses one factorization per training set; it must match the library."""
    source = _prose()

    assert "class RidgeBank" in source
    assert "assert np.allclose(bank.fit(), reference.weights_" in source
    assert "fast ridge bank verified against TransferRidgeRegressor" in source


def test_tutorial_21_runs_the_ablation_and_atlas():
    source = _prose()

    assert "v2 geometry only" in source
    assert "v0 pooled shape only" in source
    assert "positive_transfer_rate" in source
    assert "The gain is geometric, again" in source


def test_tutorial_21_reports_its_limits():
    source = _prose()

    assert "Not established" in source
    assert "754 designs per domain, not Tutorial 16b's 1,260" in source
    assert "not twelve independent physical experiments" in source
    assert "foundation *representation*" in source
    assert "More dimensions are a liability at the smallest budget" in source
