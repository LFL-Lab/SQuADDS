"""Regression checks for the executed universal-geometry-v2 tutorial."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-18_Universal_Geometry_v2_Embeddings.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _prose():
    """Join every cell source so assertions are not broken by JSON line wrapping."""
    text = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    return " ".join(text.split())


def test_tutorial_18_is_executed_and_visual():
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in code_cells for output in cell["outputs"]]
    hidden = [cell for cell in code_cells if "hide-input" in cell["metadata"].get("tags", [])]
    rendered = json.dumps(outputs)

    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "SQuADDS Tutorial (uv)",
        "language": "python",
        "name": "squadds-tutorial",
    }
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not [output for output in outputs if output["output_type"] == "error"]
    assert rendered.count("Plotly.newPlot") >= 7
    assert len(hidden) >= 7
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden)


def test_tutorial_18_states_the_v2_contract():
    source = _prose()

    assert "universal-geometry-v2" in source
    assert "pure function" in source
    assert "fit on write" in source
    assert "commensurability" in source
    assert "coupling spectrum" in source
    assert "boundary-element" in source
    assert "no catalogue-derived constant" in source


def test_tutorial_18_runs_the_decisive_geometry_ablation():
    """The headline claim fails unless geometry alone beats all of v0."""
    source = _prose()

    assert "v2 geometry only (416)" in source
    assert "v2 parameters only (96)" in source
    assert "v0 pooled shape only (144)" in source
    assert "no design parameters whatsoever" in source
    assert "The gain is geometry, not parameters" in source


def test_tutorial_18_reports_its_limits():
    source = _prose()

    assert "Not established" in source
    assert "scale extrapolation" in source
    assert "compressed spread" in source or "compressed" in source
    assert "10,000 of the 16,379" in source
    assert "not independent physical experiments" in source


def test_tutorial_18_uses_the_tutorial_16_protocol():
    source = _prose()

    assert "12 independently seeded" in source
    assert "coverage-matched generalist" in source
    assert "budget-matched generalist" in source
    assert "BASE_FINGER_COUNT" in source
    assert "Only the input vector differs" in source
