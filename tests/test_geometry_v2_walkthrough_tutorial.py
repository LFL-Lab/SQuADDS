"""Regression checks for the executed universal-geometry-v2 walkthrough."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-19_How_Universal_Geometry_v2_Works.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _prose():
    """Join every cell source so assertions survive JSON line wrapping."""
    text = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    return " ".join(text.split())


def test_walkthrough_is_executed_and_visual():
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
    assert json.dumps(outputs).count("Plotly.newPlot") >= 9
    assert len(hidden) >= 9
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden)


def test_walkthrough_covers_every_block_in_order():
    source = _prose()

    for heading in (
        "terminals are discovered, not declared",
        "the coupling spectrum",
        "physical metrics",
        "the shape spectrum",
        "parameter statistics, without a name registry",
        "a physics proxy",
        "Assembling the vector",
    ):
        assert heading in source, heading


def test_walkthrough_verifies_the_hand_derivation():
    """The tutorial must prove its explanation matches the shipped encoder."""
    source = _prose()

    assert "assert np.allclose(hand_derived, library, atol=1e-5)" in source
    assert "largest disagreement between the hand derivation and encode()" in source
    assert "identical to the block inside encode()" in source


def test_walkthrough_states_the_invariances_precisely():
    source = _prose()

    assert "Only translation is invariant everywhere" in source
    assert "orientation-free in the blocks that describe shape and coupling" in source
    assert "deliberately NOT invariant" in source or "Scale is the last row" in source


def test_walkthrough_runs_the_foreign_contributor_scenario():
    source = _prose()

    assert "build_foreign_capacitor" in source
    assert "FOREIGN_OPTIONS" in source
    assert "none of which appear in our catalogue" in source
    assert "no name in the foreign schema matches ours" in source
    assert "assert np.isfinite(similarity).all()" in source


def test_walkthrough_has_an_interactive_algorithm_replay():
    """The step-by-step slider is the notebook's answer to what encode actually does."""
    notebook = _notebook()
    html = json.dumps(
        [output for cell in notebook["cells"] if cell["cell_type"] == "code" for output in cell["outputs"]]
    )
    source = _prose()

    assert "The whole algorithm in one interactive figure" in source
    assert "The pipeline at a glance" in source
    assert "currentvalue" in html
    for step in ("roles and terminals", "physical metrics", "coupling spectrum", "physics proxy"):
        assert step in html, step


def test_walkthrough_explains_porting_to_a_new_family():
    source = _prose()

    assert "Porting this to a different design family" in source
    assert "Nothing, if you only change the component" in source
    assert "The one thing you must supply: layer roles" in source
    assert "What needs a new version of the standard" in source
    assert "Different simulation or analysis results" in source
    assert "changing an input needs nothing, changing the meaning of a coordinate" in source
