"""Regression checks for the executed cross-class v2 tutorial."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-20_Cross_Class_Transfer_with_v2.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _prose():
    """Join every cell source so assertions survive JSON line wrapping."""
    text = "\n".join("".join(cell["source"]) for cell in _notebook()["cells"])
    return " ".join(text.split())


def test_tutorial_20_is_executed_and_visual():
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
    assert json.dumps(outputs).count("Plotly.newPlot") >= 5
    assert len(hidden) >= 5
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden)


def test_tutorial_20_uses_all_three_capacitance_classes():
    source = _prose()

    for component, field in (
        ("GeneralizedCapNInterdigital", "north_to_south"),
        ("CapNInterdigitalTee", "top_to_bottom"),
        ("TransmonCross", "cross_to_claw"),
    ):
        assert component in source, component
        assert field in source, field
    assert "shared by ALL THREE classes" in source


def test_tutorial_20_runs_the_held_out_class_experiment():
    """The experiment no earlier tutorial could run with only two classes."""
    source = _prose()

    assert "held_out" in source
    assert "predict the third with no labels from it" in source
    assert "Holding out `CapNInterdigitalTee` works" in source
    assert "only representation that is ever positive" in source


def test_tutorial_20_reports_the_negative_results():
    """The residual idea failed and the similarity metric is not uniformly safe."""
    source = _prose()

    assert "The residual idea does not work" in source
    assert "A proxy is a good feature and a bad denominator" in source
    assert "Zero-shot cross-class prediction is close to break-even, not reliable" in source
    assert "Cross-class similarity is still not trustworthy" in source
    assert "`TransmonCross` is still not predictable from the other two" in source
    assert "Zero-shot cross-class prediction is close to break-even, not reliable" in source


def test_tutorial_20_fits_the_feature_map_without_leakage():
    source = _prose()

    assert "fit on the training classes only" in source
    assert "def project(matrix, fit_rows)" in source


def test_tutorial_20_includes_the_balanced_cohort():
    """Class sizes differ 15-fold, so the balanced repeat is the fair experiment."""
    source = _prose()

    assert "A balanced cohort: removing class size as a confound" in source
    assert "BALANCED_PER_CLASS" in source
    assert "EXPECTED_BALANCED_ROWS" in source
    assert "only scientific variable that changes between rotations" in source
    assert "two of the three rotations are positive" in source


def test_tutorial_20_answers_the_new_contributor_question():
    source = _prose()

    assert "The question a new contributor actually asks" in source
    assert "About ten labels" in source
    assert "only helps if the representation aligns the classes" in source


def test_tutorial_20_separates_predictive_from_transfer_blocks():
    """A block that predicts well in-class need not transfer, and vice versa."""
    source = _prose()

    assert "Which block carries prediction, and which carries transfer" in source
    assert "in-class prediction" in source
    assert "cross-class, no labels" in source
    assert "The block that predicts is not the block that transfers" in source
    assert "block ablation is almost uninformative" in source
    assert "The shape spectrum is the clearest reversal" in source
    assert "Block ablation run only in-class is misleading" in source


def test_tutorial_20_uses_the_port_complete_release():
    """The regenerated CapN and TransmonCross rows must be distinguishable."""
    source = _prose()

    assert "LAYOUT_RELEASE" in source
    assert "port-complete" in source
    assert "matrix_sha256" in source, "checkpoint fingerprint must depend on the vectors"
    assert "Ordered ports alone changed nothing measurable on `TransmonCross`" in source


def test_tutorial_20_shows_the_geometry_and_similarity_extremes():
    source = _prose()
    outputs = json.dumps(
        [output for cell in _notebook()["cells"] if cell["cell_type"] == "code" for output in cell["outputs"]]
    )

    assert "What the three families actually look like" in source
    assert "The closest and farthest shapes, by cosine similarity" in source
    assert "build_gds_index" in source
    assert "role_traces" in source
    assert "closest: design A" in outputs and "farthest: design B" in outputs
    assert "Mean standardized cosine similarity" in outputs
