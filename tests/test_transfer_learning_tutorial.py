"""Regression checks for the executed v0 transfer-learning tutorial."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-15_Transfer_Learning_with_Static_Embeddings.ipynb")


def test_transfer_learning_tutorial_is_executed_and_visual():
    notebook = json.loads(NOTEBOOK.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    hidden_plot_cells = [cell for cell in code_cells if "hide-input" in cell["metadata"].get("tags", [])]
    outputs = [output for cell in code_cells for output in cell["outputs"]]
    rendered = json.dumps(outputs)
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "SQuADDS Tutorial (uv)",
        "language": "python",
        "name": "squadds-tutorial",
    }
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not [output for output in outputs if output["output_type"] == "error"]
    assert rendered.count("Plotly.newPlot") >= 8
    assert len(hidden_plot_cells) == 8
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden_plot_cells)
    assert "2243" in rendered
    assert "1440" in rendered
    assert "V0TransferLearningStudy" in source
    assert "study.learning_curve" in source
    assert "study.similarity_learning_curves" in source
    assert "required_target_samples" in source
    assert "within_5_percent" in source
    assert "M(C,D)" in source
    assert "target-only" in source
    assert "zero-shot" in source
    assert "transfer" in source
