"""Regression checks for the executed static embedding tutorial."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-14_Exploring_Static_Layout_Embeddings.ipynb")


def test_static_embedding_tutorial_is_executed_and_visual():
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
    assert rendered.count("Plotly.newPlot") >= 7
    assert len(hidden_plot_cells) == 7
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden_plot_cells)
    assert "7727" in rendered
    assert "CapNInterdigitalTee" in rendered
    assert "GeneralizedCapNInterdigital" in rendered
    assert "StaticEmbeddingClient" in source
    assert "SQuADDS_DB.get_layout_embedding" in source
    assert "component_name=None" in source
    assert "go.Scattergl(" in source
    assert '"Finger count"' in source
    assert '"Finger length (um)"' in source
    assert '"Finger width (um)"' in source
    assert '"Log functional area"' in source
    assert '"Shape occupancy"' in source
    assert "raw_prediction = algebra_vectors" in source
    assert "finger_delta" in source
    assert "Embedding algebra" in source
