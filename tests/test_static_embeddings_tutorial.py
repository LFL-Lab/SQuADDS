"""Regression checks for the executed static embedding tutorial."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("tutorials/Tutorial-14_Exploring_Static_Layout_Embeddings.ipynb")


def test_static_embedding_tutorial_is_executed_and_visual():
    notebook = json.loads(NOTEBOOK.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in code_cells for output in cell["outputs"]]
    rendered = json.dumps(outputs)
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not [output for output in outputs if output["output_type"] == "error"]
    assert rendered.count("Plotly.newPlot") >= 5
    assert "4577" in rendered
    assert "CapNInterdigitalTee" in rendered
    assert "GeneralizedCapNInterdigital" in rendered
    assert "StaticEmbeddingClient" in source
    assert "SQuADDS_DB.get_layout_embedding" in source
    assert "component_name=None" in source
    assert 'go.Scattergl(' in source
    assert '"Finger count"' in source
    assert '"Finger length (um)"' in source
    assert '"Finger width (um)"' in source
    assert '"Log functional area"' in source
    assert '"Shape occupancy"' in source
