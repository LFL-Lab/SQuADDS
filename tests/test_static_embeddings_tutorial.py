"""Regression checks for the executed versioned embedding tutorial."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/Tutorial-14_Exploring_Static_Layout_Embeddings.ipynb")


def test_static_embedding_tutorial_is_executed_visual_and_versioned():
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
    plotly_outputs = [output for output in outputs if "application/vnd.plotly.v1+json" in output.get("data", {})]
    assert len(plotly_outputs) >= 7
    assert len(hidden_plot_cells) == 7
    assert all(cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in hidden_plot_cells)
    assert "20,062" in rendered
    assert "9,227" in rendered
    assert "512" in rendered
    assert "universal-geometry-v1" in rendered
    assert "GeneralizedCapNInterdigital" in source
    assert 'LayoutEmbeddingClient(version="v0")' in source
    assert 'LayoutEmbeddingClient(version="v1")' in source
    assert "SQuADDS_DB.get_layout_embedding" in source
    assert "v1_client.schema()" in source
    assert "v1_client.control_map()" in source
    assert "all_parameter_names" in source
    assert "capacitance_names" in source
    assert "go.Scattergl(" in source
    assert "cosine similarity" in source
    assert "signed distance" in source
    assert "Nearest-neighbor error" in source
    assert "parameter-control channel" in source
