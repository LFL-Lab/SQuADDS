from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "tutorials" / "Tutorial-24_Embedding_Representation_Acceptance.ipynb"


def sources():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]), notebook


def test_tutorial24_is_executed_and_representation_first():
    source, notebook = sources()
    assert "Decide what “physically sensible” means before scoring an encoder" in source
    assert "arbitrary-terminal" in source
    assert "39,070" in source
    assert "GeneralizedCapConcentric" not in source or "supplied" in source
    assert "R²" in source and "RMSE" in source and "median absolute error" in source
    assert source.count("Mermaid source") == 4
    assert "Can devices be assembled by adding their embeddings?" in source
    assert "Does the shared representation actually save new simulations?" in source
    assert "Final stress test: a three-terminal device" in source
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert all("hide-input" in cell.get("metadata", {}).get("tags", []) for cell in code_cells)
    assert sum(bool(cell.get("outputs")) for cell in code_cells) >= 10


def test_tutorial24_keeps_flipchip_as_future_scope():
    source, _ = sources()
    assert "future layer-stack sidecar" in source
    assert "does not manufacture flip-chip evidence" in source
    assert "JSON/XML" in source
