"""Structural checks for the executed omnibus v3 tutorial."""

from __future__ import annotations

import ast
import json
from pathlib import Path

TUTORIAL = Path("tutorials/Tutorial-23_Capacitance_Operator_v3.ipynb")


def test_tutorial_23_is_executed_and_covers_tutorials_18_through_22():
    notebook = json.loads(TUTORIAL.read_text())
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    for section in range(18, 23):
        assert f"Tutorial {section} equivalent" in markdown
    assert "R2" in markdown
    assert "RMSE" in markdown
    assert "median absolute error" in markdown
    assert "Recommended next acquisition sweep" in markdown

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 12
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    for cell in code_cells:
        ast.parse("".join(cell["source"]))


def test_tutorial_23_contains_interactive_visuals_and_no_uploads():
    notebook = json.loads(TUTORIAL.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert source.count("fig.show()") >= 8
    assert "updatemenus" in source
    assert "push_to_hub" not in source
    assert "git push" not in source
