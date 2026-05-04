from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVENMODAL_NOTEBOOKS = [
    ROOT / "tutorials" / "Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb",
    ROOT / "tutorials" / "Tutorial-11_DrivenModal_Combined_Hamiltonian_Extraction.ipynb",
    ROOT / "docs" / "source" / "tutorials" / "Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb",
    ROOT / "docs" / "source" / "tutorials" / "Tutorial-11_DrivenModal_Combined_Hamiltonian_Extraction.ipynb",
]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def test_drivenmodal_tutorials_do_not_ship_ai_apology_notes():
    banned_phrases = [
        "AI write these tutorials",
        "terrible",
        "will make them easy to read",
    ]

    for notebook_path in DRIVENMODAL_NOTEBOOKS:
        text = notebook_path.read_text()
        for phrase in banned_phrases:
            assert phrase not in text


def test_drivenmodal_tutorials_keep_code_cells_reader_sized():
    for notebook_path in DRIVENMODAL_NOTEBOOKS:
        notebook = load_notebook(notebook_path)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert len(cell["source"]) <= 35, f"{notebook_path} cell {index} is too large for a tutorial"


def test_drivenmodal_tutorials_use_dollar_math_delimiters():
    bracket_math_delimiters = ["\\[", "\\]", "\\(", "\\)"]

    for notebook_path in DRIVENMODAL_NOTEBOOKS:
        markdown = "\n".join(
            "".join(cell["source"]) for cell in load_notebook(notebook_path)["cells"] if cell["cell_type"] == "markdown"
        )
        for delimiter in bracket_math_delimiters:
            assert delimiter not in markdown, f"Use $ math delimiters in {notebook_path}, found {delimiter}"


def test_drivenmodal_tutorials_ship_reader_visible_outputs():
    for notebook_path in DRIVENMODAL_NOTEBOOKS:
        notebook = load_notebook(notebook_path)
        outputs = [output for cell in notebook["cells"] if cell["cell_type"] == "code" for output in cell["outputs"]]
        assert outputs, f"Notebook should include saved outputs for docsite readers: {notebook_path}"
