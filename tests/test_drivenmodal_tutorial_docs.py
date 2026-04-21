from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_NAMES = [
    "Tutorial-10_DrivenModal_Capacitance_Extraction",
    "Tutorial-11_DrivenModal_Combined_Hamiltonian_Extraction",
]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def tutorial_notebook_paths(name: str) -> tuple[Path, Path]:
    return (
        ROOT / "tutorials" / f"{name}.ipynb",
        ROOT / "docs" / "source" / "tutorials" / f"{name}.ipynb",
    )


def test_drivenmodal_tutorial_notebooks_exist_in_tutorials_and_docs():
    for name in TUTORIAL_NAMES:
        tutorial_path, docs_path = tutorial_notebook_paths(name)
        assert tutorial_path.exists(), f"Missing tutorial notebook: {tutorial_path}"
        assert docs_path.exists(), f"Missing docs tutorial notebook: {docs_path}"


def test_drivenmodal_tutorials_are_listed_in_docs_index():
    index_path = ROOT / "docs" / "source" / "tutorials" / "index.rst"
    index_text = index_path.read_text()

    for name in TUTORIAL_NAMES:
        assert f"{name}.ipynb" in index_text


def test_drivenmodal_tutorial_notebooks_have_intro_and_license_cells():
    for name in TUTORIAL_NAMES:
        tutorial_path, docs_path = tutorial_notebook_paths(name)
        for notebook_path in (tutorial_path, docs_path):
            notebook = load_notebook(notebook_path)
            assert all(cell.get("id") for cell in notebook["cells"]), f"Notebook has missing cell ids: {notebook_path}"
            markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
            assert markdown_cells, f"No markdown cells in {notebook_path}"

            first_markdown = "".join(markdown_cells[0]["source"])
            assert "# Tutorial" in first_markdown, f"Notebook missing title in {notebook_path}"

            license_markdown = "".join(markdown_cells[-1]["source"])
            assert "## License" in license_markdown, f"Notebook missing license footer in {notebook_path}"
            assert "This code is a part of SQuADDS" in license_markdown
