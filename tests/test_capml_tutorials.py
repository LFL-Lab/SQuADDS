from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    11: ROOT / "tutorials" / "Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.ipynb",
    12: ROOT / "tutorials" / "Tutorial-12_Planar_Capacitance_Families_in_SQuADDS.ipynb",
    13: ROOT / "tutorials" / "Tutorial-13_From_Layout_to_Hamiltonian_with_ML.ipynb",
    14: ROOT / "tutorials" / "Tutorial-14_Training_on_Your_Own_Designs.ipynb",
    15: ROOT / "tutorials" / "Tutorial-15_Contributing_Capacitance_Data_and_Models.ipynb",
    16: ROOT / "tutorials" / "Tutorial-16_Inverse_Design_by_ML_Screening.ipynb",
}
PARAMS = pytest.mark.parametrize("number", sorted(NOTEBOOKS))


def cells(number: int) -> list[dict]:
    if not NOTEBOOKS[number].exists():
        pytest.skip(f"{NOTEBOOKS[number].name} is not in this checkout")
    return json.loads(NOTEBOOKS[number].read_text(encoding="utf-8"))["cells"]


def source(cell: dict) -> str:
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def output_text(output: dict) -> str:
    if "text" in output:
        t = output["text"]
        return t if isinstance(t, str) else "".join(t)
    parts = []
    for mime, value in output.get("data", {}).items():
        if mime.startswith("text/") or mime == "application/json":
            parts.append(value if isinstance(value, str) else "".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


@PARAMS
def test_intro_cell(number):
    first = cells(number)[0]
    text = source(first)
    assert first["cell_type"] == "markdown"
    assert text.startswith(f"# Tutorial {number}: ")
    assert len(text.split("\n\n")) >= 2, "the title is followed by a short motivation"


@PARAMS
def test_autoreload_cell(number):
    second = cells(number)[1]
    assert second["cell_type"] == "code"
    assert "%load_ext autoreload" in source(second)
    assert "%autoreload 2" in source(second)


@PARAMS
def test_license_is_last(number):
    last = cells(number)[-1]
    text = source(last)
    assert last["cell_type"] == "markdown"
    assert text.startswith("## License")
    assert "background-color:#3cb1c2" in text
    assert "This tutorial is written by Sadman Ahmed Shanto" in text
    assert "Eli Levenson-Falk 2026." in text


@PARAMS
def test_congrats_and_next_steps(number):
    markdown = "\n".join(source(c) for c in cells(number) if c["cell_type"] == "markdown")
    assert "## Next Steps..." in markdown
    assert re.search(r"https://lfl-lab\.github\.io/SQuADDS/source/tutorials/Tutorial-\d+_[\w-]+\.html", markdown)


@PARAMS
def test_dollar_math_only(number):
    markdown = "\n".join(source(c) for c in cells(number) if c["cell_type"] == "markdown")
    for delimiter in ("\\[", "\\]", "\\(", "\\)"):
        assert delimiter not in markdown, f"use $ math delimiters, found {delimiter}"


@PARAMS
def test_no_placeholders(number):
    for c in cells(number):
        assert "PLACEHOLDER" not in source(c)


@PARAMS
def test_code_cells_are_reader_sized(number):
    for i, c in enumerate(cells(number)):
        if c["cell_type"] == "code":
            assert len(source(c).splitlines()) <= 35, f"cell {i} is too large for a tutorial"


@PARAMS
def test_outputs_saved_without_errors(number):
    code = [c for c in cells(number) if c["cell_type"] == "code"]
    assert all(isinstance(c.get("execution_count"), int) for c in code), "every code cell must be executed"
    outputs = [o for c in code for o in c.get("outputs", [])]
    assert outputs
    assert not [o for o in outputs if o.get("output_type") == "error"]
    assert any("image/png" in o.get("data", {}) for o in outputs), "plots must be saved inline"


@PARAMS
def test_no_absolute_paths_or_tokens(number):
    for i, c in enumerate(cells(number)):
        texts = [source(c)] + [output_text(o) for o in c.get("outputs", [])]
        for text in texts:
            assert "/Users/" not in text, f"cell {i} shows a local path"
            assert "/project2/" not in text, f"cell {i} shows a cluster path"
            assert not re.search(r"hf_[A-Za-z0-9]{30,}", text), f"cell {i} shows a token"
