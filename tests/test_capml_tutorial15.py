from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "Tutorial-15_Contributing_Capacitance_Data_and_Models.ipynb"
DATA = ROOT / "tutorials" / "data" / "tutorial15"


def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


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


def markdown() -> str:
    return "\n".join(source(c) for c in cells() if c["cell_type"] == "markdown")


def test_intro_cell():
    first = cells()[0]
    text = source(first)
    assert first["cell_type"] == "markdown"
    assert text.startswith("# Tutorial 15: ")
    assert "In this tutorial, we will" in text
    assert "By the end of this tutorial, you will be able to:" in text


def test_autoreload_cell():
    second = cells()[1]
    assert second["cell_type"] == "code"
    assert "%load_ext autoreload" in source(second)
    assert "%autoreload 2" in source(second)


def test_license_is_last():
    last = cells()[-1]
    text = source(last)
    assert last["cell_type"] == "markdown"
    assert text.startswith("## License")
    assert "background-color:#3cb1c2" in text
    assert "This tutorial is written by Sadman Ahmed Shanto" in text
    assert "Eli Levenson-Falk 2026." in text


def test_congrats_and_next_steps():
    md = markdown()
    assert "## Next Steps..." in md
    assert "https://lfl-lab.github.io/SQuADDS/source/ml_models.html" in md
    assert "Tutorial-3_Contributing_Validated_Simulation_Data_to_SQuADDS.html" in md


def test_dollar_math_only():
    md = markdown()
    for delimiter in ("\\[", "\\]", "\\(", "\\)"):
        assert delimiter not in md, f"use $ math delimiters, found {delimiter}"


def test_code_cells_are_reader_sized():
    for i, c in enumerate(cells()):
        assert "PLACEHOLDER" not in source(c)
        if c["cell_type"] == "code":
            assert len(source(c).splitlines()) <= 35, f"cell {i} is too large for a tutorial"


def test_contribution_calls_are_dry_runs():
    calls = 0
    for c in cells():
        if c["cell_type"] != "code":
            continue
        for call in re.findall(r"contribute_(?:capacitance|model)\((.*?)\)\n", source(c) + "\n", re.S):
            calls += 1
            assert "dry_run=True" in call, f"contribution call without dry_run: {call}"
    assert calls >= 3


def test_outputs_saved_without_errors():
    code = [c for c in cells() if c["cell_type"] == "code"]
    assert all(isinstance(c.get("execution_count"), int) for c in code), "every code cell must be executed"
    outputs = [o for c in code for o in c.get("outputs", [])]
    assert outputs
    assert not [o for o in outputs if o.get("output_type") == "error"]
    assert any("image/png" in o.get("data", {}) for o in outputs), "plots must be saved inline"


def test_validator_outputs_tell_the_story():
    text = "\n".join(output_text(o) for c in cells() if c["cell_type"] == "code" for o in c.get("outputs", []))
    assert "**Result: PASS**" in text
    assert "positive off-diagonal entries" in text
    assert "the contribution failed validation" in text
    assert "physics prior" in text
    assert "a test score" in text


def test_no_absolute_paths_or_tokens():
    for i, c in enumerate(cells()):
        for text in [source(c)] + [output_text(o) for o in c.get("outputs", [])]:
            assert "/Users/" not in text, f"cell {i} shows a local path"
            assert "/project2/" not in text, f"cell {i} shows a cluster path"
            assert not re.search(r"hf_[A-Za-z0-9]{30,}", text), f"cell {i} shows a token"


def test_tutorial_data_is_small():
    files = [p for p in DATA.rglob("*") if p.is_file()]
    assert len([p for p in files if p.suffix == ".gds"]) == 3
    assert sum(p.stat().st_size for p in files) < 1_000_000
