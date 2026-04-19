from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = ROOT / "tutorials"
DOCS_TUTORIALS_DIR = ROOT / "docs" / "source" / "tutorials"


@dataclass(frozen=True)
class NotebookExport:
    source_py: Path
    notebook_name: str
    replacements: tuple[tuple[str, str], ...] = ()


EXPORTS = [
    NotebookExport(
        source_py=TUTORIALS_DIR / "Tutorial-10_DrivenModal_Capacitance_Extraction.py",
        notebook_name="Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb",
    ),
    NotebookExport(
        source_py=TUTORIALS_DIR / "Tutorial-13_DrivenModal_Combined_Hamiltonian_Extraction.py",
        notebook_name="Tutorial-11_DrivenModal_Combined_Hamiltonian_Extraction.ipynb",
        replacements=(("Tutorial 13:", "Tutorial 11:"),),
    ),
]


def strip_markdown_comment_prefix(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#"):
        return line[1:]
    return line


def parse_percent_script(script_path: Path) -> list[dict]:
    lines = script_path.read_text().splitlines(keepends=True)
    cells: list[dict] = []
    current_kind: str | None = None
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_kind, current_lines
        if current_kind is None:
            return
        source_lines = current_lines
        if current_kind == "markdown":
            source_lines = [strip_markdown_comment_prefix(line) for line in source_lines]
        cell_id = hashlib.sha1("".join(source_lines).encode("utf-8")).hexdigest()[:12]
        cells.append(
            {
                "cell_type": current_kind,
                "id": cell_id,
                "metadata": {},
                "source": source_lines,
                **({"outputs": [], "execution_count": None} if current_kind == "code" else {}),
            }
        )
        current_kind = None
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped == "# %% [markdown]":
            flush_current()
            current_kind = "markdown"
            current_lines = []
            continue
        if stripped == "# %%":
            flush_current()
            current_kind = "code"
            current_lines = []
            continue
        if current_kind is None:
            continue
        current_lines.append(line)

    flush_current()
    return cells


def apply_replacements_to_cells(cells: list[dict], replacements: tuple[tuple[str, str], ...]) -> list[dict]:
    if not replacements:
        return cells

    updated_cells: list[dict] = []
    for cell in cells:
        new_cell = dict(cell)
        new_source = list(cell["source"])
        for old, new in replacements:
            new_source = [line.replace(old, new) for line in new_source]
        new_cell["source"] = new_source
        updated_cells.append(new_cell)
    return updated_cells


def build_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def export_notebook(spec: NotebookExport) -> None:
    cells = parse_percent_script(spec.source_py)
    notebook = build_notebook(apply_replacements_to_cells(cells, spec.replacements))
    tutorial_out = TUTORIALS_DIR / spec.notebook_name
    docs_out = DOCS_TUTORIALS_DIR / spec.notebook_name

    tutorial_out.write_text(json.dumps(notebook, indent=1))
    docs_out.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {tutorial_out}")
    print(f"Wrote {docs_out}")


def main() -> None:
    DOCS_TUTORIALS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in EXPORTS:
        export_notebook(spec)


if __name__ == "__main__":
    main()
