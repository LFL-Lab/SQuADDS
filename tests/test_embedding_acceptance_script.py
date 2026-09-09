from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_embedding_acceptance.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_embedding_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_acceptance_report_is_reproducible_and_contains_predeclared_gates(tmp_path):
    pytest.importorskip("klayout.db")
    module = load_script()
    scorecard = module.run(tmp_path)

    assert set(scorecard.representation) == set(module.REPRESENTATION_INFO)
    assert scorecard.deterministic_bitwise.all()
    assert (scorecard.numerical_gates_total == 9).all()
    assert (tmp_path / "acceptance_report.json").is_file()
    assert (tmp_path / "controlled_embeddings.npz").is_file()
    v4 = scorecard.set_index("representation").loc["v4 hybrid pair"]
    assert bool(v4.arbitrary_terminal_contract)
    assert abs(v4.gap_spearman) >= 0.95
