from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tutorial24_modular_studies.py"


def _module():
    specification = importlib.util.spec_from_file_location("run_tutorial24_modular_studies", SCRIPT)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_modular_study_preserves_topology_and_rejects_raw_vector_addition(tmp_path):
    pytest.importorskip("klayout.db")
    module = _module()

    summary = module.run(tmp_path / "modularity")

    assert summary["module_terminal_counts"] == {"module_a": 2, "module_b": 2, "joined": 3}
    assert summary["joined_proxy_matrix_shape"] == [4, 4]
    assert summary["naive_vector_addition_relative_error"] > 0.05
    assert summary["base_rotation_relative_drift"] < 5e-3
    assert summary["wider_gap_rotation_relative_drift"] < 5e-3
    assert summary["pose_matched_delta_cosine"] > 0.8
    assert summary["pose_matched_delta_relative_error"] < 0.02
    vectors = np.load(tmp_path / "modularity/vectors.npz")
    assert vectors["joined_three_terminal"].shape == (544,)
    assert (tmp_path / "modularity/block_arithmetic.csv").is_file()
