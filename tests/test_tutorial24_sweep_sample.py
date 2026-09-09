from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_tutorial24_sweep_sample.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_tutorial24_sweep_sample", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_campaign_map_includes_supplied_new_families():
    module = load_script()
    assert module.CAMPAIGNS["exp10_exports"] == "GeneralizedCapConcentric"
    assert module.CAMPAIGNS["exp11_exports"] == "GeneralizedCapStar"
    assert "asymmetric" in module.CAMPAIGNS["exp9_exports"]


def test_deterministic_sample_is_repeatable_and_campaign_specific(tmp_path):
    module = load_script()
    paths = [tmp_path / f"{index:03d}.json" for index in range(100)]
    first = module.deterministic_sample(paths, 12, 24, "a")
    assert first == module.deterministic_sample(paths, 12, 24, "a")
    assert first != module.deterministic_sample(paths, 12, 24, "b")

