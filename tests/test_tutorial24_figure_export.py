import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts" / "export_tutorial24_figures.py"
SPEC = spec_from_file_location("export_tutorial24_figures", SCRIPT)
figures = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figures
SPEC.loader.exec_module(figures)


def test_missing_sources_are_explicitly_recorded(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = tmp_path / "figures"

    manifest = figures.export_all(runtime, output)

    assert manifest["counts"] == {
        "generated": 0,
        "skipped_missing_source": len(figures.FIGURE_NAMES),
        "failed": 0,
    }
    assert {entry["figure"] for entry in manifest["figures"]} == set(figures.FIGURE_NAMES)
    assert all(entry["status"] == "skipped_missing_source" for entry in manifest["figures"])
    assert json.loads((output / "manifest.json").read_text())["dpi"] >= 300


def test_scorecard_png_is_high_resolution_and_data_derived(tmp_path):
    source = tmp_path / "acceptance_scorecard.csv"
    destination = tmp_path / "acceptance_scorecard.png"
    pd.DataFrame(
        {
            "representation": ["v2", "v4"],
            "pass_rotation": [False, True],
            "pass_sensitivity": [True, True],
        }
    ).to_csv(source, index=False)

    details = figures.plot_acceptance_scorecard(source, destination)

    with Image.open(destination) as image:
        assert image.info["dpi"][0] >= 300
        assert image.width >= 2500
    assert details == {"rows": 2, "gate_count": 2, "pass_cells": 3, "total_cells": 4}


def test_unknown_result_csvs_are_exported_generically(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    pd.DataFrame({"sample_count": [1, 2, 3], "score": [0.1, 0.4, 0.8]}).to_csv(
        runtime / "later_result.csv", index=False
    )

    manifest = figures.export_all(runtime, tmp_path / "output")

    later = next(entry for entry in manifest["figures"] if entry["figure"] == "later_result.png")
    assert later["status"] == "generated"
    assert Path(later["output"]).is_file()
