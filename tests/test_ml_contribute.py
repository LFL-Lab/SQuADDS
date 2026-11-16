"""Data and model contributions (squadds.ml.capacitance.contribute) with a mocked HfApi: no network."""

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gdstk")
pytest.importorskip("yaml")

import yaml  # noqa: E402

from squadds.ml.capacitance import contribute as K  # noqa: E402
from squadds.ml.capacitance import qc as Q  # noqa: E402

gdstk = pytest.importorskip("gdstk")


def maxwell(x=10.0, y=8.0, z=2.0):
    return np.array([[x + y, -x, -y], [-x, x + z, -z], [-y, -z, y + z]])


def write_gds(path, n_pads=2):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("main")
    holes = [gdstk.rectangle((40 + 120 * k, 40), (140 + 120 * k, 140)) for k in range(n_pads)]
    cell.add(*gdstk.boolean(gdstk.rectangle((0, 0), (200 + 120 * n_pads, 200)), holes, "not", layer=1, datatype=0))
    for k in range(n_pads):
        cell.add(gdstk.rectangle((60 + 120 * k, 60), (120 + 120 * k, 120), layer=1, datatype=10))
    path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(path))
    return path


TEMPLATE = Path(K.__file__).parents[2] / "database" / "templates" / "planar_capacitance_contribution"


def write_csv(path, C_fF, names=("C1", "G", "C2"), scales=(0.99, 1.0)):
    order = ["G", "C1", "C2"]
    ix = [order.index(n) for n in names]
    hdr = ["Freq [GHz]", "Pass []"] + [f"C({a},{b}) [pF]" for a in names for b in names]
    rows = [[13, k + 1] + [C_fF[i, j] * s * 1e-3 for i in ix for j in ix] for k, s in enumerate(scales)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([hdr, *rows])


def make_contribution(root: Path, n=2, C=None):
    shutil.copytree(TEMPLATE, root)
    m = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    m["contributor"]["email"] = "jane@example.edu"
    m["generator"]["url"] = "/Users/jane/private/gen.py"
    (root / "manifest.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    rows = []
    for k in range(n):
        d = f"design_{k:02d}"
        write_gds(root / "gds" / f"{d}.gds", n_pads=2) if k == 0 else _shifted(root / "gds" / f"{d}.gds", k)
        write_csv(root / "sims" / d / "q3d" / "cap_matrix.csv", maxwell() if C is None else C)
        (root / "sims" / d / "q3d" / "convergence.txt").write_text(
            "Completed : 2\nTarget    : 0.05\nCurrent   : 0.03\n", encoding="utf-8"
        )
        (root / "sims" / d / "q3d" / "run.json").write_text(
            json.dumps({"cores": 8, "log": "/home/jane/run.log"}), encoding="utf-8"
        )
        rows.append(f"{d},gds/{d}.gds,interdigital,{{}},G;C1;C2,")
    with open(root / "designs.csv", "a", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    return root


def _shifted(path, k):
    write_gds(path, n_pads=2)
    lib = gdstk.read_gds(str(path))
    for p in lib.top_level()[0].polygons:
        p.translate(k, 0)
    lib.write_gds(str(path))


REF = pd.DataFrame({"design_id": ["ref/x/1"], "geom_hash": ["sha256:none"], "split": ["train"]})


def test_template_is_complete():
    m = yaml.safe_load((TEMPLATE / "manifest.yaml").read_text(encoding="utf-8"))
    assert set(K.MANIFEST_REQUIRED) <= set(m)
    assert K.read_designs(TEMPLATE) == []
    assert (TEMPLATE / "README.md").exists()


def test_validate_clean_contribution(tmp_path):
    root = make_contribution(tmp_path / "c")
    r = K.validate_capacitance_contribution(root, reference=REF, report_path=tmp_path / "report.md")
    assert r.ok, r.all_errors()
    assert r.info["designs"] == 2 and r.info["simulations"] == 2 and r.info["licence"] == "CC-BY-4.0"
    assert any("email removed" in f for f in r.fixes) and any("absolute path" in f for f in r.fixes)
    assert "PASS" in (tmp_path / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "edit, needle",
    [
        (lambda m: m.update(licence="MIT"), "licence"),
        (lambda m: m.pop("stack"), "'stack' missing"),
        (lambda m: m["units"].update(capacitance="mF"), "units.capacitance"),
    ],
)
def test_manifest_rejections(tmp_path, edit, needle):
    root = make_contribution(tmp_path / "c", n=1)
    m = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    edit(m)
    (root / "manifest.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    r = K.validate_capacitance_contribution(root, reference=REF)
    assert not r.ok and any(needle in e for e in r.all_errors()), r.all_errors()


def test_label_and_net_rejections(tmp_path):
    bad = maxwell()
    bad[1, 2] = bad[2, 1] = 0.5
    r = K.validate_capacitance_contribution(make_contribution(tmp_path / "a", n=1, C=bad), reference=REF)
    assert any("positive off-diagonal" in e for e in r.all_errors())
    root = make_contribution(tmp_path / "b", n=1)
    txt = (root / "designs.csv").read_text(encoding="utf-8").replace("G;C1;C2", "G;C1")
    (root / "designs.csv").write_text(txt, encoding="utf-8")
    r = K.validate_capacitance_contribution(root, reference=REF)
    assert any("never merged" in e for e in r.all_errors())


def test_test_set_duplicate_is_rejected(tmp_path):
    root = make_contribution(tmp_path / "c", n=1)
    h = Q.geom_hash(root / "gds" / "design_00.gds")
    ref = pd.DataFrame({"design_id": ["ref/exp6/q3d_cap_0001"], "geom_hash": [h], "split": ["test"]})
    r = K.validate_capacitance_contribution(root, reference=ref)
    assert not r.ok and any("held-out test" in e for e in r.all_errors())


def test_long_format_matrix(tmp_path):
    root = make_contribution(tmp_path / "c", n=1)
    sim = root / "sims" / "design_00" / "q3d"
    (sim / "cap_matrix.csv").unlink()
    C = maxwell()
    names = ["G", "C1", "C2"]
    pd.DataFrame(
        [(names[i], names[j], C[i, j]) for i in range(3) for j in range(i, 3)], columns=["net_a", "net_b", "C_fF"]
    ).to_csv(sim / "matrix.csv", index=False)
    s = K.read_sim(sim)
    np.testing.assert_allclose(s["C"], C)
    assert K.validate_capacitance_contribution(root, reference=REF).ok


def test_contribute_dry_run_and_pr(tmp_path):
    root = make_contribution(tmp_path / "c")
    out = K.contribute_capacitance(root, dry_run=True, reference=REF)
    assert out["folder"] == "contrib/my-lab-2026-11-a" and out["pr_url"] is None
    assert "contrib/my-lab-2026-11-a/validation_report.md" in out["files"]
    assert "contrib/my-lab-2026-11-a/gds/design_00.gds" in out["files"]
    api = MagicMock()
    api.create_commit.return_value = SimpleNamespace(pr_url="https://huggingface.co/datasets/x/y/discussions/1")
    out = K.contribute_capacitance(root, repo_id="me/scratch", reference=REF, api=api)
    assert out["pr_url"].endswith("/discussions/1")
    kw = api.create_commit.call_args.kwargs
    assert kw["create_pr"] is True and kw["repo_type"] == "dataset" and kw["repo_id"] == "me/scratch"
    ops = {o.path_in_repo: o for o in kw["operations"]}
    man = yaml.safe_load(ops["contrib/my-lab-2026-11-a/manifest.yaml"].path_or_fileobj)
    assert "email" not in man["contributor"] and man["generator"]["url"] == "gen.py"
    run = json.loads(ops["contrib/my-lab-2026-11-a/sims/design_00/q3d/run.json"].path_or_fileobj)
    assert run["log"] == "run.log"


def test_contribute_refuses_failed(tmp_path):
    root = make_contribution(tmp_path / "c", n=1)
    m = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    m["licence"] = "proprietary"
    (root / "manifest.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    api = MagicMock()
    with pytest.raises(ValueError, match="failed validation"):
        K.contribute_capacitance(root, reference=REF, api=api)
    api.create_commit.assert_not_called()


# ---------------------------------------------------------------------------------------------------------------- models
META = {
    "name": "my-interdigital-model",
    "title": "In-family retrain on 40 interdigital designs",
    "licence": "MIT",
    "contributor": {"name": "Jane Doe", "affiliation": "Example University"},
    "training_data": {"sha256": "ab" * 32, "split_used": "train", "description": "paper TRAIN + 40 designs"},
    "val_score": {
        "median_pct": 0.25,
        "mae_fF": 0.07,
        "rmse_fF": 0.08,
        "layouts": 4608,
        "seconds": 120.0,
        "machine": "x86",
        "threads": 2,
        "split": "val",
    },
}


def model_dir(tmp_path, meta=META):
    d = tmp_path / "model"
    (d / "s0").mkdir(parents=True)
    (d / "s0" / "model.pt").write_bytes(b"weights")
    (d / "squadds_model.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return d


def stub_loader(root, meta, paths):
    return SimpleNamespace(entry={"arm": "full_gnn"})


def test_model_pr(tmp_path):
    api = MagicMock()
    api.hf_hub_download.side_effect = FileNotFoundError("404 model_zoo.yaml not found")
    api.create_commit.return_value = SimpleNamespace(pr_url="https://hf.co/pr/2")
    out = K.contribute_model(
        "jane/my-model",
        "c" * 40,
        registry_repo="me/scratch",
        local_dir=model_dir(tmp_path),
        api=api,
        loader=stub_loader,
    )
    e = out["entry"]
    assert out["pr_url"] == "https://hf.co/pr/2" and e["test_eligible"] and e["arm"] == "full_gnn"
    assert e["files"][0]["path"] == "s0/model.pt" and len(e["files"][0]["sha256"]) == 64
    ops = {o.path_in_repo: o for o in api.create_commit.call_args.kwargs["operations"]}
    zoo = yaml.safe_load(ops["model_zoo.yaml"].path_or_fileobj)
    assert zoo["models"][0]["repo_id"] == "jane/my-model"
    assert "model_zoo/my-interdigital-model/validation_report.md" in ops


def test_model_existing_zoo_and_duplicate(tmp_path):
    zp = tmp_path / "zoo.yaml"
    zp.write_text(
        yaml.safe_dump({"version": 1, "models": [{"repo_id": "jane/my-model", "revision": "c" * 40}]}), encoding="utf-8"
    )
    api = MagicMock()
    api.hf_hub_download.return_value = str(zp)
    with pytest.raises(ValueError, match="already in"):
        K.contribute_model("jane/my-model", "c" * 40, local_dir=model_dir(tmp_path), api=api, loader=stub_loader)


@pytest.mark.parametrize(
    "edit, needle",
    [
        (lambda m: m.pop("training_data"), "training_data"),
        (lambda m: m["val_score"].pop("rmse_fF"), "rmse_fF"),
        (lambda m: m["val_score"].pop("seconds"), "seconds_per_layout"),
        (lambda m: m["val_score"].update(split="test"), "test score"),
    ],
)
def test_model_rejections(tmp_path, edit, needle):
    meta = json.loads(json.dumps(META))
    edit(meta)
    rep, _ = K.validate_model("jane/m", "c" * 40, local_dir=model_dir(tmp_path, meta), loader=stub_loader)
    assert not rep.ok and any(needle in e for e in rep.all_errors()), rep.all_errors()


def test_model_load_failure_and_all_data(tmp_path):
    def broken(root, meta, paths):
        raise RuntimeError("bad checkpoint")

    rep, _ = K.validate_model("jane/m", "c" * 40, local_dir=model_dir(tmp_path), loader=broken)
    assert any("do not load" in e for e in rep.errors)
    meta = json.loads(json.dumps(META))
    meta["training_data"]["split_used"] = "all"
    rep, e = K.validate_model("jane/m", "main", local_dir=model_dir(tmp_path / "b", meta), loader=stub_loader)
    assert rep.ok and not e["test_eligible"] and any("full commit sha" in w for w in rep.warnings)


def test_model_download_path(tmp_path):
    api = MagicMock()
    api.repo_info.return_value = SimpleNamespace(sha="d" * 40)
    api.snapshot_download.return_value = str(model_dir(tmp_path))
    rep, e = K.validate_model("jane/m", "main", api=api, loader=stub_loader)
    assert rep.ok and e["revision"] == "d" * 40
    api.snapshot_download.assert_called_once_with(repo_id="jane/m", revision="d" * 40)
