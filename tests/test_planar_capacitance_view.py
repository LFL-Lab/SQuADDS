"""PlanarCapacitanceDB and the SQuADDS_DB view of the planar capacitance dataset (CI-safe: a tiny synthetic dataset)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from squadds.database.generalized_ncap_dataset import _pairwise_results
from squadds.database.planar_capacitance_view import (
    INTERDIGITAL_CONFIG,
    NOT_REPRODUCIBLE,
    VIEW_CONFIGS,
    build_view,
    sim_results_from_matrix,
)
from squadds.datasets.planar_capacitance_schema import EDGE_FEATURES, GLOBAL_FEATURES, NODE_FEATURES
from squadds.ml.capacitance.dataset import (
    DATASET_REVISION,
    ENV_VAR,
    NEAREST_FEATURES,
    PlanarCapacitanceDB,
    long_to_matrix,
)

XMON_TWO_CLAWS = {
    "cross_length": "275.972um",
    "cross_width": "22.373um",
    "cross_gap": "36.448um",
    "connection_pads": {
        "readout": {
            "connector_location": "90",
            "connector_type": "0",
            "claw_length": "199.628um",
            "claw_width": "15.441um",
            "claw_gap": "6.295um",
            "ground_spacing": "6.849um",
            "claw_cpw_length": "49.099um",
            "claw_cpw_width": "10um",
        },
        "bus": {
            "connector_location": "270",
            "connector_type": "0",
            "claw_length": "197.762um",
            "claw_width": "15.441um",
            "claw_gap": "6.295um",
            "ground_spacing": "6.849um",
            "claw_cpw_length": "49.099um",
            "claw_cpw_width": "10um",
        },
    },
}
SETUP = {
    "ansys_gds": {"q3d_signal": {"gds_layers": "1", "setup": {"AdaptiveFreq": "13GHz", "Cap": {"PerError": 0.05}}}},
    "mesh": {"q3d_signal": {}},
    "layer_stack": {"metal_thickness_um": 0.25, "substrate_thickness_um": 350, "vacuum_box_size_z_um": 1000},
}


def maxwell(n: int, seed: int) -> np.ndarray:
    """A physical Maxwell matrix (ground first): negative mutuals, diagonal = sum of mutuals (closure)."""
    rng = np.random.default_rng(seed)
    M = rng.uniform(0.5, 20.0, (n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    return np.diag(M.sum(1)) - M


# design_id, collection, config, class, split, nets, geom_hash, squadds_db_ref, options
DESIGNS = [
    ("ref/exp6/q3d_cap_0001", "exp6", INTERDIGITAL_CONFIG, "GeneralizedCapNInterdigital", "train", "GNS", "h1", "ref"),
    ("ref/exp6/q3d_cap_0002", "exp6", INTERDIGITAL_CONFIG, "GeneralizedCapNInterdigital", "train", "GNS", "h2", None),
    ("ref/exp9/q3d_cap_00001", "exp9", INTERDIGITAL_CONFIG, "GeneralizedCapNInterdigital", "val", "GNS", "h3", None),
    ("ref/exp9/q3d_cap_00002", "exp9", INTERDIGITAL_CONFIG, "GeneralizedCapNInterdigital", "train", "GNS", "h3", None),
    ("ref/exp9/q3d_cap_00003", "exp9", INTERDIGITAL_CONFIG, "GeneralizedCapNInterdigital", "test", "GNS", "h5", None),
    ("ref/exp13/q3d_cap_0000", "exp13", "qubit-Xmon-cap_matrix", "Xmon", "train", ["G", "N", "S", "P1"], "h6", None),
    ("ref/exp13/q3d_cap_0001", "exp13", "qubit-Xmon-cap_matrix", "Xmon", "train", "GNS", "h7", None),
    (
        "ref/exp11/q3d_cap_1330",
        "exp11",
        "coupler-GeneralizedCapStar-cap_matrix",
        "GeneralizedCapStar",
        "train",
        ["G", "N", "S"],
        "h8",
        None,
    ),
    (
        "ref/exp12/q3d_cap_0014",
        "exp12",
        "coupler-GeneralizedCapStar-cap_matrix",
        "GeneralizedCapStar",
        "train",
        ["G", "S", "N", "P2"],
        "h9",
        None,
    ),
    ("ref/synth/0001", "synth_q3d", None, None, "train", "GNS", "h10", None),
]


def _params(collection: str, cls: str | None) -> dict:
    if cls == "Xmon":
        opts = XMON_TWO_CLAWS
    elif cls == "GeneralizedCapStar":
        opts = {"n_points": "5", "pad_enables": "1,0,1,0,0" if collection == "exp12" else "1,0,0,0,0", "gap": "15um"}
    else:
        opts = {"finger_count": "5", "north_cpw_width": "10um"}
    return {
        "cap_instance_name": "cap",
        "margin_factor_ew": 20,
        "margin_factor_ns": 20,
        "cap_options": opts,
        "gds_north_port_length": "2um",
        "gds_north_port_layer": "2",
        "gds_south_port_length": "2um",
        "gds_south_port_layer": "3",
    }


def write_mini_dataset(root: Path) -> Path:
    """A tiny copy of the dataset layout (raw and processed tables) under ``root``."""
    designs, layouts, sims, labels, nodes, edges, glob, coarse = [], [], [], [], [], [], [], []
    for k, (did, col, cfg, cls, split, nets, gh, ref) in enumerate(DESIGNS):
        nets = list(nets)
        designs.append(
            dict(
                design_id=did,
                layout_id=f"sha256:L{k}",
                geom_hash=gh,
                collection=col,
                family=col,
                paper_family={"exp9": "paddle", "exp13": "Xmon"}.get(col, "interdigital"),
                model_family=col,
                params=json.dumps(_params(col, cls)),
                component_class=cls,
                squadds_config=cfg,
                squadds_db_ref=json.dumps({"repo": "SQuADDS/SQuADDS_DB", "row_index": 1}) if ref else None,
                n_conductors=float(len(nets)),
                net_order=nets,
                split=split,
                labelled=True,
                contributor_id="sadman_shanto" if col == "exp13" else "saikat_das",
                licence_id="CC-BY-4.0",
            )
        )
        name = did.split("/")[-1]
        sub = f"{int(name.rsplit('_', 1)[1]) // 5000}/" if col in ("exp8", "exp9") else ""
        layouts.append(dict(layout_id=f"sha256:L{k}", path=f"gds/{col}/{sub}{name}.gds", geom_hash=gh))
        sid = f"{did}#q3d"
        sims.append(
            dict(
                simulation_id=sid,
                design_id=did,
                role="reference",
                solver="Ansys Q3D Extractor",
                substrate_material="cryo_silicon",
                eps_r=11.45,
                substrate_um=350.0,
                tan_delta=1e-7,
                metal_model="thin_sheet",
                metal_t_um=np.nan,
                setup=json.dumps(SETUP),
                passes_completed=10.0,
                qc_convergence="converged",
                qc_ok=True,
                split=split,
            )
        )
        C = maxwell(len(nets), k)
        rows = [
            (sid, did, nets[i], nets[j], i, j, C[i, j], split) for i in range(len(nets)) for j in range(i, len(nets))
        ]
        if split != "test":
            labels += rows
        if did == "ref/exp13/q3d_cap_0000":
            rid = f"{did}#repro"
            sims.append({**sims[-1], "simulation_id": rid, "role": "reproducibility"})
            labels += [(rid, did, a, b, i, j, v + 1.0, split) for (_, _, a, b, i, j, v, _) in rows]
        rng = np.random.default_rng(100 + k)
        for i, net in enumerate(nets):
            nodes.append(
                dict(design_id=did, node_idx=i, net=net, split=split, **{f: float(rng.normal()) for f in NODE_FEATURES})
            )
            for j in range(i + 1, len(nets)):
                edges.append(
                    dict(
                        design_id=did,
                        i=i,
                        j=j,
                        net_a=net,
                        net_b=nets[j],
                        split=split,
                        **{f: float(rng.normal()) for f in EDGE_FEATURES},
                    )
                )
                coarse.append(
                    dict(
                        design_id=did,
                        net_a=net,
                        net_b=nets[j],
                        i=i,
                        j=j,
                        ppg=0.375,
                        mutual_fF=-0.9 * C[i, j],
                        C_fF=0.9 * C[i, j],
                        split=split,
                    )
                )
        glob.append(
            dict(
                design_id=did, n_conductors=len(nets), split=split, **{f: float(rng.normal()) for f in GLOBAL_FEATURES}
            )
        )

    lab = pd.DataFrame(
        labels, columns=["simulation_id", "design_id", "net_a", "net_b", "a_idx", "b_idx", "C_fF", "split"]
    )
    out = {
        "designs/designs-train.parquet": pd.DataFrame(d for d in designs if d["split"] != "test"),
        "designs/designs-test.parquet": pd.DataFrame(d for d in designs if d["split"] == "test"),
        "layouts/layouts.parquet": pd.DataFrame(layouts),
        "simulations/simulations-train.parquet": pd.DataFrame(sims),
        "labels/labels-train.parquet": lab[lab.split == "train"],
        "labels/labels-val.parquet": lab[lab.split == "val"],
        "geometric_embedding/nodes-train.parquet": pd.DataFrame(nodes),
        "geometric_embedding/edges-train.parquet": pd.DataFrame(edges),
        "geometric_embedding/globals-train.parquet": pd.DataFrame(glob),
        "physics_prior/physics_prior-train.parquet": pd.DataFrame(coarse),
    }
    for rel, df in out.items():
        p = root / "data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    for rec in layouts:
        p = root / rec["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"GDS")
    return root


@pytest.fixture
def mini(tmp_path):
    return write_mini_dataset(tmp_path / "SQuADDS_PlanarCapacitance")


@pytest.fixture
def pdb(mini):
    return PlanarCapacitanceDB(local_dir=mini)


# ------------------------------------------------------------------------------------------------ PlanarCapacitanceDB
def test_tables_and_filters(pdb):
    assert {"designs", "labels", "simulations", "geometric_embedding_nodes", "physics_prior"} <= set(pdb.tables())
    assert "params" not in pdb.designs().columns
    assert set(pdb.designs(split="val").design_id) == {"ref/exp9/q3d_cap_00001"}
    assert set(pdb.designs(family="paddle").design_id) == {f"ref/exp9/q3d_cap_0000{i}" for i in (1, 2, 3)}
    assert set(pdb.designs(n_conductors=4).design_id) == {"ref/exp13/q3d_cap_0000", "ref/exp12/q3d_cap_0014"}
    assert pdb.params(["ref/exp13/q3d_cap_0000"])["ref/exp13/q3d_cap_0000"]["cap_options"] == XMON_TWO_CLAWS


def test_labels_are_named_maxwell_matrices_ground_first(pdb):
    ids = ["ref/exp13/q3d_cap_0000", "ref/exp12/q3d_cap_0014", "ref/exp9/q3d_cap_00003"]
    mats = pdb.labels(ids)
    assert set(mats) == set(ids[:2])  # the test design has no public label
    M = mats["ref/exp12/q3d_cap_0014"]
    assert list(M.index) == list(M.columns) == ["G", "S", "N", "P2"]
    np.testing.assert_allclose(M.to_numpy(), maxwell(4, 8))
    every = pdb.labels(ids[:1], role=None)
    assert set(every) == {"ref/exp13/q3d_cap_0000#q3d", "ref/exp13/q3d_cap_0000#repro"}


def test_prior_without_diagonal_uses_closure(pdb):
    P = pdb.physics_prior(["ref/exp6/q3d_cap_0002"])["ref/exp6/q3d_cap_0002"]
    np.testing.assert_allclose(P.to_numpy(), 0.9 * maxwell(3, 1))


def test_long_to_matrix_keeps_stored_diagonal():
    df = pd.DataFrame(
        {
            "a_idx": [0, 0, 1],
            "b_idx": [0, 1, 1],
            "net_a": ["G", "G", "N"],
            "net_b": ["G", "N", "N"],
            "C_fF": [5.0, -2.0, 7.0],
        }
    )
    np.testing.assert_allclose(long_to_matrix(df, "C_fF").to_numpy(), [[5, -2], [-2, 7]])


def test_embedding_nearest_and_gds_path(pdb, mini):
    emb = pdb.embedding(["ref/exp13/q3d_cap_0000"])["ref/exp13/q3d_cap_0000"]
    assert list(emb["nodes"].net) == ["G", "N", "S", "P1"] and len(emb["edges"]) == 6
    assert set(GLOBAL_FEATURES) <= set(emb["globals"])
    P = pdb.pooled_embedding(design_ids=["ref/exp13/q3d_cap_0000", "ref/exp9/q3d_cap_00001"])
    assert list(P.columns) == NEAREST_FEATURES and P.shape == (2, 127)
    near = pdb.nearest("ref/exp9/q3d_cap_00001", k=2)
    assert len(near) == 2 and "ref/exp9/q3d_cap_00001" not in set(near.design_id)
    assert set(near.design_id) <= {f"ref/exp9/q3d_cap_0000{i}" for i in (2, 3)}
    assert near.distance.is_monotonic_increasing
    by_vec = pdb.nearest(P.loc["ref/exp13/q3d_cap_0000"].to_numpy(), k=1)
    assert by_vec.design_id.iloc[0] == "ref/exp13/q3d_cap_0000"
    assert pdb.gds_path("ref/exp6/q3d_cap_0002") == mini / "gds/exp6/q3d_cap_0002.gds"


def test_gds_path_reads_the_stored_bucketed_path(pdb, mini):
    p = pdb.gds_path("ref/exp9/q3d_cap_00001")
    assert p == mini / "gds/exp9/0/q3d_cap_00001.gds" and p.is_file()


def test_build_gds_relpath_buckets_exp8_exp9():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pc_build_common", Path(__file__).parents[1] / "scripts/planar_capacitance/build_dataset/common.py"
    )
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    assert common.gds_relpath("ref/exp8/q3d_cap_12345") == "gds/exp8/2/q3d_cap_12345.gds"
    assert common.gds_relpath("ref/exp9/q3d_cap_04999") == "gds/exp9/0/q3d_cap_04999.gds"
    assert common.gds_relpath("ref/exp6/q3d_cap_0002") == "gds/exp6/q3d_cap_0002.gds"
    assert common.gds_relpath("ref/multistack/exp8_q3d_cap_06211") == "gds/multistack/exp8_q3d_cap_06211.gds"


def test_env_override_and_hf_download(mini, monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(mini))
    assert PlanarCapacitanceDB().local_dir == mini
    monkeypatch.delenv(ENV_VAR)
    import huggingface_hub

    rel = [p.relative_to(mini).as_posix() for p in mini.rglob("*") if p.is_file()]
    calls = []

    class Api:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type, revision):
            calls.append(("list", repo_id, revision))
            return rel

    def download(repo_id, filename, repo_type, revision, token):
        calls.append(("get", filename, revision))
        return str(mini / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    db = PlanarCapacitanceDB(revision="abc123")
    assert len(db.designs(split="val")) == 1
    assert db.gds_path("ref/exp6/q3d_cap_0002") == mini / "gds/exp6/q3d_cap_0002.gds"
    assert {c[-1] for c in calls} == {"abc123"}
    assert ("get", "gds/exp6/q3d_cap_0002.gds", "abc123") in calls
    assert db.gds_path("ref/exp9/q3d_cap_00002") == mini / "gds/exp9/0/q3d_cap_00002.gds"
    assert ("get", "gds/exp9/0/q3d_cap_00002.gds", "abc123") in calls
    assert PlanarCapacitanceDB().revision == PlanarCapacitanceDB(revision=None).revision == DATASET_REVISION


# ------------------------------------------------------------------------------------------------ the SQuADDS view
def test_config_names_follow_the_two_dash_rule():
    assert all(c.count("-") == 2 for c in VIEW_CONFIGS)
    assert all(c.endswith("-cap_matrix") for c in VIEW_CONFIGS)


def test_interdigital_sim_results_match_the_existing_converter():
    C = maxwell(3, 3)
    ref = _pairwise_results({f"C_{a}_{b}": C[i, j] for i, a in enumerate("GNS") for j, b in enumerate("GNS")})
    ours = sim_results_from_matrix(["G", "N", "S"], C, "GeneralizedCapNInterdigital")
    assert list(ours) == list(ref)
    assert list(ours["maxwell_matrix"]) == list(ref["maxwell_matrix"])
    for k, v in ref.items():
        if k == "maxwell_matrix":
            assert ours[k] == pytest.approx(v)
        else:
            assert ours[k] == (v if isinstance(v, str) else pytest.approx(v))


def test_xmon_keys_are_transmon_cross_keys():
    sr = sim_results_from_matrix(["G", "N", "S"], maxwell(3, 4), "Xmon")
    legacy = {
        "cross_to_ground",
        "claw_to_ground",
        "cross_to_claw",
        "cross_to_cross",
        "claw_to_claw",
        "ground_to_ground",
    }
    assert legacy <= set(sr) and sr["cross_to_claw"] > 0
    assert sr["maxwell_matrix"]["C_S_N"] < 0


def test_interdigital_view_appends_only_new_unique_public_designs(pdb):
    v = build_view(pdb, INTERDIGITAL_CONFIG)
    assert [n["design_id"] for n in v.notes] == ["ref/exp6/q3d_cap_0002", "ref/exp9/q3d_cap_00001"]
    exp6, exp9 = v.iloc[0], v.iloc[1]
    assert exp6.design["design_options"]["spine_ledge_rule"] == "tangent"
    assert "spine_ledge_rule" not in exp9.design["design_options"]
    assert exp6.design["component_class"] == exp6.design["coupler_type"] == "GeneralizedCapNInterdigital"
    assert exp6.design["pin_aliases"] == {"top": "north_end", "bottom": "south_end"}
    assert exp6.sim_options["simulator"] == "Ansys Q3D" and exp6.sim_options["sim_type"] == "cap_matrix"
    assert exp6.sim_options["setup"] == SETUP["ansys_gds"]["q3d_signal"]["setup"]
    assert exp6.sim_options["layer_stack"]["eps_r"] == 11.45
    assert exp6.sim_options["layer_stack"]["substrate_thickness_um"] == 350.0
    assert exp6.contributor["uploader"] == "Saikat Das"
    assert exp6.notes["geometry_reproducible"] and exp6.notes["split"] == "train"
    assert exp6.sim_results["north_to_south"] == pytest.approx(-maxwell(3, 1)[1, 2])


def test_star_view_flags_non_reproducible_and_pads_keys(pdb):
    v = build_view(pdb, "coupler-GeneralizedCapStar-cap_matrix")
    notes = {n["design_id"]: n for n in v.notes}
    assert "ref/exp11/q3d_cap_1330" in NOT_REPRODUCIBLE
    assert not notes["ref/exp11/q3d_cap_1330"]["geometry_reproducible"]
    assert not notes["ref/exp12/q3d_cap_0014"]["geometry_reproducible"]
    keys = [list(s) for s in v.sim_results]
    assert keys[0] == keys[1] and "pad_2_to_ground" in keys[0]
    single = v.sim_results.iloc[0]
    assert single["pad_2_to_ground"] is None and single["star_to_pad_0"] > 0
    assert notes["ref/exp12/q3d_cap_0014"]["net_names"] == {"G": "ground", "S": "star", "N": "pad_0", "P2": "pad_2"}


def test_xmon_view_rebuilds_with_the_component_registry(pdb, headless_qiskit_environment):
    v = build_view(pdb, "qubit-Xmon-cap_matrix")
    two = v[[n["design_id"] == "ref/exp13/q3d_cap_0000" for n in v.notes]].iloc[0]
    assert two.sim_results["cross_to_bus_claw"] > 0 and two.contributor["uploader"] == "Sadman Ahmed Shanto"
    assert build_view(pdb, "coupler-ShieldedCapConcentric-cap_matrix").empty
    from qiskit_metal import designs

    from squadds.components import build_component_from_design

    comp = build_component_from_design(designs.DesignPlanar({}, overwrite_enabled=True), two.to_dict(), name="xmon")
    assert set(comp.options.connection_pads) == {"readout", "bus"}


def test_view_modules_do_not_import_ml_dependencies():
    code = (
        "import sys; import squadds; before = set(sys.modules); "
        "import squadds.ml.capacitance.dataset, squadds.database.planar_capacitance_view, squadds.core.db_sources; "
        "new = set(sys.modules) - before; "
        "bad = sorted(m for m in new if m.split('.')[0] in ('torch', 'gdstk', 'numba', 'threadpoolctl')); "
        "assert not bad, bad"
    )
    env = {**os.environ, "QISKIT_METAL_HEADLESS": "1", "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_star_export_styles_and_flags():
    from squadds.database import planar_capacitance_view as V

    assert V.export_style("exp12", "ref/exp12/q3d_cap_0014") == "reexport_gdspy"
    assert V.export_style("exp11", "ref/exp11/q3d_cap_0004") == "reexport"
    assert V.export_style("exp11", "ref/exp11/q3d_cap_0001") == "original"
    assert V.export_style("exp10", "ref/exp10/q3d_cap_0001") is None
    assert "ref/exp11/q3d_cap_0326" in V.NOT_REPRODUCIBLE and "ref/exp11/q3d_cap_1330" in V.NOT_REPRODUCIBLE
    assert len(V.EXP11_REEXPORT) == 154


def test_ground_gap_for_generator_families():
    from squadds.database import planar_capacitance_view as V

    xmon = {"generator": {"params": {"cross_gap": 30.123456, "ground_spacing": 9.0}}}
    assert V._ground_gap_um("Xmon", xmon) == 30.123456
    assert V._ground_gap_um("GeneralizedCapStar", {"cap_options": {}}) is None
