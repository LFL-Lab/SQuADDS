"""SQuADDS_DB data sources: the planar capacitance source is on by default, ``sources=("default",)`` is the hub alone, and
an unreachable source falls back to the hub rows with one warning."""

from __future__ import annotations

import io
import os
import warnings
from contextlib import redirect_stdout

import pandas as pd
import pytest

import squadds.core.db as dbmod
import squadds.core.db_catalog as catalog
from squadds.core.db import SQuADDS_DB
from squadds.core.db_sources import PlanarCapacitanceSource, available_sources, merge_config_names
from squadds.core.design_patterns import SingletonMeta
from squadds.core.utils import flatten_df_second_level
from squadds.database.planar_capacitance_view import INTERDIGITAL_CONFIG, VIEW_CONFIGS

HF_CONFIGS = [
    "qubit-TransmonCross-cap_matrix",
    "cavity_claw-RouteMeander-eigenmode",
    "coupler-NCap-cap_matrix",
    "coupler-CapNInterdigitalTee-cap_matrix",
    INTERDIGITAL_CONFIG,
    "measured_device_database",
]


def _hf_rows(config: str) -> pd.DataFrame:
    """Stand-in for ``load_dataset(SQuADDS_DB, config)["train"].to_pandas()``."""
    if config == "qubit-TransmonCross-cap_matrix":
        return pd.DataFrame(
            {
                "sim_results": [
                    {"cross_to_ground": 158.4, "claw_to_ground": 90.9, "cross_to_claw": 3.7, "units": "fF"},
                    {"cross_to_ground": 120.0, "claw_to_ground": 80.0, "cross_to_claw": 5.1, "units": "fF"},
                ],
                "design": [{"design_options": {"cross_length": "300um"}, "design_tool": "qiskit-metal"}] * 2,
                "notes": [{}, {}],
            }
        )
    if config == INTERDIGITAL_CONFIG:
        return pd.DataFrame(
            {
                "sim_results": [{"north_to_south": 0.7, "north_to_ground": 3.0, "units": "fF"}] * 3,
                "design": [{"design_options": {"finger_count": "5"}, "coupler_type": "GeneralizedCapNInterdigital"}]
                * 3,
                "notes": [{"source_id": f"exp6/cap_{i}"} for i in range(3)],
            }
        )
    return pd.DataFrame({"sim_results": [{"value": 1.0, "units": "x"}], "design": [{"design_options": {}}]})


class _Split:
    def __init__(self, df):
        self.df = df

    def to_pandas(self):
        return self.df.copy()


@pytest.fixture
def hf(monkeypatch):
    """Mocked HF hub: config names, dataset rows, and a counter of the cache deletion on construction."""
    calls = {"delete_cache": 0, "load": []}

    def delete_cache():
        calls["delete_cache"] += 1

    def load_dataset(repo, config, **kwargs):
        calls["load"].append(config)
        return {"train": _Split(_hf_rows(config))}

    monkeypatch.setattr(catalog, "delete_HF_cache", delete_cache)
    monkeypatch.setattr(catalog, "get_dataset_config_names", lambda repo, download_mode=None: list(HF_CONFIGS))
    monkeypatch.setattr(dbmod, "load_dataset", load_dataset)
    saved = SingletonMeta._instances.pop(SQuADDS_DB, None)
    yield calls
    SingletonMeta._instances.pop(SQuADDS_DB, None)
    if saved is not None:
        SingletonMeta._instances[SQuADDS_DB] = saved


@pytest.fixture
def mini(tmp_path):
    from test_planar_capacitance_view import write_mini_dataset

    return write_mini_dataset(tmp_path / "pc")


def _printed(fn, *args) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def _legacy_get_dataset(config: str, db) -> pd.DataFrame:
    """The pre-source ``get_dataset`` body: load, set target keys, flatten."""
    df = dbmod.load_dataset(db.repo_name, config, cache_dir=None)["train"].to_pandas()
    return flatten_df_second_level(df)


def test_default_only_construction_is_the_legacy_behaviour(hf, monkeypatch):
    monkeypatch.setattr("squadds.core.db_sources.create_source", lambda *a, **k: pytest.fail("source created"))
    db = SQuADDS_DB(sources=("default",))
    assert hf["delete_cache"] == 1
    assert db.configs == [c for c in HF_CONFIGS if c.count("-") == 2]
    assert db.enabled_sources() == ["default"]
    assert _printed(db.get_configs).strip().startswith("['qubit-TransmonCross-cap_matrix'")
    assert "GeneralizedCapStar" not in db.supported_component_names()
    db.select_system("qubit")
    db.select_qubit("TransmonCross")
    for config in db.configs:
        component, name, data_type = config.split("-")
        got = db.get_dataset(data_type=data_type, component=component, component_name=name)
        pd.testing.assert_frame_equal(got, _legacy_get_dataset(config, db))
    assert SQuADDS_DB() is db and hf["delete_cache"] == 1


def test_planar_capacitance_is_on_by_default(hf, mini, monkeypatch):
    monkeypatch.setenv("SQUADDS_PLANAR_CAP_DATA", str(mini))
    db = SQuADDS_DB()
    default = [c for c in HF_CONFIGS if c.count("-") == 2]
    assert db.enabled_sources() == ["default", "planar_capacitance"]
    assert db.configs[: len(default)] == default and set(VIEW_CONFIGS) <= set(db.configs)
    db.select_system("coupler")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows = db.get_dataset("cap_matrix", "coupler", "GeneralizedCapNInterdigital")
    assert len(rows) == len(_hf_rows(INTERDIGITAL_CONFIG)) + 2
    again = SQuADDS_DB(sources=("default",))
    assert again is db and db.enabled_sources() == ["default"] and db.configs == default


class _Unreachable(PlanarCapacitanceSource):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def load(self, config):
        raise self.error


def _access_errors():
    errors = [OSError("offline"), ConnectionError("Network is unreachable")]
    try:
        import httpx
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

        url = "https://huggingface.co/api/datasets/SQuADDS/SQuADDS_PlanarCapacitance"
        for cls, code in ((GatedRepoError, 401), (RepositoryNotFoundError, 404)):
            response = httpx.Response(code, request=httpx.Request("GET", url))
            errors.append(cls(f"{code} Client Error", response=response))
    except (ImportError, TypeError):
        pass
    return errors


@pytest.mark.parametrize("error", _access_errors(), ids=lambda e: type(e).__name__)
def test_unreachable_source_falls_back_to_the_hub_rows(hf, monkeypatch, error):
    monkeypatch.setattr("squadds.core.db.create_source", lambda name, **options: _Unreachable(error))
    db = SQuADDS_DB()
    assert db.enabled_sources() == ["default", "planar_capacitance"]
    db.select_system("coupler")
    with pytest.warns(RuntimeWarning, match="could not be read") as record:
        got = db.get_dataset("cap_matrix", "coupler", "GeneralizedCapNInterdigital")
    assert len(record) == 1 and "SQUADDS_PLANAR_CAP_DATA" in str(record[0].message)
    pd.testing.assert_frame_equal(got, _legacy_get_dataset(INTERDIGITAL_CONFIG, db))
    assert db.enabled_sources() == ["default"] and db.configs == [c for c in HF_CONFIGS if c.count("-") == 2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = db.get_dataset("cap_matrix", "coupler", "GeneralizedCapNInterdigital")
    pd.testing.assert_frame_equal(again, got)


def test_missing_local_copy_falls_back_to_the_hub_rows(hf, monkeypatch, tmp_path):
    monkeypatch.setenv("SQUADDS_PLANAR_CAP_DATA", str(tmp_path / "missing"))
    db = SQuADDS_DB()
    db.select_system("coupler")
    with pytest.warns(RuntimeWarning, match="FileNotFoundError"):
        got = db.get_dataset("cap_matrix", "coupler", "GeneralizedCapNInterdigital")
    pd.testing.assert_frame_equal(got, _legacy_get_dataset(INTERDIGITAL_CONFIG, db))


def test_sources_extend_configs(hf, mini):
    db = SQuADDS_DB(
        sources=("default", "planar_capacitance"), source_options={"planar_capacitance": {"local_dir": mini}}
    )
    default = [c for c in HF_CONFIGS if c.count("-") == 2]
    assert db.configs[: len(default)] == default
    assert db.configs == merge_config_names(default, db._extra_sources)
    assert set(VIEW_CONFIGS) <= set(db.configs) and db.configs.count(INTERDIGITAL_CONFIG) == 1
    assert hf["delete_cache"] == 1
    assert "Xmon" in db.supported_component_names() and "GeneralizedCapStar" in db.get_component_names("coupler")
    table = _printed(db.view_datasets)
    assert "Xmon" in table and "TransmonPocket" in table
    assert db.enabled_sources() == ["default", "planar_capacitance"]
    assert available_sources() == ["default", "planar_capacitance"]
    db.disable_source("planar_capacitance")
    assert db.configs == default


def test_existing_config_keeps_its_rows_first_and_appends_new_ones(hf, mini):
    plain = SQuADDS_DB(sources=("default",))
    plain.select_system("coupler")
    before = plain.get_dataset("cap_matrix", "coupler", "GeneralizedCapNInterdigital")
    db = SQuADDS_DB(
        sources=("default", "planar_capacitance"), source_options={"planar_capacitance": {"local_dir": mini}}
    )
    assert db is plain
    after = db.get_dataset("cap_matrix", "coupler", "GeneralizedCapNInterdigital")
    assert len(after) == len(before) + 2
    pd.testing.assert_frame_equal(after.iloc[: len(before)][list(before.columns)], before, check_dtype=False)
    new = after.iloc[len(before) :]
    assert list(new.design_id) == ["ref/exp6/q3d_cap_0002", "ref/exp9/q3d_cap_00001"]
    assert new.north_to_south.notna().all() and new.source_id.isna().all()


def test_singleton_applies_sources_given_after_construction(hf, mini):
    db = SQuADDS_DB(sources=("default",))
    assert "qubit-Xmon-cap_matrix" not in db.configs
    again = SQuADDS_DB(sources=("planar_capacitance",), source_options={"planar_capacitance": {"local_dir": mini}})
    assert again is db and "qubit-Xmon-cap_matrix" in db.configs
    db2 = SQuADDS_DB().enable_source("planar_capacitance", local_dir=mini)
    assert db2 is db


def test_new_qubit_config_through_select_and_analyzer(hf, mini, headless_qiskit_environment):
    db = SQuADDS_DB(
        sources=("default", "planar_capacitance"), source_options={"planar_capacitance": {"local_dir": mini}}
    )
    db.select_system("qubit")
    db.select_qubit("Xmon")
    df = db.create_system_df()
    assert len(df) == 2 and {"cross_to_claw", "cross_to_ground", "design_options"} <= set(df.columns)
    assert db.target_param_keys is not None and "cross_to_claw" in db.target_param_keys
    from squadds.core.analysis import Analyzer

    analyzer = Analyzer(db)
    closest = analyzer.find_closest({"qubit_frequency_GHz": 4.5, "anharmonicity_MHz": -200.0}, num_top=1)
    assert len(closest) == 1 and closest.iloc[0]["design_id"].startswith("ref/exp13/")


def test_new_coupler_config_through_select_coupler_and_analyzer(hf, mini):
    db = SQuADDS_DB(
        sources=("default", "planar_capacitance"), source_options={"planar_capacitance": {"local_dir": mini}}
    )
    db.select_system("coupler")
    db.select_coupler("GeneralizedCapStar")
    assert (db.selected_component_name, db.selected_data_type) == ("GeneralizedCapStar", "cap_matrix")
    df = db.create_system_df()
    assert len(df) == 2 and (df.coupler_type == "GeneralizedCapStar").all()
    from squadds.core.analysis import Analyzer

    target = float(df.star_to_pad_0.iloc[1])
    closest = Analyzer(db).find_closest({"star_to_pad_0": target}, num_top=1, display=False)
    assert closest.iloc[0]["design_id"] == df.design_id.iloc[1]


def test_default_select_coupler_leaves_component_name_alone(hf):
    db = SQuADDS_DB(sources=("default",))
    db.select_system("coupler")
    db.select_coupler("NCap")
    assert db.selected_component_name is None and db.selected_data_type is None


def test_source_view_is_cached_and_copied(mini):
    src = PlanarCapacitanceSource(local_dir=mini)
    a = src.load("qubit-Xmon-cap_matrix")
    a.loc[:, "notes"] = None
    assert src.load("qubit-Xmon-cap_matrix").notes.notna().all()


def test_source_uses_the_pinned_dataset_revision(monkeypatch):
    from squadds.ml.capacitance.dataset import DATASET_REVISION, ENV_VAR

    monkeypatch.delenv(ENV_VAR, raising=False)
    assert PlanarCapacitanceSource().db.revision == DATASET_REVISION
    assert PlanarCapacitanceSource(revision="abc123").db.revision == "abc123"


@pytest.mark.integration
def test_default_configs_and_rows_match_the_hub():
    """Networked: with ``sources=("default",)`` every current config of SQuADDS/SQuADDS_DB gives the same configs and first
    rows as a direct ``load_dataset``; with the planar capacitance source enabled (local copy via SQUADDS_PLANAR_CAP_DATA) the default
    rows of the shared config are unchanged and come first."""
    from datasets import get_dataset_config_names, load_dataset

    saved = SingletonMeta._instances.pop(SQuADDS_DB, None)
    try:
        db = SQuADDS_DB(sources=("default",))
        expected = [c for c in get_dataset_config_names(db.repo_name) if c.count("-") == 2]
        assert db.configs == expected
        db.select_system("qubit")
        db.select_qubit("TransmonCross")
        first = {}
        for config in db.configs:
            component, name, data_type = config.split("-")
            got = db.get_dataset(data_type=data_type, component=component, component_name=name)
            ref = flatten_df_second_level(load_dataset(db.repo_name, config)["train"].to_pandas())
            pd.testing.assert_frame_equal(got.head(5), ref.head(5))
            first[config] = got
        if os.environ.get("SQUADDS_PLANAR_CAP_DATA"):
            db.enable_source("planar_capacitance")
            assert db.configs[: len(expected)] == expected
            for config in expected:
                component, name, data_type = config.split("-")
                got = db.get_dataset(data_type=data_type, component=component, component_name=name)
                head = got.iloc[: len(first[config])][list(first[config].columns)]
                pd.testing.assert_frame_equal(head, first[config], check_dtype=False)
    finally:
        SingletonMeta._instances.pop(SQuADDS_DB, None)
        if saved is not None:
            SingletonMeta._instances[SQuADDS_DB] = saved
