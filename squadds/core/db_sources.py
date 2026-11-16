"""Additional data sources of ``SQuADDS_DB``.

The default source is the HF dataset ``SQuADDS/SQuADDS_DB``. An extra source contributes configs in the same row schema;
for a config that both hold (``coupler-GeneralizedCapNInterdigital-cap_matrix``) its rows are appended after the default
rows. ``SQuADDS_DB()`` enables ``DEFAULT_SOURCES`` (the hub and the planar capacitance dataset):

    db = SQuADDS_DB()                                    # "default" and "planar_capacitance"
    db = SQuADDS_DB(sources=("default",))               # SQuADDS/SQuADDS_DB only (the behaviour before 1.0)
    db = SQuADDS_DB(source_options={"planar_capacitance": {"revision": "<sha>"}})

``source_options`` go to the source (``revision``, ``local_dir``, ``token``); ``SQUADDS_PLANAR_CAP_DATA=<dir>`` points the
planar capacitance source at a local copy. If an extra source cannot be read (offline, no access, missing local copy),
``SQuADDS_DB`` warns once, disables that source and returns the default rows.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_SOURCE = "default"
DEFAULT_SOURCES = (DEFAULT_SOURCE, "planar_capacitance")


def _access_errors() -> tuple[type[BaseException], ...]:
    """Errors that mean a source cannot be reached or read (network, HTTP 401/403/404, offline mode, missing files)."""
    errors: list[type[BaseException]] = [OSError]
    try:
        from huggingface_hub import errors as hf_errors

        errors += [getattr(hf_errors, n) for n in ("HfHubHTTPError", "EntryNotFoundError") if hasattr(hf_errors, n)]
    except ImportError:
        pass
    try:
        import httpx

        errors.append(httpx.HTTPError)
    except ImportError:
        pass
    return tuple(errors)


SOURCE_ACCESS_ERRORS = _access_errors()


class PlanarCapacitanceSource:
    """SQuADDS view of ``SQuADDS/SQuADDS_PlanarCapacitance`` (see ``squadds.database.planar_capacitance_view``)."""

    name = "planar_capacitance"

    def __init__(self, revision: str | None = None, local_dir=None, token: str | None = None, db=None):
        self.revision, self.local_dir, self.token = revision, local_dir, token
        self._db = db
        self._frames: dict[str, pd.DataFrame] = {}

    def __repr__(self) -> str:
        return f"PlanarCapacitanceSource(revision={self.revision!r}, local_dir={self.local_dir!r})"

    @property
    def db(self):
        if self._db is None:
            from squadds.ml.capacitance.dataset import PlanarCapacitanceDB

            self._db = PlanarCapacitanceDB(revision=self.revision, local_dir=self.local_dir, token=self.token)
        return self._db

    def configs(self) -> list[str]:
        from squadds.database.planar_capacitance_view import VIEW_CONFIGS

        return list(VIEW_CONFIGS)

    def load(self, config: str) -> pd.DataFrame:
        """Rows of ``config`` in the SQuADDS_DB schema (nested dict columns); cached per source instance."""
        if config not in self._frames:
            from squadds.database.planar_capacitance_view import build_view

            self._frames[config] = build_view(self.db, config)
        return self._frames[config].copy()


SOURCES = {PlanarCapacitanceSource.name: PlanarCapacitanceSource}


def available_sources() -> list[str]:
    return [DEFAULT_SOURCE, *SOURCES]


def create_source(name: str, **options):
    if name not in SOURCES:
        raise ValueError(f"Unknown SQuADDS_DB source {name!r}. Available sources: {available_sources()}")
    return SOURCES[name](**options)


def merge_config_names(default_configs: list[str], sources: dict) -> list[str]:
    """Default configs first (unchanged order), then the configs only an extra source has, in source order."""
    merged = list(default_configs)
    for src in sources.values():
        merged += [c for c in src.configs() if c not in merged]
    return merged


def sources_owning(config: str, sources: dict) -> list:
    return [src for src in sources.values() if config in src.configs()]
