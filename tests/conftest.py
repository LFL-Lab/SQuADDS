import os

import matplotlib as mpl
import pytest


def configure_headless_matplotlib_and_qiskit_metal():
    if os.environ.get("DISPLAY", "") == "":
        mpl.use("Agg")
        os.environ.setdefault("QISKIT_METAL_HEADLESS", "1")
    else:
        os.environ.pop("QISKIT_METAL_HEADLESS", None)


@pytest.fixture(scope="session")
def headless_qiskit_environment():
    configure_headless_matplotlib_and_qiskit_metal()


ARTIFACTS = os.environ.get("SQUADDS_ML_ARTIFACTS", "")


@pytest.fixture(scope="session")
def released_capml_model():
    """``load_model(name)`` for the released capacitance models: the local bundle under ``SQUADDS_ML_ARTIFACTS/models`` when
    that variable is set, else the Hugging Face Hub release. A repo that cannot be read skips the test, or fails it when
    ``SQUADDS_REQUIRE_HUB_MODELS=1`` (the ml-integration CI job)."""
    pytest.importorskip("torch")
    from pathlib import Path

    from squadds.ml.capacitance import load_model
    from squadds.ml.capacitance.models import ENV_MODELS

    local = Path(ARTIFACTS) / "models" if ARTIFACTS else None
    if local is not None and local.is_dir() and not os.environ.get(ENV_MODELS):
        os.environ[ENV_MODELS] = str(local)
    cache = {}

    def load(name):
        if name not in cache:
            try:
                cache[name] = load_model(name)
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403, 404) and os.environ.get("SQUADDS_REQUIRE_HUB_MODELS") != "1":
                    pytest.skip(f"cannot read the {name} model from the Hub ({status})")
                raise
        return cache[name]

    return load
