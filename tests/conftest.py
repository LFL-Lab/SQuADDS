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
