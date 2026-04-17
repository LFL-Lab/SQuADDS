import importlib

import pytest

MODULES = [
    "squadds",
    "squadds.calcs",
    "squadds.core",
    "squadds.database",
    "squadds.interpolations",
    "squadds.core.utils",
    "squadds.core.design_patterns",
    "squadds.core.analysis",
    "squadds.database.utils",
    "squadds.interpolations.interpolator",
    "squadds.calcs.qubit",
    "squadds.calcs.transmon_cross",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
