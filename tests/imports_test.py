import importlib

import pytest

MODULES = [
    "squadds",
    "squadds.calcs",
    "squadds.core",
    "squadds.database",
    "squadds.interpolations",
    "squadds.simulations",
    "squadds.simulations.drivenmodal",
    "squadds.simulations.drivenmodal.ports",
    "squadds.simulations.drivenmodal.artifacts",
    "squadds.simulations.drivenmodal.hfss_runner",
    "squadds.simulations.drivenmodal.models",
    "squadds.simulations.drivenmodal.layer_stack",
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


def test_drivenmodal_package_exports():
    from squadds.simulations.drivenmodal import (
        CapacitanceExtractionRequest,
        CoupledSystemDrivenModalRequest,
        DrivenModalLayerStackSpec,
    )

    assert CapacitanceExtractionRequest is not None
    assert CoupledSystemDrivenModalRequest is not None
    assert DrivenModalLayerStackSpec is not None
