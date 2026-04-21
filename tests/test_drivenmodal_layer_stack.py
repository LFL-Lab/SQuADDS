from squadds.simulations.drivenmodal.layer_stack import build_layer_stack_dataframe, resolve_layer_stack
from squadds.simulations.drivenmodal.models import DrivenModalLayerStackSpec


def test_resolve_layer_stack_uses_fixed_materials():
    rows = resolve_layer_stack(DrivenModalLayerStackSpec())
    materials = {row["material"] for row in rows}

    assert materials == {"pec", "silicon"}


def test_resolve_layer_stack_applies_thickness_overrides():
    rows = resolve_layer_stack(
        DrivenModalLayerStackSpec(substrate_thickness_um=500.0, metal_thickness_um=0.3),
    )

    assert any(row["thickness"] == "0.3um" for row in rows if row["material"] == "pec")
    assert any(row["thickness"] == "-500um" for row in rows if row["material"] == "silicon")


def test_build_layer_stack_dataframe_preserves_expected_columns():
    frame = build_layer_stack_dataframe(DrivenModalLayerStackSpec())

    assert list(frame.columns) == [
        "chip_name",
        "layer",
        "datatype",
        "material",
        "thickness",
        "z_coord",
        "fill",
    ]
    assert len(frame) == 2
