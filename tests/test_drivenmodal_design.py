from pathlib import Path

from squadds.simulations.drivenmodal.design import create_multiplanar_design, write_qiskit_layer_stack_csv
from squadds.simulations.drivenmodal.models import DrivenModalLayerStackSpec


def test_write_qiskit_layer_stack_csv_writes_qiskit_expected_strings(tmp_path: Path):
    csv_path = write_qiskit_layer_stack_csv(
        DrivenModalLayerStackSpec(substrate_thickness_um=500.0, metal_thickness_um=0.3),
        tmp_path / "layer_stack.csv",
    )

    contents = csv_path.read_text()

    assert "pec" in contents
    assert "silicon" in contents
    assert "true" in contents
    assert "0.3um" in contents
    assert "-500um" in contents


def test_create_multiplanar_design_loads_explicit_layer_stack(tmp_path: Path):
    design, csv_path = create_multiplanar_design(
        layer_stack=DrivenModalLayerStackSpec(substrate_thickness_um=650.0),
        layer_stack_path=tmp_path / "layer_stack.csv",
        enable_renderers=False,
    )

    assert Path(design.ls_filename) == csv_path
    assert design.overwrite_enabled is True
    assert set(design.ls.ls_df["material"]) == {"pec", "silicon"}
    assert "-650um" in set(design.ls.ls_df["thickness"])
