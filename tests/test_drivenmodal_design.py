from types import SimpleNamespace
from pathlib import Path

from squadds.simulations.drivenmodal.design import (
    connect_renderer_to_new_ansys_design,
    create_multiplanar_design,
    format_exception_for_console,
    render_drivenmodal_design,
    write_qiskit_layer_stack_csv,
)
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
    assert design.get_chip_z("main") == "0.0mm"
    assert design.chips["main"]["material"] == "silicon"
    assert design.chips["main"]["layer_start"] == "0"
    assert design.chips["main"]["layer_end"] == "2048"
    assert design.chips["main"]["size"]["size_z"] == "-650um"
    assert design.chips["main"]["size"]["sample_holder_top"] == "790um"
    assert design.chips["main"]["size"]["sample_holder_bottom"] == "1550um"


def test_connect_renderer_to_new_ansys_design_skips_eager_setup_lookup():
    class FakePInfo:
        def __init__(self):
            self.connected_designs = []

        def connect_design(self, design_name):
            self.connected_designs.append(design_name)

    class FakeRenderer:
        def __init__(self):
            self.calls = []
            self.pinfo = FakePInfo()

        def new_ansys_design(self, design_name, solution_type, connect=True):
            self.calls.append((design_name, solution_type, connect))
            return SimpleNamespace(name=design_name)

    renderer = FakeRenderer()

    ansys_design = connect_renderer_to_new_ansys_design(renderer, "demo_dm")

    assert ansys_design.name == "demo_dm"
    assert renderer.calls == [("demo_dm", "drivenmodal", False)]
    assert renderer.pinfo.connected_designs == ["demo_dm"]


def test_format_exception_for_console_escapes_non_ascii_characters():
    message = format_exception_for_console(Exception("bad setup \U0001f914"))

    assert message == "bad setup \\U0001f914"


def test_render_drivenmodal_design_normalizes_open_pins_for_ports():
    class FakeRenderer:
        def __init__(self):
            self.kwargs = None

        def render_design(self, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    renderer = FakeRenderer()

    result = render_drivenmodal_design(
        renderer,
        selection=["xmon"],
        port_list=[("xmon", "readout", 50)],
        jj_to_port=[("xmon", "rect_jj", 50, True)],
        box_plus_buffer=False,
    )

    assert result == "rendered"
    assert renderer.kwargs["open_pins"] == []
    assert renderer.kwargs["port_list"] == [("xmon", "readout", 50)]
    assert renderer.kwargs["jj_to_port"] == [("xmon", "rect_jj", 50, True)]
    assert renderer.kwargs["box_plus_buffer"] is False
