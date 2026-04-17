from types import SimpleNamespace
from pathlib import Path

from squadds.simulations.drivenmodal.design import (
    connect_renderer_to_new_ansys_design,
    create_multiplanar_design,
    ensure_drivenmodal_setup,
    format_exception_for_console,
    render_drivenmodal_design,
    run_drivenmodal_sweep,
    safe_ansys_design_name,
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


def test_ensure_drivenmodal_setup_activates_and_edits_named_setup():
    class FakeRenderer:
        def __init__(self):
            self.calls = []

        def add_drivenmodal_setup(self, **kwargs):
            self.calls.append(("add", kwargs))
            return "setup"

        def activate_ansys_setup(self, setup_name):
            self.calls.append(("activate", setup_name))

        def edit_drivenmodal_setup(self, setup_args):
            self.calls.append(("edit", dict(setup_args)))

    renderer = FakeRenderer()

    result = ensure_drivenmodal_setup(
        renderer,
        name="DrivenModalSetup",
        freq_ghz=5.0,
        max_passes=20,
    )

    assert result == "setup"
    assert renderer.calls[:2] == [
        ("add", {"name": "DrivenModalSetup", "freq_ghz": 5.0, "max_passes": 20}),
        ("activate", "DrivenModalSetup"),
    ]


def test_run_drivenmodal_sweep_prefers_setup_handle_and_sets_current_sweep():
    class FakeSweep:
        def __init__(self):
            self.analyzed = False

        def analyze_sweep(self):
            self.analyzed = True

    class FakeSetup:
        def __init__(self):
            self.insert_calls = []
            self.sweep = FakeSweep()

        def insert_sweep(self, **kwargs):
            self.insert_calls.append(kwargs)

        def get_sweep(self, name):
            assert name == "DrivenModalSweep"
            return self.sweep

    renderer = SimpleNamespace(current_sweep=None)
    setup = FakeSetup()

    sweep = run_drivenmodal_sweep(
        renderer,
        setup,
        setup_name="DrivenModalSetup",
        start_ghz=1.0,
        stop_ghz=12.0,
        count=221,
        name="DrivenModalSweep",
        type="Fast",
        save_fields=False,
    )

    assert setup.insert_calls == [
        {
            "start_ghz": 1.0,
            "stop_ghz": 12.0,
            "count": 221,
            "name": "DrivenModalSweep",
            "type": "Fast",
            "save_fields": False,
        }
    ]
    assert sweep is setup.sweep
    assert renderer.current_sweep is setup.sweep
    assert setup.sweep.analyzed is True


def test_safe_ansys_design_name_is_short_and_stable():
    first = safe_ansys_design_name("tutorial10-qubit-claw-000-v1")
    second = safe_ansys_design_name("tutorial10-qubit-claw-000-v1")

    assert first == second
    assert first.startswith("dm_")
    assert "-" not in first
    assert len(first) <= 24
