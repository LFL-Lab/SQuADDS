from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from shapely.geometry import LineString

from squadds.simulations.drivenmodal.design import (
    apply_buffered_chip_bounds,
    apply_cryo_silicon_material_properties,
    connect_renderer_to_new_ansys_design,
    create_multiplanar_design,
    ensure_drivenmodal_setup,
    ensure_perfect_e_boundary,
    format_exception_for_console,
    render_drivenmodal_design,
    run_drivenmodal_sweep,
    safe_ansys_design_name,
    save_renderer_project,
    simplify_collinear_path_points,
    snapshot_boundary_assignments,
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


def test_save_renderer_project_persists_current_project_state(tmp_path: Path):
    saved_paths = []

    class FakeProject:
        def save(self, path):
            saved_paths.append(path)

    renderer = SimpleNamespace(pinfo=SimpleNamespace(project=FakeProject()))
    project_path = tmp_path / "demo.aedt"

    returned_path = save_renderer_project(renderer, project_path)

    assert returned_path == project_path
    assert saved_paths == [str(project_path)]


def test_apply_cryo_silicon_material_properties_updates_active_hfss_material():
    calls = []

    class FakeSilicon:
        def __init__(self):
            self.permittivity = 11.9
            self.dielectric_loss_tangent = 1e-3

    silicon = FakeSilicon()

    class FakeMaterials:
        def exists_material(self, material_name):
            calls.append(("exists", material_name))
            return False

        def checkifmaterialexists(self, material_name):
            calls.append(("check", material_name))
            return silicon

    class FakeHfss:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.materials = FakeMaterials()

        def release_desktop(self, close_projects=False, close_desktop=False):
            calls.append(("release", close_projects, close_desktop))

    renderer = SimpleNamespace(pinfo=SimpleNamespace(project_name="Project1", design_name="dm_test"))

    result = apply_cryo_silicon_material_properties(renderer, hfss_factory=FakeHfss)

    assert result == {
        "material": "silicon",
        "permittivity": 11.45,
        "dielectric_loss_tangent": 1e-07,
        "project_name": "Project1",
        "design_name": "dm_test",
    }
    assert silicon.permittivity == 11.45
    assert silicon.dielectric_loss_tangent == 1e-7
    assert calls == [
        (
            "init",
            {
                "project": "Project1",
                "design": "dm_test",
                "solution_type": "DrivenModal",
                "new_desktop": False,
                "close_on_exit": False,
            },
        ),
        ("exists", "silicon"),
        ("check", "silicon"),
        ("release", False, False),
    ]


def test_apply_cryo_silicon_material_properties_prefers_live_renderer_session():
    silicon = SimpleNamespace(permittivity=11.9, dielectric_loss_tangent=1e-3)
    calls = []

    class FakeDefinitionManager:
        def GetManager(self, name):
            calls.append(("manager", name))
            return object()

        def GetProjectMaterialNames(self):
            calls.append(("names",))
            return ["silicon"]

    class FakeProject:
        def GetDefinitionManager(self):
            calls.append(("definition_manager",))
            return FakeDefinitionManager()

    class FakeMaterials:
        def __init__(self, app):
            calls.append(("materials_init", bool(app.odesktop), bool(app.oproject), bool(app.odesign)))

        def exists_material(self, material_name):
            calls.append(("exists", material_name))
            return silicon

    renderer = SimpleNamespace(
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        pinfo=SimpleNamespace(
            project_name="Project1",
            design_name="dm_test",
            desktop=SimpleNamespace(_desktop=object()),
            project=SimpleNamespace(_project=FakeProject(), parent=SimpleNamespace(_desktop=object())),
            design=SimpleNamespace(_design=object()),
        ),
    )

    result = apply_cryo_silicon_material_properties(
        renderer,
        hfss_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
        materials_factory=FakeMaterials,
    )

    assert result == {
        "material": "silicon",
        "permittivity": 11.45,
        "dielectric_loss_tangent": 1e-07,
        "project_name": "Project1",
        "design_name": "dm_test",
    }
    assert silicon.permittivity == 11.45
    assert silicon.dielectric_loss_tangent == 1e-7
    assert calls == [
        ("materials_init", True, True, True),
        ("exists", "silicon"),
    ]


def test_ensure_perfect_e_boundary_prefers_pinfo_design_api():
    calls = []

    class FakeModeler:
        def assign_perfect_E(self, object_names, name):
            calls.append(("modeler", name, list(object_names)))

    class FakeDesign:
        def append_PerfE_assignment(self, boundary_name, object_names):
            calls.append(("design", boundary_name, list(object_names)))

    renderer = SimpleNamespace(pinfo=SimpleNamespace(design=FakeDesign()), modeler=FakeModeler())

    assigned = ensure_perfect_e_boundary(renderer, ["trace_a", "trace_b", "trace_a"])

    assert assigned == ["trace_a", "trace_b"]
    assert calls == [("design", "PerfE_Metal", ["trace_a", "trace_b"])]


def test_snapshot_boundary_assignments_reads_hfss_boundary_module():
    class FakeBoundaries:
        def GetBoundaries(self):
            return ["PerfE_Metal", "LumpPort_feedline"]

    class FakeDesign:
        _boundaries = FakeBoundaries()

        def get_boundary_assignment(self, boundary_name):
            if boundary_name == "PerfE_Metal":
                return ["trace_b", "trace_a", "trace_a"]
            return ["Port_feedline"]

    renderer = SimpleNamespace(pinfo=SimpleNamespace(design=FakeDesign()))

    snapshot = snapshot_boundary_assignments(renderer)

    assert snapshot == {
        "PerfE_Metal": ["trace_a", "trace_b"],
        "LumpPort_feedline": ["Port_feedline"],
    }


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


def test_apply_buffered_chip_bounds_sets_explicit_chip_size():
    class FakeComponent:
        def __init__(self, bounds):
            self._bounds = bounds

        def qgeometry_bounds(self):
            return self._bounds

    class FakeDesign:
        def __init__(self):
            self.components = {
                "xmon": FakeComponent((-1.0, -0.5, 1.0, 0.5)),
                "claw": FakeComponent((1.0, -0.25, 2.0, 0.25)),
            }
            self.chips = {"main": {"size": {}}}

    design = FakeDesign()

    chip_box = apply_buffered_chip_bounds(
        design,
        selection=["xmon", "claw"],
        x_buffer_mm=0.2,
        y_buffer_mm=0.2,
    )

    assert chip_box["min_x_mm"] == -1.0
    assert chip_box["max_x_mm"] == 2.0
    assert chip_box["buffered_size_x_mm"] == 3.4
    assert chip_box["buffered_size_y_mm"] == 1.4
    assert design.chips["main"]["size"]["size_x"] == "3.4mm"
    assert design.chips["main"]["size"]["size_y"] == "1.4mm"
    assert design.chips["main"]["size"]["center_x"] == "0.5mm"
    assert design.chips["main"]["size"]["center_y"] == "0mm"


def test_simplify_collinear_path_points_collapses_redundant_vertices():
    class FakeDesign:
        def __init__(self):
            self.qgeometry = SimpleNamespace(
                tables={
                    "path": pd.DataFrame(
                        [
                            {
                                "geometry": LineString(
                                    [
                                        (0.0, 2.1),
                                        (0.0, 2.10585),
                                        (0.0, -1.90585),
                                        (0.0, -1.9),
                                    ]
                                )
                            },
                            {
                                "geometry": LineString(
                                    [
                                        (0.0, 0.0),
                                        (1.0, 0.0),
                                        (1.0, 1.0),
                                    ]
                                )
                            },
                        ]
                    )
                }
            )

    design = FakeDesign()

    simplified_count = simplify_collinear_path_points(design)

    assert simplified_count == 1
    assert list(design.qgeometry.tables["path"].iloc[0]["geometry"].coords) == [(0.0, 2.1), (0.0, -1.9)]
    assert list(design.qgeometry.tables["path"].iloc[1]["geometry"].coords) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]


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


def test_run_drivenmodal_sweep_supports_interpolating_sweep_advanced_options():
    class FakeSweep:
        def __init__(self):
            self.analyzed = False

        def analyze_sweep(self):
            self.analyzed = True

    class FakeModule:
        def __init__(self):
            self.calls = []

        def InsertFrequencySweep(self, setup_name, params):
            self.calls.append((setup_name, params))

    class FakeSetup:
        def __init__(self):
            self.name = "DrivenModalSetup"
            self._setup_module = FakeModule()
            self.sweep = FakeSweep()

        def insert_sweep(self, **kwargs):
            raise AssertionError("legacy insert_sweep path should not be used")

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
        stop_ghz=10.0,
        count=400,
        name="DrivenModalSweep",
        type="Interpolating",
        save_fields=False,
        interpolation_tol=0.005,
        interpolation_max_solutions=400,
    )

    assert sweep is setup.sweep
    assert renderer.current_sweep is setup.sweep
    assert setup.sweep.analyzed is True
    assert len(setup._setup_module.calls) == 1
    _, params = setup._setup_module.calls[0]
    assert "InterpTolerance:=" in params
    assert params[params.index("InterpTolerance:=") + 1] == 0.005
    assert "InterpMaxSolns:=" in params
    assert params[params.index("InterpMaxSolns:=") + 1] == 400
    assert "RangeCount:=" in params
    assert params[params.index("RangeCount:=") + 1] == 400


def test_safe_ansys_design_name_is_short_and_stable():
    first = safe_ansys_design_name("tutorial10-qubit-claw-000-v1")
    second = safe_ansys_design_name("tutorial10-qubit-claw-000-v1")

    assert first == second
    assert first.startswith("dm_")
    assert "-" not in first
    assert len(first) == 11
