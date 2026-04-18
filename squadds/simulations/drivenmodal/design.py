"""Design helpers for explicit driven-modal layer-stack workflows."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from qiskit_metal import Dict
from qiskit_metal.designs.design_multiplanar import MultiPlanar
from shapely.geometry import LineString

from .layer_stack import build_layer_stack_dataframe, resolve_chip_metadata
from .models import DrivenModalLayerStackSpec


def write_qiskit_layer_stack_csv(
    spec: DrivenModalLayerStackSpec,
    output_path: str | Path,
) -> Path:
    """Persist the resolved layer stack in the CSV format consumed by Qiskit Metal."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = build_layer_stack_dataframe(spec).copy()
    frame["fill"] = frame["fill"].map(lambda value: str(value).lower())
    frame.to_csv(path, index=False)
    return path


def create_multiplanar_design(
    *,
    layer_stack: DrivenModalLayerStackSpec,
    layer_stack_path: str | Path,
    chip_size_x: str = "9mm",
    chip_size_y: str = "7mm",
    chip_center_z: str = "0.0mm",
    enable_renderers: bool = True,
) -> tuple[MultiPlanar, Path]:
    """Create a MultiPlanar design bound to an explicit layer-stack CSV."""
    csv_path = write_qiskit_layer_stack_csv(layer_stack, layer_stack_path)
    design = MultiPlanar(
        overwrite_enabled=True,
        enable_renderers=enable_renderers,
        layer_stack_filename=str(csv_path),
    )
    design.overwrite_enabled = True
    chip = design.chips[layer_stack.chip_name]
    chip_size = chip["size"]
    chip_size["size_x"] = chip_size_x
    chip_size["size_y"] = chip_size_y
    # QHFSSRenderer reads the active chip elevation from center_z when lifting
    # 2D polygons into 3D points. MultiPlanar does not always populate it.
    chip_size.setdefault("center_z", chip_center_z)

    chip_metadata = resolve_chip_metadata(layer_stack)
    chip.setdefault("material", chip_metadata["material"])
    chip.setdefault("layer_start", chip_metadata["layer_start"])
    chip.setdefault("layer_end", chip_metadata["layer_end"])
    chip_size.setdefault("size_z", chip_metadata["size_z"])
    chip_size.setdefault("sample_holder_top", chip_metadata["sample_holder_top"])
    chip_size.setdefault("sample_holder_bottom", chip_metadata["sample_holder_bottom"])
    return design, csv_path


def connect_renderer_to_new_ansys_design(
    renderer: Any,
    design_name: str,
    solution_type: str = "drivenmodal",
):
    """Create a new Ansys design without forcing an immediate setup lookup.

    Qiskit Metal's ``new_ansys_design(..., connect=True)`` helper reconnects via
    ``connect_ansys_design()``, which unconditionally calls ``connect_setup()``.
    Brand-new driven-modal designs do not have a setup yet, so that path fails
    before callers can create one. We avoid that eager setup lookup by creating
    the design with ``connect=False`` and then explicitly binding ``pinfo`` to
    the new design only.
    """
    ansys_design = renderer.new_ansys_design(design_name, solution_type, connect=False)
    renderer.pinfo.connect_design(ansys_design.name)
    return ansys_design


def save_renderer_project(renderer: Any, project_file: str | Path) -> Path:
    """Persist the current AEDT project state to disk."""
    project_path = Path(project_file)
    renderer.pinfo.project.save(str(project_path))
    return project_path


def apply_cryo_silicon_material_properties(
    renderer: Any,
    *,
    permittivity: float = 11.45,
    loss_tangent: float = 1e-7,
    hfss_factory: Any | None = None,
    materials_factory: Any | None = None,
) -> dict[str, float | str]:
    """Overwrite HFSS silicon properties with the cryogenic SQuADDS defaults.

    Qiskit Metal's driven-modal renderer only references the material by name in
    the layer stack. On a fresh AEDT install, that means HFSS keeps Ansys'
    default room-temperature silicon permittivity instead of the 11.45 value
    already used in the eigenmode/Q3D flows. We patch the active project
    material in-place once the renderer is connected to the target design.
    """
    pinfo = getattr(renderer, "pinfo", None)
    project_name = getattr(pinfo, "project_name", None)
    design_name = getattr(pinfo, "design_name", None)
    if not project_name or not design_name:
        raise ValueError("Renderer must be connected to an HFSS project and design before editing materials.")

    result = {
        "material": "silicon",
        "permittivity": float(permittivity),
        "dielectric_loss_tangent": float(loss_tangent),
        "project_name": str(project_name),
        "design_name": str(design_name),
    }

    class _LiveMaterialApp:
        def __init__(self, source_renderer: Any):
            live_pinfo = getattr(source_renderer, "pinfo", None)
            live_project = getattr(live_pinfo, "project", None)
            live_design = getattr(live_pinfo, "design", None)
            live_desktop = getattr(live_pinfo, "desktop", None)

            self.logger = getattr(source_renderer, "logger", logging.getLogger(__name__))
            self._oproject = getattr(live_project, "_project", None)
            self._odesign = getattr(live_design, "_design", None)
            self._desktop = getattr(live_desktop, "_desktop", None)
            if self._desktop is None:
                self._desktop = getattr(getattr(live_project, "parent", None), "_desktop", None)
            self.design_type = "HFSS"

        @property
        def odesktop(self):
            return self._desktop

        @property
        def oproject(self):
            return self._oproject

        @property
        def odesign(self):
            return self._odesign

        @property
        def odefinition_manager(self):
            if self._oproject is None:
                return None
            return self._oproject.GetDefinitionManager()

        @property
        def omaterial_manager(self):
            definition_manager = self.odefinition_manager
            if definition_manager is None:
                return None
            return definition_manager.GetManager("Material")

        def evaluate_expression(self, expression):
            return expression

    if materials_factory is None:
        try:
            from ansys.aedt.core.modules.material_lib import Materials
        except Exception:
            Materials = None
        materials_factory = Materials

    if materials_factory is not None:
        try:
            live_materials = materials_factory(_LiveMaterialApp(renderer))
            silicon = live_materials.exists_material("silicon")
            if silicon:
                silicon.permittivity = permittivity
                silicon.dielectric_loss_tangent = loss_tangent
                return result
        except Exception:
            # Fall back to opening a dedicated PyAEDT handle when the current
            # renderer session does not expose enough state for Materials(...).
            pass

    if hfss_factory is None:
        try:
            from pyaedt import Hfss
        except Exception as exc:  # pragma: no cover - only exercised on the HFSS machine
            raise RuntimeError("PyAEDT is required to update driven-modal material properties.") from exc
        hfss_factory = Hfss

    aedt = None
    try:
        aedt = hfss_factory(
            project=project_name,
            design=design_name,
            solution_type="DrivenModal",
            new_desktop=False,
            close_on_exit=False,
        )
        silicon = aedt.materials.exists_material("silicon")
        if not silicon:
            silicon = aedt.materials.checkifmaterialexists("silicon")
        silicon.permittivity = permittivity
        silicon.dielectric_loss_tangent = loss_tangent
        return result
    finally:
        if aedt is not None and hasattr(aedt, "release_desktop"):
            try:
                aedt.release_desktop(close_projects=False, close_desktop=False)
            except Exception:
                pass


def ensure_perfect_e_boundary(
    renderer: Any,
    object_names: list[str],
    *,
    boundary_name: str = "PerfE_Metal",
) -> list[str]:
    """Ensure a stable Perfect E boundary exists for the rendered metal sheets."""
    cleaned_names = sorted({str(name) for name in object_names if name})
    if not cleaned_names:
        return []

    design = getattr(getattr(renderer, "pinfo", None), "design", None)
    if design is not None and hasattr(design, "append_PerfE_assignment"):
        design.append_PerfE_assignment(boundary_name, cleaned_names)
        return cleaned_names

    modeler = getattr(renderer, "modeler", None)
    if modeler is not None and hasattr(modeler, "assign_perfect_E"):
        modeler.assign_perfect_E(cleaned_names, name=boundary_name)
        return cleaned_names

    raise AttributeError("Renderer does not expose an HFSS Perfect E assignment API.")


def snapshot_boundary_assignments(renderer: Any) -> dict[str, list[str]]:
    """Return saved HFSS boundary assignments for later artifact/debug dumps."""
    design = getattr(getattr(renderer, "pinfo", None), "design", None)
    if design is None or not hasattr(design, "_boundaries"):
        return {}

    try:
        boundary_names = list(design._boundaries.GetBoundaries())
    except Exception:
        return {}

    assignments: dict[str, list[str]] = {}
    for boundary_name in boundary_names:
        try:
            assigned_objects = sorted(set(design.get_boundary_assignment(boundary_name)))
        except Exception:
            assigned_objects = []
        assignments[str(boundary_name)] = assigned_objects
    return assignments


def format_exception_for_console(exc: BaseException) -> str:
    """Return an ASCII-safe exception string for Windows console output."""
    return str(exc).encode("ascii", "backslashreplace").decode("ascii")


def safe_ansys_design_name(identifier: str, *, prefix: str = "dm") -> str:
    """Return a short HFSS-safe design name derived from a longer run identifier.

    Older HFSS / pyEPR combinations can become unstable when driven-modal design
    names are long and heavily punctuated. We keep the user-facing run ID in the
    checkpoint and artifact layout, but use a compact deterministic alias for
    the internal Ansys design name.
    """
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def _format_mm(value_mm: float) -> str:
    return f"{value_mm:.6f}".rstrip("0").rstrip(".") + "mm"


def _is_collinear(prev_point: tuple[float, float], point: tuple[float, float], next_point: tuple[float, float]) -> bool:
    x1, y1 = prev_point
    x2, y2 = point
    x3, y3 = next_point
    area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return abs(area2) <= 1e-12


def simplify_collinear_path_points(design: Any) -> int:
    """Collapse redundant collinear path vertices before Ansys rendering.

    Qiskit Metal can emit straight CPW centerlines with tiny extra collinear
    segments near terminations. Older HFSS / COM combinations sometimes reject
    those paths in ``CreatePolyline`` even though the geometry is physically a
    simple straight line. Removing only the redundant collinear interior points
    preserves bends/fillets while producing a more stable path for rendering.
    """
    path_table = design.qgeometry.tables.get("path")
    if path_table is None or len(path_table) == 0:
        return 0

    simplified = 0
    for index, row in path_table.iterrows():
        coords = [tuple(map(float, point)) for point in row["geometry"].coords]
        if len(coords) <= 2:
            continue

        cleaned = [coords[0]]
        for point in coords[1:]:
            if point != cleaned[-1]:
                cleaned.append(point)

        if len(cleaned) <= 2:
            new_coords = [cleaned[0], cleaned[-1]]
        else:
            new_coords = [cleaned[0]]
            for position, point in enumerate(cleaned[1:-1], start=1):
                if _is_collinear(new_coords[-1], point, cleaned[position + 1]):
                    continue
                new_coords.append(point)
            new_coords.append(cleaned[-1])

        if len(new_coords) < 2:
            continue
        if new_coords != coords:
            path_table.at[index, "geometry"] = LineString(new_coords)
            simplified += 1

    return simplified


def apply_buffered_chip_bounds(
    design: Any,
    *,
    selection: list[str],
    chip_name: str = "main",
    x_buffer_mm: float = 0.2,
    y_buffer_mm: float = 0.2,
) -> dict[str, float]:
    """Set the chip size from the rendered-component bounds plus renderer buffers.

    Qiskit Metal's HFSS and Q3D renderers both use the same shared
    ``box_plus_buffer=True`` logic in ``QAnsysRenderer``. This helper makes that
    box explicit on the design before rendering so tutorials can record and
    inspect the exact chip/ground bounding box used for a run.
    """
    if not selection:
        raise ValueError("selection must include at least one component.")

    bounds = [design.components[name].qgeometry_bounds() for name in selection]
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)

    width_x = (max_x - min_x) + 2 * x_buffer_mm
    width_y = (max_y - min_y) + 2 * y_buffer_mm
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    chip_size = design.chips[chip_name]["size"]
    chip_size["size_x"] = _format_mm(width_x)
    chip_size["size_y"] = _format_mm(width_y)
    chip_size["center_x"] = _format_mm(center_x)
    chip_size["center_y"] = _format_mm(center_y)

    return {
        "min_x_mm": min_x,
        "min_y_mm": min_y,
        "max_x_mm": max_x,
        "max_y_mm": max_y,
        "buffered_size_x_mm": width_x,
        "buffered_size_y_mm": width_y,
        "buffered_center_x_mm": center_x,
        "buffered_center_y_mm": center_y,
        "x_buffer_mm": x_buffer_mm,
        "y_buffer_mm": y_buffer_mm,
    }


def render_drivenmodal_design(
    renderer: Any,
    *,
    selection: list[str],
    open_pins: list[tuple[str, str]] | None = None,
    port_list: list[tuple[str, str, float | str]] | None = None,
    jj_to_port: list[tuple[str, str, float | str, bool]] | None = None,
    ignored_jjs: list[tuple[str, str]] | None = None,
    box_plus_buffer: bool = True,
):
    """Render a driven-modal geometry with a Qiskit Metal compatibility guard.

    Qiskit Metal's current HFSS driven-modal renderer assumes ``open_pins`` is a
    list whenever ``port_list`` is present and concatenates into it internally.
    Normalizing ``None`` to ``[]`` here keeps the public call shape clean while
    avoiding that renderer-side ``TypeError``.
    """
    if open_pins is None and (port_list or jj_to_port):
        open_pins = []
    return renderer.render_design(
        selection=selection,
        open_pins=open_pins,
        port_list=port_list,
        jj_to_port=jj_to_port,
        ignored_jjs=ignored_jjs,
        box_plus_buffer=box_plus_buffer,
    )


def ensure_drivenmodal_setup(renderer: Any, **setup_kwargs: Any):
    """Create and bind a driven-modal setup across Qiskit Metal renderer versions.

    Some HFSS renderer versions create the setup but do not update
    ``renderer.pinfo.setup`` to reference the newly created name. Later calls
    such as ``add_sweep`` then fail because they look up the active setup
    through ``pinfo``. We make that state transition explicit here and reapply
    the supported editable setup fields once the setup is activated.
    """
    setup = renderer.add_drivenmodal_setup(**setup_kwargs)
    setup_name = setup_kwargs.get("name")

    if setup_name and hasattr(renderer, "activate_ansys_setup"):
        try:
            renderer.activate_ansys_setup(setup_name)
        except Exception:
            # Older pyEPR / Qiskit Metal combinations can fail to rebound an
            # existing driven-modal setup even though creation succeeded. Keep
            # the returned setup handle and continue via direct setup methods.
            pass
        if hasattr(renderer, "pinfo"):
            renderer.pinfo.setup_name = setup_name
            renderer.pinfo.setup = setup
        if hasattr(renderer, "edit_drivenmodal_setup") and getattr(renderer, "pinfo", None) is not None:
            renderer.edit_drivenmodal_setup(Dict(setup_kwargs))

    return setup


def _insert_sweep_with_interpolation_options(setup: Any, **sweep_kwargs: Any):
    """Insert a driven-modal sweep while exposing HFSS interpolating-sweep options.

    Older Qiskit Metal / pyEPR stacks only forward the basic sweep arguments,
    which hides HFSS's interpolation tolerance and maximum-basis controls. When
    callers provide those options we fall back to the underlying HFSS scripting
    payload directly, while preserving the legacy defaults from the official
    ``InsertFrequencySweep`` contract.
    """
    interpolation_tol = sweep_kwargs.pop("interpolation_tol", None)
    interpolation_max_solutions = sweep_kwargs.pop("interpolation_max_solutions", None)
    if interpolation_tol is None and interpolation_max_solutions is None:
        setup.insert_sweep(**sweep_kwargs)
        return

    sweep_type = sweep_kwargs["type"]
    if sweep_type != "Interpolating":
        setup.insert_sweep(**sweep_kwargs)
        return

    count = sweep_kwargs.get("count")
    step_ghz = sweep_kwargs.get("step_ghz")
    if (count and step_ghz) or ((not count) and (not step_ghz)):
        raise ValueError("Provide either count or step_ghz when inserting a driven-modal sweep.")

    params = [
        "NAME:" + sweep_kwargs["name"],
        "IsEnabled:=",
        True,
        "Type:=",
        sweep_type,
        "SaveFields:=",
        sweep_kwargs.get("save_fields", False),
        "SaveRadFields:=",
        False,
        "InterpTolerance:=",
        float(interpolation_tol if interpolation_tol is not None else 0.5),
        "InterpMaxSolns:=",
        int(interpolation_max_solutions if interpolation_max_solutions is not None else 250),
        "InterpMinSolns:=",
        0,
        "InterpMinSubranges:=",
        1,
        "InterpUseS:=",
        True,
        "InterpUsePortImped:=",
        False,
        "InterpUsePropConst:=",
        True,
        "UseDerivativeConvergence:=",
        False,
        "InterpDerivTolerance:=",
        0.2,
        "UseFullBasis:=",
        True,
        "EnforcePassivity:=",
        True,
        "PassivityErrorTolerance:=",
        0.0001,
        "SMatrixOnlySolveMode:=",
        "Manual",
        "SMatrixOnlySolveAbove:=",
        "1MHz",
        "ExtrapToDC:=",
        False,
    ]

    if count:
        params.extend(
            [
                "RangeType:=",
                "LinearCount",
                "RangeStart:=",
                f"{sweep_kwargs['start_ghz']:f}GHz",
                "RangeEnd:=",
                f"{sweep_kwargs['stop_ghz']:f}GHz",
                "RangeCount:=",
                count,
            ]
        )
    else:
        params.extend(
            [
                "RangeType:=",
                "LinearStep",
                "RangeStart:=",
                f"{sweep_kwargs['start_ghz']:f}GHz",
                "RangeEnd:=",
                f"{sweep_kwargs['stop_ghz']:f}GHz",
                "RangeStep:=",
                step_ghz,
            ]
        )

    setup._setup_module.InsertFrequencySweep(setup.name, params)


def run_drivenmodal_sweep(renderer: Any, setup: Any, *, setup_name: str, **sweep_kwargs: Any):
    """Insert and analyze a driven-modal sweep with compatibility fallbacks.

    Older HFSS renderer stacks expose a valid setup object from
    ``add_drivenmodal_setup`` but fail when the renderer later tries to recover
    that setup through ``pinfo.get_setup(setup_name)``. When a concrete setup
    handle is available, use it directly for sweep insertion and analysis, then
    store the resulting sweep on ``renderer.current_sweep`` so parameter export
    continues to work.
    """
    sweep_name = sweep_kwargs["name"]

    if setup is not None and hasattr(setup, "insert_sweep") and hasattr(setup, "get_sweep"):
        _insert_sweep_with_interpolation_options(setup, **sweep_kwargs)
        sweep = setup.get_sweep(sweep_name)
        if hasattr(sweep, "analyze_sweep"):
            sweep.analyze_sweep()
        renderer.current_sweep = sweep
        return sweep

    renderer.add_sweep(
        setup_name=setup_name,
        **{
            key: value
            for key, value in sweep_kwargs.items()
            if key not in {"interpolation_tol", "interpolation_max_solutions"}
        },
    )
    renderer.analyze_sweep(sweep_name, setup_name)
    return getattr(renderer, "current_sweep", None)
