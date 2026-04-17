"""Design helpers for explicit driven-modal layer-stack workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qiskit_metal.designs.design_multiplanar import MultiPlanar

from .layer_stack import build_layer_stack_dataframe
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
    design.chips[layer_stack.chip_name]["size"]["size_x"] = chip_size_x
    design.chips[layer_stack.chip_name]["size"]["size_y"] = chip_size_y
    # QHFSSRenderer reads the active chip elevation from center_z when lifting
    # 2D polygons into 3D points. MultiPlanar does not always populate it.
    design.chips[layer_stack.chip_name]["size"].setdefault("center_z", chip_center_z)
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


def format_exception_for_console(exc: BaseException) -> str:
    """Return an ASCII-safe exception string for Windows console output."""
    return str(exc).encode("ascii", "backslashreplace").decode("ascii")
