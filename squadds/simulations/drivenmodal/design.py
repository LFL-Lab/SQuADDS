"""Design helpers for explicit driven-modal layer-stack workflows."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from qiskit_metal import Dict
from qiskit_metal.designs.design_multiplanar import MultiPlanar

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
    slug = re.sub(r"[^0-9A-Za-z]+", "_", identifier).strip("_").lower()
    slug = slug[:12] or "run"
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{slug}_{digest}"


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
        setup.insert_sweep(**sweep_kwargs)
        sweep = setup.get_sweep(sweep_name)
        if hasattr(sweep, "analyze_sweep"):
            sweep.analyze_sweep()
        renderer.current_sweep = sweep
        return sweep

    renderer.add_sweep(setup_name=setup_name, **sweep_kwargs)
    renderer.analyze_sweep(sweep_name, setup_name)
    return getattr(renderer, "current_sweep", None)
