"""Parametric sweep definitions for new Qiskit Metal component families.

Everything here is deliberately generic.  The published families needed bespoke
port code because their terminals were identified by hand; a Metal component
already declares its pins and its own subtractive geometry, so ordered terminal
markers and the ground clearance can both be derived without per-family code.
Adding a family is therefore a :class:`SweepSpec` entry and nothing else.

The output follows the published layout convention exactly, so files generated
here drop straight into the registry and encode with ``universal-geometry-v2``:

* conductors on ``(1, 10)``;
* one square ground simulation domain on ``(1, 0)`` with a fixed
  :data:`~squadds.layouts.qmetal_gds.GROUND_DOMAIN_PADDING_UM` per-side margin
  and a single subtractive hole;
* ordered terminal markers on layers ``2, 3, 4, 5`` bridging each terminal to
  the ground boundary, which is what fixes terminal order in the embedding.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from .qmetal_gds import (
    PORT_LENGTH_UM,
    PortMarker,
    export_qgeometry_gds,
    marker_from_pin,
    qgeometry_role,
)

#: Terminal index to GDS layer.  Extends the published two-terminal convention.
PORT_LAYERS = (2, 3, 4, 5)


@dataclass(frozen=True)
class SweepSpec:
    """One component family and the grid to sweep it over."""

    component_name: str
    import_path: str
    port_pins: tuple[str, ...]
    grid: Mapping[str, Sequence[Any]]
    base_options: Mapping[str, Any] = field(default_factory=dict)
    clearance_paths: tuple[str, ...] = ()
    note: str = ""

    @property
    def component_class(self) -> type:
        module, _, qualname = self.import_path.rpartition(".")
        return getattr(import_module(module), qualname)

    @property
    def size(self) -> int:
        total = 1
        for values in self.grid.values():
            total *= len(values)
        return total

    def points(self) -> Iterator[dict[str, Any]]:
        keys = list(self.grid)
        for combination in itertools.product(*(self.grid[key] for key in keys)):
            yield dict(zip(keys, combination))


def _assign(options: dict, dotted: str, value: Any) -> None:
    """Set ``a.b.c`` inside a nested option mapping, creating levels as needed."""
    keys = dotted.split(".")
    cursor = options
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _lookup(options: Any, dotted: str) -> Any:
    cursor = options
    for key in dotted.split("."):
        if cursor is None:
            return None
        cursor = cursor[key] if key in cursor else None
    return cursor


def _gap_candidates(options: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield every ``*gap*`` leaf in a component's option tree."""
    try:
        items = options.items()
    except AttributeError:
        return
    for key, value in items:
        path = f"{prefix}{key}"
        if hasattr(value, "items"):
            yield from _gap_candidates(value, f"{path}.")
        elif "gap" in key.lower() and isinstance(value, (str, int, float)):
            yield path, value


def ground_clearance_um(component: Any, spec: SweepSpec | None = None) -> float:
    """Smallest signal-to-ground clearance a component declares.

    Prefers the explicit paths on the spec, then falls back to the smallest
    ``*gap*`` option anywhere in the tree.  Both are read through Metal's own
    parser, so unit suffixes are handled the same way the renderer handles them.
    """
    design = component.design
    values: list[float] = []
    if spec is not None and spec.clearance_paths:
        for path in spec.clearance_paths:
            raw = _lookup(component.options, path)
            if isinstance(raw, (str, int, float)):
                values.append(float(design.parse_value(raw)) * 1000.0)
    if not values:
        values = [float(design.parse_value(raw)) * 1000.0 for _, raw in _gap_candidates(component.options)]
    positive = [value for value in values if value > 0]
    if not positive:
        raise ValueError(
            f"No positive gap option found for {component.__class__.__name__!r}; set clearance_paths on its SweepSpec."
        )
    return min(positive)


def build_component(spec: SweepSpec, point: Mapping[str, Any], name: str = "sweep") -> Any:
    """Instantiate one design point on a fresh planar design."""
    from qiskit_metal import designs

    options: dict[str, Any] = {}
    for key, value in spec.base_options.items():
        _assign(options, key, value)
    for key, value in point.items():
        _assign(options, key, value)
    design = designs.DesignPlanar()
    design.overwrite_enabled = True
    return spec.component_class(design, name, options=options)


def port_markers(component: Any, spec: SweepSpec) -> list[PortMarker]:
    """Ordered markers, one per declared pin, on layers 2, 3, 4, 5.

    Pin order is the contract: it decides which physical conductor becomes
    terminal 0, and therefore which coupling slot each pair lands in.
    """
    if len(spec.port_pins) > len(PORT_LAYERS):
        raise ValueError(
            f"{spec.component_name} declares {len(spec.port_pins)} ports but only "
            f"{len(PORT_LAYERS)} ordered layers exist."
        )
    markers = []
    for index, pin_name in enumerate(spec.port_pins):
        if pin_name not in component.pins:
            raise ValueError(f"{spec.component_name} has no pin {pin_name!r}; found {list(component.pins)}.")
        markers.append(
            marker_from_pin(
                component.pins[pin_name],
                semantic=f"terminal_{index}_port",
                layer=PORT_LAYERS[index],
                source=f"{spec.component_name}.{pin_name}",
                length_um=PORT_LENGTH_UM,
            )
        )
    return markers


def terminal_count(component: Any) -> int:
    """Number of electrically isolated conductor islands.

    This, not the pin count, is what the embedding treats as terminals: a Tee
    with three pins on one connected trace is a single terminal, while a pocket
    transmon with two connection pads is four.
    """
    conductor = qgeometry_role(component.design, subtract=False)
    parts = list(getattr(conductor, "geoms", [conductor]))
    return sum(1 for part in parts if part.area > 0)


def write_design_point(spec: SweepSpec, point: Mapping[str, Any], destination: Path, name: str = "sweep") -> dict:
    """Render one design point to GDS and return its manifest row."""
    component = build_component(spec, point, name=name)
    markers = port_markers(component, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_qgeometry_gds(
        component.design,
        destination,
        markers=markers,
        include_ground_domain=True,
        minimum_ground_clearance_um=ground_clearance_um(component, spec),
    )
    return {
        "component_name": spec.component_name,
        "gds_path": str(destination),
        "terminal_count": terminal_count(component),
        "port_count": len(markers),
        "ground_clearance_um": round(ground_clearance_um(component, spec), 4),
        "design_options": {key: point[key] for key in sorted(point)},
    }


# ---------------------------------------------------------------------------
# The families proposed for the next simulation round
# ---------------------------------------------------------------------------

SWEEPS: dict[str, SweepSpec] = {
    # Two terminals, but coupling through parallel-line proximity rather than
    # interdigitated fingers: the first real topology change in the catalogue.
    "CoupledLineTee": SweepSpec(
        component_name="CoupledLineTee",
        import_path="qiskit_metal.qlibrary.couplers.coupled_line_tee.CoupledLineTee",
        port_pins=("prime_start", "second_end"),
        grid={
            "coupling_length": ["100um", "175um", "250um", "325um", "400um"],
            "coupling_space": ["3um", "5um", "8um", "12um", "18um"],
            "prime_width": ["10um", "15um", "20um"],
            "second_width": ["10um", "15um", "20um"],
            "prime_gap": ["4um", "6um", "9um"],
        },
        clearance_paths=("prime_gap", "second_gap"),
        note="New coupling topology at two terminals.",
    ),
    # A third interdigital comb: the control that separates topological distance
    # from mere family identity.
    "Cap3Interdigital": SweepSpec(
        component_name="Cap3Interdigital",
        import_path="qiskit_metal.qlibrary.lumped.cap_3_interdigital.Cap3Interdigital",
        port_pins=("a", "b"),
        grid={
            "finger_length": ["40um", "65um", "90um", "115um", "140um"],
            "finger_width": ["4um", "6um", "9um"],
            "cap_gap": ["3um", "5um", "8um"],
            "pocket_width": ["120um", "180um"],
            "cap_width": ["8um", "12um"],
        },
        clearance_paths=("cap_gap",),
        note="Within-topology control against CapNInterdigital.",
    ),
    # The plain (non-Tee) comb, for depth inside the existing topology group.
    "CapNInterdigital": SweepSpec(
        component_name="CapNInterdigital",
        import_path="qiskit_metal.qlibrary.lumped.cap_n_interdigital.CapNInterdigital",
        port_pins=("north_end", "south_end"),
        grid={
            "finger_length": ["40um", "70um", "100um", "130um"],
            "finger_count": ["4", "6", "8", "10"],
            "cap_gap": ["3um", "5um", "8um"],
            "cap_width": ["8um", "12um"],
            "cap_gap_ground": ["4um", "6um"],
        },
        clearance_paths=("cap_gap_ground", "cap_gap"),
        note="Within-topology depth.",
    ),
    # Four isolated islands: two transmon pads plus two connection pads.  This
    # is the first design in the catalogue to populate more than one coupling
    # pair slot, and the only proposed family that exercises all six.
    "TransmonPocket": SweepSpec(
        component_name="TransmonPocket",
        import_path="qiskit_metal.qlibrary.qubits.transmon_pocket.TransmonPocket",
        port_pins=("a", "b"),
        grid={
            "pad_width": ["350um", "425um", "500um"],
            "pad_height": ["70um", "90um", "120um"],
            "pad_gap": ["20um", "30um", "45um"],
            "connection_pads.a.pad_width": ["100um", "125um"],
            "connection_pads.a.pad_gap": ["10um", "15um"],
            "connection_pads.b.pad_width": ["100um", "125um"],
        },
        base_options={
            "connection_pads.a.loc_W": 1,
            "connection_pads.a.loc_H": 1,
            "connection_pads.b.loc_W": -1,
            "connection_pads.b.loc_H": 1,
        },
        clearance_paths=("pad_gap",),
        note="Second qubit family, and a genuine four-terminal device.",
    ),
    # Six islands: a centre pad plus five couplers.  Exceeds MAX_TERMINALS, so
    # it also tests how the encoder behaves when it must truncate.
    "StarQubit": SweepSpec(
        component_name="StarQubit",
        import_path="qiskit_metal.qlibrary.qubits.star_qubit.StarQubit",
        port_pins=("pin_cpl1", "pin_cpl2", "pin_cpl3", "pin_rdout"),
        grid={
            "radius": ["250um", "300um", "350um"],
            "center_radius": ["80um", "100um", "125um"],
            "gap_couplers": ["16um", "25um", "34um"],
            "gap_readout": ["6um", "10um", "16um"],
            "number_of_connectors": [4],
        },
        clearance_paths=("gap_couplers", "gap_readout"),
        note="Multi-terminal stress case beyond MAX_TERMINALS.",
    ),
}
