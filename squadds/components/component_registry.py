"""Resolve dataset component names to their Qiskit Metal implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit_metal.qlibrary.core import QComponent


@dataclass(frozen=True)
class CouplerSpec:
    """Construction and pin roles for a supported coupler."""

    component_class: type[QComponent]
    route_pin: str
    exposed_pins: tuple[str, ...]
    instance_suffix: str


def _normalized_component_name(component_name: str) -> str:
    return component_name.replace("_", "").replace("-", "").lower()


def get_coupler_spec(component_name: str) -> CouplerSpec:
    """Return the implementation and pin roles for a coupler name."""
    normalized = _normalized_component_name(component_name)
    if normalized == "generalizedcapninterdigital":
        from squadds.components.generalized_ncap_interdigital import GeneralizedCapNInterdigital

        return CouplerSpec(
            component_class=GeneralizedCapNInterdigital,
            route_pin="south_end",
            exposed_pins=("north_end",),
            instance_suffix="generalized_ncap_coupler",
        )

    if normalized in {"capn", "ncap", "capninterdigitaltee"}:
        from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

        return CouplerSpec(
            component_class=CapNInterdigitalTee,
            route_pin="second_end",
            exposed_pins=("prime_start", "prime_end"),
            instance_suffix="capn_coupler",
        )

    if normalized in {"clt", "coupledlinetee"}:
        from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee

        return CouplerSpec(
            component_class=CoupledLineTee,
            route_pin="second_end",
            exposed_pins=("prime_start", "prime_end"),
            instance_suffix="CLT_coupler",
        )

    raise ValueError(f"Unsupported coupler component: {component_name}")


def create_coupler(component_name: str, design, name: str, options: dict[str, Any] | None = None) -> QComponent:
    """Instantiate a named coupler using its registered implementation."""
    spec = get_coupler_spec(component_name)
    return spec.component_class(design, name, options=options or {})


def build_component_from_design(design, row: dict[str, Any], name: str = "cplr") -> QComponent:
    """Build the exact component declared by a SQuADDS dataset row.

    Both nested Hugging Face records and second-level flattened API records are
    accepted. GeneralizedCapNInterdigital is intentionally resolved only from
    ``squadds.components``.
    """
    design_record = row.get("design", row)
    component_name = (
        design_record.get("component_class") or design_record.get("component_name") or design_record.get("coupler_type")
    )
    if not component_name:
        raise ValueError("Dataset row does not declare a component class or coupler type.")

    options = design_record.get("design_options", row.get("design_options", {}))
    return create_coupler(component_name, design, name, options)
