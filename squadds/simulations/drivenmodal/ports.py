"""Reusable port-spec builders for driven-modal workflows."""

from __future__ import annotations

from collections.abc import Mapping

from .models import DrivenModalPortSpec


def _require_port_mapping(design_payload: Mapping[str, object]) -> Mapping[str, object]:
    port_mapping = design_payload.get("port_mapping")
    if not isinstance(port_mapping, Mapping):
        raise ValueError("design_payload must include a 'port_mapping' mapping.")
    return port_mapping


def _build_port_spec(kind: str, port_mapping: Mapping[str, object]) -> DrivenModalPortSpec:
    raw_spec = port_mapping.get(kind)
    if not isinstance(raw_spec, Mapping):
        raise ValueError(f"port_mapping must include '{kind}'.")

    component = raw_spec.get("component")
    pin = raw_spec.get("pin")
    if not isinstance(component, str) or not component.strip():
        raise ValueError(f"component for port '{kind}' must be a non-empty string.")
    if not isinstance(pin, str) or not pin.strip():
        raise ValueError(f"pin for port '{kind}' must be a non-empty string.")
    impedance_ohms = float(raw_spec.get("impedance_ohms", 50.0))
    metadata_raw = raw_spec.get("metadata", {})
    if not isinstance(metadata_raw, Mapping):
        raise ValueError(f"metadata for port '{kind}' must be a mapping.")
    metadata = dict(metadata_raw)
    if kind in {"cross", "jj"}:
        metadata.setdefault("hfss_target", "junction")
        metadata.setdefault("draw_inductor", False)
    else:
        metadata.setdefault("hfss_target", "pin")

    return DrivenModalPortSpec(
        kind=kind,
        component=component,
        pin=pin,
        impedance_ohms=impedance_ohms,
        metadata=metadata,
    )


def build_capacitance_port_specs(system_kind: str, design_payload: Mapping[str, object]) -> list[DrivenModalPortSpec]:
    """Build the ordered port declarations for capacitance extraction systems."""
    if system_kind == "ncap":
        ordered_kinds = ["top", "bottom"]
    elif system_kind == "qubit_claw":
        ordered_kinds = ["cross", "claw"]
    else:
        raise ValueError(f"Unsupported capacitance system_kind: {system_kind}")

    port_mapping = _require_port_mapping(design_payload)
    return [_build_port_spec(kind, port_mapping) for kind in ordered_kinds]


def build_coupled_system_port_specs(design_payload: Mapping[str, object]) -> list[DrivenModalPortSpec]:
    """Build the ordered feedline and JJ ports for coupled systems."""
    port_mapping = _require_port_mapping(design_payload)
    ordered_kinds = ["feedline_input", "feedline_output", "jj"]
    return [_build_port_spec(kind, port_mapping) for kind in ordered_kinds]


def split_rendered_ports(
    port_specs: list[DrivenModalPortSpec],
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float, bool]]]:
    """Split mixed pin/junction specs into Qiskit Metal ``port_list`` and ``jj_to_port`` payloads."""
    port_list: list[tuple[str, str, float]] = []
    jj_to_port: list[tuple[str, str, float, bool]] = []

    for spec in port_specs:
        if spec.metadata.get("hfss_target") == "junction":
            jj_to_port.append(
                (
                    spec.component,
                    spec.pin,
                    float(spec.impedance_ohms),
                    bool(spec.metadata.get("draw_inductor", False)),
                )
            )
        else:
            port_list.append(spec.to_qiskit_port_entry())

    return port_list, jj_to_port
