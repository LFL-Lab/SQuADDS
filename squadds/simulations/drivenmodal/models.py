"""Typed request and result models for HFSS driven-modal workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


def _require_mapping(name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")


@dataclass(frozen=True)
class DrivenModalLayerStackSpec:
    """Resolved from a fixed SQuADDS layer-stack preset plus thickness overrides."""

    preset: str = "squadds_hfss_v1"
    chip_name: str = "main"
    metal_layer: int = 1
    substrate_layer: int = 3
    datatype: int = 0
    metal_thickness_um: float = 0.2
    substrate_thickness_um: float = 750.0
    metal_z_coord_um: float = 0.0
    substrate_z_coord_um: float = 0.0

    def __post_init__(self) -> None:
        if not self.preset:
            raise ValueError("preset is required.")
        if self.metal_thickness_um <= 0:
            raise ValueError("metal_thickness_um must be positive.")
        if self.substrate_thickness_um <= 0:
            raise ValueError("substrate_thickness_um must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DrivenModalSetupSpec:
    """HFSS driven-modal adaptive setup parameters."""

    name: str = "Setup"
    freq_ghz: float = 5.0
    max_delta_s: float = 0.1
    max_passes: int = 10
    min_passes: int = 1
    min_converged: int = 1
    pct_refinement: int = 30
    basis_order: int = 1

    def __post_init__(self) -> None:
        if self.freq_ghz <= 0:
            raise ValueError("freq_ghz must be positive.")
        if self.max_passes < self.min_passes:
            raise ValueError("max_passes must be greater than or equal to min_passes.")
        if self.min_converged < 1:
            raise ValueError("min_converged must be at least 1.")

    def to_renderer_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DrivenModalSweepSpec:
    """Frequency sweep configuration attached to a driven-modal setup."""

    start_ghz: float
    stop_ghz: float
    count: int
    name: str = "Sweep"
    sweep_type: str = "Fast"
    save_fields: bool = False
    interpolation_tol: float | None = None
    interpolation_max_solutions: int | None = None

    def __post_init__(self) -> None:
        if self.stop_ghz <= self.start_ghz:
            raise ValueError("stop_ghz must be greater than start_ghz.")
        if self.count < 2:
            raise ValueError("count must be at least 2.")
        if self.interpolation_tol is not None and self.interpolation_tol <= 0:
            raise ValueError("interpolation_tol must be positive when provided.")
        if self.interpolation_max_solutions is not None and self.interpolation_max_solutions < 1:
            raise ValueError("interpolation_max_solutions must be at least 1 when provided.")

    def to_renderer_kwargs(self) -> dict[str, Any]:
        return {
            "start_ghz": self.start_ghz,
            "stop_ghz": self.stop_ghz,
            "count": self.count,
            "name": self.name,
            "type": self.sweep_type,
            "save_fields": self.save_fields,
            "interpolation_tol": self.interpolation_tol,
            "interpolation_max_solutions": self.interpolation_max_solutions,
        }


@dataclass(frozen=True)
class DrivenModalArtifactPolicy:
    """Artifact and checkpoint export controls for solver and post-processing outputs."""

    export_touchstone: bool = True
    export_y_parameters: bool = True
    export_capacitance_tables: bool = True
    checkpoint_after_stage: bool = True
    resume_existing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DrivenModalPortSpec:
    """A reusable Qiskit Metal / HFSS lumped-port declaration."""

    kind: str
    component: str
    pin: str
    impedance_ohms: float = 50.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("kind is required.")
        if not self.component:
            raise ValueError("component is required.")
        if not self.pin:
            raise ValueError("pin is required.")
        if self.impedance_ohms <= 0:
            raise ValueError("impedance_ohms must be positive.")
        _require_mapping("metadata", self.metadata)

    def to_qiskit_port_entry(self) -> tuple[str, str, float]:
        return (self.component, self.pin, float(self.impedance_ohms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "component": self.component,
            "pin": self.pin,
            "impedance_ohms": self.impedance_ohms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CapacitanceExtractionRequest:
    """Request to extract a frequency-dependent capacitance matrix from HFSS driven-modal data."""

    system_kind: str
    design_payload: Mapping[str, Any]
    layer_stack: DrivenModalLayerStackSpec
    setup: DrivenModalSetupSpec
    sweep: DrivenModalSweepSpec
    artifacts: DrivenModalArtifactPolicy
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.system_kind not in {"qubit_claw", "ncap"}:
            raise ValueError("system_kind must be 'qubit_claw' or 'ncap'.")
        _require_mapping("design_payload", self.design_payload)
        _require_mapping("metadata", self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_kind": self.system_kind,
            "design_payload": dict(self.design_payload),
            "layer_stack": self.layer_stack.to_dict(),
            "setup": asdict(self.setup),
            "sweep": asdict(self.sweep),
            "artifacts": self.artifacts.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CoupledSystemDrivenModalRequest:
    """Request to extract a multiport coupled-system response and post-process it into SQuADDS outputs."""

    resonator_type: str
    design_payload: Mapping[str, Any]
    layer_stack: DrivenModalLayerStackSpec
    setup: DrivenModalSetupSpec
    sweep: DrivenModalSweepSpec
    artifacts: DrivenModalArtifactPolicy
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.resonator_type not in {"quarter_wave", "half_wave"}:
            raise ValueError("resonator_type must be 'quarter_wave' or 'half_wave'.")
        _require_mapping("design_payload", self.design_payload)
        _require_mapping("metadata", self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resonator_type": self.resonator_type,
            "design_payload": dict(self.design_payload),
            "layer_stack": self.layer_stack.to_dict(),
            "setup": asdict(self.setup),
            "sweep": asdict(self.sweep),
            "artifacts": self.artifacts.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DrivenModalRunManifest:
    """Serializable checkpoint state for a driven-modal run."""

    run_id: str
    run_dir: str
    stages: Mapping[str, Any]
    request_payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "stages": dict(self.stages),
            "request_payload": dict(self.request_payload),
        }


@dataclass(frozen=True)
class CapacitanceExtractionResult:
    """Compact result surface for capacitance extraction workflows."""

    request: Mapping[str, Any]
    layer_stack: list[dict[str, Any]]
    sim_results: Mapping[str, Any]
    artifacts: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": dict(self.request),
            "layer_stack": list(self.layer_stack),
            "sim_results": dict(self.sim_results),
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class CoupledSystemDrivenModalResult:
    """Compact result surface for coupled-system workflows."""

    request: Mapping[str, Any]
    layer_stack: list[dict[str, Any]]
    sim_results: Mapping[str, Any]
    artifacts: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": dict(self.request),
            "layer_stack": list(self.layer_stack),
            "sim_results": dict(self.sim_results),
            "artifacts": dict(self.artifacts),
        }
