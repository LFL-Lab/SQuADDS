"""Selection helpers for the `SQuADDS_DB` compatibility facade."""

from __future__ import annotations

LEGACY_RESONATOR_TO_COUPLER = {"quarter": "CLT", "half": "NCap"}


def resolve_system_selection(
    components, supported_components: list[str]
) -> tuple[list[str] | str | None, str | None, str | None]:
    """Resolve selected system/component state while preserving legacy semantics."""
    if isinstance(components, list):
        for component in components:
            if component not in supported_components:
                return None, None, component
        return components, None, None

    if isinstance(components, str):
        if components not in supported_components:
            return None, None, components
        return components, components, None

    return None, None, None


def build_component_selection(
    selected_system,
    required_component: str,
    component_name: str | None,
    data_type: str,
    warning_message: str,
) -> tuple[str | None, str]:
    """Return the selected component name and data type for matching systems."""
    if (selected_system == required_component) or (required_component in selected_system):
        return component_name, data_type
    raise UserWarning(warning_message)


def resolve_resonator_coupler(resonator_type: str) -> str:
    """Map resonator types to their legacy coupler aliases."""
    if resonator_type not in LEGACY_RESONATOR_TO_COUPLER:
        raise ValueError(
            f"Invalid resonator type: {resonator_type}. Must be one of {list(LEGACY_RESONATOR_TO_COUPLER.keys())}."
        )
    return LEGACY_RESONATOR_TO_COUPLER[resonator_type]


def is_supported_coupler(coupler: str | None, supported_component_names: list[str]) -> bool:
    """Check coupler support while preserving the legacy CLT alias."""
    return coupler in supported_component_names + ["CLT"]
