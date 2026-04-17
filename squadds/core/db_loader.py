"""Dataset request helpers for the `SQuADDS_DB` compatibility facade."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRequestValidation:
    """Validation result for a dataset lookup request."""

    is_valid: bool
    message: str | None = None
    options: list[str] | None = None


def validate_dataset_request(
    component,
    component_name,
    data_type,
    supported_components: list[str],
    supported_component_names: list[str],
    supported_data_types: list[str],
) -> DatasetRequestValidation:
    """Validate a dataset request using the legacy message and option payloads."""
    if component is None or component_name is None:
        return DatasetRequestValidation(False, "Both system and component name must be defined.")

    if data_type is None:
        return DatasetRequestValidation(False, "Please specify a data type.")

    if component not in supported_components:
        return DatasetRequestValidation(
            False, "Component not supported. Available components are:", supported_components
        )

    if component_name not in supported_component_names:
        return DatasetRequestValidation(
            False,
            "Component name not supported. Available component names are:",
            supported_component_names,
        )

    if data_type not in supported_data_types:
        return DatasetRequestValidation(
            False, "Data type not supported. Available data types are:", supported_data_types
        )

    return DatasetRequestValidation(True)


def build_dataset_config(component: str, component_name: str, data_type: str) -> str:
    """Build the legacy dataset config identifier."""
    return f"{component}-{component_name}-{data_type}"
