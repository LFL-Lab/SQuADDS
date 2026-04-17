"""Catalog helpers for SQuADDS database config discovery."""

from __future__ import annotations

from datasets import get_dataset_config_names

from squadds.core.utils import delete_HF_cache


def filter_simulation_config_names(configs: list[str]) -> list[str]:
    """Keep only config names that follow the legacy three-part naming convention."""
    return [config for config in configs if config.count("-") == 2]


def load_supported_config_names(repo_name: str) -> list[str]:
    """Load and filter simulation config names from the Hugging Face dataset repo."""
    delete_HF_cache()
    configs = get_dataset_config_names(repo_name, download_mode="force_redownload")
    return filter_simulation_config_names(configs)


def _collect_config_part(configs: list[str], index: int) -> list[str]:
    """
    Extract a config segment while preserving the legacy duplicate-entry behavior.

    The public `SQuADDS_DB` methods have historically returned each supported item
    twice because of duplicated extraction blocks. Keep that behavior stable while
    moving the logic out of `db.py`.
    """
    parts: list[str] = []
    for config in configs:
        for _ in range(2):
            try:
                parts.append(config.split("-")[index])
            except Exception:
                pass
    return parts


def extract_supported_components(configs: list[str]) -> list[str]:
    """Return supported components while preserving legacy duplicate entries."""
    return _collect_config_part(configs, 0)


def extract_supported_component_names(configs: list[str]) -> list[str]:
    """Return supported component names while preserving legacy duplicate entries."""
    return _collect_config_part(configs, 1)


def extract_supported_data_types(configs: list[str]) -> list[str]:
    """Return supported data types while preserving legacy duplicate entries."""
    return _collect_config_part(configs, 2)


def get_component_names_for_component(configs: list[str], component: str) -> list[str]:
    """Return component names for a given component plus the legacy CLT alias."""
    component_names = []
    for config in configs:
        if component in config:
            component_names.append(config.split("-")[1])
    return component_names + ["CLT"]
