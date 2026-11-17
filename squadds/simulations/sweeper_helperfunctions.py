"""Helpers for expanding nested sweep dictionaries into concrete combinations."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

NestedDict = dict[str, Any]

__all__ = [
    "expand_design_options",
    "extract_QSweep_parameters",
    "extract_parameters",
    "extract_values",
    "generate_combinations",
    "create_dict_list",
]


def expand_design_options(options: NestedDict) -> list[NestedDict]:
    """Expand list-valued design parameters into one design per combination.

    Every list at a leaf of ``options`` is a parameter to sweep, and the result is the grid of all combinations
    (cartesian product, last list fastest). Leaves with the same name and the same list are swept together, so a
    parameter that appears in several blocks (e.g. ``claw_length`` in both the qubit and the cavity options of a
    coupled design) keeps one value per design. Everything else is copied unchanged; use a tuple to pass a value that
    must stay a list-like constant. Options without lists return ``[copy of options]``.

    Example::

        designs = expand_design_options(
            {"cross_length": ["200um", "250um"], "connection_pads": {"readout": {"claw_length": ["150um", "200um"]}}}
        )
        len(designs)  # 4; designs[1] == {"cross_length": "200um", "connection_pads": {"readout": {"claw_length": "200um"}}}

    Raises:
        ValueError: if a list is empty.
    """
    groups: dict[tuple[str, str], list] = {}
    paths: dict[tuple[str, str], list[tuple[str, ...]]] = {}

    def walk(node: NestedDict, path: tuple[str, ...]) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, (*path, key))
            elif isinstance(value, list):
                if not value:
                    raise ValueError(f"Empty list for design parameter '{'.'.join((*path, key))}'.")
                group = (key, repr(value))
                groups.setdefault(group, value)
                paths.setdefault(group, []).append((*path, key))

    walk(options, ())
    designs = []
    for combination in product(*groups.values()):
        design = deepcopy(options)
        for group, value in zip(groups, combination, strict=True):
            for path in paths[group]:
                target = design
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
        designs.append(design)
    return designs


def extract_QSweep_parameters(parameters: NestedDict) -> list[NestedDict]:
    """Return every concrete parameter combination from a nested sweep dictionary.

    The public API is intentionally unchanged. Nested dictionaries are flattened
    into dotted paths, leaf values are converted into lists when needed, and the
    cartesian product of all leaf lists is reconstructed back into nested
    dictionaries.
    """

    flattened_keys = extract_parameters(parameters)
    flattened_values = extract_values(parameters)
    combinations = generate_combinations(flattened_values)
    return create_dict_list(flattened_keys, combinations)


def extract_parameters(dictionary: NestedDict, keys: list[str] | None = None, prefix: str = "") -> list[str]:
    """Flatten nested keys into dotted paths in traversal order."""

    if keys is None:
        keys = []

    for key, value in dictionary.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            extract_parameters(value, keys=keys, prefix=f"{full_key}.")
        else:
            keys.append(full_key)

    return keys


def as_list(value: Any) -> list[Any]:
    """Return the input as a list while preserving existing lists."""

    return value if isinstance(value, list) else [value]


def extract_values(dictionary: NestedDict, values: list[list[Any]] | None = None) -> list[list[Any]]:
    """Collect nested leaf values in traversal order, coercing scalars to lists."""

    if values is None:
        values = []

    for value in dictionary.values():
        if isinstance(value, dict):
            extract_values(value, values=values)
        else:
            values.append(as_list(value))

    return values


def generate_combinations(value_lists: list[list[Any]]) -> list[tuple[Any, ...]]:
    """Return the cartesian product for a list of leaf-value lists."""

    return list(product(*value_lists))


def _assign_nested_value(target: NestedDict, dotted_key: str, value: Any) -> None:
    """Assign a leaf value inside ``target`` using a dotted path."""

    current = target
    parts = dotted_key.split(".")

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def create_dict_list(keys: list[str], values: list[tuple[Any, ...]]) -> list[NestedDict]:
    """Rebuild nested dictionaries for every key/value combination."""

    dict_list: list[NestedDict] = []

    for value_tuple in values:
        nested_dict: NestedDict = {}
        for dotted_key, leaf_value in zip(keys, value_tuple, strict=True):
            _assign_nested_value(nested_dict, dotted_key, leaf_value)
        dict_list.append(nested_dict)

    return dict_list
