"""Leakage-resistant split utilities for topology-general transfer studies.

The utilities in this module deliberately separate three concerns:

* group construction from raw SQuADDS records;
* deterministic assignment of groups to train, validation, and test; and
* deterministic sampling of a labelled target support set.

Every split exposes ``fit_indices`` as the only indices on which learned
preprocessing may be fit.  Call :func:`validate_preprocessing_fit_indices` at
the preprocessing boundary so an accidental full-dataset fit fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

DEFAULT_SUPPORT_SIZES: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_SUPPORT_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)

GENERALIZED_FINGER_FIELDS: Mapping[str, tuple[str, ...]] = {
    "finger_count": ("finger_count",),
    "finger_length": ("finger_length",),
    "finger_width": ("finger_width",),
    "finger_gap_north_south": ("finger_gap_north_south", "finger_gap"),
    "finger_gap_east_west": ("finger_gap_east_west", "finger_gap"),
    "finger_etch_radius": ("finger_etch_radius",),
}

GENERALIZED_CPW_SPINE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "north_cpw_length": ("north_cpw_length", "cpw_length"),
    "south_cpw_length": ("south_cpw_length", "cpw_length"),
    "north_spine_width": ("north_spine_width", "spine_width"),
    "south_spine_width": ("south_spine_width", "spine_width"),
}

CAPN_MORPHOLOGY_FIELDS: Mapping[str, tuple[str, ...]] = {
    "finger_count": ("finger_count",),
    "finger_length": ("finger_length",),
    "cap_width": ("cap_width", "finger_width"),
    "cap_gap": ("cap_gap", "finger_gap"),
}

_LENGTH_TO_UM = {
    "m": 1.0e6,
    "cm": 1.0e4,
    "mm": 1.0e3,
    "um": 1.0,
    "nm": 1.0e-3,
    "pm": 1.0e-6,
}
_NUMBER_WITH_UNIT = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)?\s*$")


def _readonly_indices(values: Iterable[int], *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(array < 0):
        raise ValueError(f"{name} contains a negative index.")
    unique = np.unique(array)
    if len(unique) != len(array):
        raise ValueError(f"{name} contains duplicate indices.")
    array = unique
    array.setflags(write=False)
    return array


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _group_token(value: Any) -> str:
    """Return a stable, type-preserving token for an arbitrary group label."""

    def typed_payload(item: Any) -> Any:
        if isinstance(item, np.generic):
            item = item.item()
        if item is None:
            return {"type": "none"}
        if isinstance(item, bool):
            return {"type": "bool", "value": item}
        if isinstance(item, int):
            return {"type": "int", "value": str(item)}
        if isinstance(item, float):
            if math.isnan(item):
                value_string = "nan"
            elif math.isinf(item):
                value_string = "+inf" if item > 0.0 else "-inf"
            else:
                value_string = item.hex()
            return {"type": "float", "value": value_string}
        if isinstance(item, str):
            return {"type": "str", "value": item}
        if isinstance(item, bytes):
            return {"type": "bytes", "value": item.hex()}
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [typed_payload(part) for part in item]}
        if isinstance(item, list):
            return {"type": "list", "items": [typed_payload(part) for part in item]}
        if isinstance(item, Mapping):
            pairs = [[typed_payload(key), typed_payload(part)] for key, part in item.items()]
            pairs.sort(key=_canonical_json)
            return {"type": "mapping", "items": pairs}
        type_name = f"{type(item).__module__}.{type(item).__qualname__}"
        return {"type": type_name, "value": str(item)}

    return _canonical_json(typed_payload(value))


def _group_order_key(value: Any) -> tuple[Any, ...]:
    """Order numeric domains numerically while retaining deterministic mixed-type behavior."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        return (0, float(value), _group_token(value))
    return (1, _group_token(value))


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = _canonical_json([int(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _canonical_scalar(value: Any) -> str:
    """Canonicalize scalar geometry values, normalizing physical lengths to um."""
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value))):
        return "<missing>"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".15g")

    text = str(value).strip().replace("µ", "u").replace("μ", "u")
    match = _NUMBER_WITH_UNIT.match(text)
    if not match:
        return text
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit in _LENGTH_TO_UM:
        return f"{format(number * _LENGTH_TO_UM[unit], '.15g')}um"
    if not unit and number.is_integer():
        return str(int(number))
    return f"{format(number, '.15g')}{unit}"


def _nested_get(record: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = record
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def _candidate_paths(name: str) -> tuple[str, ...]:
    if "." in name:
        return (name,)
    return (
        name,
        f"design_options.{name}",
        f"design.design_options.{name}",
        f"notes.{name}",
    )


def _extract_field(frame: pd.DataFrame, candidates: Sequence[str], *, logical_name: str) -> pd.Series:
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate]
        for path in _candidate_paths(candidate):
            root = path.split(".", maxsplit=1)[0]
            if root not in frame.columns:
                continue
            values: list[Any] = []
            found = True
            nested_path = path.split(".")[1:]
            for root_value in frame[root]:
                try:
                    value = root_value
                    for key in nested_path:
                        if not isinstance(value, Mapping) or key not in value:
                            raise KeyError(path)
                        value = value[key]
                    values.append(value)
                except KeyError:
                    found = False
                    break
            if found:
                return pd.Series(values, index=frame.index, name=logical_name)
    tried = ", ".join(candidates)
    raise KeyError(f"Could not resolve {logical_name!r}; tried: {tried}.")


def exact_group_labels(
    frame: pd.DataFrame,
    fields: Mapping[str, Sequence[str]],
    *,
    prefix: str,
) -> np.ndarray:
    """Build exact, unit-normalized group labels from named record fields."""
    if frame.empty:
        return np.asarray([], dtype=object)
    columns = {
        logical_name: _extract_field(frame, candidates, logical_name=logical_name).map(_canonical_scalar)
        for logical_name, candidates in fields.items()
    }
    normalized = pd.DataFrame(columns, index=frame.index)
    labels = [f"{prefix}:{_canonical_json(row)}" for row in normalized.to_dict(orient="records")]
    return np.asarray(labels, dtype=object)


def generalized_ncap_group_labels(
    frame: pd.DataFrame,
    mode: Literal["finger_profile", "campaign", "cpw_spine_block"] = "finger_profile",
) -> np.ndarray:
    """Return leakage groups for GeneralizedCapNInterdigital records.

    ``finger_profile`` uses the five independent finger-shape controls.  It is
    the preferred geometry-OOD grouping because CPW/spine variants of one
    finger design cannot cross the split boundary.  ``campaign`` isolates
    acquisition campaigns.  ``cpw_spine_block`` groups exact CPW-length and
    spine-width combinations for a complementary routing-context holdout.
    """
    if mode == "finger_profile":
        return exact_group_labels(frame, GENERALIZED_FINGER_FIELDS, prefix="generalized:finger")
    if mode == "cpw_spine_block":
        return exact_group_labels(frame, GENERALIZED_CPW_SPINE_FIELDS, prefix="generalized:cpw-spine")
    if mode == "campaign":
        campaigns = _extract_field(
            frame,
            ("source_campaign", "notes.source_campaign", "campaign"),
            logical_name="source_campaign",
        )
        return np.asarray([f"generalized:campaign:{_canonical_scalar(value)}" for value in campaigns], dtype=object)
    raise ValueError(f"Unknown GeneralizedNCap grouping mode: {mode!r}.")


def capn_group_labels(
    frame: pd.DataFrame,
    mode: Literal["finger_count", "morphology"] = "finger_count",
) -> np.ndarray:
    """Return outer-domain or exact-morphology groups for CapN records."""
    if mode == "morphology":
        return exact_group_labels(frame, CAPN_MORPHOLOGY_FIELDS, prefix="capn:morphology")
    if mode == "finger_count":
        values = _extract_field(frame, ("finger_count",), logical_name="finger_count")
        return np.asarray([f"capn:finger-count:{_canonical_scalar(value)}" for value in values], dtype=object)
    raise ValueError(f"Unknown CapN grouping mode: {mode!r}.")


@dataclass(frozen=True)
class GroupedSplit:
    """Immutable train/validation/test indices with auditable group labels."""

    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    train_groups: tuple[str, ...] = ()
    validation_groups: tuple[str, ...] = ()
    test_groups: tuple[str, ...] = ()
    group_name: str = "group"
    seed: int = 0
    name: str = "split"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_idx", _readonly_indices(self.train_idx, name="train_idx"))
        object.__setattr__(
            self,
            "validation_idx",
            _readonly_indices(self.validation_idx, name="validation_idx"),
        )
        object.__setattr__(self, "test_idx", _readonly_indices(self.test_idx, name="test_idx"))
        object.__setattr__(self, "train_groups", tuple(sorted(map(str, self.train_groups))))
        object.__setattr__(self, "validation_groups", tuple(sorted(map(str, self.validation_groups))))
        object.__setattr__(self, "test_groups", tuple(sorted(map(str, self.test_groups))))
        object.__setattr__(self, "metadata", dict(self.metadata))

        partitions = {
            "train": set(self.train_idx.tolist()),
            "validation": set(self.validation_idx.tolist()),
            "test": set(self.test_idx.tolist()),
        }
        names = tuple(partitions)
        for left_pos, left in enumerate(names):
            for right in names[left_pos + 1 :]:
                overlap = partitions[left] & partitions[right]
                if overlap:
                    preview = sorted(overlap)[:5]
                    raise ValueError(f"{left}/{right} index overlap: {preview}.")

        group_partitions = {
            "train": set(self.train_groups),
            "validation": set(self.validation_groups),
            "test": set(self.test_groups),
        }
        for left_pos, left in enumerate(names):
            for right in names[left_pos + 1 :]:
                overlap = group_partitions[left] & group_partitions[right]
                if overlap:
                    preview = sorted(overlap)[:3]
                    raise ValueError(f"{left}/{right} {self.group_name} overlap: {preview}.")

    @property
    def fit_indices(self) -> np.ndarray:
        """Indices authorized for fitting preprocessors and model parameters."""
        result = self.train_idx.copy()
        result.setflags(write=False)
        return result

    @property
    def evaluation_indices(self) -> np.ndarray:
        """Indices reserved for final evaluation."""
        result = self.test_idx.copy()
        result.setflags(write=False)
        return result

    def validate(self, groups: Sequence[Any]) -> None:
        """Validate bounds and prove that no observed group crosses partitions."""
        validate_group_separation(self, groups)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable split specification."""
        return {
            "name": self.name,
            "group_name": self.group_name,
            "seed": int(self.seed),
            "train_idx": self.train_idx.tolist(),
            "validation_idx": self.validation_idx.tolist(),
            "test_idx": self.test_idx.tolist(),
            "train_groups": list(self.train_groups),
            "validation_groups": list(self.validation_groups),
            "test_groups": list(self.test_groups),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NestedTargetSplit:
    """An outer target-domain split plus a labelled adaptation subset."""

    outer: GroupedSplit
    adaptation: GroupedSplit
    requested_support: int | float
    actual_support_size: int
    unused_target_idx: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unused_target_idx",
            _readonly_indices(self.unused_target_idx, name="unused_target_idx"),
        )
        if self.actual_support_size != len(self.adaptation.train_idx):
            raise ValueError("actual_support_size does not match adaptation.train_idx.")
        if isinstance(self.requested_support, (bool, np.bool_)) or not isinstance(
            self.requested_support,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("requested_support must be a count or fraction.")
        if not np.isfinite(self.requested_support) or self.requested_support < 0:
            raise ValueError("requested_support must be finite and non-negative.")
        outer_train = set(self.outer.train_idx.tolist())
        outer_validation = set(self.outer.validation_idx.tolist())
        outer_test = set(self.outer.test_idx.tolist())
        support = set(self.adaptation.train_idx.tolist())
        validation = set(self.adaptation.validation_idx.tolist())
        test = set(self.adaptation.test_idx.tolist())
        unused = set(self.unused_target_idx.tolist())
        if not support.issubset(outer_train):
            raise ValueError("Nested support indices must come from outer.train_idx.")
        if validation != outer_validation:
            raise ValueError("Nested validation indices must equal outer.validation_idx.")
        if test != outer_test:
            raise ValueError("Nested test indices must equal outer.test_idx.")
        if support & unused:
            raise ValueError("Nested support and unused target indices overlap.")
        if support | unused != outer_train:
            raise ValueError("Support and unused target indices must partition outer.train_idx.")

    @property
    def fit_indices(self) -> np.ndarray:
        """Train-only preprocessing indices for the target adaptation stage."""
        return self.adaptation.fit_indices

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer": self.outer.to_dict(),
            "adaptation": self.adaptation.to_dict(),
            "requested_support": self.requested_support,
            "actual_support_size": int(self.actual_support_size),
            "unused_target_idx": self.unused_target_idx.tolist(),
        }


@dataclass(frozen=True)
class SupportLevel:
    """One resolved point on a deterministic target-label learning curve."""

    size: int
    requested_by: tuple[str, ...]


def validate_group_separation(split: GroupedSplit, groups: Sequence[Any]) -> None:
    """Raise if an index is out of bounds or a group crosses a split boundary."""
    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    all_indices = np.concatenate((split.train_idx, split.validation_idx, split.test_idx))
    if all_indices.size and int(all_indices.max()) >= len(group_array):
        raise IndexError("Split index exceeds the group-label array.")

    observed: dict[str, set[str]] = {}
    for partition_name, indices in (
        ("train", split.train_idx),
        ("validation", split.validation_idx),
        ("test", split.test_idx),
    ):
        for value in group_array[indices]:
            observed.setdefault(_group_token(value), set()).add(partition_name)
    leaked = {group: names for group, names in observed.items() if len(names) > 1}
    if leaked:
        preview = list(leaked.items())[:3]
        raise ValueError(f"{split.group_name} leakage across partitions: {preview}.")

    declared = {
        "train": set(split.train_groups),
        "validation": set(split.validation_groups),
        "test": set(split.test_groups),
    }
    for partition_name, indices in (
        ("train", split.train_idx),
        ("validation", split.validation_idx),
        ("test", split.test_idx),
    ):
        observed_tokens = {_group_token(value) for value in group_array[indices]}
        if declared[partition_name] and observed_tokens != declared[partition_name]:
            raise ValueError(f"Declared {partition_name} groups do not match groups observed at its indices.")


def validate_preprocessing_fit_indices(
    split: GroupedSplit | NestedTargetSplit,
    fit_indices: Iterable[int],
    *,
    require_all_train: bool = False,
) -> np.ndarray:
    """Validate that learned preprocessing uses train indices only.

    The returned array is sorted, unique, and read-only.  Validation and test
    indices are never accepted.  ``require_all_train=True`` additionally
    requires an exact match with the split's complete training partition.
    """
    proposed = _readonly_indices(fit_indices, name="fit_indices")
    authorized = split.fit_indices
    unauthorized = set(proposed.tolist()) - set(authorized.tolist())
    if unauthorized:
        raise ValueError(f"Preprocessing fit includes non-training indices: {sorted(unauthorized)[:5]}.")
    if require_all_train and not np.array_equal(proposed, authorized):
        raise ValueError("Preprocessing fit indices must exactly equal the training indices.")
    return proposed


def make_grouped_split(
    groups: Sequence[Any],
    *,
    test_groups: Iterable[Any],
    validation_groups: Iterable[Any] = (),
    group_name: str = "group",
    seed: int = 0,
    name: str = "group-holdout",
    metadata: Mapping[str, Any] | None = None,
) -> GroupedSplit:
    """Assign complete groups to deterministic train/validation/test partitions."""
    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    tokens = np.asarray([_group_token(value) for value in group_array], dtype=object)
    test_tokens = {_group_token(value) for value in test_groups}
    validation_tokens = {_group_token(value) for value in validation_groups}
    if test_tokens & validation_tokens:
        raise ValueError("test_groups and validation_groups overlap.")
    known = set(tokens.tolist())
    unknown = (test_tokens | validation_tokens) - known
    if unknown:
        raise ValueError(f"Requested holdout groups are absent: {sorted(unknown)[:3]}.")

    test_mask = np.isin(tokens, list(test_tokens))
    validation_mask = np.isin(tokens, list(validation_tokens))
    train_mask = ~(test_mask | validation_mask)
    split = GroupedSplit(
        train_idx=np.flatnonzero(train_mask),
        validation_idx=np.flatnonzero(validation_mask),
        test_idx=np.flatnonzero(test_mask),
        train_groups=tuple(sorted(set(tokens[train_mask].tolist()))),
        validation_groups=tuple(sorted(set(tokens[validation_mask].tolist()))),
        test_groups=tuple(sorted(set(tokens[test_mask].tolist()))),
        group_name=group_name,
        seed=seed,
        name=name,
        metadata=dict(metadata or {}),
    )
    split.validate(group_array)
    return split


def leave_one_group_out_splits(
    groups: Sequence[Any],
    *,
    validation: Literal["next", "previous", "none"] = "next",
    seed: int = 0,
    group_name: str = "group",
    name_prefix: str = "logo",
) -> tuple[GroupedSplit, ...]:
    """Create deterministic outer splits with one complete test group each."""
    group_array = np.asarray(groups, dtype=object)
    tokens = np.asarray([_group_token(value) for value in group_array], dtype=object)
    token_set = set(tokens.tolist())
    representatives = {token: group_array[int(np.flatnonzero(tokens == token)[0])] for token in token_set}
    unique = sorted(token_set, key=lambda token: _group_order_key(representatives[token]))
    if len(unique) < 2:
        raise ValueError("At least two groups are required for an outer holdout.")
    if validation != "none" and len(unique) < 3:
        raise ValueError("At least three groups are required with a validation holdout.")

    offset = 1 if validation == "next" else -1
    results: list[GroupedSplit] = []
    for position, test_token in enumerate(unique):
        validation_groups: tuple[Any, ...] = ()
        validation_token: str | None = None
        if validation != "none":
            validation_token = unique[(position + offset) % len(unique)]
            validation_groups = (representatives[validation_token],)
        results.append(
            make_grouped_split(
                group_array,
                test_groups=(representatives[test_token],),
                validation_groups=validation_groups,
                group_name=group_name,
                seed=seed,
                name=f"{name_prefix}:{position:03d}",
                metadata={
                    "outer_position": position,
                    "test_group": test_token,
                    "validation_group": validation_token,
                },
            )
        )
    return tuple(results)


def generalized_ncap_holdouts(
    frame: pd.DataFrame,
    *,
    mode: Literal["finger_profile", "campaign", "cpw_spine_block"] = "campaign",
    validation: Literal["next", "previous", "none"] = "next",
    seed: int = 0,
) -> tuple[GroupedSplit, ...]:
    """Create complete GeneralizedNCap campaign or geometry-block holdouts."""
    groups = generalized_ncap_group_labels(frame, mode=mode)
    return leave_one_group_out_splits(
        groups,
        validation=validation,
        seed=seed,
        group_name=f"generalized:{mode}",
        name_prefix=f"generalized-{mode}",
    )


def capn_outer_splits(
    frame: pd.DataFrame,
    *,
    mode: Literal["finger_count", "morphology"] = "finger_count",
    validation: Literal["next", "previous", "none"] = "next",
    seed: int = 0,
) -> tuple[GroupedSplit, ...]:
    """Create CapN outer splits for cross-class transfer evaluation."""
    groups = capn_group_labels(frame, mode=mode)
    return leave_one_group_out_splits(
        groups,
        validation=validation,
        seed=seed,
        group_name=f"capn:{mode}",
        name_prefix=f"capn-{mode}",
    )


def _deterministic_sample(indices: np.ndarray, size: int, *, seed: int, label: str) -> np.ndarray:
    if size < 0:
        raise ValueError("support size must be non-negative.")
    if size >= len(indices):
        selected = np.sort(indices.copy())
    elif size == 0:
        selected = np.asarray([], dtype=np.int64)
    else:
        pool = np.sort(np.asarray(indices, dtype=np.int64))
        pool_digest = hashlib.sha256(_canonical_json(pool.tolist()).encode("ascii")).hexdigest()
        rng = np.random.default_rng(_stable_seed(seed, label, pool_digest))
        selected = np.sort(rng.permutation(pool)[:size])
    selected.setflags(write=False)
    return selected


def resolved_support_schedule(
    pool_size: int,
    *,
    sizes: Sequence[int] = DEFAULT_SUPPORT_SIZES,
    fractions: Sequence[float] = DEFAULT_SUPPORT_FRACTIONS,
    include_full: bool = True,
) -> tuple[SupportLevel, ...]:
    """Resolve absolute and fractional support requests into unique sample counts."""
    if pool_size < 0:
        raise ValueError("pool_size must be non-negative.")
    requests: dict[int, list[str]] = {}
    for size in sizes:
        if isinstance(size, (bool, np.bool_)) or int(size) != size or size < 0:
            raise ValueError(f"Invalid support size: {size!r}.")
        resolved = min(int(size), pool_size)
        requests.setdefault(resolved, []).append(f"K={int(size)}")
    for fraction in fractions:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"Support fraction must lie in [0, 1], got {fraction!r}.")
        resolved = 0 if fraction == 0.0 else min(pool_size, max(1, int(math.ceil(pool_size * float(fraction)))))
        requests.setdefault(resolved, []).append(f"fraction={format(float(fraction), '.6g')}")
    if include_full:
        requests.setdefault(pool_size, []).append("full")
    return tuple(SupportLevel(size=size, requested_by=tuple(requests[size])) for size in sorted(requests))


def make_nested_target_split(
    outer: GroupedSplit,
    groups: Sequence[Any],
    *,
    support_size: int | None = None,
    support_fraction: float | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> NestedTargetSplit:
    """Select a deterministic labelled support set inside an outer target split.

    Exactly one of ``support_size`` and ``support_fraction`` must be supplied.
    Sampling occurs only inside ``outer.train_idx``.  Validation and test groups
    remain untouched, and unused training-pool rows are explicitly recorded.
    """
    if (support_size is None) == (support_fraction is None):
        raise ValueError("Provide exactly one of support_size or support_fraction.")
    if isinstance(support_fraction, (bool, np.bool_)):
        raise ValueError("support_fraction must be a real number in [0, 1].")
    if support_fraction is not None and not 0.0 <= support_fraction <= 1.0:
        raise ValueError("support_fraction must lie in [0, 1].")

    requested: int | float
    if support_size is not None:
        if (
            isinstance(support_size, (bool, np.bool_))
            or not isinstance(support_size, (int, np.integer))
            or support_size < 0
        ):
            raise ValueError("support_size must be a non-negative integer.")
        resolved_size = min(int(support_size), len(outer.train_idx))
        requested = int(support_size)
    else:
        fraction = float(support_fraction)
        resolved_size = (
            0
            if fraction == 0.0
            else min(
                len(outer.train_idx),
                max(1, int(math.ceil(len(outer.train_idx) * fraction))),
            )
        )
        requested = fraction

    effective_seed = outer.seed if seed is None else int(seed)
    selected = _deterministic_sample(
        outer.train_idx,
        resolved_size,
        seed=effective_seed,
        label=f"{outer.name}:target-support",
    )
    selected_set = set(selected.tolist())
    unused = np.asarray([index for index in outer.train_idx if int(index) not in selected_set], dtype=np.int64)

    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    if len(group_array) <= max(
        [*outer.train_idx.tolist(), *outer.validation_idx.tolist(), *outer.test_idx.tolist()],
        default=-1,
    ):
        raise IndexError("Outer split index exceeds the group-label array.")
    adaptation = GroupedSplit(
        train_idx=selected,
        validation_idx=outer.validation_idx,
        test_idx=outer.test_idx,
        train_groups=tuple(sorted({_group_token(value) for value in group_array[selected]})),
        validation_groups=tuple(sorted({_group_token(value) for value in group_array[outer.validation_idx]})),
        test_groups=tuple(sorted({_group_token(value) for value in group_array[outer.test_idx]})),
        group_name=f"{outer.group_name}:nested",
        seed=effective_seed,
        name=name or f"{outer.name}:support-{resolved_size}",
        metadata={
            **dict(outer.metadata),
            "outer_name": outer.name,
            "requested_support": requested,
            "actual_support_size": resolved_size,
        },
    )
    adaptation.validate(group_array)
    return NestedTargetSplit(
        outer=outer,
        adaptation=adaptation,
        requested_support=requested,
        actual_support_size=resolved_size,
        unused_target_idx=unused,
    )
