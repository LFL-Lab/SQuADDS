"""Offline catalogue, target, and cache utilities for TopoCap.

The geometry pipeline in this module is deliberately target blind.  A GDS is
first aligned to an explicit net sidecar and converted to a numerical graph;
only then is its independently parsed Maxwell matrix attached to the graph.
This boundary makes it difficult for simulation labels to leak into geometry
or parameter preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .geometry_graph import (
    BOUNDARY_SAMPLE_COUNT,
    EDGE_FEATURE_NAMES,
    GEOMETRY_GRAPH_SCHEMA_VERSION,
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    PROXIMITY_SCALES_UM,
    build_geometry_graph_record,
    to_capacitance_graph,
)
from .net_extraction import (
    NET_SIDECAR_SCHEMA_VERSION,
    build_capn_interdigital_tee_sidecar,
    build_generalized_ncap_sidecar,
    canonical_json,
    content_sha256,
    read_gds_inventory,
    sha256_file,
)
from .schema import CapacitanceGraph
from .targets import maxwell_to_components

GENERALIZED_FAMILY = "GeneralizedCapNInterdigital"
CAPN_FAMILY = "CapNInterdigitalTee"
GENERALIZED_SIMULATION_CONFIG = "coupler-GeneralizedCapNInterdigital-cap_matrix"
CAPN_SIMULATION_CONFIG = "coupler-CapNInterdigitalTee-cap_matrix"
CAPN_CAMPAIGN = "generated_from_cap_matrix"

EXPECTED_GENERALIZED_RECORDS = 20_062
EXPECTED_CAPN_RECORDS = 894

CACHE_SCHEMA_VERSION = "topocap-catalogue-cache-1.0.0"
CACHE_MANIFEST_SCHEMA_VERSION = "topocap-cache-manifest-1.0.0"
CACHE_JSONL_NAME = "graphs.jsonl"
CACHE_MANIFEST_NAME = "cache-manifest.json"
SIDECAR_DIRECTORY_NAME = "sidecars"

CAPACITANCE_FACTORS_TO_FF: Mapping[str, float] = {
    "F": 1.0e15,
    "pF": 1.0e3,
    "fF": 1.0,
}

PARAMETER_DIMENSIONS = (
    "count",
    "length_um",
    "angle_deg",
    "dimensionless",
    "unknown_physical_scale",
)
PARAMETER_ROLES = (
    "count",
    "length",
    "width",
    "gap",
    "radius",
    "angle",
    "offset_position",
    "thickness",
    "active_coupling_region",
    "feed",
    "ground_clearance",
    "other",
)
PARAMETER_FEATURE_NAMES = (
    *(f"dimension_{name}" for name in PARAMETER_DIMENSIONS),
    "unit_explicit",
    *(f"role_present_{name}" for name in PARAMETER_ROLES),
    *(f"role_value_{name}" for name in PARAMETER_ROLES),
)

_GENERALIZED_MATRIX_KEYS = (
    ("C_G_G", "C_G_N", "C_G_S"),
    ("C_N_G", "C_N_N", "C_N_S"),
    ("C_S_G", "C_S_N", "C_S_S"),
)
_CAPN_RESULT_KEYS = (
    "ground_to_ground",
    "top_to_top",
    "top_to_bottom",
    "top_to_ground",
    "bottom_to_bottom",
    "bottom_to_ground",
)
_NUMBER_WITH_UNIT = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z\u00b5\u03bc\u00b0]+)?\s*$"
)
_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_EXP_SOURCE_ID = re.compile(r"^(exp[67])/(cap_\d{4})$")
_Q3D_SOURCE_FILE = re.compile(r"^(q3d_cap_(\d{5}))/(cap_(\d{5}))\.json$")

_LENGTH_TO_UM = {
    "m": 1.0e6,
    "cm": 1.0e4,
    "mm": 1.0e3,
    "um": 1.0,
    "nm": 1.0e-3,
    "pm": 1.0e-6,
}
_ANGLE_TO_DEG = {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "\u00b0": 1.0, "rad": 180.0 / math.pi}
_DIRECTION_WORDS = {"north", "south", "top", "bottom"}


class CatalogueError(ValueError):
    """Base class for an invalid catalogue or cache contract."""


class CapacitanceUnitError(CatalogueError):
    """Raised when simulation capacitance units are missing or ambiguous."""


class CataloguePairingError(CatalogueError):
    """Raised when a simulation row cannot be paired immutably to one GDS."""


class ParameterTokenError(CatalogueError):
    """Raised when a numeric design parameter has an unsupported unit."""


class CacheVerificationError(CatalogueError):
    """Raised when a cache is incomplete, stale, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class PairingDiagnostic:
    """One explicitly skipped GeneralizedNCap row with its expected artifact."""

    row_index: int
    source_id: str
    expected_gds_path: Path
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic."""
        return {
            "row_index": self.row_index,
            "source_id": self.source_id,
            "expected_gds_path": str(self.expected_gds_path),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CatalogueEntry:
    """Immutable pairing between one SQuADDS row and one local GDS artifact."""

    family: str
    dataset_role: Literal["source", "target"]
    row_index: int
    source_id: str
    campaign: str
    source_file: str
    gds_path: Path
    gds_locator: str
    design_id: str
    row_sha256: str
    dataset_sha256: str
    simulation_config: str
    row: Mapping[str, Any]

    @property
    def cache_key(self) -> str:
        """Stable family-qualified key used by cache and checkpoint code."""
        return f"{self.family}:{self.source_id}"

    @property
    def design_options(self) -> Mapping[str, Any]:
        """Return the exact parametric design options for feedback adapters."""
        try:
            options = self.row["design"]["design_options"]
        except (KeyError, TypeError) as exc:
            raise CataloguePairingError(f"{self.cache_key} has no design.design_options mapping.") from exc
        if not isinstance(options, Mapping):
            raise CataloguePairingError(f"{self.cache_key} design.design_options must be a mapping.")
        return options

    def public_metadata(self) -> dict[str, Any]:
        """Return portable provenance without a machine-specific absolute path."""
        return {
            "family": self.family,
            "dataset_role": self.dataset_role,
            "row_index": self.row_index,
            "source_id": self.source_id,
            "campaign": self.campaign,
            "source_file": self.source_file,
            "gds_locator": self.gds_locator,
            "design_id": self.design_id,
            "row_sha256": self.row_sha256,
            "dataset_sha256": self.dataset_sha256,
            "simulation_config": self.simulation_config,
        }


@dataclass(frozen=True, slots=True)
class CatalogueManifest:
    """Deterministic pairings plus explicit diagnostics for unmatched rows."""

    family: str
    entries: tuple[CatalogueEntry, ...]
    diagnostics: tuple[PairingDiagnostic, ...]
    source_row_count: int
    expected_count: int | None
    dataset_sha256: str

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CatalogueEntry]:
        return iter(self.entries)

    @property
    def is_complete(self) -> bool:
        """Whether every source row is paired and the expected count is met."""
        count_ok = self.expected_count is None or len(self.entries) == self.expected_count
        return not self.diagnostics and len(self.entries) == self.source_row_count and count_ok

    def assert_complete(self) -> None:
        """Fail with useful diagnostics unless this is an exact release pairing."""
        if self.is_complete:
            return
        expected = self.expected_count if self.expected_count is not None else self.source_row_count
        preview = "; ".join(f"row {item.row_index} {item.source_id}: {item.reason}" for item in self.diagnostics[:5])
        suffix = "" if len(self.diagnostics) <= 5 else f"; and {len(self.diagnostics) - 5} more"
        raise CataloguePairingError(
            f"{self.family} paired {len(self.entries)} of {self.source_row_count} rows "
            f"(expected {expected}). {preview}{suffix}".rstrip()
        )


@dataclass(frozen=True, slots=True)
class ParameterTokens:
    """Numeric model tokens plus exact raw feedback values."""

    names: tuple[str, ...]
    values: NDArray[np.float64]
    features: NDArray[np.float64]
    raw_flattened: Mapping[str, Any]

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True, slots=True)
class BuiltCatalogueGraph:
    """Target-blind geometry artifacts and the subsequently attached graph."""

    entry: CatalogueEntry
    sidecar: Mapping[str, Any]
    graph_record: Mapping[str, Any]
    graph: CapacitanceGraph


@dataclass(frozen=True, slots=True)
class CacheBuildSummary:
    """Summary of one deterministic streaming cache build."""

    output_dir: Path
    graph_jsonl: Path
    cache_manifest: Path
    requested_count: int
    existing_count: int
    written_count: int
    graph_jsonl_sha256: str
    pipeline_sha256: str


@dataclass(frozen=True, slots=True)
class CacheVerification:
    """Successful verification result for a complete graph cache."""

    record_count: int
    sidecar_count: int
    graph_jsonl_sha256: str
    pipeline_sha256: str


def _canonical_design_id(component_name: str, design_options: Mapping[str, Any]) -> str:
    """Match :func:`squadds.layouts.canonical_design_id` without heavy imports."""
    payload = {"component_name": component_name, "design_options": dict(design_options)}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"design:sha256:{digest}"


def _read_rows(source: str | Path | Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        dataset_sha256 = sha256_file(path)
    else:
        value = list(source)
        dataset_sha256 = content_sha256(value)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise CataloguePairingError("A SQuADDS simulation source must be a JSON list of row mappings.")
    return list(value), dataset_sha256


def _explicit_capacitance_unit(row: Mapping[str, Any]) -> str:
    sim_results = row.get("sim_results")
    if not isinstance(sim_results, Mapping):
        raise CapacitanceUnitError("Simulation row has no sim_results mapping.")
    candidates: list[tuple[str, Any]] = []
    for key in ("units", "unit", "capacitance_unit"):
        if key in sim_results:
            candidates.append((f"sim_results.{key}", sim_results[key]))
        if key in row:
            candidates.append((key, row[key]))
    if not candidates:
        raise CapacitanceUnitError("Capacitance units are missing; values must be explicitly tagged as F, pF, or fF.")
    normalized: list[tuple[str, str]] = []
    for location, value in candidates:
        if not isinstance(value, str):
            raise CapacitanceUnitError(f"{location} must be one of F, pF, or fF, not {value!r}.")
        unit = value.strip()
        if unit not in CAPACITANCE_FACTORS_TO_FF:
            raise CapacitanceUnitError(
                f"Unsupported or ambiguous capacitance unit {value!r} at {location}; use F, pF, or fF."
            )
        normalized.append((location, unit))
    unique = {unit for _, unit in normalized}
    if len(unique) != 1:
        details = ", ".join(f"{location}={unit}" for location, unit in normalized)
        raise CapacitanceUnitError(f"Conflicting capacitance units: {details}.")
    return normalized[0][1]


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CatalogueError(f"{field} must be numeric, not boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CatalogueError(f"{field} must be numeric; received {value!r}.") from exc
    if not math.isfinite(numeric):
        raise CatalogueError(f"{field} must be finite; received {value!r}.")
    return numeric


def _generalized_matrix(maxwell: Any) -> NDArray[np.float64]:
    if isinstance(maxwell, Mapping):
        if "values" in maxwell:
            node_order = maxwell.get("node_order")
            if node_order not in (["G", "N", "S"], ("G", "N", "S")):
                raise CatalogueError("maxwell_matrix.values requires explicit node_order ['G', 'N', 'S'].")
            maxwell = maxwell["values"]
        else:
            required = {key for row in _GENERALIZED_MATRIX_KEYS for key in row}
            missing = sorted(required.difference(maxwell))
            extra = sorted(set(maxwell).difference(required))
            if missing or extra:
                raise CatalogueError(
                    f"Generalized maxwell_matrix keys are incomplete or ambiguous; missing={missing}, extra={extra}."
                )
            return np.asarray(
                [
                    [_finite_float(maxwell[key], field=f"sim_results.maxwell_matrix.{key}") for key in row]
                    for row in _GENERALIZED_MATRIX_KEYS
                ],
                dtype=np.float64,
            )
    matrix = np.asarray(maxwell, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise CatalogueError("Generalized maxwell_matrix must be a finite 3x3 array in [G, N, S] order.")
    return matrix


def parse_signed_maxwell_matrix_ff(row: Mapping[str, Any]) -> NDArray[np.float64]:
    """Parse one explicit-unit SQuADDS row into signed Maxwell fF.

    GeneralizedNCap rows carry the complete matrix in canonical ``[G, N, S]``
    order.  Legacy CapN rows carry positive named magnitudes; these are rebuilt
    in canonical ``[G, top, bottom]`` order with negative off-diagonals.
    """
    unit = _explicit_capacitance_unit(row)
    factor = CAPACITANCE_FACTORS_TO_FF[unit]
    sim_results = row["sim_results"]
    if "maxwell_matrix" in sim_results:
        matrix = _generalized_matrix(sim_results["maxwell_matrix"])
    else:
        missing = [key for key in _CAPN_RESULT_KEYS if key not in sim_results]
        if missing:
            raise CatalogueError(
                "Simulation row has neither a complete maxwell_matrix nor the complete legacy CapN result set; "
                f"missing={missing}."
            )
        values = {key: _finite_float(sim_results[key], field=f"sim_results.{key}") for key in _CAPN_RESULT_KEYS}
        diagonal_keys = ("ground_to_ground", "top_to_top", "bottom_to_bottom")
        mutual_keys = ("top_to_bottom", "top_to_ground", "bottom_to_ground")
        if any(values[key] <= 0.0 for key in diagonal_keys):
            raise CatalogueError("Legacy CapN diagonal capacitances must be strictly positive magnitudes.")
        if any(values[key] <= 0.0 for key in mutual_keys):
            raise CatalogueError("Legacy CapN mutual capacitances must be strictly positive magnitudes.")
        matrix = np.asarray(
            [
                [values["ground_to_ground"], -values["top_to_ground"], -values["bottom_to_ground"]],
                [-values["top_to_ground"], values["top_to_top"], -values["top_to_bottom"]],
                [-values["bottom_to_ground"], -values["top_to_bottom"], values["bottom_to_bottom"]],
            ],
            dtype=np.float64,
        )
    matrix = matrix * factor
    if not np.isfinite(matrix).all():
        raise CatalogueError("Capacitance conversion produced a non-finite matrix.")
    # TopoCap learns logs of the strictly positive graph components.  This
    # validates symmetry, signs, and residual shunts at the catalogue boundary.
    maxwell_to_components(matrix)
    matrix.setflags(write=False)
    return matrix


def _design_options(row: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    try:
        options = row["design"]["design_options"]
    except (KeyError, TypeError) as exc:
        raise CataloguePairingError(f"{context} has no design.design_options mapping.") from exc
    if not isinstance(options, Mapping):
        raise CataloguePairingError(f"{context} design.design_options must be a mapping.")
    return options


def _entry(
    *,
    family: str,
    dataset_role: Literal["source", "target"],
    row_index: int,
    source_id: str,
    campaign: str,
    source_file: str,
    gds_path: Path,
    gds_locator: str,
    dataset_sha256: str,
    simulation_config: str,
    row: Mapping[str, Any],
) -> CatalogueEntry:
    options = _design_options(row, context=f"row {row_index}")
    return CatalogueEntry(
        family=family,
        dataset_role=dataset_role,
        row_index=row_index,
        source_id=source_id,
        campaign=campaign,
        source_file=source_file,
        gds_path=gds_path,
        gds_locator=gds_locator,
        design_id=_canonical_design_id(family, options),
        row_sha256=content_sha256(row),
        dataset_sha256=dataset_sha256,
        simulation_config=simulation_config,
        row=row,
    )


def _validate_unique_entries(entries: Sequence[CatalogueEntry]) -> None:
    for label, values in (
        ("source_id", [entry.source_id for entry in entries]),
        ("design_id", [entry.design_id for entry in entries]),
        ("GDS path", [str(entry.gds_path.resolve()) for entry in entries]),
        ("cache key", [entry.cache_key for entry in entries]),
    ):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise CataloguePairingError(f"Duplicate {label} values are forbidden: {duplicates[:5]}.")


def pair_generalized_records(
    simulation_source: str | Path | Sequence[Mapping[str, Any]],
    *,
    exp6_gds_dir: str | Path,
    exp7_gds_dir: str | Path,
    q3d_gds_root: str | Path,
    expected_count: int | None = EXPECTED_GENERALIZED_RECORDS,
    dataset_role: Literal["source", "target"] = "source",
) -> CatalogueManifest:
    """Pair every GeneralizedNCap row using its immutable acquisition ID.

    Rows whose exactly derived GDS path is absent are skipped only after a
    :class:`PairingDiagnostic` is recorded.  Malformed IDs and duplicates are
    never guessed or skipped.
    """
    rows, dataset_sha256 = _read_rows(simulation_source)
    exp_dirs = {"exp6": Path(exp6_gds_dir), "exp7": Path(exp7_gds_dir)}
    q3d_root = Path(q3d_gds_root)
    entries: list[CatalogueEntry] = []
    diagnostics: list[PairingDiagnostic] = []
    seen_source_ids: set[str] = set()
    seen_design_ids: set[str] = set()
    seen_gds_paths: set[Path] = set()

    for row_index, row in enumerate(rows):
        notes = row.get("notes")
        if not isinstance(notes, Mapping):
            raise CataloguePairingError(f"Generalized row {row_index} has no notes mapping.")
        source_id = str(notes.get("source_id", ""))
        campaign = str(notes.get("source_campaign", ""))
        source_file = str(notes.get("source_file", ""))
        if not source_id or not campaign or not source_file:
            raise CataloguePairingError(
                f"Generalized row {row_index} requires notes.source_id, source_campaign, and source_file."
            )
        if source_id in seen_source_ids:
            raise CataloguePairingError(f"Duplicate Generalized source_id: {source_id}.")
        seen_source_ids.add(source_id)

        if campaign in exp_dirs:
            match = _EXP_SOURCE_ID.fullmatch(source_id)
            if match is None or match.group(1) != campaign:
                raise CataloguePairingError(f"Malformed {campaign} source_id at row {row_index}: {source_id!r}.")
            stem = match.group(2)
            if source_file != f"{campaign}/{stem}.json":
                raise CataloguePairingError(
                    f"Row {row_index} source_file {source_file!r} disagrees with source_id {source_id!r}."
                )
            gds_path = exp_dirs[campaign] / f"{stem}.gds"
            gds_locator = f"{campaign}/{stem}.gds"
        elif campaign == "q3d_cap":
            match = _Q3D_SOURCE_FILE.fullmatch(source_file)
            if match is None or match.group(2) != match.group(4):
                raise CataloguePairingError(f"Malformed q3d source_file at row {row_index}: {source_file!r}.")
            directory, _, stem, _ = match.groups()
            if source_id != f"q3d_cap/{stem}":
                raise CataloguePairingError(
                    f"Row {row_index} source_id {source_id!r} disagrees with source_file {source_file!r}."
                )
            gds_path = q3d_root / directory / f"{stem}.gds"
            gds_locator = f"q3d_cap/{directory}/{stem}.gds"
        else:
            raise CataloguePairingError(f"Unsupported Generalized campaign at row {row_index}: {campaign!r}.")

        design_id = _canonical_design_id(GENERALIZED_FAMILY, _design_options(row, context=f"row {row_index}"))
        resolved = gds_path.resolve()
        if design_id in seen_design_ids:
            raise CataloguePairingError(f"Duplicate Generalized design_id at row {row_index}: {design_id}.")
        if resolved in seen_gds_paths:
            raise CataloguePairingError(f"Duplicate Generalized GDS mapping at row {row_index}: {gds_path}.")
        seen_design_ids.add(design_id)
        seen_gds_paths.add(resolved)
        if not gds_path.is_file():
            diagnostics.append(
                PairingDiagnostic(
                    row_index=row_index,
                    source_id=source_id,
                    expected_gds_path=gds_path,
                    reason="exactly paired GDS file is missing",
                )
            )
            continue
        entries.append(
            _entry(
                family=GENERALIZED_FAMILY,
                dataset_role=dataset_role,
                row_index=row_index,
                source_id=source_id,
                campaign=campaign,
                source_file=source_file,
                gds_path=gds_path,
                gds_locator=gds_locator,
                dataset_sha256=dataset_sha256,
                simulation_config=GENERALIZED_SIMULATION_CONFIG,
                row=row,
            )
        )
    _validate_unique_entries(entries)
    return CatalogueManifest(
        family=GENERALIZED_FAMILY,
        entries=tuple(entries),
        diagnostics=tuple(diagnostics),
        source_row_count=len(rows),
        expected_count=expected_count,
        dataset_sha256=dataset_sha256,
    )


def pair_capn_records(
    simulation_source: str | Path | Sequence[Mapping[str, Any]],
    *,
    gds_dir: str | Path,
    expected_count: int | None = EXPECTED_CAPN_RECORDS,
    dataset_role: Literal["source", "target"] = "target",
) -> CatalogueManifest:
    """Pair legacy CapN rows to the immutable generated row-index mapping."""
    rows, dataset_sha256 = _read_rows(simulation_source)
    if expected_count is not None and len(rows) != expected_count:
        raise CataloguePairingError(f"CapN source contains {len(rows)} rows; expected exactly {expected_count}.")
    root = Path(gds_dir)
    entries = []
    for row_index, row in enumerate(rows):
        stem = f"capn_{row_index:04d}"
        source_id = f"{CAPN_CAMPAIGN}/{stem}"
        gds_path = root / f"{stem}.gds"
        if not gds_path.is_file():
            raise FileNotFoundError(f"Missing generated CapN GDS for immutable row {row_index}: {gds_path}")
        entries.append(
            _entry(
                family=CAPN_FAMILY,
                dataset_role=dataset_role,
                row_index=row_index,
                source_id=source_id,
                campaign=CAPN_CAMPAIGN,
                source_file=f"row-index:{row_index}",
                gds_path=gds_path,
                gds_locator=f"{CAPN_CAMPAIGN}/{stem}.gds",
                dataset_sha256=dataset_sha256,
                simulation_config=CAPN_SIMULATION_CONFIG,
                row=row,
            )
        )
    _validate_unique_entries(entries)
    return CatalogueManifest(
        family=CAPN_FAMILY,
        entries=tuple(entries),
        diagnostics=(),
        source_row_count=len(rows),
        expected_count=expected_count,
        dataset_sha256=dataset_sha256,
    )


def flatten_parameter_mapping(options: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten all raw option names and values deterministically for feedback."""
    flattened: dict[str, Any] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                name = str(key)
                visit(value[key], f"{prefix}.{name}" if prefix else name)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                visit(item, f"{prefix}.{index}" if prefix else str(index))
            return
        if not prefix:
            raise ParameterTokenError("A flattened parameter value must have a non-empty name.")
        flattened[prefix] = value.item() if isinstance(value, np.generic) else value

    visit(options, "")
    return flattened


def _parameter_roles(name: str) -> tuple[str, ...]:
    words = [word for word in re.split(r"[^a-z0-9]+", name.lower()) if word]
    original = set(words)
    # Prime and second are both translated to the same physical feed role;
    # their semantic identity is never emitted. Direction words are discarded.
    words = [word for word in words if word not in _DIRECTION_WORDS and word not in {"prime", "second"}]
    tokens = set(words)
    roles: set[str] = set()

    if tokens.intersection({"count", "num", "number", "quantity"}):
        roles.add("count")
    if tokens.intersection({"length", "distance", "extent", "size"}):
        roles.add("length")
    if tokens.intersection({"width", "diameter"}):
        roles.add("width")
    if tokens.intersection({"gap", "spacing", "separation", "clearance"}):
        roles.add("gap")
    if tokens.intersection({"radius", "radii", "curvature"}):
        roles.add("radius")
    if tokens.intersection({"angle", "orientation", "rotation", "theta"}):
        roles.add("angle")
    if tokens.intersection({"offset", "position", "pos", "xpos", "ypos", "location"}):
        roles.add("offset_position")
    if tokens.intersection({"thickness", "height", "depth"}):
        roles.add("thickness")

    if original.intersection({"finger", "cap", "capacitor", "spine"}):
        roles.add("active_coupling_region")
    if original.intersection({"cpw", "feed", "lead", "port", "prime", "second"}):
        roles.add("feed")
    if "ground" in original and original.intersection({"gap", "distance", "spacing", "clearance", "offset"}):
        roles.add("ground_clearance")
    if not roles:
        roles.add("other")
    return tuple(role for role in PARAMETER_ROLES if role in roles)


def _parameter_numeric(value: Any, roles: Sequence[str], *, name: str) -> tuple[float, str, bool] | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    explicit_unit = False
    unit = ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    elif isinstance(value, str):
        normalized = value.strip().replace("\u00b5", "u").replace("\u03bc", "u")
        match = _NUMBER_WITH_UNIT.fullmatch(normalized)
        if match is None:
            return None
        numeric = float(match.group(1))
        unit = match.group(2) or ""
        explicit_unit = bool(unit)
    else:
        return None
    if not math.isfinite(numeric):
        raise ParameterTokenError(f"Parameter {name!r} must be finite.")

    normalized_unit = unit.lower()
    if normalized_unit in _LENGTH_TO_UM:
        return numeric * _LENGTH_TO_UM[normalized_unit], "length_um", explicit_unit
    if normalized_unit in _ANGLE_TO_DEG:
        return numeric * _ANGLE_TO_DEG[normalized_unit], "angle_deg", explicit_unit
    if unit:
        raise ParameterTokenError(
            f"Parameter {name!r} has unsupported unit {unit!r}; preserve it in feedback metadata or add an explicit conversion."
        )
    if "count" in roles:
        return numeric, "count", False
    if "angle" in roles:
        return numeric, "angle_deg", False
    physical_length_roles = {
        "length",
        "width",
        "gap",
        "radius",
        "offset_position",
        "thickness",
        "ground_clearance",
    }
    dimension = "unknown_physical_scale" if physical_length_roles.intersection(roles) else "dimensionless"
    return numeric, dimension, False


def tokenize_numeric_parameters(options: Mapping[str, Any]) -> ParameterTokens:
    """Create tool-neutral numeric tokens without encoding option names.

    The model receives values, physical dimensions, generic roles, and
    value-gated role channels.  Exact flattened names and raw values are kept
    only for later feedback into the originating layout tool.
    """
    raw = flatten_parameter_mapping(options)
    names: list[str] = []
    values: list[float] = []
    feature_rows: list[list[float]] = []
    for name, raw_value in raw.items():
        roles = _parameter_roles(name)
        parsed = _parameter_numeric(raw_value, roles, name=name)
        if parsed is None:
            continue
        value, dimension, unit_explicit = parsed
        names.append(name)
        values.append(value)
        dimension_channels = [float(dimension == candidate) for candidate in PARAMETER_DIMENSIONS]
        role_channels = [float(role in roles) for role in PARAMETER_ROLES]
        gated_channels = [value if role in roles else 0.0 for role in PARAMETER_ROLES]
        feature_rows.append([*dimension_channels, float(unit_explicit), *role_channels, *gated_channels])

    value_array = np.asarray(values, dtype=np.float64)
    feature_array = np.asarray(feature_rows, dtype=np.float64)
    if not feature_rows:
        feature_array = np.empty((0, len(PARAMETER_FEATURE_NAMES)), dtype=np.float64)
    if feature_array.shape != (len(value_array), len(PARAMETER_FEATURE_NAMES)):
        raise AssertionError("Internal parameter-token feature width mismatch.")
    value_array.setflags(write=False)
    feature_array.setflags(write=False)
    return ParameterTokens(tuple(names), value_array, feature_array, raw)


def _split_metadata(entry: CatalogueEntry, tokens: ParameterTokens) -> dict[str, Any]:
    role_values: dict[str, list[float]] = {role: [] for role in PARAMETER_ROLES}
    role_start = len(PARAMETER_DIMENSIONS) + 1
    for value, row in zip(tokens.values, tokens.features):
        for role_index, role in enumerate(PARAMETER_ROLES):
            if row[role_start + role_index] == 1.0:
                role_values[role].append(float(value))
    return {
        "dataset_family": entry.family,
        "dataset_role": entry.dataset_role,
        "source_campaign": entry.campaign,
        "source_id": entry.source_id,
        "row_index": entry.row_index,
        "design_id": entry.design_id,
        "row_sha256": entry.row_sha256,
        "dataset_sha256": entry.dataset_sha256,
        "simulation_config": entry.simulation_config,
        "simulation_solver": entry.row.get("sim_options", {}).get("simulator"),
        "gds_locator": entry.gds_locator,
        "raw_parameter_values": dict(tokens.raw_flattened),
        "split_parameter_values": dict(zip(tokens.names, (float(value) for value in tokens.values))),
        "physical_role_values": {role: values for role, values in role_values.items() if values},
        "parameter_feature_names": list(PARAMETER_FEATURE_NAMES),
    }


def build_entry_sidecar(
    entry: CatalogueEntry,
    *,
    inventory: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build the explicit matrix-node-to-GDS-net contract for one entry."""
    if entry.family == GENERALIZED_FAMILY:
        return build_generalized_ncap_sidecar(
            entry.gds_path,
            inventory=inventory,
            source_id=entry.source_id,
            design_id=entry.design_id,
        )
    if entry.family == CAPN_FAMILY:
        return build_capn_interdigital_tee_sidecar(
            entry.gds_path,
            inventory=inventory,
            signal_polygon_order=(0, 1),
            source_id=entry.source_id,
            design_id=entry.design_id,
        )
    raise CataloguePairingError(f"No net-sidecar adapter is registered for family {entry.family!r}.")


def build_catalogue_graph(
    entry: CatalogueEntry,
    *,
    canonicalize_rotation: bool = True,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
) -> BuiltCatalogueGraph:
    """Build one target-blind geometry graph, then attach its parsed target."""
    tokens = tokenize_numeric_parameters(entry.design_options)
    inventory = read_gds_inventory(entry.gds_path)
    sidecar = build_entry_sidecar(entry, inventory=inventory)
    graph_record = build_geometry_graph_record(
        entry.gds_path,
        sidecar,
        parameter_names=tokens.names,
        parameter_values=tokens.values,
        parameter_features=tokens.features,
        metadata=_split_metadata(entry, tokens),
        inventory=inventory,
        canonicalize_rotation=canonicalize_rotation,
        boundary_sample_count=boundary_sample_count,
    )
    if graph_record["capacitance_matrix_ff"] is not None:
        raise AssertionError("Geometry construction must remain target blind.")
    target_blind_graph = to_capacitance_graph(graph_record)
    target = parse_signed_maxwell_matrix_ff(entry.row)
    graph = target_blind_graph.with_target(target)
    return BuiltCatalogueGraph(entry=entry, sidecar=sidecar, graph_record=graph_record, graph=graph)


def iter_capacitance_graphs(
    entries: Iterable[CatalogueEntry],
    *,
    canonicalize_rotation: bool = True,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
) -> Iterator[CapacitanceGraph]:
    """Yield graphs one GDS at a time without retaining a geometry catalogue."""
    for entry in entries:
        yield build_catalogue_graph(
            entry,
            canonicalize_rotation=canonicalize_rotation,
            boundary_sample_count=boundary_sample_count,
        ).graph


def iter_generalized_capacitance_graphs(
    simulation_source: str | Path | Sequence[Mapping[str, Any]],
    *,
    exp6_gds_dir: str | Path,
    exp7_gds_dir: str | Path,
    q3d_gds_root: str | Path,
    expected_count: int | None = EXPECTED_GENERALIZED_RECORDS,
    canonicalize_rotation: bool = True,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
) -> Iterator[CapacitanceGraph]:
    """Pair, validate, and stream the complete GeneralizedNCap catalogue."""
    manifest = pair_generalized_records(
        simulation_source,
        exp6_gds_dir=exp6_gds_dir,
        exp7_gds_dir=exp7_gds_dir,
        q3d_gds_root=q3d_gds_root,
        expected_count=expected_count,
        dataset_role="source",
    )
    manifest.assert_complete()
    yield from iter_capacitance_graphs(
        manifest,
        canonicalize_rotation=canonicalize_rotation,
        boundary_sample_count=boundary_sample_count,
    )


def iter_capn_capacitance_graphs(
    simulation_source: str | Path | Sequence[Mapping[str, Any]],
    *,
    gds_dir: str | Path,
    expected_count: int | None = EXPECTED_CAPN_RECORDS,
    canonicalize_rotation: bool = True,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
) -> Iterator[CapacitanceGraph]:
    """Pair, validate, and stream the complete legacy CapN catalogue."""
    manifest = pair_capn_records(
        simulation_source,
        gds_dir=gds_dir,
        expected_count=expected_count,
        dataset_role="target",
    )
    manifest.assert_complete()
    yield from iter_capacitance_graphs(
        manifest,
        canonicalize_rotation=canonicalize_rotation,
        boundary_sample_count=boundary_sample_count,
    )


def _implementation_hashes() -> dict[str, str]:
    module_dir = Path(__file__).resolve().parent
    paths = {
        "datasets.py": Path(__file__).resolve(),
        "geometry_graph.py": module_dir / "geometry_graph.py",
        "net_extraction.py": module_dir / "net_extraction.py",
        "schema.py": module_dir / "schema.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def pipeline_fingerprint(
    *,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
    canonicalize_rotation: bool = True,
) -> str:
    """Hash every model-facing extraction contract and implementation file."""
    payload = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "sidecar_schema": NET_SIDECAR_SCHEMA_VERSION,
        "graph_schema": GEOMETRY_GRAPH_SCHEMA_VERSION,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "parameter_feature_names": list(PARAMETER_FEATURE_NAMES),
        "proximity_scales_um": list(PROXIMITY_SCALES_UM),
        "boundary_sample_count": int(boundary_sample_count),
        "canonicalize_rotation": bool(canonicalize_rotation),
        "implementation_sha256": _implementation_hashes(),
    }
    return content_sha256(payload)


def _safe_sidecar_path(output_dir: Path, entry: CatalogueEntry) -> Path:
    parts = [entry.family, *entry.source_id.split("/")]
    if any(not _SAFE_PATH_PART.fullmatch(part) for part in parts):
        raise CataloguePairingError(f"Unsafe sidecar path component in {entry.cache_key!r}.")
    return output_dir / SIDECAR_DIRECTORY_NAME / parts[0] / Path(*parts[1:]).with_suffix(".json")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json(value))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _verify_sidecar_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise CacheVerificationError(f"Missing sidecar file: {path}")
    with path.open(encoding="utf-8") as stream:
        sidecar = json.load(stream)
    claimed = sidecar.get("sidecar_sha256")
    payload = dict(sidecar)
    payload.pop("sidecar_sha256", None)
    observed = content_sha256(payload)
    if claimed != observed or observed != expected_sha256:
        raise CacheVerificationError(f"Sidecar hash mismatch: {path}")


def _target_payload(matrix: NDArray[np.float64], net_ids: Sequence[str]) -> dict[str, Any]:
    target = {
        "unit": "fF",
        "node_order": list(net_ids),
        "capacitance_matrix_ff": matrix.tolist(),
    }
    target["target_sha256"] = content_sha256(target)
    return target


def _cache_record(
    built: BuiltCatalogueGraph,
    *,
    output_dir: Path,
    pipeline_sha256: str,
) -> dict[str, Any]:
    sidecar_path = _safe_sidecar_path(output_dir, built.entry)
    record = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": built.entry.cache_key,
        "catalogue": built.entry.public_metadata(),
        "pipeline_sha256": pipeline_sha256,
        "sidecar_path": sidecar_path.relative_to(output_dir).as_posix(),
        "sidecar_sha256": built.sidecar["sidecar_sha256"],
        "graph_record": dict(built.graph_record),
        "target": _target_payload(built.graph.capacitance_matrix, built.graph.net_ids),
    }
    record["cache_record_sha256"] = content_sha256(record)
    return record


def _verify_graph_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != GEOMETRY_GRAPH_SCHEMA_VERSION:
        raise CacheVerificationError(f"Unsupported graph schema {record.get('schema_version')!r}.")
    claimed = record.get("record_sha256")
    payload = dict(record)
    payload.pop("record_sha256", None)
    if claimed != content_sha256(payload):
        raise CacheVerificationError("Graph record content hash mismatch.")
    if record.get("capacitance_matrix_ff") is not None:
        raise CacheVerificationError("Cached geometry records must remain target blind.")


def _verify_cache_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheVerificationError(f"Unsupported cache schema {record.get('schema_version')!r}.")
    claimed = record.get("cache_record_sha256")
    payload = dict(record)
    payload.pop("cache_record_sha256", None)
    if claimed != content_sha256(payload):
        raise CacheVerificationError(f"Cache record hash mismatch for {record.get('cache_key')!r}.")
    _verify_graph_record(record["graph_record"])
    target = dict(record["target"])
    target_sha256 = target.pop("target_sha256", None)
    if target_sha256 != content_sha256(target):
        raise CacheVerificationError(f"Target hash mismatch for {record.get('cache_key')!r}.")
    if target.get("unit") != "fF":
        raise CacheVerificationError("Cached targets must be explicitly normalized to fF.")
    maxwell_to_components(np.asarray(target["capacitance_matrix_ff"], dtype=np.float64))


def _read_cache_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise CacheVerificationError(f"Blank JSONL record at {path}:{line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CacheVerificationError(f"Malformed JSONL record at {path}:{line_number}.") from exc
            if not isinstance(record, Mapping):
                raise CacheVerificationError(f"JSONL record at {path}:{line_number} is not an object.")
            yield line_number, record


def _verify_prefix(
    entries: Sequence[CatalogueEntry],
    *,
    output_dir: Path,
    graph_path: Path,
    pipeline_sha256: str,
    verify_gds: bool,
) -> int:
    count = 0
    for line_number, record in _read_cache_records(graph_path):
        _verify_cache_record(record)
        if count >= len(entries):
            raise CacheVerificationError("Cache contains more records than the requested deterministic catalogue.")
        entry = entries[count]
        if record["cache_key"] != entry.cache_key:
            raise CacheVerificationError(
                f"Cache is not a canonical prefix at line {line_number}: expected {entry.cache_key!r}, "
                f"found {record['cache_key']!r}."
            )
        catalogue = record["catalogue"]
        for field, expected in (
            ("row_index", entry.row_index),
            ("row_sha256", entry.row_sha256),
            ("dataset_sha256", entry.dataset_sha256),
            ("design_id", entry.design_id),
        ):
            if catalogue.get(field) != expected:
                raise CacheVerificationError(f"Stale {field} for {entry.cache_key!r} at line {line_number}.")
        if record.get("pipeline_sha256") != pipeline_sha256:
            raise CacheVerificationError(
                f"Pipeline fingerprint changed for {entry.cache_key!r}; rebuild rather than resuming this cache."
            )
        sidecar_path = output_dir / record["sidecar_path"]
        _verify_sidecar_file(sidecar_path, record["sidecar_sha256"])
        if verify_gds and sha256_file(entry.gds_path) != record["graph_record"]["gds_sha256"]:
            raise CacheVerificationError(f"GDS artifact changed for {entry.cache_key!r}: {entry.gds_path}")
        count += 1
    return count


def _cache_manifest_payload(
    entries: Sequence[CatalogueEntry],
    *,
    graph_path: Path,
    pipeline_sha256: str,
    boundary_sample_count: int,
    canonicalize_rotation: bool,
) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    dataset_hashes: dict[str, str] = {}
    for entry in entries:
        family_counts[entry.family] = family_counts.get(entry.family, 0) + 1
        dataset_hashes[entry.family] = entry.dataset_sha256
    payload = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "record_count": len(entries),
        "family_counts": family_counts,
        "expected_release_counts": {
            GENERALIZED_FAMILY: EXPECTED_GENERALIZED_RECORDS,
            CAPN_FAMILY: EXPECTED_CAPN_RECORDS,
        },
        "dataset_sha256": dataset_hashes,
        "graph_jsonl": graph_path.name,
        "graph_jsonl_sha256": sha256_file(graph_path),
        "pipeline_sha256": pipeline_sha256,
        "boundary_sample_count": boundary_sample_count,
        "canonicalize_rotation": bool(canonicalize_rotation),
        "node_feature_dim": len(NODE_FEATURE_NAMES),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES),
        "global_feature_dim": len(GLOBAL_FEATURE_NAMES),
        "parameter_feature_dim": len(PARAMETER_FEATURE_NAMES),
        "parameter_feature_names": list(PARAMETER_FEATURE_NAMES),
        "target_unit": "fF",
        "target_convention": "signed_maxwell",
        "target_storage": "separate_from_target_blind_graph_record",
    }
    payload["manifest_sha256"] = content_sha256(payload)
    return payload


def write_graph_cache(
    entries: Iterable[CatalogueEntry],
    output_dir: str | Path,
    *,
    resume: bool = False,
    verify_existing_gds: bool = True,
    canonicalize_rotation: bool = True,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
    progress_every: int = 100,
    progress: Callable[[str], None] | None = None,
) -> CacheBuildSummary:
    """Build a deterministic, resumable JSONL cache one GDS at a time."""
    ordered_entries = tuple(entries)
    if not ordered_entries:
        raise CatalogueError("Cannot build an empty TopoCap graph cache.")
    if progress_every < 1:
        raise ValueError("progress_every must be positive.")
    _validate_unique_entries(ordered_entries)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / CACHE_JSONL_NAME
    manifest_path = output / CACHE_MANIFEST_NAME
    pipeline_sha256 = pipeline_fingerprint(
        boundary_sample_count=boundary_sample_count,
        canonicalize_rotation=canonicalize_rotation,
    )

    if graph_path.exists() and not resume:
        raise FileExistsError(f"Cache already exists; pass resume=True after verification: {graph_path}")
    existing = (
        _verify_prefix(
            ordered_entries,
            output_dir=output,
            graph_path=graph_path,
            pipeline_sha256=pipeline_sha256,
            verify_gds=verify_existing_gds,
        )
        if graph_path.exists()
        else 0
    )

    written = 0
    with graph_path.open("a", encoding="utf-8") as stream:
        for index in range(existing, len(ordered_entries)):
            entry = ordered_entries[index]
            try:
                built = build_catalogue_graph(
                    entry,
                    canonicalize_rotation=canonicalize_rotation,
                    boundary_sample_count=boundary_sample_count,
                )
                sidecar_path = _safe_sidecar_path(output, entry)
                if sidecar_path.exists():
                    _verify_sidecar_file(sidecar_path, built.sidecar["sidecar_sha256"])
                else:
                    _atomic_json(sidecar_path, built.sidecar)
                cache_record = _cache_record(built, output_dir=output, pipeline_sha256=pipeline_sha256)
                stream.write(canonical_json(cache_record))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            except Exception as exc:
                raise CatalogueError(
                    f"Failed TopoCap extraction at catalogue index {index} ({entry.cache_key}, {entry.gds_path}): {exc}"
                ) from exc
            written += 1
            completed = index + 1
            if progress is not None and (completed % progress_every == 0 or completed == len(ordered_entries)):
                progress(f"TopoCap cache: {completed}/{len(ordered_entries)} records complete ({entry.cache_key}).")

    manifest = _cache_manifest_payload(
        ordered_entries,
        graph_path=graph_path,
        pipeline_sha256=pipeline_sha256,
        boundary_sample_count=boundary_sample_count,
        canonicalize_rotation=canonicalize_rotation,
    )
    _atomic_json(manifest_path, manifest)
    return CacheBuildSummary(
        output_dir=output,
        graph_jsonl=graph_path,
        cache_manifest=manifest_path,
        requested_count=len(ordered_entries),
        existing_count=existing,
        written_count=written,
        graph_jsonl_sha256=manifest["graph_jsonl_sha256"],
        pipeline_sha256=pipeline_sha256,
    )


def verify_graph_cache(
    entries: Iterable[CatalogueEntry],
    output_dir: str | Path,
    *,
    verify_gds: bool = True,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
    canonicalize_rotation: bool = True,
) -> CacheVerification:
    """Verify ordering, all content hashes, targets, sidecars, and optional GDS."""
    ordered_entries = tuple(entries)
    output = Path(output_dir)
    graph_path = output / CACHE_JSONL_NAME
    manifest_path = output / CACHE_MANIFEST_NAME
    if not graph_path.is_file() or not manifest_path.is_file():
        raise CacheVerificationError(f"Incomplete cache under {output}; graph JSONL and manifest are required.")
    pipeline_sha256 = pipeline_fingerprint(
        boundary_sample_count=boundary_sample_count,
        canonicalize_rotation=canonicalize_rotation,
    )
    count = _verify_prefix(
        ordered_entries,
        output_dir=output,
        graph_path=graph_path,
        pipeline_sha256=pipeline_sha256,
        verify_gds=verify_gds,
    )
    if count != len(ordered_entries):
        raise CacheVerificationError(f"Cache has {count} records; expected {len(ordered_entries)}.")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    claimed = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    if claimed != content_sha256(payload):
        raise CacheVerificationError("Cache manifest content hash mismatch.")
    graph_sha256 = sha256_file(graph_path)
    if manifest.get("graph_jsonl_sha256") != graph_sha256:
        raise CacheVerificationError("Cache manifest points to a different graph JSONL content hash.")
    if manifest.get("record_count") != count or manifest.get("pipeline_sha256") != pipeline_sha256:
        raise CacheVerificationError("Cache manifest count or pipeline fingerprint is stale.")
    return CacheVerification(
        record_count=count,
        sidecar_count=count,
        graph_jsonl_sha256=graph_sha256,
        pipeline_sha256=pipeline_sha256,
    )


def cached_record_to_graph(record: Mapping[str, Any]) -> CapacitanceGraph:
    """Reconstruct a labelled graph while preserving target-blind features."""
    _verify_cache_record(record)
    graph = to_capacitance_graph(record["graph_record"])
    target = np.asarray(record["target"]["capacitance_matrix_ff"], dtype=np.float64)
    return graph.with_target(target)


def iter_cached_graphs(path: str | Path) -> Iterator[CapacitanceGraph]:
    """Stream validated graphs from an existing cache JSONL."""
    for _, record in _read_cache_records(Path(path)):
        yield cached_record_to_graph(record)


__all__ = [
    "CACHE_JSONL_NAME",
    "CACHE_MANIFEST_NAME",
    "CACHE_SCHEMA_VERSION",
    "CAPACITANCE_FACTORS_TO_FF",
    "CAPN_FAMILY",
    "CAPN_SIMULATION_CONFIG",
    "CatalogueEntry",
    "CatalogueError",
    "CatalogueManifest",
    "CataloguePairingError",
    "CacheBuildSummary",
    "CacheVerification",
    "CacheVerificationError",
    "CapacitanceUnitError",
    "EXPECTED_CAPN_RECORDS",
    "EXPECTED_GENERALIZED_RECORDS",
    "GENERALIZED_FAMILY",
    "GENERALIZED_SIMULATION_CONFIG",
    "PARAMETER_DIMENSIONS",
    "PARAMETER_FEATURE_NAMES",
    "PARAMETER_ROLES",
    "PairingDiagnostic",
    "ParameterTokenError",
    "ParameterTokens",
    "SIDECAR_DIRECTORY_NAME",
    "build_catalogue_graph",
    "build_entry_sidecar",
    "cached_record_to_graph",
    "flatten_parameter_mapping",
    "iter_cached_graphs",
    "iter_capacitance_graphs",
    "iter_capn_capacitance_graphs",
    "iter_generalized_capacitance_graphs",
    "pair_capn_records",
    "pair_generalized_records",
    "parse_signed_maxwell_matrix_ff",
    "pipeline_fingerprint",
    "tokenize_numeric_parameters",
    "verify_graph_cache",
    "write_graph_cache",
]
