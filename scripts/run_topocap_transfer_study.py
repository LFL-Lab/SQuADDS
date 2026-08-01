#!/usr/bin/env python3
"""Run the leakage-resistant GeneralizedNCap-to-CapN TopoCap study.

The runner is intentionally artifact driven. Source foundations are fitted once,
trial results are checkpointed atomically, and every report table is rebuilt from
those immutable trial checkpoints. Target test rows are never used for fitting,
normalization, evidence gating, or support selection.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from squadds.ml.topocap import (
    RETRIEVAL_SHUFFLE_SEED_XOR,
    RETRIEVAL_SOURCE_BUDGET,
    EBRAAdapter,
    EBRAConfig,
    EvidenceGateConfig,
    EvidenceGatedTopoCap,
    SupportConditionedSourceRetriever,
    SupportRetrievalSelection,
    TopoCapConfig,
    TopoCapFoundationModel,
    build_active_geometry_view,
    build_topology_control_view,
    canonical_edge_index,
    maxwell_to_components,
    retrieval_source_ids_sha256,
)
from squadds.ml.topocap.datasets import (
    CACHE_MANIFEST_NAME,
    CAPN_FAMILY,
    GENERALIZED_FAMILY,
    iter_cached_graphs,
)
from squadds.ml.topocap.metrics import (
    evaluate_maxwell_matrices,
    interval_calibration,
    maxwell_physical_diagnostics,
    paired_group_bootstrap,
    risk_coverage_curve,
)
from squadds.ml.topocap.net_extraction import content_sha256
from squadds.ml.topocap.schema import CapacitanceGraph
from squadds.ml.topocap.splits import (
    DEFAULT_SUPPORT_SIZES,
    capn_group_labels,
    capn_outer_splits,
    make_nested_target_split,
)

RUNNER_SCHEMA_VERSION = "topocap-transfer-study-1.1.0"
TRIAL_SCHEMA_VERSION = "topocap-transfer-trial-1.1.0"
REPORT_ARTIFACT_SCHEMA_VERSION = "topocap-report-artifacts-1.0.0"
V0_INPUT_DIMENSION = 9_227
V0_SKETCH_DIMENSION = 64
V0_SKETCH_SEED = 14
INTERVAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
PRIMARY_METRIC = "macro_component_log_mae"
PAIRED_GAIN_METRIC = "paired_gain_vs_target_control"
PAIRED_SOURCE_GAIN_METRIC = "paired_gain_vs_shuffled_source"
REPORT_METRICS = (
    PRIMARY_METRIC,
    "macro_relative_frobenius",
    "physical_valid_rate",
)
IDENTITY_FIELDS = ("gds_sha256", "design_id", "row_sha256", "source_id", "sidecar_sha256")

CONTROL_CONFIG = TopoCapConfig(
    random_feature_dimensions=0,
    ridge_alpha=30.0,
    random_seed=17,
)
ACTIVE_CONFIG = TopoCapConfig(
    random_feature_dimensions=0,
    ridge_alpha=30.0,
    random_seed=19,
)
ALL_DESCRIPTOR_CONFIG = TopoCapConfig(
    random_feature_dimensions=0,
    ridge_alpha=30.0,
    random_seed=23,
)
V0_CONFIG = TopoCapConfig(
    random_feature_dimensions=0,
    ridge_alpha=30.0,
    random_seed=29,
)
ADAPTER_CONFIG = EBRAConfig(
    prior_precision=25.0,
    minimum_residual_variance=1.0e-6,
    observation_variance_floor=1.0e-8,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a resumable, leakage-resistant TopoCap transfer study from a verified graph-cache JSONL.")
    )
    parser.add_argument("cache_jsonl", type=Path, help="Verified TopoCap graphs.jsonl path.")
    parser.add_argument("output_dir", type=Path, help="Directory for models, trials, and report artifacts.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent nested-support draws per outer finger-count domain (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=73, help="Root deterministic seed (default: 73).")
    parser.add_argument(
        "--v0-parquet",
        type=Path,
        default=None,
        help=(
            "Optional static-embedding-v0 parquet. Rows are aligned by "
            "(component_name, source_id) and sketched from 9,227 to 64 dimensions."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke mode: 2,048 source rows, three outer domains, one repeat, test subsets, "
            "and support sizes through 16. Artifacts are marked non-claim-bearing."
        ),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.cache_jsonl.is_file():
        parser.error(f"cache JSONL does not exist: {args.cache_jsonl}")
    if args.v0_parquet is not None and not args.v0_parquet.is_file():
        parser.error(f"v0 parquet does not exist: {args.v0_parquet}")
    return args


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: object) -> int:
    payload = json.dumps([int(seed), *map(str, parts)], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            payload,
            stream,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            json.dump(
                row,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
                default=_json_default,
            )
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False, lineterminator="\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_save_model(model: TopoCapFoundationModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_cache_input(cache_jsonl: Path) -> dict[str, Any]:
    cache_path = cache_jsonl.expanduser().resolve()
    manifest_path = cache_path.parent / CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest is required beside the JSONL: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_manifest_hash = manifest.get("manifest_sha256")
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_sha256", None)
    if claimed_manifest_hash != content_sha256(manifest_payload):
        raise ValueError("Cache manifest content hash is invalid.")
    actual_graph_hash = _sha256_file(cache_path)
    if actual_graph_hash != manifest.get("graph_jsonl_sha256"):
        raise ValueError("Cache JSONL SHA-256 does not match cache-manifest.json.")
    if manifest.get("graph_jsonl") != cache_path.name:
        raise ValueError("Cache manifest graph_jsonl name does not match the CLI input.")
    return {
        "path": str(cache_path),
        "manifest_path": str(manifest_path.resolve()),
        "graph_jsonl_sha256": actual_graph_hash,
        "manifest_sha256": claimed_manifest_hash,
        "pipeline_sha256": manifest.get("pipeline_sha256"),
        "record_count": int(manifest.get("record_count", 0)),
        "family_counts": dict(manifest.get("family_counts", {})),
        "dataset_sha256": dict(manifest.get("dataset_sha256", {})),
    }


def _compact_graph(graph: CapacitanceGraph) -> CapacitanceGraph:
    metadata_keys = (
        "dataset_family",
        "dataset_role",
        "source_id",
        "source_campaign",
        "row_index",
        "design_id",
        "gds_sha256",
        "row_sha256",
        "dataset_sha256",
        "capacitance_unit",
        "split_parameter_values",
    )
    metadata = {key: graph.metadata[key] for key in metadata_keys if key in graph.metadata}
    return CapacitanceGraph(
        node_features=graph.node_features,
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
        global_features=graph.global_features,
        parameter_values=graph.parameter_values,
        parameter_features=graph.parameter_features,
        net_ids=graph.net_ids,
        parameter_names=graph.parameter_names,
        capacitance_matrix=graph.capacitance_matrix,
        metadata=metadata,
    )


def _quick_source_sample(
    heap: list[tuple[int, str, CapacitanceGraph]],
    graph: CapacitanceGraph,
    limit: int,
) -> None:
    source_id = str(graph.metadata["source_id"])
    score = int.from_bytes(hashlib.sha256(source_id.encode("utf-8")).digest()[:8], "big")
    candidate = (-score, source_id, _compact_graph(graph))
    if len(heap) < limit:
        heapq.heappush(heap, candidate)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, candidate)


def _load_graphs(cache_jsonl: Path, quick: bool) -> tuple[tuple[CapacitanceGraph, ...], tuple[CapacitanceGraph, ...]]:
    source: list[CapacitanceGraph] = []
    source_heap: list[tuple[int, str, CapacitanceGraph]] = []
    target: list[CapacitanceGraph] = []
    for graph in iter_cached_graphs(cache_jsonl):
        family = graph.metadata.get("dataset_family")
        if family == GENERALIZED_FAMILY:
            if quick:
                _quick_source_sample(source_heap, graph, limit=RETRIEVAL_SOURCE_BUDGET)
            else:
                source.append(_compact_graph(graph))
        elif family == CAPN_FAMILY:
            target.append(_compact_graph(graph))
    if quick:
        source = [item[2] for item in sorted(source_heap, key=lambda item: item[1])]
    source.sort(key=lambda graph: int(graph.metadata["row_index"]))
    target.sort(key=lambda graph: int(graph.metadata["row_index"]))
    if not source or not target:
        raise ValueError("The cache must contain both GeneralizedNCap source and CapN target graphs.")
    overlaps = _cross_family_identity_overlaps(source, target)
    contaminated = {name: count for name, count in overlaps.items() if count}
    if contaminated:
        raise ValueError(f"Cross-family identity leakage detected: {contaminated}.")
    return tuple(source), tuple(target)


def _cross_family_identity_overlaps(
    source: Sequence[CapacitanceGraph],
    target: Sequence[CapacitanceGraph],
) -> dict[str, int]:
    """Count exact identity collisions that would invalidate cross-family transfer."""
    overlaps = {}
    for field in IDENTITY_FIELDS:
        source_values = {str(graph.metadata[field]) for graph in source if graph.metadata.get(field) is not None}
        target_values = {str(graph.metadata[field]) for graph in target if graph.metadata.get(field) is not None}
        overlaps[field] = len(source_values & target_values)
    return overlaps


def _graph_key(graph: CapacitanceGraph) -> tuple[str, str]:
    return str(graph.metadata["dataset_family"]), str(graph.metadata["source_id"])


def _target_frame(target: Sequence[CapacitanceGraph]) -> pd.DataFrame:
    rows = []
    for graph in target:
        values = graph.metadata["split_parameter_values"]
        rows.append(
            {
                "source_id": graph.metadata["source_id"],
                "finger_count": values["finger_count"],
                "finger_length": values["finger_length"],
                "cap_width": values["cap_width"],
                "cap_gap": values["cap_gap"],
            }
        )
    return pd.DataFrame(rows)


def _source_view(
    source: Sequence[CapacitanceGraph],
    builder: Callable[[CapacitanceGraph], CapacitanceGraph],
) -> tuple[CapacitanceGraph, ...]:
    return tuple(_compact_graph(builder(graph)) for graph in source)


def _fit_or_load_model(
    path: Path,
    config: TopoCapConfig,
    training_graphs: Callable[[], Sequence[CapacitanceGraph]],
    label: str,
    state_digest: str,
    *,
    bindings: Mapping[str, Any] | None = None,
) -> TopoCapFoundationModel:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    expected_bindings = dict(bindings or {})
    if path.is_file():
        print(f"Loading source model: {label} <- {path}", flush=True)
        if not metadata_path.is_file():
            raise ValueError(f"Stored {label} is missing its identity metadata: {metadata_path}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("state_digest") != state_digest or metadata.get("label") != label:
            raise ValueError(f"Stored {label} identity does not match this study role and state.")
        if metadata.get("bindings", {}) != expected_bindings:
            raise ValueError(f"Stored {label} retrieval bindings do not match this trial selection.")
        if metadata.get("model_sha256") != _sha256_file(path):
            raise ValueError(f"Stored {label} bytes do not match the checkpoint metadata.")
        model = TopoCapFoundationModel.load(path)
        if model.config != config:
            raise ValueError(f"Stored {label} configuration does not match this study.")
        return model
    print(f"Fitting source model once: {label}", flush=True)
    started = time.monotonic()
    model = TopoCapFoundationModel(config).fit(tuple(training_graphs()))
    _atomic_save_model(model, path)
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": "topocap-source-model-checkpoint-1.1.0",
            "state_digest": state_digest,
            "label": label,
            "model_sha256": _sha256_file(path),
            "config": asdict(config),
            "bindings": expected_bindings,
        },
    )
    print(f"Fitted {label} in {time.monotonic() - started:.1f}s -> {path}", flush=True)
    return model


def _shuffled_targets(
    graphs: Sequence[CapacitanceGraph],
    seed: int,
) -> tuple[CapacitanceGraph, ...]:
    permutation = np.random.default_rng(seed).permutation(len(graphs))
    return tuple(
        graph.with_target(graphs[int(target_index)].capacitance_matrix)
        for graph, target_index in zip(graphs, permutation)
    )


def _v0_projection() -> np.ndarray:
    rng = np.random.default_rng(V0_SKETCH_SEED)
    return rng.normal(
        0.0,
        1.0 / math.sqrt(V0_SKETCH_DIMENSION),
        size=(V0_INPUT_DIMENSION, V0_SKETCH_DIMENSION),
    ).astype(np.float32)


def _load_v0_sketches(
    parquet_path: Path,
    required_keys: set[tuple[str, str]],
    source_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("--v0-parquet requires pyarrow.") from error

    parquet = pq.ParquetFile(parquet_path)
    columns = ("component_name", "source_id", "embedding")

    source_count = 0
    source_mean = np.zeros(V0_INPUT_DIMENSION, dtype=np.float64)
    source_m2 = np.zeros(V0_INPUT_DIMENSION, dtype=np.float64)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        components = batch.column(0).to_pylist()
        source_ids = batch.column(1).to_pylist()
        selected_rows = [index for index, key in enumerate(zip(components, source_ids)) if key in source_keys]
        if not selected_rows:
            continue
        embeddings = np.vstack(
            [np.asarray(batch.column(2)[index].as_py(), dtype=np.float64) for index in selected_rows]
        )
        if embeddings.shape[1] != V0_INPUT_DIMENSION:
            raise ValueError(f"v0 embedding width is {embeddings.shape[1]}; expected {V0_INPUT_DIMENSION}.")
        batch_count = len(embeddings)
        batch_mean = embeddings.mean(axis=0)
        batch_m2 = np.sum((embeddings - batch_mean) ** 2, axis=0)
        delta = batch_mean - source_mean
        combined_count = source_count + batch_count
        source_mean += delta * (batch_count / combined_count)
        source_m2 += batch_m2 + delta**2 * source_count * batch_count / combined_count
        source_count = combined_count
    if source_count != len(source_keys):
        raise ValueError(f"v0 source-standardization pass found {source_count} rows; expected {len(source_keys)}.")
    source_scale = np.sqrt(source_m2 / max(source_count, 1))
    source_scale[source_scale < 1.0e-8] = 1.0

    projection = _v0_projection().astype(np.float64)
    sketches: dict[tuple[str, str], np.ndarray] = {}
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        components = batch.column(0).to_pylist()
        source_ids = batch.column(1).to_pylist()
        selected_rows = [index for index, key in enumerate(zip(components, source_ids)) if key in required_keys]
        if not selected_rows:
            continue
        embeddings = np.vstack(
            [np.asarray(batch.column(2)[index].as_py(), dtype=np.float64) for index in selected_rows]
        )
        if embeddings.shape[1] != V0_INPUT_DIMENSION:
            raise ValueError(f"v0 embedding width is {embeddings.shape[1]}; expected {V0_INPUT_DIMENSION}.")
        standardized = np.clip((embeddings - source_mean) / source_scale, -8.0, 8.0)
        projected = standardized @ projection
        for row_index, vector in zip(selected_rows, projected):
            key = str(components[row_index]), str(source_ids[row_index])
            if key in sketches:
                raise ValueError(f"Duplicate v0 alignment key: {key!r}.")
            sketches[key] = np.asarray(vector, dtype=np.float64)
    missing = sorted(required_keys - sketches.keys())
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"v0 parquet is missing {len(missing)} required graph rows: {preview}")
    return sketches, source_mean, source_scale


def _fit_v0_standardization(
    source: Sequence[CapacitanceGraph],
    sketches: Mapping[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    source_matrix = np.vstack([sketches[_graph_key(graph)] for graph in source])
    mean = source_matrix.mean(axis=0)
    scale = source_matrix.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return mean, scale


def _v0_graph(
    graph: CapacitanceGraph,
    sketch: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> CapacitanceGraph:
    topology = build_topology_control_view(graph)
    normalized = np.clip((sketch - mean) / scale, -8.0, 8.0)
    return CapacitanceGraph(
        node_features=topology.node_features,
        edge_index=topology.edge_index,
        edge_features=topology.edge_features,
        global_features=normalized,
        parameter_values=np.empty(0, dtype=np.float64),
        parameter_features=np.empty((0, 0), dtype=np.float64),
        net_ids=topology.net_ids,
        capacitance_matrix=topology.capacitance_matrix,
        metadata={
            "dataset_family": graph.metadata["dataset_family"],
            "source_id": graph.metadata["source_id"],
            "topocap_view": {
                "name": "static-embedding-v0-gaussian64",
                "projection_seed": V0_SKETCH_SEED,
                "projection_input_dim": V0_INPUT_DIMENSION,
                "projection_output_dim": V0_SKETCH_DIMENSION,
                "standardization_fit": ("GeneralizedCapNInterdigital source only, before and after projection"),
            },
        },
    )


def _build_state(
    args: argparse.Namespace,
    cache_identity: Mapping[str, Any],
    v0_sha256: str | None,
) -> dict[str, Any]:
    repeats = 1 if args.quick else args.repeats
    support_sizes = (0, 1, 2, 4, 8, 16) if args.quick else DEFAULT_SUPPORT_SIZES
    repository_root = Path(__file__).resolve().parents[1]
    protocol_files = [Path(__file__).resolve()]
    protocol_files.extend(sorted((repository_root / "squadds" / "ml" / "topocap").glob("*.py")))
    code_sha256 = {path.relative_to(repository_root).as_posix(): _sha256_file(path) for path in protocol_files}
    configuration = {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "experimental_protocol_revision": 3,
        "code_sha256": code_sha256,
        "cache": dict(cache_identity),
        "v0_parquet_sha256": v0_sha256,
        "v0_projection": (
            {
                "input_dimensions": V0_INPUT_DIMENSION,
                "output_dimensions": V0_SKETCH_DIMENSION,
                "seed": V0_SKETCH_SEED,
                "distribution": "Gaussian(0, 1/sqrt(64))",
                "fit": (
                    "source-standardize all 9,227 coordinates, apply target-blind fixed projection, "
                    "then source-standardize the 64-dimensional sketch"
                ),
            }
            if v0_sha256 is not None
            else None
        ),
        "quick": bool(args.quick),
        "quick_source_rows": RETRIEVAL_SOURCE_BUDGET if args.quick else None,
        "repeats": repeats,
        "root_seed": int(args.seed),
        "support_sizes": list(support_sizes),
        "outer_split": "CapNInterdigitalTee leave-one-finger-count-out with numeric-next validation",
        "support_grouping": "exact CapN morphology; gate CV leaves complete support finger counts out",
        "aggregation": "average repeats within each finger-count domain, then bootstrap outer domains",
        "primary_support_size": 16,
        "control_config": asdict(CONTROL_CONFIG),
        "active_config": asdict(ACTIVE_CONFIG),
        "all_cached_descriptor_config": asdict(ALL_DESCRIPTOR_CONFIG),
        "v0_config": asdict(V0_CONFIG),
        "adapter_config": asdict(ADAPTER_CONFIG),
        "support_conditioned_retrieval": SupportConditionedSourceRetriever.protocol(),
        "interval_levels": list(INTERVAL_LEVELS),
    }
    digest = content_sha256(configuration)
    return {"digest": digest, "configuration": configuration}


def _initialize_output(output_dir: Path, state: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "study-state.json"
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing.get("digest") != state.get("digest"):
            raise ValueError(
                "Output directory belongs to a different cache, code version, or study configuration. "
                "Use a new output directory rather than mixing checkpoints."
            )
    else:
        _atomic_write_json(state_path, state)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "trials").mkdir(exist_ok=True)


def _finger_count(graph: CapacitanceGraph) -> int:
    return int(round(float(graph.metadata["split_parameter_values"]["finger_count"])))


def _domain_for_indices(target: Sequence[CapacitanceGraph], indices: Sequence[int]) -> int:
    values = {_finger_count(target[int(index)]) for index in indices}
    if len(values) != 1:
        raise ValueError(f"Expected one finger-count domain, found {sorted(values)}.")
    return values.pop()


def _hashed_subset(indices: Sequence[int], target: Sequence[CapacitanceGraph], limit: int) -> np.ndarray:
    ranked = sorted((str(target[int(index)].metadata["source_id"]), int(index)) for index in indices)
    ranked.sort(key=lambda item: hashlib.sha256(item[0].encode("utf-8")).digest())
    return np.asarray(sorted(index for _, index in ranked[:limit]), dtype=np.int64)


def _lookup_view(views: Mapping[str, CapacitanceGraph]) -> Callable[[CapacitanceGraph], CapacitanceGraph]:
    def resolve(graph: CapacitanceGraph) -> CapacitanceGraph:
        return views[str(graph.metadata["source_id"])]

    return resolve


def _predict_many(
    model: Any,
    graphs: Sequence[CapacitanceGraph],
) -> list[Any]:
    return model.predict_many(graphs, confidence=0.9)


def _component_log_errors(
    truth: Sequence[np.ndarray],
    prediction: Sequence[np.ndarray],
) -> np.ndarray:
    errors = []
    for expected, actual in zip(truth, prediction):
        expected_components = maxwell_to_components(expected)
        actual_components = maxwell_to_components(actual)
        expected_logs = np.concatenate((expected_components.log_shunts, expected_components.log_mutuals))
        actual_logs = np.concatenate((actual_components.log_shunts, actual_components.log_mutuals))
        errors.append(float(np.mean(np.abs(actual_logs - expected_logs))))
    return np.asarray(errors, dtype=np.float64)


def _prediction_uncertainty(prediction: Any) -> float:
    variances = np.concatenate((prediction.log_shunt_variance, prediction.log_mutual_variance))
    return float(np.sqrt(np.mean(variances)))


def _evaluate_method(
    method: str,
    predictions: Sequence[Any],
    test_graphs: Sequence[CapacitanceGraph],
    test_morphologies: Sequence[str],
    choice: str | None,
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    truth = [graph.capacitance_matrix for graph in test_graphs]
    matrices = [prediction.matrix for prediction in predictions]
    sample_ids = [str(graph.metadata["source_id"]) for graph in test_graphs]
    report = evaluate_maxwell_matrices(truth, matrices, sample_ids=sample_ids)
    component_errors = _component_log_errors(truth, matrices)
    uncertainty = np.asarray(
        [_prediction_uncertainty(prediction) for prediction in predictions],
        dtype=np.float64,
    )
    calibration_rows = []
    for level in INTERVAL_LEVELS:
        bounds = [prediction.interval(level) for prediction in predictions]
        calibration = interval_calibration(
            truth,
            [bound[0] for bound in bounds],
            [bound[1] for bound in bounds],
            nominal_coverage=level,
            groups=test_morphologies,
        )
        calibration_rows.append(calibration.to_dict())
    risk = risk_coverage_curve(component_errors, uncertainty)
    physical_valid = report.physical["valid"].astype(bool).to_numpy()
    return {
        "method": method,
        "choice": choice,
        "evidence": dict(evidence) if evidence is not None else None,
        "metrics": {
            PRIMARY_METRIC: float(np.mean(component_errors)),
            "macro_relative_frobenius": float(report.aggregate["macro_relative_frobenius"]),
            "physical_valid_rate": float(report.aggregate["physical_valid_rate"]),
        },
        "sample_ids": sample_ids,
        "sample_component_log_mae": component_errors.tolist(),
        "sample_relative_frobenius": report.per_sample["relative_frobenius"].tolist(),
        "sample_physical_valid": physical_valid.tolist(),
        "sample_uncertainty": uncertainty.tolist(),
        "calibration": calibration_rows,
        "risk_coverage": risk.curve.to_dict(orient="records"),
        "aurc": float(risk.aurc),
        "excess_aurc": float(risk.excess_aurc),
    }


def _adapt_or_foundation(
    foundation: TopoCapFoundationModel,
    support: Sequence[CapacitanceGraph],
) -> TopoCapFoundationModel | EBRAAdapter:
    if not support:
        return foundation
    return EBRAAdapter(foundation, ADAPTER_CONFIG).fit(support)


def _run_trial(
    *,
    source_control: Sequence[CapacitanceGraph],
    target: Sequence[CapacitanceGraph],
    target_control: Sequence[CapacitanceGraph],
    target_active: Sequence[CapacitanceGraph],
    target_v0: Sequence[CapacitanceGraph] | None,
    control_foundation: TopoCapFoundationModel,
    active_foundation: TopoCapFoundationModel,
    all_descriptor_foundation: TopoCapFoundationModel,
    shuffled_foundation: TopoCapFoundationModel,
    v0_foundation: TopoCapFoundationModel | None,
    shuffled_v0_foundation: TopoCapFoundationModel | None,
    retrieval_selection: SupportRetrievalSelection | None,
    retrieval_model_dir: Path,
    trial_key: str,
    state_digest: str,
    support_indices: Sequence[int],
    test_indices: Sequence[int],
    morphology_groups: Sequence[str],
    trial_seed: int,
) -> list[dict[str, Any]]:
    support_raw = [target[int(index)] for index in support_indices]
    support_control = [target_control[int(index)] for index in support_indices]
    support_active = [target_active[int(index)] for index in support_indices]
    support_v0 = [target_v0[int(index)] for index in support_indices] if target_v0 is not None else None
    test_raw = [target[int(index)] for index in test_indices]
    test_control = [target_control[int(index)] for index in test_indices]
    test_active = [target_active[int(index)] for index in test_indices]
    test_v0 = [target_v0[int(index)] for index in test_indices] if target_v0 is not None else None
    support_domains = [
        int(round(float(graph.metadata["split_parameter_values"]["finger_count"]))) for graph in support_raw
    ]
    test_morphologies = [str(morphology_groups[int(index)]) for index in test_indices]
    if bool(support_control) != (retrieval_selection is not None):
        raise ValueError("Retrieval selection must be present exactly when K>0 support is available.")

    candidates: list[tuple[str, Any, Sequence[CapacitanceGraph], str | None, Mapping[str, Any] | None]] = []
    candidates.append(
        (
            "source_control_foundation",
            control_foundation,
            test_control,
            "foundation",
            None,
        )
    )
    control_transfer = _adapt_or_foundation(control_foundation, support_control)
    candidates.append(
        (
            "source_control_ebra",
            control_transfer,
            test_control,
            "foundation" if not support_control else "transfer",
            None,
        )
    )

    control_lookup = {str(graph.metadata["source_id"]): view for graph, view in zip(target, target_control)}
    active_lookup = {str(graph.metadata["source_id"]): view for graph, view in zip(target, target_active)}
    gate = EvidenceGatedTopoCap(
        control_foundation,
        _lookup_view(control_lookup),
        _lookup_view(active_lookup),
        adapter_config=ADAPTER_CONFIG,
        specialist_config=ACTIVE_CONFIG,
        gate_config=EvidenceGateConfig(
            cross_validation_folds=3,
            minimum_specialist_support=8,
            minimum_group_count=3,
            specialist_relative_margin=0.0,
            specialist_standard_error_margin=1.0,
            random_seed=trial_seed,
        ),
    )
    gate.fit(support_raw, groups=support_domains)
    candidates.append(
        (
            "evidence_gated_topocap",
            gate,
            test_raw,
            gate.choice,
            asdict(gate.evidence_) if gate.evidence_ is not None else None,
        )
    )

    active_transfer = _adapt_or_foundation(active_foundation, support_active)
    candidates.append(
        (
            "source_active_gds_ebra",
            active_transfer,
            test_active,
            "foundation" if not support_active else "transfer",
            None,
        )
    )
    all_descriptor_transfer = _adapt_or_foundation(all_descriptor_foundation, support_raw)
    candidates.append(
        (
            "source_all_cached_descriptors_ebra",
            all_descriptor_transfer,
            test_raw,
            "foundation" if not support_raw else "transfer",
            None,
        )
    )

    shuffled_transfer = _adapt_or_foundation(shuffled_foundation, support_control)
    candidates.append(
        (
            "shuffled_source_ebra",
            shuffled_transfer,
            test_control,
            "shuffled_foundation" if not support_control else "shuffled_transfer",
            None,
        )
    )

    if retrieval_selection is not None:
        retrieved_source = tuple(source_control[index] for index in retrieval_selection.source_indices)
        retrieved_ids = tuple(str(graph.metadata["source_id"]) for graph in retrieved_source)
        if retrieved_ids != retrieval_selection.source_ids:
            raise ValueError("Retrieved source IDs do not match the frozen source indices.")
        support_ids = tuple(str(graph.metadata["source_id"]) for graph in support_raw)
        common_bindings = {
            "trial_key": trial_key,
            "support_ids_sha256": retrieval_source_ids_sha256(support_ids),
            "retrieval_source_ids_sha256": retrieval_selection.source_ids_sha256,
            "retrieval_source_budget": RETRIEVAL_SOURCE_BUDGET,
            "retrieval_protocol_version": retrieval_selection.protocol_version,
        }
        trial_model_dir = retrieval_model_dir / trial_key
        retrieval_model_path = trial_model_dir / "source-retrieval-2048.npz"
        retrieval_bindings = {
            **common_bindings,
            "method": "source_retrieval_2048_ebra",
            "source_labels": "correct",
        }
        retrieval_foundation = _fit_or_load_model(
            retrieval_model_path,
            CONTROL_CONFIG,
            lambda: retrieved_source,
            "support-conditioned source retrieval foundation",
            state_digest,
            bindings=retrieval_bindings,
        )
        shuffled_seed = int(trial_seed) ^ RETRIEVAL_SHUFFLE_SEED_XOR
        shuffled_retrieval_path = trial_model_dir / "shuffled-source-retrieval-2048.npz"
        shuffled_retrieval_bindings = {
            **common_bindings,
            "method": "shuffled_source_retrieval_2048_ebra",
            "source_labels": "permuted_within_retrieval_set",
            "shuffle_seed": shuffled_seed,
        }
        shuffled_retrieval_foundation = _fit_or_load_model(
            shuffled_retrieval_path,
            CONTROL_CONFIG,
            lambda: _shuffled_targets(retrieved_source, shuffled_seed),
            "shuffled support-conditioned source retrieval foundation",
            state_digest,
            bindings=shuffled_retrieval_bindings,
        )
        retrieval_evidence = {
            **retrieval_bindings,
            "state_digest": state_digest,
            "model_checkpoint": str(retrieval_model_path),
            "model_sha256": _sha256_file(retrieval_model_path),
        }
        shuffled_retrieval_evidence = {
            **shuffled_retrieval_bindings,
            "state_digest": state_digest,
            "model_checkpoint": str(shuffled_retrieval_path),
            "model_sha256": _sha256_file(shuffled_retrieval_path),
        }
        candidates.extend(
            (
                (
                    "source_retrieval_2048_ebra",
                    EBRAAdapter(retrieval_foundation, ADAPTER_CONFIG).fit(support_control),
                    test_control,
                    "transfer",
                    retrieval_evidence,
                ),
                (
                    "shuffled_source_retrieval_2048_ebra",
                    EBRAAdapter(shuffled_retrieval_foundation, ADAPTER_CONFIG).fit(support_control),
                    test_control,
                    "shuffled_transfer",
                    shuffled_retrieval_evidence,
                ),
            )
        )

    if support_active:
        specialist = TopoCapFoundationModel(ACTIVE_CONFIG).fit(support_active)
        candidates.append(("target_active_gds_specialist", specialist, test_active, "specialist", None))
        scratch = TopoCapFoundationModel(CONTROL_CONFIG).fit(support_control)
        candidates.append(("target_control_scratch", scratch, test_control, "scratch", None))

    if v0_foundation is not None and support_v0 is not None and test_v0 is not None:
        v0_transfer = _adapt_or_foundation(v0_foundation, support_v0)
        candidates.append(
            (
                "v0_gaussian64_ebra",
                v0_transfer,
                test_v0,
                "foundation" if not support_v0 else "transfer",
                None,
            )
        )
        if support_v0:
            v0_scratch = TopoCapFoundationModel(V0_CONFIG).fit(support_v0)
            candidates.append(
                (
                    "target_v0_gaussian64_scratch",
                    v0_scratch,
                    test_v0,
                    "scratch",
                    None,
                )
            )
        if shuffled_v0_foundation is not None:
            shuffled_v0_transfer = _adapt_or_foundation(shuffled_v0_foundation, support_v0)
            candidates.append(
                (
                    "shuffled_v0_gaussian64_ebra",
                    shuffled_v0_transfer,
                    test_v0,
                    "shuffled_foundation" if not support_v0 else "shuffled_transfer",
                    None,
                )
            )

    method_results = []
    for method, model, test_graphs, choice, evidence in candidates:
        predictions = _predict_many(model, test_graphs)
        method_results.append(
            _evaluate_method(
                method,
                predictions,
                test_graphs,
                test_morphologies,
                choice,
                evidence,
            )
        )

    baseline = next(
        (result for result in method_results if result["method"] == "target_control_scratch"),
        None,
    )
    for result in method_results:
        result[PAIRED_SOURCE_GAIN_METRIC] = None
        if baseline is None:
            result["paired_gain_vs_target_control"] = None
            result["negative_transfer_rate_vs_target_control"] = None
            continue
        comparison = paired_group_bootstrap(
            baseline["sample_component_log_mae"],
            result["sample_component_log_mae"],
            test_morphologies,
            higher_is_better=False,
            n_bootstrap=500,
            confidence=0.95,
            seed=_stable_seed(trial_seed, result["method"], "paired-gain"),
        )
        result["paired_gain_vs_target_control"] = comparison.to_dict()
        result["negative_transfer_rate_vs_target_control"] = float(comparison.negative_transfer_rate)

    results_by_method = {result["method"]: result for result in method_results}
    source_control_pairs = (
        ("source_control_ebra", "shuffled_source_ebra"),
        ("source_retrieval_2048_ebra", "shuffled_source_retrieval_2048_ebra"),
        ("v0_gaussian64_ebra", "shuffled_v0_gaussian64_ebra"),
    )
    for learned_name, shuffled_name in source_control_pairs:
        learned = results_by_method.get(learned_name)
        shuffled = results_by_method.get(shuffled_name)
        if learned is None or shuffled is None:
            continue
        comparison = paired_group_bootstrap(
            shuffled["sample_component_log_mae"],
            learned["sample_component_log_mae"],
            test_morphologies,
            higher_is_better=False,
            n_bootstrap=500,
            confidence=0.95,
            seed=_stable_seed(trial_seed, learned_name, PAIRED_SOURCE_GAIN_METRIC),
        )
        learned[PAIRED_SOURCE_GAIN_METRIC] = comparison.to_dict()
    return method_results


def _trial_key(domain: int, repeat: int, support_size: int) -> str:
    return f"finger-{domain:02d}_repeat-{repeat:03d}_support-{support_size:03d}"


def _expected_method_names(*, has_support: bool, include_v0: bool) -> set[str]:
    methods = {
        "source_control_foundation",
        "source_control_ebra",
        "evidence_gated_topocap",
        "source_active_gds_ebra",
        "source_all_cached_descriptors_ebra",
        "shuffled_source_ebra",
    }
    if has_support:
        methods.update(
            {
                "source_retrieval_2048_ebra",
                "shuffled_source_retrieval_2048_ebra",
                "target_active_gds_specialist",
                "target_control_scratch",
            }
        )
    if include_v0:
        methods.update({"v0_gaussian64_ebra", "shuffled_v0_gaussian64_ebra"})
        if has_support:
            methods.add("target_v0_gaussian64_scratch")
    return methods


def _seal_trial_checkpoint(trial: dict[str, Any]) -> None:
    payload = {key: value for key, value in trial.items() if key != "checkpoint_sha256"}
    trial["checkpoint_sha256"] = content_sha256(payload)


def _verify_trial_checkpoint(trial: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    stored_hash = trial.get("checkpoint_sha256")
    payload = {key: value for key, value in trial.items() if key != "checkpoint_sha256"}
    if not isinstance(stored_hash, str) or stored_hash != content_sha256(payload):
        raise ValueError("Trial checkpoint content hash does not match its payload.")
    mismatches = [key for key, value in expected.items() if key != "expected_methods" and trial.get(key) != value]
    if mismatches:
        raise ValueError(f"Trial checkpoint identity mismatch for fields: {mismatches}.")
    observed_methods = {str(result.get("method")) for result in trial.get("methods", [])}
    if observed_methods != expected["expected_methods"]:
        raise ValueError("Trial checkpoint method set does not match this study.")


def _bootstrap_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    n_bootstrap: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    frame = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    cluster_means = frame.groupby("cluster", sort=True)["value"].mean().to_numpy(dtype=float)
    if not len(cluster_means):
        raise ValueError("Cannot bootstrap an empty result group.")
    estimate = float(np.mean(cluster_means))
    if len(cluster_means) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(cluster_means), size=len(cluster_means))
        draws[index] = float(np.mean(cluster_means[selected]))
    alpha = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def _flatten_trials(trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        shared = {
            "trial_key": trial["trial_key"],
            "state_digest": trial["state_digest"],
            "repeat": trial["repeat"],
            "trial_seed": trial["trial_seed"],
            "outer_fold": trial["outer_fold"],
            "target_domain": trial["target_domain"],
            "validation_domain": trial["validation_domain"],
            "requested_support_size": trial["requested_support_size"],
            "support_size": trial["support_size"],
            "support_ids": trial["support_ids"],
            "retrieval_source_count": len(trial["retrieval_source_ids"]),
            "retrieval_source_ids_sha256": trial["retrieval_source_ids_sha256"],
            "test_size": trial["test_size"],
            "quick": trial["quick"],
        }
        for result in trial["methods"]:
            rows.append(
                {
                    **shared,
                    "method": result["method"],
                    "choice": result["choice"],
                    **result["metrics"],
                    "paired_gain_vs_target_control": result["paired_gain_vs_target_control"],
                    PAIRED_SOURCE_GAIN_METRIC: result[PAIRED_SOURCE_GAIN_METRIC],
                    "negative_transfer_rate_vs_target_control": result["negative_transfer_rate_vs_target_control"],
                    "gate_evidence": result["evidence"],
                    "calibration": result["calibration"],
                    "aurc": result["aurc"],
                    "excess_aurc": result["excess_aurc"],
                    "sample_ids": result["sample_ids"],
                    "sample_component_log_mae": result["sample_component_log_mae"],
                    "sample_relative_frobenius": result["sample_relative_frobenius"],
                    "sample_physical_valid": result["sample_physical_valid"],
                    "sample_uncertainty": result["sample_uncertainty"],
                }
            )
    return rows


def _data_audit(
    source: Sequence[CapacitanceGraph],
    target: Sequence[CapacitanceGraph],
    cache_identity: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    overlap_counts = _cross_family_identity_overlaps(source, target)
    for family, graphs in ((GENERALIZED_FAMILY, source), (CAPN_FAMILY, target)):
        groups: dict[tuple[str, int], list[CapacitanceGraph]] = {}
        for graph in graphs:
            key = str(graph.metadata.get("source_campaign", "unknown")), graph.node_count
            groups.setdefault(key, []).append(graph)
        for (campaign, node_count), group in sorted(groups.items()):
            records = len(group)
            rows.append(
                {
                    "dataset_family": family,
                    "campaign": campaign,
                    "records": int(records),
                    "node_count": int(node_count),
                    "available_release_records": int(cache_identity.get("family_counts", {}).get(family, records)),
                    "study_role": "source" if family == GENERALIZED_FAMILY else "target",
                    "unique_gds": len({str(graph.metadata["gds_sha256"]) for graph in group}),
                    "matrix_unit": ",".join(sorted({str(graph.metadata["capacitance_unit"]) for graph in group})),
                    "cross_family_identity_overlap_total": sum(overlap_counts.values()),
                }
            )
    return pd.DataFrame(rows)


def _learning_curves(raw_rows: Sequence[Mapping[str, Any]], seed: int, quick: bool) -> pd.DataFrame:
    frame = pd.DataFrame(raw_rows)
    rows = []
    bootstrap_count = 300 if quick else 2_000
    for (method, support_size), group in frame.groupby(["method", "support_size"], sort=True):
        clusters = group["target_domain"].astype(str).tolist()
        for metric in REPORT_METRICS:
            estimate, low, high = _bootstrap_interval(
                group[metric].to_numpy(dtype=float),
                clusters,
                seed=_stable_seed(seed, method, support_size, metric),
                n_bootstrap=bootstrap_count,
            )
            rows.append(
                {
                    "method": method,
                    "support_size": int(support_size),
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "target_domain": "macro across held-out finger-count domains",
                    "n_trials": int(len(group)),
                }
            )

        for paired_metric in (PAIRED_GAIN_METRIC, PAIRED_SOURCE_GAIN_METRIC):
            paired_mask = group[paired_metric].map(lambda value: isinstance(value, Mapping))
            paired_group = group.loc[paired_mask]
            if paired_group.empty:
                continue
            paired_values = np.asarray(
                [float(value["estimate"]) for value in paired_group[paired_metric]],
                dtype=float,
            )
            paired_clusters = paired_group["target_domain"].astype(str).tolist()
            estimate, low, high = _bootstrap_interval(
                paired_values,
                paired_clusters,
                seed=_stable_seed(seed, method, support_size, paired_metric),
                n_bootstrap=bootstrap_count,
            )
            rows.append(
                {
                    "method": method,
                    "support_size": int(support_size),
                    "metric": paired_metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "target_domain": "macro across held-out finger-count domains",
                    "n_trials": int(len(paired_group)),
                }
            )
    return pd.DataFrame(rows)


def _uncertainty_artifact(
    raw_rows: Sequence[Mapping[str, Any]],
    support_sizes: Sequence[int],
) -> pd.DataFrame:
    frame = pd.DataFrame(raw_rows)
    positive = [size for size in support_sizes if size > 0]
    selected_support = min(positive, key=lambda value: abs(value - 16)) if positive else 0
    frame = frame.loc[frame["support_size"] == selected_support]
    rows = []
    for method, group in frame.groupby("method", sort=True):
        calibration_by_level: dict[float, list[float]] = {level: [] for level in INTERVAL_LEVELS}
        errors: list[float] = []
        uncertainty: list[float] = []
        for record in group.to_dict(orient="records"):
            for calibration in record["calibration"]:
                calibration_by_level[float(calibration["nominal_coverage"])].append(
                    float(calibration["group_macro_coverage"])
                )
            errors.extend(map(float, record["sample_component_log_mae"]))
            uncertainty.extend(map(float, record["sample_uncertainty"]))
        for level, observed in calibration_by_level.items():
            rows.append(
                {
                    "method": f"{method} (K={selected_support})",
                    "curve": "calibration",
                    "x": level,
                    "y": float(np.mean(observed)),
                    "support_size": selected_support,
                }
            )
        risk = risk_coverage_curve(errors, uncertainty)
        curve = risk.curve
        if len(curve) > 25:
            positions = np.unique(np.linspace(0, len(curve) - 1, 25).round().astype(int))
            curve = curve.iloc[positions]
        for record in curve.to_dict(orient="records"):
            rows.append(
                {
                    "method": f"{method} (K={selected_support})",
                    "curve": "risk_coverage",
                    "x": float(record["coverage"]),
                    "y": float(record["risk"]),
                    "support_size": selected_support,
                }
            )
    return pd.DataFrame(rows)


def _ablations(
    raw_rows: Sequence[Mapping[str, Any]],
    support_sizes: Sequence[int],
    seed: int,
    quick: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(raw_rows)
    positive = [size for size in support_sizes if size > 0]
    selected_support = min(positive, key=lambda value: abs(value - 16)) if positive else 0
    frame = frame.loc[frame["support_size"] == selected_support].copy()
    bootstrap_count = 300 if quick else 2_000
    rows = []
    clusters = frame["target_domain"].astype(str)
    for method, group in frame.groupby("method", sort=True):
        method_clusters = clusters.loc[group.index].tolist()
        for metric in REPORT_METRICS:
            estimate, low, high = _bootstrap_interval(
                group[metric].to_numpy(dtype=float),
                method_clusters,
                seed=_stable_seed(seed, "ablation", method, metric),
                n_bootstrap=bootstrap_count,
            )
            rows.append(
                {
                    "variant": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "support_size": selected_support,
                }
            )
        for paired_metric in (PAIRED_GAIN_METRIC, PAIRED_SOURCE_GAIN_METRIC):
            paired_mask = group[paired_metric].map(lambda value: isinstance(value, Mapping))
            paired_group = group.loc[paired_mask]
            if paired_group.empty:
                continue
            paired_values = np.asarray(
                [float(value["estimate"]) for value in paired_group[paired_metric]],
                dtype=float,
            )
            paired_clusters = paired_group["target_domain"].astype(str).tolist()
            estimate, low, high = _bootstrap_interval(
                paired_values,
                paired_clusters,
                seed=_stable_seed(seed, "ablation", method, paired_metric),
                n_bootstrap=bootstrap_count,
            )
            rows.append(
                {
                    "variant": method,
                    "metric": paired_metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "support_size": selected_support,
                }
            )
        finite_negative = group["negative_transfer_rate_vs_target_control"].dropna()
        if not finite_negative.empty:
            estimate, low, high = _bootstrap_interval(
                finite_negative.to_numpy(dtype=float),
                clusters.loc[finite_negative.index].tolist(),
                seed=_stable_seed(seed, "negative-transfer", method),
                n_bootstrap=bootstrap_count,
            )
            rows.append(
                {
                    "variant": method,
                    "metric": "negative_transfer_rate_vs_target_control",
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "support_size": selected_support,
                }
            )

    gate = frame.loc[frame["method"] == "evidence_gated_topocap"]
    if not gate.empty:
        for choice in ("foundation", "transfer", "specialist"):
            indicator = (gate["choice"] == choice).astype(float)
            estimate, low, high = _bootstrap_interval(
                indicator.to_numpy(),
                gate["target_domain"].astype(str).tolist(),
                seed=_stable_seed(seed, "gate-choice", choice),
                n_bootstrap=bootstrap_count,
            )
            rows.append(
                {
                    "variant": f"evidence gate chose {choice}",
                    "metric": "gate_choice_rate",
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "support_size": selected_support,
                }
            )
    return pd.DataFrame(rows)


def _topology_checks(
    model: TopoCapFoundationModel,
    control_template: CapacitanceGraph,
    model_path: Path,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for node_count in range(2, 17):
        reference = int(rng.integers(0, node_count))
        node_features = np.zeros((node_count, control_template.node_feature_dim), dtype=float)
        node_features[reference, 0] = 1.0
        edge_index = canonical_edge_index(node_count)
        edge_features = np.zeros((edge_index.shape[1], control_template.edge_feature_dim), dtype=float)
        edge_features[:, 0] = np.asarray(
            [reference in pair for pair in edge_index.T],
            dtype=float,
        )
        graph = CapacitanceGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=np.asarray([node_count, 1.0], dtype=float),
            parameter_values=control_template.parameter_values,
            parameter_features=control_template.parameter_features,
            parameter_names=control_template.parameter_names,
            net_ids=tuple(f"net_{index}" for index in range(node_count)),
            metadata={"synthetic_topology_check": True},
        )
        prediction = model.predict(graph).matrix
        diagnostics = maxwell_physical_diagnostics(prediction)
        permutation = rng.permutation(node_count)
        reordered = model.predict(graph.reorder_nodes(permutation)).matrix
        permutation_error = float(np.max(np.abs(reordered - prediction[np.ix_(permutation, permutation)])))
        components = maxwell_to_components(prediction)
        reconstruction_error = float(np.max(np.abs(components.to_matrix() - prediction)))
        checks = (
            ("permutation_equivariance", permutation_error, 1.0e-8, permutation_error <= 1.0e-8),
            (
                "symmetry",
                diagnostics.symmetry_max_abs_ff,
                1.0e-9,
                diagnostics.symmetric,
            ),
            (
                "offdiagonal_sign_violations",
                float(diagnostics.positive_offdiagonal_count),
                0.0,
                diagnostics.offdiagonal_nonpositive,
            ),
            (
                "diagonal_dominance_violation",
                max(0.0, -diagnostics.diagonal_dominance_min_margin_ff),
                1.0e-9,
                diagnostics.diagonally_dominant,
            ),
            (
                "psd_violation",
                max(0.0, -diagnostics.minimum_eigenvalue_ff),
                1.0e-9,
                diagnostics.positive_semidefinite,
            ),
            (
                "component_reconstruction",
                reconstruction_error,
                1.0e-10,
                reconstruction_error <= 1.0e-10,
            ),
        )
        for check, value, tolerance, passed in checks:
            rows.append(
                {
                    "check": check,
                    "node_count": node_count,
                    "value": float(value),
                    "tolerance": float(tolerance),
                    "passed": bool(passed),
                    "evidence_type": "architecture property test",
                }
            )

    loaded = TopoCapFoundationModel.load(model_path)
    original_matrix = model.predict(control_template).matrix
    loaded_matrix = loaded.predict(control_template).matrix
    checkpoint_error = float(np.max(np.abs(original_matrix - loaded_matrix)))
    rows.append(
        {
            "check": "checkpoint_roundtrip",
            "node_count": control_template.node_count,
            "value": checkpoint_error,
            "tolerance": 0.0,
            "passed": checkpoint_error == 0.0,
            "evidence_type": "serialization property test",
        }
    )
    return pd.DataFrame(rows)


def _diffusion_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "conditional_geometry_diffusion",
                "solver_budget": 0,
                "metric": "inverse_design_performance_not_run",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "status": "DEFERRED_NOT_RUN",
                "performance_available": False,
                "reason": (
                    "Both observed simulation families have N=3 conductors. Diffusion is deferred "
                    "until variable-N topology data and an equal-true-solver-budget inverse-design "
                    "benchmark exist; missing estimates explicitly mean no performance was measured."
                ),
                "observed_node_counts": "3",
            }
        ]
    )


def _write_report_bundle(
    *,
    output_dir: Path,
    state: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    source: Sequence[CapacitanceGraph],
    target: Sequence[CapacitanceGraph],
    trials: Sequence[Mapping[str, Any]],
    control_foundation: TopoCapFoundationModel,
    control_template: CapacitanceGraph,
    control_model_path: Path,
    support_sizes: Sequence[int],
    seed: int,
    quick: bool,
    v0_path: Path | None,
) -> None:
    raw_rows = _flatten_trials(trials)
    artifact_frames = {
        "data_audit": _data_audit(source, target, cache_identity),
        "learning_curves": _learning_curves(raw_rows, seed, quick),
        "uncertainty": _uncertainty_artifact(raw_rows, support_sizes),
        "ablations": _ablations(raw_rows, support_sizes, seed, quick),
        "topology_checks": _topology_checks(
            control_foundation,
            control_template,
            control_model_path,
            seed,
        ),
        "diffusion_decision": _diffusion_decision(),
    }
    descriptors: dict[str, dict[str, Any]] = {}
    raw_path = output_dir / "raw_trials.jsonl"
    _atomic_write_jsonl(raw_path, raw_rows)
    descriptors["raw_trials"] = {
        "path": raw_path.name,
        "sha256": _sha256_file(raw_path),
        "rows": len(raw_rows),
        "schema_version": TRIAL_SCHEMA_VERSION,
    }
    for logical_name, frame in artifact_frames.items():
        path = output_dir / f"{logical_name}.csv"
        _atomic_write_csv(path, frame)
        descriptors[logical_name] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "rows": len(frame),
            "schema_version": REPORT_ARTIFACT_SCHEMA_VERSION,
        }

    conclusions = []
    selected_support = min(
        [size for size in support_sizes if size > 0],
        key=lambda value: abs(value - 16),
    )
    ablations = artifact_frames["ablations"]
    evidence_row = ablations.loc[
        (ablations["variant"] == "evidence_gated_topocap") & (ablations["metric"] == PRIMARY_METRIC)
    ]
    if not evidence_row.empty:
        row = evidence_row.iloc[0]
        conclusions.append(
            {
                "statement": (
                    "Evidence-gated cross-family performance is reported without declaring "
                    "superiority unless its paired interval supports that claim."
                ),
                "evidence": (
                    f"ablations.csv:{PRIMARY_METRIC}, K={selected_support}, "
                    f"estimate={row['estimate']:.6g}, 95% CI=[{row['ci_low']:.6g}, {row['ci_high']:.6g}]"
                ),
            }
        )
    learning_curves = artifact_frames["learning_curves"]
    primary_comparisons = (
        (
            "evidence_gated_topocap",
            PAIRED_GAIN_METRIC,
            "EGRA versus the same-budget target-control specialist",
        ),
        (
            "source_control_ebra",
            PAIRED_SOURCE_GAIN_METRIC,
            "canonical-control source labels versus shuffled source labels",
        ),
        (
            "source_retrieval_2048_ebra",
            PAIRED_SOURCE_GAIN_METRIC,
            "support-conditioned retrieved source labels versus matched shuffled source labels",
        ),
        (
            "v0_gaussian64_ebra",
            PAIRED_SOURCE_GAIN_METRIC,
            "compressed-v0 source labels versus shuffled source labels",
        ),
    )
    for method, metric, label in primary_comparisons:
        primary = learning_curves.loc[
            (learning_curves["method"] == method)
            & (learning_curves["metric"] == metric)
            & (learning_curves["support_size"] == selected_support)
        ]
        if primary.empty:
            continue
        row = primary.iloc[0]
        classification = "positive" if row["ci_low"] > 0 else "negative" if row["ci_high"] < 0 else "inconclusive"
        conclusions.append(
            {
                "statement": (
                    f"At the predeclared primary budget K={selected_support}, the exploratory "
                    f"paired result for {label} was {classification}."
                ),
                "evidence": (
                    f"learning_curves.csv:{method}:{metric}, K={selected_support}, "
                    f"estimate={row['estimate']:.6g}, outer-domain 95% CI="
                    f"[{row['ci_low']:.6g}, {row['ci_high']:.6g}]"
                ),
            }
        )
    conclusions.append(
        {
            "statement": "Arbitrary-N behavior is architecture-tested, not empirically validated.",
            "evidence": (
                "topology_checks.csv tests N=2..16; data_audit.csv shows both observed families "
                "contain only N=3 matrices"
            ),
        }
    )
    manifest = {
        "schema_version": REPORT_ARTIFACT_SCHEMA_VERSION,
        "created_unix": time.time(),
        "state_digest": state["digest"],
        "study_status": "QUICK_SMOKE_NON_CLAIM_BEARING" if quick else "EXPLORATORY_COMPLETE",
        "cache": dict(cache_identity),
        "v0_parquet": str(v0_path.resolve()) if v0_path is not None else None,
        "artifacts": descriptors,
        "conclusions": conclusions,
        "diffusion": {
            "status": "DEFERRED_NOT_RUN",
            "reason": "Only one observed conductor count (N=3); no inverse-design solver benchmark.",
        },
        "scientific_scope": {
            "source_family": GENERALIZED_FAMILY,
            "target_family": CAPN_FAMILY,
            "outer_domains": "leave-one-finger-count-out",
            "normalization": "source-only",
            "target_test_rows_used_for_fit": False,
            "retrieval_target_test_features_or_labels_used": False,
            "retrieval_source_budget": RETRIEVAL_SOURCE_BUDGET,
            "observed_node_counts": [3],
            "evidence_tier": "exploratory; protocol developed against this public target release",
            "primary_support_size": selected_support,
            "interval_resampling_unit": "held-out finger-count domain; repeats averaged within domain",
            "learning_curve_intervals": "pointwise exploratory; no simultaneous multiplicity correction",
            "cross_family_identity_overlap_counts": _cross_family_identity_overlaps(source, target),
            "uncertainty_intervals": (
                "marginal component envelopes propagated to matrix entries; descriptive, not "
                "jointly calibrated nominal matrix intervals"
            ),
            "capn_net_alignment": (
                "explicit legacy generator polygon order; independent asymmetric golden validation pending"
            ),
        },
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)


def main() -> int:
    args = _parse_args()
    cache_jsonl = args.cache_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    v0_path = args.v0_parquet.expanduser().resolve() if args.v0_parquet is not None else None
    print("Verifying cache manifest and JSONL SHA-256...", flush=True)
    cache_identity = _verify_cache_input(cache_jsonl)
    v0_sha256 = _sha256_file(v0_path) if v0_path is not None else None
    state = _build_state(args, cache_identity, v0_sha256)
    _initialize_output(output_dir, state)

    print("Loading validated graph records...", flush=True)
    source, target = _load_graphs(cache_jsonl, args.quick)
    print(f"Loaded {len(source):,} source and {len(target):,} target graphs.", flush=True)

    target_control = tuple(build_topology_control_view(graph) for graph in target)
    target_active = tuple(build_active_geometry_view(graph) for graph in target)
    model_dir = output_dir / "models"
    retrieval_model_dir = model_dir / "support-conditioned-retrieval"
    control_model_path = model_dir / "source-control.npz"
    active_model_path = model_dir / "source-active-gds.npz"
    all_descriptor_model_path = model_dir / "source-all-cached-descriptors.npz"
    shuffled_model_path = model_dir / "shuffled-source-control.npz"

    control_source_cache: tuple[CapacitanceGraph, ...] | None = None

    def control_source() -> tuple[CapacitanceGraph, ...]:
        nonlocal control_source_cache
        if control_source_cache is None:
            control_source_cache = _source_view(source, build_topology_control_view)
        return control_source_cache

    control_foundation = _fit_or_load_model(
        control_model_path,
        CONTROL_CONFIG,
        control_source,
        "topology/control foundation",
        state["digest"],
    )
    shuffled_foundation = _fit_or_load_model(
        shuffled_model_path,
        CONTROL_CONFIG,
        lambda: _shuffled_targets(
            control_source(),
            _stable_seed(args.seed, "shuffled-source-targets"),
        ),
        "shuffled-source negative control",
        state["digest"],
    )
    source_control = control_source()
    source_retriever = SupportConditionedSourceRetriever(source_control)
    active_foundation = _fit_or_load_model(
        active_model_path,
        ACTIVE_CONFIG,
        lambda: _source_view(source, build_active_geometry_view),
        "active-GDS foundation",
        state["digest"],
    )
    all_descriptor_foundation = _fit_or_load_model(
        all_descriptor_model_path,
        ALL_DESCRIPTOR_CONFIG,
        lambda: source,
        "all-cached-descriptor foundation",
        state["digest"],
    )

    target_v0: tuple[CapacitanceGraph, ...] | None = None
    v0_foundation: TopoCapFoundationModel | None = None
    shuffled_v0_foundation: TopoCapFoundationModel | None = None
    if v0_path is not None:
        print("Streaming and aligning optional v0 embeddings...", flush=True)
        required_keys = {_graph_key(graph) for graph in (*source, *target)}
        source_keys = {_graph_key(graph) for graph in source}
        sketches, v0_input_mean, v0_input_scale = _load_v0_sketches(
            v0_path,
            required_keys,
            source_keys,
        )
        v0_mean, v0_scale = _fit_v0_standardization(source, sketches)
        preprocess_path = model_dir / "v0-gaussian64-preprocess.npz"
        with tempfile.NamedTemporaryFile(
            dir=model_dir,
            prefix=".v0-gaussian64-preprocess.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_preprocess = Path(stream.name)
            np.savez_compressed(
                stream,
                mean=v0_mean,
                scale=v0_scale,
                input_mean=v0_input_mean,
                input_scale=v0_input_scale,
                projection_seed=np.asarray(V0_SKETCH_SEED, dtype=np.int64),
                input_dimensions=np.asarray(V0_INPUT_DIMENSION, dtype=np.int64),
                output_dimensions=np.asarray(V0_SKETCH_DIMENSION, dtype=np.int64),
                source_only=np.asarray(True),
                standardize_before_projection=np.asarray(True),
            )
        os.replace(temporary_preprocess, preprocess_path)
        source_v0 = tuple(_v0_graph(graph, sketches[_graph_key(graph)], v0_mean, v0_scale) for graph in source)
        target_v0 = tuple(_v0_graph(graph, sketches[_graph_key(graph)], v0_mean, v0_scale) for graph in target)
        v0_foundation = _fit_or_load_model(
            model_dir / "source-v0-gaussian64.npz",
            V0_CONFIG,
            lambda: source_v0,
            "v0 fixed-Gaussian-64 foundation",
            state["digest"],
        )
        shuffled_v0_foundation = _fit_or_load_model(
            model_dir / "shuffled-source-v0-gaussian64.npz",
            V0_CONFIG,
            lambda: _shuffled_targets(
                source_v0,
                _stable_seed(args.seed, "shuffled-source-v0-targets"),
            ),
            "shuffled-source v0 negative control",
            state["digest"],
        )
        del source_v0, sketches

    frame = _target_frame(target)
    morphology_groups = capn_group_labels(frame, mode="morphology")
    outer_splits = sorted(
        capn_outer_splits(frame, mode="finger_count", validation="next", seed=args.seed),
        key=lambda split: _domain_for_indices(target, split.test_idx),
    )
    if args.quick:
        outer_splits = outer_splits[:3]
    support_sizes = (0, 1, 2, 4, 8, 16) if args.quick else DEFAULT_SUPPORT_SIZES
    repeats = 1 if args.quick else args.repeats

    completed_trials: list[dict[str, Any]] = []
    total_trials = len(outer_splits) * repeats * len(support_sizes)
    trial_number = 0
    for outer_index, outer in enumerate(outer_splits):
        target_domain = _domain_for_indices(target, outer.test_idx)
        validation_domain = _domain_for_indices(target, outer.validation_idx)
        test_indices = outer.test_idx
        if args.quick:
            test_indices = _hashed_subset(test_indices, target, limit=24)
        for repeat in range(repeats):
            repeat_seed = _stable_seed(args.seed, "support", target_domain, repeat)
            for requested_support in support_sizes:
                trial_number += 1
                nested = make_nested_target_split(
                    outer,
                    morphology_groups,
                    support_size=requested_support,
                    seed=repeat_seed,
                    name=f"finger-{target_domain}-repeat-{repeat}-K-{requested_support}",
                )
                support_indices = nested.adaptation.train_idx
                support_ids = [str(target[int(index)].metadata["source_id"]) for index in support_indices]
                key = _trial_key(target_domain, repeat, len(support_indices))
                trial_seed = _stable_seed(
                    args.seed,
                    "trial",
                    target_domain,
                    repeat,
                    requested_support,
                )
                retrieval_selection = None
                if len(support_indices):
                    retrieval_support = [target_control[int(index)] for index in support_indices]
                    retrieval_selection = source_retriever.retrieve(retrieval_support)
                retrieval_source_ids = list(retrieval_selection.source_ids) if retrieval_selection is not None else []
                retrieval_source_ids_hash = (
                    retrieval_selection.source_ids_sha256
                    if retrieval_selection is not None
                    else retrieval_source_ids_sha256(())
                )
                expected_checkpoint = {
                    "schema_version": TRIAL_SCHEMA_VERSION,
                    "state_digest": state["digest"],
                    "trial_key": key,
                    "outer_fold": outer_index,
                    "target_domain": target_domain,
                    "validation_domain": validation_domain,
                    "repeat": repeat,
                    "trial_seed": trial_seed,
                    "requested_support_size": requested_support,
                    "support_size": len(support_indices),
                    "support_ids": support_ids,
                    "retrieval_source_ids": retrieval_source_ids,
                    "retrieval_source_ids_sha256": retrieval_source_ids_hash,
                    "test_size": len(test_indices),
                    "quick": bool(args.quick),
                    "expected_methods": _expected_method_names(
                        has_support=len(support_indices) > 0,
                        include_v0=v0_foundation is not None,
                    ),
                }
                trial_path = output_dir / "trials" / f"{key}.json"
                if trial_path.is_file():
                    trial = json.loads(trial_path.read_text(encoding="utf-8"))
                    _verify_trial_checkpoint(trial, expected_checkpoint)
                    completed_trials.append(trial)
                    print(f"[{trial_number}/{total_trials}] resume {key}", flush=True)
                    continue

                print(f"[{trial_number}/{total_trials}] run {key}", flush=True)
                methods = _run_trial(
                    source_control=source_control,
                    target=target,
                    target_control=target_control,
                    target_active=target_active,
                    target_v0=target_v0,
                    control_foundation=control_foundation,
                    active_foundation=active_foundation,
                    all_descriptor_foundation=all_descriptor_foundation,
                    shuffled_foundation=shuffled_foundation,
                    v0_foundation=v0_foundation,
                    shuffled_v0_foundation=shuffled_v0_foundation,
                    retrieval_selection=retrieval_selection,
                    retrieval_model_dir=retrieval_model_dir,
                    trial_key=key,
                    state_digest=state["digest"],
                    support_indices=support_indices,
                    test_indices=test_indices,
                    morphology_groups=morphology_groups,
                    trial_seed=trial_seed,
                )
                trial = {
                    "schema_version": TRIAL_SCHEMA_VERSION,
                    "state_digest": state["digest"],
                    "trial_key": key,
                    "outer_fold": outer_index,
                    "target_domain": target_domain,
                    "validation_domain": validation_domain,
                    "repeat": repeat,
                    "trial_seed": trial_seed,
                    "requested_support_size": requested_support,
                    "support_size": len(support_indices),
                    "support_ids": support_ids,
                    "retrieval_source_ids": retrieval_source_ids,
                    "retrieval_source_ids_sha256": retrieval_source_ids_hash,
                    "test_size": len(test_indices),
                    "quick": bool(args.quick),
                    "methods": methods,
                }
                _seal_trial_checkpoint(trial)
                _atomic_write_json(trial_path, trial)
                completed_trials.append(trial)

    print("Building compact report artifacts from completed trial checkpoints...", flush=True)
    _write_report_bundle(
        output_dir=output_dir,
        state=state,
        cache_identity=cache_identity,
        source=source,
        target=target,
        trials=completed_trials,
        control_foundation=control_foundation,
        control_template=target_control[0],
        control_model_path=control_model_path,
        support_sizes=support_sizes,
        seed=args.seed,
        quick=args.quick,
        v0_path=v0_path,
    )
    print(f"Study complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed atomic trial checkpoints are safe to resume.", file=sys.stderr)
        raise SystemExit(130) from None
