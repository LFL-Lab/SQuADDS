"""Frozen support-conditioned TopoCap retrieval tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from squadds.ml import SupportConditionedSourceRetriever as PublicRetriever
from squadds.ml.topocap import (
    RETRIEVAL_CONTROL_NAMES,
    RETRIEVAL_SOURCE_BUDGET,
    SupportConditionedSourceRetriever,
    SupportRetrievalSelection,
    TopoCapConfig,
    retrieval_source_ids_sha256,
)
from squadds.ml.topocap.schema import CapacitanceGraph, canonical_edge_index
from squadds.ml.topocap.targets import components_to_maxwell


def _control_graph(
    source_id: str,
    *,
    count: int,
    length: float,
    width: float,
    gap: float,
    target_scale: float | None = 1.0,
) -> CapacitanceGraph:
    edges = canonical_edge_index(3)
    matrix = None
    if target_scale is not None:
        matrix = components_to_maxwell(
            np.asarray([1.0, 1.1, 1.2]) * target_scale,
            np.asarray([0.2, 0.15, 0.1]) * target_scale,
            edges,
        )
    return CapacitanceGraph(
        node_features=np.asarray([[1.0], [0.0], [0.0]]),
        edge_index=edges,
        edge_features=np.asarray([[1.0], [1.0], [0.0]]),
        global_features=np.asarray([3.0, 1.0]),
        parameter_values=np.asarray([count, length, width, gap], dtype=float),
        parameter_features=np.zeros((4, 2), dtype=float),
        parameter_names=RETRIEVAL_CONTROL_NAMES,
        net_ids=("ground", "north", "south"),
        capacitance_matrix=matrix,
        metadata={"source_id": source_id},
    )


def _stratified_source() -> tuple[CapacitanceGraph, ...]:
    graphs = []
    for count in (2, 4, 6, 8):
        for index in range(520):
            length = 10.0 + index * 0.001 if index < 512 else 1_000.0 + index
            graphs.append(
                _control_graph(
                    f"count-{count:02d}/row-{index:04d}",
                    count=count,
                    length=length,
                    width=2.0,
                    gap=1.0,
                    target_scale=1.0 + count / 100 + index / 100_000,
                )
            )
    return tuple(graphs)


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_topocap_transfer_study.py"
    spec = importlib.util.spec_from_file_location("topocap_retrieval_runner_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


@pytest.fixture(scope="module")
def frozen_source() -> tuple[CapacitanceGraph, ...]:
    return _stratified_source()


def test_frozen_retrieval_is_deterministic_exact_budget_and_equally_stratified(frozen_source):
    retriever = SupportConditionedSourceRetriever(frozen_source)
    support = [_control_graph("support", count=5, length=10.0, width=2.0, gap=1.0)]

    first = retriever.retrieve(support)
    second = retriever.retrieve(support)

    assert first == second
    assert len(first.source_ids) == RETRIEVAL_SOURCE_BUDGET == 2_048
    selected_counts = [int(frozen_source[index].parameter_values[0]) for index in first.source_indices]
    assert {count: selected_counts.count(count) for count in set(selected_counts)} == {
        2: 512,
        4: 512,
        6: 512,
        8: 512,
    }
    assert all("row-0512" not in source_id for source_id in first.source_ids)
    assert first.source_ids_sha256 == retrieval_source_ids_sha256(first.source_ids)


def test_retrieval_ignores_source_and_support_capacitance_labels(frozen_source):
    support = _control_graph("support", count=5, length=10.2, width=2.1, gap=1.1, target_scale=1.0)
    changed_support_label = support.with_target(
        components_to_maxwell(
            np.asarray([10.0, 11.0, 12.0]),
            np.asarray([2.0, 1.5, 1.0]),
            support.edge_index,
        )
    )
    permutation = np.random.default_rng(9).permutation(len(frozen_source))
    shuffled_source = tuple(
        graph.with_target(frozen_source[int(target_index)].capacitance_matrix)
        for graph, target_index in zip(frozen_source, permutation)
    )

    expected = SupportConditionedSourceRetriever(frozen_source).retrieve([support])
    changed_support = SupportConditionedSourceRetriever(frozen_source).retrieve([changed_support_label])
    changed_source = SupportConditionedSourceRetriever(shuffled_source).retrieve([support])

    assert changed_support == expected
    assert changed_source == expected


def test_source_only_median_iqr_matches_frozen_definition(frozen_source):
    retriever = SupportConditionedSourceRetriever(frozen_source)
    values = np.vstack([graph.parameter_values[1:4] for graph in frozen_source])
    transformed = np.log1p(np.maximum(values, 0.0))
    expected_center = np.median(transformed, axis=0)
    q25, q75 = np.quantile(transformed, [0.25, 0.75], axis=0)
    expected_scale = q75 - q25
    expected_scale[expected_scale < 1.0e-8] = 1.0

    np.testing.assert_allclose(retriever.source_center, expected_center)
    np.testing.assert_allclose(retriever.source_scale, expected_scale)
    assert retriever.protocol()["distance_controls"] == [
        "active_length_um",
        "active_width_um",
        "active_gap_um",
    ]
    assert retriever.protocol()["test_features_or_labels_used"] is False


def test_k_zero_has_no_retrieval_fallback_and_small_sources_are_rejected(frozen_source):
    retriever = SupportConditionedSourceRetriever(frozen_source)

    with pytest.raises(ValueError, match="K=0 has no retrieval fallback"):
        retriever.retrieve([])
    with pytest.raises(ValueError, match="At least 2048"):
        SupportConditionedSourceRetriever(frozen_source[:2_047])


def test_selection_hash_and_public_exports_reject_tampering(frozen_source):
    assert PublicRetriever is SupportConditionedSourceRetriever
    selection = SupportConditionedSourceRetriever(frozen_source).retrieve(
        [_control_graph("support", count=5, length=10.0, width=2.0, gap=1.0)]
    )

    with pytest.raises(ValueError, match="does not match"):
        SupportRetrievalSelection(
            source_indices=selection.source_indices,
            source_ids=selection.source_ids,
            source_ids_sha256="0" * 64,
        )


def test_runner_expected_methods_exclude_k_zero_retrieval_and_include_positive_k():
    runner = _load_runner()

    without_support = runner._expected_method_names(has_support=False, include_v0=False)
    with_support = runner._expected_method_names(has_support=True, include_v0=False)

    assert "source_retrieval_2048_ebra" not in without_support
    assert "shuffled_source_retrieval_2048_ebra" not in without_support
    assert "source_retrieval_2048_ebra" in with_support
    assert "shuffled_source_retrieval_2048_ebra" in with_support


def test_retrieval_model_checkpoint_is_bound_to_state_and_selection(tmp_path):
    runner = _load_runner()
    training = tuple(
        _control_graph(
            f"training-{index}",
            count=2 + index,
            length=10.0 + index,
            width=2.0,
            gap=1.0,
        )
        for index in range(4)
    )
    config = TopoCapConfig(random_feature_dimensions=0, ridge_alpha=30.0, random_seed=17)
    path = tmp_path / "retrieval.npz"
    binding = {
        "method": "source_retrieval_2048_ebra",
        "retrieval_source_ids_sha256": "a" * 64,
    }

    runner._fit_or_load_model(
        path,
        config,
        lambda: training,
        "test retrieval foundation",
        "state-a",
        bindings=binding,
    )
    runner._fit_or_load_model(
        path,
        config,
        lambda: pytest.fail("matching checkpoints must not refit"),
        "test retrieval foundation",
        "state-a",
        bindings=binding,
    )
    metadata = json.loads(path.with_suffix(".npz.metadata.json").read_text())
    assert metadata["state_digest"] == "state-a"
    assert metadata["bindings"] == binding

    with pytest.raises(ValueError, match="retrieval bindings"):
        runner._fit_or_load_model(
            path,
            config,
            lambda: training,
            "test retrieval foundation",
            "state-a",
            bindings={**binding, "retrieval_source_ids_sha256": "b" * 64},
        )
    with pytest.raises(ValueError, match="role and state"):
        runner._fit_or_load_model(
            path,
            config,
            lambda: training,
            "test retrieval foundation",
            "state-b",
            bindings=binding,
        )
