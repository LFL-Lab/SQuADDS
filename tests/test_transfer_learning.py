"""Tests for transfer learning over static v0 embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from squadds.layouts import (
    EMBEDDING_DIMENSIONS,
    TransferRidgeRegressor,
    V0TransferLearningStudy,
    compress_v0_embeddings,
    regression_scores,
    required_target_samples,
    target_to_source_similarity,
)


def _synthetic_v0_embeddings(values: np.ndarray) -> np.ndarray:
    embeddings = np.zeros((len(values), EMBEDDING_DIMENSIONS), dtype=np.float32)
    embeddings[:, 0] = 1.0
    embeddings[:, 1] = values
    embeddings[:, 11:] = values[:, None]
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def test_compress_v0_embeddings_retains_blocks_and_pools_shape():
    embeddings = np.zeros((2, EMBEDDING_DIMENSIONS), dtype=np.float32)
    embeddings[:, :11] = np.arange(11)
    embeddings[0, 11:] = 1.0

    compressed = compress_v0_embeddings(embeddings, pooled_shape_size=12)

    assert compressed.shape == (2, 155)
    assert compressed[0, :11] == pytest.approx(np.arange(11))
    assert compressed[0, 11:] == pytest.approx(np.ones(144))
    assert compressed[1, 11:] == pytest.approx(np.zeros(144))


def test_transfer_ridge_uses_a_source_prior():
    source_x = np.linspace(-1, 1, 40)[:, None]
    source_y = np.column_stack([2.0 * source_x[:, 0] + 1.0, -source_x[:, 0] + 0.5])
    target_x = np.asarray([[-0.8], [0.8]])
    target_y = np.column_stack([2.0 * target_x[:, 0] + 1.1, -target_x[:, 0] + 0.6])
    test_x = np.linspace(-1, 1, 20)[:, None]
    test_y = np.column_stack([2.0 * test_x[:, 0] + 1.1, -test_x[:, 0] + 0.6])

    source_model = TransferRidgeRegressor(alpha=1.0).fit(source_x, source_y)
    target_only = TransferRidgeRegressor(alpha=1.0).fit(target_x, target_y)
    transfer = TransferRidgeRegressor(alpha=1.0).fit(target_x, target_y, prior=source_model)

    target_only_mae = regression_scores(test_y, target_only.predict(test_x)).iloc[-1]["mae"]
    transfer_mae = regression_scores(test_y, transfer.predict(test_x)).iloc[-1]["mae"]

    assert transfer_mae < target_only_mae


def test_transfer_study_runs_reproducible_similarity_curves():
    source_values = np.linspace(0.2, 1.0, 60)
    target_values = np.linspace(0.1, 0.9, 48)
    source_embeddings = _synthetic_v0_embeddings(source_values)
    target_embeddings = _synthetic_v0_embeddings(target_values)
    source_targets = np.column_stack([3.0 * source_values, source_values**2])
    target_targets = np.column_stack([3.0 * target_values + 0.1, target_values**2 + 0.05])

    study = V0TransferLearningStudy(
        source_embeddings,
        source_targets,
        target_embeddings,
        target_targets,
        target_names=["linear", "quadratic"],
        pooled_shape_size=12,
        alpha=10.0,
    )
    curves = study.learning_curve([4, 8], repeats=3, test_fraction=0.25, random_seed=4)
    similarity_curves = study.similarity_learning_curves(
        [4, 8],
        bands=2,
        repeats=3,
        test_fraction=0.25,
        random_seed=4,
    )
    pool_indices, test_indices = study.target_split(test_fraction=0.25, random_seed=4)
    models = study.fit_models(pool_indices[:4])

    assert set(curves["method"]) == {"zero-shot", "target-only", "transfer"}
    assert set(curves["target"]) == {"linear", "quadratic", "macro"}
    assert set(similarity_curves["similarity_band"]) == {"Q1", "Q2"}
    assert study.domain_similarity()["minimum"] <= study.domain_similarity()["maximum"]
    assert target_to_source_similarity(source_embeddings, target_embeddings).shape == (48,)
    assert len(test_indices) == 12
    assert set(models) == {"zero-shot", "target-only", "transfer"}


def test_required_target_samples_uses_the_lower_confidence_bound():
    curves = pd.DataFrame(
        [
            {
                "sample_size": sample_size,
                "repeat": repeat,
                "method": "transfer",
                "target": "macro",
                "r2": score,
            }
            for sample_size, scores in ((8, [0.78, 0.80, 0.82]), (16, [0.91, 0.92, 0.93]))
            for repeat, score in enumerate(scores)
        ]
    )

    requirements = required_target_samples(
        curves,
        [0.75, 0.90, 0.95],
        confidence=0.8,
    )

    assert requirements["required_samples"].tolist()[:2] == [8, 16]
    assert pd.isna(requirements.iloc[2]["required_samples"])
