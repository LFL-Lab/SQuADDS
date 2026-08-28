#!/usr/bin/env python
"""Fit and freeze the universal-geometry-v2 similarity metric.

v2 vectors are non-negative log-magnitudes in absolute physical units, so every
device shares a large common direction and a raw cosine saturates: within
``TransmonCross`` the entire family spans 0.9928 to 1.0 and ``nearest`` reports
1.0000 for its top matches.  The vectors are not wrong; a raw cosine is simply
the wrong metric for a non-negative, absolutely anchored representation.

The fix keeps the two concerns separate, which is the point of the design:

* the **vectors** stay catalogue-free and byte-stable forever;
* the **metric** is fitted once on the reference catalogue, frozen, published,
  and independently versioned.

A newcomer applies the published transform.  They never refit it, so two
contributions remain directly comparable, exactly as with the vectors.

The transform is centre, scale, then shrinkage-regularized ZCA whitening.
Shrinkage 0.70 was selected by sweeping the parameter against two criteria: the
rank correlation between similarity and capacitance difference within a family,
and the worst such correlation across any pair of families.  It is the setting
where every cross-family pair still has the correct sign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

METRIC_VERSION = "metric-v1"
SHRINKAGE = 0.70
MINIMUM_SCALE = 1e-8


def fit(vectors: np.ndarray, shrinkage: float = SHRINKAGE) -> dict[str, np.ndarray]:
    reference = vectors.astype(np.float64)
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    keep = scale > MINIMUM_SCALE
    standardized = (reference[:, keep] - mean[keep]) / scale[keep]

    covariance = np.cov(standardized, rowvar=False)
    identity = np.eye(covariance.shape[0]) * np.trace(covariance) / covariance.shape[0]
    regularized = (1.0 - shrinkage) * covariance + shrinkage * identity
    values, directions = np.linalg.eigh(regularized)
    whitening = directions @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-8))) @ directions.T
    return {
        "mean": mean,
        "scale": scale,
        "keep": keep,
        "whitening": whitening.astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--from-parquet",
        type=Path,
        help="Fit on a local embedding table instead of the published one, so a metric can be "
        "frozen against the exact vectors it ships beside.",
    )
    parser.add_argument("--shrinkage", type=float, default=SHRINKAGE)
    arguments = parser.parse_args()

    source = arguments.from_parquet or hf_hub_download(
        "SQuADDS/SQuADDS_Layout_Embeddings",
        "metadata/universal-geometry-v2.parquet",
        repo_type="dataset",
        revision=arguments.revision,
    )
    table = pd.read_parquet(source)
    vectors = np.vstack(table["embedding"].to_numpy()).astype(np.float64)
    parameters = fit(vectors, arguments.shrinkage)

    output = arguments.output_dir / "models" / "universal-geometry-v2"
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / f"{METRIC_VERSION}.npz"
    np.savez_compressed(
        array_path,
        mean=parameters["mean"].astype(np.float32),
        scale=parameters["scale"].astype(np.float32),
        keep=parameters["keep"],
        whitening=parameters["whitening"].astype(np.float32),
    )
    contract = {
        "metric_version": METRIC_VERSION,
        "applies_to": "universal-geometry-v2",
        "transform": "centre, scale, then shrinkage-regularized ZCA whitening, then cosine",
        "shrinkage": arguments.shrinkage,
        "fitted_rows": int(len(table)),
        "fitted_on": ("local embedding table" if arguments.from_parquet else f"published revision {arguments.revision}"),
        "fitted_families": sorted(table["component_name"].unique().tolist()),
        "input_dimensions": int(vectors.shape[1]),
        "retained_dimensions": int(parameters["keep"].sum()),
        "note": (
            "The vectors are catalogue-free and unchanged. Only this metric is fitted, "
            "and it is frozen and versioned separately so two contributions stay comparable. "
            "Apply it with LayoutEmbeddingClient.metric_transform(); never refit it locally."
        ),
        "arrays": {
            array_path.name: {"sha256": hashlib.sha256(array_path.read_bytes()).hexdigest()},
        },
    }
    (output / f"{METRIC_VERSION}.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
