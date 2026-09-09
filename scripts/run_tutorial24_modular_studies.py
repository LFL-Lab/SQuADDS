#!/usr/bin/env python
"""Generate the planar module-composition and vector-arithmetic evidence for Tutorial 24."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import box

from squadds.layouts.embedding_benchmark import rectangular_pair, write_standard_planar_gds
from squadds.layouts.geometry_v4 import encode_v4_graph, layout_view_v4

LAYOUT_BLOCKS = {
    "global": slice(0, 64),
    "node_mean": slice(64, 128),
    "node_std": slice(128, 192),
    "node_max": slice(192, 256),
    "edge_mean": slice(256, 352),
    "edge_std": slice(352, 448),
    "edge_max": slice(448, 544),
}


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0 else 0.0


def _encode(path: Path):
    graph = encode_v4_graph(path, layer_contract=path.with_suffix(".layers.json"))
    return graph, layout_view_v4(graph).astype(np.float64)


def run(output: Path) -> dict[str, object]:
    gds_dir = output / "gds"
    gds_dir.mkdir(parents=True, exist_ok=True)

    left = box(-65.0, -18.0, -35.0, 18.0)
    shared = box(-15.0, -22.0, 15.0, 22.0)
    right = box(35.0, -18.0, 65.0, 18.0)
    shifted_right = affinity.translate(right, xoff=10.0)
    fixtures = {
        "module_a": [left, shared],
        "module_b": [shared, right],
        "module_b_right_shifted": [shared, shifted_right],
        "joined_three_terminal": [left, shared, right],
        "joined_right_shifted": [left, shared, shifted_right],
    }

    base_pair = rectangular_pair(gap_um=6.0, facing_length_um=32.0, width_um=36.0)
    wider_pair = rectangular_pair(gap_um=12.0, facing_length_um=32.0, width_um=36.0)
    fixtures.update(
        {
            "delta_base": base_pair,
            "delta_wider_gap": wider_pair,
            "delta_base_rotated": [affinity.rotate(shape, 37.0, origin=(0.0, 0.0)) for shape in base_pair],
            "delta_wider_gap_rotated": [
                affinity.rotate(shape, 37.0, origin=(0.0, 0.0)) for shape in wider_pair
            ],
        }
    )

    manifest = []
    graphs = {}
    vectors = {}
    for name, terminals in fixtures.items():
        path = gds_dir / f"{name}.gds"
        write_standard_planar_gds(
            path,
            terminals,
            clearance_um=5.0,
            domain_padding_um=90.0,
            metadata={"study": "tutorial24_modular_arithmetic", "fixture": name},
        )
        graph, vector = _encode(path)
        graphs[name] = graph
        vectors[name] = vector
        manifest.append(
            {
                "fixture": name,
                "gds_path": str(path),
                "layer_contract_path": str(path.with_suffix(".layers.json")),
                "terminal_count": int(graph.metadata["terminal_count"]),
                "unmarked_conductor_count": int(graph.metadata["unmarked_conductor_count"]),
                "vector_dimensions": len(vector),
            }
        )

    naive_sum = vectors["module_a"] + vectors["module_b"]
    joined = vectors["joined_three_terminal"]
    module_delta = vectors["module_b_right_shifted"] - vectors["module_b"]
    joined_delta = vectors["joined_right_shifted"] - joined
    original_delta = vectors["delta_wider_gap"] - vectors["delta_base"]
    rotated_delta = vectors["delta_wider_gap_rotated"] - vectors["delta_base_rotated"]

    block_rows = []
    for name, block in LAYOUT_BLOCKS.items():
        reference = joined[block]
        residual = naive_sum[block] - reference
        block_rows.append(
            {
                "block": name,
                "naive_addition_relative_error": float(
                    np.linalg.norm(residual) / max(np.linalg.norm(reference), 1e-12)
                ),
                "module_delta_norm": float(np.linalg.norm(module_delta[block])),
                "joined_delta_norm": float(np.linalg.norm(joined_delta[block])),
                "delta_cosine": _cosine(module_delta[block], joined_delta[block]),
            }
        )
    block_frame = pd.DataFrame(block_rows)
    block_frame.to_csv(output / "block_arithmetic.csv", index=False)
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)
    np.savez_compressed(output / "vectors.npz", **vectors)
    np.savez_compressed(
        output / "proxy_matrices.npz",
        **{name: graph.proxy_matrix_fF for name, graph in graphs.items()},
    )

    summary = {
        "schema_version": "tutorial24-modular-studies-v1",
        "fixtures": len(fixtures),
        "module_terminal_counts": {
            "module_a": int(graphs["module_a"].metadata["terminal_count"]),
            "module_b": int(graphs["module_b"].metadata["terminal_count"]),
            "joined": int(graphs["joined_three_terminal"].metadata["terminal_count"]),
        },
        "joined_proxy_matrix_shape": list(graphs["joined_three_terminal"].proxy_matrix_fF.shape),
        "naive_vector_addition_relative_error": float(
            np.linalg.norm(naive_sum - joined) / max(np.linalg.norm(joined), 1e-12)
        ),
        "naive_vector_addition_cosine": _cosine(naive_sum, joined),
        "module_vs_joined_delta_cosine": _cosine(module_delta, joined_delta),
        "module_vs_joined_delta_norm_ratio": float(
            np.linalg.norm(joined_delta) / max(np.linalg.norm(module_delta), 1e-12)
        ),
        "pose_matched_delta_relative_error": float(
            np.linalg.norm(original_delta - rotated_delta) / max(np.linalg.norm(original_delta), 1e-12)
        ),
        "pose_matched_delta_cosine": _cosine(original_delta, rotated_delta),
        "base_rotation_relative_drift": float(
            np.linalg.norm(vectors["delta_base_rotated"] - vectors["delta_base"])
            / max(np.linalg.norm(vectors["delta_base"]), 1e-12)
        ),
        "wider_gap_rotation_relative_drift": float(
            np.linalg.norm(vectors["delta_wider_gap_rotated"] - vectors["delta_wider_gap"])
            / max(np.linalg.norm(vectors["delta_wider_gap"]), 1e-12)
        ),
        "interpretation": {
            "raw_addition": "diagnostic only; it double-counts shared conductors and global summaries",
            "raw_subtraction": (
                "diagnostic only; subtracting near-equal invariant vectors amplifies their small numerical drift"
            ),
            "valid_composition": "merge port/net geometry first, then re-encode the resulting conductor graph",
            "proxy_matrix": "deterministic low-fidelity 2-D feature, not high-fidelity simulation truth",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tutorials/runtime/tutorial24/modularity"),
    )
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
