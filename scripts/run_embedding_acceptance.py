#!/usr/bin/env python
"""Run the controlled, representation-first acceptance suite for Tutorial 24."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from squadds.layouts.embedding_benchmark import generate_controlled_planar_suite
from squadds.layouts.geometry_v2 import METRIC_NAMES, encode, scale_conditioned_coupling
from squadds.layouts.geometry_v3 import V3_BLOCK_SLICES, encode_v3
from squadds.layouts.geometry_v4 import encode_v4_graph, encode_v4_pair, layout_view_v4, pair_view_v4

ENVIRONMENT = {
    "relative_permittivity": 11.45,
    "substrate_thickness_um": 350.0,
    "metal_thickness_um": 0.25,
    "vacuum_height_um": 1000.0,
}

REPRESENTATION_INFO = {
    "v2": {"arbitrary_terminals": False, "view": "fixed four-terminal vector"},
    "v2 conditioned": {"arbitrary_terminals": False, "view": "fixed four-terminal vector"},
    "v3 core": {"arbitrary_terminals": False, "view": "selected pair; other conductors omitted from core interactions"},
    "v3 complete": {"arbitrary_terminals": False, "view": "selected pair plus fixed environment"},
    "v4 layout": {"arbitrary_terminals": True, "view": "permutation-invariant pooled graph view"},
    "v4 graph pair": {"arbitrary_terminals": True, "view": "graph-native reciprocal pair query"},
    "v4 hybrid pair": {"arbitrary_terminals": True, "view": "v3 invariant pair core plus all-conductor context"},
}


def layer_roles(terminal_count: int) -> dict[tuple[int, int], str]:
    return {
        (1, 0): "domain",
        (1, 10): "conductor",
        **{(2 + index, 0): "port" for index in range(terminal_count)},
    }


def encode_all(row: pd.Series) -> dict[str, np.ndarray]:
    path = Path(row.gds_path)
    roles = layer_roles(int(row.terminal_count))
    v2 = encode(path, layer_roles=roles)
    v3 = encode_v3(path, environment=ENVIRONMENT, layer_roles=roles)
    graph = encode_v4_graph(path, layer_contract=Path(row.sidecar_path))
    return {
        "v2": np.asarray(v2, dtype=np.float64),
        "v2 conditioned": np.asarray(scale_conditioned_coupling(v2)[0], dtype=np.float64),
        "v3 core": np.asarray(v3[V3_BLOCK_SLICES["fixed_stack_core"]], dtype=np.float64),
        "v3 complete": np.asarray(v3, dtype=np.float64),
        "v4 layout": np.asarray(layout_view_v4(graph), dtype=np.float64),
        "v4 graph pair": np.asarray(pair_view_v4(graph, 0, 1), dtype=np.float64),
        "v4 hybrid pair": np.asarray(
            encode_v4_pair(path, 0, 1, layer_contract=Path(row.sidecar_path), environment=ENVIRONMENT, graph=graph),
            dtype=np.float64,
        ),
    }


def relative_drift(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-12))


def vector_key(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def analyze(frame: pd.DataFrame, vectors: dict[str, list[np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_index = int(frame.index[frame.fixture_name == "pair_baseline"][0])
    nuisance_names = [
        "pair_translated",
        "pair_rotated_90",
        "pair_rotated_37",
        "pair_reflected",
        "pair_ground_crop_small",
        "pair_ground_crop_large",
    ]
    physical_groups = ["uniform_scale", "terminal_gap_um", "facing_length_um", "ground_clearance_um"]
    rows = []
    trajectories = []
    for name, values in vectors.items():
        reference = values[baseline_index]
        repeated = encode_all(frame.iloc[baseline_index])[name]
        nuisance = {
            fixture: relative_drift(reference, values[int(frame.index[frame.fixture_name == fixture][0])])
            for fixture in nuisance_names
        }
        physical_drifts = []
        worst_continuity = 0.0
        for group in physical_groups:
            subset = frame[frame.perturbation_group == group].sort_values("control_value")
            group_vectors = [values[int(index)] for index in subset.index]
            steps = np.asarray(
                [relative_drift(group_vectors[index], group_vectors[index + 1]) for index in range(len(group_vectors) - 1)]
            )
            continuity = float(np.max(steps) / max(float(np.median(steps)), 1e-12)) if len(steps) else 0.0
            worst_continuity = max(worst_continuity, continuity)
            for index, record in subset.iterrows():
                drift = relative_drift(reference, values[int(index)])
                physical_drifts.append(drift)
                trajectories.append(
                    {
                        "representation": name,
                        "group": group,
                        "control_value": float(record.control_value),
                        "distance_from_baseline": drift,
                    }
                )
        gap_subset = frame[frame.perturbation_group == "terminal_gap_um"].sort_values("control_value")
        if name.startswith("v2"):
            gap_coordinate = [values[int(index)][METRIC_NAMES.index("log1p_minimum_pair_gap_um")] for index in gap_subset.index]
        elif name.startswith("v3"):
            gap_coordinate = [values[int(index)][18] for index in gap_subset.index]
        elif name == "v4 graph pair":
            # global 64 + endpoint sum 64 + endpoint difference 64 + edge metric 2
            gap_coordinate = [values[int(index)][194] for index in gap_subset.index]
        elif name == "v4 hybrid pair":
            gap_coordinate = [values[int(index)][18] for index in gap_subset.index]
        elif name == "v4 layout":
            gap_coordinate = [values[int(index)][12] for index in gap_subset.index]
        else:
            gap_coordinate = [relative_drift(reference, values[int(index)]) for index in gap_subset.index]
        rho = float(spearmanr(gap_subset.control_value.astype(float), gap_coordinate).statistic)
        physical_median = float(np.median(physical_drifts))
        nuisance_median = float(np.median(list(nuisance.values())))
        rows.append(
            {
                "representation": name,
                "dimensions": len(reference),
                "deterministic_bitwise": bool(np.array_equal(reference, repeated)),
                "translation_drift": nuisance["pair_translated"],
                "rotation_90_drift": nuisance["pair_rotated_90"],
                "rotation_37_drift": nuisance["pair_rotated_37"],
                "reflection_drift": nuisance["pair_reflected"],
                "outer_crop_max_drift": max(
                    nuisance["pair_ground_crop_small"], nuisance["pair_ground_crop_large"]
                ),
                "median_nuisance_drift": nuisance_median,
                "median_physical_drift": physical_median,
                "sensitivity_to_nuisance": physical_median / max(nuisance_median, 1e-12),
                "worst_continuity_ratio": worst_continuity,
                "gap_spearman": rho,
                "arbitrary_terminal_contract": REPRESENTATION_INFO[name]["arbitrary_terminals"],
                "distinct_control_vectors": len({vector_key(vector) for vector in values}),
                "total_control_vectors": len(values),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(trajectories)


def add_pass_columns(scorecard: pd.DataFrame) -> pd.DataFrame:
    answer = scorecard.copy()
    answer["pass_determinism"] = answer.deterministic_bitwise
    answer["pass_translation"] = answer.translation_drift <= 1e-7
    answer["pass_rotation_90"] = answer.rotation_90_drift <= 5e-4
    answer["pass_rotation_37"] = answer.rotation_37_drift <= 3e-3
    answer["pass_reflection"] = answer.reflection_drift <= 5e-4
    answer["pass_outer_crop"] = answer.outer_crop_max_drift <= 1e-5
    answer["pass_sensitivity_ratio"] = answer.sensitivity_to_nuisance >= 10.0
    answer["pass_continuity"] = answer.worst_continuity_ratio <= 5.0
    answer["pass_gap_ordering"] = answer.gap_spearman.abs() >= 0.95
    required = [column for column in answer if column.startswith("pass_")]
    answer["numerical_gates_passed"] = answer[required].sum(axis=1)
    answer["numerical_gates_total"] = len(required)
    return answer


def run(output: Path) -> pd.DataFrame:
    gds_dir = output / "gds"
    frame = pd.DataFrame(generate_controlled_planar_suite(gds_dir)).reset_index(drop=True)
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATION_INFO}
    for _, row in frame.iterrows():
        encoded = encode_all(row)
        for name, vector in encoded.items():
            vectors[name].append(vector)
    scorecard, trajectories = analyze(frame, vectors)
    scorecard = add_pass_columns(scorecard)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "controlled_manifest.csv", index=False)
    scorecard.to_csv(output / "acceptance_scorecard.csv", index=False)
    trajectories.to_csv(output / "physical_trajectories.csv", index=False)
    np.savez_compressed(
        output / "controlled_embeddings.npz",
        **{name.replace(" ", "_"): np.stack(values) for name, values in vectors.items()},
    )
    report = {
        "thresholds": {
            "translation_drift": 1e-7,
            "rotation_90_and_reflection_drift": 5e-4,
            "rotation_37_drift": 3e-3,
            "outer_crop_drift": 1e-5,
            "sensitivity_to_nuisance": 10.0,
            "continuity_ratio": 5.0,
            "absolute_gap_spearman": 0.95,
        },
        "representation_contracts": REPRESENTATION_INFO,
        "rows": len(frame),
        "scorecard": scorecard.to_dict(orient="records"),
    }
    (output / "acceptance_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tutorials/runtime/tutorial24"),
    )
    arguments = parser.parse_args()
    scorecard = run(arguments.output)
    columns = [
        "representation",
        "numerical_gates_passed",
        "numerical_gates_total",
        "rotation_37_drift",
        "sensitivity_to_nuisance",
        "gap_spearman",
        "arbitrary_terminal_contract",
    ]
    print(scorecard[columns].to_string(index=False))


if __name__ == "__main__":
    main()
