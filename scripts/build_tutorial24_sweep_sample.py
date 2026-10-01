#!/usr/bin/env python
"""Build a deterministic v2/v3/v4 sample from the supplied GDS campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from squadds.layouts.geometry_v2 import encode, read_layer_geometry, scale_conditioned_coupling
from squadds.layouts.geometry_v3 import V3_BLOCK_SLICES, encode_v3
from squadds.layouts.geometry_v4 import encode_v4_graph, layout_view_v4

ENVIRONMENT = {
    "relative_permittivity": 11.45,
    "substrate_thickness_um": 350.0,
    "metal_thickness_um": 0.25,
    "vacuum_height_um": 1000.0,
}

CAMPAIGNS = {
    "exports": "GeneralizedCapNInterdigital / baseline",
    "exp7_exports": "GeneralizedCapNInterdigital / symmetric-7",
    "exp8_exports": "GeneralizedCapNInterdigital / symmetric-8",
    "exp9_exports": "GeneralizedCapNInterdigital / asymmetric",
    "exp10_exports": "GeneralizedCapConcentric",
    "exp11_exports": "GeneralizedCapStar",
}


def deterministic_sample(paths: list[Path], count: int, seed: int, campaign: str) -> list[Path]:
    if len(paths) <= count:
        return sorted(paths)
    digest = int(hashlib.sha256(campaign.encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed + digest).permutation(len(paths))[:count]
    return [sorted(paths)[int(index)] for index in sorted(order)]


def infer_roles(path: Path) -> dict[tuple[int, int], str]:
    geometry = read_layer_geometry(path)
    roles = {}
    for key in geometry:
        if key == (1, 0):
            roles[key] = "domain"
        elif key == (1, 10):
            roles[key] = "conductor"
        elif key == (1, 11):
            roles[key] = "etch"
        elif key[1] == 0 and key[0] >= 2:
            roles[key] = "port"
    return roles


def build(source_root: Path, output: Path, per_campaign: int, seed: int) -> pd.DataFrame:
    selected = []
    for campaign, family in CAMPAIGNS.items():
        json_root = source_root / campaign / "json"
        gds_root = source_root / campaign / "gds"
        paths = deterministic_sample(list(json_root.glob("cap_*.json")), per_campaign, seed, campaign)
        for json_path in paths:
            gds_path = gds_root / f"{json_path.stem}.gds"
            if not gds_path.is_file():
                raise FileNotFoundError(f"No paired GDS for {json_path}")
            selected.append((campaign, family, json_path, gds_path))

    records = []
    started = time.perf_counter()
    for index, (campaign, family, json_path, gds_path) in enumerate(selected, start=1):
        configuration = json.loads(json_path.read_text())
        options = configuration.get("cap_options", {})
        roles = infer_roles(gds_path)
        v2 = encode(gds_path, design_options=options, layer_roles=roles)
        v3 = encode_v3(gds_path, environment=ENVIRONMENT, layer_roles=roles)
        graph = encode_v4_graph(gds_path)
        geometry = read_layer_geometry(gds_path)
        conductor = geometry[(1, 10)]
        records.append(
            {
                "campaign": campaign,
                "family": family,
                "source_id": json_path.stem,
                "gds_path": str(gds_path),
                "json_path": str(json_path),
                "layout_sha256": hashlib.sha256(gds_path.read_bytes()).hexdigest(),
                "terminal_count_v4": graph.metadata["terminal_count"],
                "unmarked_conductor_count_v4": graph.metadata["unmarked_conductor_count"],
                "conductor_area_um2": float(conductor.area),
                "conductor_perimeter_um": float(conductor.length),
                "embedding_v2": np.asarray(v2, dtype=np.float32).tolist(),
                "embedding_v2_conditioned": np.asarray(scale_conditioned_coupling(v2)[0], dtype=np.float32).tolist(),
                "embedding_v3_core": np.asarray(v3[V3_BLOCK_SLICES["fixed_stack_core"]], dtype=np.float32).tolist(),
                "embedding_v3_complete": np.asarray(v3, dtype=np.float32).tolist(),
                "embedding_v4_layout": layout_view_v4(graph).tolist(),
            }
        )
        if index % 25 == 0 or index == len(selected):
            print(f"encoded {index:4d}/{len(selected)} supplied layouts in {time.perf_counter()-started:6.1f} s", flush=True)
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    provenance = {
        "source_root": str(source_root),
        "rows": len(frame),
        "per_campaign": per_campaign,
        "seed": seed,
        "campaigns": frame.campaign.value_counts().sort_index().to_dict(),
        "families": frame.family.value_counts().sort_index().to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/shanto/Downloads/gds_json_exports"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tutorials/runtime/tutorial24/supplied-sweep-sample.parquet"),
    )
    parser.add_argument("--per-campaign", type=int, default=24)
    parser.add_argument("--seed", type=int, default=24)
    arguments = parser.parse_args()
    frame = build(arguments.source_root, arguments.output, arguments.per_campaign, arguments.seed)
    print(
        frame.groupby("family")[["terminal_count_v4", "unmarked_conductor_count_v4", "conductor_area_um2"]]
        .agg(["count", "median", "max"])
        .round(3)
    )


if __name__ == "__main__":
    main()
