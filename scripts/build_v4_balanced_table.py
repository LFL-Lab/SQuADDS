#!/usr/bin/env python
"""Encode the Tutorial 23 balanced cohort with planar-terminal-graph-v4."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from squadds.layouts.geometry_v4 import encode_v4_graph, encode_v4_pair, layout_view_v4

ENVIRONMENT = {
    "relative_permittivity": 11.45,
    "substrate_thickness_um": 500.0,
    "metal_thickness_um": 0.2,
    "penetration_depth_um": 0.05,
}


def build(source: Path, output: Path) -> pd.DataFrame:
    source_frame = pd.read_parquet(source)
    records = []
    started = time.perf_counter()
    for index, row in enumerate(source_frame.itertuples(), start=1):
        path = Path(row.absolute_gds_path)
        graph = encode_v4_graph(path)
        pair = encode_v4_pair(path, 0, 1, graph=graph, environment=ENVIRONMENT)
        records.append(
            {
                "design_id": row.design_id,
                "component_name": row.component_name,
                "source_id": row.source_id,
                "mutual_fF": row.mutual_fF,
                "ground_sum_fF": row.ground_sum_fF,
                "terminal_count_v4": graph.metadata["terminal_count"],
                "unmarked_conductor_count_v4": graph.metadata["unmarked_conductor_count"],
                "proxy_mutual_fF_v4": abs(float(graph.proxy_matrix_fF[0, 1])),
                "embedding_v4_pair": pair.tolist(),
                "embedding_v4_layout": layout_view_v4(graph).tolist(),
                "absolute_gds_path": str(path),
            }
        )
        if index % 250 == 0 or index == len(source_frame):
            print(f"encoded {index:4d}/{len(source_frame)} v4 graphs in {time.perf_counter()-started:7.1f} s", flush=True)
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(source),
                "rows": len(frame),
                "families": frame.component_name.value_counts().to_dict(),
                "elapsed_seconds": time.perf_counter() - started,
                "environment": ENVIRONMENT,
            },
            indent=2,
        )
        + "\n"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("tutorials/runtime/tutorial23/capacitance-operator-v3-balanced.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tutorials/runtime/tutorial24/planar-terminal-graph-v4-balanced.parquet"),
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.is_file() and not arguments.force:
        print(f"already exists: {arguments.output}")
        return
    frame = build(arguments.source, arguments.output)
    print(frame.groupby("component_name")[["mutual_fF", "proxy_mutual_fF_v4"]].agg(["count", "median"]))


if __name__ == "__main__":
    main()
