#!/usr/bin/env python
"""Build the balanced three-family CapOp-v3 table used by Tutorial 23."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from squadds.layouts import canonical_design_id, encode_v3

SEED = 23
BALANCED_PER_FAMILY = 894
ENVIRONMENT = {
    "relative_permittivity": 11.45,
    "substrate_thickness_um": 350.0,
    "metal_thickness_um": 0.25,
    "vacuum_height_um": 1000.0,
}
FAMILIES = {
    "GeneralizedCapNInterdigital": {
        "database": "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
        "mutual": "north_to_south",
        "grounds": ("north_to_ground", "south_to_ground"),
    },
    "CapNInterdigitalTee": {
        "database": "coupler-CapNInterdigitalTee-cap_matrix.json",
        "mutual": "top_to_bottom",
        "grounds": ("top_to_ground", "bottom_to_ground"),
    },
    "TransmonCross": {
        "database": "qubit-TransmonCross-cap_matrix.json",
        "mutual": "cross_to_claw",
        "grounds": ("cross_to_ground", "claw_to_ground"),
    },
}


def target_table() -> pd.DataFrame:
    records = []
    for component, specification in FAMILIES.items():
        database = Path(
            hf_hub_download("SQuADDS/SQuADDS_DB", specification["database"], repo_type="dataset")
        )
        for row in json.loads(database.read_text()):
            options = row["design"]["design_options"]
            results = row["sim_results"]
            mutual = abs(float(results[specification["mutual"]]))
            ground = sum(abs(float(results[name])) for name in specification["grounds"])
            records.append(
                {
                    "design_id": canonical_design_id(component, options),
                    "component_name": component,
                    "mutual_fF": mutual,
                    "ground_sum_fF": ground,
                    "log_mutual": float(np.log1p(mutual)),
                }
            )
    return pd.DataFrame(records).drop_duplicates("design_id")


def layout_index(port_complete_root: Path) -> pd.DataFrame:
    published_manifest = Path(
        hf_hub_download("SQuADDS/SQuADDS_Layouts", "metadata/manifest.parquet", repo_type="dataset")
    )
    published_root = published_manifest.parent.parent
    published = pd.read_parquet(published_manifest)
    published = published[published.component_name == "GeneralizedCapNInterdigital"].copy()
    published["absolute_gds_path"] = published.gds_path.map(lambda path: published_root / path)
    published = published[published.absolute_gds_path.map(Path.is_file)]

    local_manifest = port_complete_root / "metadata/manifest.parquet"
    local = pd.read_parquet(local_manifest).copy()
    local["absolute_gds_path"] = local.gds_path.map(lambda path: port_complete_root / path)
    combined = pd.concat([published, local], ignore_index=True)
    return combined.drop_duplicates("design_id", keep="last")


def balanced_cohort(targets: pd.DataFrame, layouts: pd.DataFrame) -> pd.DataFrame:
    available = targets.merge(
        layouts[["design_id", "source_id", "absolute_gds_path"]],
        on="design_id",
        how="inner",
    )
    picked = []
    for component in FAMILIES:
        family = available[available.component_name == component].sort_values("design_id")
        if len(family) < BALANCED_PER_FAMILY:
            raise RuntimeError(f"{component} has only {len(family)} usable layouts.")
        order = np.random.default_rng(SEED).permutation(len(family))[:BALANCED_PER_FAMILY]
        picked.append(family.iloc[order])
    result = pd.concat(picked, ignore_index=True)
    return result.sort_values(["component_name", "design_id"]).reset_index(drop=True)


def build(output: Path, port_complete_root: Path) -> pd.DataFrame:
    cohort = balanced_cohort(target_table(), layout_index(port_complete_root))
    records = []
    started = time.perf_counter()
    for index, row in enumerate(cohort.itertuples(), start=1):
        vector, metadata = encode_v3(
            row.absolute_gds_path,
            environment=ENVIRONMENT,
            return_metadata=True,
        )
        proxy = metadata["electrostatic_proxy"]["matrix_scaled_fF"]
        records.append(
            {
                "design_id": row.design_id,
                "component_name": row.component_name,
                "source_id": row.source_id,
                "mutual_fF": row.mutual_fF,
                "ground_sum_fF": row.ground_sum_fF,
                "log_mutual": row.log_mutual,
                "characteristic_scale_um": metadata["characteristic_scale_um"],
                "minimum_gap_um": float(np.expm1(vector[18])),
                "proxy_mutual_fF": abs(float(proxy[0, 1])),
                "proxy_ground_sum_fF": abs(float(proxy[0, 2])) + abs(float(proxy[1, 2])),
                "embedding": vector.tolist(),
                "absolute_gds_path": str(row.absolute_gds_path),
            }
        )
        if index % 250 == 0 or index == len(cohort):
            elapsed = time.perf_counter() - started
            print(f"encoded {index:4d}/{len(cohort)} layouts in {elapsed:7.1f} s", flush=True)
    table = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    provenance = {
        "rows": len(table),
        "families": table.component_name.value_counts().to_dict(),
        "environment": ENVIRONMENT,
        "seed": SEED,
        "balanced_per_family": BALANCED_PER_FAMILY,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tutorials/runtime/tutorial23/capacitance-operator-v3-balanced.parquet"),
    )
    parser.add_argument(
        "--port-complete-root",
        type=Path,
        default=Path("../SQuADDS-port-gds-artifacts/layout-dataset"),
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.is_file() and not arguments.force:
        print(f"already exists: {arguments.output}")
        return
    table = build(arguments.output, arguments.port_complete_root)
    print(table.groupby("component_name")[["mutual_fF", "proxy_mutual_fF"]].agg(["count", "median"]))


if __name__ == "__main__":
    main()
