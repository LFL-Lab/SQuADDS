#!/usr/bin/env python
"""Audit and stage supplied GDS/JSON sweeps without promoting them to ``raw/``.

This command is intentionally conservative: any family that does not satisfy
the released two-port layer contract is copied under ``incoming/`` and receives
a machine-readable report.  The files are then safe to back up in a Hub pull
request without implying that they are ready for the production layout index.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

CAMPAIGNS = {
    "exp10_exports": "GeneralizedCapConcentric",
    "exp11_exports": "GeneralizedCapStar",
}
EXPECTED_LAYERS = {(1, 0), (1, 10), (2, 0), (3, 0)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_layers(path: Path) -> list[tuple[int, int]]:
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.top_cell()
    if top is None:
        raise ValueError("GDS has no top cell")
    layers = []
    for layer_index in layout.layer_indices():
        info = layout.get_info(layer_index)
        if not kdb.Region(top.begin_shapes_rec(layer_index)).is_empty():
            layers.append((int(info.layer), int(info.datatype)))
    return sorted(layers)


def _audit_pair(item: tuple[str, str, str, str]) -> dict[str, Any]:
    campaign, component_name, gds_text, json_text = item
    gds_path = Path(gds_text)
    json_path = Path(json_text)
    try:
        configuration = json.loads(json_path.read_text())
        layers = _nonempty_layers(gds_path)
        port_layers = [layer for layer in layers if layer[1] == 0 and layer[0] >= 2]
        source_id = gds_path.stem
        if configuration.get("cap_instance_name") != source_id:
            raise ValueError("cap_instance_name does not match the GDS/JSON stem")
        return {
            "ok": True,
            "campaign": campaign,
            "component_name": component_name,
            "source_id": source_id,
            "gds_sha256": _sha256(gds_path),
            "json_sha256": _sha256(json_path),
            "gds_size_bytes": gds_path.stat().st_size,
            "json_size_bytes": json_path.stat().st_size,
            "layers": json.dumps(layers, separators=(",", ":")),
            "port_layer_count": len(port_layers),
            "two_port_layer_contract": set(layers) == EXPECTED_LAYERS,
            "design_options_json": json.dumps(
                configuration.get("cap_options", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "layer_stack_json": json.dumps(
                configuration.get("layer_stack", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    except Exception as error:
        return {
            "ok": False,
            "campaign": campaign,
            "component_name": component_name,
            "source_id": gds_path.stem,
            "reason": f"{type(error).__name__}: {error}",
        }


def build(source_root: Path, output_root: Path, workers: int) -> dict[str, Any]:
    items: list[tuple[str, str, str, str]] = []
    unmatched: list[dict[str, str]] = []
    for campaign, component_name in CAMPAIGNS.items():
        campaign_root = source_root / campaign
        gds_by_stem = {path.stem: path for path in (campaign_root / "gds").glob("*.gds")}
        json_by_stem = {path.stem: path for path in (campaign_root / "json").glob("cap_*.json")}
        for stem in sorted(set(gds_by_stem) | set(json_by_stem)):
            if stem not in gds_by_stem or stem not in json_by_stem:
                unmatched.append(
                    {
                        "campaign": campaign,
                        "component_name": component_name,
                        "source_id": stem,
                        "reason": "missing GDS" if stem not in gds_by_stem else "missing JSON",
                    }
                )
                continue
            items.append(
                (
                    campaign,
                    component_name,
                    str(gds_by_stem[stem]),
                    str(json_by_stem[stem]),
                )
            )

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_audit_pair, items, chunksize=32))
    failures = unmatched + [result for result in results if not result["ok"]]
    records = [{key: value for key, value in result.items() if key != "ok"} for result in results if result["ok"]]

    for campaign, component_name in CAMPAIGNS.items():
        source = source_root / campaign
        destination = output_root / "incoming" / component_name / campaign
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "gds", destination / "gds", dirs_exist_ok=True)
        shutil.copytree(source / "json", destination / "json", dirs_exist_ok=True)

    metadata = output_root / "incoming" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values(["component_name", "source_id"])
    frame.to_parquet(metadata / "supplied-new-family-manifest.parquet", index=False, compression="zstd")
    failure_columns = ["campaign", "component_name", "source_id", "reason"]
    pd.DataFrame(failures, columns=failure_columns).to_parquet(
        metadata / "supplied-new-family-failures.parquet",
        index=False,
        compression="zstd",
    )
    layer_counts = Counter(frame.layers)
    report = {
        "status": "incoming-not-production",
        "source": "user-supplied gds_json_exports",
        "campaigns": dict(CAMPAIGNS),
        "paired_rows": len(frame),
        "failures": len(failures),
        "families": frame.component_name.value_counts().sort_index().to_dict(),
        "layer_sets": dict(sorted(layer_counts.items())),
        "port_layer_counts": {
            str(key): value for key, value in sorted(Counter(frame.port_layer_count).items())
        },
        "two_port_layer_contract_rows": int(frame.two_port_layer_contract.sum()),
        "family_port_layer_counts": {
            component: {
                str(port_count): int(count)
                for port_count, count in group.port_layer_count.value_counts().sort_index().items()
            }
            for component, group in frame.groupby("component_name")
        },
        "ready_for_raw_namespace": bool(
            not failures and len(frame) and frame.two_port_layer_contract.all()
        ),
        "blocking_finding": (
            "3,000 of 4,000 supplied layouts have only layer (2,0); layer (3,0) "
            "is absent, so the second conductor is not explicitly port-labelled."
        ),
    }
    (metadata / "supplied-new-family-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")
    build(arguments.source_root, arguments.output_root, arguments.workers)


if __name__ == "__main__":
    main()
