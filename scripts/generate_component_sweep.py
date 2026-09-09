#!/usr/bin/env python
"""Generate a simulation-ready layout sweep for a new component family.

Produces GDS in the published SQuADDS convention plus the design-option JSON a
simulation pipeline consumes, so the round trip is: run this, simulate, hand back
results keyed by ``design_index``.

    uv run --extra gds python scripts/generate_component_sweep.py \\
        TransmonPocket /path/to/output --limit 200

Families are declared in ``squadds.layouts.component_sweeps.SWEEPS``.  Adding one
is a ``SweepSpec`` entry; no per-family code is required, because terminal
markers come from the component's own declared pins and the ground clearance
from its own gap options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from squadds.layouts.component_sweeps import SWEEPS, write_design_point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=sorted(SWEEPS), nargs="?")
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--limit", type=int, help="stop after this many design points")
    parser.add_argument("--list", action="store_true", help="show the declared families and exit")
    arguments = parser.parse_args()

    if arguments.list or not arguments.family:
        print(f"{'family':20s} {'points':>7s}  {'ports':>5s}  note")
        for name, spec in sorted(SWEEPS.items()):
            print(f"{name:20s} {spec.size:7d}  {len(spec.port_pins):5d}  {spec.note}")
        return
    if arguments.output_dir is None:
        parser.error("output_dir is required unless --list is given")

    spec = SWEEPS[arguments.family]
    root = arguments.output_dir
    gds_root = root / "raw" / spec.component_name
    gds_root.mkdir(parents=True, exist_ok=True)

    rows, options_payload, failures = [], [], []
    for index, point in enumerate(spec.points()):
        if arguments.limit is not None and index >= arguments.limit:
            break
        relative = f"raw/{spec.component_name}/{spec.component_name.lower()}_{index:05d}.gds"
        try:
            row = write_design_point(spec, point, root / relative, name=f"sweep_{index:05d}")
        except Exception as error:  # noqa: BLE001 - one bad point must not stop the sweep
            failures.append({"design_index": index, "error": f"{type(error).__name__}: {error}", **point})
            continue
        row["gds_path"] = relative
        row["design_index"] = index
        rows.append(row)
        options_payload.append({"design_index": index, "design": {"design_options": row["design_options"]}})
        if (index + 1) % 100 == 0:
            print(f"  generated {index + 1} ...", flush=True)

    if not rows:
        print("No design points generated.", file=sys.stderr)
        raise SystemExit(1)

    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(metadata / "manifest.parquet", index=False)
    (root / "design-options.json").write_text(json.dumps(options_payload, indent=2, sort_keys=True) + "\n")
    if failures:
        pd.DataFrame(failures).to_parquet(metadata / "failures.parquet", index=False)

    counts = frame.terminal_count.value_counts().sort_index()
    print(
        json.dumps(
            {
                "family": spec.component_name,
                "generated": len(frame),
                "failed": len(failures),
                "grid_size": spec.size,
                "conductor_islands": {int(k): int(v) for k, v in counts.items()},
                "ground_clearance_um": sorted(frame.ground_clearance_um.unique().tolist()),
                "output": str(root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
