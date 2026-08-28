#!/usr/bin/env python
"""Generate resumable GDS artifacts from supported SQuADDS simulation rows."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class SimulationLayoutSpec:
    """Describe one simulation dataset that can deterministically render GDS."""

    key: str
    filename: str
    prefix: str


SPECS = {
    "cavity-claw": SimulationLayoutSpec(
        key="cavity-claw",
        filename="cavity_claw-RouteMeander-eigenmode.json",
        prefix="cavity_claw",
    ),
    "transmon-cross": SimulationLayoutSpec(
        key="transmon-cross",
        filename="qubit-TransmonCross-cap_matrix.json",
        prefix="transmon_cross",
    ),
}


def _enable_qiskit_metal_pandas_compatibility() -> None:
    """Restore pandas 1.x ``append`` API expected by Qiskit Metal's GDS renderer."""
    import pandas as pd

    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = pd.DataFrame._append  # type: ignore[attr-defined]
    try:
        import geopandas as gpd
    except ImportError:
        return
    if not hasattr(gpd.GeoDataFrame, "append"):
        gpd.GeoDataFrame.append = gpd.GeoDataFrame._append  # type: ignore[attr-defined]


def export_row(kind: str, row: dict[str, Any], destination: Path) -> None:
    """Render one source row into an atomic GDS artifact."""
    _enable_qiskit_metal_pandas_compatibility()
    from qiskit_metal import designs

    from squadds.layouts.qmetal_gds import export_qgeometry_gds, transmon_cross_port_markers

    options = row["design"]["design_options"]
    design = designs.DesignPlanar()
    if kind == "cavity-claw":
        from squadds.simulations.utils_component_factory import create_claw, create_clt_coupler, create_cpw

        # Match the placement and safe meander options used in the eigenmode sweep.
        cpw_length = int(float(str(options["cpw_opts"]["total_length"]).removesuffix("um")))
        create_claw(options["claw_opts"], cpw_length, design)
        coupler = create_clt_coupler(options["cplr_opts"], design)
        create_cpw(options["cpw_opts"], coupler, design)
    elif kind == "transmon-cross":
        from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross

        component = TransmonCross(design, "qubit", options=options)
        markers = transmon_cross_port_markers(component, design)
    else:
        raise ValueError(f"Unsupported simulation layout kind: {kind}")
    export_qgeometry_gds(design, destination, markers=markers if kind == "transmon-cross" else ())


def _worker(kind: str, row: dict[str, Any], destination: str) -> tuple[str, str | None]:
    try:
        export_row(kind, row, Path(destination))
    except Exception as exc:  # pragma: no cover - exercised in full sweep runs.
        return destination, f"{type(exc).__name__}: {exc}"
    return destination, None


def generate(
    kind: str,
    output_dir: Path,
    source_json: Path,
    *,
    workers: int = 1,
    limit: int | None = None,
    overwrite: bool = False,
) -> tuple[int, int, list[dict[str, str]]]:
    """Generate missing GDS files and return generated, skipped, and failures."""
    spec = SPECS[kind]
    rows = json.loads(source_json.read_text())
    if limit is not None:
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped = 0
    for index, row in enumerate(rows):
        destination = output_dir / f"{spec.prefix}_{index:04d}.gds"
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        pending.append((row, destination))

    failures: list[dict[str, str]] = []
    if workers == 1:
        for row, destination in pending:
            _, error = _worker(kind, row, str(destination))
            if error:
                failures.append({"path": str(destination), "error": error})
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, kind, row, str(destination)) for row, destination in pending]
            for future in as_completed(futures):
                destination, error = future.result()
                if error:
                    failures.append({"path": destination, "error": error})
    return len(pending) - len(failures), skipped, sorted(failures, key=lambda item: item["path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(SPECS))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    spec = SPECS[args.kind]
    source_json = args.source_json or Path(hf_hub_download("SQuADDS/SQuADDS_DB", spec.filename, repo_type="dataset"))
    generated, skipped, failures = generate(
        args.kind,
        args.output_dir,
        source_json,
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    if failures:
        failure_path = args.output_dir / "generation-failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(f"Generated {generated} files; skipped {skipped}; {len(failures)} failures in {failure_path}.")
    print(f"Generated {generated} GDS files; skipped {skipped} existing files.")


if __name__ == "__main__":
    main()
