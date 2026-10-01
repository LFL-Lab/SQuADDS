#!/usr/bin/env python
"""Build universal-geometry-v1 embeddings from GDS layouts and design options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from squadds.layouts import canonical_design_id, write_universal_embedding_dataset


def _mapping(value: str) -> tuple[str, Path]:
    key, separator, raw_path = value.partition("=")
    if not separator or not key or not raw_path:
        raise argparse.ArgumentTypeError("Mappings must use key=/absolute/path format.")
    return key, Path(raw_path)


def _design_options(design_sources: list[tuple[str, Path]]) -> dict[str, dict]:
    options_by_id = {}
    for component_name, source_path in design_sources:
        for row in json.loads(source_path.read_text()):
            options = row["design"]["design_options"]
            options_by_id[canonical_design_id(component_name, options)] = options
    return options_by_id


def _artifact_resolver(gds_sources: list[tuple[str, Path]]):
    sources = sorted(gds_sources, key=lambda item: len(item[0]), reverse=True)

    def resolve(record: dict) -> Path:
        relative_path = Path(record["gds_path"])
        for prefix, source_dir in sources:
            if record["gds_path"].startswith(prefix + "/"):
                for candidate in (
                    source_dir / relative_path,
                    source_dir / relative_path.relative_to(prefix),
                    source_dir / relative_path.name,
                ):
                    if candidate.is_file():
                        return candidate
                raise FileNotFoundError(f"Missing GDS artifact for {record['gds_path']!r}.")
        raise LookupError(f"No --gds-source mapping covers {record['gds_path']!r}.")

    return resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--component-name", default="GeneralizedCapNInterdigital")
    parser.add_argument("--design-json", action="append", required=True, type=_mapping)
    parser.add_argument("--gds-source", action="append", required=True, type=_mapping)
    args = parser.parse_args()

    count, dimensions = write_universal_embedding_dataset(
        pd.read_parquet(args.manifest),
        _design_options(args.design_json),
        _artifact_resolver(args.gds_source),
        args.output_dir,
        component_name=args.component_name,
    )
    print(f"Wrote {count} universal-geometry-v1 embeddings with {dimensions} dimensions.")


if __name__ == "__main__":
    main()
