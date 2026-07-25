#!/usr/bin/env python
"""Build deterministic geometry-vector-v1 embeddings from layout features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from squadds.layouts.embeddings import write_geometry_vector_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_features", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    count, dimensions = write_geometry_vector_dataset(pd.read_parquet(args.geometry_features), args.output_dir)
    print(f"Wrote {count} geometry-vector-v1 embeddings with {dimensions} dimensions.")


if __name__ == "__main__":
    main()
