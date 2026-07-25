#!/usr/bin/env python
"""Build versioned numerical geometry features from a layout manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from squadds.layouts import build_geometry_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    features = build_geometry_features(pd.read_parquet(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"Wrote {len(features)} geometry feature records to {args.output}")


if __name__ == "__main__":
    main()
