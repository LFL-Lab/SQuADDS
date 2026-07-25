#!/usr/bin/env python
"""Build the GeneralizedCapNInterdigital Hugging Face dataset artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from squadds.database.generalized_ncap_dataset import convert_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    count = convert_dataset(args.source_root, args.output_path)
    print(f"Wrote {count} rows to {args.output_path}")


if __name__ == "__main__":
    main()
