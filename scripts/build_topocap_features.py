#!/usr/bin/env python
"""Build or verify the offline, topology-aware TopoCap graph catalogue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from squadds.ml.topocap.datasets import (
    EXPECTED_CAPN_RECORDS,
    EXPECTED_GENERALIZED_RECORDS,
    CatalogueEntry,
    CatalogueError,
    pair_capn_records,
    pair_generalized_records,
    verify_graph_cache,
    write_graph_cache,
)

HOME = Path.home()
DEFAULT_DATA_ROOT = HOME / "Downloads" / "squadds_data"
DEFAULT_GENERALIZED_JSON = (
    HOME
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--SQuADDS--SQuADDS_DB"
    / "snapshots"
    / "0e25705f54c343fb96571ff15b6fd8375ca899aa"
    / "coupler-GeneralizedCapNInterdigital-cap_matrix.json"
)
DEFAULT_CAPN_JSON = (
    HOME
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--SQuADDS--SQuADDS_DB"
    / "snapshots"
    / "150ebf9fe01a291b5f36a77eb51ff09e5efc2623"
    / "coupler-CapNInterdigitalTee-cap_matrix.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pair local SQuADDS simulation rows to immutable GDS files and stream target-blind "
            "TopoCap geometry graphs to a deterministic JSONL cache. No network access is used."
        )
    )
    parser.add_argument("output_dir", type=Path, help="Directory for graphs.jsonl, sidecars, and cache manifest.")
    parser.add_argument(
        "--family",
        choices=("all", "generalized", "capn"),
        default="all",
        help="Build both families (default) or only one release family.",
    )
    parser.add_argument("--generalized-json", type=Path, default=DEFAULT_GENERALIZED_JSON)
    parser.add_argument("--capn-json", type=Path, default=DEFAULT_CAPN_JSON)
    parser.add_argument("--exp6-gds-dir", type=Path, default=DEFAULT_DATA_ROOT / "gds2")
    parser.add_argument("--exp7-gds-dir", type=Path, default=DEFAULT_DATA_ROOT / "gds")
    parser.add_argument("--q3d-gds-root", type=Path, default=DEFAULT_DATA_ROOT / "exports" / "squadds")
    parser.add_argument(
        "--capn-gds-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "capn_interdigital_tee_gds",
    )
    parser.add_argument(
        "--expected-generalized",
        type=int,
        default=EXPECTED_GENERALIZED_RECORDS,
        help=f"Expected complete GeneralizedNCap count (default: {EXPECTED_GENERALIZED_RECORDS:,}).",
    )
    parser.add_argument(
        "--expected-capn",
        type=int,
        default=EXPECTED_CAPN_RECORDS,
        help=f"Expected complete CapN count (default: {EXPECTED_CAPN_RECORDS:,}).",
    )
    parser.add_argument(
        "--allow-incomplete-generalized",
        action="store_true",
        help="Continue after printing every unmatched Generalized row diagnostic.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Build only the first N canonically ordered pairings for a smoke test.",
    )
    parser.add_argument("--boundary-samples", type=int, default=96)
    parser.add_argument("--no-canonical-rotation", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Verify and continue an exact cache prefix.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing complete cache without extracting geometry.",
    )
    parser.add_argument(
        "--skip-gds-rehash",
        action="store_true",
        help="Skip re-hashing GDS artifacts while resuming/verifying; all internal hashes are still checked.",
    )
    return parser


def _catalogue(args: argparse.Namespace) -> list[CatalogueEntry]:
    entries: list[CatalogueEntry] = []
    if args.family in {"all", "generalized"}:
        generalized = pair_generalized_records(
            args.generalized_json,
            exp6_gds_dir=args.exp6_gds_dir,
            exp7_gds_dir=args.exp7_gds_dir,
            q3d_gds_root=args.q3d_gds_root,
            expected_count=args.expected_generalized,
            dataset_role="source",
        )
        for diagnostic in generalized.diagnostics:
            print(
                f"UNMATCHED row={diagnostic.row_index} source_id={diagnostic.source_id} "
                f"path={diagnostic.expected_gds_path} reason={diagnostic.reason}",
                file=sys.stderr,
            )
        if not args.allow_incomplete_generalized:
            generalized.assert_complete()
        entries.extend(generalized.entries)
        print(
            f"Paired GeneralizedCapNInterdigital: {len(generalized):,}/{generalized.source_row_count:,} "
            f"(expected {args.expected_generalized:,})."
        )

    if args.family in {"all", "capn"}:
        capn = pair_capn_records(
            args.capn_json,
            gds_dir=args.capn_gds_dir,
            expected_count=args.expected_capn,
            dataset_role="target",
        )
        capn.assert_complete()
        entries.extend(capn.entries)
        print(f"Paired CapNInterdigitalTee: {len(capn):,}/{capn.source_row_count:,} (expected {args.expected_capn:,}).")

    if args.limit is not None:
        if args.limit < 1:
            raise CatalogueError("--limit must be positive.")
        entries = entries[: args.limit]
        print(f"Smoke-test limit selected the first {len(entries):,} canonical records.")
    return entries


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        entries = _catalogue(args)
        if not entries:
            raise CatalogueError("No catalogue entries were selected.")
        if args.verify_only:
            verification = verify_graph_cache(
                entries,
                args.output_dir,
                verify_gds=not args.skip_gds_rehash,
                boundary_sample_count=args.boundary_samples,
                canonicalize_rotation=not args.no_canonical_rotation,
            )
            print(
                f"Verified {verification.record_count:,} graph/sidecar pairs; "
                f"graphs sha256={verification.graph_jsonl_sha256}."
            )
            return

        summary = write_graph_cache(
            entries,
            args.output_dir,
            resume=args.resume,
            verify_existing_gds=not args.skip_gds_rehash,
            canonicalize_rotation=not args.no_canonical_rotation,
            boundary_sample_count=args.boundary_samples,
            progress_every=args.progress_every,
            progress=print,
        )
        verification = verify_graph_cache(
            entries,
            args.output_dir,
            verify_gds=not args.skip_gds_rehash,
            boundary_sample_count=args.boundary_samples,
            canonicalize_rotation=not args.no_canonical_rotation,
        )
        print(
            f"TopoCap cache complete: requested={summary.requested_count:,}, "
            f"resumed={summary.existing_count:,}, written={summary.written_count:,}."
        )
        print(f"Graph JSONL: {summary.graph_jsonl} (sha256={verification.graph_jsonl_sha256})")
        print(f"Sidecars: {summary.output_dir / 'sidecars'} ({verification.sidecar_count:,} verified)")
        print(f"Manifest: {summary.cache_manifest}")
    except (CatalogueError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
