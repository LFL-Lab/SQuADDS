"""Merge helpers for composing qubit and cavity datasets."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from squadds.core.json_utils import deserialize_json_like
from squadds.core.processing import merge_dfs
from squadds.core.utils import create_unified_design_options


def add_merger_terms_columns(qubit_df: pd.DataFrame, cavity_df: pd.DataFrame, merger_terms: list[str]):
    """Populate merger-term columns using the legacy nested design-option lookups."""
    for merger_term in merger_terms:
        qubit_df[merger_term] = qubit_df["design_options"].map(
            lambda x: x["connection_pads"]["readout"].get(merger_term)
        )
        cavity_df[merger_term] = cavity_df["design_options"].map(
            lambda x: x["claw_opts"]["connection_pads"]["readout"].get(merger_term)
        )
    return qubit_df, cavity_df


def create_qubit_cavity_dataframe(
    qubit_df: pd.DataFrame,
    cavity_df: pd.DataFrame,
    merger_terms: list[str] | None = None,
    parallelize: bool = False,
    num_cpu=None,
) -> pd.DataFrame:
    """Merge qubit and cavity dataframes using the legacy SQuADDS_DB behavior."""
    merger_terms = [] if merger_terms is None else merger_terms
    qubit_df, cavity_df = add_merger_terms_columns(qubit_df, cavity_df, merger_terms)

    qubit_df = qubit_df.reset_index().rename(columns={"index": "index_qc"})

    if not merger_terms:
        merged_df = qubit_df.merge(cavity_df, how="cross", suffixes=("_qubit", "_cavity_claw"))
    elif parallelize:
        n_cores = cpu_count() if num_cpu is None else num_cpu
        qubit_df_splits = np.array_split(qubit_df, n_cores)

        with Pool(n_cores) as pool:
            merged_df_parts = list(
                tqdm(
                    pool.starmap(merge_dfs, [(split, cavity_df, merger_terms) for split in qubit_df_splits]),
                    total=n_cores,
                )
            )

        merged_df = pd.concat(merged_df_parts).reset_index(drop=True)
    else:
        merged_df = merge_dfs(qubit_df, cavity_df, merger_terms)

    merged_df["design_options"] = merged_df.apply(create_unified_design_options, axis=1)

    # Inflate any JSON-string sub-payloads (e.g. `cplr_opts`, `meander`, `lead`)
    # that the HuggingFace dataset stores as strings, so consumers that mutate
    # `design_options_qubit` / `design_options_cavity_claw` / `design_options`
    # via nested dict access always see dicts instead of raw JSON strings.
    for col in ("design_options_qubit", "design_options_cavity_claw", "design_options"):
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].map(deserialize_json_like)
    return merged_df
