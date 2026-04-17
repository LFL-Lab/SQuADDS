"""Merge helpers for composing qubit and cavity datasets."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    qubit_df, cavity_df = add_merger_terms_columns(qubit_df, cavity_df, merger_terms)

    qubit_df = qubit_df.reset_index().rename(columns={"index": "index_qc"})

    if parallelize:
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
    return merged_df
