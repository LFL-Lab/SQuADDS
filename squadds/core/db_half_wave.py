"""Helpers for the half-wave cavity flow in `SQuADDS_DB`."""

from __future__ import annotations

import os


NCAP_SIM_COLS = [
    "bottom_to_bottom",
    "bottom_to_ground",
    "ground_to_ground",
    "top_to_bottom",
    "top_to_ground",
    "top_to_top",
]


def filter_and_validate_ncap_cavity_df(cavity_df, *, filter_df_by_conditions_fn, coupler_type: str = "NCap"):
    """Filter a cavity dataframe to the required coupler type and validate the result."""
    cavity_df = filter_df_by_conditions_fn(cavity_df, {"coupler_type": coupler_type})
    if not all(cavity_df["coupler_type"] == coupler_type):
        raise ValueError(f"All entries in the 'coupler_type' column of the cavity_df must be '{coupler_type}'.")
    return cavity_df


def optimize_half_wave_dataframe(
    df,
    *,
    process_design_options_fn,
    compute_memory_usage_fn,
    optimize_dataframe_fn,
    delete_object_columns_fn,
    delete_categorical_columns_fn,
):
    """Run the legacy half-wave dataframe optimization pipeline."""
    opt_df = process_design_options_fn(df)
    initial_mem = compute_memory_usage_fn(df)
    opt_df = optimize_dataframe_fn(opt_df)
    opt_df = delete_object_columns_fn(opt_df)
    opt_df = delete_categorical_columns_fn(opt_df)
    final_mem = compute_memory_usage_fn(opt_df)
    return opt_df, initial_mem, final_mem


def save_half_wave_parquet_outputs(cavity_df, merged_df, opt_df, data_dir: str = "data"):
    """Persist the half-wave parquet outputs using the legacy filenames."""
    os.makedirs(data_dir, exist_ok=True)
    cavity_df.to_parquet(os.path.join(data_dir, "half-wave-cavity_df.parquet"))
    merged_df.to_parquet(os.path.join(data_dir, "qubit_half-wave-cavity_df_uncompressed.parquet"))
    opt_df.to_parquet(os.path.join(data_dir, "qubit_half-wave-cavity_df.parquet"))
