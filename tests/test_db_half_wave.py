import pandas as pd
import pytest

from squadds.core.db_half_wave import (
    NCAP_SIM_COLS,
    filter_and_validate_ncap_cavity_df,
    optimize_half_wave_dataframe,
    save_half_wave_parquet_outputs,
)


def test_ncap_sim_cols_matches_legacy_column_list():
    assert NCAP_SIM_COLS == [
        "bottom_to_bottom",
        "bottom_to_ground",
        "ground_to_ground",
        "top_to_bottom",
        "top_to_ground",
        "top_to_top",
    ]


def test_filter_and_validate_ncap_cavity_df_filters_by_coupler_type():
    cavity_df = pd.DataFrame({"coupler_type": ["NCap", "CLT", "NCap"], "value": [1, 2, 3]})

    filtered = filter_and_validate_ncap_cavity_df(
        cavity_df,
        filter_df_by_conditions_fn=lambda df, conditions: df[df["coupler_type"] == conditions["coupler_type"]],
    )

    assert filtered["value"].tolist() == [1, 3]


def test_filter_and_validate_ncap_cavity_df_raises_if_filtered_rows_still_mismatch():
    cavity_df = pd.DataFrame({"coupler_type": ["NCap", "CLT"]})

    with pytest.raises(ValueError, match="must be 'NCap'"):
        filter_and_validate_ncap_cavity_df(cavity_df, filter_df_by_conditions_fn=lambda df, conditions: df)


def test_optimize_half_wave_dataframe_runs_pipeline_in_order():
    call_order = []

    def process_design_options_fn(df):
        call_order.append("process")
        return df.assign(step="processed")

    def compute_memory_usage_fn(df):
        call_order.append(f"mem:{df['step'].iloc[0] if 'step' in df.columns else 'raw'}")
        return len(df.columns)

    def optimize_dataframe_fn(df):
        call_order.append("optimize")
        return df.assign(step="optimized")

    def delete_object_columns_fn(df):
        call_order.append("delete_object")
        return df.assign(step="objects_deleted")

    def delete_categorical_columns_fn(df):
        call_order.append("delete_categorical")
        return df.assign(step="categoricals_deleted")

    opt_df, initial_mem, final_mem = optimize_half_wave_dataframe(
        pd.DataFrame({"value": [1]}),
        process_design_options_fn=process_design_options_fn,
        compute_memory_usage_fn=compute_memory_usage_fn,
        optimize_dataframe_fn=optimize_dataframe_fn,
        delete_object_columns_fn=delete_object_columns_fn,
        delete_categorical_columns_fn=delete_categorical_columns_fn,
    )

    assert call_order == [
        "process",
        "mem:raw",
        "optimize",
        "delete_object",
        "delete_categorical",
        "mem:categoricals_deleted",
    ]
    assert initial_mem == 1
    assert final_mem == 2
    assert opt_df["step"].tolist() == ["categoricals_deleted"]


def test_save_half_wave_parquet_outputs_writes_expected_files(tmp_path):
    cavity_df = pd.DataFrame({"value": [1]})
    merged_df = pd.DataFrame({"value": [2]})
    opt_df = pd.DataFrame({"value": [3]})

    save_half_wave_parquet_outputs(cavity_df, merged_df, opt_df, data_dir=str(tmp_path / "artifacts"))

    assert (tmp_path / "artifacts" / "half-wave-cavity_df.parquet").exists()
    assert (tmp_path / "artifacts" / "qubit_half-wave-cavity_df_uncompressed.parquet").exists()
    assert (tmp_path / "artifacts" / "qubit_half-wave-cavity_df.parquet").exists()


def test_save_half_wave_parquet_outputs_overwrites_existing_directory(tmp_path):
    data_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    (data_dir / "half-wave-cavity_df.parquet").write_text("stale")

    save_half_wave_parquet_outputs(
        pd.DataFrame({"value": [1]}),
        pd.DataFrame({"value": [2]}),
        pd.DataFrame({"value": [3]}),
        data_dir=str(data_dir),
    )

    reloaded = pd.read_parquet(data_dir / "half-wave-cavity_df.parquet")
    assert reloaded["value"].tolist() == [1]
