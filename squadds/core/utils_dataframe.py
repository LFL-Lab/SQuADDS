import numpy as np
import pandas as pd

from squadds.core.json_utils import deserialize_json_like
from squadds.core.utils_converters import convert_numpy


def save_intermediate_df(df, filename, file_idx):
    df.to_parquet(f"{filename}_{file_idx}.parquet", index=False)


def get_sim_results_keys(dataframes):
    all_keys = []
    if not isinstance(dataframes, list):
        dataframes = [dataframes]

    for df in dataframes:
        if "sim_results" not in df.columns:
            continue
        for row in df["sim_results"]:
            if isinstance(row, dict):
                all_keys.extend(row.keys())

    return list(set(all_keys))


def create_unified_design_options(row):
    # Deserialize any JSON-string payloads (e.g. `cplr_opts`, `meander`, `lead`)
    # that the HuggingFace dataset stores as strings so the unified output
    # always contains fully-nested dicts.
    cavity_dict = convert_numpy(deserialize_json_like(row["design_options_cavity_claw"]))
    coupler_type = row["coupler_type"]
    qubit_dict = convert_numpy(deserialize_json_like(row["design_options_qubit"]))

    qubit_dict["connection_pads"]["readout"]["claw_cpw_width"] = "0um"
    qubit_dict["connection_pads"]["readout"]["claw_cpw_length"] = "0um"
    cavity_dict["claw_opts"]["connection_pads"]["readout"]["claw_cpw_width"] = "0um"
    cavity_dict["claw_opts"]["connection_pads"]["readout"]["claw_cpw_length"] = "0um"
    cavity_dict["claw_opts"]["connection_pads"]["readout"]["ground_spacing"] = qubit_dict["connection_pads"]["readout"][
        "ground_spacing"
    ]

    return {
        "cavity_claw_options": {
            "coupler_type": coupler_type,
            "coupler_options": cavity_dict.get("cplr_opts", {}),
            "cpw_opts": {"left_options": cavity_dict.get("cpw_opts", {})},
        },
        "qubit_options": qubit_dict,
    }


def flatten_df_second_level(df):
    flattened_data = {}
    for column in df.columns:
        if isinstance(df[column].iloc[0], dict):
            for key in df[column].iloc[0].keys():
                flattened_data[f"{key}"] = df[column].apply(lambda value: value[key] if key in value else None)
        else:
            flattened_data[column] = df[column]
    return pd.DataFrame(flattened_data)


def filter_df_by_conditions(df, conditions):
    if not isinstance(conditions, dict):
        print("Conditions must be provided as a dictionary.")
        return None

    filtered_df = df
    for column, value in conditions.items():
        if column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column] == value]

    if filtered_df.empty:
        print("Warning: No rows match the given conditions. Returning the original DataFrame.")
        return df
    return filtered_df


def compute_memory_usage(df):
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {mem} MB")
    return mem


def can_be_categorical(column):
    try:
        hashable = all(isinstance(item, (str, int, float, tuple)) or item is None for item in column)
        if hashable:
            pd.Categorical(column)
            return True
        return False
    except TypeError:
        return False


def optimize_dataframe(df):
    df_optimized = df.copy()
    initial_memory_usage = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    memory_before_floats = df_optimized.memory_usage(deep=True).sum() / (1024**2)

    for col in df_optimized.select_dtypes(include=["float"]):
        df_optimized[col] = df_optimized[col].astype("float32")
    memory_after_floats = df_optimized.memory_usage(deep=True).sum() / (1024**2)

    for col in df_optimized.select_dtypes(include=["int"]):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="unsigned")
    memory_after_ints = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    memory_before_objects = df_optimized.memory_usage(deep=True).sum() / (1024**2)

    for col in df_optimized.select_dtypes(include=["object"]):
        if can_be_categorical(df_optimized[col]):
            df_optimized[col] = df_optimized[col].astype("category")

    memory_after_objects = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    float_savings = memory_before_floats - memory_after_floats
    int_savings = memory_after_floats - memory_after_ints
    object_savings = memory_before_objects - memory_after_objects
    total_memory_usage = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    total_savings = initial_memory_usage - total_memory_usage
    percentage_saved = (total_savings / initial_memory_usage) * 100
    float_savings_percentage = (float_savings / initial_memory_usage) * 100
    int_savings_percentage = (int_savings / initial_memory_usage) * 100
    object_savings_percentage = (object_savings / initial_memory_usage) * 100

    print(f"Initial memory usage: {initial_memory_usage:.2f} MB")
    print(f"Memory usage after optimizing floats: {memory_after_floats:.2f} MB")
    print(f"Memory usage after optimizing integers: {memory_after_ints:.2f} MB")
    print(f"Memory usage after optimizing objects: {memory_after_objects:.2f} MB")
    print(f"Total memory usage after optimization: {total_memory_usage:.2f} MB")
    print(f"Memory saved by float optimization: {float_savings:.2f} MB ({float_savings_percentage:.2f}%)")
    print(f"Memory saved by integer optimization: {int_savings:.2f} MB ({int_savings_percentage:.2f}%)")
    print(f"Memory saved by object optimization: {object_savings:.2f} MB ({object_savings_percentage:.2f}%)")
    print(f"Total memory saved: {total_savings:.2f} MB")
    print(f"Memory efficiency: {percentage_saved:.2f}%")
    return df_optimized


def process_design_options(merged_df):
    def convert_value(value):
        if value is None:
            return value
        if isinstance(value, str) and value.endswith("um"):
            return np.float16(value[:-2])
        return np.int16(value)

    design_options = merged_df["design_options"]
    merged_df["finger_count"] = design_options.apply(
        lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["finger_count"])
    )
    merged_df["finger_length"] = design_options.apply(
        lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["finger_length"])
    )
    merged_df["cap_gap"] = design_options.apply(
        lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["cap_gap"])
    )
    merged_df["cap_width"] = design_options.apply(
        lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["cap_width"])
    )
    merged_df["total_length"] = design_options.apply(
        lambda x: convert_value(x["cavity_claw_options"]["cpw_opts"]["left_options"]["total_length"])
    )
    merged_df["meander_spacing"] = design_options.apply(
        lambda x: convert_value(x["cavity_claw_options"]["cpw_opts"]["left_options"]["meander"]["spacing"])
    )
    merged_df["meander_asymmetry"] = design_options.apply(
        lambda x: convert_value(x["cavity_claw_options"]["cpw_opts"]["left_options"]["meander"]["asymmetry"])
    )
    merged_df["claw_length"] = design_options.apply(
        lambda x: convert_value(x["qubit_options"]["connection_pads"]["readout"]["claw_length"])
    )
    merged_df["claw_width"] = design_options.apply(
        lambda x: convert_value(x["qubit_options"]["connection_pads"]["readout"]["claw_width"])
    )
    merged_df["ground_spacing"] = design_options.apply(
        lambda x: convert_value(x["qubit_options"]["connection_pads"]["readout"]["ground_spacing"])
    )
    merged_df["cross_length"] = design_options.apply(lambda x: convert_value(x["qubit_options"]["cross_length"]))
    merged_df.drop(columns=["design_options"], inplace=True)
    return merged_df


def print_column_types(df):
    column_types = df.dtypes
    for col, dtype in column_types.items():
        print(f"Column: {col}, Data Type: {dtype}")


def delete_object_columns(df):
    return df.drop(columns=df.select_dtypes(include=["object"]).columns)


def columns_memory_usage(df):
    mem_usage = df.memory_usage(deep=True) / (1024**2)
    total_mem_usage = mem_usage.sum()
    mem_usage_df = pd.DataFrame(
        {
            "Column": mem_usage.index,
            "Memory Usage (MB)": mem_usage.values,
            "Percentage of Total Memory Usage": (mem_usage.values / total_mem_usage) * 100,
        }
    )
    return mem_usage_df.sort_values(by="Memory Usage (MB)", ascending=False).reset_index(drop=True)


def delete_categorical_columns(df):
    return df.drop(columns=df.select_dtypes(include=["category"]).columns)
