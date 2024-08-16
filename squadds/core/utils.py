import getpass
import os
import platform
import re
import shutil
import urllib.parse
import webbrowser

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, HfFolder
from tabulate import tabulate

from squadds.core.globals import ENV_FILE_PATH


def view_contributors_from_rst(rst_file_path):
    """
    Extract and print relevant contributor information from the index.rst file.

    Args:
        rst_file_path (str): The path to the `index.rst` file.

    Returns:
        None
    """

    contributors_data = []

    with open(rst_file_path, 'r') as file:
        content = file.read()

        # Find the Contributors section
        contributors_match = re.search(r'Contributors\s+-{3,}\s+(.*?)(\n\n|$)', content, re.S)
        if contributors_match:
            contributors_section = contributors_match.group(1).strip()

            # Extract individual contributor entries
            contributor_entries = contributors_section.split("\n| ")

            for entry in contributor_entries:
                if entry.strip():
                    # Extract name, institution, and contribution
                    match = re.match(r'\*\*(.*?)\*\* \((.*?)\) - (.*)', entry.strip())
                    if match:
                        name = match.group(1)
                        institution = match.group(2)
                        contribution = match.group(3)
                        contributors_data.append([name, institution, contribution])

    if contributors_data:
        headers = ["Name", "Institution", "Contribution"]
        print(tabulate(contributors_data, headers=headers, tablefmt="grid"))
    else:
        print("No contributors found in the RST file.")


def save_intermediate_df(df, filename, file_idx):
    """
    Save the intermediate DataFrame to disk in Parquet format.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The base name of the file to save the DataFrame to.
        file_idx (int): The index of the file chunk.
    """
    df.to_parquet(f"{filename}_{file_idx}.parquet", index=False)

def set_github_token():
    """
    Sets the GitHub token by appending it to the .env file.
    If the token already exists in the .env file, it does not add it again.
    If the GitHub token is not found, it raises a ValueError.
    """
    # Check if token already exists
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'r') as file:
            existing_keys = file.read()
            if 'GITHUB_TOKEN=' in existing_keys:
                print('Token already exists in .env file.')
                return

    # Ask for the new token
    token = getpass.getpass("Enter your GitHub PAT token (with at least repo scope): ")
    # Append the new token to the .env file
    with open(ENV_FILE_PATH, 'a') as file:
        file.write(f'\nGITHUB_TOKEN={token}\n')
        print('Token added to .env file.')

def get_type(value):
    if isinstance(value, dict):
        return 'dict'
    elif isinstance(value, list):
        return 'list' if not value else get_type(value[0])
    else:
        return type(value).__name__.lower()

# Recursive function to validate types
def validate_types(data_part, schema_part):
    """
    Recursively validates the types of data_part against the expected types defined in schema_part.

    Args:
        data_part (dict): The data to be validated.
        schema_part (dict): The schema defining the expected types.

    Raises:
        TypeError: If the type of any key in data_part does not match the expected type in schema_part.

    Returns:
        None
    """
    for key, expected_type in schema_part.items():
        if isinstance(expected_type, dict):
            validate_types(data_part[key], expected_type)
        else:
            actual_type = get_type(data_part[key])
            if actual_type != expected_type:
                raise TypeError(f"Invalid type for {key}. Expected {expected_type}, got {actual_type}.")

def get_config_schema(entry):
    """
    Generates the schema for the given entry with specific rules.
    The 'sim_results' are fully expanded, while others are expanded to the first level.
    """
    def get_type(value):
        # Return the type as a string representation
        if isinstance(value, dict):
            return 'dict'
        elif isinstance(value, list):
            # Check the type of the first item if the list is not empty
            return 'list' if not value else get_type(value[0])
        else:
            return type(value).__name__.lower()

    schema = {}
    for key, value in entry.items():
        if key == 'sim_results':
            # Fully expand 'sim_results'
            schema[key] = {k: get_type(v) for k, v in value.items()}
        elif key in ['sim_options', 'design', 'notes'] and isinstance(value, dict):
            # Expand to the first level for 'sim_options', 'design', and 'notes'
            schema[key] = {k: get_type(v) for k, v in value.items()}
        else:
            schema[key] = get_type(value)

    return schema

def get_schema(obj):
    """
    Returns the schema of the given object.

    Args:
        obj: The object for which the schema needs to be determined.

    Returns:
        The schema of the object. If the object is a dictionary, the schema will be a dictionary
        with the same keys as the original dictionary, where the values represent the schema of
        the corresponding values in the original dictionary. If the object is a list, the schema
        will be either 'dict' if the list contains dictionaries, or the type name of the first
        element in the list. For any other type of object, the schema will be the type name of
        the object.

    """
    if isinstance(obj, dict):
        return {k: 'dict' if isinstance(v, dict) else get_schema(v) for k, v in obj.items() if k != 'contributor'}
    elif isinstance(obj, list):
        # If the list contains dictionaries, just return 'dict', else get the type of the first element
        return 'dict' if any(isinstance(elem, dict) for elem in obj) else type(obj[0]).__name__
    else:
        return type(obj).__name__

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def compare_schemas(data_schema, expected_schema, path=''):
    """
    Compare two schemas and raise an error if there are any mismatches.

    Args:
        data_schema (dict): The data schema to compare.
        expected_schema (dict): The expected schema to compare against.
        path (str, optional): The current path in the schema. Used for error messages. Defaults to ''.

    Raises:
        ValueError: If there is a key in the data schema that is not present in the expected schema.
        ValueError: If there is a type mismatch between the data schema and the expected schema.

    """
    for key, data_type in data_schema.items():
        # Check if the key exists in the expected schema
        if key not in expected_schema:
            raise ValueError(f"Unexpected key '{path}{key}' found in data schema.")

        expected_type = expected_schema[key]

        # Compare types for nested dictionaries
        if isinstance(expected_type, dict):
            if not isinstance(data_type, dict):
                raise ValueError(f"Type mismatch for '{path}{key}'. Expected a dict, Got: {get_type(data_type)}")
            compare_schemas(data_type, expected_type, path + key + '.')
        else:
            # Compare types for simple fields
            if get_type(data_type) != expected_type:
                #! TODO: fix this:: if float is expected but got str then ignore
                if expected_type == 'float' and get_type(data_type) == "str":
                    continue
                else:
                    raise ValueError(f"Type mismatch for '{path}{key}'. Expected: {expected_type}, Got: {get_type(data_type)}")

def convert_to_numeric(value):
    """
    Converts a value to a numeric type if possible.

    Args:
        value: The value to be converted.

    Returns:
        The converted value if it can be converted to int or float, otherwise returns the original value.
    """
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        elif is_float(value):
            return float(value)
    return value

def convert_to_str(value:float, units: str):
    """
    Converts the given value to a string with the given units.
    Args:
        value (float): The value to be converted.
        units (str): The units to be appended to the value.
    Returns:
        str: The value as a string with the units.
    """
    return f"{value} {units}"

def convert_list_to_str(lst):
    """
    Converts the given list of floats to a string representation.
    Args:
        lst (list): The list of floats to be converted.
    Returns:
        str: The string representation of the list.
    """

    return [convert_to_str(item) for item in lst]

def get_entire_schema(obj):
    """
    Recursively traverses the given object and returns a schema representation.

    Args:
        obj: The object to generate the schema for.

    Returns:
        The schema representation of the object.
    """
    if isinstance(obj, dict):
        return {k: get_entire_schema(v) for k, v in obj.items() if k != 'contributor'}
    elif isinstance(obj, list):
        return [get_entire_schema(o) for o in obj][0] if obj else []
    else:
        return type(obj).__name__

def delete_HF_cache():
    """
    Deletes the cache directory for the specific dataset.
    """
    # Determine the root cache directory for 'datasets'
    # Default cache directory is '~/.cache/huggingface/datasets' on Unix systems
    # and 'C:\\Users\\<username>\\.cache\\huggingface\\datasets' on Windows
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")

    # Adjust the path for Windows if necessary
    if platform.system() == "Windows":
        cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "huggingface", "datasets")

    # Define the specific dataset cache directory name
    dataset_cache_dir_name = "SQuADDS___s_qu_adds_db"

    # Path for the specific dataset cache
    dataset_cache_dir = os.path.join(cache_dir, dataset_cache_dir_name)

    # Check if the cache directory exists
    if os.path.exists(dataset_cache_dir):
        try:
            # Delete the dataset cache directory
            shutil.rmtree(dataset_cache_dir)
        except OSError as e:
            print(f"Error occurred while deleting cache: {e}")
    else:
        pass

def get_sim_results_keys(dataframes):
    """
    Get the unique keys from the 'sim_results' column of the given dataframes.

    Args:
        dataframes (list or pandas.DataFrame): A list of dataframes or a single dataframe.

    Returns:
        list: A list of unique keys extracted from the 'sim_results' column.
    """
    # Initialize an empty list to store all keys
    all_keys = []

    # Ensure the input is a list, even if it's a single dataframe
    if not isinstance(dataframes, list):
        dataframes = [dataframes]

    # Iterate over each dataframe
    for df in dataframes:
        # Check if 'sim_results' column exists in the dataframe
        if 'sim_results' in df.columns:
            # Extract keys from each row's 'sim_results' and add them to the list
            for row in df['sim_results']:
                if isinstance(row, dict):  # Ensure the row is a dictionary
                    all_keys.extend(row.keys())

    # Remove duplicates from the list
    unique_keys = list(set(all_keys))

    return unique_keys

def convert_numpy(obj):
    """
    Converts NumPy arrays to Python lists recursively.

    Args:
        obj: The object to be converted.

    Returns:
        The converted object.

    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

# Function to create a unified design_options dictionary
def create_unified_design_options(row):
    # TODO: no hardcoding
    """
    Create a unified design options dictionary based on the given row.

    Args:
        row (pandas.Series): The row containing the design options.

    Returns:
        dict: The unified design options dictionary.
    """
    cavity_dict = convert_numpy(row["design_options_cavity_claw"])
    coupler_type = row["coupler_type"]

    qubit_dict = convert_numpy(row["design_options_qubit"])

    # setting the `claw_cpw_*` params to zero
    qubit_dict['connection_pads']['readout']['claw_cpw_width'] = "0um"
    qubit_dict['connection_pads']['readout']['claw_cpw_length'] = "0um"
    cavity_dict['claw_opts']['connection_pads']['readout']['claw_cpw_width'] = "0um"
    cavity_dict['claw_opts']['connection_pads']['readout']['claw_cpw_length'] = "0um"

    # replacing the ground spacing of the cavity by that of the qubit
    cavity_dict["claw_opts"]['connection_pads']["readout"]["ground_spacing"] = qubit_dict['connection_pads']['readout']['ground_spacing']

    # setting the `claw_cpw_*` params to zero
    qubit_dict['connection_pads']['readout']['claw_cpw_width'] = "0um"
    qubit_dict['connection_pads']['readout']['claw_cpw_length'] = "0um"
    cavity_dict['claw_opts']['connection_pads']['readout']['claw_cpw_width'] = "0um"
    cavity_dict['claw_opts']['connection_pads']['readout']['claw_cpw_length'] = "0um"

    # replacing the ground spacing of the cavity by that of the qubit
    cavity_dict["claw_opts"]['connection_pads']["readout"]["ground_spacing"] = qubit_dict['connection_pads']['readout']['ground_spacing']

    device_dict = {
        "cavity_claw_options": {
            "coupler_type": coupler_type,
            "coupler_options": cavity_dict.get("cplr_opts", {}),
            "cpw_opts": {
                "left_options": cavity_dict.get("cpw_opts", {})
            }
        },
        "qubit_options": qubit_dict
    }

    return device_dict


def flatten_df_second_level(df):
    """
    Flattens a DataFrame by expanding dictionary-like data in the second level of columns.

    Args:
        df (pandas.DataFrame): The DataFrame to be flattened.

    Returns:
        pandas.DataFrame: A new DataFrame with the flattened data.
    """
    # Initialize an empty dictionary to collect flattened data
    flattened_data = {}

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the column contains dictionary-like data
        if isinstance(df[column].iloc[0], dict):
            # Iterate over second-level keys and create new columns
            for key in df[column].iloc[0].keys():
                flattened_data[f"{key}"] = df[column].apply(lambda x: x[key] if key in x else None)
        else:
            # For non-dictionary data, keep as is
            flattened_data[column] = df[column]

    # Create a new DataFrame with the flattened data
    new_df = pd.DataFrame(flattened_data)
    return new_df

def filter_df_by_conditions(df, conditions):
    """
    Filter a DataFrame based on given conditions.

    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        conditions (dict): A dictionary containing column-value pairs as conditions.

    Returns:
        pandas.DataFrame: The filtered DataFrame.

    Raises:
        None

    """
    # Ensure conditions is a dictionary
    if not isinstance(conditions, dict):
        print("Conditions must be provided as a dictionary.")
        return None

    # Start with the original DataFrame
    filtered_df = df

    # Apply each condition
    for column, value in conditions.items():
        if column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column] == value]


    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        print("Warning: No rows match the given conditions. Returning the original DataFrame.")
        return df
    else:
        return filtered_df

def set_huggingface_api_key():
    """
    Sets the Hugging Face API key by appending it to the .env file.
    If the API key already exists in the .env file, it does not add it again.
    If the Hugging Face token is not found, it raises a ValueError.
    """
    # Check if API key already exists
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'r') as file:
            existing_keys = file.read()
            if 'HUGGINGFACE_API_KEY=' in existing_keys:
                print('API key already exists in .env file.')
                return

    # Ask for the new API key
    api_key = getpass.getpass("Enter your Hugging Face API key: ")
    # Append the new API key to the .env file
    with open(ENV_FILE_PATH, 'a') as file:
        file.write(f'\nHUGGINGFACE_API_KEY={api_key}\n')
        print('API key added to .env file.')

    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")


def create_mailto_link(recipients, subject, body):
    """
    Create a mailto link with the given recipients, subject, and body.

    Args:
        recipients (list): A list of email addresses of the recipients.
        subject (str): The subject of the email.
        body (str): The body of the email.

    Returns:
        str: The generated mailto link.

    """
    # Encode the subject and body using urllib.parse.quote_plus to handle special characters
    subject_encoded = urllib.parse.quote_plus(subject)
    body_encoded = urllib.parse.quote_plus(body)

    # Construct the mailto link with the encoded subject and body
    mailto_link = f"mailto:{','.join(recipients)}?subject={subject_encoded}&body={body_encoded}"

    # Replace '+' with '%20' for proper space encoding
    mailto_link = mailto_link.replace('+', '%20')
    return mailto_link


def send_email_via_client(dataset_name, institute, pi_name, date, dataset_link):
    """
    Sends an email notification to recipients with the details of the created dataset.

    Args:
        dataset_name (str): The name of the dataset.
        institute (str): The name of the institute where the dataset was created.
        pi_name (str): The name of the principal investigator who created the dataset.
        date (str): The date when the dataset was created.
        dataset_link (str): The link to the created dataset.

    Returns:
        None
    """
    recipients = ["shanto@usc.edu", "elevenso@usc.edu"]
    subject = f"SQuADDS: Dataset Created - {dataset_name} ({date})"
    body = f"{dataset_name} has been created by {pi_name} at {institute} on {date}.\nHere is the link - {dataset_link}"

    mailto_link = create_mailto_link(recipients, subject, body)
    webbrowser.open(mailto_link)

def compute_memory_usage(df):
    """
    Compute the memory usage of the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to compute the memory usage for.

    Returns:
        float: The memory usage of the DataFrame in megabytes.
    """
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Memory usage: {mem} MB")
    return mem

def print_column_types(df):
    """
    Prints out the columns in the DataFrame that have floats, integers, objects, datetimes, and strings.

    Parameters:
    - df: DataFrame to analyze.
    """
    float_cols = df.select_dtypes(include=['float']).columns.tolist()
    int_cols = df.select_dtypes(include=['int']).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    string_cols = df.select_dtypes(include=['string']).columns.tolist()
    
    print("Columns with float types:", float_cols)
    print("Columns with int types:", int_cols)
    print("Columns with object types:", object_cols)
    print("Columns with datetime types:", datetime_cols)
    print("Columns with string types:", string_cols)

def can_be_categorical(column):
    """
    Check if all elements in the column are hashable.
    """
    try:
        hashable = all(isinstance(item, (str, int, float, tuple)) or item is None for item in column)
        if hashable:
            pd.Categorical(column)
            return True
        else:
            return False
    except TypeError:
        return False

def optimize_dataframe(df):
    """
    Optimize the memory usage of a pandas DataFrame by downcasting data types.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be optimized.

    Returns:
    - df_optimized (pandas.DataFrame): The optimized DataFrame.

    """
    df_optimized = df.copy()
    
    # Calculate initial memory usage
    initial_memory_usage = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    
    # Memory usage before optimization
    memory_before_floats = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    
    # Optimize float columns
    for col in df_optimized.select_dtypes(include=['float']):
        df_optimized[col] = df_optimized[col].astype('float32')
    
    # Memory usage after optimizing floats
    memory_after_floats = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    
    # Optimize integer columns
    for col in df_optimized.select_dtypes(include=['int']):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='unsigned')
    
    # Memory usage after optimizing integers
    memory_after_ints = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    
    # Memory usage before optimizing object columns
    memory_before_objects = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    
    # Optimize object columns
    for col in df_optimized.select_dtypes(include=['object']):
        if can_be_categorical(df_optimized[col]):
            df_optimized[col] = df_optimized[col].astype('category')
    
    # Memory usage after optimizing object columns
    memory_after_objects = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    
    # Calculate memory savings and percentages
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
    """
    Processes the 'design_options' column in merged_df, appends new columns, converts values, and drops 'design_options'.
    
    Parameters:
    - merged_df: DataFrame containing the 'design_options' column.
    
    Returns:
    - merged_df: Modified DataFrame with new columns added and 'design_options' dropped.
    """
    def convert_value(value):
        if value is None:
            return value
        if isinstance(value, str) and value.endswith("um"):
            return np.float16(value[:-2])
        return np.int16(value)
    
    design_options = merged_df["design_options"]

    merged_df["finger_count"] = design_options.apply(lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["finger_count"]))
    merged_df["finger_length"] = design_options.apply(lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["finger_length"]))
    merged_df["cap_gap"] = design_options.apply(lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["cap_gap"]))
    merged_df["cap_width"] = design_options.apply(lambda x: convert_value(x["cavity_claw_options"]["coupler_options"]["cap_width"]))
    merged_df["total_length"] = design_options.apply(lambda x: convert_value(x["cavity_claw_options"]["cpw_opts"]["left_options"]["total_length"]))
    merged_df["meander_spacing"] = design_options.apply(lambda x: convert_value(x["cavity_claw_options"]["cpw_opts"]["left_options"]["meander"]["spacing"]))
    merged_df["meander_asymmetry"] = design_options.apply(lambda x: convert_value(x["cavity_claw_options"]["cpw_opts"]["left_options"]["meander"]["asymmetry"]))
    merged_df["claw_length"] = design_options.apply(lambda x: convert_value(x["qubit_options"]["connection_pads"]["readout"]["claw_length"]))
    merged_df["claw_width"] = design_options.apply(lambda x: convert_value(x["qubit_options"]["connection_pads"]["readout"]["claw_width"]))
    merged_df["ground_spacing"] = design_options.apply(lambda x: convert_value(x["qubit_options"]["connection_pads"]["readout"]["ground_spacing"]))
    merged_df["cross_length"] = design_options.apply(lambda x: convert_value(x["qubit_options"]["cross_length"]))

    # Drop the 'design_options' column
    merged_df.drop(columns=["design_options"], inplace=True)
    
    return merged_df

def print_column_types(df):
    """
    Prints out the data type of each column in the DataFrame.

    Parameters:
    - df: DataFrame to analyze.
    """
    column_types = df.dtypes
    for col, dtype in column_types.items():
        print(f"Column: {col}, Data Type: {dtype}")

def delete_object_columns(df):
    """
    Deletes all columns of type 'object' from the DataFrame.

    Parameters:
    - df: DataFrame to process.

    Returns:
    - df: DataFrame with 'object' columns removed.
    """
    # Select columns that are of type 'object'
    object_columns = df.select_dtypes(include=['object']).columns

    # Drop the selected columns
    df = df.drop(columns=object_columns)
    
    return df


def columns_memory_usage(df):
    """
    Calculates the memory usage of each column and returns a DataFrame showing each column's memory usage and percentage of total memory usage.

    Parameters:
    - df: DataFrame to process.

    Returns:
    - mem_usage_df: DataFrame with columns 'Column', 'Memory Usage (MB)', and 'Percentage of Total Memory Usage'.
    """
    # Calculate the memory usage for each column
    mem_usage = df.memory_usage(deep=True) / (1024 ** 2)  # Convert to MB
    
    # Calculate total memory usage
    total_mem_usage = mem_usage.sum()
    
    # Create a DataFrame with memory usage and percentage of total memory usage
    mem_usage_df = pd.DataFrame({
        'Column': mem_usage.index,
        'Memory Usage (MB)': mem_usage.values,
        'Percentage of Total Memory Usage': (mem_usage.values / total_mem_usage) * 100
    })
    
    # Sort the DataFrame by memory usage in descending order
    mem_usage_df = mem_usage_df.sort_values(by='Memory Usage (MB)', ascending=False).reset_index(drop=True)
    
    return mem_usage_df

# drop all categorical columns
def delete_categorical_columns(df):
    """
    Deletes all columns of type 'category' from the DataFrame.

    Parameters:
    - df: DataFrame to process.

    Returns:
    - df: DataFrame with 'category' columns removed.
    """
    # Select columns that are of type 'category'
    category_columns = df.select_dtypes(include=['category']).columns

    # Drop the selected columns
    df = df.drop(columns=category_columns)
    
    return df