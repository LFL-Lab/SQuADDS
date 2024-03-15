import getpass
import os
import platform
import shutil
import urllib.parse
import webbrowser

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, HfFolder

from squadds.core.globals import ENV_FILE_PATH


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
    try:
        coupler_type = row["coupler_type"]
    except:
        coupler_type = "CLT"
    qubit_dict = convert_numpy(row["design_options_qubit"])

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
