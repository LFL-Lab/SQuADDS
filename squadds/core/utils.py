import urllib.parse
import webbrowser
import getpass
import os
from huggingface_hub import HfApi, HfFolder
from squadds.core.globals import ENV_FILE_PATH
import pandas as pd
import numpy as np

def get_sim_results_keys(dataframes):
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

    device_dict = {
        "cavity_claw_options": {
            "coupling_type": coupler_type,
            "coupler_options": cavity_dict.get("cplr_opts", {}),
            "cpw_options": {
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
