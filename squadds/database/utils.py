"""Utilities for the database package."""
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path

from dotenv import dotenv_values, set_key

from squadds.core.globals import ENV_FILE_PATH


def copy_files_to_new_location(data_path, new_path):
    """
    Copy files from the given data path to the new location.

    Args:
        data_path (str): The path to the directory containing the files to be copied.
        new_path (str): The path to the directory where the files will be copied to.

    Returns:
        None

    Raises:
        None
    """
    new_names = []
    for file in glob.glob(os.path.join(data_path)):
        new_name = generate_file_name(file)
        # copy file to new location
        location = os.path.dirname(new_path)
        new_file = os.path.join(location, new_name)
        new_names.append(new_file) 
        shutil.copy(file, new_file)
    # alert if there are duplicates and show the files
    if len(new_names) != len(set(new_names)):
        print('There are duplicates!')
        print(new_names)

def generate_file_name(data_file):
    """
    Generate a unique file name based on the given data file.

    Args:
        data_file (str): The path to the data file.

    Returns:
        str: The generated file name.

    """
    with open(data_file, 'r') as file:
        data = json.load(file)
    grp = data['contributor']['group']
    inst = data['contributor']['institution']
    dc = data["contributor"]['date_created']
    # create hash based on data
    hash_fn = hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16]
    return f"{grp}_{inst}_{hash_fn}.json"

def create_contributor_info():
    """
    Prompt the user for information and update the .env file.

    This function prompts the user to enter information such as institution name, group name,
    PI name, user name, and an optional contrib_misc. It then validates the input and updates
    the corresponding fields in the .env file. If the fields already exist in the .env file,
    the function prompts the user to confirm whether to overwrite the existing values.

    Raises:
        ValueError: If any of the input fields are empty (except for contrib_misc).
    """

    # Define the keys we want to update in the .env file
    keys = ["GROUP_NAME", "PI_NAME", "INSTITUTION", "USER_NAME", "CONTRIB_MISC"]
    user_inputs = {}

    # Load existing .env values
    existing_env = dotenv_values(ENV_FILE_PATH)

    # Prompt the user for each field and validate
    for key in keys:
        user_input = input(f"Enter your {key.replace('_', ' ').lower()}: ").strip()
        if not user_input and key != "CONTRIB_MISC":  # contrib_misc is optional
            raise ValueError(f"{key} cannot be empty.")
        user_inputs[key] = user_input

        # Check if key exists and ask for confirmation to overwrite
        if key in existing_env:
            overwrite = input(f"{key} already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
            if overwrite not in ["yes", "y"]:
                continue  # Skip updating this key

        # Update or append the key-value pair
        set_key(ENV_FILE_PATH, key, user_inputs[key])

    print(f"Contributor information updated in .env file ({ENV_FILE_PATH}).")