"""Utilities for the database package."""
from pathlib import Path
from squadds.core.globals import *

import json
import hashlib
import shutil
import glob
import os

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
    PI name, and user name. It then validates the input and updates the corresponding fields
    in the .env file. If the fields already exist in the .env file, the function prompts the
    user to confirm whether to overwrite the existing values.

    Raises:
        ValueError: If any of the input fields are empty.
    """
    # Check if .env file exists
    if not Path(ENV_FILE_PATH).exists():
        ENV_FILE_PATH.touch()

    # Read all lines from .env file
    with open(ENV_FILE_PATH, "r") as env_file:
        lines = env_file.readlines()

    # Convert lines to a dictionary
    existing_fields = {}
    for line in lines:
        if "=" in line:
            key, value = line.strip().split("=")
            existing_fields[key.strip()] = value.strip().strip("\"")

    # Prompt the user for information
    institution = input("Enter institution name: ")
    group_name = input("Enter your group name: ")
    pi_name = input("Enter your PI name: ")
    user_name = input("Enter your name: ")

    # Validate the input
    if not group_name:
        raise ValueError("Group name cannot be empty")
    if not pi_name:
        raise ValueError("PI name cannot be empty")
    if not institution:
        raise ValueError("Institution cannot be empty")
    if not user_name:
        raise ValueError("User name cannot be empty")

    # Update or write field values in .env file
    with open(ENV_FILE_PATH, "w") as env_file:
        for line in lines:
            if "=" in line:
                key, _ = line.strip().split("=")
                key = key.strip()
                if key in ["GROUP_NAME", "PI_NAME", "INSTITUTION", "USER_NAME"]:
                    value = locals()[key.lower()]
                    if key in existing_fields and existing_fields[key] != "":
                        overwrite = input(f"{key} already exists with value: {existing_fields[key]}. Do you want to overwrite? (y/n): ")
                        if overwrite.lower() == "y":
                            env_file.write(f"{key} = \"{value}\"\n")
                            print(f"{key} overwritten with value: {value}")
                        else:
                            env_file.write(line)
                            print(f"{key} not overwritten. Existing value: {existing_fields[key]}")
                    else:
                        env_file.write(f"{key} = \"{value}\"\n")
                        print(f"{key} updated with value: {value}")
                else:
                    env_file.write(line)
            else:
                env_file.write(line)

    print("Contributor information updated successfully!")