"""Utilities for the database package."""
from pathlib import Path
from squadds.core.globals import *

def create_contributor_info():
    """
    Prompt the user for information and update the .env file.

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