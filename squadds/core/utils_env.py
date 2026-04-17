import getpass
import os

from squadds.core.globals import ENV_FILE_PATH


def set_github_token():
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH) as file:
            existing_keys = file.read()
            if "GITHUB_TOKEN=" in existing_keys:
                print("Token already exists in .env file.")
                return

    token = getpass.getpass("Enter your GitHub PAT token (with at least repo scope): ")
    with open(ENV_FILE_PATH, "a") as file:
        file.write(f"\nGITHUB_TOKEN={token}\n")
        print("Token added to .env file.")
