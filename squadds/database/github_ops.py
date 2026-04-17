"""Pure helpers for the GitHub contribution workflow."""

from __future__ import annotations

import json
import os
from datetime import datetime

from dotenv import load_dotenv

DEFAULT_TEMP_CLONE_DIR = "./temp_forked_repo"
MEASURED_DEVICE_JSON_PATH = "measured_device_database.json"
ORIGINAL_DATASET_REPO_NAME = "LFL-Lab/SQuADDS_DB"


def load_github_token() -> str:
    """Load the GitHub token from environment variables using the legacy dotenv flow."""
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token is None:
        raise ValueError("GitHub token not found in environment variables.")
    return github_token


def build_forked_repo_name(github_username: str) -> str:
    """Return the legacy forked repository name for measured-data contributions."""
    return f"{github_username}/SQuADDS_DB"


def build_measured_data_commit_message(timestamp: datetime | None = None) -> str:
    """Build the measured-dataset commit message using the current legacy format."""
    timestamp = timestamp or datetime.now()
    return f"Update JSON dataset with new entry - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"


def build_authenticated_remote_url(remote_url: str, github_token: str) -> str | None:
    """Inject a GitHub token into an HTTPS remote URL and leave non-HTTPS URLs unsupported."""
    if remote_url.startswith("https://"):
        repo_name = remote_url.split("github.com/")[1]
        return f"https://{github_token}@github.com/{repo_name}"
    return None


def read_json_file(file_path):
    """Read and return the contents of the specified JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Reading JSON file from {file_path}...")
    with open(file_path) as file:
        data = json.load(file)
    return data


def append_to_json(data, new_entry):
    """Append a new entry to the loaded JSON structure using the legacy in-place behavior."""
    print("Appending new entry to JSON data...")
    data.append(new_entry)
    return data


def save_json_file(file_path, data):
    """Save the updated JSON payload back to disk."""
    print(f"Saving updated JSON data to {file_path}...")
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
