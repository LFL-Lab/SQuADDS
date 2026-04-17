"""Helpers for Hugging Face dataset naming and uploads."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime


def build_dataset_name(
    components: list[str],
    data_type: str,
    data_nature: str,
    data_source: str,
    institute: str | None,
    pi_name: str | None,
    date: str | None = None,
) -> str:
    """Build the legacy dataset repository name."""
    components_joined = "-".join(components)
    date = date or datetime.now().strftime("%Y%m%d")
    base_string = f"{components_joined}_{data_type}_{data_nature}_{data_source}_{institute}_{pi_name}_{date}"
    uid_hash = hashlib.sha256(base_string.encode()).hexdigest()[:8]
    return f"{base_string}_{uid_hash}"


def ensure_dataset_repository(api, token: str | None, dataset_name: str) -> None:
    """Create the dataset repository and preserve the existing print-based UX."""
    try:
        api.create_repo(repo_id=dataset_name, token=token, repo_type="dataset")
        print(f"Dataset repository {dataset_name} created.")
    except Exception as error:
        print(f"Error creating dataset repository: {error}")


def upload_dataset_files(api, token: str | None, dataset_name: str, files: list[str]) -> None:
    """Upload files to a Hugging Face dataset repository with legacy print output."""
    for file_path in files:
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=dataset_name,
                repo_type="dataset",
                token=token,
            )
            print(f"Uploaded {file_path} to {dataset_name}.")
        except Exception as error:
            print(f"Error uploading file {file_path}: {error}")
