"""Environment helpers for contributor-facing workflows."""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token, login

from squadds.core.globals import ENV_FILE_PATH


def load_contributor_environment() -> None:
    """Load contributor environment variables from the project `.env` file."""
    load_dotenv(ENV_FILE_PATH)


def build_contributor_record(timestamp: datetime | None = None) -> dict[str, str | None]:
    """Build the legacy contributor metadata payload from environment variables."""
    load_contributor_environment()
    timestamp = timestamp or datetime.now()
    return {
        "group": os.getenv("GROUP_NAME"),
        "PI": os.getenv("PI_NAME"),
        "institution": os.getenv("INSTITUTION"),
        "uploader": os.getenv("USER_NAME"),
        "misc": os.getenv("CONTRIB_MISC"),
        "date_created": timestamp.strftime("%Y-%m-%d %H%M%S"),
    }


def get_hf_api_and_token() -> tuple[HfApi, str | None]:
    """
    Return the Hugging Face API client and token using the legacy login flow.

    This preserves the current behavior: require a cached/authenticated token to
    exist first, then read `HUGGINGFACE_API_KEY` from the environment and use it
    for the explicit `login()` call.
    """
    load_contributor_environment()
    api = HfApi()
    token = get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")

    token = os.getenv("HUGGINGFACE_API_KEY")
    login(token)
    return api, token
