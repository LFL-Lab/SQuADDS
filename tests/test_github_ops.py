import json
from datetime import datetime

import pytest

from squadds.database import github_ops
from squadds.database.github_ops import (
    DEFAULT_TEMP_CLONE_DIR,
    MEASURED_DEVICE_JSON_PATH,
    ORIGINAL_DATASET_REPO_NAME,
    append_to_json,
    build_authenticated_remote_url,
    build_forked_repo_name,
    build_measured_data_commit_message,
    load_github_token,
    read_json_file,
    save_json_file,
)


def test_github_workflow_constants_match_legacy_values():
    assert DEFAULT_TEMP_CLONE_DIR == "./temp_forked_repo"
    assert MEASURED_DEVICE_JSON_PATH == "measured_device_database.json"
    assert ORIGINAL_DATASET_REPO_NAME == "LFL-Lab/SQuADDS_DB"


def test_load_github_token_requires_environment_variable(monkeypatch):
    monkeypatch.setattr(github_ops, "load_dotenv", lambda: None)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(ValueError, match="GitHub token not found"):
        load_github_token()


def test_load_github_token_returns_environment_value(monkeypatch):
    monkeypatch.setattr(github_ops, "load_dotenv", lambda: None)
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")

    assert load_github_token() == "secret-token"


def test_build_forked_repo_name_uses_sqadds_db_suffix():
    assert build_forked_repo_name("shanto268") == "shanto268/SQuADDS_DB"


def test_build_measured_data_commit_message_matches_legacy_format():
    message = build_measured_data_commit_message(datetime(2026, 4, 16, 20, 15, 30))

    assert message == "Update JSON dataset with new entry - 2026-04-16 20:15:30"


def test_build_authenticated_remote_url_only_supports_https():
    assert (
        build_authenticated_remote_url("https://github.com/foo/bar.git", "abc123")
        == "https://abc123@github.com/foo/bar.git"
    )
    assert build_authenticated_remote_url("git@github.com:foo/bar.git", "abc123") is None


def test_json_file_round_trip_and_append(tmp_path):
    file_path = tmp_path / "payload.json"
    save_json_file(file_path, [{"name": "alpha"}])

    loaded = read_json_file(file_path)
    updated = append_to_json(loaded, {"name": "beta"})
    save_json_file(file_path, updated)

    assert json.loads(file_path.read_text()) == [{"name": "alpha"}, {"name": "beta"}]
