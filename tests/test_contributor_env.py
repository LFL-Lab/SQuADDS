from datetime import datetime

import pytest

from squadds.database import contributor_env


def test_build_contributor_record_uses_environment_values(monkeypatch):
    monkeypatch.setattr(contributor_env, "load_contributor_environment", lambda: None)
    monkeypatch.setenv("GROUP_NAME", "LFL")
    monkeypatch.setenv("PI_NAME", "Schuster")
    monkeypatch.setenv("INSTITUTION", "Stanford")
    monkeypatch.setenv("USER_NAME", "shanto")
    monkeypatch.setenv("CONTRIB_MISC", "notes")

    record = contributor_env.build_contributor_record(datetime(2026, 4, 16, 19, 30, 0))

    assert record == {
        "group": "LFL",
        "PI": "Schuster",
        "institution": "Stanford",
        "uploader": "shanto",
        "misc": "notes",
        "date_created": "2026-04-16 193000",
    }


def test_get_hf_api_and_token_requires_existing_hf_login(monkeypatch):
    class FakeApi:
        pass

    monkeypatch.setattr(contributor_env, "load_contributor_environment", lambda: None)
    monkeypatch.setattr(contributor_env, "HfApi", FakeApi)
    monkeypatch.setattr(contributor_env, "get_token", lambda: None)

    with pytest.raises(ValueError, match="Hugging Face token not found"):
        contributor_env.get_hf_api_and_token()


def test_get_hf_api_and_token_logs_in_with_env_token(monkeypatch):
    calls = []

    class FakeApi:
        pass

    monkeypatch.setattr(contributor_env, "load_contributor_environment", lambda: None)
    monkeypatch.setattr(contributor_env, "HfApi", FakeApi)
    monkeypatch.setattr(contributor_env, "get_token", lambda: "cached-token")
    monkeypatch.setattr(contributor_env, "login", lambda token: calls.append(token))
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "env-token")

    api, token = contributor_env.get_hf_api_and_token()

    assert isinstance(api, FakeApi)
    assert token == "env-token"
    assert calls == ["env-token"]


def test_get_hf_api_and_token_falls_back_to_cached_token_when_env_var_missing(monkeypatch):
    calls = []

    class FakeApi:
        pass

    monkeypatch.setattr(contributor_env, "load_contributor_environment", lambda: None)
    monkeypatch.setattr(contributor_env, "HfApi", FakeApi)
    monkeypatch.setattr(contributor_env, "get_token", lambda: "cached-token")
    monkeypatch.setattr(contributor_env, "login", lambda token: calls.append(token))
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    api, token = contributor_env.get_hf_api_and_token()

    assert isinstance(api, FakeApi)
    assert token == "cached-token"
    assert calls == ["cached-token"]
