"""Internal helpers for normalizing legacy JSON-like payloads."""

from __future__ import annotations

import json


def deserialize_json_like(value):
    """Recursively deserialize dataset fields that may already be dicts or JSON strings."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return value
        else:
            return value

    if isinstance(value, dict):
        return {key: deserialize_json_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deserialize_json_like(item) for item in value]
    return value


def normalize_setup_payload(value):
    """Return the inner setup mapping when payloads are wrapped in legacy containers."""
    payload = deserialize_json_like(value)

    if isinstance(payload, dict) and "setup" in payload:
        return normalize_setup_payload(payload["setup"])
    if isinstance(payload, list) and len(payload) == 1:
        return normalize_setup_payload(payload[0])
    return payload


def extract_setup_payload(row, *keys):
    """Read the first present setup payload from a row-like mapping."""
    for key in keys:
        if key not in row or row[key] is None:
            continue
        payload = normalize_setup_payload(row[key])
        if payload is not None:
            return payload
    raise KeyError(f"None of the setup keys were found: {keys}")


def extract_optional_setup_payload(row, *keys, default=None):
    """Best-effort variant of extract_setup_payload()."""
    try:
        return extract_setup_payload(row, *keys)
    except KeyError:
        return default
