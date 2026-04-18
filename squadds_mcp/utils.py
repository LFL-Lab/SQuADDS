"""
Shared utilities for the SQuADDS MCP server.
=============================================

Helpers for serialization, formatting, and error handling used across
all tools and resources.

Adding a New Utility
--------------------
1. Add your function here.
2. Keep it stateless — no imports of ``SQuADDS_DB`` or ``Analyzer`` at module level.
3. Add a docstring with Args/Returns so AI agents can understand usage.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def dataframe_to_records(
    df: pd.DataFrame,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Convert a DataFrame slice to a list of JSON-safe dicts.

    Args:
        df: The pandas DataFrame.
        limit: Maximum number of rows to return.
        offset: Starting row index.

    Returns:
        List of dicts, one per row, with numpy types converted to Python builtins.
    """
    sliced = df.iloc[offset : offset + limit]
    records = sliced.to_dict(orient="records")
    return [sanitize_for_json(r) for r in records]


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types.

    Handles: ndarray, int64/float64, NaN, Timestamp, bytes, sets.

    Args:
        obj: Any Python object that may contain numpy/pandas types.

    Returns:
        A JSON-safe version of the object.
    """
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def format_design_options(design_options: dict[str, Any]) -> str:
    """Format a design options dict as a human-readable JSON string.

    Args:
        design_options: The raw design options dictionary.

    Returns:
        Pretty-printed JSON string.
    """
    return json.dumps(sanitize_for_json(design_options), indent=2)


def extract_h_params_from_row(row: pd.Series, h_param_keys: list[str]) -> dict[str, Any]:
    """Extract Hamiltonian parameter values from a DataFrame row.

    Args:
        row: A single row from the analysis DataFrame.
        h_param_keys: List of H-parameter column names.

    Returns:
        Dict of parameter name → value.
    """
    params = {}
    for key in h_param_keys:
        if key in row.index:
            val = row[key]
            params[key] = sanitize_for_json(val)
    return params


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts/objects.

    Args:
        obj: The root object to traverse.
        *keys: Sequence of keys to follow.
        default: Value to return if any key is missing.

    Returns:
        The value at the nested path, or *default*.
    """
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif hasattr(current, key):
            current = getattr(current, key, default)
        else:
            return default
        if current is default:
            return default
    return current


def build_error_response(message: str, details: dict[str, Any] | None = None) -> str:
    """Build a formatted error message for tool responses.

    Args:
        message: Human-readable error message.
        details: Optional extra context.

    Returns:
        Formatted error string.
    """
    parts = [f"❌ Error: {message}"]
    if details:
        for k, v in details.items():
            parts.append(f"  • {k}: {v}")
    return "\n".join(parts)
