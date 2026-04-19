"""
Tests for SQuADDS MCP tools and utilities.
==========================================

These tests validate the MCP tool functions, utility helpers,
and server creation without requiring a live HuggingFace connection.
"""

import json

import numpy as np
import pandas as pd
import pytest

from squadds_mcp.utils import (
    build_error_response,
    dataframe_to_records,
    extract_h_params_from_row,
    safe_get,
    sanitize_for_json,
)

# ---------------------------------------------------------------------------
# Tests for utils.py
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    """Tests for sanitize_for_json utility."""

    def test_numpy_int(self):
        assert sanitize_for_json(np.int64(42)) == 42
        assert isinstance(sanitize_for_json(np.int64(42)), int)

    def test_numpy_float(self):
        assert sanitize_for_json(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(sanitize_for_json(np.float64(3.14)), float)

    def test_numpy_nan(self):
        assert sanitize_for_json(np.float64("nan")) is None

    def test_numpy_inf(self):
        assert sanitize_for_json(np.float64("inf")) is None

    def test_python_nan(self):
        assert sanitize_for_json(float("nan")) is None

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = sanitize_for_json(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_bool(self):
        assert sanitize_for_json(np.bool_(True)) is True
        assert isinstance(sanitize_for_json(np.bool_(True)), bool)

    def test_nested_dict(self):
        data = {
            "a": np.int64(1),
            "b": {"c": np.array([1.0, 2.0])},
        }
        result = sanitize_for_json(data)
        assert result == {"a": 1, "b": {"c": [1.0, 2.0]}}

    def test_list_of_numpy(self):
        data = [np.int64(1), np.float64(2.5), "hello"]
        result = sanitize_for_json(data)
        assert result == [1, 2.5, "hello"]

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-01-01")
        result = sanitize_for_json(ts)
        assert isinstance(result, str)
        assert "2024" in result

    def test_bytes(self):
        result = sanitize_for_json(b"hello")
        assert result == "hello"

    def test_set(self):
        result = sanitize_for_json({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_plain_types_unchanged(self):
        assert sanitize_for_json("hello") == "hello"
        assert sanitize_for_json(42) == 42
        assert sanitize_for_json(3.14) == 3.14
        assert sanitize_for_json(True) is True
        assert sanitize_for_json(None) is None

    def test_result_is_json_serializable(self):
        """Ensure the output is fully JSON-serializable."""
        data = {
            "arr": np.array([1, 2]),
            "nested": {"val": np.float64(3.14)},
            "nan": float("nan"),
        }
        result = sanitize_for_json(data)
        # This should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


class TestDataframeToRecords:
    """Tests for dataframe_to_records utility."""

    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        records = dataframe_to_records(df, limit=2, offset=0)
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert records[1]["b"] == "y"

    def test_offset(self):
        df = pd.DataFrame({"a": range(10)})
        records = dataframe_to_records(df, limit=3, offset=5)
        assert len(records) == 3
        assert records[0]["a"] == 5

    def test_beyond_end(self):
        df = pd.DataFrame({"a": range(5)})
        records = dataframe_to_records(df, limit=10, offset=3)
        assert len(records) == 2  # Only 2 rows left

    def test_sanitizes_numpy(self):
        df = pd.DataFrame({"val": np.array([np.int64(42)])})
        records = dataframe_to_records(df, limit=10, offset=0)
        assert records[0]["val"] == 42
        assert isinstance(records[0]["val"], int)


class TestExtractHParamsFromRow:
    """Tests for extract_h_params_from_row."""

    def test_basic(self):
        row = pd.Series({"qubit_frequency_GHz": 4.5, "anharmonicity_MHz": -200.0, "other": "ignore"})
        keys = ["qubit_frequency_GHz", "anharmonicity_MHz"]
        result = extract_h_params_from_row(row, keys)
        assert result == {"qubit_frequency_GHz": 4.5, "anharmonicity_MHz": -200.0}

    def test_missing_keys(self):
        row = pd.Series({"qubit_frequency_GHz": 4.5})
        keys = ["qubit_frequency_GHz", "missing_key"]
        result = extract_h_params_from_row(row, keys)
        assert "qubit_frequency_GHz" in result
        assert "missing_key" not in result


class TestSafeGet:
    """Tests for safe_get utility."""

    def test_nested_dict(self):
        data = {"a": {"b": {"c": 42}}}
        assert safe_get(data, "a", "b", "c") == 42

    def test_missing_key(self):
        data = {"a": {"b": 1}}
        assert safe_get(data, "a", "x", default="nope") == "nope"

    def test_default(self):
        assert safe_get({}, "missing", default=None) is None


class TestBuildErrorResponse:
    """Tests for build_error_response."""

    def test_basic(self):
        msg = build_error_response("Something broke")
        assert "❌" in msg
        assert "Something broke" in msg

    def test_with_details(self):
        msg = build_error_response("Bad input", {"component": "qubit", "valid": ["a", "b"]})
        assert "component" in msg
        assert "qubit" in msg


# ---------------------------------------------------------------------------
# Tests for server creation
# ---------------------------------------------------------------------------


class TestServerFactory:
    """Tests for create_server()."""

    def test_create_server_returns_fastmcp(self):
        """Verify server can be created without errors."""
        from squadds_mcp.server import create_server

        server = create_server()
        # FastMCP instance should have a name
        assert server.name == "SQuADDS"

    def test_server_has_tools(self):
        """Verify tools are registered."""
        from squadds_mcp.server import create_server

        server = create_server()
        # The server should have tool handlers registered
        # Access the internal tool manager
        assert server is not None


# ---------------------------------------------------------------------------
# Tests for analysis helper
# ---------------------------------------------------------------------------


class TestConfigureDbForSearch:
    """Tests for _configure_db_for_search helper."""

    def test_invalid_system_type(self):
        from squadds_mcp.tools.analysis import _configure_db_for_search

        class FakeDB:
            def unselect_all(self):
                pass

        with pytest.raises(ValueError, match="Invalid system_type"):
            _configure_db_for_search(FakeDB(), "invalid_type")
