import pytest

from squadds.database.contributor_schema import validate_design_payload, validate_sim_setup_payload


def test_validate_sim_setup_payload_accepts_matching_schema():
    validate_sim_setup_payload(
        {"setup": {}, "simulator": ""},
        {"setup": "dict", "simulator": "str"},
        lambda v: "dict" if isinstance(v, dict) else type(v).__name__.lower(),
    )


def test_validate_sim_setup_payload_rejects_missing_key():
    with pytest.raises(ValueError, match="Missing required simulation setup option: simulator"):
        validate_sim_setup_payload(
            {"setup": {}},
            {"setup": "dict", "simulator": "str"},
            lambda v: "dict" if isinstance(v, dict) else type(v).__name__.lower(),
        )


def test_validate_design_payload_accepts_optional_design_tool():
    validate_design_payload(
        {"design_options": {"cross_length": "120um"}},
        "dict",
        lambda value: "dict" if isinstance(value, dict) else type(value).__name__.lower(),
        require_design_tool=False,
    )


def test_validate_design_payload_requires_string_design_tool_in_legacy_mode():
    with pytest.raises(TypeError, match="Incorrect type for design tool. Expected 'str'"):
        validate_design_payload(
            {"design_options": {"cross_length": "120um"}, "design_tool": None},
            "dict",
            lambda value: "dict" if isinstance(value, dict) else type(value).__name__.lower(),
            require_design_tool=True,
        )
