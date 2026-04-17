from squadds.core.db_catalog import (
    extract_supported_component_names,
    extract_supported_components,
    extract_supported_data_types,
    filter_simulation_config_names,
    get_component_names_for_component,
)


def test_filter_simulation_config_names_preserves_legacy_two_hyphen_filter():
    configs = [
        "qubit-TransmonCross-cap_matrix",
        "measured_device_database",
        "half-wave-cavity_df.parquet",
        "cavity_claw-RouteMeander-eigenmode",
    ]

    assert filter_simulation_config_names(configs) == [
        "qubit-TransmonCross-cap_matrix",
        "half-wave-cavity_df.parquet",
        "cavity_claw-RouteMeander-eigenmode",
    ]


def test_extract_supported_components_preserves_legacy_duplicates():
    configs = ["qubit-TransmonCross-cap_matrix", "cavity_claw-RouteMeander-eigenmode"]

    assert extract_supported_components(configs) == [
        "qubit",
        "qubit",
        "cavity_claw",
        "cavity_claw",
    ]


def test_extract_supported_component_names_preserves_legacy_duplicates():
    configs = ["qubit-TransmonCross-cap_matrix"]

    assert extract_supported_component_names(configs) == ["TransmonCross", "TransmonCross"]


def test_extract_supported_data_types_preserves_legacy_duplicates():
    configs = ["qubit-TransmonCross-cap_matrix"]

    assert extract_supported_data_types(configs) == ["cap_matrix", "cap_matrix"]


def test_get_component_names_for_component_appends_clt_alias():
    configs = ["qubit-TransmonCross-cap_matrix"]

    assert get_component_names_for_component(configs, "qubit") == ["TransmonCross", "CLT"]
