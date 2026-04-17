from squadds.core.db_loader import build_dataset_config, validate_dataset_request


def test_validate_dataset_request_requires_component_and_component_name():
    validation = validate_dataset_request(
        None,
        "TransmonCross",
        "cap_matrix",
        ["qubit"],
        ["TransmonCross"],
        ["cap_matrix"],
    )

    assert not validation.is_valid
    assert validation.message == "Both system and component name must be defined."
    assert validation.options is None


def test_validate_dataset_request_requires_data_type():
    validation = validate_dataset_request(
        "qubit",
        "TransmonCross",
        None,
        ["qubit"],
        ["TransmonCross"],
        ["cap_matrix"],
    )

    assert not validation.is_valid
    assert validation.message == "Please specify a data type."
    assert validation.options is None


def test_validate_dataset_request_returns_supported_component_options():
    validation = validate_dataset_request(
        "cavity_claw",
        "TransmonCross",
        "cap_matrix",
        ["qubit"],
        ["TransmonCross"],
        ["cap_matrix"],
    )

    assert not validation.is_valid
    assert validation.message == "Component not supported. Available components are:"
    assert validation.options == ["qubit"]


def test_validate_dataset_request_returns_supported_component_name_options():
    validation = validate_dataset_request(
        "qubit",
        "RouteMeander",
        "cap_matrix",
        ["qubit"],
        ["TransmonCross"],
        ["cap_matrix"],
    )

    assert not validation.is_valid
    assert validation.message == "Component name not supported. Available component names are:"
    assert validation.options == ["TransmonCross"]


def test_validate_dataset_request_returns_supported_data_type_options():
    validation = validate_dataset_request(
        "qubit",
        "TransmonCross",
        "eigenmode",
        ["qubit"],
        ["TransmonCross"],
        ["cap_matrix"],
    )

    assert not validation.is_valid
    assert validation.message == "Data type not supported. Available data types are:"
    assert validation.options == ["cap_matrix"]


def test_validate_dataset_request_accepts_supported_triplet():
    validation = validate_dataset_request(
        "qubit",
        "TransmonCross",
        "cap_matrix",
        ["qubit"],
        ["TransmonCross"],
        ["cap_matrix"],
    )

    assert validation.is_valid
    assert validation.message is None
    assert validation.options is None


def test_build_dataset_config_formats_triplet():
    assert build_dataset_config("qubit", "TransmonCross", "cap_matrix") == "qubit-TransmonCross-cap_matrix"
