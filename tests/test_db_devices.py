from squadds.core.db_devices import (
    SIMULATION_CONTRIBUTOR_KEYS,
    build_measured_device_records,
    build_measured_device_rows,
    build_recipe_rows,
    build_reference_device_records,
    collect_all_simulation_contributors,
    find_device_contributor_info,
    find_reference_device_info,
    find_simulation_results_for_device,
    unique_contributor_records,
)


def _measured_dataset():
    return {
        "contrib_info": [
            {
                "name": "device-a",
                "group": "LFL",
                "measured_by": ["alice", "bob"],
                "PI": "pi-a",
                "institution": "usc",
                "uploader": "shanto",
                "foundry": "fab-1",
            }
        ],
        "sim_results": [{"qubit-TransmonCross-cap_matrix": {"g_MHz": 55}}],
        "design_code": ["https://github.com/example/device-a"],
        "paper_link": ["paper-a"],
        "image": ["image-a"],
        "foundry": ["fab-1"],
        "fabrication_recipe": ["recipe-a"],
        "substrate": ["Si"],
        "materials": ["Nb"],
        "junction_style": ["bridge"],
        "junction_material": ["Al"],
    }


def test_unique_contributor_records_preserves_first_seen_order():
    records = unique_contributor_records(
        [
            {"uploader": "a", "PI": "p1", "group": "g1", "institution": "i1"},
            {"uploader": "a", "PI": "p1", "group": "g1", "institution": "i1"},
            {"uploader": "b", "PI": "p2", "group": "g2", "institution": "i2"},
        ],
        SIMULATION_CONTRIBUTOR_KEYS,
    )

    assert records == [
        {"uploader": "a", "PI": "p1", "group": "g1", "institution": "i1"},
        {"uploader": "b", "PI": "p2", "group": "g2", "institution": "i2"},
    ]


def test_collect_all_simulation_contributors_adds_config_column():
    datasets = {
        "cfg-a": {"train": {"contributor": [{"uploader": "a", "PI": "p1", "group": "g1", "institution": "i1"}]}},
        "cfg-b": {"train": {"contributor": [{"uploader": "b", "PI": "p2", "group": "g2", "institution": "i2"}]}},
    }

    rows = collect_all_simulation_contributors(
        ["cfg-a", "cfg-b"],
        "repo",
        load_dataset_fn=lambda repo_name, config: datasets[config],
    )

    assert rows == [
        {"uploader": "a", "PI": "p1", "group": "g1", "institution": "i1", "Config": "cfg-a"},
        {"uploader": "b", "PI": "p2", "group": "g2", "institution": "i2", "Config": "cfg-b"},
    ]


def test_build_measured_device_records_matches_legacy_dataframe_payload():
    records = build_measured_device_records(_measured_dataset())

    assert records == [
        {
            "Name": "device-a",
            "Design Code": "https://github.com/example/device-a",
            "Paper Link": "paper-a",
            "Image": "image-a",
            "Foundry": "fab-1",
            "Substrate": "Si",
            "Materials": "Nb",
            "Junction Style": "bridge",
            "Junction Materials": "Al",
        }
    ]


def test_build_measured_device_rows_matches_printed_subset():
    rows = build_measured_device_rows(_measured_dataset())

    assert rows == [
        {
            "Name": "device-a",
            "Design Code": "https://github.com/example/device-a",
            "Paper Link": "paper-a",
            "Image": "image-a",
            "Foundry": "fab-1",
            "Fabrication Recipe": "recipe-a",
        }
    ]


def test_find_simulation_results_for_device_returns_matching_mapping():
    assert find_simulation_results_for_device(_measured_dataset(), "device-a") == {
        "qubit-TransmonCross-cap_matrix": {"g_MHz": 55}
    }


def test_find_device_contributor_info_extracts_matching_reference_device():
    info = find_device_contributor_info(_measured_dataset(), "qubit-TransmonCross-cap_matrix")

    assert info == {
        "Foundry": "fab-1",
        "PI": "pi-a",
        "Group": "LFL",
        "Institution": "usc",
        "Measured By": "alice, bob",
        "Reference Device Name": "device-a",
        "Uploader": "shanto",
    }


def test_find_reference_device_info_combines_metadata_and_contrib_info():
    info = find_reference_device_info(_measured_dataset(), "qubit-TransmonCross-cap_matrix")

    assert info["Design Code"] == "https://github.com/example/device-a"
    assert info["Paper Link"] == "paper-a"
    assert info["name"] == "device-a"


def test_build_recipe_rows_appends_fabrication_tree_suffix():
    assert build_recipe_rows(_measured_dataset(), "device-a") == [
        ["Foundry", "fab-1"],
        ["Fabublox Link", "recipe-a"],
        ["Fabrication Recipe Links", "https://github.com/example/device-a/tree/main/Fabrication"],
    ]


def test_build_reference_device_records_uses_lookup_callback():
    dataset = _measured_dataset()
    rows = build_reference_device_records(dataset, lambda device_name: {"device": device_name})

    assert rows == [
        {
            "name": "device-a",
            "group": "LFL",
            "measured_by": ["alice", "bob"],
            "simulations": {"device": "device-a"},
        }
    ]
