from squadds.core.db_views import build_dataset_rows, describe_dataset


def test_build_dataset_rows_matches_legacy_zip_and_dedup_behavior():
    rows = build_dataset_rows(
        components=["qubit", "qubit"],
        component_names=["TransmonCross", "TransmonCross"],
        data_types=["cap_matrix", "cap_matrix"],
    )

    assert rows == [
        [
            "qubit",
            "TransmonCross",
            "cap_matrix",
            "https://github.com/LFL-Lab/SQuADDS/tree/master/docs/_static/images/TransmonCross.png",
        ]
    ]


def test_describe_dataset_collects_printed_metadata_fields():
    class FakeDataset:
        features = {"value": "float"}
        description = "desc"
        citation = "cite"
        homepage = "home"
        license = "mit"
        size_in_bytes = 42

    assert describe_dataset(FakeDataset()) == {
        "features": {"value": "float"},
        "description": "desc",
        "citation": "cite",
        "homepage": "home",
        "license": "mit",
        "size_in_bytes": 42,
    }
