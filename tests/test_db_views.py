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


def test_build_dataset_rows_preserves_first_seen_order_when_deduplicating():
    rows = build_dataset_rows(
        components=["qubit", "cavity_claw", "qubit"],
        component_names=["TransmonCross", "RouteMeander", "TransmonCross"],
        data_types=["cap_matrix", "eigenmode", "cap_matrix"],
    )

    assert rows[0][:3] == ["qubit", "TransmonCross", "cap_matrix"]
    assert rows[1][:3] == ["cavity_claw", "RouteMeander", "eigenmode"]


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
