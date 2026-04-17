from squadds.database.hf_dataset_ops import build_dataset_name, ensure_dataset_repository, upload_dataset_files


def test_build_dataset_name_is_stable_for_fixed_inputs():
    dataset_name = build_dataset_name(
        ["qubit", "cavity_claw"],
        "cap_matrix",
        "simulated",
        "ansys",
        "Stanford",
        "Schuster",
        date="20260416",
    )

    assert dataset_name == "qubit-cavity_claw_cap_matrix_simulated_ansys_Stanford_Schuster_20260416_6dad1950"


def test_ensure_dataset_repository_calls_create_repo(capsys):
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(kwargs)

    ensure_dataset_repository(FakeApi(), "token-123", "example-dataset")
    output = capsys.readouterr().out

    assert calls == [{"repo_id": "example-dataset", "token": "token-123", "repo_type": "dataset"}]
    assert "Dataset repository example-dataset created." in output


def test_upload_dataset_files_uses_repo_basename_and_token(capsys):
    calls = []

    class FakeApi:
        def upload_file(self, **kwargs):
            calls.append(kwargs)

    upload_dataset_files(FakeApi(), "token-123", "example-dataset", ["/tmp/alpha.json", "/tmp/beta.parquet"])
    output = capsys.readouterr().out

    assert calls == [
        {
            "path_or_fileobj": "/tmp/alpha.json",
            "path_in_repo": "alpha.json",
            "repo_id": "example-dataset",
            "repo_type": "dataset",
            "token": "token-123",
        },
        {
            "path_or_fileobj": "/tmp/beta.parquet",
            "path_in_repo": "beta.parquet",
            "repo_id": "example-dataset",
            "repo_type": "dataset",
            "token": "token-123",
        },
    ]
    assert "Uploaded /tmp/alpha.json to example-dataset." in output
    assert "Uploaded /tmp/beta.parquet to example-dataset." in output
