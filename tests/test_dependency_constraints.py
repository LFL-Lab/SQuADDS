from pathlib import Path


def test_pandas_is_capped_below_three_until_pyepr_matrix_parser_is_updated():
    pyproject = Path("pyproject.toml").read_text()
    pandas_specs = [line.strip().strip('",') for line in pyproject.splitlines() if '"pandas' in line]

    assert pandas_specs == ["pandas>=1.5,<3.0"]
