import pytest

from squadds.simulations.utils_display import make_table


def test_make_table_rejects_unknown_title():
    with pytest.raises(ValueError, match="Unsupported table title"):
        make_table("unknown", {"anything": 1})
