import pandas as pd
import pytest

from squadds.core.db_state import format_selection_lines, get_unselect_attr_name, reset_selections, update_target_param_keys


class DummyDB:
    def __init__(self):
        self.selected_component_name = "name"
        self.selected_component = "component"
        self.selected_data_type = "dtype"
        self.selected_qubit = "qubit"
        self.selected_cavity = "cavity"
        self.selected_coupler = "coupler"
        self.selected_system = ["qubit", "cavity_claw"]
        self.selected_resonator_type = "half"


def test_reset_selections_clears_all_selection_attributes():
    db = DummyDB()

    reset_selections(db)

    assert db.selected_component_name is None
    assert db.selected_component is None
    assert db.selected_data_type is None
    assert db.selected_qubit is None
    assert db.selected_cavity is None
    assert db.selected_coupler is None
    assert db.selected_system is None
    assert db.selected_resonator_type is None


def test_format_selection_lines_for_multi_component_system():
    lines = format_selection_lines(
        ["qubit", "cavity_claw"],
        None,
        None,
        None,
        "TransmonCross",
        "RouteMeander",
        "NCap",
        "half",
    )

    assert lines == [
        "Selected qubit:  TransmonCross",
        "Selected cavity:  RouteMeander",
        "Selected coupler to feedline:  NCap",
        "Selected resonator type:  half",
        "Selected system:  ['qubit', 'cavity_claw']",
    ]


def test_format_selection_lines_for_single_component_system():
    lines = format_selection_lines("qubit", "qubit", "TransmonCross", "cap_matrix", None, None, None, None)

    assert lines == [
        "Selected component:  qubit",
        "Selected component name:  TransmonCross",
        "Selected data type:  cap_matrix",
        "Selected system:  qubit",
        "Selected coupler:  None",
    ]


def test_update_target_param_keys_handles_initial_and_multi_system_updates():
    df = pd.DataFrame({"sim_results": [{"g_MHz": 55, "unit_g_MHz": "MHz"}]})
    getter = lambda frame: ["g_MHz", "unit_g_MHz"]

    assert update_target_param_keys(None, "qubit", df, getter) == ["g_MHz"]
    assert update_target_param_keys(["existing"], ["qubit", "cavity_claw"], df, getter) == ["existing", "g_MHz"]


def test_update_target_param_keys_raises_when_selected_system_missing():
    with pytest.raises(UserWarning, match="No selected system df is created"):
        update_target_param_keys(None, None, pd.DataFrame(), lambda frame: [])


def test_get_unselect_attr_name_matches_public_arguments():
    assert get_unselect_attr_name("component_name") == "selected_component_name"
    assert get_unselect_attr_name("unknown") is None
