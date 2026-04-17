UNSELECT_ATTR_MAP = {
    "component": "selected_component",
    "component_name": "selected_component_name",
    "data_type": "selected_data_type",
    "qubit": "selected_qubit",
    "cavity_claw": "selected_cavity",
    "coupler": "selected_coupler",
    "system": "selected_system",
}

SELECTION_ATTRS = (
    "selected_component_name",
    "selected_component",
    "selected_data_type",
    "selected_qubit",
    "selected_cavity",
    "selected_coupler",
    "selected_system",
    "selected_resonator_type",
)


def reset_selections(instance):
    """
    Clear the mutable selection state on an ``SQuADDS_DB`` instance.
    """
    for attr in SELECTION_ATTRS:
        setattr(instance, attr, None)


def format_selection_lines(
    selected_system,
    selected_component,
    selected_component_name,
    selected_data_type,
    selected_qubit,
    selected_cavity,
    selected_coupler,
    selected_resonator_type,
):
    """
    Return the printed selection lines for ``show_selections``.
    """
    if isinstance(selected_system, list):
        lines = [
            f"Selected qubit:  {selected_qubit}",
            f"Selected cavity:  {selected_cavity}",
            f"Selected coupler to feedline:  {selected_coupler}",
        ]
        if selected_resonator_type is not None:
            lines.append(f"Selected resonator type:  {selected_resonator_type}")
        lines.append(f"Selected system:  {selected_system}")
        return lines

    if isinstance(selected_system, str):
        lines = [
            f"Selected component:  {selected_component}",
            f"Selected component name:  {selected_component_name}",
            f"Selected data type:  {selected_data_type}",
            f"Selected system:  {selected_system}",
            f"Selected coupler:  {selected_coupler}",
        ]
        if selected_resonator_type is not None:
            lines.append(f"Selected resonator type:  {selected_resonator_type}")
        return lines

    return []


def update_target_param_keys(current_keys, selected_system, df, get_sim_results_keys_fn):
    """
    Apply the legacy target-parameter update rules.
    """
    if selected_system is None:
        raise UserWarning("No selected system df is created. Please check `self.selected_df`")

    if current_keys is None:
        updated = get_sim_results_keys_fn(df)
    elif isinstance(current_keys, list) and len(selected_system) == 2:
        updated = current_keys + get_sim_results_keys_fn(df)
    elif isinstance(current_keys, list) and len(selected_system) != 1:
        updated = get_sim_results_keys_fn(df)
    else:
        raise UserWarning("target_param_keys is not None or a list. Please check `self.target_param_keys`")

    return [key for key in updated if not key.startswith("unit")]


def get_unselect_attr_name(param):
    """
    Map the public ``unselect`` parameter to the instance attribute name.
    """
    return UNSELECT_ATTR_MAP.get(param)
