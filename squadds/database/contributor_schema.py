def validate_sim_setup_payload(sim_setup, sim_setup_schema, get_type_fn):
    """
    Validate simulation setup payloads against the config schema.
    """
    if not isinstance(sim_setup, dict):
        raise ValueError("Simulation setup options must be provided as a dictionary.")

    for key, expected_type in sim_setup_schema.items():
        if key not in sim_setup:
            raise ValueError(f"Missing required simulation setup option: {key}")
        if get_type_fn(sim_setup[key]) != expected_type:
            raise TypeError(f"Incorrect type for {key}. Expected {expected_type}, got {get_type_fn(sim_setup[key])}.")


def validate_design_payload(design, design_options_schema, get_type_fn, require_design_tool=False):
    """
    Validate design payloads against the config schema.
    """
    if not isinstance(design, dict):
        raise ValueError("Design must be provided as a dictionary.")

    design_options = design.get("design_options", {})
    design_tool = design.get("design_tool")

    if get_type_fn(design_options) != design_options_schema:
        raise TypeError(
            f"Incorrect type for design options. Expected {design_options_schema}, got {get_type_fn(design_options)}."
        )

    if require_design_tool:
        if get_type_fn(design_tool) != "str":
            raise TypeError(f"Incorrect type for design tool. Expected 'str', got {get_type_fn(design_tool)}.")
    elif design_tool and get_type_fn(design_tool) != "str":
        raise TypeError(f"Incorrect type for design tool. Expected 'str', got {get_type_fn(design_tool)}.")
