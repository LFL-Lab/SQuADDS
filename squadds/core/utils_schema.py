def get_type(value):
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list" if not value else get_type(value[0])
    return type(value).__name__.lower()


def validate_types(data_part, schema_part):
    for key, expected_type in schema_part.items():
        if isinstance(expected_type, dict):
            validate_types(data_part[key], expected_type)
        else:
            actual_type = get_type(data_part[key])
            if actual_type != expected_type:
                raise TypeError(f"Invalid type for {key}. Expected {expected_type}, got {actual_type}.")


def get_config_schema(entry):
    schema = {}
    for key, value in entry.items():
        if key == "sim_results":
            schema[key] = {field: get_type(field_value) for field, field_value in value.items()}
        elif key in ["sim_options", "design", "notes"] and isinstance(value, dict):
            schema[key] = {field: get_type(field_value) for field, field_value in value.items()}
        else:
            schema[key] = get_type(value)
    return schema


def get_schema(obj):
    if isinstance(obj, dict):
        return {key: "dict" if isinstance(value, dict) else get_schema(value) for key, value in obj.items() if key != "contributor"}
    if isinstance(obj, list):
        return "dict" if any(isinstance(elem, dict) for elem in obj) else type(obj[0]).__name__
    return type(obj).__name__


def get_entire_schema(obj):
    if isinstance(obj, dict):
        return {key: get_entire_schema(value) for key, value in obj.items() if key != "contributor"}
    if isinstance(obj, list):
        return [get_entire_schema(item) for item in obj][0] if obj else []
    return type(obj).__name__


def compare_schemas(data_schema, expected_schema, path=""):
    for key, data_type in data_schema.items():
        if key not in expected_schema:
            raise ValueError(f"Unexpected key '{path}{key}' found in data schema.")

        expected_type = expected_schema[key]
        if isinstance(expected_type, dict):
            if not isinstance(data_type, dict):
                raise ValueError(f"Type mismatch for '{path}{key}'. Expected a dict, Got: {get_type(data_type)}")
            compare_schemas(data_type, expected_type, path + key + ".")
        elif get_type(data_type) != expected_type:
            if expected_type == "float" and get_type(data_type) == "str":
                continue
            raise ValueError(
                f"Type mismatch for '{path}{key}'. Expected: {expected_type}, Got: {get_type(data_type)}"
            )
