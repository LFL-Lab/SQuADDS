import numpy as np


def float_to_string(value, units):
    return f"{value}{units}"


def string_to_float(string):
    return float(string[:-2])


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def convert_to_numeric(value):
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        if is_float(value):
            return float(value)
    return value


def convert_to_str(value: float, units: str):
    return f"{value} {units}"


def convert_list_to_str(lst):
    return [convert_to_str(item) for item in lst]


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(value) for value in obj]
    return obj
