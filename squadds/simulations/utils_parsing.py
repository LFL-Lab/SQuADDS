import re


def string_to_float(string):
    return float(string[:-2])


def extract_value(dictionary, key):
    if key in dictionary:
        return dictionary[key]
    for value in dictionary.values():
        if isinstance(value, dict):
            result = extract_value(value, key)
            if result is not None:
                return result
    return None


def convert_str_to_float(value):
    return float(value[:-2])


def extract_number(string):
    return float(re.sub(r"[^\d.]", "", string))


def unpack(parent_key, parent_value, delimiter=","):
    if isinstance(parent_value, dict):
        return [(parent_key + delimiter + key, value) for key, value in parent_value.items()]
    return [(parent_key, parent_value)]


def flatten_dict(dictionary_, delimiter=","):
    while True:
        dictionary_ = dict(
            item for group in [unpack(key, value, delimiter) for key, value in dictionary_.items()] for item in group
        )
        if all(not isinstance(value, dict) for value in dictionary_.values()):
            break
    return dictionary_
