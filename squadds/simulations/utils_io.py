import json
import os


def save_simulation_data_to_json(data, filename):
    with open(f"{filename}.json", "w") as outfile:
        json.dump(data, outfile, indent=4)


def read_json_files(directory):
    json_files = [file for file in os.listdir(directory) if file.endswith(".json")]
    data = []
    for file in json_files:
        with open(os.path.join(directory, file)) as json_file:
            data.append(json.load(json_file))
    return data
