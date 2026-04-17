from prettytable import PrettyTable

from squadds.simulations.utils_parsing import extract_value


def make_table(title, data):
    if title == "qubit":
        pars = [
            "cross_width",
            "cross_length",
            "cross_gap",
            "claw_cpw_length",
            "claw_cpw_width",
            "claw_gap",
            "claw_length",
            "claw_width",
            "ground_spacing",
        ]
    elif title == "cavity":
        pars = ["total_length"]
    elif title == "coupler":
        pars = ["coupling_length", "coupling_space"]
    elif title == "purcell_filter":
        pars = ["total_length", "cap_gap_ground", "finger_length", "cap_width", "cap_gap"]
    else:
        raise ValueError(f"Unsupported table title: {title}")

    table = PrettyTable()
    table.title = title
    table.field_names = ["param", "value"]
    for key in pars:
        table.add_row([key, extract_value(dictionary=data, key=key)])
    print(table)
