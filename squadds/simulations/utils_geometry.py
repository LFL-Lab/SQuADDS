from qiskit_metal import Dict
from qiskit_metal.qlibrary.core import QComponent


def get_cavity_claw_options_keys(cavity_dict):
    cpw_opts_key = None
    cplr_opts_key = None
    for key in cavity_dict.keys():
        if key.startswith("cpw"):
            cpw_opts_key = key
        elif key.startswith("cplr"):
            cplr_opts_key = key

    return cpw_opts_key, cplr_opts_key


def generate_bbox(component: QComponent) -> Dict[str, float]:
    bounds = component.qgeometry_bounds()
    return {"min_x": bounds[0], "max_x": bounds[2], "min_y": bounds[1], "max_y": bounds[3]}


def calculate_center_and_dimensions(bbox):
    center_x = (bbox["min_x"] + bbox["max_x"]) / 2
    center_y = (bbox["min_y"] + bbox["max_y"]) / 2
    center_z = 0
    x_size = bbox["max_x"] - bbox["min_x"]
    y_size = bbox["max_y"] - bbox["min_y"]
    z_size = 0
    return (center_x, center_y, center_z), (x_size, y_size, z_size)
