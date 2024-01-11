"""
========================================================================================================================
This file contains utility functions for the simulation.
========================================================================================================================
"""
import json
from collections import OrderedDict
from datetime import datetime

import numpy as np
import qiskit_metal as metal
import scqubits as scq
from pyaedt import Hfss
from qiskit_metal import Dict, MetalGUI, designs, draw
from qiskit_metal.qlibrary.core import QComponent
from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import \
    CapNInterdigitalTee
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.couplers.line_tee import LineTee
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
from qiskit_metal.qlibrary.tlines.anchored_path import RouteAnchors
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.mixed_path import RouteMixed
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from qiskit_metal.toolbox_metal import math_and_overrides

from squadds.components.claw_coupler import TransmonClaw
from squadds.components.coupled_systems import QubitCavity


def getMeshScreenshot(projectname, designname, solutiontype="Eigenmode"):
    """
    Get a screenshot of the mesh for a given project, design, and solution type.

    Parameters:
        projectname (str): The name of the project.
        designname (str): The name of the design.
        solutiontype (str, optional): The type of solution. Defaults to "Eigenmode".

    Raises:
        NotImplementedError: This function is not implemented yet.
    """
    raise NotImplementedError()

def generate_bbox(component: QComponent) -> Dict[str, float]:
    """
    Generates a bounding box dictionary from a given QComponent.

    Parameters:
    component (QComponent): The component for which to generate a bounding box.

    Returns:
    Dict[str, float]: A dictionary representing the bounding box with keys 'min_x', 'max_x', 'min_y', 'max_y'.
    """
    bounds = component.qgeometry_bounds()
    bbox = {
        'min_x': bounds[0],
        'max_x': bounds[2],
        'min_y': bounds[1],
        'max_y': bounds[3]
    }
    return bbox

def setMaterialProperties(projectname,designname,solutiontype="Eigenmode"):
    """
    Interfaces with ANSYS via pyEPR for more custom automation.
    1. Connects to ANSYS.
    2. Changes Silicon permittivity to 11.45, representing ultra cold silicon.
    3. Deletes any preexisting setups.

    Parameters:
        projectname (str): The name of the project.
        designname (str): The name of the design.
        solutiontype (str, optional): The type of solution. Defaults to "Eigenmode".
    """

    aedt = Hfss(projectname=projectname, 
                designname=designname, 
                solution_type=solutiontype,
                new_desktop_session=False, 
                close_on_exit=False)


    ultra_cold_silicon(aedt)
    delete_old_setups(aedt)

    aedt.release_desktop(close_projects=False, close_desktop=False)

def ultra_cold_silicon(aedt):
    """Change silicon properties to ultra cold silicon

    Args:
        aedt (pyAEDT Desktop obj)
    """
    materials = aedt.materials
    silicon = materials.checkifmaterialexists('silicon')
    silicon.permittivity = 11.45
    silicon.dielectric_loss_tangent = 1E-7

def delete_old_setups(aedt):
    """Delete old setups

    Args:
        aedt (pyAEDT Desktop obj)
    """
    # Clear setups
    if len(aedt.setups) != 0:
        aedt.setups[0].delete()

def calculate_center_and_dimensions(bbox):
    """
    Calculate the center and dimensions from the bounding box.

    :param bbox: The bounding box dictionary with keys 'min_x', 'max_x', 'min_y', 'max_y'.
    :return: A tuple containing the center coordinates and dimensions.
    """
    center_x = (bbox['min_x'] + bbox['max_x']) / 2 
    center_y = (bbox['min_y'] + bbox['max_y']) / 2 
    center_z = 0

    x_size = bbox['max_x'] - bbox['min_x']  
    y_size = bbox['max_y'] - bbox['min_y']
    z_size = 0

    return (center_x, center_y, center_z), (x_size, y_size, z_size)

def get_freq(epra, test_hfss):
    """
    Analyze the simulation, plot the results, and report the frequencies, Q, and kappa.

    :param epra: The EPR analysis object.
    :param test_hfss: The HFSS object.
    """
    project_name = test_hfss.pinfo.project_name
    design_name = test_hfss.pinfo.design_name

    setMaterialProperties(project_name, design_name, solutiontype="Eigenmode")
    epra.sim._analyze()
    try:
        epra.sim.plot_convergences()
        epra.sim.save_screenshot()
        epra.sim.plot_fields('main')
        epra.sim.save_screenshot()
    except:
        print("couldn't generate plots.")
    f = epra.get_frequencies()

    freq = f.values[0][0] * 1e9
    print(f"freq = {round(freq/1e9, 3)} GHz")
    return freq

def get_freq_Q_kappa(epra, test_hfss):
    """
    Analyze the simulation, plot the results, and report the frequencies, Q, and kappa.

    :param epra: The EPR analysis object.
    :param test_hfss: The HFSS object.
    """
    project_name = test_hfss.pinfo.project_name
    design_name = test_hfss.pinfo.design_name

    setMaterialProperties(project_name, design_name, solutiontype="Eigenmode")
    epra.sim._analyze()
    # epra.sim.plot_convergences()
    epra.sim.save_screenshot()
    # epra.sim.plot_fields('main')
    # epra.sim.save_screenshot()
    f = epra.get_frequencies()

    freq = f.values[0][0] * 1e9
    Q = f.values[0][1]
    kappa = freq / Q
    print(f"freq = {round(freq/1e9, 3)} GHz")
    print(f"Q = {round(Q, 1)}")
    print(f"kappa = {round(kappa/1e6, 3)} MHz")
    return freq, Q, kappa

def mesh_objects(modeler, mesh_lengths):
    """
    Draw the rectangle in the Ansys modeler, update the model, and set the mesh based on the input dictionary.

    :param modeler: The modeler object.
    :param center: The center coordinates tuple.
    :param dimensions: The dimensions tuple.
    :param cpw: The cpw object.
    :param claw: The claw object.
    :param mesh_lengths: Dictionary containing mesh names, associated objects, and MaxLength values.
    """
    for mesh_name, mesh_info in mesh_lengths.items():
        modeler.mesh_length(mesh_name, mesh_info['objects'], MaxLength=mesh_info['MaxLength'])

def add_ground_strip_and_mesh(modeler, coupler, mesh_lengths):
    """
    Draw the rectangle in the Ansys modeler, update the model, and set the mesh based on the input dictionary.

    :param modeler: The modeler object.
    :param center: The center coordinates tuple.
    :param dimensions: The dimensions tuple.
    :param coupler: The coupler object.
    :param cpw: The cpw object.
    :param claw: The claw object.
    :param mesh_lengths: Dictionary containing mesh names, associated objects, and MaxLength values.
    """
    bounds = coupler.qgeometry_bounds()
    bbox = {'min_x': bounds[0], 'max_x': bounds[2], 'min_y': bounds[1], 'max_y': bounds[3]}
    center, dimensions = calculate_center_and_dimensions(bbox)
    gs = modeler.draw_rect_center(
        [coord * 1e-3 for coord in center],
        x_size=dimensions[0] * 1e-3,
        y_size=dimensions[1] * 1e-3,
        name='ground_strip'
    )

    modeler.intersect(["ground_strip", "ground_main_plane"], True)
    modeler.subtract("ground_main_plane", ["ground_strip"], True)
    modeler.assign_perfect_E(["ground_strip"])
    mesh_lengths.update({'mesh_ground_strip': {"objects": ["ground_strip"], "MaxLength": '4um'}})

    for mesh_name, mesh_info in mesh_lengths.items():
        modeler.mesh_length(mesh_name, mesh_info['objects'], MaxLength=mesh_info['MaxLength'])

def create_qubitcavity(opts, design):
    """
    Create a QubitCavity object.

    Args:
        opts (dict): Options for the QubitCavity object.
        design (str): Design name.

    Returns:
        QubitCavity: The created QubitCavity object.
    """
    qubitcavity = QubitCavity(design, "qubitcavity", options=opts)
    return qubitcavity

def create_claw(opts, cpw_length, design):
    """
    Create a TransmonClaw object with the given options, cpw_length, and design.

    Args:
        opts (dict): A dictionary of options for the TransmonClaw object.
        cpw_length (int): The length of the cpw.
        design (str): The design name.

    Returns:
        TransmonClaw: The created TransmonClaw object.
    """
    opts["orientation"] = "-90"
    opts["pos_x"] = "-1500um" if cpw_length > 2500 else "-1000um"
    claw = TransmonClaw(design, 'claw', options=opts)
    return claw

def create_coupler(opts, design):
    """
    Create a coupler based on the given options and design.

    Args:
        opts (dict): A dictionary containing the options for the coupler.
        design: The design object.

    Returns:
        The created coupler object.
    """
    opts["orientation"] = "-90"
    cplr = CapNInterdigitalTee(design, 'cplr', options = opts) if "finger_count" in opts.keys() else CoupledLineTee(design, 'cplr', options = opts)
    return cplr

def create_cpw(opts, cplr, design):
    """
    Create a coplanar waveguide (CPW) based on the given options, coupler, and design.

    Args:
        opts (dict): Options for creating the CPW.
        cplr (Coupler): Coupler object used for creating the CPW.
        design (Design): Design object used for creating the CPW.

    Returns:
        RouteMeander: The created coplanar waveguide (CPW).
    """
    adj_distance = int("".join(filter(str.isdigit, cplr.options["coupling_length"]))) if int("".join(filter(str.isdigit, cplr.options["coupling_length"]))) > 150 else 0
    jogs = OrderedDict()
    jogs[0] = ["R90", f'{adj_distance/(1.5)}um']
    opts.update({"lead" : Dict(
                            start_straight = "100um",
                            end_straight = "50um",
                            
                            start_jogged_extension = jogs
                            )})
    opts.update({"pin_inputs" : Dict(start_pin = Dict(component = cplr.name,
                                                    pin = 'second_end'),
                                   end_pin = Dict(component = 'claw',
                                                  pin = 'readout'))})
    opts.update({"meander" : Dict(
                                spacing = "100um",
                                asymmetry = f'{adj_distance/(3)}um' # need this to make CPW asymmetry half of the coupling length
                                )})                                 # if not, sharp kinks occur in CPW :(
    cpw = RouteMeander(design, 'cpw', options = opts)
    return cpw

def save_simulation_data_to_json(data, filename):
    """
    Save simulation data to a JSON file.

    Args:
        data (dict): The simulation data to be saved.
        filename (str): The name of the file to save the data to.

    Returns:
        None
    """
    filename = f"{filename}.json"
    
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def chunk_sweep_options(sweep_opts, N):
    """
    Divide the sweep options into multiple chunks based on the number of computers.

    Args:
        sweep_opts (dict): The sweep options dictionary.
        N (int): The number of computers to divide the sweep options into.

    Returns:
        list: A list of dictionaries, each containing a chunk of the sweep options.
    """
    # Extract claw_lengths and total_lengths from sweep_opts
    claw_lengths = sweep_opts['claw_opts']['connection_pads']['readout']['claw_length']
    total_lengths = sweep_opts['cpw_opts']['total_length']

    # Determine the number of claw_lengths to be assigned to each chunk
    base_chunk_size = len(claw_lengths) // N
    remainder = len(claw_lengths) % N

    chunks = []
    start_idx = 0
    for i in range(N):
        # Calculate chunk size for this computer
        chunk_size = base_chunk_size + (1 if i < remainder else 0)

        # Slice the claw_lengths for this chunk
        claw_length_chunk = claw_lengths[start_idx:start_idx + chunk_size]

        # Each chunk gets a copy of the full total_lengths list
        new_sweep_opts = {
            'claw_opts': {
                'connection_pads': {
                    'readout': sweep_opts['claw_opts']['connection_pads']['readout'].copy()
                }
            },
            'cpw_opts': sweep_opts['cpw_opts'].copy(),
            'cplr_opts': sweep_opts['cplr_opts'].copy()
        }

        new_sweep_opts['claw_opts']['connection_pads']['readout']['claw_length'] = claw_length_chunk
        new_sweep_opts['cpw_opts']['total_length'] = total_lengths

        chunks.append(new_sweep_opts)

        # Update the start index for the next chunk
        start_idx += chunk_size

    return chunks

def find_a_fq(C_g, C_B, Lj):
    """
    Calculate the anharmonicity and frequency of a transmon qubit.

    Args:
        C_g (float): Gate capacitance in Farads.
        C_B (float): Bias capacitance in Farads.
        Lj (float): Josephson inductance in Henries.

    Returns:
        tuple: A tuple containing the anharmonicity (a) in linear MHz and the frequency (f_q) in linear GHz.
    """
    # Constants
    e = 1.602e-19  # elementary charge in C
    hbar = 1.054e-34  # reduced Planck constant in Js
    Z_0 = 50  # in Ohms

    C_Sigma = C_g + C_B # + 1.5e-15
    EJ = ((hbar / 2 / e) ** 2) / Lj * (1.5092e24) # 1J = 1.5092e24 GHz
    EC = e**2/(2*C_Sigma) * (1.5092e24) # 1J = 1.5092e24 GHz

    transmon = scq.Transmon(EJ=EJ,
                            EC=EC,
                            ng = 0,
                            ncut = 30)

    a = transmon.anharmonicity() * 1000 # linear MHz
    # g = ((C_g / C_Sigma) * omega_r * np.sqrt(N * Z_0 * e**2 / (hbar * np.pi) )* (EJ/(8*EC))**(1/4)) / 1E6 / (2 * np.pi) # linear MHz
    f_q = transmon.E01() # Linear GHz
    
    return a, f_q

def find_g_a_fq(C_g, C_B, f_r, Lj, N):
    """
    Calculate the values of g, a, and f_q for a transmon qubit.

    Args:
        C_g (float): Capacitance of the gate in Farads.
        C_B (float): Capacitance of the bias in Farads.
        f_r (float): Resonance frequency of the resonator in Hz.
        Lj (float): Josephson inductance in Henries.
        N (int): Number of photons in the resonator.

    Returns:
        tuple: A tuple containing the values of g, a, and f_q.
            - g (float): Coupling strength in MHz.
            - a (float): Anharmonicity in MHz.
            - f_q (float): Transition frequency in GHz.
    """
    # Constants
    e = 1.602e-19  # elementary charge in C
    hbar = 1.054e-34  # reduced Planck constant in Js
    Z_0 = 50  # in Ohms

    C_Sigma = C_g + C_B # + 1.5e-15
    omega_r = 2 * np.pi * f_r
    EJ = ((hbar / 2 / e) ** 2) / Lj * (1.5092e24) # 1J = 1.5092e24 GHz
    EC = e**2/(2*C_Sigma) * (1.5092e24) # 1J = 1.5092e24 GHz

    transmon = scq.Transmon(EJ=EJ,
                            EC=EC,
                            ng = 0,
                            ncut = 30)

    a = transmon.anharmonicity() * 1000 # linear MHz
    g = ((C_g / C_Sigma) * omega_r * np.sqrt(N * Z_0 * e**2 / (hbar * np.pi) )* (EJ/(8*EC))**(1/4)) / 1E6 / (2 * np.pi) # linear MHz
    f_q = transmon.E01() # Linear GHz
    
    return g, a, f_q