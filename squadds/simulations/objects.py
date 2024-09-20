"""
========================================================================================================================
SimulationConfig
========================================================================================================================
"""

from datetime import datetime

from qiskit_metal.analyses.quantization import EPRanalysis, LOManalysis

from .sweeper_helperfunctions import extract_QSweep_parameters
from .utils import *


class SimulationConfig:
    """
    Represents the configuration for a simulation.

    Args:
        design_name (str): The name of the design.
        renderer_type (str): The type of renderer.
        sim_type (str): The type of simulation.
        setup_name (str): The name of the setup.
        max_passes (int): The maximum number of passes.
        max_delta_f (float): The maximum delta frequency.
        min_converged_passes (int): The minimum number of converged passes.
        Lj (float): The value of Lj.
        Cj (float): The value of Cj.
    """

    def __init__(self, design_name="CavitySweep", renderer_type="hfss", sim_type="eigenmode",
                 setup_name="Setup", max_passes=49, max_delta_f=0.05, min_converged_passes=2, Lj=0, Cj=0):
        """
        Initialize the Simulation object.

        Args:
            design_name (str): The name of the design.
            renderer_type (str): The type of renderer to be used.
            sim_type (str): The type of simulation.
            setup_name (str): The name of the setup.
            max_passes (int): The maximum number of passes.
            max_delta_f (float): The maximum change in frequency.
            min_converged_passes (int): The minimum number of converged passes.
            Lj (float): The value of inductance.
            Cj (float): The value of capacitance.
        """
        self.design_name = design_name
        self.renderer_type = renderer_type
        self.sim_type = sim_type
        self.setup_name = setup_name
        self.max_passes = max_passes
        self.max_delta_f = max_delta_f
        self.min_converged_passes = min_converged_passes
        self.Lj = Lj
        self.Cj = Cj

def simulate_whole_device(design, device_dict, eigenmode_options, LOM_options, open_gui=False):
    """
    Simulates the whole device by running eigenmode and LOM simulations.

    Args:
        design (metal.designs.design_planar.DesignPlanar): The design object.
        cross_dict (dict): Dictionary containing qubit options.
        cavity_dict (dict): Dictionary containing cavity options.
        LOM_options (dict): Dictionary containing LOM setup options.
        eigenmode_options (dict): Dictionary containing eigenmode setup options.
        open_gui (bool, optional): If True, the Metal GUI is opened. Default is False.

    Returns:
        tuple: A tuple containing the simulation results, LOM analysis object, and eigenmode analysis object.
    """
    
    cross_dict = device_dict["design_options_qubit"]
    cavity_dict = device_dict["design_options_cavity_claw"]
    
    design.delete_all_components()
    if device_dict["coupler_type"].upper() == "CLT":
        emode_df, emode_obj = run_eigenmode(design, cavity_dict, eigenmode_options)
        lom_df, lom_obj = run_xmon_LOM(design, cross_dict, LOM_options)
        data = get_sim_results(emode_df = emode_df, lom_df = lom_df)

    elif device_dict["coupler_type"].lower() == "ncap":
        emode_df, emode_obj = run_eigenmode(design, cavity_dict, eigenmode_options)
        ncap_lom_df, ncap_lom_obj = run_capn_LOM(design, cavity_dict["cplr_opts"], LOM_options)
        lom_df, lom_obj = run_xmon_LOM(design, cross_dict, LOM_options)
        data = get_sim_results(emode_df = emode_df, lom_df = lom_df, ncap_lom_df=ncap_lom_df)

    device_dict_format = Dict(
        cavity_options = Dict(
            coupler_type = device_dict["coupler_type"],
            coupler_options = cavity_dict["cplr_opts"],
            cpw_opts = Dict (
                left_options = cavity_dict["cpw_opts"],
            )
            
        ),
        qubit_options = cross_dict
    )

    design = metal.designs.design_planar.DesignPlanar()
    if open_gui:
        gui = metal.MetalGUI(design)
    else:
        pass
    design.overwrite_enabled = True
    QC = create_qubitcavity(device_dict_format, design)
    
    return_df = dict(
        sim_options = dict(
            setup = dict(
                eigenmode_setup = eigenmode_options,
                LOM_setup = LOM_options
            ),
            renderer_options = dict(
                eigenmode_renderer_options = emode_obj.sim.renderer.options,
                lom_renderer_options = lom_obj.sim.renderer.options
            ),
            simulator = "Ansys HFSS"
        ),
        sim_results = data,
        design = dict(
            design_options = device_dict_format
        )
    )

    return return_df, lom_obj, emode_obj

def simulate_single_design(design, device_dict, emode_options={}, lom_options={}, coupler_type="CLT"):
    """
    Simulates a single design using the provided parameters.

    Args:
        design (Design): The design object representing the design.
        device_dict (dict): A dictionary containing device options.
        emode_options (dict): A dictionary containing the eigenmode simulation options.
        lom_options (dict): A dictionary containing the LOM simulation options.
        coupler_type (str): The type of coupler to be used.
        sim_options (dict): A dictionary containing simulation options.

    Returns:
        dict or tuple: The simulation results. If eigenmode simulation is performed, returns a dictionary
        containing the eigenmode results. If LOM simulation is performed, returns a tuple containing the
        LOM dataframe and LOM object.
    """
    design.delete_all_components()
    emode_df = {}
    lom_df = {}
    return_df = {}

    emode_obj = None
    lom_obj = None

    if "cpw_opts" in device_dict.keys():
        emode_df, emode_obj = run_eigenmode(design, device_dict, emode_options)
        if coupler_type.lower() == "ncap":
            # emode_df, emode_obj = run_eigenmode(design, device_dict, sim_options)
            ncap_lom_df, lom_obj = run_capn_LOM(design, device_dict["cplr_opts"], lom_options)
            f_est, kappa_est = find_kappa(emode_df["sim_results"]["cavity_frequency"], ncap_lom_df["sim_results"]["C_top2ground"], ncap_lom_df["sim_results"]["C_top2bottom"])
            # emode_df["sim_results"]["cavity_frequency"] = f_est
            # emode_df["sim_results"]["kappa"] = kappa_est
            return_df.update({"lom_df": ncap_lom_df})
        return_df.update({"eigenmode_df": emode_df})
        return_df["final_sim_results"] = {}
        return_df["final_sim_results"]["cavity_frequency"] = f_est
        return_df["final_sim_results"]["kappa"] = kappa_est
        return_df["final_sim_results"].update({"cavity_frequency_unit": "GHz"})
        return_df["final_sim_results"].update({"kappa_unit": "kHz"})
    else:
        lom_df, lom_obj = run_xmon_LOM(design, device_dict["design_options"], lom_options) if "cross_length" in device_dict["design_options"] else run_capn_LOM(design, device_dict, lom_options)
        return_df = lom_df

    return return_df, emode_obj, lom_obj

def get_sim_results(emode_df = {}, lom_df = {}, ncap_lom_df = {}):
    """
    Retrieves simulation results from the provided dataframes and calculates additional parameters.

    Args:
        emode_df (dict): Dataframe containing eigenmode simulation results.
        lom_df (dict): Dataframe containing lumped element model simulation results.
        ncap_lom_df (dict): Dataframe containing lumped element model simulation results for NCap coupler.

    Returns:
        dict: A dictionary containing the calculated simulation results.

    """
    data_emode = {} if emode_df == {} else emode_df["sim_results"]
    data_lom = {} if lom_df == {} else lom_df["sim_results"]

    data = {}

    cross2cpw = abs(lom_df["sim_results"]["cross_to_claw"]) * 1e-15
    cross2ground = abs(lom_df["sim_results"]["cross_to_ground"]) * 1e-15
    f_r = emode_df["sim_results"]["cavity_frequency"]
    Lj = lom_df["design"]["design_options"]["aedt_q3d_inductance"] * (1 if lom_df["design"]["design_options"]["aedt_q3d_inductance"] > 1e-9 else 1e-9)
    # print(Lj)
    gg, aa, ff_q = find_g_a_fq(cross2cpw, cross2ground, f_r, Lj, N=4)
    kappa = emode_df["sim_results"]["kappa"]
    Q = emode_df["sim_results"]["Q"]
    if ncap_lom_df != {}:
        f_r, kappa = find_kappa(emode_df["sim_results"]["cavity_frequency"], ncap_lom_df["sim_results"]["C_top2ground"], ncap_lom_df["sim_results"]["C_top2bottom"])

    data = dict(
        cavity_frequency_GHz = f_r,
        Q = Q,
        kappa_kHz = kappa,
        g_MHz = gg,
        anharmonicity_MHz = aa,
        qubit_frequency_GHz = ff_q
    )

    return data


def run_eigenmode(design, geometry_dict, sim_options):
    """
    Runs the eigenmode simulation for a given design using Ansys HFSS.

    Args:
        design (str): The name of the design.
        geometry_dict (dict): A dictionary containing the geometry options for the simulation.
        sim_options (dict): A dictionary containing the simulation options.

    Returns:
        tuple: A tuple containing the simulation results and the EPRAnalysis object.
            The simulation results are stored in a dictionary with the following structure:
            {
                "design": {
                    "coupler_type": "CLT",
                    "design_options": geometry_dict,
                    "design_tool": "Qiskit Metal"
                },
                "sim_options": {
                    "sim_type": "epr",
                    "setup": setup,
                    "simulator": "Ansys HFSS"
                },
                "sim_results": {
                    "cavity_frequency": f_rough,
                    "Q": Q,
                    "kappa": kappa
                },
                "misc": data
            }
            The EPRAnalysis object is returned for further analysis or post-processing.
    """

    cpw_length = int("".join(filter(str.isdigit, geometry_dict["cpw_opts"]["total_length"])))
    claw = create_claw(geometry_dict["claw_opts"], cpw_length, design)
    coupler = create_coupler(geometry_dict["cplr_opts"], design)
    cpw = create_cpw(geometry_dict["cpw_opts"], coupler, design)
    config = SimulationConfig(min_converged_passes=3)

    epra, hfss = start_simulation(design, config)
    hfss.clean_active_design()
    # setup = set_simulation_hyperparameters(epra, config)
    # ["setup"]
    epra.sim.setup = Dict(sim_options)
    epra.sim.setup.name = "test_setup"
    epra.sim.renderer.options.max_mesh_length_port = '7um'
    setup = epra.sim.setup
    # print(setup)
    # print(type(setup))
    # print(type(sim_options["setup"]))

    mesh_lengths = {}
    coupler_type = "CLT"
    # "finger_count" in geometry_dict["cplr_opts"]
    if geometry_dict['cplr_opts'].get('finger_count') is not None :
        coupler_type = "NCap"
        render_simulation_no_ports(epra, [cpw,claw], [(cpw.name, "start")], config.design_name, setup.vars)
        mesh_lengths = {'mesh1': {"objects": [f"trace_{cpw.name}", f"readout_connector_arm_{claw.name}"], "MaxLength": '4um'}}
    else:
        render_simulation_with_ports(epra, config.design_name, setup.vars, coupler)
        mesh_lengths = {'mesh1': {"objects": [f"prime_cpw_{coupler.name}", f"second_cpw_{coupler.name}", f"trace_{cpw.name}", f"readout_connector_arm_{claw.name}"], "MaxLength": '7um'}}
    
    modeler = hfss.pinfo.design.modeler
    
    #add_ground_strip_and_mesh(modeler, coupler, mesh_lengths=mesh_lengths)
    print(mesh_lengths)
    mesh_objects(modeler, mesh_lengths)
    f_rough, Q, kappa = get_freq_Q_kappa(epra, hfss)

    data = epra.get_data()

    data_df = {
        "design": {
            "coupler_type": coupler_type,
            "design_options": geometry_dict,
            "design_tool": "Qiskit Metal"
        },
        "sim_options": {
            "sim_type": "epr",
            "setup": setup,
            "renderer_options": epra.sim.renderer.options,
            "simulator": "Ansys HFSS"
        },
        "sim_results": {
            "cavity_frequency": f_rough,
            "cavity_frequency_unit": "GHz",
            "Q": Q,
            "kappa": kappa,
            "kappa_unit": "kHz"
        },
        "misc": data
    }

    return data_df, epra

def run_capn_LOM(design, param, sim_options):
    """
    Run capacitance analysis using Qiskit Metal and Ansys HFSS.

    Args:
        design (metal.designs.design_planar.DesignPlanar): The design object.
        param (dict): Design options for the coupler.
        sim_options (dict): Simulation options.

    Returns:
        tuple: A tuple containing the following:
            - data_df (dict): Dictionary containing design, simulation options, simulation results, and miscellaneous data.
            - loma (LOManalysis): The LOManalysis object.

    """
    # design = metal.designs.design_planar.DesignPlanar()
    # gui = metal.MetalGUI(design)
    # design.overwrite_enabled = True

    coupler = create_coupler(param, design)

    loma = LOManalysis(design, "q3d")
    loma.sim.setup.reuse_selected_design = False
    loma.sim.setup.reuse_setup = False

    # example: update single setting
    loma.sim.setup.max_passes = 33
    loma.sim.setup.min_converged_passes = 3
    loma.sim.setup.percent_error = 0.1
    loma.sim.setup.auto_increase_solution_order = 'False'
    loma.sim.setup.solution_order = 'Medium'

    loma.sim.setup.name = 'lom_setup'

    loma.sim.setup = sim_options["setup"]

    loma.sim.run(name = 'LOMv2.01', components=[coupler.name],
    open_terminations=[(coupler.name, pin_name) for pin_name in coupler.pin_names])
    cap_df = loma.sim.capacitance_matrix
    data = loma.get_data()
    setup = loma.sim.setup

    data_df = {
        "design": {
            "coupler_type": "NCap",
            "design_options": param,
            "design_tool": "Qiskit Metal"
        },
        "sim_options": {
            "sim_type": "lom",
            "setup": setup,
            "simulator": "Ansys HFSS"
        },
        "sim_results": {
            "C_top2top" : abs(cap_df[f"cap_body_0_{coupler.name}"].values[0]),
            "C_top2bottom" : abs(cap_df[f"cap_body_0_{coupler.name}"].values[1]),
            "C_top2ground" : abs(cap_df[f"cap_body_0_{coupler.name}"].values[2]),
            "C_bottom2bottom" : abs(cap_df[f"cap_body_1_{coupler.name}"].values[1]),
            "C_bottom2ground" : abs(cap_df[f"cap_body_1_{coupler.name}"].values[2]),
            "C_ground2ground" : abs(cap_df[f"ground_main_plane"].values[2]),
        },
        "misc": data
    }

    return data_df, loma

def run_xmon_LOM(design, cross_dict, sim_options):
    """
    Runs the XMON LOM simulation.

    Args:
        design (metal.designs.design_planar.DesignPlanar): The design object.
        cross_dict (dict): The dictionary containing cross connection information.
        sim_options (dict): The simulation options.

    Returns:
        tuple: A tuple containing the simulation data and the LOManalysis object.
    """
    # design = metal.designs.design_planar.DesignPlanar()
    # gui = metal.MetalGUI(design)
    # design.overwrite_enabled = True

    c1 = LOManalysis(design, "q3d")

    c1.sim.setup.reuse_selected_design = False
    c1.sim.setup.reuse_setup = False

    c1.sim.setup.max_passes = 50
    c1.sim.setup.min_converged_passes = 2
    c1.sim.setup.percent_error = 0.1
    c1.sim.setup.name = 'sweep_setup'

    c1.sim.setup = sim_options #["setup"]

    qname = 'xmon'
    cnames = cross_dict["connection_pads"].keys()
    cname = list(cnames)[0]

    temp_arr = np.repeat(qname, len(cnames))
    ports_zip = list(zip(temp_arr, cnames))
    q = TransmonCross(design, qname, options=cross_dict)
    design.rebuild()
    selection = [qname]
    open_pins = ports_zip
    # print(q.options)
    c1.sim.renderer.clean_active_design()
    c1.sim.run(name = 'LOMv2.0', components=selection,
               open_terminations=open_pins)
    cap_df = c1.sim.capacitance_matrix

    # print("#"*100)
    # print(c1.sim.renderer.options)

    data = {
        "design": {
            "design_options": design.components[qname].options,
            "design_tool": "Qiskit Metal"
        },
        "sim_options": {
            "sim_type": "lom",
            "setup": c1.sim.setup,
            "renderer_options": c1.sim.renderer.options,
            "simulator": "Ansys HFSS"
        },
        "sim_results": {
            "cross_to_ground": 0 if 'ground_main_plane' not in cap_df.loc[f'cross_{qname}'] else abs(cap_df.loc[f'cross_{qname}']['ground_main_plane']),
            "claw_to_ground": 0 if 'ground_main_plane' not in cap_df.loc[f'{cname}_connector_arm_{qname}'] else abs(cap_df.loc[f'{cname}_connector_arm_{qname}']['ground_main_plane']),
            "cross_to_claw": abs(cap_df.loc[f'cross_{qname}'][f'{cname}_connector_arm_{qname}']),
            "cross_to_cross": abs(cap_df.loc[f'cross_{qname}'][f'cross_{qname}']),
            "claw_to_claw": abs(cap_df.loc[f'{cname}_connector_arm_{qname}'][f'{cname}_connector_arm_{qname}']),
            "ground_to_ground": 0 if 'ground_main_plane' not in cap_df.loc[f'cross_{qname}'] else abs(cap_df.loc['ground_main_plane']['ground_main_plane']),
            "units": "fF"
        },
    }
    # save_simulation_data_to_json(data, filename = f"qubitonly_num{i}_{comp_id}_v{version}")
    return data, c1

def run_sweep(design, sweep_opts, emode_options, lom_options, filename="default_sweep"):
    '''
    Runs a parameter sweep for the specified design.

    Args:
        design (Design): The design object.
        sweep_opts (dict): The sweep options.
        emode_options (dict): The eigenmode setup options.
        lom_options (dict): The LOM setup options.
        filename (str): The filename for the simulation results.

    Returns:
        None
        Saves each sweep iteration to a JSON file with the specified filename.
    '''
    for param in extract_QSweep_parameters(sweep_opts["geometry_dict"]):
        print(param)
        return_df, _, __ = simulate_single_design(design, param, emode_options, lom_options, sweep_opts["coupler_type"])
        filename = f"filename_{datetime.now().strftime('%d%m%Y_%H.%M.%S')}"
        save_simulation_data_to_json(return_df, filename)

def run_qubit_cavity_sweep(design, device_options, emode_setup=None, lom_setup=None, filename="default_sweep"):
    """
    Runs a parameter sweep for the specified design.
    
    Args:
        design (Design): The design object.
        sweep_opts (dict): The sweep options.
        device_dict (dict): The device dictionary containing the design options and setup.
        emode_setup (dict): The eigenmode setup options.
        lom_setup (dict): The LOM setup options.
        filename (str): The filename for the simulation results.
        
        Returns:"""
    
    simulated_params_list = [] 
    for param in extract_QSweep_parameters(device_options):
        cpw_claw_qubit_df, _, _ = simulate_whole_device(design, param, emode_setup, lom_setup)
        filename = f"filename_{datetime.now().strftime('%d%m%Y_%H.%M.%S')}"
        simulated_params_list.append(cpw_claw_qubit_df)
        save_simulation_data_to_json(cpw_claw_qubit_df, filename)

    return simulated_params_list
    
def start_simulation(design, config):
    """
    Starts the simulation with the specified design and configuration.

    :param design: The design to be simulated.
    :param config: The configuration settings for the simulation.
    :return: A tuple containing the EPR analysis object and the HFSS object.
    """
    epra = EPRanalysis(design, config.renderer_type)
    hfss = epra.sim.renderer
    print("Starting the Simulation")
    hfss.start()
    hfss.new_ansys_design(config.design_name, config.sim_type)
    return epra, hfss


def set_simulation_hyperparameters(epra, config):
    """
    Sets the simulation hyperparameters based on the provided configuration.

    :param epra: The EPR analysis object.
    :param config: The configuration settings for the simulation.
    :return: The setup object with the updated settings.
    """
    setup = epra.sim.setup
    setup.name = config.setup_name
    setup.max_passes = config.max_passes
    setup.max_delta_f = config.max_delta_f
    setup.min_converged = config.min_converged_passes
    setup.n_modes = 1
    setup.vars = {'Lj': f'{config.Lj}nH', 'Cj': f'{config.Cj}fF'}
    return setup


def render_simulation_with_ports(epra, ansys_design_name, setup_vars, coupler):
    """
    Renders the simulation into HFSS.

    :param epra: The EPR analysis object.
    :param ansys_design_name: The name of the Ansys design.
    :param setup_vars: The setup variables for the rendering.
    :param coupler: The coupler object.
    """
    print(epra.sim)
    epra.sim.renderer.clean_active_design()
    epra.sim._render(name=ansys_design_name,
                     solution_type='eigenmode',
                     vars_to_initialize=setup_vars,
                     open_pins=[(coupler.name, "prime_start"), (coupler.name, "prime_end")],
                     port_list=[(coupler.name, 'prime_start', 50), (coupler.name, "prime_end", 50)],
                     box_plus_buffer=True)
    print("Sim rendered into HFSS!")

def render_simulation_no_ports(epra, components, open_pins, ansys_design_name, setup_vars):
    """
    Renders the simulation into HFSS.

    :param epra: The EPR analysis object.
    :param ansys_design_name: The name of the Ansys design.
    :param setup_vars: The setup variables for the rendering.
    :param components: List of QComponent object.
    """
    epra.sim._render(name=ansys_design_name,
                     selection=[qcomp.name for qcomp in components],
                     open_pins=open_pins,
                     solution_type='eigenmode',
                     vars_to_initialize=setup_vars,
                     box_plus_buffer=True)
    print("Sim rendered into HFSS!")
