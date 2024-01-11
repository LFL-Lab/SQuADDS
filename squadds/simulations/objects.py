"""
========================================================================================================================
SimulationConfig
========================================================================================================================
"""

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

def simulate_whole_device(design, cross_dict, cavity_dict, LOM_options, eigenmode_options):
    """
    Simulates the whole device by running eigenmode and LOM simulations.

    Args:
        design (metal.designs.design_planar.DesignPlanar): The design object.
        cross_dict (dict): Dictionary containing qubit options.
        cavity_dict (dict): Dictionary containing cavity options.
        LOM_options (dict): Dictionary containing LOM setup options.
        eigenmode_options (dict): Dictionary containing eigenmode setup options.

    Returns:
        tuple: A tuple containing the simulation results, LOM analysis object, and eigenmode analysis object.
    """
    design.delete_all_components()
    # print(cavity_dict)
    emode_df, epra = run_eigenmode(design, cavity_dict, eigenmode_options)
    lom_df, loma = run_xmon_LOM(design, cross_dict, LOM_options)
    data = get_sim_results(emode_df = emode_df, lom_df = lom_df)

    device_dict = Dict(
        cavity_options = Dict(
            coupling_type = "CLT",
            coupler_options = cavity_dict["cplr_opts"],
            cpw_options = Dict (
                left_options = cavity_dict["cpw_opts"],
            )
            
        ),
        qubit_options = cross_dict
    )

    design = metal.designs.design_planar.DesignPlanar()
    gui = metal.MetalGUI(design)
    design.overwrite_enabled = True
    QC = create_qubitcavity(device_dict, design)

    # gui.rebuild()
    # gui.autoscale()
    # gui.screenshot()

    return_df = dict(
        sim_options = dict(
            setup = dict(
                eigenmode_setup = eigenmode_options,
                LOM_setup = LOM_options
            ),
            simulator = "Ansys HFSS"
        ),
        sim_results = data,
        design = dict(
            design_options = device_dict
        )
    )

    return return_df, loma, epra

def simulate_single_design(design, gui, device_dict, sim_options):
    """
    Simulates a single design using the provided parameters.

    Args:
        design (Design): The design object representing the design.
        gui (GUI): The GUI object for displaying simulation results.
        device_dict (dict): A dictionary containing device options.
        sim_options (dict): A dictionary containing simulation options.

    Returns:
        dict or tuple: The simulation results. If eigenmode simulation is performed, returns a dictionary
        containing the eigenmode results. If LOM simulation is performed, returns a tuple containing the
        LOM dataframe and LOM object.
    """
    design.delete_all_components()
    emode_df = {}
    lom_df = {}
    if "cpw_opts" in device_dict.keys():
        emode_df = run_eigenmode(design, device_dict, sim_options)
    else:
        lom_df, lom_obj = run_xmon_LOM(design, device_dict, sim_options) if "cross_length" in device_dict else run_capn_LOM(design, device_dict, sim_options)
    # return get_sim_results(lom_df=lom_df, emode_df=emode_df)
    # gui.rebuild()
    # gui.autoscale()
    # gui.screenshot()

    return emode_df if emode_df != {} else (lom_df, lom_obj)

def get_sim_results(emode_df = {}, lom_df = {}):
    """
    Retrieves simulation results from the provided dataframes and calculates additional parameters.

    Args:
        emode_df (dict): Dataframe containing eigenmode simulation results.
        lom_df (dict): Dataframe containing lumped element model simulation results.

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
    data = dict(
        cavity_frequency_GHz = f_r,
        Q = emode_df["sim_results"]["Q"],
        kappa_kHz = emode_df["sim_results"]["kappa"],
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
    epra.sim.setup = Dict(sim_options["setup"])
    epra.sim.setup.name = "test_setup"
    epra.sim.renderer.options.max_mesh_length_port = '7um'
    setup = epra.sim.setup
    # print(setup)
    # print(type(setup))
    # print(type(sim_options["setup"]))

    render_simulation_with_ports(epra, config.design_name, setup.vars, coupler)
    modeler = hfss.pinfo.design.modeler

    mesh_lengths = {'mesh1': {"objects": [f"prime_cpw_{coupler.name}", f"second_cpw_{coupler.name}", f"trace_{cpw.name}", f"readout_connector_arm_{claw.name}"], "MaxLength": '7um'}}
    #add_ground_strip_and_mesh(modeler, coupler, mesh_lengths=mesh_lengths)
    print(mesh_lengths)
    mesh_objects(modeler, mesh_lengths)
    f_rough, Q, kappa = get_freq_Q_kappa(epra, hfss)

    data = epra.get_data()

    data_df = {
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
    c1.sim.setup.min_converged_passes = 1
    c1.sim.setup.percent_error = 0.1
    c1.sim.setup.name = 'sweep_setup'

    c1.sim.setup = sim_options #["setup"]

    qname = 'xmon'
    cnames = cross_dict["connection_pads"].keys()
    cname = list(cnames)[0]

    temp_arr = np.repeat(qname, len(cnames))
    ports_zip = zip(temp_arr, cnames)
    q = TransmonCross(design, qname, options=cross_dict)
    design.rebuild()
    selection = [qname]
    open_pins = ports_zip
    print(q.options)
    c1.sim.renderer.clean_active_design()
    c1.sim.run(name = 'LOMv2.0', components=selection,
               open_terminations=open_pins)
    cap_df = c1.sim.capacitance_matrix

    # print(cap_df)

    data = {
        "design": {
            "design_options": design.components[qname].options,
            "design_tool": "Qiskit Metal"
        },
        "sim_options": {
            "sim_type": "lom",
            "setup": c1.sim.setup,
            "simulator": "Ansys HFSS"
        },
        "sim_results": {
            "cross_to_ground": 0 if 'ground_main_plane' not in cap_df.loc[f'cross_{qname}'] else cap_df.loc[f'cross_{qname}']['ground_main_plane'],
            "claw_to_ground": 0 if 'ground_main_plane' not in cap_df.loc[f'{cname}_connector_arm_{qname}'] else cap_df.loc[f'{cname}_connector_arm_{qname}']['ground_main_plane'],
            "cross_to_claw": cap_df.loc[f'cross_{qname}'][f'{cname}_connector_arm_{qname}'],
            "cross_to_cross": cap_df.loc[f'cross_{qname}'][f'cross_{qname}'],
            "claw_to_claw": cap_df.loc[f'{cname}_connector_arm_{qname}'][f'{cname}_connector_arm_{qname}'],
            "ground_to_ground": 0 if 'ground_main_plane' not in cap_df.loc[f'cross_{qname}'] else cap_df.loc['ground_main_plane']['ground_main_plane']
        },
    }
    # save_simulation_data_to_json(data, filename = f"qubitonly_num{i}_{comp_id}_v{version}")
    return data, c1

def CLT_epr_sweep(design, sweep_opts, filename):    
    """
    Perform a parameter sweep for a CLT (Coupled-Line T-Junction) EPR (Electrically Parallel Resonator) simulation.

    Args:
        design (str): The design name.
        sweep_opts (dict): The sweep options.
        filename (str): The filename to save the simulation data.

    Returns:
        None
    """
    for param in extract_QSweep_parameters(sweep_opts):
        cpw_length = int("".join(filter(str.isdigit, param["cpw_opts"]["total_length"])))
        claw = create_claw(param["claw_opts"], cpw_length, design)
        coupler = create_coupler(param["cplr_opts"], design)
        cpw = create_cpw(param["cpw_opts"], coupler, design)
        # gui.rebuild()
        # gui.autoscale()

        config = SimulationConfig(min_converged_passes=3)

        epra, hfss = start_simulation(design, config)
        setup = set_simulation_hyperparameters(epra, config)
        epra.sim.renderer.options.max_mesh_length_port = '7um'

        render_simulation_with_ports(epra, config.design_name, setup.vars, coupler)
        modeler = hfss.pinfo.design.modeler

        mesh_lengths = {'mesh1': {"objects": [f"prime_cpw_{coupler.name}", f"second_cpw_{coupler.name}", f"trace_{cpw.name}", f"readout_connector_arm_{claw.name}"], "MaxLength": '7um'}}
        #add_ground_strip_and_mesh(modeler, coupler, mesh_lengths=mesh_lengths)
        print(mesh_lengths)
        mesh_objects(modeler, mesh_lengths)
        f_rough, Q, kappa = get_freq_Q_kappa(epra, hfss)

        data = epra.get_data()

        data_df = {
            "design_options": {
                "coupling_type": "CLT",
                "geometry_dict": param
            },
            "sim_options": {
                "sim_type": "epr",
                "setup": setup,
            },
            "sim_results": {
                "cavity_frequency": f_rough,
                "Q": Q,
                "kappa": kappa
            },
            "misc": data
        }
        
        # filename = f"CLT_cpw{cpw.options.total_length}_claw{claw.options.connection_pads.readout.claw_width}_clength{coupler.options.coupling_length}"
        save_simulation_data_to_json(data_df, filename)

def NCap_epr_sweep(design, sweep_opts):    
    """
    Perform a parameter sweep for NCap EPR simulations.

    Args:
        design (Design): The design object.
        sweep_opts (dict): The sweep options.

    Returns:
        None
    """
    for param in extract_QSweep_parameters(sweep_opts):
        claw = create_claw(param["claw_opts"], design)
        coupler = create_coupler(param["cplr_opts"], design)
        cpw = create_cpw(param["cpw_opts"], design)
        # gui.rebuild()
        # gui.autoscale()
        
        config = SimulationConfig()

        epra, hfss = start_simulation(design, config)
        setup = set_simulation_hyperparameters(epra, config)
        
        render_simulation_no_ports(epra, [cpw,claw], [(cpw.name, "start")], config.design_name, setup.vars)
        modeler = hfss.pinfo.design.modeler

        mesh_lengths = {'mesh1': {"objects": [f"trace_{cpw.name}", f"readout_connector_arm_{claw.name}"], "MaxLength": '4um'}}
        mesh_objects(modeler,  mesh_lengths)
        f_rough = get_freq(epra, hfss)

        data = epra.get_data()

        data_df = {
            "design_options": {
                "coupling_type": "NCap",
                "geometry_dict": param
            },
            "sim_options": {
                "sim_type": "epr",
                "setup": setup,
            },
            "sim_results": {
                "cavity_frequency": f_rough
            },
            "misc": data
        }
        
        filename = f"CLT_cpw{cpw.options.total_length}_claw{claw.options.connection_pads.readout.claw_width}_clength{coupler.options.coupling_length}"
        save_simulation_data_to_json(data_df, filename)

def NCap_LOM_sweep(design, sweep_opts):
    """
    Perform a sweep analysis for NCap LOManalysis.

    Args:
        design (Design): The design object.
        sweep_opts (dict): The sweep options.

    Returns:
        None
    """
    for param in extract_QSweep_parameters(sweep_opts):
        # claw = create_claw(param["claw_opts"], design)
        coupler = create_coupler(param, design)
        # coupler.options[""]
        # cpw = create_cpw(param["cpw_opts"], design)
        # gui.rebuild()
        # gui.autoscale()

        loma = LOManalysis(design, "q3d")
        loma.sim.setup.reuse_selected_design = False
        loma.sim.setup.reuse_setup = False

        # example: update single setting
        loma.sim.setup.max_passes = 30
        loma.sim.setup.min_converged_passes = 5
        loma.sim.setup.percent_error = 0.1
        loma.sim.setup.auto_increase_solution_order = 'False'
        loma.sim.setup.solution_order = 'Medium'

        loma.sim.setup.name = 'lom_setup'

        loma.sim.run(name = 'LOMv2.01', components=[coupler.name],
        open_terminations=[(coupler.name, pin_name) for pin_name in coupler.pin_names])
        cap_df = loma.sim.capacitance_matrix
        data = loma.get_data()
        setup = loma.sim.setup

        data_df = {
            "design_options": {
                "coupling_type": "NCap",
                "geometry_dict": param
            },
            "sim_options": {
                "sim_type": "lom",
                "setup": setup,
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

        filename = f"NCap_LOM_fingerwidth{coupler.options.cap_width}_fingercount{coupler.options.finger_count}_fingerlength{coupler.options.finger_length}_fingergap{coupler.options.cap_gap}"
        save_simulation_data_to_json(data_df, filename)

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
