# from utils import *
import sys
# warn using `warnings` if os is mac that this is not supported
import warnings

import qiskit_metal as metal
from qiskit_metal import Dict, MetalGUI, designs, draw
from qiskit_metal.toolbox_metal import math_and_overrides

from squadds.simulations.objects import *


class AnsysSimulator:
    """
    AnsysSimulator class for simulating devices using Ansys.

    Attributes:
        analyzer: The analyzer object.
        design_options: The design options.
        lom_analysis_obj: The LOM analysis object.
        epr_analysis_obj: The EPR analysis object.
        design: The design planar object.
        gui: The MetalGUI object.
    """

    def __init__(self, analyzer, design_options):
        """
        Initialize the AnsysSimulator object.

        Args:
            analyzer: The analyzer object.
            design_options: The design options.
        """
        self.analyzer = analyzer
        self.design_options = design_options

        self.lom_analysis_obj = None
        self.epr_analysis_obj = None

        self.design = metal.designs.design_planar.DesignPlanar()
        self.gui = metal.MetalGUI(self.design)
        self.design.overwrite_enabled = True
        self._warnings()

        print(f"selected system: {self.analyzer.selected_system}")

    def _warnings(self):
            """
            Displays a warning message if the operating system is macOS.

            Raises:
                UserWarning: If the operating system is macOS, a warning is raised indicating that `AnsysSimulator` is not supported on MacOS.
            """
            if sys.platform == "darwin":  # Checks if the operating system is macOS
                warnings.warn("`AnsysSimulator` is not supported on MacOS since Ansys does not have a Mac App. Please use Windows or Linux for simulations.")

    def get_design_screenshot(self):
        """
        Saves a screenshot of the design.

        Returns:
            None
        """
        self.gui.rebuild()
        self.gui.autoscale()
        self.gui.screenshot()

    def simulate(self, device_dict):
        """
        Simulates the device based on the provided device dictionary.

        Args:
            device_dict (dict): A dictionary containing the device design options and setup.

        Returns:
            pandas.DataFrame: The simulation results.

        Raises:
            None
        """
        if isinstance(self.analyzer.selected_system, list): # have a qubit_cavity object
            self.geom_dict = Dict(
                qubit_geoms = device_dict["design_options_qubit"],
                cavity_geoms = device_dict["design_options_cavity_claw"]
            )
            self.setup_dict = Dict(
                qubit_setup = device_dict["setup_qubit"],
                cavity_setup = device_dict["setup_cavity_claw"]
            )
            return_df, self.lom_analysis_obj, self.epr_analysis_obj = simulate_whole_device(design=self.design, cross_dict=self.geom_dict.qubit_geoms, cavity_dict=self.geom_dict.cavity_geoms, LOM_options=self.setup_dict.qubit_setup, eigenmode_options=self.setup_dict.cavity_setup)

        else: # have a non-qubit_cavity object
            self.geom_dict = device_dict["design_options"]
            self.setup_dict = device_dict["setup"]
            return_df, self.lom_analysis_obj = simulate_single_design(design=self.design, gui=self.gui, device_dict=self.geom_dict, sim_options=self.setup_dict)
        
        return return_df

    def get_renderer_screenshot(self):
        """
        Saves a screenshot of the renderer.

        If the EPR analysis object is not None, it saves a screenshot of the EPR analysis simulation.
        If the LOM analysis object is not None, it saves a screenshot of the LOM analysis simulation.
        """
        if self.epr_analysis_obj is not None:
            self.epr_analysis_obj.sim.save_screenshot()
        if self.lom_analysis_obj is not None:
            self.lom_analysis_obj.sim.save_screenshot()
    
    def get_xmon_info(self, xmon_dict):
        """
        Retrieves information about the Xmon qubit from the given xmon_dict.

        Parameters:
            xmon_dict (dict): A dictionary containing simulation results and design options.

        Returns:
            dict: A dictionary containing the qubit frequency in GHz and the anharmonicity in MHz.
        """
        # data = xmon_dict["sim_results"]
        cross2cpw = abs(xmon_dict["sim_results"]["cross_to_claw"]) * 1e-15
        cross2ground = abs(xmon_dict["sim_results"]["cross_to_ground"]) * 1e-15
        Lj = xmon_dict["design"]["design_options"]["aedt_q3d_inductance"] * (1 if xmon_dict["design"]["design_options"]["aedt_q3d_inductance"] > 1e-9 else 1e-9)
        a, fq = find_a_fq(cross2cpw, cross2ground, Lj)
        print(f"qubit anharmonicity = {round(a)} MHz \nqubit frequency = {round(fq, 3)} GHz")
        # return a json object
        return {"qubit_frequency_GHz": fq, "anharmonicity_MHz": a}

    def plot_device(self, device_dict):
        """
        Plot the device based on the given device dictionary.

        Parameters:
        - device_dict (dict): A dictionary containing the device information.

        Returns:
        None
        """
        self.design.delete_all_components()
        if "g" in device_dict["sim_results"]:
            qc = QubitCavity(self.design, "qubit_cavity", options=device_dict["design"]["design_options"])

        self.gui.rebuild()
        self.gui.autoscale()
        self.gui.screenshot()