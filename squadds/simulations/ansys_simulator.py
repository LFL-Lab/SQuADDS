# from utils import *
from squadds.simulations.objects import *

from qiskit_metal import draw, Dict, designs, MetalGUI
from qiskit_metal.toolbox_metal import math_and_overrides
import qiskit_metal as metal

class AnsysSimulator:

    def __init__(self, analyzer, design_options):
        self.analyzer = analyzer
        self.design_options = design_options

        self.lom_analysis_obj = None
        self.epr_analysis_obj = None

        self.design = metal.designs.design_planar.DesignPlanar()
        self.gui = metal.MetalGUI(self.design)
        self.design.overwrite_enabled = True

        print(f"selected system: {self.analyzer.selected_system}")

    def get_design_screenshot(self):
        self.gui.rebuild()
        self.gui.autoscale()
        self.gui.screenshot()

    def simulate(self, device_dict):
        if isinstance(self.analyzer.selected_system, list): # have a qubit_cavity object
            # print("if")
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
            # print("else")
            self.geom_dict = device_dict["design_options"]
            self.setup_dict = device_dict["setup"]
            return_df, self.lom_analysis_obj = simulate_single_design(design=self.design, gui=self.gui, device_dict=self.geom_dict, sim_options=self.setup_dict)
        
        return return_df
        # print(f"geoms: {self.geom_dict}")
        # print(f"setup: {self.setup_dict}")

    def get_renderer_screenshot(self):
        if self.epr_analysis_obj is not None:
            self.epr_analysis_obj.sim.save_screenshot()
        if self.lom_analysis_obj is not None:
            self.lom_analysis_obj.sim.save_screenshot()
    
    def get_xmon_info(self, xmon_dict):
        # data = xmon_dict["sim_results"]
        cross2cpw = abs(xmon_dict["sim_results"]["cross_to_claw"]) * 1e-15
        cross2ground = abs(xmon_dict["sim_results"]["cross_to_ground"]) * 1e-15
        Lj = xmon_dict["design"]["design_options"]["aedt_q3d_inductance"] * (1 if xmon_dict["design"]["design_options"]["aedt_q3d_inductance"] > 1e-9 else 1e-9)
        a, fq = find_a_fq(cross2cpw, cross2ground, Lj)
        print(f"qubit anharmonicity = {round(a)} MHz \nqubit frequency = {round(fq, 3)} GHz")

    def plot_device(device_dict):
        print(device_dict)