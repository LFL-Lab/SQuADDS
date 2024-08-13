import pandas as pd
from pyEPR.calcs import Convert

from squadds import Analyzer
from squadds.core.utils import *
from squadds.interpolations.interpolator import Interpolator


def string_to_float(string):
    """
    Converts a string representation of a number to a float.

    Args:
        string (str): The string representation of the number.

    Returns:
        float: The converted float value.
    """
    return float(string[:-2])

class ScalingInterpolator(Interpolator):
    """Class for scaling-based interpolation."""
    def __init__(self, analyzer: Analyzer, target_params: dict):
        super().__init__(analyzer, target_params)

    def get_design(self) -> pd.DataFrame:
        """
        Retrieves the design options for qubit and cavity based on target parameters.
        
        Returns:
            pd.DataFrame: A DataFrame containing the design options for qubit and cavity.
        """
        # Extract target parameters
        f_q_target = self.target_params['qubit_frequency_GHz']
        g_target = self.target_params['g_MHz']
        alpha_target = self.target_params['anharmonicity_MHz']
        f_res_target = self.target_params['cavity_frequency_GHz']
        kappa_target = self.target_params['kappa_kHz']
        try:
            res_type = self.target_params['resonator_type']
        except:
            res_type = self.analyzer.selected_resonator_type

        self.df = self.analyzer.df
        
        # Find the closest qubit-claw design
        if self.analyzer.selected_resonator_type == 'half':
            closest_qubit_claw_design = self.analyzer.find_closest({"qubit_frequency_GHz": f_q_target,'anharmonicity_MHz': alpha_target, 'g_MHz': g_target},parallel=True, num_cpu="auto", num_top=1)
        else:
            closest_qubit_claw_design = self.analyzer.find_closest({"qubit_frequency_GHz": f_q_target,'anharmonicity_MHz': alpha_target, 'g_MHz': g_target}, num_top=1)

        # Scale values
        alpha_scaling = closest_qubit_claw_design['anharmonicity_MHz'] / alpha_target
        g_scaling = g_target / closest_qubit_claw_design['g_MHz']

        # Scale qubit and claw dimensions
        updated_cross_length = string_to_float(closest_qubit_claw_design["design_options_qubit"].iloc[0]['cross_length']) * alpha_scaling.values[0]
        updated_claw_length = string_to_float(closest_qubit_claw_design["design_options_qubit"].iloc[0]["connection_pads"]["readout"]['claw_length']) * g_scaling.values[0] * alpha_scaling.values[0]

        # Scaling logic for cavity-coupler designs
        # Filter DataFrame based on qubit coupling claw capacitance
        try:
            cross_to_claw_cap_chosen = closest_qubit_claw_design['cross_to_claw'].iloc[0]
        except:
            cross_to_claw_cap_chosen = closest_qubit_claw_design['cross_to_claw_closest'].iloc[0]
        
        threshold = 0.3  # 30% threshold
        try:
            filtered_df = self.df[(self.df['cross_to_claw'] >= (1 - threshold) * cross_to_claw_cap_chosen) &
                                    (self.df['cross_to_claw'] <= (1 + threshold) * cross_to_claw_cap_chosen)]
        except:
            filtered_df = self.df[(self.df['cross_to_claw_closest'] >= (1 - threshold) * cross_to_claw_cap_chosen) &
                                    (self.df['cross_to_claw_closest'] <= (1 + threshold) * cross_to_claw_cap_chosen)]

        # Find the closest cavity-coupler design
        merged_df = self.analyzer.df.copy()
        system_chosen = self.analyzer.selected_system
        H_params_chosen = self.analyzer.H_param_keys

        self.analyzer.df = filtered_df
        self.analyzer.selected_system = 'cavity_claw' 
        self.analyzer.H_param_keys = ['resonator_type','cavity_frequency_GHz', 'kappa_kHz']
        self.analyzer.target_params = {'cavity_frequency_GHz': f_res_target, 'kappa_kHz': kappa_target, 'resonator_type': res_type}

        target_params_cavity = {'cavity_frequency_GHz': f_res_target, 'kappa_kHz': kappa_target, 'resonator_type': res_type}

        if self.analyzer.selected_resonator_type == 'half':
            closest_cavity_cpw_design = self.analyzer.find_closest(target_params_cavity,parallel=True, num_cpu="auto", num_top=1)
        else:
            closest_cavity_cpw_design = self.analyzer.find_closest(target_params_cavity, num_top=1)

        closest_kappa = closest_cavity_cpw_design['kappa_kHz'].values[0]
        closest_f_cavity = closest_cavity_cpw_design['cavity_frequency_GHz'].values[0]

        if self.analyzer.selected_resonator_type == 'half':
            closest_coupler_length = string_to_float(closest_cavity_cpw_design["design_options_cavity_claw"].iloc[0]['cplr_opts']['finger_length'])
        else:
            closest_coupler_length = string_to_float(closest_cavity_cpw_design["design_options_cavity_claw"].iloc[0]['cplr_opts']['coupling_length'])

        # Scale resonator and coupling element dimensions
        updated_resonator_length = string_to_float(closest_cavity_cpw_design["design_options_cavity_claw"].iloc[0]["cpw_opts"]['total_length']) * (closest_cavity_cpw_design['cavity_frequency_GHz'] / f_res_target).values[0]

        res_scaling = closest_f_cavity / f_res_target
        res_scaling = closest_f_cavity / f_res_target
        kappa_scaling = np.sqrt(kappa_target / closest_kappa)

        print("="*50)
        print(f"Kappa scaling: {kappa_scaling}")
        print(f"g scaling: {g_scaling.values[0]}")
        print(f"alpha scaling: {alpha_scaling.values[0]}")
        print(f"resonator scaling: {res_scaling}")
        print("="*50)

        updated_coupling_length = closest_coupler_length * kappa_scaling
        # round updated_coupling_length to nearest integer
        updated_coupling_length = round(updated_coupling_length)


        # Reset the analyzer's DataFrame
        self.analyzer.df = merged_df
        self.analyzer.selected_system = system_chosen
        self.analyzer.H_param_keys = H_params_chosen

        # a dataframe with three empty colums
        interpolated_designs_df = pd.DataFrame(columns=["design_options_qubit", "design_options_cavity_claw", "design_options"])

        # Update the qubit and cavity design options
        qubit_design_options = closest_qubit_claw_design["design_options_qubit"].iloc[0]
        qubit_design_options['cross_length'] = f"{updated_cross_length}um"
        qubit_design_options["connection_pads"]["readout"]['claw_length'] = f"{updated_claw_length}um"
        required_Lj = Convert.Lj_from_Ej(closest_qubit_claw_design['EJ'].iloc[0], units_in='GHz', units_out='nH') 
        qubit_design_options['aedt_hfss_inductance'] = required_Lj*1e-9
        qubit_design_options['aedt_q3d_inductance'] = required_Lj*1e-9
        qubit_design_options['q3d_inductance'] = required_Lj*1e-9
        qubit_design_options['hfss_inductance'] = required_Lj*1e-9
        qubit_design_options["connection_pads"]["readout"]['Lj'] = f"{required_Lj}nH"

        # setting the `claw_cpw_*` params to zero
        qubit_design_options["connection_pads"]['readout']['claw_cpw_length'] = "0um"
        qubit_design_options["connection_pads"]['readout']['claw_cpw_width'] = "0um"

        cavity_design_options = closest_cavity_cpw_design["design_options_cavity_claw"].iloc[0]
        cavity_design_options["cpw_opts"]['total_length'] = f"{updated_resonator_length}um"

        if self.analyzer.selected_resonator_type == 'half':
            cavity_design_options['cplr_opts']['finger_length'] = f"{updated_coupling_length}um"
        else:
            cavity_design_options['cplr_opts']['coupling_length'] = f"{updated_coupling_length}um"

        # update the claw of the cavity based on the one from the qubit
        cavity_design_options["claw_opts"]["connection_pads"] = qubit_design_options["connection_pads"]

        interpolated_designs_df = pd.DataFrame({
            "coupler_type": self.analyzer.selected_coupler,
            "design_options_qubit": [qubit_design_options],
            "design_options_cavity_claw": [cavity_design_options],
            "setup_qubit": [closest_qubit_claw_design["setup_qubit"].iloc[0]],
            "setup_cavity_claw": [closest_cavity_cpw_design["setup_cavity_claw"].iloc[0]],
        })

        device_design_options = create_unified_design_options(interpolated_designs_df.iloc[0])

        # add the device design options to the dataframe
        interpolated_designs_df["design_options"] = [device_design_options]
        interpolated_designs_df.iloc[0]["design_options"]["qubit_options"]["connection_pads"]["readout"]["claw_cpw_length"] = "0um"

        return interpolated_designs_df