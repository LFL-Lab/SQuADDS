import numpy as np
import pandas as pd
import warnings
import os
import scqubits as scq
from scqubits.core.transmon import TunableTransmon
from pymongo import MongoClient
from pandasai import SmartDataframe
from pandasai.llm import OpenAI, Starcoder, Falcon
from dotenv import load_dotenv

from squadds import logging
from squadds.core.sweeper_helperfunctions import create_dict_list
from squadds.utils.metrics import *
from squadds.core.database import SQuADDS_DB


"""
=====================================================================================
HELPER FUNCTIONS
=====================================================================================
"""
# Helper function to scale values with 'um' in them
def scale_value(value, ratio):
    # Remove 'um' from the value, convert to float, scale and convert back to string
    scaled_value = str(float(value.replace('um', '')) * ratio) + 'um'
    return scaled_value

"""
=====================================================================================
Analyzer
=====================================================================================
"""
class Analyzer:

    __supported_metrics__ = ['Euclidean', 'Manhattan', 'Chebyshev', 'Weighted Euclidean' , 'Custom']
    __supported_estimation_methods__ = ['Interpolation']

    def __init__(self, db):
        """
        Args:
            db (SQuADDS_DB): The database to analyze.

        Attributes:
            db (SQuADDS_DB): The database to analyze.
            component_name (str): The name of the component to analyze.
            component_type (str): The type of the component to analyze.
            df (pd.DataFrame): The dataframe of the component to analyze.
            metric_strategy (MetricStrategy): The strategy to use for calculating the distance metric.
            custom_metric_func (function): The custom function to use for calculating the distance metric.
            metric_weights (dict): The weights to use for calculating the weighted distance metric.
            smart_df (SmartDataframe): The SmartDataframe of the component to analyze.
            H_param_keys (list): The keys of the Hamiltonian parameters.

        Raises:
            ValueError: If the specified metric is not supported.
        """

        self.db = db

        self.selected_component_name = self.db.selected_component_name
        self.selected_component = self.db.selected_component
        self.selected_data_type = self.db.selected_data_type
        self.selected_confg = self.db.selected_confg
        self.selected_qubit = self.db.selected_qubit
        self.selected_cavity = self.db.selected_cavity
        self.selected_coupler = self.db.selected_coupler
        self.selected_system = self.db.selected_system
        self.selected_df = self.db.selected_df

        self.metric_strategy = None  # Will be set dynamically
        self.custom_metric_func = None
        self.metric_weights = None
        self.smart_df = None
        self.coupling_type = None

    def set_metric_strategy(self, strategy: MetricStrategy):
        """
        Sets the metric strategy to use for calculating the distance metric.

        Args:
            strategy (MetricStrategy): The strategy to use for calculating the distance metric.

        Raises:
            ValueError: If the specified metric is not supported.
        """

        self.metric_strategy = strategy

    def _outside_bounds(self, df: pd.DataFrame, params: dict, display=True) -> bool:
        """
        Check if entered parameters are outside the bounds of a dataframe.

        Args:
            df (pd.DataFrame): Dataframe to give warning.
            params (dict): Keys are column names of `df`. Values are values to check for bounds.
        
        Returns:
            bool: True if any value is outside of bounds. False if all values are inside bounds.
        """
        outside_bounds = False
        filtered_df = df.copy()

        for param, value in params.items():
            if param not in df.columns:
                raise ValueError(f"{param} is not a column in dataframe: {df}")

            if isinstance(value, (int, float)):
                if value < df[param].min() or value > df[param].max():
                    if display:
                        logging.info(f"\033[1mNOTE TO USER:\033[0m the value \033[1m{value} for {param}\033[0m is outside the bounds of our library.\nIf you find a geometry which corresponds to these values, please consider contributing it! 😁🙏\n")
                    outside_bounds = True

            elif isinstance(value, str):
                filtered_df = filtered_df[filtered_df[param] == value]

            else:
                raise ValueError(f"Unsupported type {type(value)} for parameter {param}")

        if filtered_df.empty:
            categorical_params = {key: value for key, value in params.items() if isinstance(value, str)}
            if display and categorical_params:
                logging.info(f"\033[1mNOTE TO USER:\033[0m There are no geometries with the specified categorical parameters - \033[1m{categorical_params}\033[0m.\nIf you find a geometry which corresponds to these values, please consider contributing it! 😁🙏\n")
            outside_bounds = True

        return outside_bounds


    def find_closest(self,
                     target_params: dict,
                     num_top: int,
                     metric: str = 'Euclidean',
                     display: bool = True):
        """
        Finds the rows in the DataFrame with the closest matching characteristics
        to the given target parameters using a specified metric.
        
        Args:
            target_params (dict): A dictionary containing the target values for columns in `self.df`.
                                  Keys are column names and values are the target values.
            num_top (int): The number of closest matching rows to return.
            metric (str, optional): The distance metric to use for finding the closest matches.
                                    Available options are specified in `self.__supported_metrics__`.
                                    Defaults to 'Euclidean'.
            display (bool, optional): Whether to display warnings and logs. Defaults to True.
        
        Returns:
            pd.DataFrame: A DataFrame containing the rows with the closest matching characteristics,
                          sorted by the distance metric.
        
        Raises:
            ValueError: If the specified metric is not supported or `num_top` exceeds the DataFrame size.
        """
        ### Checks
        # Check for supported metric
        if metric not in self.__supported_metrics__:
            raise ValueError(f'`metric` must be one of the following: {self.__supported_metrics__}')
        # Check for improper size of library
        if (num_top > len(self.df)):
            raise ValueError('`num_top` cannot be bigger than size of read-in library.')

        # Log if parameters outside of library
        filtered_df = self.df[self.H_param_keys]  # Filter DataFrame based on H_param_keys
        self._outside_bounds(df=filtered_df, params=target_params, display=display)

        # Set strategy dynamically based on the metric parameter
        if metric == 'Euclidean':
            self.set_metric_strategy(EuclideanMetric())
        elif metric == 'Manhattan':
            self.set_metric_strategy(ManhattanMetric())
        elif metric == 'Chebyshev':
            self.set_metric_strategy(ChebyshevMetric())
        elif metric == 'Weighted Euclidean':
            self.set_metric_strategy(WeightedEuclideanMetric(self.metric_weights))
        elif metric == 'Custom':
            self.set_metric_strategy(CustomMetric(self.custom_metric_func))

        if not self.metric_strategy:
            raise ValueError("Invalid metric.")

        # Main logic
        distances = filtered_df.apply(lambda row: self.metric_strategy.calculate(target_params, row), axis=1)

        # Sort distances and get the closest ones
        sorted_indices = distances.nsmallest(num_top).index
        closest_df = self.df.loc[sorted_indices]

        return closest_df

    def chat(self, question, llm="OpenAI"):
        """
        Chat with the library using a language model.

        Args:
            question (str): The question to ask the library.
            llm (str, optional): The language model to use for answering the question.

        Returns:
            SmartDataframe: The answer to the question in a dataframe format.
        """

        # set up LLM
        load_dotenv()
        if llm == "OpenAI":
            llm = OpenAI()
        elif llm == "Starcoder":
            llm = Starcoder(api_token=os.getenv("HUGGING_FACE_API_KEY"))
        elif llm == "Falcon":
            llm = Falcon(api_token=os.getenv("HUGGING_FACE_API_KEY"))

        # filter the df
        filtered_df = self.df[self.H_param_keys]  # Filter DataFrame based on H_param_keys

        # chat with the dataframe using pandasai
        self.smart_df = SmartDataframe(filtered_df, config={"llm": llm})
        response = self.smart_df.chat(question)

        # merge the response with the original dataframe
        pd_response = pd.DataFrame(response, columns=self.H_param_keys)
        df_response = self.df.merge(pd_response, on=list(pd_response.columns), how='inner')
        if len(df_response) == 0:
            contrib_message = f"\nIf you find a geometry which corresponds to these values, please consider contributing it! 😁🙏\n"
            raise ValueError("No matching geometries were found for your query 😢\n" + contrib_message)
        return df_response.head(len(pd_response))


    def get_interpolated_design(self,
                     target_params: dict,
                     metric: str = 'Euclidean',
                     display: bool = True):
        """
        Implement the 7-step interpolation procedure to find the best qubit and resonator design.

        Args:
            target_params (dict): A dictionary containing the target parameters for qubit and resonator.
                                    The dictionary should contain two keys: 'qubit_params' and 'resonator_params'.
            metric (str, optional): The distance metric to use for finding the closest matches.
                                    Defaults to 'Euclidean'.
            display (bool, optional): Whether to display warnings and logs. Defaults to True.

        Returns:
            dict: A dictionary containing the interpolated design for the qubit and resonator.

        Raises:
            ValueError: If the specified metric is not supported or if `num_top` exceeds the DataFrame size.
            NotImplementedError: If auxiliary methods for calculating parameters are not implemented.
        """
        ### Checks
        # Validate target_params
        if not isinstance(target_params, dict):
            raise ValueError("`target_params` must be a dictionary.")
        if not all(key in target_params for key in self.H_param_keys):
            # remind the characteristics allowed
            characteristics = self.db.get_characteristic_info(self.component_name, self.component_type)
            raise ValueError("The target parameters must be one of the following: \n" + str(characteristics))

        # Validate metric
        if metric not in self.__supported_metrics__:
            raise ValueError(f'`metric` must be one of the following: {self.__supported_metrics__}')

        # Set num_top to 1
        num_top = 1

        ### Start the interpolation algorithm

        # Step 0: Extract the target parameters for qubit and resonator
        f_q, alpha_target, g_target = target_params.get("Qubit_Frequency_GHz"), target_params.get("Qubit_Anharmonicity_MHz"), target_params.get("Coupling_Strength_MHz")
        f_r, kappa, wavelength_type = target_params.get("Cavity_Frequency_GHz"), target_params.get("kappa_MHz"), target_params.get("wavelength")
        self.coupling_type = target_params.get("feedline_coupling")

        # Step 1: Compute the coupling parameters
        C_q, C_r, C_c, E_J, E_C = self.get_coupling_parameters(f_q, f_r, alpha_target, g_target, wavelength_type)

        # Step 2: Search database for best matching design for anharmonicity, coulping strength, resonator frequency and qubit frequency
        target_params_1 = {"Qubit_Frequency_GHz": f_q, "Qubit_Anharmonicity_MHz": alpha_target, "Coupling_Strength_MHz": g_target, "Cavity_Frequency_GHz": f_r}
        df_flagged_1 = self.find_closest(target_params_1, num_top, metric, display=False)

        # Step 3: Scale qubit and coupling capacitor areas

        # Step 3.1: Extract the best matching design parameters
        alpha_sim, g_sim = df_flagged_1['Qubit_Anharmonicity_MHz'].values[0], df_flagged_1['Coupling_Strength_MHz'].values[0]
        f_q_sim, f_r_sim = df_flagged_1['Qubit_Frequency_GHz'].values[0], df_flagged_1['Cavity_Frequency_GHz'].values[0]
        # C_q_sim, C_r_sim, C_c_sim, E_J_sim, E_C_sim = self.get_coupling_parameters(f_q_sim, f_r_sim, alpha_sim, g_sim, wavelength_type)

        # Step 3.2: Scale the qubit and coupling capacitor areas
        qubit_ratio, coupling_ratio =  alpha_sim/alpha_target, (alpha_sim/alpha_target)*(g_target/g_sim)

        # Step 3.3: Get the design dict for the best matching design
        design = self.get_design(df_flagged_1)

        # Step 3.4: Scale the qubit and coupling capacitor areas and update the design dict
        updated_design1 = self.get_updated_design_with_scaled_qubit_and_coupling_capacitor_areas(design, qubit_ratio, coupling_ratio)


        # Step 4: Search database for best matching resonator design
        target_params_resonator = {"Cavity_Frequency_GHz": f_r, "kappa_MHz": kappa, "wavelength": wavelength_type}
        df_flagged_2 = self.find_closest(target_params_resonator, num_top, metric, display=False)

        # Step 4.1: Extract the best matching resonator design parameters
        kappa_sim, f_r_sim = df_flagged_2['kappa_MHz'].values[0], df_flagged_2['Cavity_Frequency_GHz'].values[0]
        alpha_sim, g_sim = df_flagged_2['Qubit_Anharmonicity_MHz'].values[0], df_flagged_2['Coupling_Strength_MHz'].values[0]
        f_q_sim = df_flagged_2['Qubit_Frequency_GHz'].values[0]
        # C_q_sim, C_r_sim, C_c_sim, E_J_sim, E_C_sim = self.get_coupling_parameters(f_q_sim, f_r_sim, alpha_sim, g_sim, wavelength_type)

        # Steps 5: Scale the resonator length and coupling dimension

        # Step 5.1: Get the design dict for the best matching resonator design
        design = self.get_design(df_flagged_2)

        # Step 5.2: Scale the resonator length and coupling dimension
        res_length_ratio, coupling_dim_ratio = f_r_sim/f_r, np.sqrt(kappa/kappa_sim)

        # Step 5.3: Scale the resonator length and coupling dimension and update the design dict
        updated_design2 = self.get_updated_design_scaled_resonator_length_and_coupling_dim(design, res_length_ratio, coupling_dim_ratio)


        # Step 6: C_c / C_r > 0.01 check and rescale the resonator length as necessary to hit the target

        # Step 6.1: Calculate the target resonator length
        L_r = 1 / (C_r * (2*np.pi*f_r)** 2)

        if C_c / C_r > 0.01:
            # Step 6.2: rescale the resonator length
            omega_r = 1 / np.sqrt(L_r*(C_r + C_c))
            res_length_ratio = (2*np.pi*f_r_sim) / omega_r
            updated_design2 = self.get_updated_design_scaled_resonator_length_and_coupling_dim(design, res_length_ratio, coupling_dim_ratio)


        # TODO: create a `interpolated_design` dict using the updated information
        # interpolated_designs = self.interpolate_design(updated_design1, updated_design2)
        interpolated_designs = [updated_design1, updated_design2]


        # Step 7: Return interpolated design and best matching df
        df_design= self.find_closest(target_params, num_top, metric, display=False)

        return interpolated_designs, df_design

    def extract_area_parameters(self, options: dict[str, str]) -> dict[str, float]:
        """
        Extracts parameters of interest related to the qubit and cavity capacitor
        areas from the input options dictionary.

        Args:
            options (dict): Configuration options for qubit and cavity.

        Returns:
            dict: A dictionary containing the parameters of interest.
        """
        # Define keys of interest
        if "Transmon" in self.component_name:
            keys_of_interest = [
                'cavity_options.coupler_options.coupling_length',
                'cavity_options.coupler_options.prime_width',
                'qubit_options.cross_width',
                'qubit_options.cross_length'
            ]
        else:
            raise NotImplementedError("The method for calculating qubit parameters for the chosen qubit type is not implemented.")

        # Extract parameters of interest
        params_of_interest = {}
        for key in keys_of_interest:
            if key not in options:
                raise KeyError(f"Key {key} not found in options")
            params_of_interest[key] = float(options[key].strip('um'))
        return params_of_interest


    
    def get_updated_design_with_scaled_qubit_and_coupling_capacitor_areas(self, design, qubit_ratio, coupling_ratio):
        """
        Scales the qubit and coupling capacitor areas by a given ratio and updates the design dict accordingly.

        Args:
            design (dict): A dictionary containing the design parameters.
            qubit_ratio (float): The ratio by which to scale the qubit area.
            coupling_ratio (float): The ratio by which to scale the coupling capacitor area.

        Returns:
            dict: Updated design dict with scaled areas.
        """

        # Check and scale qubit capacitor area
        if 'Transmon' in self.component_name:
            design['qubit_options.cross_length'] = scale_value(design['qubit_options.cross_length'], qubit_ratio)
        else:
            raise NotImplementedError("The method for calculating qubit parameters for the chosen qubit type is not implemented.")

        # Check and scale coupling capacitor area
        coupling_type = design.get('cavity_options.coupling_type', '')
        if coupling_type in ['capacitive', 'inductive']:
            design['cavity_options.coupler_options.coupling_length'] = scale_value(design['cavity_options.coupler_options.coupling_length'], coupling_ratio)
        elif 'interdig' in coupling_type:
            # If finger_length is not in the coupler_options dictionary, set its default value
            if 'finger_length' not in design.get('cavity_options.coupler_options', {}):
                design['cavity_options.coupler_options']['finger_length'] = '50um'
            design['cavity_options.coupler_options']['finger_length'] = scale_value(design['cavity_options.coupler_options']['finger_length'], coupling_ratio)

        return design

    def get_updated_design_scaled_resonator_length_and_coupling_dim(self, design, res_length_ratio, coupling_dim_ratio):
        """
        Scales the resonator length and coupling dimension by a given ratio and updates the design dict accordingly.

        Args:
            design (dict): A dictionary containing the design parameters.
            res_length_ratio (float): The ratio by which to scale the resonator length.
            coupling_dim_ratio (float): The ratio by which to scale the coupling dimension.

        Returns:
            dict: Updated design dict with scaled areas.
        """

        # Check and scale resonator length
        design['cavity_options.cpw_options.total_length'] = scale_value(design['cavity_options.cpw_options.total_length'], res_length_ratio)

        # Check and scale coupling dimensions
        coupling_type = design.get('cavity_options.coupling_type', '')
        if coupling_type in ['capacitive', 'inductive']:
            design['cavity_options.coupler_options.coupling_length'] = scale_value(design['cavity_options.coupler_options.coupling_length'], coupling_dim_ratio)
        elif 'interdig' in coupling_type:
            # If finger_length is not in the coupler_options dictionary, set its default value
            if 'finger_length' not in design.get('cavity_options.coupler_options', {}):
                design['cavity_options.coupler_options']['finger_length'] = '50um'
            design['cavity_options.coupler_options']['finger_length'] = scale_value(design['cavity_options.coupler_options']['finger_length'], coupling_dim_ratio)

        return design


    def get_design(self, df):
        """
        Extracts the design parameters from the dataframe and returns a dict.
        """
        return df["design_options"].to_dict()[0]

    def get_param(self, design, param):
        """
        Extracts a specific parameter from the design dict.
        """
        raise NotImplementedError

    def interpolate_design(self, updated_design1, updated_design2):
        """
        Interpolates the design parameters of the resonator and qubit to the design dict.
        """
        raise NotImplementedError

    def get_coupling_parameters(self, f_q, f_r, alpha, g, wavelength_type,  Z_q=50, Z_r=50):
        """
        Numerically calculate the required qubit capacitance, coupling capacitance, and E_J.
        For demonstration, using theoretical expressions.

        Args:
            f_q (float): Qubit frequency in GHz.
            f_r (float): Resonator frequency in GHz.
            alpha (float): Qubit anharmonicity in MHz.
            g (float): Coupling strength in MHz.
            wavelength_type (str): Type of wavelength to use for calculating coupling capacitance.
            Z_q (float): Qubit impedance in Ohms.
            Z_r (float): Resonator impedance in Ohms.
        Returns:
            tuple: A tuple containing the calculated qubit capacitance, coupling capacitance, and E_J.
        """
        PHI_0 = 2.067833848E-15 # Magnetic flux quantum
        H = 6.62607015E-34  # Planck's constant
        Q = 1.602176634E-19  # Elementary charge

        scq.set_units("GHz")


        if "Transmon" in self.component_name:
            # alpha to gigahertz
            alpha = - alpha/1000 #to gigahertz
            E_J, E_C = TunableTransmon.find_EJ_EC(E01=f_q, anharmonicity=alpha, ncut=30)

            C_q = Q ** 2 / (2 * E_C)  # Qubit capacitance

            if wavelength_type == "lambda/4" or "quarter":
                C_r = np.pi / (4 * Z_r * f_r) # Resonator capacitance
            elif wavelength_type == "lambda/2" or "half":
                C_r = np.pi / (2 * Z_r * f_r)  # Resonator capacitance

            C_c = C_q * g * (C_r / (Q**2 * f_r))**1/2 * ((8*E_C)/E_J )**1/4  # Coupling capacitance (Here E_{C,q} = E_C)

            if E_J / E_C < 30:
                warnings.warn("E_J/E_C < 30, the design may not be optimal.")
            return C_q, C_r, C_c, E_J, E_C

        else:
            raise NotImplementedError("The method for calculating qubit parameters for the chosen qubit type is not implemented.")
