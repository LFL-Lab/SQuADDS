import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from squadds.calcs.transmon_cross import TransmonCrossHamiltonian
from squadds.core.metrics import *

"""
=====================================================================================
HELPER FUNCTIONS
=====================================================================================
"""
# Helper function to scale values with 'um' in them
def scale_value(value, ratio):
    """
    Scales the given value by the specified ratio.

    Args:
        - value (str): The value to be scaled, in the format 'Xum' where X is a number.
        - ratio (float): The scaling ratio.

    Returns:
        scaled_value (str): The scaled value in the format 'Xum' where X is the scaled number.
    """
    scaled_value = str(float(value.replace('um', '')) * ratio) + 'um'
    return scaled_value

"""
=====================================================================================
Analyzer
=====================================================================================
"""
class Analyzer:
    """
    The Analyzer class is responsible for analyzing designs and finding the closest designs based on target parameters.

    Methods:
        _add_target_params_columns(): Adds target parameter columns to the dataframe based on the selected system.
        _fix_cavity_claw_df(): Fixes the cavity claw DataFrame by renaming columns and updating values.
        _get_H_param_keys(): Gets the parameter keys for the Hamiltonian based on the selected system.
        target_param_keys(): Returns the target parameter keys.
        set_metric_strategy(strategy: MetricStrategy): Sets the metric strategy to use for calculating the distance metric.
        _outside_bounds(df: pd.DataFrame, params: dict, display=True) -> bool: Checks if entered parameters are outside the bounds of a dataframe.
        find_closest(target_params: dict, num_top: int, metric: str = 'Euclidean', display: bool = True): Finds the closest designs in the library based on the target parameters.
        get_interpolated_design(target_params: dict, metric: str = 'Euclidean', display: bool = True): Gets the interpolated design based on the target parameters.
        get_design(df): Extracts the design parameters from the dataframe and returns a dict.
    """

    __supported_metrics__ = ['Euclidean', 'Manhattan', 'Chebyshev', 'Weighted Euclidean' , 'Custom']
    __supported_estimation_methods__ = ['Interpolation']

    def __init__(self, db):
        """
        Initializes an instance of the Analysis class.

        Parameters:
            - db: The database object.

        Attributes:
            - db: The database object.
            - selected_component_name: The name of the selected component.
            - selected_component: The selected component.
            - selected_data_type: The selected data type.
            - selected_confg: The selected configuration.
            - selected_qubit: The selected qubit.
            - selected_cavity: The selected cavity.
            - selected_coupler: The selected coupler.
            - selected_system: The selected system.
            - df: The selected dataframe.
            - closest_df_entry: The closest dataframe entry.
            - closest_design: The closest design.
            - presimmed_closest_cpw_design: The presimmed closest CPW design.
            - presimmed_closest_qubit_design: The presimmed closest qubit design.
            - presimmed_closest_coupler_design: The presimmed closest coupler design.
            - interpolated_design: The interpolated design.
            - metric_strategy: The metric strategy (will be set dynamically).
            - custom_metric_func: The custom metric function.
            - metric_weights: The metric weights.
            - target_params: The target parameters.
            - H_param_keys: The H parameter keys.
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
        self.df = self.db.selected_df
        self.closest_df_entry = None
        self.closest_design = None
        self.presimmed_closest_cpw_design = None
        self.presimmed_closest_qubit_design = None
        self.presimmed_closest_coupler_design = None
        self.interpolated_design = None

        self.metric_strategy = None  # Will be set dynamically
        self.custom_metric_func = None
        self.metric_weights = None
        self.target_params = None
        
        self.H_param_keys = self._get_H_param_keys()
        
    def _add_target_params_columns(self):
        """
        Adds target parameter columns to the dataframe based on the selected system.

        If the selected system is "qubit", it adds qubit Hamiltonian parameters to the dataframe.
        If the selected system is "cavity_claw", it fixes the dataframe for the cavity_claw system.
        If the selected system is "coupler", it does nothing.
        If the selected system is ["qubit", "cavity_claw"] or ["cavity_claw", "qubit"], it fixes the dataframe for the cavity_claw system and adds cavity-coupled Hamiltonian parameters to the dataframe.

        Raises:
            a ValueError if the selected system is invalid.
        """
        #! TODO: make this more general and read the param keys from the database
        if self.selected_system == "qubit":
            qubit_H = TransmonCrossHamiltonian(self)
            qubit_H.add_qubit_H_params()
            self.df = qubit_H.df 
        elif self.selected_system == "cavity_claw":
            self._fix_cavity_claw_df()
        elif self.selected_system == "coupler":
            pass
        elif (self.selected_system == ["qubit","cavity_claw"]) or (self.selected_system == ["cavity_claw","qubit"]):
            self._fix_cavity_claw_df()
            qubit_H = TransmonCrossHamiltonian(self)
            qubit_H.add_cavity_coupled_H_params()
            self.df = qubit_H.df 
        else:
            raise ValueError("Invalid system.")
    
    def _fix_cavity_claw_df(self):
        """
        Fix the cavity claw DataFrame by renaming columns and updating values.

        If the columns 'cavity_frequency' or 'kappa' exist in the DataFrame, they will be renamed to
        'cavity_frequency_GHz' and 'kappa_kHz' respectively. The values in these columns will also be
        updated by multiplying them with appropriate conversion factors.

        Args:
            None

        Returns:
            None
        """
        if ("cavity_frequency" in self.df.columns) or ("kappa" in self.df.columns):
            self.df = self.df.rename(columns={"cavity_frequency": "cavity_frequency_GHz", "kappa": "kappa_kHz"})
            self.df["cavity_frequency_GHz"] = self.df["cavity_frequency_GHz"] * 1e-9
            self.df["kappa_kHz"] = self.df["kappa_kHz"] * 1e-3
        else:
            pass
    
    def _get_H_param_keys(self):
        """
        Get the parameter keys for the Hamiltonian (H) based on the selected system.

        Returns:
            list: A list of parameter keys for the Hamiltonian.
        
        Raises:
            ValueError: If the selected system is invalid.
        """
        #! TODO: make this more general and read the param keys from the database
        self.H_param_keys = None
        if self.selected_system == "qubit":
            self.H_param_keys = ["qubit_frequency_GHz", "anharmonicity_MHz"]
        elif self.selected_system == "cavity_claw":
            self.H_param_keys = ["resonator_type", "cavity_frequency_GHz", "kappa_kHz"]
        elif self.selected_system == "coupler":
            pass
        elif (self.selected_system == ["qubit","cavity_claw"]) or (self.selected_system == ["cavity_claw","qubit"]):
            self.H_param_keys = ["qubit_frequency_GHz", "anharmonicity_MHz", "resonator_type", "cavity_frequency_GHz", "kappa_kHz", "g_MHz"]
        else:
            raise ValueError("Invalid system.")
        return self.H_param_keys

    def target_param_keys(self):
        """
        Returns:
            list: The target parameter keys.
        """
        return self.H_param_keys

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
        Find the closest designs in the library based on the target parameters.

        Args:
            - target_params (dict): A dictionary containing the target parameters.
            - num_top (int): The number of closest designs to retrieve.
            - metric (str, optional): The distance metric to use for calculating distances. Defaults to 'Euclidean'.
            - display (bool, optional): Whether to display warnings for parameters outside of the library bounds. Defaults to True.

        Returns:
            - closest_df (DataFrame): A DataFrame containing the closest designs.

        Raises:
            - ValueError: If the specified metric is not supported or if num_top is bigger than the size of the library.
            - ValueError: If the metric is invalid.
        """
        ### Checks
        # Check for supported metric
        if metric not in self.__supported_metrics__:
            raise ValueError(f'`metric` must be one of the following: {self.__supported_metrics__}')
        # Check for improper size of library
        if (num_top > len(self.df)):
            raise ValueError('`num_top` cannot be bigger than size of read-in library.')

        self.target_params = target_params
        self._add_target_params_columns()

        # Log if parameters outside of library
        filtered_df = self.df[self.target_params]  # Filter DataFrame based on H_param_keys
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

        # Filter DataFrame based on target parameters that are string
        for param, value in target_params.items():
            if isinstance(value, str):
                filtered_df = filtered_df[filtered_df[param] == value]

        # Calculate distances
        distances = filtered_df.apply(lambda row: self.metric_strategy.calculate(target_params, row), axis=1)

        # Sort distances and get the closest ones
        sorted_indices = distances.nsmallest(num_top).index
        closest_df = self.df.loc[sorted_indices]

        # store the best design 
        self.closest_df_entry = closest_df.iloc[0]
        self.closest_design = closest_df.iloc[0]["design_options"]

        if len(self.selected_system) == 2: #! TODO: make this more general
            self.presimmed_closest_cpw_design = self.closest_df_entry["design_options_cavity_claw"]
            self.presimmed_closest_qubit_design = self.closest_df_entry["design_options_qubit"]

        return closest_df

    def get_interpolated_design(self,
                     target_params: dict,
                     metric: str = 'Euclidean',
                     display: bool = True):
        """
        """
        raise NotImplementedError
    

    def get_design(self, df):
        """
        Extracts the design parameters from the dataframe and returns a dict.

        Returns:
            dict: A dict containing the design parameters.
        """
        return df["design_options"].to_dict()[0]

    def get_param(self, design, param):
        """
        Extracts a specific parameter from the design dict.
        """
        raise NotImplementedError


    def closest_design_in_H_space(self):
        """
        Plots a scatter plot of the closest design in the H-space.

        This method creates a scatter plot with two subplots. The first subplot shows the relationship between 'cavity_frequency_GHz' and 'kappa_kHz', while the second subplot shows the relationship between 'anharmonicity_MHz' and 'g_MHz'. The scatter plot includes pre-simulated data, target data, and the closest design entry from the database.

        Returns:
            None
        """
        # Set Seaborn style and context
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.4)

        # Create a colormap for the scatter plot points
        viridis_cmap = plt.cm.get_cmap('viridis')
        color_sim = viridis_cmap(0.2)
        color_presim = viridis_cmap(0.9)
        color_database = viridis_cmap(0.6)

        # Create the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # First subplot: kappa_kHz vs fres
        ax1.scatter(x=self.df['cavity_frequency_GHz'], y=self.df['kappa_kHz'], color=color_presim, marker=".", s=50, label="Pre-Simulated")
        ax1.scatter(x=self.target_params["cavity_frequency_GHz"], y=self.target_params["kappa_kHz"], color='red', s=100, marker='x', label='Target')
        closest_fres = self.closest_df_entry["cavity_frequency_GHz"]
        closest_kappa_kHz = self.closest_df_entry["kappa_kHz"]
        ax1.scatter(closest_fres, closest_kappa_kHz, color=[color_database], s=100, marker='s', alpha=0.7, label='Closest')
        ax1.set_xlabel(r'$f_{res}$ (Hz)', fontweight='bold', fontsize=24)
        ax1.set_ylabel(r'$\kappa / 2 \pi$ (Hz)', fontweight='bold', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        legend1 = ax1.legend(loc='upper left', fontsize=16)
        for text in legend1.get_texts():
            text.set_fontweight('bold')

        # Second subplot: g vs alpha
        ax2.scatter(x=self.df['anharmonicity_MHz'], y=self.df['g_MHz'], color=color_presim, marker=".", s=50, label="Pre-Simulated")
        ax2.scatter(x=self.target_params["anharmonicity_MHz"], y=self.target_params["g_MHz"], color='red', s=100, marker='x', label='Target')
        closest_alpha = [self.closest_df_entry["anharmonicity_MHz"]]
        closest_g = [self.closest_df_entry["g_MHz"]]
        ax2.scatter(closest_alpha, closest_g, color=[color_database], s=100, marker='s', alpha=0.7, label='Closest')
        ax2.set_xlabel(r'$\alpha / 2 \pi$ (MHz)', fontweight='bold', fontsize=24)
        ax2.set_ylabel(r'$g / 2 \pi$ (MHz)', fontweight='bold', fontsize=24)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        legend2 = ax2.legend(loc='lower left', fontsize=16)
        for text in legend2.get_texts():
            text.set_fontweight('bold')

        plt.tight_layout()
        plt.show()
