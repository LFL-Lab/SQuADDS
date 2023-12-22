from squadds.core.design_patterns import SingletonMeta
from datasets import get_dataset_config_names
from datasets import load_dataset
from tabulate import tabulate
import pprint
import pandas as pd
from squadds.core.utils import *

class SQuADDS_DB(metaclass=SingletonMeta):
    """
    SQuADDS_DB is a singleton class that represents the database for the SQuADDS project.
    It provides methods to interact with the database and retrieve information about supported components, datasets, contributors, etc.
    """

    def __init__(self):
        """
        Initializes the SQuADDS_DB instance.
        """
        self.repo_name = "SQuADDS/SQuADDS_DB"
        self.configs = self.supported_config_names()
        self.selected_component_name = None
        self.selected_component = None
        self.selected_data_type = None
        self.selected_confg = None
        self.selected_qubit = None
        self.selected_cavity = None
        self.selected_coupler = None
        self.selected_system = None
        self.selected_df = None
        self.target_param_keys = None
        self.units = None

    def supported_components(self):
        """
        Returns a list of supported components based on the available dataset configurations.

        Returns:
            list: A list of supported components.
        """
        components = []
        for config in self.configs:
            components.append(config.split("-")[0])
        return components
    
    def supported_component_names(self):
        """
        Returns a list of supported component names based on the available dataset configurations.

        Returns:
            list: A list of supported component names.
        """
        component_names = []
        for config in self.configs:
            component_names.append(config.split("-")[1])
        return component_names
    
    def supported_data_types(self):
        """
        Returns a list of supported data types based on the available dataset configurations.

        Returns:
            list: A list of supported data types.
        """
        data_types = []
        for config in self.configs:
            data_types.append(config.split("-")[2])
        return data_types

    def supported_config_names(self):
        """
        Returns a list of supported dataset configuration names.

        Returns:
            list: A list of supported dataset configuration names.
        """
        configs = get_dataset_config_names(self.repo_name)
        return configs

    def get_configs(self):
        """
        Prints the supported dataset configuration names.
        """
        pprint.pprint(self.configs)

    def get_component_names(self, component=None):
        """
        Returns a list of component names for the specified component.

        Args:
            component (str): The component for which to retrieve the component names.

        Returns:
            list: A list of component names.
        """
        if component is None:
            print("Please specify a component")
            return
        if component not in self.supported_components():
            print("Component not supported. Available components are:")
            print(self.supported_components())
            return
        else:
            component_names = []
            for config in self.configs:
                if component in config:
                    component_names.append(config.split("-")[1])
            return component_names
        
    def view_component_names(self, component=None):
        """
        Prints the component names for the specified component.

        Args:
            component (str): The component for which to view the component names.
        """
        if component is None:
            print("Please specify a component")
        if component not in self.supported_components():
            print("Component not supported. Available components are:")
            print(self.supported_components()+["CLT"]) #TODO: handle dynamically
        else:
            component_names = []
            for config in self.configs:
                if component in config:
                    component_names.append(config.split("-")[1])
            print(component_names+["CLT"]) #TODO: handle dynamically

        #self.selected_data_type = "cap_matrix" # TODO: handle dynamically
        
        # check if coupler is supported
        if self.selected_coupler not in self.supported_component_names()+["CLT"]: # TODO: handle dynamically

            print(f"Coupler `{self.selected_coupler}` not supported. Available couplers are:")
            self.view_component_names("coupler")
            return

    def get_dataset(self, data_type=None, component=None, component_name=None):
        """
        Retrieves a dataset based on the specified data type, component, and component name.

        Args:
            data_type (str): The type of data to retrieve.
            component (str): The component to retrieve the dataset from.
            component_name (str): The name of the component to retrieve the dataset from.

        Returns:
            pandas.DataFrame: The retrieved dataset.

        Raises:
            ValueError: If the system or component name is not defined.
            ValueError: If the data type is not specified.
            ValueError: If the component is not supported.
            ValueError: If the component name is not supported.
            ValueError: If the data type is not supported.
            Exception: If an error occurs while loading the dataset.
        """
        # Use the instance attributes if the user does not provide them
        component = component if component is not None else self.selected_system
        component_name = component_name if component_name is not None else self.selected_component_name
        
        # Check if system and component_name are still None
        if component is None or component_name is None:
            raise ValueError("Both system and component name must be defined.")
        
        if data_type is None:
            raise ValueError("Please specify a data type.")
        
        # Check if the component is supported
        if component not in self.supported_components():
            raise ValueError("Component not supported. Available components are: {}".format(self.supported_components()))
        
        # Check if the component name is supported
        if component_name not in self.supported_component_names():
            raise ValueError("Component name not supported. Available component names are: {}".format(self.supported_component_names()))
        
        # Check if the data type is supported
        if data_type not in self.supported_data_types():
            raise ValueError("Data type not supported. Available data types are: {}".format(self.supported_data_types()))

        # Construct the configuration string based on the provided or default values
        config = f"{component}-{component_name}-{data_type}"
        try:
            df = load_dataset(self.repo_name, config)["train"].to_pandas()
            self._set_target_param_keys(df)
            return flatten_df_second_level(df)
        except Exception as e:
            raise Exception("An error occurred while loading the dataset: {}".format(e))

    def create_system_df(self):
        """
        Creates and returns a DataFrame based on the selected system.

        If the selected system is a single component, it retrieves the dataset based on the selected data type, component, and component name.
        If a coupler is selected, it filters the DataFrame by the coupler.
        The resulting DataFrame is stored in `self.selected_df`.

        If the selected system is a list of components (qubit and cavity), it retrieves the qubit and cavity DataFrames.
        It then creates a new DataFrame by merging the qubit and cavity DataFrames using the specified merger terms.
        The resulting DataFrame is stored in `self.selected_df`.

        Raises:
            UserWarning: If the selected system is either not specified or does not contain a cavity.

        Returns:
            pandas.DataFrame: The created DataFrame based on the selected system.
        """
        if self.selected_system is None:
            print("Selected system is not defined.")
            return
        elif isinstance(self.selected_system, str):
            df = self.get_dataset(data_type=self.selected_data_type, component=self.selected_component, component_name=self.selected_component_name)
            # if coupler is selected, filter by coupler
            if self.selected_coupler is not None:
                df = filter_df_by_conditions(df, {"coupler": self.selected_coupler}) 
            self.selected_df = df
        elif isinstance(self.selected_system, list):
            # get the qubit and cavity dfs
            qubit_df = self.get_dataset(data_type="cap_matrix", component="qubit", component_name=self.selected_qubit) #TODO: handle dynamically
            cavity_df = self.get_dataset(data_type="eigenmode", component="cavity_claw", component_name=self.selected_cavity) #TODO: handle dynamically
            df = self.create_qubit_cavity_df(qubit_df, cavity_df, merger_terms=['claw_width', 'claw_length', 'claw_gap']) #TODO: handle with user awareness
            self.selected_df = df
        else:
            raise UserWarning("Selected system is either not specified or does not contain a cavity! Please check `self.selected_system`")
        return df

    def create_qubit_cavity_df(self, qubit_df, cavity_df, merger_terms=None):
        """
        Creates a merged DataFrame by merging the qubit_df and cavity_df based on the specified merger terms.

        Args:
            qubit_df (pandas.DataFrame): DataFrame containing qubit data.
            cavity_df (pandas.DataFrame): DataFrame containing cavity data.
            merger_terms (list): List of column names to be used for merging the DataFrames. Defaults to None.

        Returns:
            pandas.DataFrame: Merged DataFrame with qubit and cavity data.

        Raises:
            None
        """
        for merger_term in merger_terms:
            # process the dfs to make them ready for merger
            qubit_df[merger_term] = qubit_df['design_options'].apply(lambda x: x['connection_pads']['c'][merger_term])
            cavity_df[merger_term] = cavity_df['design_options'].apply(lambda x: x['claw_opts']['connection_pads']['readout'][merger_term])

        # Merging the data frames based on merger terms
        merged_df = pd.merge(qubit_df, cavity_df, on=merger_terms, how="inner", suffixes=('_qubit', '_cavity_claw'))

        # Dropping the merger terms
        merged_df.drop(columns=merger_terms, inplace=True)

        # Combining the qubit and cavity design options into one
        merged_df['design_options'] = merged_df.apply(create_unified_design_options, axis=1)

        return merged_df

    def unselect_all(self):
        """
        Unselects all the components and data types in the system.
        """
        self.selected_component_name = None
        self.selected_component = None
        self.selected_data_type = None
        self.selected_qubit = None
        self.selected_cavity = None
        self.selected_coupler = None
        self.selected_system = None

    def show_selections(self):
        """
        Prints the selected system, component, and data type.

        If the selected system is a list, it prints the selected qubit, cavity, coupler, and system.
        If the selected system is a string, it prints the selected component, component name, data type, system, and coupler.
        """
        if isinstance(self.selected_system, list): #TODO: handle dynamically
            print("Selected qubit: ", self.selected_qubit)
            print("Selected cavity: ", self.selected_cavity)
            print("Selected coupler: ", self.selected_coupler)
            print("Selected system: ", self.selected_system)
        elif isinstance(self.selected_system, str):
            print("Selected component: ", self.selected_component)
            print("Selected component name: ", self.selected_component_name)
            print("Selected data type: ", self.selected_data_type)
            print("Selected system: ", self.selected_system)
            print("Selected coupler: ", self.selected_coupler)

    def _set_target_param_keys(self, df):
        """
        Sets the target parameter keys based on the provided dataframe.

        Args:
            df (pandas.DataFrame): The dataframe containing simulation results.

        Raises:
            UserWarning: If no selected system dataframe is created or if target_param_keys is not None or a list.

        """
        # ensure selected_df is not None
        if self.selected_system is None:
            raise UserWarning("No selected system df is created. Please check `self.selected_df`")
        else:
            # check if self.target_param_keys is None
            if self.target_param_keys is None:
                self.target_param_keys = get_sim_results_keys(df)
            #check if target_param_keys is type list and system has more than one element
            elif isinstance(self.target_param_keys, list) and len(self.selected_system) == 2:
                self.target_param_keys += get_sim_results_keys(df)
            #check if target_param_keys is type list and system has only one element
            elif isinstance(self.target_param_keys, list) and len(self.selected_system) != 1:
                self.target_param_keys = get_sim_results_keys(df)
            else:
                raise UserWarning("target_param_keys is not None or a list. Please check `self.target_param_keys`")

            # update the attribute to remove any elements that start with "unit"
            self.target_param_keys = [key for key in self.target_param_keys if not key.startswith("unit")]
    
    def _get_units(self, df):
        """
        Retrieves the units from the given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.

        Returns:
            list: A list of units extracted from the DataFrame.
        """
        # TODO: needs implementation 
        raise NotImplementedError()

    def unselect(self, param):
        if param == "component":
            self.selected_component = None
        elif param == "component_name":
            self.selected_component_name = None
        elif param == "data_type":
            self.selected_data_type = None
        elif param == "qubit":
            self.selected_qubit = None
        elif param == "cavity_claw":
            self.selected_cavity = None
        elif param == "coupler":
            self.selected_coupler = None
        elif param == "system":
            self.selected_system = None
        else:
            print("Please specify a valid parameter to unselect.")
            return
    
    def show_selected_system(self):
        raise NotImplementedError("Waiting on Andre's code")