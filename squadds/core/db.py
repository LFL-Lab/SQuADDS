import os
import platform
import pprint
import shutil
import sys
import warnings

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from tabulate import tabulate

from squadds.core.design_patterns import SingletonMeta
from squadds.core.utils import *

#* HANDLE WARNING MESSAGES
if sys.platform == "darwin":  # Checks if the operating system is macOS
    warnings.filterwarnings("ignore", category=UserWarning, module="pyaedt") # ANSYS is not a mac product

class SQuADDS_DB(metaclass=SingletonMeta):
    """
    A class representing the SQuADDS database.

    Methods:
        supported_components(): Get a list of supported components.
        supported_component_names(): Get a list of supported component names.
        supported_data_types(): Get a list of supported data types.
        _delete_cache(): Delete the dataset cache directory.
        supported_config_names(): Get a list of supported configuration names.
        get_configs(): Print the supported configuration names.
        get_component_names(component): Get a list of component names for a given component.
        view_component_names(component): Print the component names for a given component.
        view_datasets(): Print a table of available datasets.
        get_dataset_info(component, component_name, data_type): Print information about a specific dataset.
        view_all_contributors(): Print a table of all contributors.
        view_contributors_of_config(config): Print a table of contributors for a specific configuration.
        view_contributors_of(component, component_name, data_type): Print a table of contributors for a specific component, component name, and data type.
        select_components(component_dict): Select a configuration based on a component dictionary or string.
        select_system(components): Select a system based on a list of components or a single component.
        select_qubit(qubit): Select a qubit.
        select_cavity_claw(cavity): Select a cavity.
    """
    
    def __init__(self):
        """
        Constructor for the SQuADDS_DB class.

        Attributes:
            repo_name (str): The name of the repository.
            configs (list): List of supported configuration names.
            selected_component_name (str): The name of the selected component.
            selected_component (str): The selected component.
            selected_data_type (str): The selected data type.
            selected_confg (str): The selected configuration.
            selected_qubit (str): The selected qubit.
            selected_cavity (str): The selected cavity.
            selected_coupler (str): The selected coupler.
            selected_system (str): The selected system.
            selected_df (str): The selected dataframe.
            target_param_keys (str): The target parameter keys.
            units (str): The units.
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
        Returns a list of supported components based on the configurations.

        Returns:
            list: A list of supported components.
        """
        components = []
        for config in self.configs:
            components.append(config.split("-")[0])
        return components
    
    def supported_component_names(self):
        """
        Returns a list of supported component names extracted from the configs.

        Returns:
            list: A list of supported component names.
        """
        component_names = []
        for config in self.configs:
            component_names.append(config.split("-")[1])
        return component_names
    
    def supported_data_types(self):
        """
        Returns a list of supported data types.

        Returns:
            list: A list of supported data types.
        """
        data_types = []
        for config in self.configs:
            data_types.append(config.split("-")[2])
        return data_types

    def _delete_cache(self):
        """
        Deletes the cache directory for the specific dataset.
        """
        # Determine the root cache directory for 'datasets'
        # Default cache directory is '~/.cache/huggingface/datasets' on Unix systems
        # and 'C:\\Users\\<username>\\.cache\\huggingface\\datasets' on Windows
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
        
        # Adjust the path for Windows if necessary
        if platform.system() == "Windows":
            cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "huggingface", "datasets")

        # Define the specific dataset cache directory name
        dataset_cache_dir_name = "SQuADDS___s_qu_adds_db"

        # Path for the specific dataset cache
        dataset_cache_dir = os.path.join(cache_dir, dataset_cache_dir_name)

        # Check if the cache directory exists
        if os.path.exists(dataset_cache_dir):
            try:
                # Delete the dataset cache directory
                shutil.rmtree(dataset_cache_dir)
            except OSError as e:
                print(f"Error occurred while deleting cache: {e}")
        else:
            pass
        
    def supported_config_names(self):
        """
        Retrieves the supported configuration names from the repository.

        Returns:
            A list of supported configuration names.
        """
        self._delete_cache()
        configs = get_dataset_config_names(self.repo_name, download_mode='force_redownload')
        return configs

    def get_configs(self):
        """
        Returns the configurations stored in the database.

        Returns:
            list: A list of configuration names.
        """
        # pretty print the config names
        pprint.pprint(self.configs)

    def get_component_names(self, component=None):
        """
        Get the names of the components associated with a specific component.

        Args:
            component (str): The specific component to retrieve names for.

        Returns:
            list: A list of component names associated with the specified component.
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
        Prints the names of the components available in the database.

        Args:
            component (str): The specific component to view names for. If None, all component names will be printed.

        Returns:
            None
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


    def view_datasets(self):
        """
        View the datasets available in the database.

        This method retrieves the supported components, component names, and data types
        from the database and displays them in a tabular format.
        """
        components = self.supported_components()
        component_names = self.supported_component_names()
        data_types = self.supported_data_types()

        # Create a list of rows for the table
        table = [components, component_names, data_types]

        # Transpose the table (convert columns to rows)
        table = list(map(list, zip(*table)))

        # Print the table with headers
        print(tabulate(table, headers=["Component", "Component Name", "Data Available"],tablefmt="fancy_grid"))

    def get_dataset_info(self, component=None, component_name=None, data_type=None):
        """
        Retrieves and prints information about a dataset.

        Args:
            component (str): The component of the dataset.
            component_name (str): The name of the component.
            data_type (str): The type of data.

        Returns:
            None
        """
        # do checks
        if component is None:
            print("Please specify a component")
            return
        if component_name is None:
            print("Please specify a component type")
            return
        if data_type is None:
            print("Please specify a data type")
            return
        
        if component not in self.supported_components():
            print("Component not supported. Available components are:")
            print(self.supported_components())
            return
        
        if component_name not in self.supported_component_names():
            print("Component name not supported. Available component names are:")
            print(self.supported_component_names())
            return
        if data_type not in self.supported_data_types():
            print("Data type not supported. Available data types are:")
            print(self.supported_data_types())
            return
        
        # print the table of the dataset configs
        config = component + "-" + component_name + "-" + data_type
        
        dataset = load_dataset(self.repo_name, config)["train"]
        # describe the dataset and print in table format
        print("="*80)
        print("Dataset Features:")
        pprint.pprint(dataset.features, depth=2)
        print("\nDataset Description:")
        print(dataset.description)
        print("\nDataset Citation:")
        print(dataset.citation)
        print("\nDataset Homepage:")
        print(dataset.homepage)
        print("\nDataset License:")
        print(dataset.license)
        print("\nDataset Size in Bytes:")
        print(dataset.size_in_bytes)
        print("="*80)
        
    def view_all_contributors(self):
        """
        View all unique contributors and their relevant information.

        This method iterates through the configurations and extracts the relevant information
        of each contributor. It checks if the combination of uploader, PI, group, and institution
        is already in the list of unique contributors. If not, it adds the relevant information
        to the list. Finally, it prints the list of unique contributors in a tabular format.
        """
        # Placeholder for the full contributor info
        unique_contributors_info = []

        for config in self.configs:
            dataset = load_dataset(self.repo_name, config)["train"]
            configs_contrib_info = dataset["contributor"]
            
            for contrib_info in configs_contrib_info:
                # Extracting the relevant information
                relevant_info = {key: contrib_info[key] for key in ['uploader', 'PI', 'group', 'institution']}
                relevant_info['config'] = config  # Add the config to the relevant info

                # Check if this combination of info is already in the list
                if not any(existing_info['config'] == config and
                            existing_info['uploader'] == relevant_info['uploader'] and
                            existing_info['PI'] == relevant_info['PI'] and
                            existing_info['group'] == relevant_info['group'] and
                            existing_info['institution'] == relevant_info['institution']
                            for existing_info in unique_contributors_info):
                    unique_contributors_info.append(relevant_info)

        print(tabulate(unique_contributors_info, headers="keys", tablefmt="fancy_grid"))

    def view_contributors_of_config(self, config):
        """
        View the contributors of a specific configuration.

        Args:
            config (str): The name of the configuration.

        Returns:
            None
        """
        dataset = load_dataset(self.repo_name, config)["train"]
        configs_contrib_info = dataset["contributor"]
        unique_contributors_info = []
        
        for contrib_info in configs_contrib_info:
            # Extracting the relevant information
            relevant_info = {key: contrib_info[key] for key in ['uploader', 'PI', 'group', 'institution']}
            if relevant_info not in unique_contributors_info:
                unique_contributors_info.append(relevant_info)
        
        print(tabulate(unique_contributors_info, headers='keys', tablefmt="fancy_grid"))

    def view_contributors_of(self, component=None, component_name=None, data_type=None):
        """
        View contributors of a specific component, component name, and data type.

        Args:
            component (str): The component of interest.
            component_name (str): The name of the component.
            data_type (str): The type of data.

        Returns:
            None
        """
        config = component + "-" + component_name + "-" + data_type
        self.view_contributors_of_config(config)

    def select_components(self, component_dict=None):
        """
        Selects components based on the provided component dictionary or string.

        Args:
            component_dict (dict or str): A dictionary containing the component details
                (component, component_name, data_type) or a string representing the component.

        Returns:
            None

        """
        # check if dict or string
        if isinstance(component_dict, dict):
            config = component_dict["component"] + "-" + component_dict["component_name"] + "-" + component_dict["data_type"]
        elif isinstance(component_dict, str):
            config = component_dict
        print("Selected config: ", config)
        
    def select_system(self, components=None):
        """
        Selects the system and component(s) to be used.

        Args:
            components (list or str): The component(s) to be selected. If a list is provided,
                each component will be checked against the supported components. If a string
                is provided, it will be checked against the supported components.

        Returns:
            None

        Raises:
            None
        """
        # Validation and checks
        if isinstance(components, list):
            for component in components:
                if component not in self.supported_components():
                    print(f"Component `{component}` not supported. Available components are:")
                    print(self.supported_components())
                    return
                else:
                    self.selected_system = components

        elif isinstance(components, str):
            if components not in self.supported_components():
                print(f"Component `{components}` not supported. Available components are:")
                print(self.supported_components())
                return
            else:
                self.selected_system = components
                self.selected_component = components
    
    def select_qubit(self, qubit=None):
        """
        Selects a qubit and sets the necessary attributes for the selected qubit.

        Args:
            qubit (str): The name of the qubit to be selected.

        Raises:
            UserWarning: If the selected system is not specified or does not contain a qubit.

        Returns:
            None
        """
        # check whether selected_component is qubit
        if (self.selected_system == "qubit") or ("qubit" in self.selected_system):
            self.selected_qubit = qubit
            self.selected_component_name = qubit
            self.selected_data_type = "cap_matrix" # TODO: handle dynamically
        else:
            raise UserWarning("Selected system is either not specified or does not contain a qubit! Please check `self.selected_system`")
        
        # check if qubit is supported
        if self.selected_qubit not in self.supported_component_names():
            print(f"Qubit `{self.selected_qubit}` not supported. Available qubits are:")
            self.view_component_names("qubit")
            return

    def select_cavity_claw(self, cavity=None):
        """
        Selects a cavity claw component.

        Args:
            cavity (str): The name of the cavity to select.

        Raises:
            UserWarning: If the selected system is not specified or does not contain a cavity.

        Returns:
            None
        """
        # check whether selected_component is cavity
        if (self.selected_system == "cavity_claw") or ("cavity_claw" in self.selected_system):
            self.selected_cavity = cavity
            self.selected_component_name = cavity
            self.selected_data_type = "eigenmode" # TODO: handle dynamically
        else:
            raise UserWarning("Selected system is either not specified or does not contain a cavity! Please check `self.selected_system`")
        
        # check if cavity is supported
        if self.selected_cavity not in self.supported_component_names():
            print(f"Cavity `{self.selected_cavity}` not supported. Available cavities are:")
            self.view_component_names("cavity_claw")
            return

    def select_cavity(self, cavity=None):
        """
        Selects a cavity and sets the necessary attributes for further operations.

        Parameters:
            cavity (str): The name of the cavity to be selected.

        Raises:
            UserWarning: If the selected system is either not specified or does not contain a cavity.

        Returns:
            None
        """
        # check whether selected_component is cavity
        if (self.selected_system == "cavity") or ("cavity" in self.selected_system):
            self.selected_cavity = cavity
            self.selected_component_name = cavity
            self.selected_data_type = "eigenmode" # TODO: handle dynamically
        else:
            raise UserWarning("Selected system is either not specified or does not contain a cavity! Please check `self.selected_system`")
        
        # check if cavity is supported
        if self.selected_cavity not in self.supported_component_names():
            print(f"Cavity `{self.selected_cavity}` not supported. Available cavities are:")
            self.view_component_names("cavity")
            return
        
    def select_coupler(self, coupler=None):
        """
        Selects a coupler for the database.

        Args:
            coupler (str, optional): The name of the coupler to select. Defaults to None.

        Returns:
            None
        """
        #! TODO: fix this method to work on NCap coupler sims
        self.selected_coupler = coupler
        #self.selected_component_name = coupler
        #self.selected_data_type = "cap_matrix" # TODO: handle dynamically
        
        # check if coupler is supported
        if self.selected_coupler not in self.supported_component_names()+["CLT"]: # TODO: handle dynamically

            print(f"Coupler `{self.selected_coupler}` not supported. Available couplers are:")
            self.view_component_names("coupler")
            return

    def see_dataset(self, data_type=None, component=None, component_name=None):
        """
        View a dataset based on the provided data type, component, and component name.

        Args:
            data_type (str): The type of data to view.
            component (str): The component to use. If not provided, the selected system will be used.
            component_name (str): The name of the component. If not provided, the selected component name will be used.

        Returns:
            pandas.DataFrame: The flattened dataset.

        Raises:
            ValueError: If both system and component name are not defined.
            ValueError: If data type is not specified.
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
            print("Both system and component name must be defined.")
            return
        
        if data_type is None:
            print("Please specify a data type.")
            return
        
        # Check if the component is supported
        if component not in self.supported_components():
            print("Component not supported. Available components are:")
            print(self.supported_components())
            return
        
        # Check if the component name is supported
        if component_name not in self.supported_component_names():
            print("Component name not supported. Available component names are:")
            print(self.supported_component_names())
            return
        
        # Check if the data type is supported
        if data_type not in self.supported_data_types():
            print("Data type not supported. Available data types are:")
            print(self.supported_data_types())
            return

        # Construct the configuration string based on the provided or default values
        config = f"{component}-{component_name}-{data_type}"
        try:
            df = load_dataset(self.repo_name, config)["train"].to_pandas()
            return flatten_df_second_level(df)
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return


    def get_dataset(self, data_type=None, component=None, component_name=None):
        """
        Retrieves a dataset based on the specified data type, component, and component name.

        Args:
            data_type (str): The type of data to retrieve.
            component (str): The component to retrieve the data from.
            component_name (str): The name of the component to retrieve the data from.

        Returns:
            pandas.DataFrame: The retrieved dataset.

        Raises:
            ValueError: If the system and component name are not defined.
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
            print("Both system and component name must be defined.")
            return
        
        if data_type is None:
            print("Please specify a data type.")
            return
        
        # Check if the component is supported
        if component not in self.supported_components():
            print("Component not supported. Available components are:")
            print(self.supported_components())
            return
        
        # Check if the component name is supported
        if component_name not in self.supported_component_names():
            print("Component name not supported. Available component names are:")
            print(self.supported_component_names())
            return
        
        # Check if the data type is supported
        if data_type not in self.supported_data_types():
            print("Data type not supported. Available data types are:")
            print(self.supported_data_types())
            return

        # Construct the configuration string based on the provided or default values
        config = f"{component}-{component_name}-{data_type}"
        try:
            df = load_dataset(self.repo_name, config, cache_dir=None)["train"].to_pandas()
            self._set_target_param_keys(df)
            return flatten_df_second_level(df)
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return

    def create_system_df(self):
        """
        Creates and returns a DataFrame based on the selected system.

        If the selected system is a single component, it retrieves the dataset based on the selected data type, component, and component name.
        If a coupler is selected, the DataFrame is filtered by the coupler.
        The resulting DataFrame is stored in the `selected_df` attribute.

        If the selected system is a list of components (qubit and cavity), it retrieves the qubit and cavity DataFrames.
        The qubit DataFrame is obtained based on the selected qubit component name and data type "cap_matrix".
        The cavity DataFrame is obtained based on the selected cavity component name and data type "eigenmode".
        The qubit and cavity DataFrames are merged into a single DataFrame using the merger terms ['claw_width', 'claw_length', 'claw_gap'].
        The resulting DataFrame is stored in the `selected_df` attribute.

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
        Creates a merged DataFrame by merging the qubit and cavity DataFrames based on the specified merger terms.

        Args:
            qubit_df (pandas.DataFrame): The DataFrame containing qubit data.
            cavity_df (pandas.DataFrame): The DataFrame containing cavity data.
            merger_terms (list): A list of column names to be used for merging the DataFrames. Defaults to None.

        Returns:
            pandas.DataFrame: The merged DataFrame.

        Raises:
            None
        """
        for merger_term in merger_terms:
            # process the dfs to make them ready for merger
            qubit_df[merger_term] = qubit_df['design_options'].apply(lambda x: x['connection_pads']['readout'][merger_term])
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
        Clears the selected component, data type, qubit, cavity, coupler, and system.
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
        Sets the target parameter keys based on the provided DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing simulation results.

        Raises:
            UserWarning: If no selected system DataFrame is created or if target_param_keys is not None or a list.
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
        # TODO: needs implementation 
        raise NotImplementedError()

    def unselect(self, param):
        """
        Unselects the specified parameter.

        Parameters:
        param (str): The parameter to unselect. Valid options are:
            - "component"
            - "component_name"
            - "data_type"
            - "qubit"
            - "cavity_claw"
            - "coupler"
            - "system"

        Returns:
        None
        """
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
