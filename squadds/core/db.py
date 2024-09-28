"""
!TODO: add FULL support for half-wave cavity
"""
import json
import os
import pprint
import sys
import warnings

import pandas as pd
import requests
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import login
from tabulate import tabulate
from tqdm import tqdm

from squadds.core.design_patterns import SingletonMeta
from squadds.core.processing import *
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
            selected_resonator_type (str): The selected resonator type.
            selected_system (str): The selected system.
            selected_df (str): The selected dataframe.
            target_param_keys (str): The target parameter keys.
            units (str): The units.
            _internal_call (bool): Flag to track internal calls.
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
        self.selected_resonator_type = None
        self.selected_system = None
        self.selected_df = None
        self.qubit_df = None
        self.cavity_df = None
        self.coupler_df = None
        self.target_param_keys = None
        self.units = None
        self.measured_device_database = None
        self._internal_call = False  # Flag to track internal calls
        self.hwc_fname = "half-wave-cavity_df.parquet"
        self.merged_df_hwc_fname = "qubit_half-wave-cavity_df.parquet"
        #self.merger_terms = ['claw_width', 'claw_length', 'claw_gap']
        self.claw_merger_terms = ['claw_length'] # 07/2024 -> claw_length is the only parameter that is common between qubit and cavity
        self.ncap_merger_terms = ['prime_width', 'prime_gap', 'second_width', 'second_gap']

    def check_login(self):
        """
        Checks if the user is logged in to Hugging Face.
        """
        token = HfFolder.get_token()
        if not token:
            print("You are not logged in. Please login to Hugging Face.")
            login()  # This will prompt the user to login

    def get_existing_files(self):
        """
        Retrieves the list of existing files in the repository.
        
        Returns:
            list: A list of existing file names in the repository.
        """
        api = HfApi()
        repo_info = api.dataset_info(repo_id=self.repo_name)
        existing_files = [file.rfilename for file in repo_info.siblings]
        return existing_files

    def upload_dataset(self, file_paths, repo_file_names, overwrite=False):
        """
        Uploads a dataset to the repository.
        
        Args:
            file_paths (list): A list of file paths to upload.
            repo_file_names (list): A list of file names to use in the repository.
            overwrite (bool): Whether to overwrite an existing dataset. Defaults to False. 
        """
        self.check_login()
        api = HfApi()
        existing_files = self.get_existing_files()

        for file_path, repo_file_name in zip(file_paths, repo_file_names):
            if repo_file_name in existing_files and not overwrite:
                print(f"File {repo_file_name} already exists in the repository. Skipping upload.")
                continue
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=repo_file_name,
                    repo_id=self.repo_name,
                    repo_type="dataset",
                )
                print(f"Uploaded {repo_file_name} to {self.repo_name}.")
            except Exception as e:
                print(f"Failed to upload {repo_file_name}: {e}")

    def supported_components(self):
        """
        Returns a list of supported components based on the configurations.

        Returns:
            list: A list of supported components.
        """
        components = []
        for config in self.configs:
            try:
                components.append(config.split("-")[0])
            except:
                pass
            try:
                components.append(config.split("-")[0])
            except:
                pass
        return components
    
    def supported_component_names(self):
        """
        Returns a list of supported component names extracted from the configs.

        Returns:
            list: A list of supported component names.
        """
        component_names = []
        for config in self.configs:
            try:
                component_names.append(config.split("-")[1])
            except:
                pass
            try:
                component_names.append(config.split("-")[1])
            except:
                pass
        return component_names
    
    def supported_data_types(self):
        """
        Returns a list of supported data types.

        Returns:
            list: A list of supported data types.
        """
        data_types = []
        for config in self.configs:
            try:
                data_types.append(config.split("-")[2])
            except:
                pass
            try:
                data_types.append(config.split("-")[2])
            except:
                pass
        return data_types

        
    def supported_config_names(self):
        """
        Retrieves the supported configuration names from the repository.

        Returns:
            A list of supported configuration names.
        """
        delete_HF_cache()
        configs = get_dataset_config_names(self.repo_name, download_mode='force_redownload')
        # if there are not two "-" in the config name, remove it (since it does conform to the simulation naming convention)
        configs = [config for config in configs if config.count('-') == 2]
        # if there are not two "-" in the config name, remove it (since it does conform to the simulation naming convention)
        configs = [config for config in configs if config.count('-') == 2]
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
            print(self.supported_components()+["CLT"]) #TODO: handle dynamically
            return
        else:
            component_names = []
            for config in self.configs:
                if component in config:
                    component_names.append(config.split("-")[1])
            return component_names+["CLT"]
        
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
        component_urls = [f"https://github.com/LFL-Lab/SQuADDS/tree/master/docs/_static/images/{name}.png" for name in component_names]
        
        component_images = []
        
        for url in component_urls:
            component_images.append(url)
            table = [components, component_names, data_types, component_images]
        component_urls = [f"https://github.com/LFL-Lab/SQuADDS/tree/master/docs/_static/images/{name}.png" for name in component_names]
        
        component_images = []
        
        for url in component_urls:
            component_images.append(url)
            table = [components, component_names, data_types, component_images]

        # Transpose the table (convert columns to rows)
        table = list(map(list, zip(*table)))

        # Remove duplicate entries in table
        table = [list(x) for x in set(tuple(x) for x in table)]

        # Print the table with headers
        print(tabulate(table, headers=["Component", "Component Name", "Data Available", "Component Image"],tablefmt="grid"))

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
        View all unique contributors and their relevant information from simulation configurations.

        This method iterates through the simulation configurations and extracts the relevant information
        of each contributor. It checks if the combination of uploader, PI, group, and institution
        is already in the list of unique contributors. If not, it adds the relevant information
        to the list. Finally, it prints the list of unique contributors in a tabular format with a banner.
        """
        view_contributors_from_rst('../docs/source/developer/index.rst')

    def view_all_simulation_contributors(self):
        """
        View all unique simulation contributors and their relevant information.
        """
        # Placeholder for the full contributor info
        unique_contributors_info = []

        banner = "=" * 80
        title = "SIMULATION DATA CONTRIBUTORS"
        print(f"\n{banner}\n{title.center(80)}\n{banner}\n")

        for config in self.configs:
            dataset = load_dataset(self.repo_name, config)["train"]
            configs_contrib_info = dataset["contributor"]

            for contrib_info in configs_contrib_info:
                # Extracting the relevant information
                relevant_info = {
                    "Uploader": contrib_info.get('uploader', 'N/A'),
                    "PI": contrib_info.get('PI', 'N/A'),
                    "Group": contrib_info.get('group', 'N/A'),
                    "Institution": contrib_info.get('institution', 'N/A'),
                    "Config": config  # Add the config to the relevant info
                }

                # Check if this combination of info is already in the list
                if not any(existing_info['Config'] == config and
                        existing_info['Uploader'] == relevant_info['Uploader'] and
                        existing_info['PI'] == relevant_info['PI'] and
                        existing_info['Group'] == relevant_info['Group'] and
                        existing_info['Institution'] == relevant_info['Institution']
                        for existing_info in unique_contributors_info):
                    unique_contributors_info.append(relevant_info)

        print(tabulate(unique_contributors_info, headers="keys", tablefmt="grid"))
        print(f"\n{banner}\n")  # End with a banner

    def get_measured_devices(self):
        """
        Retrieve all measured devices with their corresponding design codes, paper links, images, foundries, and fabrication recipes.

        Returns:
            pd.DataFrame: A DataFrame containing the name, design code, paper link, image, foundry, and fabrication recipe for each device.
        """
        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]

        all_devices_info = []

        for entry in zip(dataset["contrib_info"], dataset["design_code"], dataset["paper_link"], 
                        dataset["image"], dataset["foundry"], dataset["fabrication_recipe"], dataset["substrate"], dataset["materials"], dataset["junction_style"], dataset["junction_material"]):
            contrib_info, design_code, paper_link, image, foundry, recipe, substrate, materials, junction_style, junction_materials = entry

            device_info = {
                "Name": contrib_info.get('name', 'N/A'),
                "Design Code": design_code,
                "Paper Link": paper_link,
                "Image": image,
                "Foundry": foundry,
                "Substrate": substrate,
                "Materials": materials,
                "Junction Style": junction_style,
                "Junction Materials": junction_materials,
                # "Fabrication Recipe": recipe
            }
            all_devices_info.append(device_info)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(all_devices_info)
        
        return df

    def view_measured_devices(self):
        """
        View all measured devices with their corresponding design codes, paper links, images, foundries, and fabrication recipes.

        This method retrieves and displays the relevant information for each device in the dataset in a well-formatted table.
        """
        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]

        all_devices_info = []

        for entry in zip(dataset["contrib_info"], dataset["design_code"], dataset["paper_link"], 
                        dataset["image"], dataset["foundry"], dataset["fabrication_recipe"]):
            contrib_info, design_code, paper_link, image, foundry, recipe = entry

            device_info = {
                "Name": contrib_info.get('name', 'N/A'),
                "Design Code": design_code,
                "Paper Link": paper_link,
                "Image": image,
                "Foundry": foundry,
                "Fabrication Recipe": recipe
            }
            all_devices_info.append(device_info)

        # Prepare the data for tabular display
        headers = ["Name", "Design Code", "Paper Link", "Image", "Foundry", "Fabrication Recipe"]
        rows = [[device_info[header] for header in headers] for device_info in all_devices_info]

        # Print the table with tabulate
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left", numalign="left"))

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
        
        print(tabulate(unique_contributors_info, headers='keys', tablefmt="grid"))

    def view_contributors_of(self, component=None, component_name=None, data_type=None, measured_device_name=None):
        """
        View contributors of a specific component, component name, and data type.

        Args:
            component (str): The component of interest.
            component_name (str): The name of the component.
            data_type (str): The type of data.
            measured_device_name (str): The name of the measured device.

        Returns:
            None
        """
        config = component + "-" + component_name + "-" + data_type
        try:
            print("="*80)
            print(f"\t\t\tMeasured Device Contributor(s):")
            print("="*80)
            self.view_device_contributors_of(component, component_name, data_type)
        except:
            pass
        try:
            print("="*80)
            print(f"\t\t\tSimulation Data Contributor(s):")
            print("="*80)
            self.view_contributors_of_config(config)
        except:
            pass

    def view_simulation_results(self, device_name):
        """
        View the simulation results of a specific device specified with a device name.

        Args:
           device_name (str): the name of the experimentally validated device within the database.

        Returns:
            dict: a dict of sim results.
        """       
        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]
        configs_contrib_info = dataset["contrib_info"]
        simulation_info = dataset["sim_results"]
            
        for contrib_info, sim_results in zip(configs_contrib_info, simulation_info):
                if contrib_info['name'] == device_name:
                    return sim_results
        return {}

    def get_device_contributors_of(self, component=None, component_name=None, data_type=None):
        """
        View the reference/source experimental device that was used to validate a specific simulation configuration.  
        
        Args:
            component (str): The component of interest.
            component_name (str): The name of the component.
            data_type (str): The type of data.

        Returns: 
            dict: The relevant contributor information.
        """
        if not (component and component_name and data_type):
            return "Component, component_name, and data_type must all be provided."

        config = f"{component}-{component_name}-{data_type}"
        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]
        
        for entry in zip(dataset["contrib_info"], dataset["sim_results"]):
            contrib_info, sim_results = entry
            
            if config in sim_results:
                relevant_info = {
                    "Foundry": contrib_info.get("foundry", "N/A"),
                    "PI": contrib_info.get("PI", "N/A"),
                    "Group": contrib_info.get("group", "N/A"),
                    "Institution": contrib_info.get("institution", "N/A"),
                    "Measured By": ", ".join(contrib_info.get("measured_by", [])),
                    "Reference Device Name": contrib_info.get("name", "N/A"),
                    "Uploader": contrib_info.get("uploader", "N/A")
                }
                
                print(tabulate(relevant_info.items(), tablefmt="grid"))
                return relevant_info

        return None

    def view_device_contributors_of(self, component=None, component_name=None, data_type=None):
        """
        View the reference/source experimental device that was used to validate a specific simulation configuration.  
        
        Args:
            component (str): The component of interest.
            component_name (str): The name of the component.
            data_type (str): The type of data.

        Returns: 
            str: The name of the experimentally validated reference device, or an error message if not found.
        """
        if not (component and component_name and data_type):
            return "Component, component_name, and data_type must all be provided."

        config = f"{component}-{component_name}-{data_type}"
        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]
        
        for entry in zip(dataset["contrib_info"], dataset["sim_results"]):
            contrib_info, sim_results = entry
            
            if config in sim_results:
                relevant_info = {
                    "Foundry": contrib_info.get("foundry", "N/A"),
                    "PI": contrib_info.get("PI", "N/A"),
                    "Group": contrib_info.get("group", "N/A"),
                    "Institution": contrib_info.get("institution", "N/A"),
                    "Measured By": ", ".join(contrib_info.get("measured_by", [])),
                    "Reference Device Name": contrib_info.get("name", "N/A"),
                    "Uploader": contrib_info.get("uploader", "N/A")
                }
                
                print(tabulate(relevant_info.items(), tablefmt="grid"))

        return "The reference device could not be retrieved."


    def view_reference_device_of(self, component=None, component_name=None, data_type=None):
        """
        View the reference/source experimental device that was used to validate a specific simulation configuration.  
        
        Args:
            component (str): The component of interest.
            component_name (str): The name of the component.
            data_type (str): The type of data.

        """
        if not (component and component_name and data_type):
            return "Component, component_name, and data_type must all be provided."

        config = f"{component}-{component_name}-{data_type}"
        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]
        
        for entry in zip(dataset["contrib_info"], dataset["sim_results"], dataset["design_code"], 
                        dataset["paper_link"], dataset["image"], dataset["foundry"], dataset["fabrication_recipe"]):
            contrib_info, sim_results, design_code, paper_link, image, foundry, recipe = entry
            
            if config in sim_results:
                combined_info = {
                    "Design Code": design_code,
                    "Paper Link": paper_link,
                    "Image": image,
                    "Foundry": foundry,
                    "Fabrication Recipe": recipe
                }
                combined_info.update(contrib_info)
                
                print(tabulate(combined_info.items(), tablefmt="grid"))


    def view_recipe_of(self, device_name):
        """
        Retrieve the foundry and fabrication recipe information for a specified device.
        
        Args:
            device_name (str): The name of the device to retrieve information for.
        
        Returns:
            dict: A dictionary containing foundry and fabrication recipe information.
        """
        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]
        
        for contrib_info, foundry, recipe, github_url in zip(dataset["contrib_info"], dataset["foundry"], dataset["fabrication_recipe"], dataset["design_code"],):
            if contrib_info['name'] == device_name:
                # append tree/main/Fabrication to the github_url
                github_url = f"{github_url}/tree/main/Fabrication"
                # Prepare the data for tabulation
                data = [["Foundry", foundry], ["Fabublox Link", recipe], ["Fabrication Recipe Links", github_url]]
                
                # Print the data in a tabulated format
                print(tabulate(data, tablefmt="grid"))
                return
        
        print("Error: Device not found in the dataset.")
  
    def view_reference_devices(self):
        """
        View all unique reference (experimental) devices and their relevant information.

        This method iterates through the configurations and extracts the chip's name within the SQuADDS DB, group, and who the chip was measured by. 
        It also finds the simulation results for the device.It checks if the combination of simulation results uploader, PI, group, and institution
        is already in the list of unique contributors. If not, it adds the relevant information to the list. 
        Finally, it prints the list of unique devices in a tabular format.

        """

        dataset = load_dataset(self.repo_name, 'measured_device_database')["train"]
        configs_contrib_info = dataset["contrib_info"]
        unique_contributors_info = []

        for contrib_info in configs_contrib_info:

            relevant_info = {key: contrib_info[key] for key in ['name', 'group', 'measured_by']}
 
            device_name = contrib_info['name']
            relevant_info['simulations']= self.view_simulation_results(device_name)
            if relevant_info not in unique_contributors_info:
                unique_contributors_info.append(relevant_info)

        print(tabulate(unique_contributors_info, headers='keys', tablefmt="grid"))


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
        
    def select_resonator_type(self, resonator_type):
        """
        Select the coupler based on the resonator type.

        Args:
            resonator_type (str): The type of resonator, e.g., "quarter" or "half".
        """
        resonator_to_coupler = {
            "quarter": "CLT",
            "half": "NCap"
        }

        if resonator_type not in resonator_to_coupler:
            raise ValueError(f"Invalid resonator type: {resonator_type}. Must be one of {list(resonator_to_coupler.keys())}.")

        self._internal_call = True  # Set the flag to indicate an internal call
        self.select_coupler(resonator_to_coupler[resonator_type])
        self.selected_resonator_type = resonator_type
        self._internal_call = False  # Reset the flag after the call

    def select_coupler(self, coupler=None):
        """
        Selects a coupler for the database.

        Args:
            coupler (str, optional): The name of the coupler to select. Defaults to None.

        Returns:
            None
        """
        if not self._internal_call:
            print("WARNING:DeprecationWarning: select_coupler() is deprecated and will be removed in a future release. Use select_resonator_type() instead.")
            warnings.warn(
                "select_coupler() is deprecated and will be removed in a future release. Use select_resonator_type() instead.",
                PendingDeprecationWarning
            )
        # E
        #! TODO: fix this method to work on CapNInterdigitalTee coupler sims
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

    def create_system_df(self, parallelize=False, num_cpu=None):
        """
        Creates and returns a DataFrame based on the selected system.

        Args:
            parallelize (bool): Whether to use multiprocessing to speed up the merging. Defaults to False.
            num_cpu (int): The number of CPU cores to use for multiprocessing. If not specified, the function will use the maximum number of available cores.

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
            raise UserWarning("Selected system is not defined.")
        
        if isinstance(self.selected_system, str):
            df = self._create_single_component_df()
        elif isinstance(self.selected_system, list):
            df = self._create_multi_component_df(parallelize, num_cpu)
        else:
            raise UserWarning("Selected system is either not specified or does not contain a cavity! Please check `self.selected_system`")

        self.selected_df = df
        return df

    def _create_single_component_df(self):
        """Creates a DataFrame for a single component system."""
        df = self.get_dataset(data_type=self.selected_data_type, component=self.selected_component, component_name=self.selected_component_name)

        if self.selected_coupler:
            df = filter_df_by_conditions(df, {"coupler_type": self.selected_coupler})
            if self.selected_coupler == "CapNInterdigitalTee":
                df = self._update_cap_interdigital_tee_parameters(df)

        return df

    def generate_updated_half_wave_cavity_df(self, parallelize=False, num_cpu=None):
        """
        !TODO: speed this up!
        """
        cavity_df = self.get_dataset(data_type="eigenmode", component="cavity_claw", component_name=self.selected_cavity)

        assert self.selected_coupler in ["NCap", "CapNInterdigitalTee"], "Selected coupler must be either 'NCap' or 'CapNInterdigitalTee'."

        cavity_df = filter_df_by_conditions(cavity_df, {"coupler_type": "NCap"})

        if not all(cavity_df["coupler_type"] == "NCap"):
            raise ValueError("All entries in the 'coupler_type' column of the cavity_df must be 'NCap'.")
       
        # update the kappa and cavity_frequency values 
        cavity_df = self._update_cap_interdigital_tee_parameters(cavity_df)
        
        return cavity_df

    def generate_qubit_half_wave_cavity_df(self, parallelize=False, num_cpu=None, save_data=False):
        """
        Generates a DataFrame that combines the qubit and half-wave cavity data.

        Args:
            parallelize (bool, optional): Flag indicating whether to parallelize the computation. Defaults to False.
            num_cpu (int, optional): Number of CPUs to use for parallelization. Defaults to None.
            save_data (bool, optional): Flag indicating whether to save the generated data. Defaults to False.

        Returns:
            pandas.DataFrame: The generated DataFrame.

        Raises:
            None

        Notes:
            - This method generates a DataFrame by combining the qubit and half-wave cavity data.
            - The qubit and cavity data are obtained from the `get_dataset` and `generate_updated_half_wave_cavity_df` methods, respectively.
            - The generated DataFrame is optimized to reduce memory usage using various optimization techniques.
            - If `save_data` is True, the generated DataFrames are saved in the "data" directory.

        TODO:
            - Speed up the generation process.
        """
        print("Generating half-wave-cavity DataFrame...")
        qubit_df = self.get_dataset(data_type="cap_matrix", component="qubit", component_name=self.selected_qubit)
        cavity_df = self.generate_updated_half_wave_cavity_df(parallelize=parallelize, num_cpu=num_cpu)

        self.qubit_df = qubit_df
        self.cavity_df = cavity_df

        print("Creating qubit-half-wave-cavity DataFrame...")
        df = self.create_qubit_cavity_df(qubit_df, cavity_df, merger_terms=self.claw_merger_terms, parallelize=parallelize, num_cpu=num_cpu)
        
        # process the df to reduce the memory usage
        print("Optimizing the DataFrame...")
        opt_df = process_design_options(df)
        initial_mem = compute_memory_usage(df)
        opt_df = optimize_dataframe(opt_df)
        opt_df = delete_object_columns(opt_df)
        opt_df = delete_categorical_columns(opt_df)
        final_mem = compute_memory_usage(opt_df)
        print(f"Memory usage reduced by {100*(initial_mem - final_mem)/initial_mem:.2f}%")

        if save_data:

            # create a data directory if it does not exist using os.makedirs
            if not os.path.exists("data"):
                os.makedirs("data")
                cavity_df.to_parquet("data/half-wave-cavity_df.parquet")
                df.to_parquet("data/qubit_half-wave-cavity_df_uncompressed.parquet")
                opt_df.to_parquet("data/qubit_half-wave-cavity_df.parquet")

        return opt_df

    def _create_multi_component_df(self, parallelize, num_cpu):
        """Creates a DataFrame for a multi-component system."""
        qubit_df = self.get_dataset(data_type="cap_matrix", component="qubit", component_name=self.selected_qubit)
        cavity_df = self.get_dataset(data_type="eigenmode", component="cavity_claw", component_name=self.selected_cavity)

        self.qubit_df = qubit_df

        if self.selected_coupler == "CLT":
            cavity_df = filter_df_by_conditions(cavity_df, {"coupler_type": self.selected_coupler})
            self.cavity_df = cavity_df
        if self.selected_coupler == "NCap":
            self.coupler_df = self.get_dataset(data_type="cap_matrix", component="coupler", component_name=self.selected_coupler)
            self.cavity_df = self.read_parquet_file(self.hwc_fname)
            df = self.read_parquet_file(self.merged_df_hwc_fname)
            return df
                
        df = self.create_qubit_cavity_df(qubit_df, cavity_df, merger_terms=self.claw_merger_terms, parallelize=parallelize, num_cpu=num_cpu)
        return df

    def _update_cap_interdigital_tee_parameters(self, cavity_df):
        """Updates parameters for CapNInterdigitalTee coupler."""
        ncap_df = self.get_dataset(data_type="cap_matrix", component="coupler", component_name="NCap")
        ncap_sim_cols = ['bottom_to_bottom', 'bottom_to_ground', 'ground_to_ground', 'top_to_bottom', 'top_to_ground', 'top_to_top']
        
        df = update_ncap_parameters(cavity_df, ncap_df, self.ncap_merger_terms, ncap_sim_cols)
        return df

    def create_qubit_cavity_df(self, qubit_df, cavity_df, merger_terms=None, parallelize=False, num_cpu=None):
        """
        Creates a merged DataFrame by merging the qubit and cavity DataFrames based on the specified merger terms.

        Args:
            qubit_df (pandas.DataFrame): The DataFrame containing qubit data.
            cavity_df (pandas.DataFrame): The DataFrame containing cavity data.
            merger_terms (list): A list of column names to be used for merging the DataFrames. Defaults to None.
            parallelize (bool): Whether to use multiprocessing to speed up the merging. Defaults to False.
            num_cpu (int): The number of CPU cores to use for multiprocessing. If not specified, the function will use the maximum number of available cores.

        Returns:
            pandas.DataFrame: The merged DataFrame.

        Raises:
            None
        """
        for merger_term in merger_terms:
            qubit_df[merger_term] = qubit_df['design_options'].map(lambda x: x['connection_pads']['readout'].get(merger_term))
            cavity_df[merger_term] = cavity_df['design_options'].map(lambda x: x['claw_opts']['connection_pads']['readout'].get(merger_term))

        # Add index column to qubit_df
        qubit_df = qubit_df.reset_index().rename(columns={'index': 'index_qc'})

        if parallelize:
            n_cores = cpu_count() if num_cpu is None else num_cpu
            qubit_df_splits = np.array_split(qubit_df, n_cores)

            with Pool(n_cores) as pool:
                merged_df_parts = list(tqdm(pool.starmap(merge_dfs, [(split, cavity_df, merger_terms) for split in qubit_df_splits]), total=n_cores))

            merged_df = pd.concat(merged_df_parts).reset_index(drop=True)
        else:
            merged_df = merge_dfs(qubit_df, cavity_df, merger_terms)

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
        self.selected_resonator_type = None

    def show_selections(self):
        """
        Prints the selected system, component, and data type.

        If the selected system is a list, it prints the selected qubit, cavity, coupler, and system.
        If the selected system is a string, it prints the selected component, component name, data type, system, and coupler.
        """
        if isinstance(self.selected_system, list): #TODO: handle dynamically
            print("Selected qubit: ", self.selected_qubit)
            print("Selected cavity: ", self.selected_cavity)
            print("Selected coupler to feedline: ", self.selected_coupler)
            if self.selected_resonator_type is not None:
                print("Selected resonator type: ", self.selected_resonator_type)
            print("Selected system: ", self.selected_system)
        elif isinstance(self.selected_system, str):
            print("Selected component: ", self.selected_component)
            print("Selected component name: ", self.selected_component_name)
            print("Selected data type: ", self.selected_data_type)
            print("Selected system: ", self.selected_system)
            print("Selected coupler: ", self.selected_coupler)
            if self.selected_resonator_type is not None:
                print("Selected resonator type: ", self.selected_resonator_type)

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

    def find_parquet_files(self):
        """
        Searches for `parquet` files in the repository and returns their paths/filenames.
        
        Returns:
            list: A list of paths/filenames of `parquet` files in the repository.
        """
        existing_files = self.get_existing_files()
        parquet_files = [file for file in existing_files if file.endswith('.parquet')]
        return parquet_files

    def read_parquet_file(self, file_name):
        """
        Takes in the filename and returns the object to be read as a pandas dataframe.
        
        Args:
            file_name (str): The name of the parquet file to read.
            
        Returns:
            pandas.DataFrame: The dataframe read from the parquet file.
        """
        base_url = f"https://huggingface.co/datasets/{self.repo_name}/resolve/main/{file_name}"
        response = requests.get(base_url)
        with open(file_name, 'wb') as f:
            f.write(response.content)
        df = pd.read_parquet(file_name)
        os.remove(file_name)  # Cleanup the downloaded file
        return df
