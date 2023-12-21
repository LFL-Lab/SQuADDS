from squadds.core.design_patterns import SingletonMeta
from datasets import get_dataset_config_names
from datasets import load_dataset
from tabulate import tabulate
import pprint
import pandas as pd
from addict import Dict
import numpy as np
from squadds.core.utils import *

class SQuADDS_DB(metaclass=SingletonMeta):
    
    def __init__(self):
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
        components = []
        for config in self.configs:
            components.append(config.split("-")[0])
        return components
    
    def supported_component_names(self):
        component_names = []
        for config in self.configs:
            component_names.append(config.split("-")[1])
        return component_names
    
    def supported_data_types(self):
        data_types = []
        for config in self.configs:
            data_types.append(config.split("-")[2])
        return data_types

    def supported_config_names(self):
        configs = get_dataset_config_names(self.repo_name)
        return configs

    def get_configs(self):
        # pretty print the config names
        pprint.pprint(self.configs)

    def get_component_names(self, component=None):
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
        config = component + "-" + component_name + "-" + data_type
        self.view_contributors_of_config(config)

    def select_components(self, component_dict=None):
        # check if dict or string
        if isinstance(component_dict, dict):
            config = component_dict["component"] + "-" + component_dict["component_name"] + "-" + component_dict["data_type"]
        elif isinstance(component_dict, str):
            config = component_dict
        print("Selected config: ", config)
        
    def select_system(self, components=None):
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
        # TODO: fix this method to work on NCap coupler sims
        self.selected_coupler = coupler
        #self.selected_component_name = coupler
        #self.selected_data_type = "cap_matrix" # TODO: handle dynamically
        
        # check if coupler is supported
        if self.selected_coupler not in self.supported_component_names()+["CLT"]: # TODO: handle dynamically

            print(f"Coupler `{self.selected_coupler}` not supported. Available couplers are:")
            self.view_component_names("coupler")
            return

    def get_dataset(self, data_type=None, component=None, component_name=None):
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
            self._set_target_param_keys(df)
            return flatten_df_second_level(df)
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return

    def selected_system_df(self):
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
        self.selected_component_name = None
        self.selected_component = None
        self.selected_data_type = None
        self.selected_qubit = None
        self.selected_cavity = None
        self.selected_coupler = None
        self.selected_system = None

    def show_selections(self):
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