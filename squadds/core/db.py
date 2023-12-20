from squadds.core.design_patterns import SingletonMeta
from datasets import get_dataset_config_names
from datasets import load_dataset
from tabulate import tabulate
import pprint
import pandas as pd

def flatten_df_second_level(df):
    # Initialize an empty dictionary to collect flattened data
    flattened_data = {}

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the column contains dictionary-like data
        if isinstance(df[column].iloc[0], dict):
            # Iterate over second-level keys and create new columns
            for key in df[column].iloc[0].keys():
                flattened_data[f"{key}"] = df[column].apply(lambda x: x[key] if key in x else None)
        else:
            # For non-dictionary data, keep as is
            flattened_data[column] = df[column]

    # Create a new DataFrame with the flattened data
    new_df = pd.DataFrame(flattened_data)
    return new_df


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
        # check whether selected_component is coupler
        self.selected_coupler = coupler
        self.selected_component_name = coupler
        self.selected_data_type = "cap_matrix" # TODO: handle dynamically
        
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
            return flatten_df_second_level(df)
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return

    def selected_system_df(self):
        if self.selected_system is None:
            print("Selected system is not defined.")
            return
        elif isinstance(self.selected_system, str):
            df = self.get_dataset(data_type=self.selected_data_type, component=self.selected_system, component_name=self.selected_component_name)
            return df
        elif isinstance(self.selected_system, list):
            pass
