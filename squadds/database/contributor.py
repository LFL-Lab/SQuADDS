import glob
import json
import os
import subprocess
from datetime import datetime

from datasets import get_dataset_config_names, load_dataset
from dotenv import load_dotenv

from squadds.core.globals import *
from squadds.core.utils import (compare_schemas, delete_HF_cache,
                                get_config_schema, get_entire_schema, get_type,
                                is_float, validate_types)

"""
! TODO:
* Inputs the config/system data
* required schema generated based on the config/system data
"""

class ExistingConfigData:
    """
    Represents an existing configuration data object.

    Attributes:
        config (str): The name of the configuration.
        sim_results (dict): A dictionary containing simulation results.
        design (dict): A dictionary containing design options and the design tool.
        sim_options (dict): A dictionary containing simulation setup options.
        units (set): A set containing the units used in the simulation results.
        notes (dict): A dictionary containing additional notes.
        ref_entry (dict): A dictionary containing the reference entry.
        contributor (dict): A dictionary containing contributor information.
        entry (dict): A dictionary containing the contribution data.
        local_repo_path (str): The local repository path.
        sweep_data (list): A list containing sweep data.

    Methods:
        _validate_config_name(): Validates the configuration name.
        get_config_schema(): Retrieves the schema for the given configuration name.
        show_config_schema(): Prints the schema for the given configuration name.
        _supported_config_names(): Retrieves the supported configuration names.
        show(): Prints the contribution data.
        __set_contributor_info(): Sets the contributor information.
        get_contributor_info(): Retrieves the contributor information.
        add_sim_result(result_name, result_value, unit): Adds a simulation result.
        add_sim_setup(sim_setup): Adds simulation setup options to the contribution.
        add_design(design): Adds a design to the contribution.
        add_design_v0(design): Adds a design to the contribution (version 0).
        to_dict(): Converts the contribution data to a dictionary.
        clear(): Clears the contribution data.
        add_notes(notes): Adds notes to the contribution.
        validate_structure(actual_structure): Validates the structure of the contributor object.
        _validate_structure(): Validates the structure of the contributor object.
        validate_types(data): Validates the types of the data.
        _validate_types(): Validates the types of the data.
        _validate_content_v0(): Validates the content of the contribution against the dataset schema.
    """
    def __init__(self, config=""):
        self.__repo_name = "SQuADDS/SQuADDS_DB"
        self.config = config
        self._validate_config_name()
        load_dotenv(ENV_FILE_PATH) 
        self.sim_results = {}
        self.design = {"design_tool": "", "design_options": {}}
        self.sim_options = {"setup": {}, "simulator": ""}
        self.units = set()
        self.notes = {} 
        self.ref_entry = {}
        self.__set_contributor_info()
        self.entry = self.to_dict()
        self.__isValidated = False
        self.local_repo_path = ""
        self.sweep_data = []

    def _validate_config_name(self):
            """
            Validates the config name against the supported config names.

            Raises:
                ValueError: If the config name is invalid.
            """
            configs = self._supported_config_names()
            if self.config not in configs:
                raise ValueError(f"Invalid config name: {self.config}. Supported config names: {configs}")
        
    def get_config_schema(self):
        """
        Connects to the repository with the given configuration name. Chooses the first entry from the config dataset and extracts the schema.

        Returns:
            A dictionary containing the schema for the given configuration name.
        """
        # get the first entry
        config_dataset = load_dataset(self.__repo_name, self.config)
        entry = config_dataset['train'][0]
        self.ref_entry = entry
        schema = get_config_schema(entry)
        return schema  # Return the schema as a dictionary

    def show_config_schema(self):
        """
        Connects to the repository with the given configuration name. Chooses the first entry from the config dataset and extracts the schema.

        Returns:
            None
        """
        # get the first entry
        config_dataset = load_dataset(self.__repo_name, self.config)
        entry = config_dataset['train'][0]
        schema = get_config_schema(entry)
        print(json.dumps(schema, indent=2))


    def _supported_config_names(self):
        """
        Retrieves the supported configuration names from the repository.

        Returns:
            A list of supported configuration names.
        """
        delete_HF_cache()
        configs = get_dataset_config_names(self.__repo_name, download_mode='force_redownload')
        return configs

    # method that returns the contribution data in a dictionary format
    def show(self):
        """
        Print the contribution data in a pretty format.

        Args:
            None

        Returns:
            None
        """
        # pretty print the contribution data
        print(json.dumps(self.to_dict(), indent=4))
        
    def __set_contributor_info(self):
        self.contributor = {
            "group": os.getenv('GROUP_NAME'),
            "PI": os.getenv('PI_NAME'),
            "institution": os.getenv('INSTITUTION'),
            "uploader": os.getenv('USER_NAME'),
            "misc": os.getenv('CONTRIB_MISC'),
            "date_created": datetime.now().strftime("%Y-%m-%d %H%M%S")
        }

    def get_contributor_info(self):
        """
        Returns the contributor information.

        Returns:
            str: The contributor information.
        """
        return self.contributor

    def add_sim_result(self, result_name, result_value, unit):
        """
        Add a simulation result to the contributor.

        Args:
            result_name (str): The name of the simulation result.
            result_value (float): The value of the simulation result.
            unit (str): The unit of measurement for the simulation result.

        Returns:
            None
        """
        self.units.add(unit)  # Add unit to the set
        self.sim_results[result_name] = result_value
        self.sim_results[f"{result_name}_unit"] = unit  # Keep the individual unit keys for now

    def add_sim_setup(self, sim_setup):
        """
        Adds simulation setup options to the contribution.

        Args:
            sim_setup (dict): A dictionary containing simulation setup options that match the configs schema.
        """
        # Retrieve the schema for simulation options
        schema = self.get_config_schema()

        # Validate the provided simulation setup options against the schema
        sim_setup_schema = schema.get('sim_options', {})
        if not isinstance(sim_setup, dict):
            raise ValueError('Simulation setup options must be provided as a dictionary.')

        # Check if all keys are present and have correct types
        for key, expected_type in sim_setup_schema.items():
            if key not in sim_setup:
                raise ValueError(f'Missing required simulation setup option: {key}')
            if get_type(sim_setup[key]) != expected_type:
                raise TypeError(f'Incorrect type for {key}. Expected {expected_type}, got {get_type(sim_setup[key])}.')

        # All checks passed, add the simulation setup options
        self.sim_options.update(sim_setup)

    def add_design(self, design):
        """
        Adds a design to the contribution.

        Args:
            design (dict): A dictionary containing design options and the design tool.
        """
        # Retrieve the schema for design
        schema = self.get_config_schema()

        # Validate the provided design against the schema
        if not isinstance(design, dict):
            raise ValueError('Design must be provided as a dictionary.')

        design_options = design.get('design_options', {})
        design_tool = design.get('design_tool')

        # Validate design options and design tool
        design_options_schema = schema.get('design', {}).get('design_options', {})
        if get_type(design_options) != design_options_schema:
            raise TypeError(f"Incorrect type for design options. Expected {design_options_schema}, got {get_type(design_options)}.")

        if design_tool and get_type(design_tool) != 'str':
            raise TypeError(f"Incorrect type for design tool. Expected 'str', got {get_type(design_tool)}.")

        # All checks passed, add the design options and tool
        self.design.update(design)

    def add_design_v0(self, design):
        """
        Adds a design to the contribution.

        Args:
            design (dict): A dictionary containing design options and the design tool.
        """
        # Retrieve the schema for design
        schema = self.get_config_schema()

        # Validate the provided design against the schema
        if not isinstance(design, dict):
            raise ValueError('Design must be provided as a dictionary.')

        # Extract design options and design tool from the input dictionary
        design_options = design.get('design_options')
        design_tool = design.get('design_tool')

        # Validate design options and design tool
        design_options_schema = schema.get('design', {}).get('design_options', {})
        if get_type(design_options) != design_options_schema:
            raise TypeError(f"Incorrect type for design options. Expected {design_options_schema}, got {get_type(design_options)}.")

        if get_type(design_tool) != 'str':
            raise TypeError(f"Incorrect type for design tool. Expected 'str', got {get_type(design_tool)}.")

        # All checks passed, add the design options and tool
        self.design.update(design)
    
    def to_dict(self):
        """
        Converts the Contributor object to a dictionary.

        Returns:
            dict: A dictionary representation of the Contributor object.
        """
        # Check if all units are the same
        if len(self.units) == 1:
            common_unit = self.units.pop()  # Get the common unit
            self.sim_results['units'] = common_unit
            # Remove individual unit keys
            for result_name in list(self.sim_results.keys()):
                if '_unit' in result_name:
                    del self.sim_results[result_name]
        return {
            "design": self.design,
            "sim_options": self.sim_options,
            "sim_results": self.sim_results,
            "contributor": self.contributor,
            "notes": self.notes
        }

    def clear(self):
        """
        Clears the contribution data.
        """
        self.sim_results = {}
        self.design = {"design_tool": "", "design_options": {}}
        self.sim_options = {"setup": {}, "simulator": ""}
        self.units = set()
        self.notes = {}
        self.__isValidated = False

    def add_notes(self, notes={}):
        """
        Adds notes to the contribution.

        Args:
            notes (dict): A dictionary containing notes.
        """
        if not isinstance(notes, dict):
            raise ValueError('Notes must be provided as a dictionary.')

        # Merge new notes with existing ones
        self.notes.update(notes)

    def validate_structure(self, actual_structure):
        """
        Validates the structure of the contributor object.

        Args:
            actual_structure (dict): The actual structure of the contributor object.

        Raises:
            ValueError: If any required key or sub-key is missing in the actual structure.
        """
        expected_structure = self.get_config_schema()

        # Compare the structure of actual data with the expected schema
        for key, value in expected_structure.items():
            if key not in actual_structure:
                raise ValueError(f"Missing required key: {key}")
            if isinstance(value, dict):
                for sub_key in value:
                    if sub_key not in actual_structure[key]:
                        raise ValueError(f"Missing required sub-key '{sub_key}' in '{key}'")
        print("Structure validated successfully....")

    def _validate_structure(self):
        """
        Validates the structure of the contributor object.

        Raises:
            ValueError: If any required key or sub-key is missing in the actual structure.
        """
        expected_structure = self.get_config_schema()
        actual_structure = self.to_dict()

        # Compare the structure of actual data with the expected schema
        for key, value in expected_structure.items():
            if key not in actual_structure:
                raise ValueError(f"Missing required key: {key}")
            if isinstance(value, dict):
                for sub_key in value:
                    if sub_key not in actual_structure[key]:
                        raise ValueError(f"Missing required sub-key '{sub_key}' in '{key}'")
        print("Structure validated successfully....")

    def validate_types(self, data):
        """
        Args:
            data (dict): The data to be validated.
        Validates the types of the data using the schema defined in the config.
        """
        schema = self.get_config_schema()
        validate_types(data, schema)
        print("Types validated successfully....")

    def _validate_types(self):
        """
        Validates the types of the data using the schema defined in the config.
        """
        schema = self.get_config_schema()
        data = self.to_dict()
        validate_types(data, schema)
        print("Types validated successfully....")

    def _validate_content_v0(self):
        """
        Validates the content of the contribution against the dataset schema.
        """
        data = self.to_dict()
        ref = self.ref_entry
        # print data and ref nicely json
        # print(f"Data: {json.dumps(data, indent=2)}")
        # print(f"Ref: {json.dumps(ref, indent=2)}")
        
        # Validate 'sim_options.setup' and 'design.design_options'
        for key in ['design', 'sim_options']:
            sub_key = 'setup' if key == 'sim_options' else 'design_options'
            data_schema = get_entire_schema(data[key][sub_key])
            expected_schema = get_entire_schema(ref[key][sub_key])
            print(f"Key: {key}, Sub-key: {sub_key}")
            print(f"Data schema: {json.dumps(data_schema, indent=2)}")
            print(f"Expected schema: {json.dumps(expected_schema, indent=2)}")

            if data_schema != expected_schema:
                raise ValueError(f"Structure mismatch in '{key}.{sub_key}'. Expected: {expected_schema}, Got: {data_schema}")

    def validate_content(self, data):
        """
        Args:
            data (dict): The data to be validated.
        Validates the content of the contribution against the dataset schema.
        """
        ref = self.ref_entry

        def get_nested(dictionary, keys):
            for key in keys.split('.'):
                if dictionary is not None and key in dictionary:
                    dictionary = dictionary[key]
                else:
                    return None
            return dictionary
    
    def _validate_content(self):
        """
        Validates the content of the contribution against the dataset schema.
        """
        data = self.to_dict()
        ref = self.ref_entry

        def get_nested(dictionary, keys):
            for key in keys.split('.'):
                if dictionary is not None and key in dictionary:
                    dictionary = dictionary[key]
                else:
                    return None
            return dictionary

        def find_common_keys(dict1, dict2, path=""):
            """
            Recursively find and compare common keys in two dictionaries.
            """
            common_keys = set(dict1.keys()) & set(dict2.keys())
            diff_keys = (set(dict1.keys()) - set(dict2.keys())) | (set(dict2.keys()) - set(dict1.keys()))
            for key in common_keys:
                new_path = f"{path}.{key}" if path else key
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    yield from find_common_keys(dict1[key], dict2[key], new_path)
                else:
                    if type(dict1[key]) != type(dict2[key]):
                        yield new_path, False
                    else:
                        yield new_path, True
            for key in diff_keys:
                new_path = f"{path}.{key}" if path else key
                yield new_path, None

        result = list(find_common_keys(data, ref))
        common_keys = [key for key, match in result if match is not None]
        mismatched_keys = [key for key, match in result if match is False]
        missing_keys = [key for key, match in result if match is None]

        if mismatched_keys:
            print("\nMismatched keys found. These keys are present in both dictionaries but have values of different types:\n")
            for key in mismatched_keys:
                print(f"Key: {key}, data type in 'data': {type(get_nested(data, key))}, data type in 'ref': {type(get_nested(ref, key))}")

        if missing_keys:
            print("\nMissing keys found. These keys are present in one dictionary but not the other:\n")
            for key in missing_keys:
                if get_nested(data, key) is not None:
                    print(f"Key: {key} is missing in 'ref'")
                else:
                    print(f"Key: {key} is missing in 'data'")

        # return common_keys, mismatched_keys, missing_keys

    def _validate_content_v1(self):
        """
        Validates the content of the contribution against the dataset schema.
        """
        data = self.to_dict()
        ref = self.ref_entry
        
        for key in ['design', 'sim_options']:
            sub_key = 'setup' if key == 'sim_options' else 'design_options'
            data_schema = get_entire_schema(data[key][sub_key])
            expected_schema = get_entire_schema(ref[key][sub_key])
            print(f"Key: {key}, Sub-key: {sub_key}")
            # print(f"Data schema: {json.dumps(data_schema, indent=2)}")
            # print(f"Expected schema: {json.dumps(expected_schema, indent=2)}")

            compare_schemas(data_schema, expected_schema, f"{key}.{sub_key}.")

        print("Content validation passed.")

    def validate(self):
        """
        Validates the contribution by performing various checks.

        Raises:
            Exception: If any validation check fails.
        """
        # Perform all validation checks
        # if no errors then set isValidated to True
        if not self.is_validated:
            try:
                self._validate_structure()
                self._validate_types()
                self._validate_content()
                self.__isValidated = True
            except Exception as e:
                print("Validation failed.")
                raise e
        else:
            print("This contribution has already been validated.")

    def validate_sweep(self):
        """
        Validates the sweep data by performing structure, type, and content validation on each entry.

        Raises:
            Exception: If the validation fails.

        Returns:
            None
        """
        if not self.is_validated:
            try:
                for entry in self.sweep_data:
                    print(f"Validating entry {self.sweep_data.index(entry)+1} of {len(self.sweep_data)}...") 
                    self.validate_structure(entry)
                    self.validate_types(entry)
                    self.validate_content(entry)
                    print(f"Entry {self.sweep_data.index(entry)+1} of {len(self.sweep_data)} validated successfully.")
                    print("--------------------------------------------------")
                self.__isValidated = True
            except Exception as e:
                print("Validation failed.")
                raise e
        else:
            print("This contribution has already been validated.")

    @property
    def invalidate(self):
        """
        Invalidates the contributor by setting the isValidated flag to False.
        """
        self.__isValidated = False

    def update_repo(self, path_to_repo):
        """
        Updates the repository at the specified path.

        Args:
            path_to_repo (str): The path to the repository.

        Raises:
            subprocess.CalledProcessError: If the git commands fail.

        """
        original_cwd = os.getcwd()
        try:
            # Check if data is validated
            if not self.is_validated:
                raise ValueError("Data must be validated before updating the repository.")
            # Create the path to the repo if it doesn't exist
            if not os.path.exists(path_to_repo):
                os.makedirs(path_to_repo)
            # Check if the repo exists by looking for .git file in the path_to_repo + "SQuADDS_DB" directory
            if os.path.exists(path_to_repo+"/"+self.__repo_name.split('/')[-1]):
                # Pull the latest changes
                os.chdir(path_to_repo+"/"+self.__repo_name.split('/')[-1])
                subprocess.run(["git", "pull"], check=True)
            else:
                print(f"Cloning dataset repository from to {path_to_repo}...")
                os.chdir(path_to_repo)
                dataset_endpoint = f"git@hf.co:datasets/{self.__repo_name}"
                # Clone the repo
                # subprocess.run(["git", "clone", dataset_endpoint], check=True)
                subprocess.run(["git", "-c", "core.sshCommand=ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no", "clone", dataset_endpoint], check=True)
            
            # Create a new branch and checkout to it
            # uploader_name = self.contributor['uploader'].replace(" ", "")
            # uid = self.contributor['date_created'].replace(" ", "")
            # branch_name = f"add_{self.config}_{uploader_name}_{uid}"
            # subprocess.run(["git", "checkout", "-b", branch_name], check=True)
        finally:
            # Revert to the original current working directory
            os.chdir(original_cwd)

    def update_db(self, path_to_repo, is_sweep=False):
        """
        Updates the local repository with the validated data.

        Args:
            path_to_repo (str): The path to the local repository.

        Raises:
            ValueError: If the data has not been validated.
        """
        if not is_sweep:
            if not self.is_validated:
                raise ValueError("Data must be validated before updating the repository.")
            # update the local repo
            os.chdir(path_to_repo+"/"+self.__repo_name.split('/')[-1])
            dataset_file = f"{self.config}.json"
            with open(dataset_file, "r+") as file:
                data = json.load(file)
                data.append(self.to_dict())
                file.seek(0)
                json.dump(data, file, indent=4)
            print(f"Data added to {dataset_file} successfully.")
        else:
            if not self.is_validated:
                raise ValueError("Data must be validated before updating the repository.")
            # update the local repo
            os.chdir(path_to_repo+"/"+self.__repo_name.split('/')[-1])
            dataset_file = f"{self.config}.json"
            with open(dataset_file, "r+") as file:
                data = json.load(file)
                for entry in self.sweep_data:
                    data.append(entry)
                file.seek(0)
                json.dump(data, file, indent=4)
            print(f"Data added to {dataset_file} successfully.")
            
    def upload_to_HF(self, path_to_repo):
        """
        Uploads validated data to the specified repository.

        Args:
            path_to_repo (str): The path to the repository.

        Raises:
            ValueError: If the data has not been validated.
            subprocess.CalledProcessError: If the git commands fail.

        Returns:
            None
        """
        if not self.is_validated:
            raise ValueError("Data must be validated before updating the repository.")
        # navigate to the repo
        os.chdir(path_to_repo+"/"+self.__repo_name.split('/')[-1])
        # create a commit message based on the contributor info
        commit_message = f"Add {self.config} data from {self.contributor['group']} group by {self.contributor['uploader']} on {self.contributor['date_created']}"
        uploader_name = self.contributor['uploader'].replace(" ", "")
        uid = self.contributor['date_created'].replace(" ", "")
        branch_name = f"add_{self.config}_{uploader_name}_{uid}"

        try:
            # Commit and push changes
            subprocess.run(["git", "add", f"{self.config}.json"], check=True)
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to commit changes to {self.config}.json")
            raise e

        try:
            # create upstream branch
            os.environ['GITHUB_TOKEN'] = os.getenv('GITHUB_TOKEN')
            subprocess.run(["git", "push", "--set-upstream", "origin", branch_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to create upstream branch for {self.config}.json")
            raise e

        try:
            # Push changes - ensure you have the necessary permissions and authentication set up
            subprocess.run(["git", "push"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to push changes to {self.config}.json")
            raise e

    def from_json(self, json_file, is_sweep=False):
        """
        Loads a contribution from a JSON file.

        Args:
            json_file (str): The path to the JSON file.
            is_sweep (bool): True if the contribution is a sweep, False otherwise.
        """
        if not is_sweep:
            file_path = os.path.abspath(json_file)
            
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")

            with open(file_path, "r") as file:
                data = json.load(file)
                self.design = data['design']
                self.sim_options = data['sim_options']
                self.sim_results = data['sim_results']
                self.__set_contributor_info()
                try:
                    self.notes = data['notes']
                except KeyError:
                    pass

            print("Contribution loaded successfully.")
        else:
            
            json_files = glob.glob(os.path.abspath(json_file+"*.json"))
            if not json_files:
                raise ValueError(f"Files not found: {json_files}") 
            for file in json_files:
                entry = {}
                with open(file, "r") as f:
                    data = json.load(f)
                    entry["design"] = data['design']
                    entry["sim_options"] = data['sim_options']
                    entry["sim_results"] = data['sim_results']
                    entry["contributor"] = self.get_contributor_info()
                    try:
                        entry["notes"] = data['notes']
                    except KeyError:
                        entry["notes"] = {}

                    self.sweep_data.append(entry)
            
            print("Sweep data loaded successfully.")

    @property
    def is_validated(self):
        """
        Returns True if the contribution is validated, False otherwise.

        Returns:
            bool: True if the contribution is validated, False otherwise.
        """
        return self.__isValidated

    def contribute(self, path_to_repo, is_sweep=False):
        """
        Contributes to the repository by updating the local repo, updating the database, and uploading to HF.

        Args:
            path_to_repo (str): The path to the repository.
            is_sweep (bool): True if the contribution is a sweep, False otherwise.

        Returns:
            None
        """
        if not self.is_validated:
            raise ValueError("Data must be validated before contributing.")
        self.update_repo(path_to_repo)
        self.update_db(path_to_repo, is_sweep)
        # self.upload_to_HF(path_to_repo)
        print("Contribution ready for PR")
    
    def submit(self):
        """
        Sends the data and the config name to a remote server.
        """
        raise NotImplementedError("This method is not implemented yet.")  