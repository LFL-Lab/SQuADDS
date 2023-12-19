import os
import hashlib
import json

from datetime import datetime
from dotenv import load_dotenv
from squadds.core.globals import *
from squadds.core.utils import *
from squadds.database.checker import Checker
from huggingface_hub import HfApi, HfFolder, login

load_dotenv(ENV_FILE_PATH)

class Contribute:
    """
    Class representing a contributor for dataset creation and upload.

    Attributes:
        dataset_files (list): List of dataset file paths.
        institute (str): Institution name.
        pi_name (str): PI (Principal Investigator) name.
        api (HfApi): Hugging Face API object.
        token (str): Hugging Face API token.
        dataset_name (str): Name of the dataset.
        dataset_files (list): List of dataset file paths.
        dataset_link (str): Link to the dataset.

    Methods:
        check_for_api_key: Checks for the presence of Hugging Face API key.
        create_dataset_name: Creates a unique name for the dataset.
        get_dataset_link: Retrieves the link to the dataset.
        upload_dataset: Uploads the dataset to Hugging Face.
        create_dataset_repository: Creates a repository for the dataset on Hugging Face.
        upload_dataset_no_validation: Uploads the dataset to Hugging Face without validation.
    """

    def __init__(self, data_files):
        self.dataset_files = data_files
        self.institute = os.getenv('INSTITUTION')
        self.pi_name = os.getenv('PI_NAME')
        self.api, self.token = self.check_for_api_key()
        self.dataset_name = None
        self.dataset_files = None
        self.dataset_link = None
        
    def check_for_api_key(self):
        """
        Checks for the presence of Hugging Face API key.

        Returns:
            api (HfApi): Hugging Face API object.
            token (str): Hugging Face API token.

        Raises:
            ValueError: If Hugging Face token is not found.
        """
        api = HfApi()
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")
        else:
            token = os.getenv('HUGGINGFACE_API_KEY')
            login(token)
        return api, token
            
    def create_dataset_name(self, components, data_type, data_nature, data_source, date=None):
        """
        Creates a unique name for the dataset.

        Args:
            components (list): List of components.
            data_type (str): Type of the data.
            data_nature (str): Nature of the data.
            data_source (str): Source of the data.
            date (str, optional): Date of the dataset creation. Defaults to None.

        Returns:
            str: Unique name for the dataset.
        """
        components_joined = "-".join(components)
        date = date or datetime.now().strftime('%Y%m%d')
        base_string = f"{components_joined}_{data_type}_{data_nature}_{data_source}_{self.institute}_{self.pi_name}_{date}"
        uid_hash = hashlib.sha256(base_string.encode()).hexdigest()[:8]  # Short hash
        self.dataset_name = f"{base_string}_{uid_hash}"
        return f"{base_string}_{uid_hash}"

    def get_dataset_link(self):
        """
        Retrieves the link to the dataset.

        Returns:
            str: Link to the dataset.
        """
        raise NotImplementedError()

    def upload_dataset(self):
        """
        Uploads the dataset to Hugging Face.

        Raises:
            NotImplementedError: If dataset upload is not implemented.
        """
        checker = Checker()  
        for file in self.dataset_files:
            checker.check(file)
            if not checker.upload_ready: 
                raise NotImplementedError()
            else:
                # Upload the dataset to Hugging Face
                raise NotImplementedError()

        # generate the link to the dataset
        
        # Send notification email after successful upload
        send_email_via_client("Example Dataset", "Institute Name", "PI Name", "2023-01-01")
        return

    def create_dataset_repository(self, components, data_type, data_nature, data_source):
        """
        Creates a repository for the dataset on HuggingFace (if it doesn't exist).

        Args:
            components (list): List of components.
            data_type (str): Type of the data.
            data_nature (str): Nature of the data.
            data_source (str): Source of the data.
        """
        date = datetime.now().strftime('%Y%m%d')
        dataset_name = self.create_dataset_name(components, data_type, data_nature, data_source, date)
        
        # Create a repository for the dataset on HuggingFace (if it doesn't exist)
        try:
            self.api.create_repo(repo_id=dataset_name, token=self.token, repo_type="dataset")
            print(f"Dataset repository {dataset_name} created.")
        except Exception as e:
            print(f"Error creating dataset repository: {e}")

        
    def upload_dataset_no_validation(self, components, data_type, data_nature, data_source, files, date=None):
        """
        Uploads the dataset to HuggingFace without validation.

        Args:
            components (list): List of components.
            data_type (str): Type of the data.
            data_nature (str): Nature of the data.
            data_source (str): Source of the data.
            files (list): List of file paths.
            date (str, optional): Date of the dataset creation. Defaults to None.
        """
        dataset_name = self.create_dataset_name(components, data_type, data_nature, data_source, date)
        
        # Create a repository for the dataset on HuggingFace (if it doesn't exist)
        try:
            self.api.create_repo(repo_id=dataset_name, token=self.token, repo_type="dataset")
            print(f"Dataset repository {dataset_name} created.")
        except Exception as e:
            print(f"Error creating dataset repository: {e}")

        # Upload files to the dataset
        for file_path in files:
            try:
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=self.token
                )
                print(f"Uploaded {file_path} to {dataset_name}.")
            except Exception as e:
                print(f"Error uploading file {file_path}: {e}")

                    