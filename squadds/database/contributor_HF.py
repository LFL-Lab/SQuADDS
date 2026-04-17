import os
from datetime import datetime

from squadds.core.utils import *
from squadds.database.checker import Checker
from squadds.database.contributor_env import get_hf_api_and_token, load_contributor_environment
from squadds.database.hf_dataset_ops import build_dataset_name, ensure_dataset_repository, upload_dataset_files


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
        load_contributor_environment()
        self.dataset_files = data_files
        self.institute = os.getenv("INSTITUTION")
        self.pi_name = os.getenv("PI_NAME")
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
        return get_hf_api_and_token()

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
        self.dataset_name = build_dataset_name(
            components,
            data_type,
            data_nature,
            data_source,
            self.institute,
            self.pi_name,
            date=date,
        )
        return self.dataset_name

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
        date = datetime.now().strftime("%Y%m%d")
        dataset_name = self.create_dataset_name(components, data_type, data_nature, data_source, date)

        # Create a repository for the dataset on HuggingFace (if it doesn't exist)
        ensure_dataset_repository(self.api, self.token, dataset_name)

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
        ensure_dataset_repository(self.api, self.token, dataset_name)
        upload_dataset_files(self.api, self.token, dataset_name, files)
