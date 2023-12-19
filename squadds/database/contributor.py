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
    def __init__(self, data_files):
        self.dataset_files = data_files
        self.institute = os.getenv('INSTITUTION')
        self.pi_name = os.getenv('PI_NAME')
        self.api, self.token = self.check_for_api_key()
        self.dataset_name = None
        self.dataset_files = None
        self.dataset_link = None
        
    def check_for_api_key(self):
        api = HfApi()
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")
        else:
            token = os.getenv('HUGGINGFACE_API_KEY')
            login(token)
        return api, token
        """
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not api_key:
            # Show the user the following page to get the API key
            print("="*80)
            print("\nPlease get your API key from the following page and add it to the .env file.")
            print("https://huggingface.co/settings/token")
            print("If you don't have an account, please create one here: https://huggingface.co/join (it's free)\n")
            print("OR you can run the following command to set the API key:\n")
            print("python -m squadds.core.utils set_huggingface_api_key\n")
            print("="*80)
        return api_key
        """
            
    def create_dataset_name(self, components, data_type, data_nature, data_source, date=None):
        components_joined = "-".join(components)
        date = date or datetime.now().strftime('%Y%m%d')
        base_string = f"{components_joined}_{data_type}_{data_nature}_{data_source}_{self.institute}_{self.pi_name}_{date}"
        uid_hash = hashlib.sha256(base_string.encode()).hexdigest()[:8]  # Short hash
        self.dataset_name = f"{base_string}_{uid_hash}"
        return f"{base_string}_{uid_hash}"

    def get_dataset_link(self):
        pass

    def upload_dataset(self):
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

    def create_dataset_repository(self, components, data_type, data_nature, data_source):
        date = datetime.now().strftime('%Y%m%d')
        dataset_name = self.create_dataset_name(components, data_type, data_nature, data_source, date)
        
        # Create a repository for the dataset on HuggingFace (if it doesn't exist)
        try:
            self.api.create_repo(repo_id=dataset_name, token=self.token, repo_type="dataset")
            print(f"Dataset repository {dataset_name} created.")
        except Exception as e:
            print(f"Error creating dataset repository: {e}")

        
    def upload_dataset_no_validation(self, components, data_type, data_nature, data_source, files, date=None):
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

                    