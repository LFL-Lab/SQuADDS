import getpass
import os
import platform
import shutil

from huggingface_hub import HfApi, get_token

from squadds.core.globals import ENV_FILE_PATH


def delete_HF_cache():
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    if platform.system() == "Windows":
        cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "huggingface", "datasets")

    dataset_cache_dir = os.path.join(cache_dir, "SQuADDS___s_qu_adds_db")
    if os.path.exists(dataset_cache_dir):
        try:
            shutil.rmtree(dataset_cache_dir)
        except OSError as exc:
            print(f"Error occurred while deleting cache: {exc}")


def set_huggingface_api_key():
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH) as file:
            existing_keys = file.read()
            if "HUGGINGFACE_API_KEY=" in existing_keys:
                print("API key already exists in .env file.")
                return

    api_key = getpass.getpass("Enter your Hugging Face API key: ")
    with open(ENV_FILE_PATH, "a") as file:
        file.write(f"\nHUGGINGFACE_API_KEY={api_key}\n")
        print(f"API key added to {ENV_FILE_PATH} file.")

    HfApi()
    if get_token() is None:
        raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")
