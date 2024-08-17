import os

from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository, login

from squadds.core.globals import *


def fork_dataset(repo_id: str, dataset_name: str, new_dataset_name: str, private: bool = True):
    """
    Fork a dataset from Hugging Face Hub.
    
    Args:
        repo_id (str): The repo ID (namespace/repo) of the dataset to fork.
        dataset_name (str): Name of the dataset to fork.
        new_dataset_name (str): Name of the new dataset.
        private (bool): Whether the new dataset should be private or public.
    
    Returns:
        None
    """
    dataset = load_dataset(repo_id, dataset_name)
    dataset.push_to_hub(repo_id, new_dataset_name, private=private)
    print(f"Forked dataset '{dataset_name}' to '{new_dataset_name}'.")

def create_PR(repo_id: str, branch_name: str, title: str, description: str):
    """
    Create a Pull Request (PR) on Hugging Face Hub.
    
    Args:
        repo_id (str): The repo ID (namespace/repo) where the PR will be created.
        branch_name (str): The branch name where the changes are made.
        title (str): The title of the PR.
        description (str): A description of the changes made in the PR.
    
    Returns:
        dict: Information about the created PR.
    """
    api = HfApi()
    
    try:
        pr_info = api.create_pull_request(
            repo_id=repo_id,
            head=branch_name,  # The branch with your changes
            title=title,
            body=description
        )
        print(f"Created PR '{title}' on repo '{repo_id}'.")
        return pr_info
    
    except Exception as e:
        print(f"Failed to create PR: {e}")
        raise

# Make sure you are logged in to Hugging Face
def login_to_huggingface():
    """
    Log into Hugging Face using an API token from environment variables.
    """
    load_dotenv(ENV_FILE_PATH)  # Load environment variables from .env file
    token = os.getenv("HUGGINGFACE_API_KEY")  # Retrieve the token from environment variables
    
    if token is None:
        raise ValueError("Hugging Face API token not found in environment variables.")
    
    login(token)
    print("Successfully logged in to Hugging Face")


# Load the dataset from the Hugging Face Hub
def load_hf_dataset(dataset_name: str, config: str = None):
    """
    Load a dataset from Hugging Face Hub.
    
    Args:
        dataset_name (str): The name or path of the dataset on the Hugging Face Hub.
        config (str): Specific configuration or version of the dataset.
    
    Returns:
        Dataset or DatasetDict: Loaded dataset.
    """
    
    dataset = load_dataset(dataset_name, config)
    
    print(f"Loaded dataset: {dataset_name}")
    return dataset

# Add a new column to the dataset
def add_column_to_dataset(dataset: Dataset, column_name: str, column_data: list):
    """
    Add a new column to a dataset.
    
    Args:
        dataset (Dataset): Hugging Face dataset to which you want to add a column.
        column_name (str): Name of the new column.
        column_data (list): Data for the new column.
    
    Returns:
        Dataset: Dataset with the new column.
    """
    new_dataset = dataset.add_column(column_name, column_data)
    print(f"Added new column '{column_name}' to dataset.")
    return new_dataset

# Remove a column from the dataset
def remove_column_from_dataset(dataset: Dataset, column_name: str):
    """
    Remove a column from a dataset.
    
    Args:
        dataset (Dataset): Hugging Face dataset from which you want to remove a column.
        column_name (str): Name of the column to remove.
    
    Returns:
        Dataset: Dataset with the column removed.
    """
    new_dataset = dataset.remove_columns([column_name])
    print(f"Removed column '{column_name}' from dataset.")
    return new_dataset

# View a specific column in the dataset
def view_column_in_dataset(dataset: Dataset, column_name: str, num_values: int):
    """
    View a specific column in the dataset by its name.
    
    Args:
        dataset (Dataset): Hugging Face dataset.
        column_name (str): Name of the column to view.
    
    Returns:
        list: Data from the specified column.
    """
    if column_name not in dataset.column_names:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")

    column_data = dataset[column_name]
    print(f"Data from column '{column_name}':")
    print(column_data[:num_values])  # Print values for preview
    return column_data

# Update a specific column in the dataset
def update_column_in_dataset(dataset: Dataset, column_name: str, new_column_data: list):
    """
    Update a specific column in the dataset.
    
    Args:
        dataset (Dataset): Hugging Face dataset to update.
        column_name (str): Name of the column to update.
        new_column_data (list): List of new data to replace the existing column.
    
    Returns:
        Dataset: Updated dataset.
    """
    if column_name not in dataset.column_names:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    if len(new_column_data) != len(dataset):
        raise ValueError(f"The new data length ({len(new_column_data)}) does not match the dataset length ({len(dataset)}).")
    
    updated_dataset = dataset.map(lambda x, idx: {column_name: new_column_data[idx]}, with_indices=True)
    print(f"Updated column '{column_name}' in the dataset.")
    return updated_dataset

# Add a new row to the dataset
def add_row_to_dataset(dataset: Dataset, row_data: dict):
    """
    Add a new row to a dataset.
    
    Args:
        dataset (Dataset): The Hugging Face dataset to which you want to add a row.
        row_data (dict): The row data in dictionary format.
    
    Returns:
        Dataset: Dataset with the new row added.
    """
    # Convert the dataset to a list, append the new row, and convert back to a Dataset
    new_dataset = Dataset.from_dict({k: dataset[k] + [v] for k, v in row_data.items()})
    print("Added new row to dataset.")
    return new_dataset


# Remove a row from the dataset by index
def remove_row_from_dataset(dataset: Dataset, row_index: int):
    """
    Remove a row from a dataset by index.
    
    Args:
        dataset (Dataset): Hugging Face dataset from which you want to remove a row.
        row_index (int): Index of the row to remove.
    
    Returns:
        Dataset: Dataset with the row removed.
    """
    new_dataset = dataset.select([i for i in range(len(dataset)) if i != row_index])
    print(f"Removed row at index {row_index} from dataset.")
    return new_dataset


# Update a row in the dataset
def update_row_in_dataset(dataset: Dataset, row_index: int, new_row_data: dict):
    """
    Update an existing row in a dataset by index.
    
    Args:
        dataset (Dataset): Hugging Face dataset to update.
        row_index (int): Index of the row to update.
        new_row_data (dict): The new data for the row.
    
    Returns:
        Dataset: Updated dataset.
    """
    updated_rows = dataset.to_dict()
    
    for key in new_row_data:
        updated_rows[key][row_index] = new_row_data[key]
    
    updated_dataset = Dataset.from_dict(updated_rows)
    print(f"Updated row at index {row_index} in the dataset.")
    return updated_dataset


# View a specific row in the dataset
def view_row_in_dataset(dataset: Dataset, row_index: int):
    """
    View a specific row in the dataset by index.
    
    Args:
        dataset (Dataset): Hugging Face dataset.
        row_index (int): Index of the row to view.
    
    Returns:
        dict: Data for the specified row.
    """
    row_data = dataset[row_index]
    print(f"Row {row_index}: {row_data}")
    return row_data

# Merge two datasets
def merge_datasets(dataset1: Dataset, dataset2: Dataset):
    """
    Merge two datasets into one.
    
    Args:
        dataset1 (Dataset): First dataset.
        dataset2 (Dataset): Second dataset.
    
    Returns:
        Dataset: Merged dataset.
    """
    merged_dataset = concatenate_datasets([dataset1, dataset2])
    print("Merged two datasets.")
    return merged_dataset

# Save the dataset to Hugging Face Hub
def save_dataset_to_hf(dataset: Dataset, repo_id: str, dataset_name: str, private: bool = True):
    """
    Push a dataset to Hugging Face Hub.
    
    Args:
        dataset (Dataset): The dataset to push to Hugging Face Hub.
        repo_id (str): The repo ID (namespace/repo) on Hugging Face Hub.
        dataset_name (str): Name of the dataset on Hugging Face Hub.
        private (bool): Whether the dataset should be private or public.
    
    Returns:
        None
    """
    dataset.push_to_hub(repo_id, dataset_name, private=private)
    print(f"Dataset '{dataset_name}' saved to Hugging Face Hub.")

# Filter the dataset based on a condition
def filter_dataset(dataset: Dataset, filter_fn):
    """
    Filter a dataset based on a custom condition.
    
    Args:
        dataset (Dataset): Hugging Face dataset to filter.
        filter_fn (function): Function that returns True or False for filtering.
    
    Returns:
        Dataset: Filtered dataset.
    """
    filtered_dataset = dataset.filter(filter_fn)
    print("Filtered dataset based on condition.")
    return filtered_dataset

if __name__ == "__main__":
    # Log in to Hugging Face
    login_to_huggingface()

    # Define repo_id, dataset_name, and new_dataset_name
    repo_id = "SQuADDS/SQuADDS_DB"