import json
import os
import shutil
from datetime import date, datetime

import git
from dotenv import load_dotenv
from github import Github
from github.Auth import Auth
from github.GithubException import GithubException


def login_to_github():
    """Logs in to GitHub using a token stored in environment variables."""
    load_dotenv()  # Load environment variables
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token is None:
        raise ValueError("GitHub token not found in environment variables.")
    return Github(github_token)

def clone_repository(repo_url, clone_dir):
    """
    Clone the given repository into the specified directory.

    Parameters:
    - repo_url (str): URL of the repository to clone.
    - clone_dir (str): Path to the directory where the repo should be cloned.

    Returns:
    - git.Repo: The cloned Git repository object.
    """
    # Remove the directory if it exists
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)

    # Clone the repository
    print(f"Cloning repository from {repo_url} into {clone_dir}...")
    repo = git.Repo.clone_from(repo_url, clone_dir)
    return repo

def fork_repository(github_token):
    """
    Forks the specified GitHub repository to the authenticated user's account.
    
    - github_token (str): GitHub Personal Access Token with appropriate permissions.
    
    Returns:
    - str: URL of the forked repository if successful, None otherwise.
    """

    original_repo_name = "LFL-Lab/SQuADDS_DB"
    # Authenticate with GitHub
    g = Github(github_token)
    
    # Get the authenticated user
    user = g.get_user()
    
    try:
        # Get the original repository
        original_repo = g.get_repo(original_repo_name)
        
        # Fork the repository to the authenticated user's account
        print(f"Forking the repository: {original_repo_name} to {user.login}'s account...")
        forked_repo = user.create_fork(original_repo)
        print(f"Repository forked successfully: {forked_repo.html_url}")
        
        return forked_repo.html_url
    
    except Exception as e:
        print(f"Error forking repository: {e}")
        return None

def read_json_file(file_path):
    """
    Read and return the contents of the specified JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: The contents of the JSON file as a Python dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Reading JSON file from {file_path}...")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def append_to_json(data, new_entry):
    """
    Append the new entry to the JSON data, ensuring the format is maintained.

    Parameters:
    - data (dict): Existing JSON data.
    - new_entry (dict): New entry to add to the JSON data.

    Returns:
    - dict: Updated JSON data.
    """
    print("Appending new entry to JSON data...")
    data.append(new_entry)
    return data

def save_json_file(file_path, data):
    """
    Save the updated data back to the JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.
    - data (dict): Updated JSON data.
    """
    print(f"Saving updated JSON data to {file_path}...")
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def commit_changes(repo, file_path, commit_message):
    """
    Commit changes to the specified file in the repository.

    Parameters:
    - repo (git.Repo): The Git repository object.
    - file_path (str): Path to the file to commit.
    - commit_message (str): Commit message.

    Returns:
    - str: The commit hash if successful.
    """
    print(f"Staging and committing changes to {file_path}...")
    repo.git.add(file_path)
    commit = repo.index.commit(commit_message)
    print(f"Commit successful with message: {commit_message}")
    return commit.hexsha

def push_changes(repo, branch_name='main', github_token=None):
    """
    Push the committed changes to the remote repository.

    Parameters:
    - repo (git.Repo): The Git repository object.
    - branch_name (str): The name of the branch to push to.
    - github_token (str): GitHub Personal Access Token (optional).

    Returns:
    - bool: True if push is successful, False otherwise.
    """
    try:
        # Check and update remote URL if GitHub token is provided
        if github_token:
            # Construct the URL with the token for authentication
            remote_url = repo.remotes.origin.url
            if remote_url.startswith("https://"):
                repo_name = remote_url.split("github.com/")[1]
                authenticated_url = f"https://{github_token}@github.com/{repo_name}"
                repo.remotes.origin.set_url(authenticated_url)
                print(f"Updated remote URL with token for authentication")
            else:
                print("Remote URL is not HTTPS, cannot update with token.")

        # Push changes to the remote repository
        print(f"Pushing changes to the remote {branch_name} branch...")
        push_result = repo.remote(name='origin').push(refspec=f'{branch_name}:{branch_name}')
        print("Push successful.")
        return True
    except Exception as e:
        print(f"Push failed: {e}")
        return False

def contribute_measured_data(new_entry, pr_title="PR For Contributing New Data", pr_body="This PR is for contributing new data to the SQuADDS Measured Devices Database."):
    """
    Update the JSON file in the given repository by appending a new entry and committing the changes.

    Parameters:
    - new_entry (dict): New entry to append to the JSON file.
    - pr_title (str): The title of the pull request.
    - pr_body (str): The body description of the pull request.
    
    Returns:
    - str: Commit hash if successful, None otherwise.
    """
    # Variable setup
    original_repo_name = "LFL-Lab/SQuADDS_DB"
    TEMP_CLONE_DIR = "./temp_forked_repo"
    json_file_path = "measured_device_database.json" 
    load_dotenv()  
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token is None:
        raise ValueError("GitHub token not found in environment variables.")
    gh_name = get_github_username(github_token)
    forked_repo_name = f"{gh_name}/SQuADDS_DB"
    branch_name = "main"

    # Step 0: Fork the repository
    forked_repo_url = fork_repository(github_token)

    # Step 1: Clone the repository
    repo = clone_repository(forked_repo_url, TEMP_CLONE_DIR)

    # Step 2: Check out the main branch
    print("Checking out the main branch...")
    repo.git.checkout('main')

    # Step 3: Read the JSON file
    full_json_file_path = os.path.join(TEMP_CLONE_DIR, json_file_path)
    data = read_json_file(full_json_file_path)

    # Step 4: Append the new entry to the JSON data
    updated_data = append_to_json(data, new_entry)

    # Step 5: Save the updated JSON data back to the file
    save_json_file(full_json_file_path, updated_data)

    # Step 6: Commit the changes with a unique datetime-based message
    commit_message = f"Update JSON dataset with new entry - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    commit_hash = commit_changes(repo, json_file_path, commit_message)

    # Step 7: Push the changes to the remote repository
    if (push_changes(repo, "main", github_token)):
        print("Successfully pushed changes to the remote repository.")
    else:
        print("Failed to push changes to the remote repository.")
        return
    
    # Step 8: Create a Pull Request on GitHub
    pr_url = create_pull_request(forked_repo_name, branch_name, pr_title, pr_body, github_token)
    print(f"Pull request URL: {pr_url}")
    
def create_pull_request(forked_repo_name, branch_name, pr_title, pr_body, github_token):
    """
    Creates a pull request from the specified branch in the forked repository to the original repository.

    Parameters:
    - forked_repo_name (str): The full name of the forked repository (e.g., 'your_username/repo_name').
    - branch_name (str): The name of the branch from which to create the PR.
    - pr_title (str): The title of the pull request.
    - pr_body (str): The body description of the pull request.
    - github_token (str): GitHub Personal Access Token for authentication.

    Returns:
    - str: URL of the created pull request if successful, None otherwise.
    """

    original_repo_name = "LFL-Lab/SQuADDS_DB"

    # Authenticate with GitHub using the provided token
    g = Github(github_token)
    user = g.get_user()

    try:
        # Get the original repository
        original_repo = g.get_repo(original_repo_name)

        # Create a pull request
        print(f"Creating a pull request from {forked_repo_name}:{branch_name} to {original_repo_name}:main...")
        pull_request = original_repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=f"{user.login}:{branch_name}",  # Forked repository and branch
            base="main"  # Target branch in the original repository
        )

        print("Pull request created successfully")
        return pull_request.html_url

    except Exception as e:
        print(f"Error creating pull request: {e}")
        return None

def get_github_username(github_token):
    """
    Get the GitHub username associated with the provided token.

    Parameters:
    - github_token (str): GitHub Personal Access Token.

    Returns:
    - str: GitHub username if successful, None otherwise.
    """
    try:
        # Authenticate with GitHub using the token
        g = Github(github_token)
        
        # Get the authenticated user
        user = g.get_user()
        
        # Return the username
        return user.login

    except Exception as e:
        print(f"Error retrieving GitHub username: {e}")
        return None