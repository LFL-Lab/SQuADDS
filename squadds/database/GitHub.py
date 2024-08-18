from github import Github
from github.Auth import Auth
from dotenv import load_dotenv
import os

def login_to_github():
    """Logs in to GitHub using a token stored in environment variables."""
    load_dotenv()  # Load environment variables
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token is None:
        raise ValueError("GitHub token not found in environment variables.")
    return Github(github_token)

def fork_repo(github, repo_name: str, org=None):
    """
    Forks a repository on GitHub.
    
    Args:
        github: Authenticated GitHub instance.
        repo_name (str): Full name of the repository to fork (e.g., 'org/repo').
        org (str): The organization where you want to create the fork (optional).
    
    Returns:
        Repository: The forked repository.
    """
    try:
        repo = github.get_repo(repo_name)
        forked_repo = repo.create_fork(organization=org)
        print(f"Forked repository {repo_name} to {forked_repo.full_name}")
        return forked_repo
    except Exception as e:
        print(f"Error forking repository: {e}")
        raise

def create_pr(github, repo_name: str, head: str, base: str, title: str, body: str):
    """
    Creates a pull request on GitHub.
    
    Args:
        github: Authenticated GitHub instance.
        repo_name (str): Full name of the repository (e.g., 'org/repo').
        head (str): The branch with the changes.
        base (str): The branch to merge into.
        title (str): Title of the PR.
        body (str): Description of the PR.
    
    Returns:
        PullRequest: The created pull request.
    """
    try:
        repo = github.get_repo(repo_name)
        pr = repo.create_pull(title=title, body=body, head=head, base=base)
        print(f"Created PR '{title}' in {repo_name}")
        return pr
    except Exception as e:
        print(f"Error creating pull request: {e}")
        raise
