import re

import requests
from tabulate import tabulate


def view_contributors_from_rst(rst_file_path):
    """
    Extract and return relevant contributor information from the index.rst file in Markdown format.

    Args:
        rst_file_path (str): The path to the `index.rst` file.

    Returns:
        str: A Markdown-formatted table of contributors.
    """

    contributors_data = []

    with open(rst_file_path, 'r') as file:
        content = file.read()

        # Find the Contributors section
        contributors_match = re.search(r'Contributors\s+-{3,}\s+(.*?)(\n\n|$)', content, re.S)
        if contributors_match:
            contributors_section = contributors_match.group(1).strip()

            # Extract individual contributor entries
            contributor_entries = contributors_section.split("\n| ")

            for entry in contributor_entries:
                if entry.strip():
                    # Extract name, institution, and contribution
                    match = re.match(r'\*\*(.*?)\*\* \((.*?)\) - (.*)', entry.strip())
                    if match:
                        name = match.group(1)
                        institution = match.group(2)
                        contribution = match.group(3)
                        contributors_data.append([name, institution, contribution])

    if contributors_data:
        headers = ["Name", "Institution", "Contribution"]
        markdown_table = tabulate(contributors_data, headers=headers, tablefmt="pipe")
        return f"\n\n{markdown_table}\n"
    else:
        return "\n"

def fetch_github_contributors():
    """
    Fetch GitHub contributors and return them as a Markdown-formatted string.

    Returns:
        str: A Markdown-formatted string of GitHub contributors.
    """
    # GitHub API URL for contributors
    url = "https://api.github.com/repos/lfl-lab/SQuADDS/contributors"

    # Fetch contributors data from GitHub API
    response = requests.get(url)
    contributors = response.json()

    # Initialize variables for contributions
    shanto_contributions = 0
    other_contributors = []

    # Process the contributors
    for contributor in contributors:
        if contributor['login'] == 'shanto268' or contributor['login'] == 'actions-user':
            shanto_contributions += contributor['contributions']
        else:
            other_contributors.append(f"- [{contributor['login']}]({contributor['html_url']}) - {contributor['contributions']} contributions")

    # Prepare the final output
    contributors_output = [
        f"- [shanto268](https://github.com/shanto268) - {shanto_contributions} contributions"
    ] + other_contributors

    # Convert the list to a string
    final_output = "\n".join(contributors_output)
    
    return final_output

def update_readme(readme_path, rst_file_path):
    """
    Update the README.md file with the latest contributors section.

    Args:
        readme_path (str): Path to the README.md file.
        rst_file_path (str): Path to the index.rst file to extract contributors.

    Returns:
        None
    """
    # Fetch GitHub contributors
    github_contributors = fetch_github_contributors()

    # Fetch contributors from the index.rst file
    rst_contributors = view_contributors_from_rst(rst_file_path)

    # Combine both sections
    header = "## Contributors\n" 
    combined_contributors = header + rst_contributors + "\n" + "## Developers\n" + github_contributors + "\n---"

    # Read the existing README.md content
    with open(readme_path, "r") as file:
        content = file.read()

    # Replace the Contributors and Developers sections
    new_content = re.sub(
        r"## Contributors.*?## Developers.*?(\n---|\Z)",  # Match from '## Contributors' to the '## Developers' section and its end
        combined_contributors + "\n",  # Replace with the updated Contributors and Developers sections
        content,
        flags=re.DOTALL  # Enable matching across multiple lines
    )

    # Write the updated content back to README.md
    with open(readme_path, "w") as file:
        file.write(new_content)

    # Optionally, print the output for verification
    print(new_content)

# Example usage:
# Assuming your README.md is located in the root of your project and the index.rst file is in the `docs` directory
readme_path = "README.md"
rst_file_path = "../docs/source/developer/index.rst"
update_readme(readme_path, rst_file_path)