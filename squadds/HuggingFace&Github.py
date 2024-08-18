# Correct imports based on file structure
from database.GitHub import (
    login_to_github, fork_repo, create_pr
)

from database.HuggingFace import (
    login_to_huggingface, fork_dataset, save_dataset_to_hf, add_row_to_dataset
)

# Step 1: Log in to Hugging Face and GitHub
login_to_huggingface()
github = login_to_github()

# Step 2: Fork the GitHub repository
repo_name = "SQuADDS/squadds-db-repo"
forked_repo = fork_repo(github, repo_name)

# Step 3: Fork the Hugging Face dataset
dataset_name = "SQuADDS/SQuADDS_DB"
fork_dataset(dataset_name, "SQuADDS_DB", "my_forked_dataset")

# Step 4: Add a new row to the dataset
new_row_data = {
    "design_code": "GITHUB LINK",
    "contrib_info": {
        "group": "Research Group XYZ",
        "PI": "Dr. John Doe",
        "institution": "University ABC",
        "uploader": "Jane Doe",
        "measured_by": ["John", "Jane"],
        "date_created": "2024-08-16",
        "name": "New Experimental Device"
    },
    "measured_results": [{
        "H_params": [
            {
                "qubit_1": {
                    "F_res_GHz": 6.116,
                    "F_01_GHz": 4.216,
                    "Anharmonicity_MHz": -153,
                    "L_j_nH": 9.686
                }
            }
        ]
    }],
    "sim_results": [],
    "image": "GITHUB IMAGE LINK",
    "paper_link": "PAPER LINK",
    "foundry": "FOUNDRY NAME",
    "fabrication_recipe": "FABUBLOX LINK",
    "notes": "Additional notes"
}

# Step 5: Add the new row to your forked dataset and save it
updated_dataset = add_row_to_dataset(forked_repo, new_row_data)
save_dataset_to_hf(updated_dataset, "myusername/my_forked_dataset", "my_forked_dataset")

# Step 6: Create a pull request on GitHub
pr_title = "Added new experimental device"
pr_description = "This PR adds a newly validated experimental device to the SQuADDS database."
create_pr(github, forked_repo.full_name, "my_branch", "main", pr_title, pr_description)

print("Successfully contributed to Hugging Face and GitHub!")
