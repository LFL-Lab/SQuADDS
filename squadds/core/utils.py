import urllib.parse
import webbrowser
import getpass
import os
from huggingface_hub import HfApi, HfFolder
from squadds.core.globals import ENV_FILE_PATH

def set_huggingface_api_key():
    """
    Sets the Hugging Face API key by appending it to the .env file.
    If the API key already exists in the .env file, it does not add it again.
    If the Hugging Face token is not found, it raises a ValueError.
    """
    # Check if API key already exists
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'r') as file:
            existing_keys = file.read()
            if 'HUGGINGFACE_API_KEY=' in existing_keys:
                print('API key already exists in .env file.')
                return
    
    # Ask for the new API key
    api_key = getpass.getpass("Enter your Hugging Face API key: ")
    # Append the new API key to the .env file
    with open(ENV_FILE_PATH, 'a') as file:
        file.write(f'\nHUGGINGFACE_API_KEY={api_key}\n')
        print('API key added to .env file.')

    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")


def create_mailto_link(recipients, subject, body):
    """
    Create a mailto link with the given recipients, subject, and body.

    Args:
        recipients (list): A list of email addresses of the recipients.
        subject (str): The subject of the email.
        body (str): The body of the email.

    Returns:
        str: The generated mailto link.

    """
    # Encode the subject and body using urllib.parse.quote_plus to handle special characters
    subject_encoded = urllib.parse.quote_plus(subject)
    body_encoded = urllib.parse.quote_plus(body)

    # Construct the mailto link with the encoded subject and body
    mailto_link = f"mailto:{','.join(recipients)}?subject={subject_encoded}&body={body_encoded}"

    # Replace '+' with '%20' for proper space encoding
    mailto_link = mailto_link.replace('+', '%20')
    return mailto_link


def send_email_via_client(dataset_name, institute, pi_name, date, dataset_link):
    """
    Sends an email notification to recipients with the details of the created dataset.

    Args:
        dataset_name (str): The name of the dataset.
        institute (str): The name of the institute where the dataset was created.
        pi_name (str): The name of the principal investigator who created the dataset.
        date (str): The date when the dataset was created.
        dataset_link (str): The link to the created dataset.

    Returns:
        None
    """
    recipients = ["shanto@usc.edu", "elevenso@usc.edu"]
    subject = f"SQuADDS: Dataset Created - {dataset_name} ({date})"
    body = f"{dataset_name} has been created by {pi_name} at {institute} on {date}.\nHere is the link - {dataset_link}"

    mailto_link = create_mailto_link(recipients, subject, body)
    webbrowser.open(mailto_link)
