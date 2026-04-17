import urllib.parse
import webbrowser


def create_mailto_link(recipients, subject, body):
    subject_encoded = urllib.parse.quote_plus(subject)
    body_encoded = urllib.parse.quote_plus(body)
    mailto_link = f"mailto:{','.join(recipients)}?subject={subject_encoded}&body={body_encoded}"
    return mailto_link.replace("+", "%20")


def send_email_via_client(dataset_name, institute, pi_name, date, dataset_link):
    recipients = ["shanto@usc.edu", "elevenso@usc.edu"]
    subject = f"SQuADDS: Dataset Created - {dataset_name} ({date})"
    body = f"{dataset_name} has been created by {pi_name} at {institute} on {date}.\nHere is the link - {dataset_link}"
    webbrowser.open(create_mailto_link(recipients, subject, body))
