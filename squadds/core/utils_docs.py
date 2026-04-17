import re

import requests
from tabulate import tabulate


def view_contributors_from_rst(rst_url):
    contributors_data = []
    try:
        response = requests.get(rst_url, timeout=10)
    except requests.RequestException as exc:
        print(f"Failed to fetch the file: {exc}")
        return

    if response.status_code == 200:
        content = response.text
        contributors_match = re.search(r"Contributors\s+-{3,}\s+(.*?)(\n\n|$)", content, re.S)
        if contributors_match:
            contributors_section = contributors_match.group(1).strip()
            contributor_entries = contributors_section.split("\n| ")

            for entry in contributor_entries:
                if not entry.strip():
                    continue
                match = re.match(r"\*\*(.*?)\*\* \((.*?)\) - (.*)", entry.strip())
                if match:
                    contributors_data.append([match.group(1), match.group(2), match.group(3)])

        if contributors_data:
            print(tabulate(contributors_data, headers=["Name", "Institution", "Contribution"], tablefmt="grid"))
        else:
            print("No contributors found in the RST file.")
    else:
        print(f"Failed to fetch the file. Status code: {response.status_code}")
