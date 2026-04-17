SIMULATION_CONTRIBUTOR_KEYS = ("uploader", "PI", "group", "institution")


def unique_contributor_records(contributor_rows, keys):
    """
    Return unique contributor mappings preserving their first-seen order.
    """
    unique_rows = []
    for contributor in contributor_rows:
        record = {key: contributor.get(key, "N/A") for key in keys}
        if record not in unique_rows:
            unique_rows.append(record)
    return unique_rows


def collect_all_simulation_contributors(configs, repo_name, load_dataset_fn):
    """
    Gather unique simulation contributors across all dataset configs.
    """
    unique_rows = []
    for config in configs:
        dataset = load_dataset_fn(repo_name, config)["train"]
        for row in unique_contributor_records(dataset["contributor"], SIMULATION_CONTRIBUTOR_KEYS):
            row_with_config = dict(row)
            row_with_config["Config"] = config
            if row_with_config not in unique_rows:
                unique_rows.append(row_with_config)
    return unique_rows


def build_measured_device_records(dataset):
    """
    Build the measured-device dataframe payload used by ``get_measured_devices``.
    """
    records = []
    for entry in zip(
        dataset["contrib_info"],
        dataset["design_code"],
        dataset["paper_link"],
        dataset["image"],
        dataset["foundry"],
        dataset["fabrication_recipe"],
        dataset["substrate"],
        dataset["materials"],
        dataset["junction_style"],
        dataset["junction_material"],
    ):
        (
            contrib_info,
            design_code,
            paper_link,
            image,
            foundry,
            _recipe,
            substrate,
            materials,
            junction_style,
            junction_materials,
        ) = entry

        records.append(
            {
                "Name": contrib_info.get("name", "N/A"),
                "Design Code": design_code,
                "Paper Link": paper_link,
                "Image": image,
                "Foundry": foundry,
                "Substrate": substrate,
                "Materials": materials,
                "Junction Style": junction_style,
                "Junction Materials": junction_materials,
            }
        )
    return records


def build_measured_device_rows(dataset):
    """
    Build the rows printed by ``view_measured_devices``.
    """
    rows = []
    for entry in zip(
        dataset["contrib_info"],
        dataset["design_code"],
        dataset["paper_link"],
        dataset["image"],
        dataset["foundry"],
        dataset["fabrication_recipe"],
    ):
        contrib_info, design_code, paper_link, image, foundry, recipe = entry
        rows.append(
            {
                "Name": contrib_info.get("name", "N/A"),
                "Design Code": design_code,
                "Paper Link": paper_link,
                "Image": image,
                "Foundry": foundry,
                "Fabrication Recipe": recipe,
            }
        )
    return rows


def find_simulation_results_for_device(dataset, device_name):
    """
    Return simulation results for a measured device name.
    """
    for contrib_info, sim_results in zip(dataset["contrib_info"], dataset["sim_results"]):
        if contrib_info["name"] == device_name:
            return sim_results
    return {}


def find_device_contributor_info(dataset, config):
    """
    Return contributor information for the measured device that validates a config.
    """
    for contrib_info, sim_results in zip(dataset["contrib_info"], dataset["sim_results"]):
        if config in sim_results:
            return {
                "Foundry": contrib_info.get("foundry", "N/A"),
                "PI": contrib_info.get("PI", "N/A"),
                "Group": contrib_info.get("group", "N/A"),
                "Institution": contrib_info.get("institution", "N/A"),
                "Measured By": ", ".join(contrib_info.get("measured_by", [])),
                "Reference Device Name": contrib_info.get("name", "N/A"),
                "Uploader": contrib_info.get("uploader", "N/A"),
            }
    return None


def find_reference_device_info(dataset, config):
    """
    Return the combined measured-device metadata for a config.
    """
    for entry in zip(
        dataset["contrib_info"],
        dataset["sim_results"],
        dataset["design_code"],
        dataset["paper_link"],
        dataset["image"],
        dataset["foundry"],
        dataset["fabrication_recipe"],
    ):
        contrib_info, sim_results, design_code, paper_link, image, foundry, recipe = entry

        if config in sim_results:
            combined_info = {
                "Design Code": design_code,
                "Paper Link": paper_link,
                "Image": image,
                "Foundry": foundry,
                "Fabrication Recipe": recipe,
            }
            combined_info.update(contrib_info)
            return combined_info
    return None


def build_recipe_rows(dataset, device_name):
    """
    Return the printable recipe rows for a measured device.
    """
    for contrib_info, foundry, recipe, github_url in zip(
        dataset["contrib_info"],
        dataset["foundry"],
        dataset["fabrication_recipe"],
        dataset["design_code"],
    ):
        if contrib_info["name"] == device_name:
            return [
                ["Foundry", foundry],
                ["Fabublox Link", recipe],
                ["Fabrication Recipe Links", f"{github_url}/tree/main/Fabrication"],
            ]
    return None


def build_reference_device_records(dataset, simulation_lookup_fn):
    """
    Return unique reference-device rows for ``view_reference_devices``.
    """
    unique_rows = []
    for contrib_info in dataset["contrib_info"]:
        record = {key: contrib_info[key] for key in ["name", "group", "measured_by"]}
        device_name = contrib_info["name"]
        record["simulations"] = simulation_lookup_fn(device_name)
        if record not in unique_rows:
            unique_rows.append(record)
    return unique_rows
