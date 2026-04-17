DEFAULT_COMPONENT_IMAGE_BASE_URL = "https://github.com/LFL-Lab/SQuADDS/tree/master/docs/_static/images"


def build_dataset_rows(components, component_names, data_types, image_base_url=DEFAULT_COMPONENT_IMAGE_BASE_URL):
    """
    Build the tabular dataset rows displayed by ``SQuADDS_DB.view_datasets``.
    """
    component_urls = [f"{image_base_url}/{name}.png" for name in component_names]
    rows = list(map(list, zip(components, component_names, data_types, component_urls)))
    seen = set()
    deduped_rows = []
    for row in rows:
        row_key = tuple(row)
        if row_key in seen:
            continue
        seen.add(row_key)
        deduped_rows.append(row)
    return deduped_rows


def describe_dataset(dataset):
    """
    Collect the printable metadata fields for a Hugging Face dataset split.
    """
    return {
        "features": dataset.features,
        "description": dataset.description,
        "citation": dataset.citation,
        "homepage": dataset.homepage,
        "license": dataset.license,
        "size_in_bytes": dataset.size_in_bytes,
    }
