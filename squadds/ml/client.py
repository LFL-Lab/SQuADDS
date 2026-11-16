"""Client of the SQuADDS ML API (the models hosted on the SQuADDS Hugging Face Space).

    from squadds.ml import client
    client.list_models()
    client.get_model("transmon_cross_hamiltonian_inverse")     # inputs, outputs, units, limits, example request
    client.predict("transmon_cross_hamiltonian_inverse", inputs={"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0})
    client.predict("<file model id>", file="chip.gds", inputs={"ground_layer": "1/0"})

The base URL is ``SQUADDS_ML_API_URL`` when set, else ``DEFAULT_URL``. Gated models need a Hugging Face token of a member
of the SQuADDS organization (``token=`` or ``HF_TOKEN``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_URL = "https://squadds-squadds-ml-inference-api.hf.space"


def _url(base_url: str | None) -> str:
    return (base_url or os.environ.get("SQUADDS_ML_API_URL") or DEFAULT_URL).rstrip("/")


def _headers(token: str | None) -> dict:
    token = token or os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _json(r) -> dict:
    try:
        body = r.json()
    except ValueError:
        body = {"error": {"message": r.text}}
    if r.status_code >= 400:
        err = body.get("error") or body
        raise RuntimeError(f"SQuADDS ML API {r.status_code}: {err}")
    return body


def list_models(base_url: str | None = None, token: str | None = None, timeout: float = 30) -> list[dict]:
    import requests

    return _json(requests.get(f"{_url(base_url)}/v1/models", headers=_headers(token), timeout=timeout))["models"]


def get_model(model_id: str, base_url: str | None = None, token: str | None = None, timeout: float = 30) -> dict:
    """One model's listing entry and contract (``GET /v1/models/{model_id}``)."""
    import requests

    return _json(requests.get(f"{_url(base_url)}/v1/models/{model_id}", headers=_headers(token), timeout=timeout))


def predict(
    model_id: str,
    inputs: dict | list | None = None,
    file=None,
    options: dict | None = None,
    base_url: str | None = None,
    token: str | None = None,
    timeout: float = 600,
) -> dict:
    """The response envelope (``outputs``, ``metadata``, ``timing_s``, ``cached``). ``file``: a path or bytes (file models)."""
    import requests

    url = f"{_url(base_url)}/v1/models/{model_id}/predict"
    if file is None:
        r = requests.post(
            url, json={"inputs": inputs or {}, "options": options or {}}, headers=_headers(token), timeout=timeout
        )
    else:
        data = Path(file).read_bytes() if isinstance(file, (str, os.PathLike)) else bytes(file)
        form = {"inputs": json.dumps(inputs or {}), "options": json.dumps(options or {})}
        r = requests.post(
            url, files={"file": ("layout.gds", data)}, data=form, headers=_headers(token), timeout=timeout
        )
    return _json(r)
