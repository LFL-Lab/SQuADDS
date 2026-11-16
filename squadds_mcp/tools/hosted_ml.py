"""
Hosted ML API tools.
====================

The models served by the SQuADDS ML API (the SQuADDS Hugging Face Space), through ``squadds.ml.client``. No ``[ml]``
extra is needed: the models run on the Space.

Tools:
    - ``list_hosted_ml_models``          the hosted models (``GET /v1/models``), each with its contract
    - ``predict_with_hosted_ml_model``   one prediction with JSON inputs (``POST /v1/models/{model_id}/predict``)

The base URL is ``SQUADDS_ML_API_URL`` when set, else the Space. Gated models need ``HF_TOKEN`` in the server's environment.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP

from squadds_mcp.utils import sanitize_for_json


def _error(e: Exception, base_url: str) -> dict[str, Any]:
    from squadds.ml import client

    return {"error": f"{type(e).__name__}: {e}", "api": client._url(base_url or None)}


def list_hosted(base_url: str = "", details: bool = True) -> dict[str, Any]:
    """Synchronous body of ``list_hosted_ml_models``."""
    from squadds.ml import client

    try:
        models = client.list_models(base_url=base_url or None)
        if details:
            models = [client.get_model(m["id"], base_url=base_url or None) for m in models]
    except Exception as e:  # noqa: BLE001
        return _error(e, base_url)
    return {"api": client._url(base_url or None), "models": sanitize_for_json(models)}


def predict_hosted(model_id: str, inputs: Any, options: dict | None = None, base_url: str = "") -> dict[str, Any]:
    """Synchronous body of ``predict_with_hosted_ml_model``."""
    from squadds.ml import client

    try:
        return sanitize_for_json(client.predict(model_id, inputs=inputs, options=options, base_url=base_url or None))
    except Exception as e:  # noqa: BLE001
        return {**_error(e, base_url), "model_id": model_id}


def register_hosted_ml_tools(mcp: FastMCP) -> None:
    """Register the hosted ML API tools on the MCP server."""

    @mcp.tool()
    async def list_hosted_ml_models(base_url: str = "", details: bool = True) -> dict[str, Any]:
        """List the models served by the SQuADDS ML API. This list is the source of truth for what is available.

        Args:
            base_url: API base URL (default: ``SQUADDS_ML_API_URL`` or the SQuADDS Space).
            details: also fetch each model's contract (``GET /v1/models/{model_id}``): input and output names and units,
                limits and an example request.

        Returns ``models`` (``id``, ``family``, ``display_name``, ``description``, ``input_kind``, ``revision``,
        ``status``, plus the contract when ``details``) and ``api``. Call ``predict_with_hosted_ml_model`` with a model
        whose ``status`` is ``ready``, using the input names it lists.
        """
        return await asyncio.to_thread(list_hosted, base_url, details)

    @mcp.tool()
    async def predict_with_hosted_ml_model(
        model_id: str,
        inputs: dict[str, Any] | list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        base_url: str = "",
    ) -> dict[str, Any]:
        """Run one hosted SQuADDS ML model with JSON inputs (``POST /v1/models/{model_id}/predict``).

        Args:
            model_id: a model id from ``list_hosted_ml_models``, e.g. ``transmon_cross_hamiltonian_inverse``.
            inputs: the model's input fields with their exact names, e.g. ``{"qubit_frequency_GHz": 4.85,
                "anharmonicity_MHz": -205.0}``, or a list of such objects (a batch). A model that takes a file accepts
                it as ``inputs.file_base64``.
            options: model options (see the model's contract), e.g. ``{"include_scaled_outputs": false}``.
            base_url: API base URL (default: ``SQUADDS_ML_API_URL`` or the SQuADDS Space).

        Returns the response envelope: ``outputs``, ``metadata`` (names, orders and units), ``timing_s``, ``revision``
        and ``cached``; or ``error`` with the API's message (an unknown or unlisted model is ``unknown_model``).
        """
        return await asyncio.to_thread(predict_hosted, model_id, inputs, options, base_url)
