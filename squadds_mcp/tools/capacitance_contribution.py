"""
Planar capacitance contribution tools (data and models -> Hugging Face pull requests).
======================================================================================

Outward-facing actions are dry runs by default. A real pull request is opened only when the caller passes
``dry_run=False`` AND ``confirm=True``; anything else validates and returns the planned files and report. The real PR
needs a Hugging Face token.

Tools:
    - ``init_capacitance_contribution``    copy the contribution template (manifest.yaml, designs.csv, README) to a folder
    - ``contribute_capacitance_data``      validate a contribution folder and (only with confirm) open the dataset PR
    - ``validate_capacitance_model``       validate a model repository (checkpoints, sha256, metadata, val score)
    - ``contribute_capacitance_model``     validate and (only with confirm) open the model-zoo PR

``validate_capacitance_contribution`` (read-only data validation) lives in ``capacitance_ml.py``.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from squadds_mcp.tools.capacitance_ml import INSTALL_HINT, _ml_missing
from squadds_mcp.utils import sanitize_for_json

CONFIRM_NOTE = (
    "dry run: nothing was uploaded. To open the pull request, call again with dry_run=False and confirm=True after "
    "the user has reviewed the planned files and the validation report"
)


def _template_dir() -> Path:
    import squadds

    return Path(squadds.__file__).parent / "database" / "templates" / "planar_capacitance_contribution"


def init_contribution(dest_dir: str) -> dict[str, Any]:
    """Synchronous body of ``init_capacitance_contribution``."""
    src = _template_dir()
    dest = Path(dest_dir).expanduser()
    if dest.exists() and any(dest.iterdir()):
        return {"error": f"destination is not empty: {dest_dir}"}
    shutil.copytree(src, dest, dirs_exist_ok=True)
    for sub in ("gds", "sims"):
        (dest / sub).mkdir(exist_ok=True)
    files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    readme = dest / "README.md"
    return {
        "path": str(dest.resolve()),
        "files": files,
        "readme": readme.read_text(encoding="utf-8") if readme.is_file() else "",
        "layout": "manifest.yaml (contributor, licence CC-BY-4.0 / CC0-1.0, solver, stack, setup, layer map, units, "
        "family); designs.csv (design, gds, family, params_json, net_order = G first then conductors in GDS file order, "
        "notes); gds/<design>.gds; sims/<design>/<sim_tag>/cap_matrix.csv (Q3D export as is) [+ convergence.txt, run.json]",
        "next": "fill the files, then validate_capacitance_contribution(path) and contribute_capacitance_data(path)",
    }


def _outward(dry_run: bool, confirm: bool) -> dict[str, Any] | None:
    if not dry_run and not confirm:
        return {
            "error": "a real pull request needs dry_run=False AND confirm=True (outward-facing action); run the dry run "
            "first and get the user's go-ahead"
        }
    return None


def contribute_data(
    path: str,
    dry_run: bool = True,
    confirm: bool = False,
    org: str | None = None,
    batch: str | None = None,
    repo_id: str | None = None,
    reference: str = "auto",
    check_prior: bool = False,
) -> dict[str, Any]:
    """Synchronous body of ``contribute_capacitance_data``."""
    refused = _outward(dry_run, confirm)
    if refused:
        return refused
    p = Path(path).expanduser()
    if not p.is_dir():
        return {"error": f"contribution folder not found: {path}"}
    if check_prior:
        err = _ml_missing()
        if err:
            return err
    try:
        from squadds.datasets.planar_capacitance_schema import REPO_ID
        from squadds.ml.capacitance.contribute import contribute_capacitance

        out = contribute_capacitance(
            p,
            repo_id=repo_id or REPO_ID,
            dry_run=bool(dry_run),
            org=org,
            batch=batch,
            reference=None if reference == "none" else "auto",
            check_prior=check_prior,
        )
    except ImportError as e:
        return {"error": f"{e}", "install": INSTALL_HINT}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    out = {**out, "dry_run": bool(dry_run)}
    if dry_run:
        out["note"] = CONFIRM_NOTE
    return sanitize_for_json(out)


def validate_model_repo(
    model_repo_id: str,
    revision: str | None = None,
    local_dir: str | None = None,
    metadata: dict | None = None,
    val_data: str | None = None,
    threads: int = 2,
) -> dict[str, Any]:
    """Synchronous body of ``validate_capacitance_model``."""
    err = _ml_missing()
    if err:
        return err
    try:
        from squadds.ml.capacitance.contribute import validate_model

        rep, entry = validate_model(model_repo_id, revision, local_dir, metadata, val_data, threads=int(threads))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return sanitize_for_json(
        {
            "ok": rep.ok,
            "errors": rep.all_errors(),
            "warnings": rep.all_warnings(),
            "entry": entry,
            "report_markdown": rep.to_markdown("Model validation report"),
        }
    )


def contribute_model_repo(
    model_repo_id: str,
    revision: str | None = None,
    local_dir: str | None = None,
    metadata: dict | None = None,
    val_data: str | None = None,
    dry_run: bool = True,
    confirm: bool = False,
    registry_repo: str | None = None,
    threads: int = 2,
) -> dict[str, Any]:
    """Synchronous body of ``contribute_capacitance_model``."""
    refused = _outward(dry_run, confirm)
    if refused:
        return refused
    err = _ml_missing()
    if err:
        return err
    try:
        from squadds.datasets.planar_capacitance_schema import REPO_ID
        from squadds.ml.capacitance.contribute import contribute_model

        out = contribute_model(
            model_repo_id,
            revision,
            registry_repo=registry_repo or REPO_ID,
            metadata=metadata,
            val_data=val_data,
            dry_run=bool(dry_run),
            local_dir=local_dir,
            threads=int(threads),
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    out = {**out, "dry_run": bool(dry_run)}
    if dry_run:
        out["note"] = CONFIRM_NOTE
    return sanitize_for_json(out)


def register_capacitance_contribution_tools(mcp: FastMCP) -> None:
    """Register the planar capacitance contribution tools (dry run by default)."""

    @mcp.tool()
    async def init_capacitance_contribution(dest_dir: str) -> dict[str, Any]:
        """Copy the planar capacitance contribution template into a new (or empty) folder and explain its layout.

        The template holds ``manifest.yaml`` (contributor, licence, solver, stack, setup, layer map, units, family),
        ``designs.csv`` (one row per design: gds, family, params_json, net_order with ground ``G`` first) and a README;
        ``gds/`` and ``sims/`` are created empty. Fill them, then validate with ``validate_capacitance_contribution``.
        """
        return await asyncio.to_thread(init_contribution, dest_dir)

    @mcp.tool()
    async def contribute_capacitance_data(
        path: str,
        dry_run: bool = True,
        confirm: bool = False,
        org: str | None = None,
        batch: str | None = None,
        repo_id: str | None = None,
        reference: str = "auto",
        check_prior: bool = False,
    ) -> dict[str, Any]:
        """Validate a contribution folder and plan (or open) the pull request on ``SQuADDS/SQuADDS_PlanarCapacitance``.

        Default ``dry_run=True``: validates and returns the target folder ``contrib/<org>-<batch>/``, the files and the
        validation report; no network write. A real pull request is opened only with ``dry_run=False`` AND
        ``confirm=True``: show the user the dry-run result first and get their go-ahead. A contribution with validation
        errors is refused. Needs a Hugging Face token.

        Args:
            path: the contribution folder (see ``init_capacitance_contribution``).
            org / batch: name parts of the target folder (default: from the manifest; batch = today's date).
            repo_id: target dataset (default ``SQuADDS/SQuADDS_PlanarCapacitance``).
            reference: ``auto`` (duplicate check against the dataset) or ``none``.
            check_prior: also compare each label with the optimized physics prior (``[ml]`` extra; seconds per design).
        """
        return await asyncio.to_thread(
            contribute_data, path, dry_run, confirm, org, batch, repo_id, reference, check_prior
        )

    @mcp.tool()
    async def validate_capacitance_model(
        model_repo_id: str,
        revision: str | None = None,
        local_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
        val_data: str | None = None,
        threads: int = 2,
    ) -> dict[str, Any]:
        """Validate a capacitance model repository for the model zoo (read-only).

        Checks that the ``*.pt`` checkpoints load, records their sha256, and checks ``squadds_model.yaml``: name, title,
        licence, contributor, ``training_data`` (sha256, split used) and ``val_score`` (median %, MAE and RMSE in fF,
        layouts, seconds, machine, threads). ``val_data`` (a validation shard dir) recomputes the score here. Self-reported
        test scores are rejected (test scoring is maintainer-run, once per declared model).

        Args:
            model_repo_id: the Hugging Face model repo (``you/your-model``). revision: commit or branch.
            local_dir: a local copy of the repository (no download). metadata: overrides ``squadds_model.yaml``.
        """
        return await asyncio.to_thread(
            validate_model_repo, model_repo_id, revision, local_dir, metadata, val_data, threads
        )

    @mcp.tool()
    async def contribute_capacitance_model(
        model_repo_id: str,
        revision: str | None = None,
        local_dir: str | None = None,
        metadata: dict[str, Any] | None = None,
        val_data: str | None = None,
        dry_run: bool = True,
        confirm: bool = False,
        registry_repo: str | None = None,
        threads: int = 2,
    ) -> dict[str, Any]:
        """Validate a model repository and plan (or open) the pull request adding it to ``model_zoo.yaml``.

        Default ``dry_run=True``: returns the zoo entry, the planned files and the validation report, no write (with
        ``local_dir`` and no network at all). A real pull request needs ``dry_run=False`` AND ``confirm=True`` (show the
        user the dry run first). A model with validation errors or an existing repo@revision entry is refused.

        Args:
            model_repo_id / revision / local_dir / metadata / val_data: as in ``validate_capacitance_model``.
            registry_repo: the dataset holding ``model_zoo.yaml`` (default ``SQuADDS/SQuADDS_PlanarCapacitance``).
        """
        return await asyncio.to_thread(
            contribute_model_repo,
            model_repo_id,
            revision,
            local_dir,
            metadata,
            val_data,
            dry_run,
            confirm,
            registry_repo,
            threads,
        )
