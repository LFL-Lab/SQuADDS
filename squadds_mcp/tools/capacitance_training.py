"""
Planar capacitance training, scoring and label QC tools.
========================================================

Retraining runs on the user's own compute. Nothing long runs inside the server: the planning tool writes a driver script
and a generic SLURM job (no site-specific accounts or tooling) and returns the commands; the in-process tools are bounded
(a few designs for shard building, a few hundred steps for a smoke fit, a few layouts for end-to-end scoring).

Tools:
    - ``check_capacitance_label``           label QC of a Q3D ``cap_matrix.csv`` or a matrix (symmetry, signs, closure, PD)
    - ``build_capacitance_training_shards`` a few GDS + labels -> a training shard dir (embedding + physics priors)
    - ``plan_capacitance_training``         driver + SLURM scripts for fit / in-family / collective retraining
    - ``run_capacitance_fit_smoke``         a short in-process fit (<= ``MAX_SMOKE_STEPS``) to check a shard dir end to end
    - ``score_capacitance_model``           validation-split score: median % with MAE and RMSE in fF, time, machine, threads
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from squadds_mcp.tools.capacitance_ml import _ml_missing
from squadds_mcp.utils import sanitize_for_json

MAX_INPROCESS_DESIGNS = 16
MAX_SMOKE_STEPS = 500
MAX_SCORE_LAYOUTS = 20
MODES = ("build", "fit", "in_family", "collective")
FULL_FIT_NOTE = (
    "paper recipe: 18,000 steps per seed, about 240 s per seed on a GH200 node CPU with 2 threads (release fit 239 s); "
    "plan minutes per seed on a cluster CPU, longer on a laptop"
)


def check_label(
    path: str | None = None,
    matrix_fF: list[list[float]] | None = None,
    names: list[str] | None = None,
    unit: str = "fF",
    net_order: list[str] | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``check_capacitance_label``."""
    from squadds.ml.capacitance import qc as Q

    try:
        if path:
            p = Path(path).expanduser()
            if not p.is_file():
                return {"error": f"file not found: {path}"}
            data = Q.read_cap_matrix(p)
            done = [(k, m) for k, m in data["passes"] if not np.isnan(m).any()]
            if not done:
                return {"error": f"{path}: no complete pass", "names": data["names"]}
            rep = Q.label_qc(done[-1][1], data["names"], net_order=net_order, passes=done, subject=p.name)
            extra = {"passes": len(done), "file_units": data.get("units"), "freq_GHz": data.get("freq_GHz")}
        elif matrix_fF is not None and names is not None:
            rep = Q.label_qc(matrix_fF, names, net_order=net_order, unit=unit, subject="matrix")
            extra = {}
        else:
            return {"error": "pass path (a Q3D cap_matrix.csv) or matrix_fF + names"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    info = {k: v for k, v in rep.info.items() if k != "matrix_fF"}
    out = {
        "ok": rep.ok,
        "errors": rep.all_errors(),
        "warnings": rep.all_warnings(),
        "info": info,
        **extra,
        "report_markdown": rep.to_markdown("Label QC"),
    }
    if "matrix_fF" in rep.info:
        out["names"] = list(net_order or (data["names"] if path else names))
        out["matrix_fF"] = rep.info["matrix_fF"]
    return sanitize_for_json(out)


def build_training_shards(
    designs: list[dict],
    out_dir: str,
    prior: str = "optimized",
    threads: int = 2,
    on_error: str = "skip",
) -> dict[str, Any]:
    """Synchronous body of ``build_capacitance_training_shards``."""
    err = _ml_missing()
    if err:
        return err
    if not designs:
        return {"error": "no designs"}
    if len(designs) > MAX_INPROCESS_DESIGNS:
        return {
            "error": f"{len(designs)} designs: the server builds at most {MAX_INPROCESS_DESIGNS} in process",
            "hint": "write the designs to a JSON file and call plan_capacitance_training(mode='build', designs_json=...) "
            "to build them on your own compute",
        }
    out = Path(out_dir).expanduser()
    if out.exists() and any(out.iterdir()):
        return {"error": f"out_dir is not empty: {out_dir}"}
    try:
        from squadds.ml.capacitance.training_data import build_shards, shard_summary

        res = build_shards(designs, out, prior=prior, threads=int(threads), on_error=on_error)
        summary = shard_summary(out)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return sanitize_for_json(
        {
            **res,
            "summary": summary,
            "threads": threads,
            "note": "priors computed on this machine (macOS differs from the released x86 priors by up to ~2.4e-3 "
            "relative); next: plan_capacitance_training or run_capacitance_fit_smoke",
        }
    )


def _sbatch_header(job_name, log_dir, cpus, mem, time, partition, account, gpus, extra_sbatch) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={log_dir}/%x_%j.out",
    ]
    lines += [f"#SBATCH --partition={partition}"] if partition else []
    lines += [f"#SBATCH --account={account}"] if account else []
    lines += [f"#SBATCH --gpus={int(gpus)}"] if gpus else []
    lines += [f"#SBATCH {x}" for x in extra_sbatch or ()]
    return lines


def _driver(mode, designs_json, shard_dir, base_shards, out_dir, seeds, steps, threads, device, k_folds, family) -> str:
    cfg = f"TrainConfig(steps={int(steps)})"
    designs = f"json.loads(Path({str(designs_json)!r}).read_text(encoding='utf-8'))" if designs_json else "None"
    body = {
        "build": f"res = build_shards(DESIGNS, {shard_dir!r}, threads={threads}, on_error='skip')\nprint(res)",
        "in_family": (
            f"res = fit_in_family(DESIGNS if DESIGNS is not None else {shard_dir!r}, {str(out_dir)!r}, "
            f"k_folds={k_folds}, config={cfg}, device={device!r}, threads={threads}, family={family!r}, "
            f"final_seeds={tuple(seeds)!r})\nprint(res)\nprint(res.report)"
        ),
        "collective": (
            "user = "
            + (
                f"build_shards(DESIGNS, {str(Path(out_dir) / 'user_shards')!r}, threads={threads})['out_dir']"
                if designs_json
                else f"{shard_dir!r}"
            )
            + f"\nmodel, runs = retrain_collective(user, {base_shards!r}, {str(out_dir)!r}, seeds={tuple(seeds)!r}, "
            f"config={cfg}, device={device!r}, threads={threads})\nprint(model)\n"
            "print([{k: r[k] for k in ('seed', 'train_seconds', 'machine', 'threads') if k in r} for r in runs])"
        ),
    }[mode]
    return "\n".join(
        [
            f'"""Generated by the SQuADDS MCP server (plan_capacitance_training, mode={mode!r}). Edit freely."""',
            "",
            "import json",
            "from pathlib import Path",
            "",
            "from squadds.ml.capacitance import TrainConfig",
            "from squadds.ml.capacitance.train import fit_in_family, retrain_collective",
            "from squadds.ml.capacitance.training_data import build_shards",
            "",
            f"DESIGNS = {designs}",
            "",
            body,
            "",
        ]
    )


def plan_training(
    mode: str,
    out_dir: str,
    script_dir: str,
    shard_dir: str | None = None,
    designs_json: str | None = None,
    base_shards: str | None = None,
    seeds: list[int] | None = None,
    steps: int = 18000,
    threads: int = 2,
    device: str = "cpu",
    k_folds: int = 5,
    family: str = "user",
    partition: str | None = None,
    account: str | None = None,
    time: str = "01:00:00",
    mem: str = "6G",
    gpus: int = 0,
    setup: str = "",
    python: str | None = None,
    extra_sbatch: list[str] | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``plan_capacitance_training``."""
    if mode not in MODES:
        return {"error": f"mode must be one of {MODES}"}
    seeds = [int(s) for s in (seeds if seeds is not None else ([0] if mode == "in_family" else [0, 1, 2]))]
    python = python or sys.executable
    sd = Path(script_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if designs_json and not Path(designs_json).expanduser().is_file():
        return {"error": f"designs_json not found: {designs_json}"}
    if mode in ("build", "fit") and not shard_dir:
        return {"error": f"mode={mode!r} needs shard_dir (build: where to write it; fit: the shards to train on)"}
    if mode == "build" and not designs_json:
        return {"error": "mode='build' needs designs_json (a JSON list of designs: gds, labels, label_names, key, ...)"}
    if mode == "in_family" and not (designs_json or shard_dir):
        return {"error": "mode='in_family' needs designs_json or shard_dir"}
    if mode == "collective" and not base_shards:
        return {
            "error": "mode='collective' needs base_shards: the QCap all-data shard dir (the dataset's "
            "model_input converted with training_data.shards_from_model_input)"
        }
    if mode == "collective" and not (designs_json or shard_dir):
        return {"error": "mode='collective' needs designs_json or shard_dir (your designs)"}
    sd.mkdir(parents=True, exist_ok=True)
    logs = sd / "logs"
    designs_json = str(Path(designs_json).expanduser().resolve()) if designs_json else None
    shard_dir = str(Path(shard_dir).expanduser().resolve()) if shard_dir else None
    base_shards = str(Path(base_shards).expanduser().resolve()) if base_shards else None
    files: dict[str, str] = {}
    if mode == "fit":
        from squadds.ml.capacitance.slurm import write_slurm_array

        job = write_slurm_array(
            sd / "train_array.sbatch",
            shard_dir,
            seeds,
            out_dir=out,
            partition=partition,
            account=account,
            cpus=threads,
            time=time,
            mem=mem,
            python=python,
            steps=steps,
            device=device,
            gpus=gpus,
            extra_sbatch=tuple(extra_sbatch or ()),
            setup=setup,
        )
        files["slurm"] = str(job)
        local = (
            f"{shlex.quote(python)} -m squadds.ml.capacitance.train --data {shlex.quote(shard_dir)} "
            f"--out {shlex.quote(str(out))} --seeds {','.join(map(str, seeds))} --steps {int(steps)} "
            f"--threads {int(threads)} --device {device}"
        )
        cluster = f"sbatch {shlex.quote(str(job))}   # one seed per array task"
    else:
        driver = sd / f"{mode}_driver.py"
        driver.write_text(
            _driver(mode, designs_json, shard_dir, base_shards, out, seeds, steps, threads, device, k_folds, family),
            encoding="utf-8",
        )
        job = sd / f"{mode}.sbatch"
        lines = _sbatch_header(f"squadds-ml-{mode}", logs, threads, mem, time, partition, account, gpus, extra_sbatch)
        lines += ["set -euo pipefail", *([setup] if setup else [])]
        lines += [
            f"export OMP_NUM_THREADS={int(threads)} MKL_NUM_THREADS={int(threads)} OPENBLAS_NUM_THREADS={int(threads)}"
        ]
        lines += [f"mkdir -p {shlex.quote(str(logs))}", f"{shlex.quote(python)} {shlex.quote(str(driver))}", ""]
        logs.mkdir(parents=True, exist_ok=True)
        job.write_text("\n".join(lines), encoding="utf-8")
        job.chmod(0o755)
        files.update(driver=str(driver), slurm=str(job))
        local = f"{shlex.quote(python)} {shlex.quote(str(driver))}"
        cluster = f"sbatch {shlex.quote(str(job))}"
    return {
        "mode": mode,
        "files": files,
        "run_locally": local,
        "run_on_slurm": cluster,
        "out_dir": str(out),
        "seeds": seeds,
        "steps": int(steps),
        "threads": int(threads),
        "device": device,
        "time_note": FULL_FIT_NOTE,
        "reproducibility": "released models were fit on Linux aarch64 (GH200) CPU, 2 threads, numpy 2.5.3, torch 2.6.0; "
        "other machines, devices or thread counts drift slightly (each checkpoint records its environment)",
        "next": "after training: score_capacitance_model(model=<run dir>, data=<shard dir>, split='val'); then "
        "validate_capacitance_model / contribute_capacitance_model to share it",
        "edit": "the scripts are plain text: add modules, partitions or accounts of your cluster; nothing is site-specific",
        **(
            {"note": f"{mode} creates out_dir itself: it must not exist when the job starts"}
            if mode in ("in_family", "collective")
            else {}
        ),
    }


def fit_smoke(
    shard_dir: str,
    out_dir: str,
    steps: int = 200,
    seed: int = 0,
    threads: int = 2,
    arm: str = "full_gnn",
) -> dict[str, Any]:
    """Synchronous body of ``run_capacitance_fit_smoke``."""
    err = _ml_missing()
    if err:
        return err
    if int(steps) > MAX_SMOKE_STEPS:
        return {
            "error": f"steps={steps}: in-process fits are capped at {MAX_SMOKE_STEPS} steps (a smoke test)",
            "hint": "use plan_capacitance_training(mode='fit', ...) for a full fit on your own compute",
        }
    if not Path(shard_dir).expanduser().is_dir():
        return {"error": f"shard dir not found: {shard_dir}"}
    out = Path(out_dir).expanduser()
    if out.exists() and any(out.iterdir()):
        return {"error": f"out_dir is not empty: {out_dir}"}
    try:
        from squadds.ml.capacitance import TrainConfig
        from squadds.ml.capacitance.train import fit

        cfg = TrainConfig(steps=int(steps), log_every=max(1, int(steps) // 4))
        run = fit(shard_dir, out, arm=arm, seed=int(seed), config=cfg, threads=int(threads), verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    keep = ("arm", "seed", "steps", "train_seconds", "machine", "threads", "device", "val", "release_environment_match")
    return sanitize_for_json(
        {
            "run": {k: run[k] for k in keep if k in run},
            "model_path": str((out / "model.pt").resolve()),
            "warning": f"a {steps}-step smoke fit checks the pipeline; it is not a usable model (the recipe is 18,000 steps)",
        }
    )


def score_model(
    model: str = "QCap",
    data: str | None = None,
    layouts: list[dict] | None = None,
    split: str = "val",
    threads: int = 2,
    families: list[str] | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``score_capacitance_model``."""
    err = _ml_missing()
    if err:
        return err
    if split == "test":
        return {
            "error": "test scoring is maintainer-only (once per declared model, recorded in a ledger); score on 'val'",
        }
    try:
        from squadds.ml.capacitance.benchmark import score

        if data:
            if not Path(data).expanduser().is_dir():
                return {"error": f"shard dir not found: {data}"}
            rep = score(model, str(Path(data).expanduser()), split=split, threads=int(threads), families=families)
        elif layouts:
            if len(layouts) > MAX_SCORE_LAYOUTS:
                return {
                    "error": f"{len(layouts)} layouts: end-to-end scoring in the server is capped at {MAX_SCORE_LAYOUTS}",
                    "hint": "build shards (plan_capacitance_training mode='build') and score the shard dir instead",
                }
            items = [x["gds"] for x in layouts]
            labels = [x["labels"] if isinstance(x["labels"], str) else (x["names"], x["labels"]) for x in layouts]
            rep = score(model, items, labels=labels, split=split, threads=int(threads))
        else:
            return {
                "error": "pass data (a shard dir) or layouts ([{gds, labels: cap_matrix.csv path or matrix, names}])"
            }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    d = rep.to_dict()
    d["summary"] = str(rep)
    d["metric"] = "per layout: median of |p - y| / max(|y|, 0.1 fF) over mutual entries, then the mean over layouts"
    return sanitize_for_json(d)


def register_capacitance_training_tools(mcp: FastMCP) -> None:
    """Register the planar capacitance training, scoring and label-QC tools."""

    @mcp.tool()
    async def check_capacitance_label(
        path: str | None = None,
        matrix_fF: list[list[float]] | None = None,
        names: list[str] | None = None,
        unit: str = "fF",
        net_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Label QC of a capacitance matrix before training or contributing it (no ``[ml]`` extra needed).

        Pass ``path`` (a Q3D ``cap_matrix.csv`` export: every adaptive pass is read, any unit) or ``matrix_fF`` + ``names``
        (``unit`` converts F / nF / pF / fF / aF). Checks: square over the net order, ground ``G`` present and first,
        unique names, symmetry, Maxwell signs (diagonal > 0, off-diagonal < 0), closure, positive-definite signal block,
        unit plausibility, the last pass equal to the label, monotone passes. ``net_order`` (G first) reorders and checks
        the net names. Returns ``ok``, errors, warnings, the measured values and the matrix in fF.
        """
        return await asyncio.to_thread(check_label, path, matrix_fF, names, unit, net_order)

    @mcp.tool()
    async def build_capacitance_training_shards(
        designs: list[dict[str, Any]],
        out_dir: str,
        prior: str = "optimized",
        threads: int = 2,
        on_error: str = "skip",
    ) -> dict[str, Any]:
        """Turn a few of your layouts and labels into a training shard dir (embedding + physics priors, on this machine).

        For more than 16 designs use ``plan_capacitance_training(mode="build")`` and run it on your own compute.

        Args:
            designs: one dict per design: ``gds`` (path), ``labels`` (a Q3D ``cap_matrix.csv`` path, or a Maxwell matrix
                in fF with ``label_names``, ground ``G``), optional ``key``, ``family``, ``split`` (``train`` / ``val``),
                ``names`` (``auto`` default; how layout bodies map to label nets: never merged), ``ground_layers`` /
                ``conductor_layers`` (lists of [layer, datatype]).
            out_dir: a new (or empty) folder for the shards.
            prior: ``optimized`` (the prior of the released models). threads: solver threads per design.
            on_error: ``skip`` (default: a design failing QC is listed in ``refused``) or ``raise``.
        """
        return await asyncio.to_thread(build_training_shards, designs, out_dir, prior, threads, on_error)

    @mcp.tool()
    async def plan_capacitance_training(
        mode: str,
        out_dir: str,
        script_dir: str,
        shard_dir: str | None = None,
        designs_json: str | None = None,
        base_shards: str | None = None,
        seeds: list[int] | None = None,
        steps: int = 18000,
        threads: int = 2,
        device: str = "cpu",
        k_folds: int = 5,
        family: str = "user",
        partition: str | None = None,
        account: str | None = None,
        time: str = "01:00:00",
        mem: str = "6G",
        gpus: int = 0,
        setup: str = "",
        python: str | None = None,
        extra_sbatch: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write the scripts to retrain the capacitance networks on your own compute (laptop, GPU box or SLURM cluster).

        Nothing is run: the result gives the files, the local command and the ``sbatch`` command.

        Modes:
            - ``build``: ``designs_json`` (a JSON list of designs, as in ``build_capacitance_training_shards``) ->
              shards in ``shard_dir``.
            - ``fit``: the paper recipe on ``shard_dir`` (its ``train`` rows; validated on its ``val`` rows), one seed per
              SLURM array task (``squadds.ml.capacitance.slurm``) or ``squadds-ml train`` locally.
            - ``in_family``: ``fit_in_family`` on your k designs (``designs_json`` or ``shard_dir``): k-fold CV of a
              from-scratch fit against the zero-shot QCap; zero-shot wins ties.
            - ``collective``: ``retrain_collective``: your designs added to the all-data training set (``base_shards``),
              ``seeds`` networks (the QCap recipe).

        Args:
            out_dir: where runs go. script_dir: where the scripts are written.
            seeds: default [0, 1, 2] ([0] for in_family). steps: 18,000 = the released recipe.
            threads / device (``cpu``, ``cuda``, ``mps``): per job. partition / account / time / mem / gpus /
                extra_sbatch: your cluster's values (left out when empty). setup: shell lines before training (e.g.
                ``module load ...; source venv/bin/activate``). python: interpreter (default: the server's).
        """
        return await asyncio.to_thread(
            plan_training,
            mode,
            out_dir,
            script_dir,
            shard_dir,
            designs_json,
            base_shards,
            seeds,
            steps,
            threads,
            device,
            k_folds,
            family,
            partition,
            account,
            time,
            mem,
            gpus,
            setup,
            python,
            extra_sbatch,
        )

    @mcp.tool()
    async def run_capacitance_fit_smoke(
        shard_dir: str,
        out_dir: str,
        steps: int = 200,
        seed: int = 0,
        threads: int = 2,
        arm: str = "full_gnn",
    ) -> dict[str, Any]:
        """Run a short in-process fit (at most 500 steps) on a shard dir to check the training pipeline end to end.

        Trains on the shard dir's ``train`` rows and scores its ``val`` rows (if any); writes ``out_dir/model.pt``. The
        result is a pipeline check, not a usable model: full fits (18,000 steps per seed) go through
        ``plan_capacitance_training``.
        """
        return await asyncio.to_thread(fit_smoke, shard_dir, out_dir, steps, seed, threads, arm)

    @mcp.tool()
    async def score_capacitance_model(
        model: str = "QCap",
        data: str | None = None,
        layouts: list[dict[str, Any]] | None = None,
        split: str = "val",
        threads: int = 2,
        families: list[str] | None = None,
    ) -> dict[str, Any]:
        """Score a capacitance model with the paper's metric: layout-mean median % over mutual entries, MAE and RMSE in fF,
        per entry kind and per family, with wall time, machine and threads.

        Args:
            model: ``QCap`` (default) / ``QCap-Fast`` (both trained on all data: a score on the dataset's rows is in-sample,
                not held out; on your own labelled layouts it is a zero-shot score), or a run dir / ``model.pt`` path of your
                own fit.
            data: a shard dir (fast: stored embedding and priors; ``split`` rows, ``families`` filter), or
            layouts: up to 20 ``{"gds", "labels"}`` items (``labels``: a ``cap_matrix.csv`` path, or a matrix in fF with
                ``names``), predicted end to end.
            split: ``val`` (default) or ``train``. Test scoring is maintainer-only and refused here.
        """
        return await asyncio.to_thread(score_model, model, data, layouts, split, threads, families)


__all__ = ["register_capacitance_training_tools"]
