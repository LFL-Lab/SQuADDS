"""
Planar capacitance ML tools.
============================

GDS layout -> full N x N Maxwell capacitance matrix (fF, ground first; nets are never merged) with the SQuADDS planar
capacitance models, plus the ``SQuADDS/SQuADDS_PlanarCapacitance`` dataset and its contribution validator.

Every tool degrades to a clear message when the optional ML extra is missing (``pip install "squadds[ml]"``): the heavy
imports (torch, gdstk, numba) happen inside the tools, never at module import.

Tools:
    - ``list_capacitance_models``             released models (QCap, QCap-Fast) with their reference accuracy and time
    - ``predict_capacitance_from_gds``        GDS path -> matrix, names, checks, timing (model ``QCap`` by default)
    - ``query_planar_capacitance``            dataset designs filtered by family / conductor count / split
    - ``get_planar_capacitance_conventions``  units, signs, ground first, layer map, stack, models and the QCap-Fast warning
    - ``validate_capacitance_contribution``   the dataset validator on a local contribution folder
    - ``compute_capacitance_prior``           the physics prior alone (one coarse solve, or the extrapolated optimized prior), no network
    - ``inspect_capacitance_layout``          the conductors and the ground a prediction would use (no prediction; PNG)
    - ``get_planar_capacitance_design``       one dataset design: options, Q3D QC and convergence, label, stored priors, GDS
    - ``nearest_planar_capacitance_designs``  the dataset designs geometrically closest to a design

Design (component -> GDS -> Hamiltonian), training / scoring and contribution tools live in ``capacitance_design.py``,
``capacitance_training.py`` and ``capacitance_contribution.py``.
"""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from squadds_mcp.utils import dataframe_to_records, sanitize_for_json

INSTALL_HINT = "pip install 'squadds[ml]'"
_PREDICTORS: dict = {}


def _ml_missing() -> dict[str, Any] | None:
    """An error payload when the ``[ml]`` extra is not installed, else None."""
    from squadds.ml import missing_ml_dependencies

    missing = missing_ml_dependencies()
    if missing:
        return {
            "error": f"the planar capacitance models need the optional ML dependencies (missing: {', '.join(missing)})",
            "install": INSTALL_HINT,
        }
    return None


def _layers(s: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in x.split("/")) for x in str(s).split(",") if x.strip()]


def _pieces(s: str) -> list[int] | None:
    """'1,4' -> [1, 4]; empty -> None (automatic ground detection)."""
    return [int(x) for x in str(s or "").split(",") if x.strip()] or None


def conventions() -> dict[str, Any]:
    """The planar capacitance conventions (also served by ``get_planar_capacitance_conventions``)."""
    from squadds.datasets import planar_capacitance_schema as S

    models = {}
    try:
        from squadds.ml.capacitance.models import list_models

        for row in list_models():
            models[row["model"]] = {
                "title": row["title"],
                "default": row["default"],
                "warning": row["warning"],
            }
    except Exception as e:  # noqa: BLE001
        models = {"error": f"model manifest not readable: {e}"}
    return {
        "matrix": (
            "full N x N Maxwell (short-circuit) capacitance matrix over every conductor of the layout, ground included; "
            "N is inferred from the layout and nets are never merged"
        ),
        "units": {"capacitance": "fF", "length": "um", "time": "s"},
        "signs": "diagonal > 0 (self capacitance), off-diagonal < 0 (= -mutual); mutual_fF = -C_ij",
        "ground": "net G is always row/column 0; signal conductors C1..Cn follow in GDS file order",
        "layer_map": {
            "ground": [[1, 0]],
            "conductor": [[1, 10]],
            "note": "configurable (ground_layers / conductor_layers); other layers such as port markers are ignored",
            "single_layer": "ground and conductors on one layer: pass that layer as both ground_layers and "
            "conductor_layers; the metal is split into numbered pieces and the ground is detected automatically "
            "(ground_detection in the result, with a warning); inspect_capacitance_layout shows the pieces before a "
            "prediction, and ground_pieces (e.g. '1' or '1,4') overrides the detected ground",
        },
        "stack": {
            "substrate": "Si (cryogenic)",
            "eps_r": 11.45,
            "thickness_um": 350.0,
            "metal": "zero-thickness sheet",
            "note": "training stack. Other stacks (eps_r, thickness_um): QCap solves the physics prior at the "
            "requested stack and correct it with the unchanged networks (calibrated route; on 15 held-out designs 0.21-1.02 % "
            "(MAE 0.07-0.33, RMSE 0.09-0.45 fF) on every tested stack vs 0.23 % (0.080 / 0.111 fF) on the training stack; self "
            "terms up to 1.22 % (0.79 / 0.86 fF)); QCap-Fast cannot solve a prior and rescales its training-stack prediction by "
            "(1 + eps_r) / (1 + 11.45), with a warning (measured off the stack: 6.36 % (MAE 1.74, RMSE 2.39 fF); use QCap "
            "there). Tested range: Si at 10-1000 um, or eps_r 3.8-24 at 350 um; other "
            "combinations warn (untested). result.substrate records the route and the stack the prior was solved at",
            "tested_range": {"si_thickness_um": [10.0, 1000.0], "eps_r_at_350_um": [3.8, 24.0]},
        },
        "models": models,
        "model_guide": {
            "QCap": "default: two thin-film method-of-moments solves (1.5 and 3 panels per gap, Richardson-"
            "extrapolated) corrected by residual graph networks; about 1 s for a small component",
            "QCap-Fast": "no field solve (learned prior surrogate + networks); quick screening of trained families only. "
            "WARNING: on new designs of trained families (B5) it was worse than the physics prior alone, 3.175 % (MAE "
            "0.583 / RMSE 0.824 fF) against 1.765 % (MAE 0.275 / RMSE 0.376 fF); the paper's version of it was also "
            "worse on a held-out family, 3.11 % (MAE 0.533 / RMSE 0.633 fF) against 1.55 % (MAE 0.166 / RMSE 0.205 fF)",
        },
        "priors": {
            "physics_prior": f"one solve at {S.COARSE_PPG} panels per gap ({S.COARSE_PRIOR_VERSION})",
            "optimized_physics_prior": f"C^R = R(1.5, 3) on the corrected stack ({S.OPTIMIZED_PRIOR_VERSION})",
        },
        "dataset": {
            "repo_id": S.REPO_ID,
            "design_id": "ref/<collection>/<design> (community/<org>-<batch>/<design> for contributions)",
            "geom_hash": "sha256 of the canonical polygon multiset (1 nm): the duplicate key across SQuADDS",
            "splits": list(S.SPLITS),
            "test_labels": "held out (private repo); never used for fitting, tuning or selection",
        },
        "metric": "layout-mean median relative error over mutual entries (0.1 fF floor), always with MAE and RMSE in fF "
        "and the wall time with machine and thread count",
    }


def _predictor(model: str, threads: int | None):
    from squadds.ml.capacitance import CapacitancePredictor, load_model

    key = (model, threads)
    if key not in _PREDICTORS:
        _PREDICTORS[key] = CapacitancePredictor(load_model(model), threads=threads)
    return _PREDICTORS[key]


def predict_gds(
    gds_path: str,
    model: str = "QCap",
    threads: int | None = None,
    ground_layers: str = "1/0",
    conductor_layers: str = "1/10",
    eps_r: float = 11.45,
    thickness_um: float = 350.0,
    ground_pieces: str = "",
) -> dict[str, Any]:
    """Synchronous body of ``predict_capacitance_from_gds``."""
    err = _ml_missing()
    if err:
        return err
    p = Path(gds_path).expanduser()
    if not p.is_file():
        return {"error": f"GDS file not found: {gds_path}"}
    try:
        pred = _predictor(model, threads)
        res = pred.predict(
            str(p),
            ground_layers=_layers(ground_layers),
            conductor_layers=_layers(conductor_layers),
            ground_pieces=_pieces(ground_pieces),
            threads=threads,
            eps_r=float(eps_r),
            thickness_um=float(thickness_um),
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return sanitize_for_json(
        {
            "names": list(res.names),
            "matrix_fF": res.matrix_fF,
            "units": "fF",
            "convention": res.convention,
            "model": res.model,
            "title": res.title,
            "checks": res.checks,
            "warnings": res.warnings,
            "timing_s": res.timing_s,
            "threads": res.threads,
            "machine": res.machine,
            "prior_version": res.prior_version,
            "substrate": res.substrate,
            "ground_detection": res.ground_detection,
            "sim_results": res.to_sim_results(),
        }
    )


def _dataset(revision: str | None = None, local_dir: str | None = None):
    from squadds.ml.capacitance.dataset import PlanarCapacitanceDB

    return PlanarCapacitanceDB(revision=revision, local_dir=local_dir)


def _dataset_error(e: Exception) -> dict[str, Any]:
    return {
        "error": f"dataset not readable ({type(e).__name__}: {e})",
        "hint": "check the network connection and the Hugging Face login, or set SQUADDS_PLANAR_CAP_DATA to a local copy",
    }


def _matrix(df) -> dict[str, Any]:
    return {"names": list(df.index), "matrix_fF": df.to_numpy().tolist()}


def ground_area_in_file(path, ground_layers) -> float:
    """Summed area (um^2) of the top-level polygons on the ground layers of a GDS, as written in the file."""
    import gdstk

    lib = gdstk.read_gds(str(path))
    scale = (lib.unit / 1e-6) ** 2
    layers = {tuple(x) for x in ground_layers}
    return scale * sum(
        q.area() for cell in lib.top_level() for q in cell.get_polygons() if (q.layer, q.datatype) in layers
    )


def inspect_gds(
    gds_path: str,
    ground_layers: str = "1/0",
    conductor_layers: str = "1/10",
    ground_pieces: str = "",
    png_path: str | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``inspect_capacitance_layout``."""
    err = _ml_missing()
    if err:
        return err
    p = Path(gds_path).expanduser()
    if not p.is_file():
        return {"error": f"GDS file not found: {gds_path}"}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from squadds.ml.capacitance import inspect_layout

        g, c = _layers(ground_layers), _layers(conductor_layers)
        info = inspect_layout(str(p), ground_layers=g, conductor_layers=c, ground_pieces=_pieces(ground_pieces))
        fig = info["figure"]
        png = None
        if png_path:
            png = Path(png_path).expanduser()
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        lay = info["prepared"]
        out: dict[str, Any] = {
            "names": ["G", *lay.names],
            "n_conductors": len(lay.names) + 1,
            "single_layer": bool(info["detected"]),
            "pieces": info["pieces"],
            "ground_pieces": info["ground_pieces"],
            "reason": info["reason"],
            "confidence": info["confidence"],
            "warnings": info["warnings"],
            "ground_area_um2": float(lay.ground.area),
            "conductor_areas_um2": {n: float(b.area) for n, b in zip(lay.names, lay.conductors)},
            "png_path": str(png.resolve()) if png else None,
        }
        if set(g).isdisjoint(c):
            out["ground_area_in_file_um2"] = ground_area_in_file(p, g)
            out["ground_area_note"] = (
                "the two areas should agree; a smaller area read by the model means part of the ground was lost "
                "(e.g. a keyhole cut along an edge of an exactly mirror-symmetric layout)"
            )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out["next"] = (
        "predict_capacitance_from_gds with the same layers; if the detected ground is wrong, pass ground_pieces with the "
        "piece numbers of the figure"
    )
    return sanitize_for_json(out)


DESIGN_PARTS = (
    "params",
    "simulations",
    "label",
    "label_passes",
    "optimized_physics_prior",
    "physics_prior",
    "gds_path",
)
DEFAULT_DESIGN_PARTS = ("params", "simulations", "label", "optimized_physics_prior")


def design_record(
    design_id: str,
    parts: list[str] | None = None,
    local_dir: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``get_planar_capacitance_design``."""
    parts = list(parts or DEFAULT_DESIGN_PARTS)
    bad = [x for x in parts if x not in DESIGN_PARTS]
    if bad:
        return {"error": f"unknown parts {bad}; choose from {list(DESIGN_PARTS)}"}
    try:
        db = _dataset(revision, local_dir)
        row = db.designs(design_ids=[design_id])
    except Exception as e:  # noqa: BLE001
        return _dataset_error(e)
    if row.empty:
        return {"error": f"unknown design_id {design_id!r}"}
    out: dict[str, Any] = {"design": dataframe_to_records(row, limit=1)[0]}
    errors = {}
    for part in parts:
        try:
            if part == "params":
                out["params"] = db.params([design_id]).get(design_id, {})
            elif part in ("simulations", "label_passes"):
                df = getattr(db, part)([design_id])
                out[part] = dataframe_to_records(df, limit=max(1, len(df)))
            elif part == "label":
                lab = db.labels([design_id]).get(design_id)
                out["label"] = _matrix(lab) if lab is not None else None
                if lab is None:
                    out["label_note"] = "no public label (test-split labels are held out)"
            elif part == "gds_path":
                out["gds_path"] = str(db.gds_path(design_id))
            else:
                m = getattr(db, part)([design_id]).get(design_id)
                out[part] = _matrix(m) if m is not None else None
        except Exception as e:  # noqa: BLE001
            errors[part] = f"{type(e).__name__}: {e}"
    if errors:
        out["errors"] = errors
    return sanitize_for_json(out)


def nearest_designs(
    design_id: str,
    k: int = 5,
    family: str | None = None,
    split: str | None = None,
    local_dir: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``nearest_planar_capacitance_designs``."""
    k = max(1, min(int(k), 50))
    try:
        df = _dataset(revision, local_dir).nearest(design_id, k=k, family=family, split=split)
    except (KeyError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001
        return _dataset_error(e)
    return sanitize_for_json(
        {
            "query": design_id,
            "rows": dataframe_to_records(df, limit=len(df)),
            "note": "geometric similarity in the pooled embedding (standardized Euclidean), not a capacitance distance; "
            "candidates default to the query's model_family",
        }
    )


def query_dataset(
    family: str | None = None,
    n_conductors: int | None = None,
    split: str | None = None,
    collection: str | None = None,
    config: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_labels: bool = False,
    local_dir: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``query_planar_capacitance``."""
    try:
        db = _dataset(revision, local_dir)
        df = db.designs(family=family, split=split, n_conductors=n_conductors, collection=collection, config=config)
    except Exception as e:  # noqa: BLE001
        return _dataset_error(e)
    limit = max(1, min(int(limit), 500))
    out: dict[str, Any] = {
        "squadds_db": "the families with a Qiskit Metal component are also SQuADDS_DB configs: "
        "SQuADDS_DB() (the planar_capacitance source is on by default), then select_* / Analyzer as usual",
        "total": int(len(df)),
        "offset": int(offset),
        "limit": limit,
        "columns": list(df.columns),
        "rows": dataframe_to_records(df, limit=limit, offset=offset),
    }
    if include_labels and len(df):
        ids = df.design_id.iloc[offset : offset + limit].tolist()
        try:
            labs = db.labels(ids)
            out["labels_fF"] = {k: _matrix(v) for k, v in labs.items()}
            out["labels_note"] = "test-split designs have no public label"
        except Exception as e:  # noqa: BLE001
            out["labels_error"] = f"{type(e).__name__}: {e}"
    return sanitize_for_json(out)


def validate_contribution(path: str, check_prior: bool = False, reference: str = "auto") -> dict[str, Any]:
    """Synchronous body of ``validate_capacitance_contribution``."""
    try:
        from squadds.ml.capacitance.contribute import validate_capacitance_contribution as V
    except ImportError as e:
        return {"error": f"{e}", "install": INSTALL_HINT}
    p = Path(path).expanduser()
    if not p.is_dir():
        return {"error": f"contribution folder not found: {path}"}
    if check_prior:
        err = _ml_missing()
        if err:
            return err
    try:
        rep = V(p, reference=None if reference == "none" else "auto", check_prior=check_prior)
    except ImportError as e:
        return {"error": f"{e}", "install": INSTALL_HINT}
    return sanitize_for_json(
        {
            "ok": rep.ok,
            "errors": rep.all_errors(),
            "warnings": rep.all_warnings(),
            "fixes": rep.fixes,
            "summary": {k: v for k, v in rep.info.items() if not isinstance(v, (list, dict))},
            "report_markdown": rep.to_markdown("Validation report"),
        }
    )


def physics_prior(
    gds_path: str,
    prior: str = "optimized",
    threads: int | None = None,
    ground_layers: str = "1/0",
    conductor_layers: str = "1/10",
    eps_r: float = 11.45,
    thickness_um: float = 350.0,
    ground_pieces: str = "",
) -> dict[str, Any]:
    """Synchronous body of ``compute_capacitance_prior``."""
    err = _ml_missing()
    if err:
        return err
    p = Path(gds_path).expanduser()
    if not p.is_file():
        return {"error": f"GDS file not found: {gds_path}"}
    if prior not in ("optimized", "coarse"):
        return {"error": "prior must be 'optimized' or 'coarse'"}
    try:
        from squadds.ml.capacitance import optimized_physics_prior, prepare_layout
        from squadds.ml.capacitance import physics_prior as one_solve_prior

        fn = optimized_physics_prior if prior == "optimized" else one_solve_prior
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lay = prepare_layout(
                str(p),
                ground_layers=_layers(ground_layers),
                conductor_layers=_layers(conductor_layers),
                ground_pieces=_pieces(ground_pieces),
            )
            res = fn(lay, threads=threads, eps_r=float(eps_r), thickness_um=float(thickness_um))
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    d = res.to_dict()
    warns = [w for w in lay.warnings if "beyond the validated range" not in w]
    warns = list(dict.fromkeys(warns + [str(w.message) for w in caught if issubclass(w.category, UserWarning)]))
    return sanitize_for_json(
        {
            "names": d["names"],
            "matrix_fF": d["C_fF"],
            "solves_fF": d["solves_fF"],
            "units": "fF",
            "kind": d["kind"],
            "prior_version": d["prior_version"],
            "levels_ppg": d["levels_ppg"],
            "stack": d["stack"],
            "ground_detection": lay.info.get("ground_detection"),
            "warnings": warns,
            "timing_s": d["timing_s"],
            "threads": d["threads"],
            "machine": d["machine"],
            "note": "physics only (no learned parameter); macOS priors differ from the released x86 ones by up to "
            "~2.4e-3 relative",
        }
    )


def register_capacitance_ml_tools(mcp: FastMCP) -> None:
    """Register the planar capacitance ML tools on the MCP server."""

    @mcp.tool()
    async def list_capacitance_models() -> dict[str, Any]:
        """List the released planar capacitance models (GDS -> full Maxwell capacitance matrix): ``QCap``
        and ``QCap-Fast``.

        Each row gives the model, whether it is the default (``QCap``), the Hugging Face repository and pinned revision,
        the warning (the ``QCap-Fast`` model is for quick screening of trained families only), the reference accuracy (median %
        with MAE and RMSE in fF): ``own`` = the model's in-sample fit, ``b5`` = its one never-seen check, ``reference`` = the
        held-out numbers of the paper's version of the model (TRAIN split only; not the released model's own, quote them as
        the paper's version), and the time with machine and threads.
        """
        from squadds.ml.capacitance.models import list_models

        out: dict[str, Any] = {"models": sanitize_for_json(list_models())}
        err = _ml_missing()
        if err:
            out["note"] = f"listing only: running a model needs {INSTALL_HINT} ({err['error']})"
        return out

    @mcp.tool()
    async def predict_capacitance_from_gds(
        gds_path: str,
        model: str = "QCap",
        threads: int | None = None,
        ground_layers: str = "1/0",
        conductor_layers: str = "1/10",
        eps_r: float = 11.45,
        thickness_um: float = 350.0,
        ground_pieces: str = "",
    ) -> dict[str, Any]:
        """Predict the full N x N Maxwell capacitance matrix (fF, ground first) of a GDS layout on this machine.

        Args:
            gds_path: path of a GDS file readable by the server process.
            model: ``QCap`` (default; trained on all data, for design work, ~1 s for a small component), or ``QCap-Fast`` (no
                field solve; screening of trained families only, worse than physics alone on new designs).
            threads: CPU threads (results are deterministic for a given machine and thread count).
            ground_layers: comma-separated ``layer/datatype`` pairs of the ground metal, e.g. ``1/0``.
            conductor_layers: comma-separated ``layer/datatype`` pairs of the conductors, e.g. ``1/10``.
            eps_r: substrate permittivity (default Si 11.45, the training stack).
            thickness_um: substrate thickness in um (default 350, the training stack). Off the training stack ``QCap``
                solves the physics prior at this stack (calibrated route: 0.21-1.02 % (MAE 0.07-0.33, RMSE
                0.09-0.45 fF) on 15 held-out designs at every tested stack, vs 0.23 % (MAE 0.080 / RMSE 0.111 fF) on the
                training stack; a bounded error,
                not training-stack accuracy); ``QCap-Fast`` rescales by (1 + eps_r) / (1 + 11.45) and warns (use ``QCap``).
                Tested: Si at 10-1000 um, or eps_r 3.8-24 at 350 um; other combinations warn.
            ground_pieces: only when ground and conductors share one layer (pass the same ``ground_layers`` and
                ``conductor_layers``, or a single-layer file): the metal is split into numbered pieces and the ground is
                detected automatically (``ground_detection`` in the result: pieces, reason, confidence, plus a warning).
                If the detected ground is wrong, resubmit with the piece numbers of the ground, e.g. ``"1"`` or ``"1,4"``
                (``inspect_capacitance_layout`` shows the numbered pieces without predicting).

        Returns names (G first), ``matrix_fF``, physical checks (symmetric, diagonal positive, off-diagonal negative,
        closure, positive-definite signal block; ``checks.passed``), warnings, timing with machine and threads, the
        substrate route (``substrate``: the stack the prior was solved at), ``ground_detection`` and the SQuADDS
        ``sim_results`` view of the matrix. A failed check puts "physical check failed ... do not use it" first in
        ``warnings``: do not use that matrix; a model that solves the physics (``QCap``) is the safer choice.
        """
        return await asyncio.to_thread(
            predict_gds,
            gds_path,
            model,
            threads,
            ground_layers,
            conductor_layers,
            eps_r,
            thickness_um,
            ground_pieces,
        )

    @mcp.tool()
    async def query_planar_capacitance(
        family: str | None = None,
        n_conductors: int | None = None,
        split: str | None = None,
        collection: str | None = None,
        config: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_labels: bool = False,
    ) -> dict[str, Any]:
        """Query designs of the ``SQuADDS/SQuADDS_PlanarCapacitance`` dataset.

        Reads with ``squadds.ml.capacitance.dataset.PlanarCapacitanceDB`` (no ML extra needed). The families that have a
        Qiskit Metal component (interdigital, paddle, concentric, star, Xmon, TransmonPocket, shielded concentric) are also
        SQuADDS_DB configs: the ``planar_capacitance`` source of ``SQuADDS_DB()`` is on by default.

        Args:
            family: e.g. ``interdigital``, ``concentric``, ``xmon`` (matches family, paper_family or model_family).
            n_conductors: conductor count including ground.
            split: ``train``, ``val``, ``test``, ``excluded`` or ``none`` (test designs carry no public label).
            collection: e.g. ``exp6``, ``atlas``, ``twonet``.
            config: SQuADDS config name, e.g. ``coupler-GeneralizedCapStar-cap_matrix``.
            limit: rows to return (max 500). offset: first row.
            include_labels: also return the reference Maxwell matrices (fF) of the returned designs.
        """
        return await asyncio.to_thread(
            query_dataset, family, n_conductors, split, collection, config, limit, offset, include_labels
        )

    @mcp.tool()
    async def get_planar_capacitance_conventions() -> dict[str, Any]:
        """Conventions of the planar capacitance models and dataset: units (fF), Maxwell signs, ground first, never merging
        nets, the GDS layer map, the served substrate stack, what the models are (QCap default; the QCap-Fast warning), the
        physics priors and the accuracy metric."""
        return conventions()

    @mcp.tool()
    async def validate_capacitance_contribution(
        path: str, check_prior: bool = False, reference: str = "auto"
    ) -> dict[str, Any]:
        """Validate a local planar capacitance contribution folder (template: manifest.yaml, designs.csv, gds/, sims/).

        Runs the dataset validator: schema, GDS readback, conductor count vs matrix size (nets never merged), net names,
        label QC (symmetry, signs, closure, positive definiteness, convergence, units), duplicates (a duplicate of a held-out
        test design is rejected), stack, licence (CC-BY-4.0 / CC0-1.0) and privacy. Read-only: nothing is uploaded.

        Args:
            path: the contribution folder.
            check_prior: also compare each label with the optimized physics prior (needs the ML extra; seconds per design).
            reference: ``auto`` (duplicate check against the dataset when reachable) or ``none``.
        """
        return await asyncio.to_thread(validate_contribution, path, check_prior, reference)

    @mcp.tool()
    async def compute_capacitance_prior(
        gds_path: str,
        prior: str = "optimized",
        threads: int | None = None,
        ground_layers: str = "1/0",
        conductor_layers: str = "1/10",
        eps_r: float = 11.45,
        thickness_um: float = 350.0,
        ground_pieces: str = "",
    ) -> dict[str, Any]:
        """Physics-only Maxwell matrix of a GDS (fF, ground first): the thin-film method-of-moments prior, no network.

        ``optimized`` (default): C^R = R(1.5, 3), two solves Richardson-extrapolated, the prior QCap corrects.
        ``coarse``: the physics prior, one solve at 0.375 panels per gap (faster, less accurate). Use it as a baseline next to
        ``predict_capacitance_from_gds`` (the model = this prior + learned correction). ``eps_r`` / ``thickness_um``: the
        substrate the prior is solved at (default Si 11.45, 350 um). ``ground_pieces``: as in
        ``predict_capacitance_from_gds`` (ground and conductors on one layer). Needs the ``[ml]`` extra.

        Returns names, ``matrix_fF``, ``solves_fF`` (each thin-film solve), the stack (with the substrate and whether it is
        in the tested range), ``ground_detection``, ``warnings`` (layout reading, untested substrate), timing with machine
        and threads.
        """
        return await asyncio.to_thread(
            physics_prior, gds_path, prior, threads, ground_layers, conductor_layers, eps_r, thickness_um, ground_pieces
        )

    @mcp.tool()
    async def inspect_capacitance_layout(
        gds_path: str,
        ground_layers: str = "1/0",
        conductor_layers: str = "1/10",
        ground_pieces: str = "",
        png_path: str | None = None,
    ) -> dict[str, Any]:
        """Look at the metal before predicting: the conductors and the ground a prediction would use (no prediction).

        Returns the conductor names (``G`` first) and count, their areas, and the ground area read; for a two-layer file
        also the ground area summed in the file (they should agree: a smaller read area means part of the ground was
        lost). When ground and conductors share one layer (pass the same layer twice) the metal is split into numbered
        ``pieces`` and the detected ``ground_pieces`` come with a ``reason`` and ``confidence``; if the ground is wrong,
        pass ``ground_pieces`` (e.g. ``"1"`` or ``"1,4"``) here and to ``predict_capacitance_from_gds``.
        ``png_path``: also save the labelled figure (ground grey, conductors coloured, every piece numbered) as a PNG.
        Needs the ``[ml]`` extra (no model download).
        """
        return await asyncio.to_thread(inspect_gds, gds_path, ground_layers, conductor_layers, ground_pieces, png_path)

    @mcp.tool()
    async def get_planar_capacitance_design(design_id: str, parts: list[str] | None = None) -> dict[str, Any]:
        """One design of the ``SQuADDS/SQuADDS_PlanarCapacitance`` dataset in detail (``PlanarCapacitanceDB``; no ML extra).

        Args:
            design_id: e.g. from ``query_planar_capacitance`` or ``nearest_planar_capacitance_designs``.
            parts: any of ``params`` (generator / component options), ``simulations`` (the Q3D solve: solver, stack,
                passes, convergence, cores, solve time and the QC flags symmetric / signs / closure / PD / convergence),
                ``label`` (final Maxwell matrix, fF, ground first; test-split labels are held out), ``label_passes`` (every
                adaptive pass, for convergence), ``optimized_physics_prior`` / ``physics_prior`` (the stored priors
                as Maxwell matrices, fF), ``gds_path`` (downloads the solved GDS). Default: params, simulations, label,
                optimized_physics_prior.
        """
        return await asyncio.to_thread(design_record, design_id, parts)

    @mcp.tool()
    async def nearest_planar_capacitance_designs(
        design_id: str, k: int = 5, family: str | None = None, split: str | None = None
    ) -> dict[str, Any]:
        """The ``k`` dataset designs geometrically closest to ``design_id`` (pooled geometric embedding, standardized
        Euclidean distance; a similarity of shapes, not of capacitances). Candidates: ``family`` (default the query's
        model family) and ``split`` (e.g. ``train``). Use it to see whether a design sits inside the training data."""
        return await asyncio.to_thread(nearest_designs, design_id, k, family, split)
