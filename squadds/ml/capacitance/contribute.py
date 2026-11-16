"""Contributing planar capacitance data and models to ``SQuADDS/SQuADDS_PlanarCapacitance`` (Hugging Face pull requests).

Data: a contribution directory in the template layout (``squadds/database/templates/planar_capacitance_contribution``)::

    manifest.yaml       contributor, licence, solver, stack, setup, layer map, units, generator, family
    designs.csv         design, gds, family, params_json, net_order (G first, then conductors in GDS file order), notes
    gds/<design>.gds
    sims/<design>/<sim_tag>/cap_matrix.csv    Q3D export as is (one row per pass)   or   matrix.csv / matrix.parquet
    sims/<design>/<sim_tag>/convergence.txt   the Q3D .conv file (optional)
    sims/<design>/<sim_tag>/run.json          solver version, cores, solve_s, wall_s, peak_mem_GB, machine (optional)
    nets/<design>.json                        {net: [conductor-layer polygon indices]} (optional explicit net map)

``validate_capacitance_contribution(dir)`` runs the dataset validator (schema, GDS readback, net names, label QC, duplicates,
stack, licence, privacy) and ``contribute_capacitance(dir)`` opens a pull request adding ``contrib/<org>-<batch>/`` and its
``validation_report.md``. Maintainers merge, compute the priors on HPC and ingest the rows as ``source="community"``,
``split="none"``.

Models: ``contribute_model(repo_id, revision)`` validates a model repository (checkpoints load, sha256, required metadata,
validation score with median % and MAE / RMSE in fF plus time, machine and threads) and opens a pull request adding its entry
to ``model_zoo.yaml``. Test-set scores are never self-reported: they are run by the maintainers, once per declared model, only
for models trained on the TRAIN split.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from squadds.datasets.planar_capacitance_schema import REPO_ID

from . import qc as Q

LICENCES = {"cc-by-4.0": "CC-BY-4.0", "cc0-1.0": "CC0-1.0", "cc0": "CC0-1.0", "cc-by 4.0": "CC-BY-4.0"}
MANIFEST_REQUIRED = {
    "contributor": dict,
    "licence": str,
    "solver": dict,
    "stack": dict,
    "setup": dict,
    "layer_map": dict,
    "ground_net_name": str,
    "units": dict,
    "generator": dict,
    "family": str,
    "description": str,
}
DESIGN_COLUMNS = ("design", "gds", "family", "params_json", "net_order", "notes")
MATRIX_FILES = ("cap_matrix.csv", "matrix.csv", "matrix.parquet")
CONV_FILES = ("convergence.txt",)
UPLOAD_DIRS = ("gds", "sims", "nets")
ZOO_FILE = "model_zoo.yaml"
MODEL_META_FILES = ("squadds_model.yaml", "squadds_model.json")
MODEL_REQUIRED = ("name", "title", "licence", "training_data", "val_score", "contributor")
VAL_REQUIRED = ("median_pct", "mae_fF", "rmse_fF", "layouts", "machine", "threads")
VAL_TIME = ("seconds", "seconds_per_layout")
_ABS = re.compile(r"^(/|~/|[A-Za-z]:[\\/]|\\\\)")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(s).lower()).strip("-.")


def _yaml():
    try:
        import yaml
    except ImportError as e:
        raise ImportError("contributions need pyyaml: pip install pyyaml (or 'squadds[ml]')") from e
    return yaml


def _api(api=None, token=None):
    if api is not None:
        return api
    from huggingface_hub import HfApi

    from squadds.database.contributor_env import load_contributor_environment

    load_contributor_environment()
    return HfApi(token=token or os.getenv("HUGGINGFACE_API_KEY") or None)


def submitter_record() -> dict:
    """The contributor block of the repo-root ``.env`` (GROUP_NAME, PI_NAME, INSTITUTION, USER_NAME, CONTRIB_MISC)."""
    from squadds.database.contributor_env import build_contributor_record

    return build_contributor_record()


# ---------------------------------------------------------------------------------------------------------------- privacy
def _scrub(o, path: str, fixes: list):
    """Absolute local paths -> file names (recorded in ``fixes``)."""
    if isinstance(o, dict):
        return {k: _scrub(v, f"{path}.{k}", fixes) for k, v in o.items()}
    if isinstance(o, list):
        return [_scrub(v, f"{path}[{i}]", fixes) for i, v in enumerate(o)]
    if isinstance(o, str) and _ABS.match(o.strip()) and len(o.strip()) > 1:
        fixes.append(f"{path}: absolute path replaced by its file name")
        return re.split(r"[\\/]", o.strip().rstrip("/\\"))[-1]
    return o


def sanitized_manifest(manifest: dict) -> tuple[dict, list[str]]:
    """The manifest as it is uploaded: absolute paths stripped, the e-mail dropped unless ``contributor.email_public``."""
    fixes: list[str] = []
    m = _scrub(manifest, "manifest", fixes)
    c = dict(m.get("contributor") or {})
    if c.get("email") and not c.get("email_public"):
        c.pop("email")
        fixes.append("manifest.contributor.email removed (email_public is not true)")
    m["contributor"] = c
    return m, fixes


def _sanitized_designs_csv(rows: list[dict], fixes: list[str]) -> str:
    out = io.StringIO()
    cols = list(dict.fromkeys([*DESIGN_COLUMNS, *(k for r in rows for k in r)]))
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: _scrub(r.get(k, ""), f"designs.csv[{r.get('design')}].{k}", fixes) for k in cols})
    return out.getvalue()


# ---------------------------------------------------------------------------------------------------------------- reading
def load_manifest(path) -> dict:
    p = Path(path) / "manifest.yaml"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (copy the template manifest.yaml)")
    return _yaml().safe_load(p.read_text(encoding="utf-8")) or {}


def read_designs(path) -> list[dict]:
    p = Path(path) / "designs.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (copy the template designs.csv)")
    with open(p, newline="", encoding="utf-8") as f:
        return [
            {k.strip(): (v or "").strip() for k, v in r.items() if k}
            for r in csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        ]


def parse_net_order(s: str) -> list[str]:
    s = (s or "").strip()
    if s.startswith("["):
        return [str(x) for x in json.loads(s)]
    return [x.strip() for x in re.split(r"[;|,\s]+", s) if x.strip()]


def read_sim(sim_dir, unit: str = "fF") -> dict:
    """One simulation directory -> {"names", "C" (label, given unit), "passes", "unit", "convergence", "run"}."""
    d = Path(sim_dir)
    out: dict = {"convergence": None, "run": None}
    if (d / "cap_matrix.csv").exists():
        m = Q.read_cap_matrix(d / "cap_matrix.csv")
        complete = [(p, C) for p, C in m["passes"] if np.isfinite(C).all()]
        out.update(names=m["names"], passes=m["passes"], unit="fF", units_found=m["units"], freq_GHz=m["freq_GHz"])
        out["C"] = complete[-1][1] if complete else np.full((len(m["names"]),) * 2, np.nan)
    else:
        f = next((d / n for n in ("matrix.parquet", "matrix.csv") if (d / n).exists()), None)
        if f is None:
            raise FileNotFoundError(f"{d}: no {' / '.join(MATRIX_FILES)}")
        import pandas as pd

        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        val = "C_fF" if "C_fF" in df else "C"
        if val not in df or not {"net_a", "net_b"} <= set(df.columns):
            raise ValueError(f"{f}: needs columns net_a, net_b and C_fF (or C in the manifest unit)")
        u = "fF" if val == "C_fF" else unit
        order = list(dict.fromkeys([*map(str, df.net_a), *map(str, df.net_b)]))
        names = ([Q.GROUND] if Q.GROUND in order else []) + [x for x in order if x != Q.GROUND]
        ix = {n: i for i, n in enumerate(names)}

        def mat(sub):
            C = np.full((len(names), len(names)), np.nan)
            for a, b, v in zip(sub.net_a, sub.net_b, sub[val]):
                C[ix[str(a)], ix[str(b)]] = C[ix[str(b)], ix[str(a)]] = float(v)
            return C

        if "pass" in df:
            passes = [(int(p), mat(g)) for p, g in df.groupby("pass")]
            out.update(passes=passes, C=passes[-1][1])
        else:
            out.update(passes=None, C=mat(df))
        out.update(names=names, unit=u)
    conv = next((p for p in [*(d / n for n in CONV_FILES), *sorted(d.glob("*.conv"))] if p.exists()), None)
    if conv is not None:
        out["convergence"] = Q.read_conv(conv)
    if (d / "run.json").exists():
        out["run"] = json.loads((d / "run.json").read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------------------------------------------- validator
def _manifest_qc(m: dict) -> Q.QCReport:
    r = Q.QCReport("manifest")
    for k, t in MANIFEST_REQUIRED.items():
        if k not in m or m[k] in (None, "", {}):
            r.error(f"manifest field {k!r} missing")
        elif not isinstance(m[k], t):
            r.error(f"manifest field {k!r} must be a {t.__name__}")
    c = m.get("contributor") or {}
    if isinstance(c, dict) and not (c.get("name") and c.get("affiliation")):
        r.error("contributor needs name and affiliation")
    lic = LICENCES.get(str(m.get("licence", "")).strip().lower())
    if lic is None:
        r.error(f"licence {m.get('licence')!r} not accepted: data contributions must be CC-BY-4.0 or CC0-1.0")
    r.info["licence"] = lic
    solver = m.get("solver") or {}
    if isinstance(solver, dict) and not solver.get("name"):
        r.error("solver.name missing")
    units = m.get("units") or {}
    cap = str(units.get("capacitance", "")).strip() if isinstance(units, dict) else ""
    try:
        Q.to_fF(1.0, cap)
    except ValueError:
        r.error(f"units.capacitance {cap!r} not one of {sorted(Q.TO_FF)} (or 'pF*1e15')")
    lm = m.get("layer_map") or {}
    if isinstance(lm, dict) and not (lm.get("ground") and lm.get("conductor")):
        r.error("layer_map needs ground and conductor layer lists, e.g. {ground: [[1, 0]], conductor: [[1, 10]]}")
    return r


def validate_capacitance_contribution(
    path,
    reference="auto",
    check_prior: bool = False,
    threads: int = 2,
    report_path=None,
    revision: str | None = None,
) -> Q.QCReport:
    """Validate a contribution directory (``PLAN_DATASET_HF.md`` §4.2). Returns a ``QCReport`` with one child per design.

    reference: the dataset's designs (DataFrame with ``design_id``, ``geom_hash``, ``split``) for the duplicate check;
    ``"auto"`` reads them with ``PlanarCapacitanceDB`` (a warning when the dataset is not reachable), None skips it.
    check_prior: also compare every label with ``optimized_physics_prior`` (needs ``[ml]``; seconds per design; report only).
    report_path: also write the markdown report there."""
    root = Path(path)
    rep = Q.QCReport(f"contribution {root.name}")
    try:
        m = load_manifest(root)
        rows = read_designs(root)
    except (FileNotFoundError, ValueError) as e:
        rep.error(str(e))
        return _finish(rep, report_path)
    except Exception as e:  # noqa: BLE001  (YAML / CSV syntax)
        rep.error(f"cannot parse the manifest or designs.csv: {e}")
        return _finish(rep, report_path)
    rep.extend(_manifest_qc(m), "manifest")
    _, fixes = sanitized_manifest(m)
    rep.fixes += fixes
    _sanitized_designs_csv(rows, rep.fixes)
    st = m.get("stack") if isinstance(m.get("stack"), dict) else {}
    freq = (m.get("setup") or {}).get("freq_GHz") if isinstance(m.get("setup"), dict) else None
    rep.extend(Q.stack_qc(st, freq), "stack")
    if not rows:
        rep.error("designs.csv has no rows")
    missing_cols = [c for c in ("design", "gds", "net_order") if rows and c not in rows[0]]
    if missing_cols:
        rep.error(f"designs.csv misses columns {missing_cols} (template columns: {', '.join(DESIGN_COLUMNS)})")
        return _finish(rep, report_path)
    names = [r["design"] for r in rows]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        rep.error(f"design names repeated in designs.csv: {dup_names[:5]}")
    unit = str((m.get("units") or {}).get("capacitance", "fF"))
    layer_map = m.get("layer_map") or Q.DEFAULT_LAYERS
    gname = str(m.get("ground_net_name") or Q.GROUND)
    hashes = {}
    for row in rows:
        child = _validate_design(root, row, unit, layer_map, gname, check_prior, threads)
        rep.children.append(child)
        hashes[row["design"]] = child.info.get("geom_hash")
    if isinstance(reference, str) and reference == "auto":
        try:
            reference = Q.reference_hashes(revision=revision)
        except Exception as e:  # noqa: BLE001
            rep.warn(f"dataset not reachable for the duplicate check ({type(e).__name__}: {e}); maintainers rerun it")
            reference = None
    rep.extend(Q.duplicate_qc(hashes, reference), "duplicates")
    rep.info.update(
        designs=len(rows),
        designs_ok=sum(c.ok for c in rep.children),
        simulations=sum(c.info.get("simulations", 0) for c in rep.children),
        family=m.get("family"),
        licence=rep.info.get("licence"),
    )
    return _finish(rep, report_path)


def _finish(rep: Q.QCReport, report_path) -> Q.QCReport:
    if report_path is not None:
        Path(report_path).write_text(rep.to_markdown("Validation report"), encoding="utf-8")
    return rep


def _validate_design(root: Path, row: dict, unit, layer_map, gname, check_prior, threads) -> Q.QCReport:
    d = row["design"]
    r = Q.QCReport(d)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", d or ""):
        r.error(f"design name {d!r}: use letters, digits, '.', '_' or '-'")
        return r
    order = parse_net_order(row.get("net_order", ""))
    order = [Q.GROUND if x == gname else x for x in order]
    if not order:
        r.error("net_order is empty")
        return r
    gds = row.get("gds") or f"gds/{d}.gds"
    gp = root / gds if not _ABS.match(gds) else root / "gds" / Path(gds).name
    if not gp.exists():
        r.error(f"GDS {gds} not found")
        return r
    nets = None
    if (root / "nets" / f"{d}.json").exists():
        nets = json.loads((root / "nets" / f"{d}.json").read_text(encoding="utf-8"))
    g = Q.gds_qc(gp, layer_map, nets, subject="gds")
    r.extend(g, "gds")
    if "n_conductors" in g.info:
        r.extend(Q.conductor_count_qc(g.info["n_conductors"], order), "nets")
    if nets and [Q.GROUND, *nets] != order:
        r.error(f"net_order {order} differs from the nets map order {[Q.GROUND, *nets]}")
    sims = sorted(p for p in (root / "sims" / d).glob("*") if p.is_dir()) if (root / "sims" / d).exists() else []
    r.info["simulations"] = len(sims)
    if not sims and str(row.get("unlabelled", "")).lower() not in ("1", "true", "yes"):
        r.error(
            f"no simulation under sims/{d}/<sim_tag>/ (declare the design unlabelled=true to contribute geometry only)"
        )
    for s in sims:
        try:
            sim = read_sim(s, unit)
        except Exception as e:  # noqa: BLE001
            r.error(f"{s.name}: cannot read the matrix ({e})")
            continue
        sim_names = [Q.GROUND if x == gname else x for x in sim["names"]]
        lq = Q.label_qc(
            sim["C"],
            sim_names,
            net_order=order,
            unit=sim["unit"],
            passes=sim["passes"],
            convergence=sim["convergence"],
            flagged_zero=_flagged(row),
            subject=s.name,
        )
        mat = lq.info.pop("matrix_fF", None)
        r.extend(lq, f"sim {s.name}")
        if sim.get("run"):
            fx: list[str] = []
            _scrub(sim["run"], f"sims/{d}/{s.name}/run.json", fx)
            r.fixes += fx
        if check_prior and lq.ok and mat is not None:
            r.extend(Q.prior_sanity(mat, order, gp, layer_map, threads=threads), f"sim {s.name}")
    return r


def _flagged(row: dict) -> list[tuple[str, str]]:
    s = row.get("zero_entries", "")
    return [tuple(p.split("-", 1)) for p in re.split(r"[;\s]+", s) if "-" in p]


# ---------------------------------------------------------------------------------------------------------------- data PR
def _upload_files(root: Path) -> list[Path]:
    files = []
    for top in UPLOAD_DIRS:
        if (root / top).exists():
            files += [p for p in sorted((root / top).rglob("*")) if p.is_file() and not p.name.startswith(".")]
    return files


def contribute_capacitance(
    path,
    repo_id: str = REPO_ID,
    dry_run: bool = False,
    org: str | None = None,
    batch: str | None = None,
    reference="auto",
    check_prior: bool = False,
    api=None,
    token: str | None = None,
) -> dict:
    """Validate ``path`` and open a pull request on the dataset ``repo_id`` adding ``contrib/<org>-<batch>/`` (manifest,
    designs, GDS, simulations, ``validation_report.md``, ``submission.json`` with the ``.env`` contributor block).

    A contribution with errors is refused (``ValueError``). dry_run: validate and return the planned files without any
    network call. Returns ``{"ok", "pr_url", "folder", "files", "report"}``."""
    from huggingface_hub import CommitOperationAdd

    root = Path(path)
    rep = validate_capacitance_contribution(root, reference=reference, check_prior=check_prior)
    if not rep.ok:
        raise ValueError("the contribution failed validation:\n- " + "\n- ".join(rep.all_errors()[:20]))
    m = load_manifest(root)
    org = _slug(org or m.get("org") or (m.get("contributor") or {}).get("affiliation") or "community")
    batch = _slug(batch or m.get("batch") or time.strftime("%Y%m%d"))
    folder = f"contrib/{org}-{batch}"
    man, _ = sanitized_manifest(m)
    fx: list[str] = []
    designs_csv = _sanitized_designs_csv(read_designs(root), fx)
    try:
        sub = submitter_record()
    except Exception as e:  # noqa: BLE001
        sub = {"note": f".env contributor block not read: {e}"}
    md = rep.to_markdown(f"Validation report: {folder}")
    md += "\n## Submitted by (.env)\n\n" + "\n".join(f"- {k}: {v}" for k, v in sub.items() if v) + "\n"
    yaml = _yaml()
    blobs = {
        "manifest.yaml": yaml.safe_dump(man, sort_keys=False).encode(),
        "designs.csv": designs_csv.encode(),
        "validation_report.md": md.encode(),
        "submission.json": json.dumps(
            {"submitted_by": sub, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "validator": _version()}, indent=1
        ).encode(),
    }
    ops = [CommitOperationAdd(path_in_repo=f"{folder}/{k}", path_or_fileobj=v) for k, v in blobs.items()]
    for p in _upload_files(root):
        rel = p.relative_to(root).as_posix()
        if p.name == "run.json":
            fxr: list[str] = []
            data = json.dumps(_scrub(json.loads(p.read_text(encoding="utf-8")), rel, fxr), indent=1).encode()
            ops.append(CommitOperationAdd(path_in_repo=f"{folder}/{rel}", path_or_fileobj=data))
        else:
            ops.append(CommitOperationAdd(path_in_repo=f"{folder}/{rel}", path_or_fileobj=str(p)))
    files = [o.path_in_repo for o in ops]
    out = {"ok": True, "folder": folder, "files": files, "report": md, "pr_url": None, "repo_id": repo_id}
    if dry_run:
        return out
    info = _api(api, token).create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Contribution {org}-{batch}: {rep.info.get('designs')} designs ({m.get('family')})",
        commit_description=md[:60000],
        create_pr=True,
    )
    out["pr_url"] = getattr(info, "pr_url", None) or str(info)
    return out


def _version() -> str:
    try:
        from squadds import __version__

        return f"squadds {__version__}"
    except Exception:  # noqa: BLE001
        return "squadds"


# ---------------------------------------------------------------------------------------------------------------- models
def _model_meta(root: Path) -> dict:
    for n in MODEL_META_FILES:
        p = root / n
        if p.exists():
            return (
                _yaml().safe_load(p.read_text(encoding="utf-8"))
                if p.suffix == ".yaml"
                else json.loads(p.read_text(encoding="utf-8"))
            )
    return {}


def _load_contributed(root: Path, meta: dict, paths: list[Path]):
    base = meta.get("base") or {}
    from .models import is_model_name, load_model

    if is_model_name(base.get("name")):
        return load_model(base["name"], path=root)
    from .train import trained_model

    return trained_model(paths, name=_slug(meta.get("name", "contributed")) or "contributed", title=meta.get("title"))


def validate_model(
    model_repo_id: str,
    revision: str | None = None,
    local_dir=None,
    metadata: dict | None = None,
    val_data=None,
    threads: int = 2,
    api=None,
    token: str | None = None,
    loader=None,
) -> tuple[Q.QCReport, dict]:
    """Validate a model repository for the model zoo. Returns (report, zoo entry).

    The repository holds ``*.pt`` checkpoints (``squadds-ml train`` run dirs or ``model.pt`` files) and ``squadds_model.yaml``
    with ``name``, ``title``, ``licence``, ``contributor``, ``training_data`` (``sha256``, ``split_used``, ``description``) and
    ``val_score`` (``median_pct``, ``mae_fF``, ``rmse_fF``, ``layouts``, ``seconds`` or ``seconds_per_layout``, ``machine``,
    ``threads``); ``metadata``
    overrides the file. val_data: a validation shard dir: the score is then computed here with ``benchmark.score`` and replaces
    the self-reported one. local_dir: a local copy of the repository (no download)."""
    from .models import sha256_file

    rep = Q.QCReport(f"model {model_repo_id}@{revision or 'main'}")
    hf = None
    if local_dir is None:
        hf = _api(api, token)
        sha = hf.repo_info(model_repo_id, revision=revision).sha
        local_dir = hf.snapshot_download(repo_id=model_repo_id, revision=sha)
        revision = sha
    elif not (revision and re.fullmatch(r"[0-9a-f]{40}", str(revision))):
        rep.warn(f"revision {revision!r} is not a full commit sha: the zoo entry must pin one")
    root = Path(local_dir)
    meta = {**_model_meta(root), **(metadata or {})}
    for k in MODEL_REQUIRED:
        if not meta.get(k):
            rep.error(f"model metadata field {k!r} missing (squadds_model.yaml)")
    lic = str(meta.get("licence", "")).strip()
    if lic and not re.fullmatch(r"[A-Za-z0-9.+-]+", lic):
        rep.error(f"licence {lic!r} is not an SPDX identifier")
    td = meta.get("training_data") or {}
    if not td.get("sha256"):
        rep.error("training_data.sha256 missing (sha256 of the training shards / model_input used)")
    split_used = str(td.get("split_used", "")).lower()
    if not split_used:
        rep.error("training_data.split_used missing (train | train+val | all | custom)")
    test_eligible = split_used == "train"
    if "test" in split_used or split_used == "all":
        rep.warn(
            "trained on test layouts: the model is listed without any test score (not eligible for maintainer scoring)"
        )
    paths = sorted(p for p in root.rglob("*.pt") if not any(x.startswith(".") for x in p.relative_to(root).parts))
    if not paths:
        rep.error("no *.pt checkpoint in the repository")
    files = [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)} for p in paths]
    model = None
    if paths:
        try:
            model = (loader or _load_contributed)(root, meta, paths)
            rep.info["arm"] = (getattr(model, "entry", {}) or {}).get("arm")
        except Exception as e:  # noqa: BLE001
            rep.error(f"checkpoints do not load: {type(e).__name__}: {e}")
    val = dict(meta.get("val_score") or {})
    if val_data is not None and model is not None:
        from .benchmark import score

        s = score(model, val_data, split="val", threads=threads)
        h = s.headline()
        computed = {
            "median_pct": h["median_pct"],
            "mae_fF": h["mae_fF"],
            "rmse_fF": h["rmse_fF"],
            "layouts": h["layouts"],
            "seconds": s.timing.get("seconds"),
            "machine": s.timing.get("machine"),
            "threads": s.timing.get("threads"),
            "split": "val",
            "source": "computed by contribute_model",
        }
        if val and val.get("median_pct") is not None and abs(float(val["median_pct"]) - h["median_pct"]) > 1e-3:
            rep.warn(
                f"self-reported val median {val['median_pct']} % differs from the computed {h['median_pct']:.4f} %"
            )
        val = computed
    elif val:
        val.setdefault("source", "self-reported")
        rep.warn("validation score self-reported (pass val_data to compute it here)")
    if str(val.get("split", "val")).lower() == "test":
        rep.error("val_score is a test score: test scores are maintainer-run only, report the validation split")
    miss = [k for k in VAL_REQUIRED if val.get(k) in (None, "")]
    if all(val.get(k) in (None, "") for k in VAL_TIME):
        miss.append(" or ".join(VAL_TIME))
    if meta.get("val_score") is not None or val_data is not None:
        if miss:
            rep.error(f"val_score misses {miss} (median % with MAE and RMSE in fF, plus time, machine and threads)")
    rep.info.update(files=len(files), val_median_pct=val.get("median_pct"), val_mae_fF=val.get("mae_fF"))
    entry = {
        "id": _slug(meta.get("name", model_repo_id.split("/")[-1])),
        "repo_id": model_repo_id,
        "revision": revision,
        "title": meta.get("title"),
        "arm": rep.info.get("arm"),
        "base": meta.get("base"),
        "licence": lic or None,
        "contributor": meta.get("contributor"),
        "files": files,
        "training_data": td,
        "val_score": val,
        "test_eligible": test_eligible,
        "validated": {"date": time.strftime("%Y-%m-%d"), "by": _version()},
    }
    return rep, entry


def contribute_model(
    model_repo_id: str,
    revision: str | None = None,
    registry_repo: str = REPO_ID,
    metadata: dict | None = None,
    val_data=None,
    dry_run: bool = False,
    local_dir=None,
    threads: int = 2,
    api=None,
    token: str | None = None,
    loader=None,
) -> dict:
    """Validate a model repository (``validate_model``) and open a pull request on ``registry_repo`` adding its entry to
    ``model_zoo.yaml`` plus ``model_zoo/<id>/validation_report.md``. A model with errors is refused (``ValueError``).
    Returns ``{"ok", "pr_url", "entry", "report", "files"}``."""
    from huggingface_hub import CommitOperationAdd

    hf = None if (dry_run and local_dir is not None and api is None) else _api(api, token)
    rep, entry = validate_model(
        model_repo_id, revision, local_dir, metadata, val_data, threads, api=hf, token=token, loader=loader
    )
    if not rep.ok:
        raise ValueError("the model failed validation:\n- " + "\n- ".join(rep.all_errors()[:20]))
    zoo = {"version": 1, "models": []}
    if hf is not None:
        try:
            p = hf.hf_hub_download(repo_id=registry_repo, repo_type="dataset", filename=ZOO_FILE)
            zoo = _yaml().safe_load(Path(p).read_text(encoding="utf-8")) or zoo
        except Exception as e:  # noqa: BLE001
            if "404" not in str(e) and "EntryNotFound" not in type(e).__name__ and "not found" not in str(e).lower():
                raise
    zoo.setdefault("models", [])
    if any(x.get("repo_id") == entry["repo_id"] and x.get("revision") == entry["revision"] for x in zoo["models"]):
        raise ValueError(f"{entry['repo_id']}@{entry['revision']} is already in {ZOO_FILE}")
    zoo["models"].append(entry)
    md = rep.to_markdown(f"Model validation report: {model_repo_id}@{entry['revision']}")
    ops = [
        CommitOperationAdd(path_in_repo=ZOO_FILE, path_or_fileobj=_yaml().safe_dump(zoo, sort_keys=False).encode()),
        CommitOperationAdd(path_in_repo=f"model_zoo/{entry['id']}/validation_report.md", path_or_fileobj=md.encode()),
    ]
    out = {"ok": True, "entry": entry, "report": md, "files": [o.path_in_repo for o in ops], "pr_url": None}
    if dry_run:
        return out
    info = hf.create_commit(
        repo_id=registry_repo,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Model zoo: add {entry['id']} ({model_repo_id}@{str(entry['revision'])[:8]})",
        commit_description=md[:60000],
        create_pr=True,
    )
    out["pr_url"] = getattr(info, "pr_url", None) or str(info)
    return out
