"""Scoring of capacitance models with the paper's metric rules.

Per layout and entry kind: the median over its entries of ``|p - y| / max(|y|, 0.1 fF)`` in %, the MAE and the RMSE in fF; then
the mean over layouts (the layout mean), so a 68-conductor chip and a two-pad capacitor count once each. Kinds: ``mutual``
(every off-diagonal pair, the headline), ``signal-signal``, ``signal-ground``, ``signal_self``, ``ground_self``. Unknown
(NaN) label entries are dropped and counted. Every report carries the wall time, machine and thread count.

Validation scoring is open to everyone. Test scoring (``split="test"``) is maintainer-only: once per declared model, recorded
in a ledger, and only for models whose training keys are disjoint from the scored layouts (models trained on the TRAIN split).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FLOOR_FF = 0.1
KINDS = ("mutual", "signal-signal", "signal-ground", "signal_self", "ground_self")


# ---------------------------------------------------------------------------------------------------------------- models
def checkpoint_paths(model) -> list[Path]:
    """Checkpoint files of a model given as a CapacitanceModel, a run dir / .pt path, or a list of them."""
    from .models import CapacitanceModel

    if isinstance(model, CapacitanceModel):
        if model.entry.get("learned_prior") or model.entry.get("stack") is None:
            raise ValueError(
                f"{model.name}: array scoring needs a prior-based model (not a learned-prior / solver-free model)"
            )
        return [Path(model.root) / c["path"] for c in model.entry["checkpoints"]]
    if isinstance(model, (str, os.PathLike)):
        p = Path(model)
        return [p / "model.pt" if p.is_dir() else p]
    if isinstance(model, (list, tuple)):
        return [q for m in model for q in checkpoint_paths(m)]
    raise TypeError(f"cannot read checkpoints from {type(model).__name__}")


def load_checkpoints(model, threads: int = 2) -> list[tuple[dict, object]]:
    """[(checkpoint dict, network in eval mode)] of a model (CapacitanceModel, path(s), run dir(s), or checkpoint dict(s))."""
    import torch

    from .network import build_model

    torch.set_num_threads(int(threads))
    items = model if isinstance(model, (list, tuple)) and model and isinstance(model[0], dict) else None
    if isinstance(model, dict):
        items = [model]
    if items is None:
        items = [torch.load(p, map_location="cpu", weights_only=False) for p in checkpoint_paths(model)]
    out = []
    for c in items:
        s = c["scales"]
        net = build_model(len(s["X"]["median"]), len(s["E"]["median"]), len(s["G"]["median"]), c["rounds"])
        net.load_state_dict(c["state"])
        net.eval()
        out.append((c, net))
    if len({c["arm"] for c, _ in out}) != 1:
        raise ValueError("an ensemble must share one arm")
    return out


def _cols(src: list, dst: list) -> list[int]:
    missing = [k for k in dst if k not in src]
    if missing:
        raise KeyError(f"features {missing[:5]} of the checkpoint are not in the data")
    return [src.index(k) for k in dst]


def predict_arrays(models, d: dict, schema: dict, rows=None, chunk: int = 64) -> np.ndarray:
    """Ensemble-mean Maxwell matrices (fF) of shard rows ``rows`` of one shard dict ``d`` (all rows by default): the
    checkpoint's own standardisation, intercepts and arm, as ``infer.Predictor.from_features`` does per layout."""
    import torch

    rows = np.arange(len(d["key"])) if rows is None else np.asarray(rows)
    pairs = d["pairs"]
    out = np.zeros((len(rows), d["sig"].shape[1], d["sig"].shape[1]))
    for c, net in models:
        cs = c.get("schema", schema)
        nk, ek, gk = cs["nodekeys"], cs["edgekeys"], cs.get("globkeys", schema["globkeys"])
        xi, ei_, gi = _cols(schema["nodekeys"], nk), _cols(schema["edgekeys"], ek), _cols(schema["globkeys"], gk)
        arm = c["arm"]
        direct = arm == "direct_gnn"
        eg = ek.index("is_ground_pair")
        ni = nk.index("role_ground")
        for s in range(0, len(rows), chunk):
            b = rows[s : s + chunk]
            X, E, G = d["X"][b][:, :, xi].copy(), d["E"][b][:, :, ei_].copy(), d["G"][b][:, gi].copy()
            kinds = E[:, :, eg].astype(int)
            if direct:
                di = c["direct_intercepts"]
                base = np.where(kinds == 0, di["0"], di["1"]).astype(np.float32)
            else:
                lp = np.log(d["pri"][b].astype(np.float32))
                base = (2 * lp[:, :, 1] - lp[:, :, 0]).astype(np.float32)
                E = np.concatenate([E, (lp[:, :, 1] - lp[:, :, 0])[:, :, None]], -1).astype(np.float32)
            if arm in ("pair_mlp", "mesh", "lofo_pair"):
                X, G = X[:, :, [ni]], np.zeros((len(b), 1), np.float32)
            zs = []
            for name, z in (("X", X), ("E", E), ("G", G)):
                med, iqr = np.array(c["scales"][name]["median"]), np.array(c["scales"][name]["iqr"])
                zs.append(np.clip((np.where(np.isfinite(z), z, med) - med) / iqr, -10, 10).astype(np.float32))
            cl = c["closure_intercepts"]
            bbase = np.where(d["sig"][b], cl.get("1", cl.get("True")), cl.get("0", cl.get("False"))).astype(np.float32)
            T = torch.from_numpy
            M = torch.ones(d["sig"][b].shape, dtype=torch.bool)
            PM = torch.ones(base.shape, dtype=torch.bool)
            with torch.inference_mode():
                C, _ = net(T(zs[0]), T(zs[1]), T(zs[2]), M, PM, T(pairs), T(base), T(bbase), direct)
            out[s : s + len(b)] += C.numpy().astype(np.float64)
    out /= len(models)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite prediction")
    return out


# ---------------------------------------------------------------------------------------------------------------- metrics
def entry_kinds(sig: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    i, j = pairs.T
    return np.where(sig[i] & sig[j], "signal-signal", "signal-ground")


def layout_rows(y: np.ndarray, p: np.ndarray, sig: np.ndarray, pairs: np.ndarray, key: str, family: str) -> list[dict]:
    """Per-kind metrics of one layout (y, p: Maxwell matrices in fF)."""
    i, j = pairs.T
    yk, pk = -y[i, j], -p[i, j]
    kk = entry_kinds(sig, pairs)
    dy, dp = np.diag(y), np.diag(p)
    sets = {
        "mutual": (yk, pk),
        "signal-signal": (yk[kk == "signal-signal"], pk[kk == "signal-signal"]),
        "signal-ground": (yk[kk == "signal-ground"], pk[kk == "signal-ground"]),
        "signal_self": (dy[sig], dp[sig]),
        "ground_self": (dy[~sig], dp[~sig]),
    }
    out = []
    for kind, (a, b) in sets.items():
        m = np.isfinite(a)
        if not m.any():
            continue
        a, b = a[m].astype(float), b[m].astype(float)
        err = np.abs(b - a)
        out.append(
            dict(
                key=key,
                family=family,
                kind=kind,
                n=int(m.sum()),
                dropped=int((~m).sum()),
                median_pct=float(np.median(err / np.maximum(np.abs(a), FLOOR_FF)) * 100),
                mae_fF=float(err.mean()),
                rmse_fF=float(np.sqrt(np.mean(err**2))),
            )
        )
    return out


def summarize(layouts) -> tuple:
    """(per family x kind layout means, overall per kind layout means) of a layout-metric frame."""
    import pandas as pd

    L = layouts if isinstance(layouts, pd.DataFrame) else pd.DataFrame(layouts)
    agg = dict(
        layouts=("key", "size"), entries=("n", "sum"), median_pct=("median_pct", "mean"), mae_fF=("mae_fF", "mean")
    )
    agg["rmse_fF"] = ("rmse_fF", "mean")
    fam = L.groupby(["family", "kind"]).agg(**agg).reset_index()
    tot = L.groupby("kind").agg(**agg).reset_index()
    return fam, tot


@dataclass
class ScoreReport:
    layouts: object
    by_family: object
    overall: object
    timing: dict = field(default_factory=dict)
    split: str = "val"
    model: str = ""

    def headline(self, kind: str = "mutual") -> dict:
        r = self.overall[self.overall.kind == kind].iloc[0]
        return dict(
            kind=kind,
            layouts=int(r.layouts),
            median_pct=float(r.median_pct),
            mae_fF=float(r.mae_fF),
            rmse_fF=float(r.rmse_fF),
        )

    def __str__(self) -> str:
        h = self.headline()
        t = self.timing
        return (
            f"{self.model} on {self.split}: {h['layouts']} layouts, layout-mean median {h['median_pct']:.4f} % "
            f"(MAE {h['mae_fF']:.4f} fF, RMSE {h['rmse_fF']:.4f} fF); {t.get('seconds', 0):.1f} s, "
            f"{t.get('threads')} threads, {t.get('machine')}"
        )

    def to_dict(self) -> dict:
        return dict(
            model=self.model,
            split=self.split,
            headline=self.headline(),
            overall=self.overall.to_dict("records"),
            by_family=self.by_family.to_dict("records"),
            timing=self.timing,
        )


# ---------------------------------------------------------------------------------------------------------------- guards
def _guard(split, maintainer, keys, models, ledger_dir, model_id):
    if split != "test":
        return None
    if not maintainer:
        raise PermissionError(
            "test scoring is maintainer-only (once per declared model, TRAIN-split models only); score on split='val'"
        )
    if ledger_dir is None:
        raise ValueError("maintainer test scoring needs ledger_dir (the once-per-model ledger)")
    if models is not None:
        trained = set().union(*[set(map(str, c.get("train_keys", []))) for c, _ in models])
        overlap = trained & set(map(str, keys))
        if overlap:
            raise PermissionError(
                f"{len(overlap)} scored test layouts are training layouts of this model (e.g. {sorted(overlap)[:3]})"
            )
    lp = Path(ledger_dir) / f"test__{model_id}.scored"
    lp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise PermissionError(
            f"model {model_id} was already scored on test ({lp.read_text(encoding='utf-8').strip()})"
        ) from None
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), layouts=len(keys))))
    return lp


def _model_id(models) -> str:
    import hashlib

    h = hashlib.sha256()
    for _, net in models:
        for k, v in sorted(net.state_dict().items()):
            h.update(k.encode())
            h.update(v.numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- score
def score(
    model,
    data,
    labels=None,
    split: str = "val",
    maintainer: bool = False,
    threads: int = 2,
    families=None,
    keys=None,
    ledger_dir=None,
    name: str | None = None,
    **predict_kwargs,
) -> ScoreReport:
    """Score a model on the ``split`` rows of a shard dir, or on layouts with labels.

    model: ``"QCap"`` / ``"QCap-Fast"`` (released), a CapacitanceModel, a run dir /
    .pt path (or a list = ensemble), or checkpoint dict(s). data: a shard dir (array route: the stored embedding and priors) or
    a list of layouts (GDS paths / PreparedLayouts, predicted end to end) with ``labels`` = one (names, Maxwell matrix fF) or
    ``cap_matrix.csv`` path per layout. ``families`` / ``keys`` restrict the shard rows."""
    from ._runtime import machine
    from .training_data import read_q3d_csv, read_schema, read_shards

    if split == "test" and not maintainer:
        _guard(split, maintainer, [], None, ledger_dir, "")
    t0 = time.perf_counter()
    rows = []
    if isinstance(data, (str, os.PathLike)) and Path(data).is_dir():
        from .models import is_model_name, load_model

        if is_model_name(model):
            model = load_model(model)
        models = load_checkpoints(model, threads)
        schema = read_schema(data)
        sel = []
        for d in read_shards(data):
            m = d["split"] == split
            if families is not None:
                m &= np.isin(d["family"], list(families))
            if keys is not None:
                m &= np.isin(d["key"], list(keys))
            if m.any():
                sel.append((d, np.flatnonzero(m)))
        all_keys = [str(k) for d, ix in sel for k in d["key"][ix]]
        if not all_keys:
            raise ValueError(f"no {split!r} rows in {data}")
        mid = _model_id(models)
        _guard(split, maintainer, all_keys, models, ledger_dir, mid)
        tn = 0.0
        for d, ix in sel:
            t = time.perf_counter()
            P = predict_arrays(models, d, schema, ix)
            tn += time.perf_counter() - t
            for b, p in zip(ix, P):
                rows += layout_rows(
                    d["C"][b].astype(float), p, d["sig"][b], d["pairs"], str(d["key"][b]), str(d["family"][b])
                )
        timing = dict(
            network_s=tn, per_layout_network_s=tn / len(all_keys), route="shard arrays (stored embedding + priors)"
        )
        label = name or f"{models[0][0]['arm']} x{len(models)} ({mid})"
    else:
        from .models import CapacitanceModel, load_model
        from .predictor import CapacitancePredictor

        items = list(data) if isinstance(data, (list, tuple)) else [data]
        if labels is None or len(labels) != len(items):
            raise ValueError("layout scoring needs one label per layout")
        if split == "test":
            raise PermissionError("maintainer test scoring runs on shard dirs (train keys are checked there)")
        if isinstance(model, str):
            model = load_model(model)
        if not isinstance(model, CapacitanceModel):
            from .train import trained_model

            model = trained_model(checkpoint_paths(model))
        pred = CapacitancePredictor(model, threads=threads)
        times = []
        for k, (lay, lab) in enumerate(zip(items, labels)):
            names, C = (read_q3d_csv(lab)[:2]) if isinstance(lab, (str, os.PathLike)) else lab
            r = pred.predict(lay, **predict_kwargs)
            times.append(r.timing_s["total_s"])
            if sorted(r.names) != sorted(names):
                raise ValueError(f"layout {k}: predicted nets {r.names} differ from the label nets {names}")
            idx = [list(names).index(x) for x in r.names]
            y = np.asarray(C, float)[np.ix_(idx, idx)]
            sig = np.array([x != "G" for x in r.names])
            rows += layout_rows(
                y,
                r.matrix_fF,
                sig,
                np.array([(i, j) for i in range(len(sig)) for j in range(i + 1, len(sig))]),
                str(getattr(lay, "source", lay)),
                "layouts",
            )
        timing = dict(
            total_s=float(np.sum(times)),
            per_layout_median_s=float(np.median(times)),
            route="end to end from the layout",
        )
        label = name or model.name
    import pandas as pd

    L = pd.DataFrame(rows)
    fam, tot = summarize(L)
    timing.update(seconds=time.perf_counter() - t0, threads=int(threads), machine=machine())
    return ScoreReport(L, fam, tot, timing, split, label)


def paired_difference(a, b, kind: str = "mutual", n_boot: int = 2000, seed: int = 0) -> dict:
    """Layout-paired difference a - b of the layout means (median % and MAE fF) with a layout-bootstrap 95% interval.
    a, b: ScoreReports or layout frames scored on the same layouts."""
    La = (a.layouts if hasattr(a, "layouts") else a).query("kind == @kind").set_index("key")
    Lb = (b.layouts if hasattr(b, "layouts") else b).query("kind == @kind").set_index("key")
    common = La.index.intersection(Lb.index)
    if len(common) != len(La) or len(common) != len(Lb):
        raise ValueError("the two scores are not on the same layouts")
    rng = np.random.default_rng(seed)
    out = {"layouts": len(common)}
    for col in ("median_pct", "mae_fF", "rmse_fF"):
        d = (La.loc[common, col] - Lb.loc[common, col]).to_numpy()
        boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1) if len(d) > 1 else np.array([d.mean()] * 2)
        out[col] = dict(diff=float(d.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))
    return out
