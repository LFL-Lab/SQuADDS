"""learnedprior surrogate (PROTOCOL.md section 3): tri-GNN on the P2d scene-graph features -> (base, delta) per pair.

  python lp_surrogate.py train --p2d DIR [--gen DIR] --out DIR [--steps N] [--raw] [--gen-frac F] [--seed S]
  python lp_surrogate.py predict --ckpt FILE --data DIR --kind p2d|gen --out FILE.npz

Data buckets are homogeneous in conductor count n (no padding). Pairs i < j in the stored order.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import argparse
import glob
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "schema_p2d.json").read_text())
NK, EK, GK = SCHEMA["nodekeys"], SCHEMA["edgekeys"], SCHEMA["globkeys"]
NODE_DIM = {"log_area": 2, "log_perimeter": 1, "log_min_halfwidth": 1, "log_width": 1}
EDGE_DIM = {"log_dmin": 1, "log_cps": 1, "mom_0.5": 0.5, "mom_1": 0.0, "mom_2": -1.0}
EDGE_DIM.update({f"lg_{k}": v for k, v in list(EDGE_DIM.items())})
GLOB_DIM = {"log_chip_area": 2, "log_pocket_area": 2, "log_substrate_t": 1}
GROUPS = ("geomfast", "base", "large", "scaled", "near")
T_GROUP = {"t1_near": "near", "t2_large": "large", "t3_scaled": "scaled"}


# ----------------------------------------------------------------------------------------------------------- data
def load_p2d(root) -> list:
    out = []
    for p in sorted(glob.glob(f"{root}/n*.npz")):
        d = np.load(p, allow_pickle=False)
        out.append(dict(n=d["X"].shape[1], X=d["X"], E=d["E"][:, :, :len(EK)], G=d["G"], pri=d["pri"].astype(np.float64), key=d["key"],
                        split=d["split"], family=d["family"], group=np.full(len(d["key"]), "geomfast"), scale=np.ones(len(d["key"]))))
    return out


def load_gen(roots) -> list:
    """roots: one packed dir or several separated by "+" (v1 corpus + targeted corpus; no commas: sbatch --export splits on them)."""
    out = []
    for p in sorted(f for root in str(roots).replace(",", "+").split("+") for f in glob.glob(f"{root}/n*.npz")):
        d = np.load(p, allow_pickle=False)
        grp = np.array([T_GROUP.get(str(s), str(s)) for s in d["stratum"]])
        out.append(dict(n=d["X"].shape[1], X=d["X"], E=d["E"], G=d["G"], pri=d["pri"].astype(np.float64), key=d["key"], split=d["split"],
                        family=np.char.add("gen_", d["stratum"].astype(str)), group=grp, scale=d["scale"]))
    return out


def merge_buckets(bs: list) -> list:
    by = {}
    for b in bs:
        by.setdefault(b["n"], []).append(b)
    return [{k: (v[0][k] if k == "n" else np.concatenate([x[k] for x in v])) for k in v[0]} for n, v in sorted(by.items())]


def subset(b: dict, m: np.ndarray) -> dict:
    return {k: (v if k == "n" else v[m]) for k, v in b.items()}


def log_L(X: np.ndarray) -> np.ndarray:
    """(B,) log characteristic length: mean over signal nodes of 0.5 log_area."""
    sig = X[:, :, NK.index("role_ground")] < 0.5
    la = X[:, :, NK.index("log_area")]
    return 0.5 * (la * sig).sum(1) / sig.sum(1).clip(1)


class Prep:
    """Scale shift (optional) + robust scaling; TRAIN statistics only."""

    def __init__(self, raw: bool = False):
        self.raw = raw
        self.nshift = np.array([NODE_DIM.get(k, 0.0) for k in NK])
        self.eshift = np.array([EDGE_DIM.get(k, 0.0) for k in EK])
        self.gshift = np.array([GLOB_DIM.get(k, 0.0) for k in GK])

    def features(self, b: dict):
        X, E, G = b["X"].astype(np.float64), b["E"].astype(np.float64), b["G"].astype(np.float64)
        lL = log_L(b["X"]) if not self.raw else np.zeros(len(X))
        if not self.raw:
            X = X - self.nshift * lL[:, None, None]
            E = E - self.eshift * lL[:, None, None]
            G = G - self.gshift * lL[:, None]
            G = np.concatenate([G, lL[:, None]], 1)
        return X, E, G, lL

    def targets(self, b: dict, lL: np.ndarray):
        y = np.log(b["pri"]) - lL[:, None, None]
        return np.stack([2 * y[..., 1] - y[..., 0], y[..., 1] - y[..., 0]], -1)

    def fit(self, buckets: list):
        acc = {k: [] for k in "XEGT"}
        rng = np.random.default_rng(0)
        for b in buckets:
            tr = np.flatnonzero(b["split"] == "train")
            if not len(tr):
                continue
            tr = rng.choice(tr, min(len(tr), 3000), replace=False)
            X, E, G, lL = self.features(subset(b, np.isin(np.arange(len(b["key"])), tr)))
            acc["X"].append(X.reshape(-1, X.shape[-1]))
            acc["E"].append(E.reshape(-1, E.shape[-1]))
            acc["G"].append(G)
            acc["T"].append(self.targets(subset(b, np.isin(np.arange(len(b["key"])), tr)), lL).reshape(-1, 2))
        self.stats = {}
        for k in "XEG":
            v = np.concatenate(acc[k])
            med = np.nanmedian(v, 0)
            iq = np.nanpercentile(v, 75, 0) - np.nanpercentile(v, 25, 0)
            self.stats[k] = (med, np.where(iq > 1e-6, iq, 1.0))
        t = np.concatenate(acc["T"])
        self.stats["T"] = (t.mean(0), t.std(0))
        return self

    def apply(self, b: dict):
        X, E, G, lL = self.features(b)
        out = []
        for k, z in (("X", X), ("E", E), ("G", G)):
            med, iq = self.stats[k]
            out.append(np.clip((np.where(np.isfinite(z), z, med) - med) / iq, -10, 10).astype(np.float32))
        return (*out, lL)

    def state(self):
        return dict(raw=self.raw, stats={k: [np.asarray(x).tolist() for x in v] for k, v in self.stats.items()})

    @classmethod
    def from_state(cls, s):
        p = cls(s["raw"])
        p.stats = {k: tuple(np.array(x) for x in v) for k, v in s["stats"].items()}
        return p


# ---------------------------------------------------------------------------------------------------------- model
def mlp(i, o, w):
    return nn.Sequential(nn.Linear(i, w), nn.SiLU(), nn.Linear(w, o))


class Round(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.me, self.mh = nn.Linear(w, w), nn.Linear(w, w)
        self.node = mlp(3 * w, w, w)
        self.edge = mlp(3 * w, w, w)
        self.ln_t = nn.LayerNorm(w)
        self.ta, self.tb, self.tg = nn.Linear(w, w), nn.Linear(w, w), nn.Linear(w, w)
        self.tout = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w))
        self.ln_h, self.ln_e = nn.LayerNorm(w), nn.LayerNorm(w)

    def forward(self, h, e, off):
        # h (B,N,w), e (B,N,N,w) symmetric, off (N,N,1) off-diagonal mask
        N = h.shape[1]
        m = self.me(e) * self.mh(h)[:, None, :, :] * off  # message j -> i
        mean = m.sum(2) / max(N - 1, 1)
        mx = (m + (off - 1) * 1e4).amax(2) if N > 1 else torch.zeros_like(mean)
        h = h + self.node(torch.cat([self.ln_h(h), mean, mx], -1))
        hs, hd = h[:, :, None] + h[:, None], (h[:, :, None] - h[:, None]).abs()
        e = e + self.edge(torch.cat([self.ln_e(e), hs, hd], -1)) * off
        z = self.ln_t(e)
        a, b = self.ta(z) * off, self.tb(z) * off
        t = torch.einsum("bikc,bjkc->bijc", a, b) / max(N - 2, 1)
        t = 0.5 * (t + t.transpose(1, 2))
        e = e + torch.sigmoid(self.tg(z)) * self.tout(t) * off
        return h, e


class TriGNN(nn.Module):
    def __init__(self, dx, de, dg, w=128, rounds=4):
        super().__init__()
        self.node_in, self.edge_in = mlp(dx + dg, w, w), mlp(de, w, w)
        self.rounds = nn.ModuleList([Round(w) for _ in range(rounds)])
        self.head = mlp(3 * w, 2, w)

    def forward(self, X, Eu, G, iu):
        """X (B,N,dx), Eu (B,P,de) upper-triangle pairs iu=(i,j), G (B,dg) -> (B,P,2) standardised (b, d)."""
        B, N, _ = X.shape
        i, j = iu
        h = self.node_in(torch.cat([X, G[:, None].expand(-1, N, -1)], -1))
        eu = self.edge_in(Eu)
        e = torch.zeros(B, N, N, eu.shape[-1], device=X.device, dtype=eu.dtype)
        e[:, i, j] = eu
        e[:, j, i] = eu
        off = (1 - torch.eye(N, device=X.device, dtype=eu.dtype))[:, :, None]
        for r in self.rounds:
            h, e = r(h, e, off)
        return self.head(torch.cat([e[:, i, j], h[:, i] + h[:, j], (h[:, i] - h[:, j]).abs()], -1))


def build(dx, de, dg, w=128, rounds=4):
    return TriGNN(dx, de, dg, w, rounds)


# ------------------------------------------------------------------------------------------------------- training
def pairs_of(n):
    return np.array([(i, j) for i in range(n) for j in range(i + 1, n)]).T.reshape(2, -1)


def predict_bucket(model, prep, b, device, max_pairs=2**17):
    """(B,P,2) log P (ppg 1.5, 3) predicted for every layout of bucket b."""
    n = b["n"]
    iu = torch.as_tensor(pairs_of(n), device=device)
    X, E, G, lL = prep.apply(b)
    mu, sd = prep.stats["T"]
    bs = max(1, max_pairs // (n * n))
    out = []
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(X), bs):
            T = lambda a: torch.as_tensor(a[s:s + bs], device=device)  # noqa: E731
            z = model(T(X), T(E), T(G), iu).double().cpu().numpy() * sd + mu
            out.append(z)
    z = np.concatenate(out)
    bb, dd = z[..., 0], z[..., 1]
    y3, y1 = bb - dd, bb - 2 * dd
    return np.stack([y1, y3], -1) + lL[:, None, None]


def fidelity(logp_hat, pri):
    """Per-entry C0 errors (C0 = exp(2 log P3 - log P1.5)): |rel| with 0.1 fF floor, abs fF; |delta error|."""
    c0 = np.exp(2 * np.log(pri[..., 1]) - np.log(pri[..., 0]))
    c0h = np.exp(2 * logp_hat[..., 1] - logp_hat[..., 0])
    d = np.log(pri[..., 1] / pri[..., 0])
    dh = logp_hat[..., 1] - logp_hat[..., 0]
    err = np.abs(c0h - c0)
    return err / np.maximum(c0, 0.1), err, np.abs(dh - d), c0


def summarize(rel, err):
    rel, err = np.ravel(rel), np.ravel(err)
    return dict(entries=int(len(rel)), median_percent=float(np.median(rel) * 100), p95_percent=float(np.percentile(rel, 95) * 100),
                mae_fF=float(err.mean()), rmse_fF=float(np.sqrt((err ** 2).mean())))


def cmd_train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    buckets = load_p2d(a.p2d)
    if a.p2d_holdout_fold is not None:  # out-of-fold surrogate for geomfast TRAIN (PROTOCOL amendment 2026-10-17)
        for b in buckets:
            f = np.array([int(hashlib.sha256(str(k).encode()).hexdigest(), 16) % a.p2d_nfolds for k in b["key"]])
            b.update(subset(b, ~((b["split"] == "train") & (f == a.p2d_holdout_fold))))
    if a.gen:
        g = load_gen(a.gen)
        if a.gen_frac < 1:  # nested subsets by parent index hash (learning curve)
            for b in g:
                parent = np.array(["/".join(k.split("/")[:2]) for k in b["key"]])
                u = np.array([int(hashlib.sha256(p.encode()).hexdigest(), 16) % 10**6 / 1e6 for p in parent])
                keep = (b["split"] != "train") | (u < a.gen_frac)
                b.update(subset(b, keep))
        buckets += g
    buckets = [b for b in merge_buckets(buckets) if b["n"] >= 2]
    prep = Prep(a.raw).fit(buckets)
    data = []
    for b in buckets:
        X, E, G, lL = prep.apply(b)
        Y = ((prep.targets(b, lL) - prep.stats["T"][0]) / prep.stats["T"][1]).astype(np.float32)
        data.append(dict(n=b["n"], X=torch.as_tensor(X), E=torch.as_tensor(E), G=torch.as_tensor(G), Y=torch.as_tensor(Y),
                         split=b["split"], group=b["group"], iu=torch.as_tensor(pairs_of(b["n"]), device=dev)))
    index = {g: [] for g in GROUPS}
    for k, d in enumerate(data):
        for g in GROUPS:
            idx = np.flatnonzero((d["split"] == "train") & (d["group"] == g))
            if len(idx):
                index[g].append((k, idx))
    groups = [g for g in GROUPS if index[g]]
    counts = {g: int(sum(len(i) for _, i in index[g])) for g in groups}
    val = [(k, np.flatnonzero(d["split"] == "val")) for k, d in enumerate(data)]
    model = build(data[0]["X"].shape[-1], data[0]["E"].shape[-1], data[0]["G"].shape[-1], a.width, a.rounds).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03, anneal_strategy="cos")
    sd = torch.as_tensor(prep.stats["T"][1], device=dev, dtype=torch.float32)
    meta = dict(args=vars(a), host=platform.node(), device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
                train_counts=counts, val_layouts=int(sum(len(i) for _, i in val)), params=sum(p.numel() for p in model.parameters()))
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)

    def batch_loss(k, idx):
        d = data[k]
        T = lambda z: z[idx].to(dev, non_blocking=True)  # noqa: E731
        z = model(T(d["X"]), T(d["E"]), T(d["G"]), d["iu"]) * sd
        y = T(d["Y"]) * sd
        return (F.huber_loss(z[..., 0], y[..., 0], delta=0.05, reduction="none") + F.huber_loss(z[..., 1], y[..., 1], delta=0.05, reduction="none")).mean(1)

    def evaluate(maxn=600):
        model.eval()
        res = {}
        with torch.inference_mode():
            for k, idx in val:
                if not len(idx):
                    continue
                d = data[k]
                bs = max(1, a.pair_budget // (d["n"] ** 2))
                for s in range(0, min(len(idx), maxn), bs):
                    l = batch_loss(k, torch.as_tensor(idx[s:s + bs]))
                    for gname in set(d["group"][idx[s:s + bs]]):
                        m = d["group"][idx[s:s + bs]] == gname
                        res.setdefault(gname, []).extend(l[torch.as_tensor(m, device=dev)].tolist())
        model.train()
        return {g: float(np.mean(v)) for g, v in res.items()}

    hist, t0, skipped = [], time.perf_counter(), 0
    for step in range(a.steps):
        g = groups[rng.integers(len(groups))]
        w = np.array([len(i) for _, i in index[g]], float)
        k, idx = index[g][rng.choice(len(w), p=w / w.sum())]
        n = data[k]["n"]
        bs = int(max(1, min(a.max_batch, a.pair_budget // (n * n))))
        b = torch.as_tensor(rng.choice(idx, bs))
        loss = batch_loss(k, b).mean()
        if not torch.isfinite(loss):  # skipped and counted (PROTOCOL amendment 2026-10-17); abort if frequent
            skipped += 1
            print(json.dumps(dict(step=step, skipped_nonfinite=skipped, n=int(n), batch=bs)), flush=True)
            if skipped > 100:
                raise RuntimeError(f"non-finite loss at step {step} ({skipped} skipped)")
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            ev = evaluate()
            hist.append(dict(step=step, loss=float(loss), seconds=time.perf_counter() - t0, val=ev))
            print(json.dumps(hist[-1]), flush=True)
            torch.save(dict(state=model.state_dict(), prep=prep.state(), width=a.width, rounds=a.rounds, dims=[data[0]["X"].shape[-1],
                       data[0]["E"].shape[-1], data[0]["G"].shape[-1]], step=step, history=hist), out / "model.pt")
    meta.update(train_seconds=time.perf_counter() - t0, history=hist, skipped_nonfinite=skipped)
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print("TRAIN_DONE", time.perf_counter() - t0, flush=True)


def load_model(path, device="cpu"):
    c = torch.load(path, map_location=device, weights_only=False)
    m = build(*c["dims"], c["width"], c["rounds"]).to(device)
    m.load_state_dict(c["state"])
    m.eval()
    return m, Prep.from_state(c["prep"]), c


def cmd_predict(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, prep, _ = load_model(a.ckpt, dev)
    buckets = load_p2d(a.data) if a.kind == "p2d" else load_gen(a.data)
    keys, logp = [], {}
    for b in merge_buckets(buckets):
        if b["n"] < 2:
            continue
        sel = b["split"] == "val" if a.split == "val" else np.ones(len(b["key"]), bool)
        b = subset(b, sel)
        if not len(b["key"]):
            continue
        lp = predict_bucket(model, prep, b, dev)
        for k, v in zip(b["key"], lp):
            logp[str(k)] = v.astype(np.float64)
    np.savez_compressed(a.out, keys=np.array(list(logp)), **{f"k{i}": v for i, v in enumerate(logp.values())})
    print("PREDICTED", len(logp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("train")
    p.add_argument("--p2d", required=True)
    p.add_argument("--gen")
    p.add_argument("--gen-frac", type=float, default=1.0)
    p.add_argument("--out", required=True)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--pair-budget", type=int, default=2**16)
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p2d-holdout-fold", type=int)
    p.add_argument("--p2d-nfolds", type=int, default=2)
    p = sp.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--kind", choices=["p2d", "gen"], default="p2d")
    p.add_argument("--split", choices=["val", "all"], default="val")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict}[a.cmd](a)
