"""Fast tier: the learned prior surrogate (agents/learnedprior, t100_norm) in place of the two field solves.

priors(model, prep, row) = the statements of scripts/learnedprior/lp_bench.py `Learned.from_gds` between the features and the
network: pack the scene-graph row with the P2d schema (scripts/learnedprior/lp_label.py `pack`), predict log P at 1.5 and 3
panels per gap with lp_surrogate.predict_bucket (one layout), and return the symmetric positive mutual matrices that
infer.Predictor.from_features reads. Self / closure terms are not inputs, so the diagonals stay zero.
"""

from __future__ import annotations

import numpy as np

from . import lp_surrogate as S
from ._tlsq.uge.blocks import edge_matrix

PPG = (1.5, 3.0)


def load(path):
    """(model, prep, checkpoint) on CPU (lp_surrogate.load_model)."""
    return S.load_model(str(path), "cpu")


def pack(row: dict) -> dict:
    """lp_label.pack: X (n, nodekeys), E (P, edgekeys), G (globkeys); pairs i < j in scene order."""
    n = len(row["names"])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    nk = S.SCHEMA["nodekeys"]
    X = np.array([[(float(not np.isfinite(nd["log_min_halfwidth"])) if k == "missing_log_min_halfwidth" else nd[k]) for k in nk]
                  for nd in row["nodes"]], np.float32)
    E = edge_matrix(row["edges"], S.SCHEMA["edgekeys"], n)
    Gv = np.array([row["global"][k] for k in S.SCHEMA["globkeys"]], np.float32)
    return dict(X=X, E=E, G=Gv, names=row["names"])


def priors(model, prep, row: dict) -> dict:
    z = pack(row)
    n = len(z["names"])
    b = dict(n=n, X=z["X"][None], E=z["E"][None], G=z["G"][None])
    lp = S.predict_bucket(model, prep, b, "cpu")[0]
    iu = np.triu_indices(n, 1)
    out = {}
    for k, p in enumerate(PPG):
        M = np.zeros((n, n))
        v = np.exp(lp[:, k])
        fl = getattr(prep, "floor_ff", None)
        M[iu] = v if fl is None else np.maximum(v, fl)  # registry learned_prior.floor_ff: the training minimum (agents/embedspeed)
        out[p] = M + M.T
    return out
