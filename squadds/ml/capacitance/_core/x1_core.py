"""X1 shared model code (agents/v4test/PROTOCOL.md): row -> arrays, multifid forward, surrogate priors. No solver, no labels."""
import numpy as np

from ._tlsq.uge.blocks import edge_matrix

PPG = (0.375, 0.75, 1.5, 3.0)


def schema_of(ck):
    s = ck.get('schema', ck)
    return s['nodekeys'], s['edgekeys'], s.get('globkeys', ['log_chip_area', 'log_pocket_area', 'log_substrate_t', 'eps_eff'])


def raw_arrays(row, nk, ek, gk):
    """Unnormalised X (n, nk), E (P, ek), G (gk) exactly as geomfast infer.Predictor.from_features builds them."""
    n = len(row['names'])
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    X = np.array([[(float(not np.isfinite(nd['log_min_halfwidth'])) if k == 'missing_log_min_halfwidth' else nd[k]) for k in nk]
                  for nd in row['nodes']], np.float32)
    E = edge_matrix(row['edges'], ek, n)
    G = np.array([row['global'][k] for k in gk], np.float32)
    return X, E, G, pairs


def norm(z, scale):
    med, iqr = np.array(scale['median']), np.array(scale['iqr'])
    return np.clip((np.where(np.isfinite(z), z, med) - med) / iqr, -10, 10).astype(np.float32)


class Multifid:
    """multifid residual arm (scripts/multifid/train_mf.py Net), seeds averaged as full matrices (analyze.py)."""

    def __init__(self, paths, threads=2):
        import torch
        from .train_mf import build
        torch.set_num_threads(threads)
        self.models = []
        for p in paths:
            c = torch.load(p, map_location='cpu', weights_only=False)
            assert not c['curve']
            m = build(len(c['scales']['X']['median']), len(c['scales']['E']['median']), len(c['scales']['G']['median']), 2)
            m.load_state_dict(c['state'])
            m.eval()
            self.models.append((c, m))
        self.levels = self.models[0][0]['levels']
        assert all(c['levels'] == self.levels for c, _ in self.models)

    def predict(self, row, levels):
        """levels: {ppg: positive mutual matrix (n, n)} in row order."""
        nk, ek, gk = schema_of(self.models[0][0])
        X, E, G, pairs = raw_arrays(row, nk, ek, gk)
        i, j = pairs.T
        sig = np.array([nm != 'G' for nm in row['names']])
        return self.predict_arrays(X, E, G, pairs, sig, {p: np.asarray(levels[p], float)[i, j] for p in self.levels})

    def predict_arrays(self, X0, E0, G0, pairs, sig, lev):
        """raw (unnormalised) X (n, f), E (P, f), G (g), pairs (P, 2), sig (n,), lev {ppg: positive mutual (P,)}."""
        import torch
        out = []
        for c, m in self.models:
            assert schema_of(c) == schema_of(self.models[0][0])
            L = [np.log(np.asarray(lev[p], float)) for p in self.levels]
            assert all(np.isfinite(v).all() for v in L)
            X, E, G = X0, E0, G0
            lf, lc = L[-1], L[-2]
            anchor, slope = 2 * lf - lc, lf - lc
            E = np.concatenate([E] + [(L[k + 1] - L[k])[:, None].astype(np.float32) for k in range(len(L) - 1)], -1)
            X, E, G = norm(X, c['scales']['X'])[None], norm(E, c['scales']['E'])[None], norm(G[None], c['scales']['G'])
            bbase = np.where(sig, c['closure']['1'], c['closure']['0']).astype(np.float32)[None]
            T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
            n = len(sig)
            with torch.inference_mode():
                C = m(T(X), T(E), T(G), torch.ones((1, n), dtype=torch.bool), torch.ones((1, len(pairs)), dtype=torch.bool),
                      torch.from_numpy(pairs), T(anchor[None]), T(slope[None]), T(bbase), False)[0]
            out.append(C[0].numpy().astype(np.float64))
        r = np.mean(out, 0)
        assert np.isfinite(r).all()
        return r


class Surrogate:
    """learnedprior surrogate (lp_surrogate.predict_bucket on lp_label.pack(row)) -> priors at ppg 1.5, 3."""

    def __init__(self, path):
        from . import lp_surrogate as S
        self.S = S
        self.model, self.prep, _ = S.load_model(path, 'cpu')
        sch = S.SCHEMA
        self.nk, self.ek, self.gk = sch['nodekeys'], sch['edgekeys'], sch['globkeys']

    def priors(self, row):
        X, E, G, pairs = raw_arrays(row, self.nk, self.ek, self.gk)
        n = len(row['names'])
        lp = self.S.predict_bucket(self.model, self.prep, dict(n=n, X=X[None], E=E[None], G=G[None]), 'cpu')[0]
        iu = np.triu_indices(n, 1)
        pri = {}
        for k, p in enumerate((1.5, 3.0)):
            M = np.zeros((n, n))
            M[iu] = np.exp(lp[:, k])
            pri[p] = M + M.T
        return pri


def c0(levels, lo=1.5, hi=3.0):
    """Richardson extrapolation in log of positive mutual / self matrices: exp(2 log c(hi) - log c(lo)); Maxwell form."""
    a, b = np.abs(np.asarray(levels[lo], float)), np.abs(np.asarray(levels[hi], float))
    m = np.exp(2 * np.log(np.maximum(b, 1e-300)) - np.log(np.maximum(a, 1e-300)))
    n = len(m)
    off = ~np.eye(n, dtype=bool)
    C = np.where(off, -m, m)
    return C


def to_maxwell(P):
    """stored prior P = -C (positive mutuals) -> Maxwell C."""
    return -np.asarray(P, float)
