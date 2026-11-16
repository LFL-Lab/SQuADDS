"""Vectorised drop-in replacements for the H-matrix kernel of tlsq.embed.proxy (same mathematics, no per-row scipy
sparse slicing, no Python loop over image charges or ACA ranks). install() patches the loaded module in place."""
import numpy as np

from ._tlsq.embed import proxy as P

_orig = {}


def _images(eps_r):
    P._image_correction(np.zeros(1), eps_r)
    key = (P.SUBSTRATE["model"], float(P.SUBSTRATE["t_um"]), float(eps_r))
    w, D = P._IMG_CACHE[key][2:]
    return np.asarray(w, float), np.asarray(D, float) ** 2


def _kernel_init(self, c, wh, eps_r, perm):
    _orig["Kernel.__init__"](self, c, wh, eps_r, perm)
    N = self.N.tocsr()
    N.sort_indices()
    self.indptr, self.indices, self.data = N.indptr, N.indices, N.data
    self.w, self.D2 = _images(eps_r) if self.img else (None, None)


def _corr(self, d):
    return np.sqrt((d * d)[..., None] + self.D2) ** -1 @ self.w


def _row(self, i, b0, b1):
    d = np.sqrt((self.x[i] - self.x[b0:b1]) ** 2 + (self.y[i] - self.y[b0:b1]) ** 2)
    r = 1.0 / d
    if self.img:
        r += _corr(self, d)
    s, e = self.indptr[i], self.indptr[i + 1]
    if e > s:
        ix = self.indices[s:e]
        lo, hi = np.searchsorted(ix, (b0, b1))
        if hi > lo:
            r[ix[lo:hi] - b0] += self.data[s + lo:s + hi]
    return r / self.k


def _block(self, a0, a1, b0, b1):
    d = np.sqrt((self.x[a0:a1, None] - self.x[None, b0:b1]) ** 2 + (self.y[a0:a1, None] - self.y[None, b0:b1]) ** 2)
    corr = _corr(self, d) if self.img else None
    if a0 == b0:
        d[np.arange(a1 - a0), np.arange(a1 - a0)] = 1.0
    A = 1.0 / d
    if corr is not None:
        A += corr
    s, e = self.indptr[a0], self.indptr[a1]
    if e > s:
        ix = self.indices[s:e]
        rr = np.repeat(np.arange(a1 - a0), np.diff(self.indptr[a0:a1 + 1]))
        m = (ix >= b0) & (ix < b1)
        np.add.at(A, (rr[m], ix[m] - b0), self.data[s:e][m])
    return A / self.k


def _aca(K, a0, a1, b0, b1, eps, max_rank):
    m, n = a1 - a0, b1 - b0
    R = min(m, n, max_rank)
    U, V = np.empty((m, R)), np.empty((n, R))
    used_r, used_c = np.zeros(m, bool), np.zeros(n, bool)
    i, norm2, k = 0, 0.0, 0
    for _ in range(R):
        r = K.row(a0 + i, b0, b1)
        if k:
            r -= V[:, :k] @ U[i, :k]
        rm = np.where(used_c, 0.0, np.abs(r))
        j = int(np.argmax(rm))
        if rm[j] == 0.0:
            break
        v = r / r[j]
        u = K.row(b0 + j, a0, a1)
        if k:
            u -= U[:, :k] @ V[j, :k]
        used_r[i], used_c[j] = True, True
        nu, nv = float(u @ u), float(v @ v)
        norm2 += nu * nv + (2 * float((U[:, :k].T @ u) @ (V[:, :k].T @ v)) if k else 0.0)
        U[:, k], V[:, k] = u, v
        k += 1
        if nu * nv <= eps**2 * norm2:
            break
        um = np.where(used_r, 0.0, np.abs(u))
        i = int(np.argmax(um))
        if um[i] == 0.0:
            break
    else:
        return None
    return U[:, :k].copy(), V[:, :k].copy()


def _image_correction(d, eps_r, exact=False):
    if exact or P.SUBSTRATE["model"] == "halfspace":
        return _orig["image"](d, eps_r, exact)
    _orig["image"](np.zeros(1), eps_r)
    key = (P.SUBSTRATE["model"], float(P.SUBSTRATE["t_um"]), float(eps_r))
    du, T = P._IMG_CACHE[key][:2]
    u = np.asarray(d, float) / du
    j = np.minimum(u.astype(np.int64), len(T) - 2)
    f = u - j
    t0 = T[j]
    return t0 + f * (T[j + 1] - t0)


REFINE = dict(steps=6, tol=1e-13)


def _chol_solve(A, B):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.linalg.blas import dsymm

    try:
        cf = cho_factor(A.astype(np.float32, order="F"), lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        return _orig["chol_solve"](A, B)
    B = np.asfortranarray(B, dtype=float)
    X = cho_solve(cf, B.astype(np.float32), check_finite=False).astype(float)
    bn = np.linalg.norm(B, axis=0)
    for _ in range(REFINE["steps"]):
        R = B - dsymm(1.0, A, X, side=0, lower=1)
        if (np.linalg.norm(R, axis=0) <= REFINE["tol"] * bn).all():
            return X
        X += cho_solve(cf, R.astype(np.float32), check_finite=False)
    return _orig["chol_solve"](A, B)


def install(dense_max=None, mixed=False):
    if not _orig:
        _orig["Kernel.__init__"] = P._Kernel.__init__
        _orig.update(row=P._Kernel.row, block=P._Kernel.block, aca=P._aca, dense_max=P.DENSE_MAX, image=P._image_correction,
                     chol_solve=P._chol_solve)
        P._Kernel.__init__, P._Kernel.row, P._Kernel.block, P._aca = _kernel_init, _row, _block, _aca
        P._image_correction = _image_correction
    P._chol_solve = _chol_solve if mixed else _orig["chol_solve"]
    if dense_max is not None:
        P.DENSE_MAX = dense_max


def uninstall():
    if _orig:
        P._Kernel.__init__, P._Kernel.row, P._Kernel.block, P._aca = (_orig["Kernel.__init__"], _orig["row"], _orig["block"],
                                                                      _orig["aca"])
        P._image_correction, P._chol_solve = _orig["image"], _orig["chol_solve"]
        P.DENSE_MAX = _orig["dense_max"]
        _orig.clear()
