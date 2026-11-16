"""Prior v2 = frozen geomfast prior + rejtail fix fc (rescue panels), as a monkeypatch of fastprior._window_solve.
Logic copied from scripts/rejtail/rt.py (policies 'exact' and 'fc' only; no wp dependency). Vendor files untouched.
Import after the geomfast vendor dir is on sys.path. Usage:
    import priorv2; with priorv2.installed(): ... fastprior.micro_mom / micro_mom_loc ...
fc acts only on windowed solves (all_rhs False, N > 9) whose FIRST attempt leaves a metal without a panel; everything
else runs the frozen code path."""
import time
import contextlib
import numpy as np
import shapely
from ._tlsq.embed import fastprior as F, proxy as P

PRIOR_VERSION = 'v2_fc_dipfill'
TRACE = []
POLICY = ['fc']


def _solve(c, wh, o, metals, cols, eps_r, all_rhs):
    if len(c) <= P.DENSE_MAX:
        A = P._assemble(c, wh, P.rect_self(wh[:, 0], wh[:, 1]), eps_r, P.rect_potential)
        Bm = (o[:, None] == cols[None, :]).astype(float)
        Q = P._chol_solve(A, Bm) if all_rhs else F._sym_solve(A, Bm)
    else:
        Q = P.mom_solve_fast(c, wh, o, len(metals), eps_r)[:, cols]
    return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)


def _mesh(metals, n_core, hh, cap, refined, majority, levels):
    if not refined:
        return F._lite_mesh(metals, hh, cap, n_core, levels)
    if levels:
        return F._graded_mesh(metals, n_core, hh, levels, majority)
    return P.mesh_flagship(metals, hh, cap, majority=majority)


def _rescue_panels(metals, miss, h):
    """Each panel-less metal cut by a lattice of pitch min(h, sqrt(area) / 2) anchored at its bbox; one panel per piece,
    centre = centroid (point on surface if outside), size = piece bbox (the frozen cut-piece form)."""
    cs, whs, os_ = [], [], []
    for a in miss:
        g = metals[a]
        x0, y0, x1, y1 = g.bounds
        p = min(h, np.sqrt(g.area) / 2)
        nx, ny = max(1, int(np.ceil((x1 - x0) / p))), max(1, int(np.ceil((y1 - y0) / p)))
        gx, gy = np.meshgrid(x0 + p * np.arange(nx), y0 + p * np.arange(ny))
        pcs = shapely.intersection(shapely.box(gx.ravel(), gy.ravel(), gx.ravel() + p, gy.ravel() + p), g)
        pcs = pcs[shapely.area(pcs) > 1e-12]
        bb = shapely.bounds(pcs)
        cen = shapely.get_coordinates(shapely.centroid(pcs))
        inside = shapely.contains_xy(g, cen[:, 0], cen[:, 1])
        if not inside.all():
            cen[~inside] = shapely.get_coordinates(shapely.point_on_surface(pcs[~inside]))
        cs.append(cen)
        whs.append(np.maximum(np.stack([bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]], 1), 1e-6))
        os_.append(np.full(len(pcs), a))
    return np.concatenate(cs), np.concatenate(whs), np.concatenate(os_).astype(int)


def window_solve(metals, n_core, h, budget, eps_r, all_rhs, levels=()):
    pol = POLICY[0] if not all_rhs else 'exact'
    cols = np.arange(len(metals)) if all_rhs else np.arange(1, 1 + n_core)
    t_mesh = t_sol = 0.0
    hh, cap, refined, majority = h, budget, 0, False
    tr = dict(n_core=n_core, n_metals=len(metals), all_rhs=all_rhs, policy=pol, attempts=[])
    best, hist = None, []
    while True:
        t = time.perf_counter()
        c, wh, o, hh, n_raw = _mesh(metals, n_core, hh, cap, refined, majority, levels)
        miss = np.nonzero(np.bincount(o, minlength=len(metals)) == 0)[0]
        att = dict(refined=refined, majority=majority, panels=len(c), missing=miss.tolist())
        t_mesh += time.perf_counter() - t
        tr['attempts'].append(att)
        cs, whs, os_ = c, wh, o
        if pol != 'exact' and len(miss):
            empty = np.array([metals[a].is_empty for a in miss])
            if pol == 'fc' and not refined and (not empty.any() or len(c) <= P.DENSE_MAX):
                t = time.perf_counter()
                if not empty.all():
                    r = _rescue_panels(metals, miss[~empty], hh)
                    cs, whs, os_ = np.concatenate([c, r[0]]), np.concatenate([wh, r[1]]), np.concatenate([o, r[2]]).astype(o.dtype)
                    att['rescued'] = int(len(r[0]))
                if empty.any():
                    att['accept'] = 'empty'
                t_mesh += time.perf_counter() - t
            else:
                att['defer'] = True
        if att.get('defer'):
            doomed = len(c) > P.DENSE_MAX
            hist.append(('doomed',) if doomed else ('mesh', c, wh, o))
            att['solve'] = 'doomed' if doomed else 'deferred'
            ok = False
        else:
            t = time.perf_counter()
            try:
                R = _solve(cs, whs, os_, metals, cols, eps_r, all_rhs)
                off = np.ones(R.shape, bool)
                off[np.arange(len(cols)), cols] = False
                ok = bool(np.all(np.bincount(os_, minlength=len(metals)) > 0) and (not all_rhs or np.all(R[off] < 0)))
                if att.get('accept') == 'empty':
                    ok = True
                att['solve'] = 'ok'
                best = R
                hist.append(('R', R))
            except np.linalg.LinAlgError as e:
                ok = False
                att['solve'], att['err'] = 'error', str(e)[:120]
                hist.append(('err',))
            t_sol += time.perf_counter() - t
        if ok or majority:
            break
        if refined < F.LITE_REFINE:
            hh, cap, refined = hh / 2, np.inf, refined + 1
        else:
            majority = True
    if not ok:  # frozen R binding: the latest attempt that solved (deferred dense attempts solved lazily, newest first)
        R = None
        t = time.perf_counter()
        for e in reversed(hist):
            if e[0] == 'R':
                R = e[1]
                break
            if e[0] == 'mesh':
                try:
                    R = _solve(e[1], e[2], e[3], metals, cols, eps_r, all_rhs)
                    break
                except np.linalg.LinAlgError:
                    continue
        t_sol += time.perf_counter() - t
        assert R is not None, 'no attempt solved (the frozen code raises here too)'
    else:
        R = best
    tr['fired'] = bool(pol == 'fc' and tr['attempts'][0]['missing'])
    TRACE.append(tr)
    return R, {"h_um": hh, "panels": len(c), "raw": n_raw, "refined": refined, "majority": majority, "mesh_s": t_mesh,
               "assemble_s": 0.0, "solve_s": t_sol}


FILL = []    # one entry per formula_prior call replaced by the dipole fill: number of pairs
MARGIN = []  # one entry per window whose margin was widened: (d, margin)
_FROZEN = dict(formula_prior=F.formula_prior, lite_window=F._lite_window, dipole_far=F.dipole_far)
K_TOL = 1e-4


def slab_R(d, t_um=350.0, eps_r=11.45):
    """src/tlsq/embed/fastprior_farv4.slab_R verbatim: R(d) = 1 + 2 sum_n (-K)^n (1 - 8 n^2 t^2/d^2)/(1 + 4 n^2 t^2/d^2)^(5/2)."""
    d = np.asarray(d, float)
    K = (eps_r - 1) / (eps_r + 1)
    n_max = int(np.ceil(np.log(K_TOL) / np.log(K))) if K > 0 else 0
    n = np.arange(1, n_max + 1)[:, None]
    u = (n * t_um / d[None]) ** 2
    return 1 + 2 * np.sum((-K) ** n * (1 - 8 * u) / (1 + 4 * u) ** 2.5, 0)


def dipole_far_slab(G, conds, pairs, eps_r=11.45, gap=None):
    """src/tlsq/embed/fastprior_farv4.dipole_far_v4 verbatim: the frozen dipole_far with (1 + eps_r) -> 1 + eps_r R(d)."""
    base = _FROZEN['dipole_far'](G, conds, pairs, eps_r, gap)
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    cen = shapely.get_coordinates(shapely.centroid(np.array([G] + list(conds), dtype=object)))
    d = np.hypot(*(cen[pairs[:, 0]] - cen[pairs[:, 1]]).T)
    return base * (1 + eps_r * slab_R(d, float(P.SUBSTRATE["t_um"]), eps_r)) / (1 + eps_r)


def formula_as_dipole(G, conds, eps_r=11.45, step_um=None, R_gaps=None, pairs=None, gap=None, screen_um=None, workers=1, max_core=8):
    """Replaces formula_prior inside micro_mom (its only caller there: same-pocket far pairs): the same pairs get
    dipole_far(G, conds, pairs, eps_r, gap), the fill micro_mom uses for cross-pocket far pairs."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    N = len(conds) + 1
    C = np.zeros((N, N))
    v = F.dipole_far(G, conds, pairs, eps_r, gap)
    C[pairs[:, 0], pairs[:, 1]] = v
    C[pairs[:, 1], pairs[:, 0]] = v
    FILL.append(len(pairs))
    return C, {"dipole_fill_pairs": len(pairs)}


def lite_window_margin(G, conds, core, R_g, margin, fill_um, tree, etched):
    """Frozen _lite_window, except margin' = 2 d when d = max distance of a core conductor to the chip ground > margin."""
    d = max(shapely.distance(conds[i], G) for i in core)
    if d > margin:
        MARGIN.append((float(d), float(margin)))
        margin = 2 * d
    return _FROZEN['lite_window'](G, conds, core, R_g, margin, fill_um, tree, etched)


@contextlib.contextmanager
def installed(policy='fc', dip=True, margin=False, slab_alpha=None):
    """Install prior v2 pieces: window solve policy 'fc' (or 'exact' = traced frozen copy, or None = vendor as is), the
    dipole fill of same-pocket far pairs (dip), the window margin widening (margin) and, for arm v2-slab, the slab dipole
    for every far pair with ALPHA_GAP_SHARE = slab_alpha. Always restored."""
    saved = (F._window_solve, F.formula_prior, F._lite_window, F.dipole_far, F.ALPHA_GAP_SHARE)
    if policy is not None:
        POLICY[0] = policy
        F._window_solve = window_solve
    if dip:
        F.formula_prior = formula_as_dipole
    if margin:
        F._lite_window = lite_window_margin
    if slab_alpha is not None:
        F.dipole_far, F.ALPHA_GAP_SHARE = dipole_far_slab, float(slab_alpha)
    try:
        yield
    finally:
        F._window_solve, F.formula_prior, F._lite_window, F.dipole_far, F.ALPHA_GAP_SHARE = saved
        POLICY[0] = 'fc'


def clear():
    TRACE.clear()
    FILL.clear()
    MARGIN.clear()


def fired(trace):
    """True if fc changed the solve of any window of the traced call."""
    return any(t.get('fired') for t in trace)
