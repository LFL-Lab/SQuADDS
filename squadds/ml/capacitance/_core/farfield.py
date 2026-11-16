"""farfield arms (agents/farfield/PROTOCOL.md sec. 2): the far-pair dipole form, installed on top of an installed v2m
prior (priorv2.installed(policy='fc', dip=True, margin=True)). install(arm, F, P, PV) patches the bound vendor modules
F (fastprior) / P (proxy) through PV (the priorv2 module patching them) and returns the undo callable.
  a: half-space dipole for every far pair (v2m as deployed; values bit-identical, only timed)
  b: slab dipole for signal-signal pairs of micro_mom's cross-pocket call; same-pocket and ground far pairs half-space
  c: slab dipole for every far pair
Slab factor (1 + eps_r R(d)) / (1 + eps_r), R = PV.slab_R at the substrate thickness of P.SUBSTRATE, d = centroid
distance: PV.dipole_far_slab's expression, applied per pair."""
import time

import numpy as np
import shapely

STATS = dict(cross_calls=0, cross_pairs=0, same_calls=0, same_pairs=0, slab_pairs=0, fill_s=0.0)
MODES = {'a': ('half', 'half'), 'b': ('slab_sig', 'half'), 'c': ('slab_all', 'slab_all')}


def reset():
    for k in STATS:
        STATS[k] = 0.0 if k == 'fill_s' else 0


def _dip(mode, frozen, P, PV, tag):
    def dip(G, conds, pairs, eps_r=11.45, gap=None):
        t = time.perf_counter()
        v = frozen(G, conds, pairs, eps_r, gap)
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        if mode != 'half':
            cen = shapely.get_coordinates(shapely.centroid(np.array([G] + list(conds), dtype=object)))
            d = np.hypot(*(cen[pairs[:, 0]] - cen[pairs[:, 1]]).T)
            s = v * (1 + eps_r * PV.slab_R(d, float(P.SUBSTRATE["t_um"]), eps_r)) / (1 + eps_r)
            use = np.ones(len(pairs), bool) if mode == 'slab_all' else (pairs > 0).all(1)
            v = np.where(use, s, v)
            STATS['slab_pairs'] += int(use.sum())
        STATS[f'{tag}_calls'] += 1
        STATS[f'{tag}_pairs'] += len(pairs)
        STATS['fill_s'] += time.perf_counter() - t
        return v
    return dip


def install(arm, F, P, PV):
    assert F.formula_prior is PV.formula_as_dipole and F._window_solve is not None, 'v2m fill must be installed first'
    frozen = PV._FROZEN['dipole_far']
    assert F.dipole_far is frozen, 'dipole_far must be the vendor function under v2m'
    saved = (F.dipole_far, F.formula_prior)
    cross, same = (_dip(m, frozen, P, PV, t) for m, t in zip(MODES[arm], ('cross', 'same')))

    def formula(G, conds, eps_r=11.45, step_um=None, R_gaps=None, pairs=None, gap=None, screen_um=None, workers=1, max_core=8):
        F.dipole_far = same
        try:
            return PV.formula_as_dipole(G, conds, eps_r, step_um, R_gaps, pairs, gap, screen_um, workers, max_core)
        finally:
            F.dipole_far = cross

    F.dipole_far, F.formula_prior = cross, formula

    def undo():
        F.dipole_far, F.formula_prior = saved
    return undo


def install_sentinel(F):
    """G4: any far-fill call raises."""
    saved = (F.dipole_far, F.formula_prior)

    def boom(*a, **k):
        raise RuntimeError('far fill called')
    F.dipole_far = F.formula_prior = boom

    def undo():
        F.dipole_far, F.formula_prior = saved
    return undo
