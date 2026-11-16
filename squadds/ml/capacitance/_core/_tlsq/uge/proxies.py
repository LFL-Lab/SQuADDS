"""Physics-proxy plugins. A proxy maps a Scene to extra node / edge / global quantities that are cheap to compute and
physically aligned with the target. Swap the proxy to move the recipe to another observable (e.g. a resonator dataset
would plug in a quasi-TEM transmission-line proxy for f0 and kappa); the rest of the pipeline is unchanged."""

from __future__ import annotations

import numpy as np

from ..embed.proxy import mom_capacitance_gds, mom_capacitance_gds_avg, mom_capacitance_gds_flagship, mom_capacitance_gds_rich
from .scene import Scene

EPS0 = 8.8541878128e-3  # fF/µm


class Proxy:
    name = "none"

    def compute(self, scene: Scene, edges: dict) -> dict:
        """Return {"node": {i: {...}}, "edge": {(i, j): {...}}, "global": {...}}; `edges` holds the geometry edge features."""
        return {"node": {}, "edge": {}, "global": {}}


class NoProxy(Proxy):
    name = "none"


class ConformalCoupling(Proxy):
    """Solver-free: per-pair coplanar-strip conformal-map coupling (already a geometry edge feature) turned into a
    capacitance estimate C_ij ~ eps0 eps_eff * sum L K(k')/K(k), and node totals."""

    name = "conformal"

    def compute(self, scene, edges):
        e_eff = (scene.stack["substrate_eps_r"] + 1) / 2
        out = {"node": {i: {"log_Cself": -np.inf} for i in range(scene.n)}, "edge": {}, "global": {}}
        tot = np.zeros(scene.n)
        for (i, j), f in edges.items():
            c = EPS0 * e_eff * f["_cps"]
            out["edge"][(i, j)] = {"log_C": float(np.log(max(c, 1e-9)))}
            tot[i] += c
            tot[j] += c
        for i in range(scene.n):
            out["node"][i] = {"log_Cself": float(np.log(max(tot[i], 1e-9)))}
        return out


class CapacitanceMoM(Proxy):
    """Thin-film method of moments on the exact polygons (quadtree mesh, whole ground plane): full Maxwell matrix."""

    name = "mom"

    def __init__(self, max_cells: int = 8000, cells_per_gap: float = 3.0, n_offsets: int = 1, mesh: str = "v2",
                 substrate: str = "slab_open"):
        self.max_cells, self.cells_per_gap, self.n_offsets, self.mesh = max_cells, cells_per_gap, n_offsets, mesh
        self.substrate = substrate  # MoM Green's function (embed.proxy.SUBSTRATE; PRIOR_SCHEME.md 14.6)

    def compute(self, scene, edges):
        from ..embed.proxy import substrate

        with substrate(self.substrate, scene.stack.get("substrate_thickness_um")):
            return self._compute(scene, edges)

    def _compute(self, scene, edges):
        metals, kw = [c.geom for c in scene.conductors], dict(eps_r=scene.stack["substrate_eps_r"], max_cells=self.max_cells,
                                                              cells_per_gap=self.cells_per_gap)
        if self.mesh == "v3-rich":  # conductor-anchored mesh, Richardson-extrapolated (gap/3 and gap/6)
            C, n, hm = mom_capacitance_gds_rich(metals, kw["eps_r"], self.cells_per_gap, self.max_cells, 2 * self.max_cells)
        elif self.mesh in ("lite", "formula"):  # flagship-fast cheap priors (PRIOR_SCHEME.md Part III): micro-MoM at a panel budget / closed form
            from ..embed.fastprior import formula_prior, micro_mom

            if self.mesh == "lite":
                C, _, info = micro_mom(metals[0], metals[1:], kw["eps_r"], budget=self.max_cells, cells_per_gap=self.cells_per_gap)
                n, hm = int(sum(info["panels"])), float(min(info["h_um"]))
                mg = float(info["min_gap_um"])  # the mesh's smallest conductor gap (h = gap / ppg off the floor)
            else:
                C, info = formula_prior(metals[0], metals[1:], kw["eps_r"])
                n, hm = 0, float(info["min_gap_um"])
        elif self.mesh == "flagship":  # strip-merged, resolution-controlled Richardson pair, H-matrix solve (PRIOR_SCHEME.md Part II)
            C, n, hm, _ = mom_capacitance_gds_flagship(metals, kw["eps_r"], self.cells_per_gap, self.max_cells)
        else:
            if self.mesh != "v2":
                kw["mesh"] = self.mesh
            try:
                C, n, hm = mom_capacitance_gds_avg(metals, self.n_offsets, **kw) if self.n_offsets > 1 else mom_capacitance_gds(metals, **kw)
            except np.linalg.LinAlgError:
                if self.mesh != "v3":
                    raise
                # v3 single solve: sliver edge panels can make the gap/3 system indefinite (~0.2 % of designs); use gap/6
                kw.update(cells_per_gap=2 * self.cells_per_gap, max_cells=2 * self.max_cells)
                C, n, hm = mom_capacitance_gds(metals, **kw)
        out = {"node": {i: {"log_Cself": float(np.log(max(C[i, i], 1e-9)))} for i in range(scene.n)}, "edge": {},
               "global": {"cells": float(n), "h_min_um": float(hm)}, "matrix": C}
        if self.mesh != "lite":  # every variant records the layout's smallest conductor gap (review W1); lite: the mesh's own
            from ..embed.fastprior import gap_of
            mg = float(info["min_gap_um"]) if self.mesh == "formula" else float(gap_of(metals))
        out["global"]["min_gap_um"] = mg
        for (i, j) in edges:
            out["edge"][(i, j)] = {"log_C": float(np.log(max(-C[i, j], 1e-9)))}
        return out


PROXIES = {p.name: p for p in (NoProxy, ConformalCoupling, CapacitanceMoM)}
