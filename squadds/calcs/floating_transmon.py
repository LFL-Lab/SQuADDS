"""Transmon Hamiltonians computed from the full Maxwell capacitance matrix of a planar qubit.

Rows carry positive magnitudes ``<a>_to_<b>`` (fF): the self terms ``<a>_to_<a>`` are the Maxwell diagonal and the
mutuals are :math:`|C_{ab}|`. The Maxwell matrix of the present conductors is rebuilt from them (ground excluded, every
conductor that is not part of the qubit mode held at ground: coupler pads are loaded by their resonators, whose
capacitance is much larger than the qubit's).

* :class:`FloatingTransmonHamiltonian`: two islands (``TransmonPocket``), the pad-to-pad (differential) mode.
* :class:`XmonHamiltonian`: one grounded island (``Xmon``) with any number of claws, :math:`C_\\Sigma` = Maxwell diagonal.

Both reuse the ``TransmonCross`` transmon approximations: :math:`E_J` from the target :math:`(f_q, \\alpha)` via
``Transmon.find_EJ_EC``, :math:`f_q` and :math:`\\alpha` of each row from scqubits with that :math:`E_J` and the row's :math:`E_C`, and :math:`g` from
``TransmonCrossHamiltonian.g_from_cap_matrix`` on the equivalent grounded two-node circuit.
"""

from __future__ import annotations

import numpy as np
from pyEPR.calcs import Convert

from squadds.calcs.transmon_cross import (
    _TRANSMON_CACHE_PRECISION,
    TransmonCrossHamiltonian,
    get_transmon_E01_alpha,
)


def _value(row, key):
    v = row.get(key) if hasattr(row, "get") else None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else abs(f)


def _pair(row, a, b):
    v = _value(row, f"{a}_to_{b}")
    return v if v is not None else _value(row, f"{b}_to_{a}")


def maxwell_from_row(row, names, ground: str = "ground") -> tuple[np.ndarray, list[str]]:
    """Signed Maxwell matrix (fF) of the conductors in ``names`` that a row carries (a conductor is present when its
    self term or its ground term is a number). A missing self term is rebuilt by closure,
    :math:`C_{aa} = C_{ag} + \\sum_{b \\ne a} |C_{ab}|`, over every present conductor."""
    present = [n for n in names if _pair(row, n, n) is not None or _pair(row, n, ground) is not None]
    k = len(present)
    C = np.zeros((k, k))
    for i, a in enumerate(present):
        for j in range(i + 1, k):
            C[i, j] = C[j, i] = -(_pair(row, a, present[j]) or 0.0)
    for i, a in enumerate(present):
        diag = _pair(row, a, a)
        C[i, i] = diag if diag is not None else (_pair(row, a, ground) or 0.0) - C[i].sum()
    return C, present


def floating_transmon_capacitances(C, names, pads=("pad_top", "pad_bot"), coupler=None) -> tuple[float, float]:
    """Effective capacitance of the pad-to-pad mode and its effective coupling capacitance to one coupler (fF).

    With every other conductor at ground, the two-pad block of the Maxwell matrix is
    :math:`\\begin{pmatrix} C_{11} & -C_{12} \\\\ -C_{12} & C_{22} \\end{pmatrix}`, :math:`C_{i\\Sigma} = C_{ii} - C_{12}` (pad :math:`i` to
    ground plus every grounded coupler pad). The mode :math:`\\varphi = V_1 - V_2` (with :math:`Q_1 + Q_2 = 0`) has

    :math:`C_\\mathrm{eff} = \\frac{1}{(C^{-1})_{11} + (C^{-1})_{22} - 2 (C^{-1})_{12}} = C_{12} + \\frac{C_{1\\Sigma} C_{2\\Sigma}}{C_{1\\Sigma} + C_{2\\Sigma}},`

    and a coupler :math:`k` at voltage :math:`V_k` drives it through the capacitive division

    :math:`C_{g,\\mathrm{eff}} = \\frac{C_{2\\Sigma} C_{1k} - C_{1\\Sigma} C_{2k}}{C_{1\\Sigma} + C_{2\\Sigma}}, \\qquad \\beta = C_{g,\\mathrm{eff}} / C_\\mathrm{eff}.`

    Returns ``(C_eff, |C_g_eff|)``; the second value is 0 when ``coupler`` is None or absent."""
    C = np.asarray(C, float)
    names = list(names)
    i, j = names.index(pads[0]), names.index(pads[1])
    c12 = -C[i, j]
    x, y = C[i, i] - c12, C[j, j] - c12
    c_eff = c12 + x * y / (x + y)
    if coupler is None or coupler not in names:
        return float(c_eff), 0.0
    k = names.index(coupler)
    c1k, c2k = -C[i, k], -C[j, k]
    return float(c_eff), float(abs((y * c1k - x * c2k) / (x + y)))


def EC_GHz_from_Cs(C_fF: float) -> float:
    """:math:`E_C = e^2 / (2 C)` in GHz for :math:`C` in fF (the ``TransmonCross`` conversion)."""
    return Convert.Ec_from_Cs(C_fF, units_in="fF", units_out="GHz")


class _MaxwellTransmonHamiltonian(TransmonCrossHamiltonian):
    """Shared row loop: subclasses give :math:`C_\\Sigma` and the coupling capacitance of each row (fF)."""

    readout_coupler = "claw"

    def capacitances(self, row) -> tuple[float, float]:
        """``(C_sigma, C_g)`` of the qubit mode (fF) for one row; ``C_g`` to ``readout_coupler``."""
        raise NotImplementedError

    def _caps(self, df):
        caps = [self.capacitances(row) for _, row in df.iterrows()]
        return np.array([c[0] for c in caps], float), np.array([c[1] for c in caps], float)

    def add_qubit_H_params(self):
        """Add ``C_sigma_fF``, ``EC``, ``EJ``, ``qubit_frequency_GHz`` and ``anharmonicity_MHz`` columns."""
        self.df = self.add_qubit_H_params_chunk(self.df)

    def add_qubit_H_params_chunk(self, df):
        EJ_target = self.EJ(self.target_params["qubit_frequency_GHz"], self.target_params["anharmonicity_MHz"] * 1e-3)
        c_sigma, _ = self._caps(df)
        EC_values = np.array([EC_GHz_from_Cs(c) for c in c_sigma])
        df["C_sigma_fF"] = c_sigma
        df["EC"] = EC_values
        df["EJ"] = EJ_target
        fq_alpha = {
            ec: get_transmon_E01_alpha(float(EJ_target), float(ec))
            for ec in np.unique(np.round(EC_values, _TRANSMON_CACHE_PRECISION))
        }
        rounded = np.round(EC_values, _TRANSMON_CACHE_PRECISION)
        df["qubit_frequency_GHz"] = [fq_alpha[ec][0] for ec in rounded]
        df["anharmonicity_MHz"] = [fq_alpha[ec][1] for ec in rounded]
        return df

    def add_cavity_coupled_H_params(self, num_chunks="auto", Z_0=50):
        """Qubit columns plus ``g_MHz`` to ``readout_coupler`` from ``g_from_cap_matrix`` on the equivalent grounded
        circuit (:math:`C_Q = C_\\Sigma - C_g`, so :math:`C_Q + C_g = C_\\Sigma`)."""
        self.add_qubit_H_params()
        _, c_g = self._caps(self.df)
        res_types = (
            self.df["resonator_type"] if "resonator_type" in self.df else [self.selected_resonator_type] * len(self.df)
        )
        self.df["g_MHz"] = [
            self.g_from_cap_matrix(C=cs - cg, C_c=cg, EJ=ej, f_r=fr, res_type=rt, Z0=Z_0)
            for cs, cg, ej, fr, rt in zip(
                self.df["C_sigma_fF"], c_g, self.df["EJ"], self.df["cavity_frequency_GHz"], res_types
            )
        ]


class FloatingTransmonHamiltonian(_MaxwellTransmonHamiltonian):
    """Floating (two-pad) transmon, e.g. ``TransmonPocket``: :math:`E_C = e^2 / (2 C_\\mathrm{eff})` of the pad-to-pad mode
    with the coupler pads loaded to ground (:func:`floating_transmon_capacitances`); :math:`g` to the readout pad through
    :math:`C_{g,\\mathrm{eff}}`."""

    pads = ("pad_top", "pad_bot")
    couplers = ("readout_pad", "bus_pad")
    readout_coupler = "readout_pad"

    def capacitances(self, row):
        C, names = maxwell_from_row(row, [*self.pads, *self.couplers])
        return floating_transmon_capacitances(C, names, self.pads, self.readout_coupler)


class XmonHamiltonian(_MaxwellTransmonHamiltonian):
    """Grounded cross with every claw: :math:`C_\\Sigma = C_{\\mathrm{cross,cross}}` (Maxwell diagonal
    :math:`= C_{\\mathrm{cross},g} + C_{\\mathrm{cross,claw}} + C_{\\mathrm{cross,bus\\_claw}}`), :math:`E_C = e^2 / (2 C_\\Sigma)`.
    ``TransmonCross`` keeps its own convention :math:`C_{\\mathrm{cross},g} + C_{\\mathrm{cross,claw}}`."""

    island = "cross"
    couplers = ("claw", "bus_claw")
    readout_coupler = "claw"

    def capacitances(self, row):
        C, names = maxwell_from_row(row, [self.island, *self.couplers])
        i = names.index(self.island)
        k = names.index(self.readout_coupler) if self.readout_coupler in names else None
        return float(C[i, i]), (float(-C[i, k]) if k is not None else 0.0)


QUBIT_HAMILTONIANS = {"TransmonPocket": FloatingTransmonHamiltonian, "Xmon": XmonHamiltonian}


def qubit_hamiltonian_class(qubit_name):
    """Hamiltonian class for a qubit component name, or ``None`` for the default ``TransmonCross`` path."""
    return QUBIT_HAMILTONIANS.get(qubit_name) if isinstance(qubit_name, str) else None
