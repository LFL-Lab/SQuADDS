"""Two transmons coupled to each other, each read out through a half-wave resonator: exact design from Hamiltonian targets.

Targets (GHz, f = omega / 2 pi) are the parameters of the Hamiltonian

    H = sum_q H_transmon(E_J^q, E_C^q) + sum_r f_r c_r^+ c_r + K n_A n_B + sum_qr G_qr n_q (c_r + c_r^+) + R x_A x_B

``f_q`` and ``eta`` of each exact transmon (charge basis), the bare ``f_r``, ``J = K n01_A n01_B`` with the exact charge
matrix elements, and ``chi`` = half the dispersive shift computed from the full Hamiltonian (6 transmon levels in their
exact eigenbasis with every charge matrix element, 5 photons per resonator, counter-rotating terms kept; no rotating-wave
approximation). The circuit side maps a Maxwell matrix (fF; ground, cross A, cross B, claw A, claw B) plus lambda/2 lines
at the claws to E_C, K, G and R through the full inverse of the node capacitance matrix.

Used by the MCP tool ``solve_coupled_transmon_hamiltonian``.
Only numpy and scipy are needed.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import e, h, hbar
from scipy.optimize import fsolve, least_squares

PHI0 = h / (2 * e)
STATES = {
    "g": (0, 0, 0, 0),
    "A1": (1, 0, 0, 0),
    "A2": (2, 0, 0, 0),
    "B1": (0, 1, 0, 0),
    "B2": (0, 2, 0, 0),
    "rA": (0, 0, 1, 0),
    "rB": (0, 0, 0, 1),
    "A1rA": (1, 0, 1, 0),
    "B1rB": (0, 1, 0, 1),
}
KEYS = ["f_q_A", "f_q_B", "eta_A", "eta_B", "chi_A", "chi_B", "J", "f_r_A", "f_r_B"]
FIVE = [(1, 0), (2, 0), (1, 2), (1, 3), (2, 4)]  # (body, body): cross A-ground, cross B-ground, A-B, A-claw A, B-claw B


# ------------------------------------------------------------------------------------------------ transmons
def transmon(EJ, EC, ng=0.0, ncut=30, levels=6):
    """Exact transmon in the charge basis: level energies (GHz, ground at 0) and the charge operator in its eigenbasis."""
    n = np.arange(-ncut, ncut + 1)
    H = np.diag(4 * EC * (n - ng) ** 2) - EJ / 2 * (np.eye(2 * ncut + 1, k=1) + np.eye(2 * ncut + 1, k=-1))
    E, V = np.linalg.eigh(H)
    V = V[:, :levels]
    return E[:levels] - E[0], V.T @ np.diag(n - ng) @ V


def fit_transmon(f01, eta):
    """E_J, E_C (GHz) of the transmon whose exact f01 and anharmonicity eta = f12 - f01 are the targets."""

    def residual(x):
        E, _ = transmon(*x)
        return [E[1] - f01, E[2] - 2 * E[1] - eta]

    EC0 = -eta
    return least_squares(residual, [(f01 + EC0) ** 2 / (8 * EC0), EC0], xtol=1e-14, ftol=1e-14).x


def transmon_parameters(targets):
    """{"A": (E_J, E_C), "B": (E_J, E_C)} (GHz) of the two transmons, solved exactly from the targets."""
    return {q: fit_transmon(targets[f"f_q_{q}"], targets[f"eta_{q}"]) for q in "AB"}


def charge_matrix_elements(EJ_EC):
    """``|<0|n|1>|`` of the two transmons."""
    return np.array([abs(transmon(*EJ_EC[q])[1][0, 1]) for q in "AB"])


# ------------------------------------------------------------------------------------------------ full Hamiltonian
def kron(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def hamiltonian(c, nq=6, nr=5):
    """Full Hamiltonian (GHz) of two transmons and two resonators; c: EC, EJ, fr (pairs), K, G (2x2) and R (optional)."""
    (EA, nA), (EB, nB) = (transmon(c["EJ"][i], c["EC"][i], levels=nq) for i in range(2))
    a = np.diag(np.sqrt(np.arange(1, nr)), 1)
    x, num, Iq, Ir = a + a.T, a.T @ a, np.eye(nq), np.eye(nr)
    nA, nB = kron(nA, Iq, Ir, Ir), kron(Iq, nB, Ir, Ir)
    xA, xB = kron(Iq, Iq, x, Ir), kron(Iq, Iq, Ir, x)
    H = kron(np.diag(EA), Iq, Ir, Ir) + kron(Iq, np.diag(EB), Ir, Ir)
    H = H + c["fr"][0] * kron(Iq, Iq, num, Ir) + c["fr"][1] * kron(Iq, Iq, Ir, num)
    H = H + c["K"] * nA @ nB + c.get("R", 0.0) * xA @ xB
    G = np.asarray(c["G"])
    H = H + G[0, 0] * nA @ xA + G[1, 1] * nB @ xB + G[0, 1] * nA @ xB + G[1, 0] * nB @ xA
    return H, (nq, nq, nr, nr)


def dressed(c, nq=6, nr=5):
    """Dressed f_q, eta, f_r, chi (half the dispersive shift) of both sides and J = K n01_A n01_B (GHz)."""
    H, dims = hamiltonian(c, nq, nr)
    E, V = np.linalg.eigh(H)
    lev = {k: E[np.argmax(np.abs(V[np.ravel_multi_index(s, dims)]) ** 2)] for k, s in STATES.items()}
    lev = {k: v - lev["g"] for k, v in lev.items()}
    n01 = [abs(transmon(c["EJ"][i], c["EC"][i])[1][0, 1]) for i in range(2)]
    return {
        "f_q_A": lev["A1"],
        "f_q_B": lev["B1"],
        "eta_A": lev["A2"] - 2 * lev["A1"],
        "eta_B": lev["B2"] - 2 * lev["B1"],
        "chi_A": (lev["A1rA"] - lev["A1"] - lev["rA"]) / 2,
        "chi_B": (lev["B1rB"] - lev["B1"] - lev["rB"]) / 2,
        "J": c["K"] * n01[0] * n01[1],
        "f_r_A": lev["rA"],
        "f_r_B": lev["rB"],
        "n01": n01,
    }


# ------------------------------------------------------------------------------------------------ the design
def design_circuit(targets, G, EJ_EC=None):
    """The circuit of the targets' Hamiltonian: exact transmons, bare resonators, K = J / (n01_A n01_B), and G."""
    EJ_EC = EJ_EC or transmon_parameters(targets)
    N01 = charge_matrix_elements(EJ_EC)
    return {
        "EJ": np.array([EJ_EC[q][0] for q in "AB"]),
        "EC": np.array([EJ_EC[q][1] for q in "AB"]),
        "fr": np.array([targets["f_r_A"], targets["f_r_B"]]),
        "K": targets["J"] / N01.prod(),
        "G": np.diag(G),
    }


def rwa_g(targets, q, chi):
    """The textbook (rotating-wave, two-level-like) coupling g for a dispersive shift chi."""
    D, eta = targets[f"f_q_{q}"] - targets[f"f_r_{q}"], targets[f"eta_{q}"]
    return np.sqrt(abs(chi * D * (D + eta) / eta))


def solve_couplings(targets, chi_A, chi_B):
    """G_A, G_B (GHz) for which the full Hamiltonian gives ``|chi_A|`` and ``|chi_B|`` (GHz)."""
    EJ_EC = transmon_parameters(targets)
    N01 = charge_matrix_elements(EJ_EC)

    def residual(G):
        d = dressed(design_circuit(targets, G, EJ_EC))
        return [1e6 * (abs(d["chi_A"]) - chi_A), 1e6 * (abs(d["chi_B"]) - chi_B)]  # kHz

    G0 = np.array([rwa_g(targets, "A", chi_A), rwa_g(targets, "B", chi_B)]) / N01
    return least_squares(residual, G0, x_scale=0.05 * G0, xtol=1e-13, ftol=1e-13).x


def h_params(c):
    """The Hamiltonian parameters of a circuit (GHz); chi from the full Hamiltonian."""
    out, d = {"J": dressed(c)["J"]}, dressed(c)
    for i, q in enumerate("AB"):
        E, _ = transmon(c["EJ"][i], c["EC"][i])
        out |= {f"f_q_{q}": E[1], f"eta_{q}": E[2] - 2 * E[1], f"chi_{q}": d[f"chi_{q}"], f"f_r_{q}": c["fr"][i]}
    return out


def compare(targets, c):
    """Target vs achieved Hamiltonian parameters of a circuit (MHz, magnitudes) with the miss in MHz and %."""
    import pandas as pd

    p = h_params(c)
    df = pd.DataFrame({k: {"target (MHz)": 1e3 * abs(targets[k]), "achieved (MHz)": 1e3 * abs(p[k])} for k in KEYS}).T
    df["miss (MHz)"] = df["achieved (MHz)"] - df["target (MHz)"]
    df["miss (%)"] = 100 * df["miss (MHz)"] / df["target (MHz)"]
    return df


def design_table(c):
    """E_J, E_C, E_J/E_C, L_J, I_c, n01, G, g = G n01 and ``|chi|`` of both transmons, K and J (a DataFrame)."""
    import pandas as pd

    rows, d = {}, dressed(c)
    for i, q in enumerate("AB"):
        E, n = transmon(c["EJ"][i], c["EC"][i])
        EJ_joule = h * 1e9 * c["EJ"][i]
        rows[q] = {
            "E_J (GHz)": c["EJ"][i],
            "E_C (MHz)": 1e3 * c["EC"][i],
            "E_J/E_C": c["EJ"][i] / c["EC"][i],
            "L_J (nH)": 1e9 * (PHI0 / (2 * np.pi)) ** 2 / EJ_joule,
            "I_c (nA)": 1e9 * 2 * np.pi * EJ_joule / PHI0,
            "|<0|n|1>|": abs(n[0, 1]),
            "G (MHz)": 1e3 * np.asarray(c["G"])[i, i],
            "g = G n01 (MHz)": 1e3 * np.asarray(c["G"])[i, i] * abs(n[0, 1]),
            "|chi| (MHz)": 1e3 * abs(d[f"chi_{q}"]),
        }
    out = pd.DataFrame(rows)
    out.loc["K (MHz)"] = 1e3 * c["K"]
    out.loc["J = K n01_A n01_B (MHz)"] = 1e3 * d["J"]
    return out


# ------------------------------------------------------------------------------------------------ circuit <-> capacitances
def line_capacitance(f_line, Z0=50.0):
    """Lumped capacitance (F) of a lambda/2 line with fundamental f_line (GHz) near resonance."""
    return np.pi / (2 * 2 * np.pi * np.asarray(f_line) * 1e9 * Z0)


def circuit(M, fr_bare, Z0=50.0):
    """Circuit parameters (GHz) of a Maxwell matrix M (fF; ground, cross A, cross B, claw A, claw B) whose claws end
    lambda/2 lines that resonate (loaded) at fr_bare (GHz)."""
    Cs, fr = np.asarray(M, float)[1:, 1:] * 1e-15, np.asarray(fr_bare, float)
    f_line = fr.copy()
    for _ in range(60):
        C = Cs + np.diag([0, 0, *line_capacitance(f_line, Z0)])
        Ci = np.linalg.inv(C)
        L = 1 / ((2 * np.pi * f_line * 1e9) ** 2 * line_capacitance(f_line, Z0))
        f_line = f_line * fr / (np.sqrt(np.diag(Ci)[2:] / L) / (2 * np.pi * 1e9))
    Qz = np.sqrt(hbar * 2 * np.pi * fr * 1e9 / (2 * np.diag(Ci)[2:]))
    G = np.array([[2 * e * Ci[i, 2 + j] * Qz[j] for j in range(2)] for i in range(2)]) / h / 1e9
    return {
        "EC": e**2 * np.diag(Ci)[:2] / (2 * h * 1e9),
        "K": 4 * e**2 * Ci[0, 1] / h / 1e9,
        "G": G,
        "R": Ci[2, 3] * Qz[0] * Qz[1] / h / 1e9,
        "fr": fr,
        "f_line": f_line,
        "C_line_fF": 1e15 * line_capacitance(f_line, Z0),
        "C_node_fF": 1e15 / np.diag(Ci)[2:],
    }


def partials(M):
    """Partial capacitances of a Maxwell matrix: P[i, j] = -M[i, j] between any two bodies (the ground included) and
    P[i, i] = row sum (body i to infinity)."""
    M = np.asarray(M, float)
    P = -M.copy()
    np.fill_diagonal(P, M.sum(1))
    return P


def maxwell_from(P):
    """The Maxwell matrix of partial capacitances P (the inverse of ``partials``)."""
    M = -np.array(P, float)
    np.fill_diagonal(M, np.sum(P, 1))
    return M


def goal(c):
    """The five circuit numbers a layout controls: E_C^A, E_C^B, K, G_A, G_B (GHz)."""
    return np.array([c["EC"][0], c["EC"][1], c["K"], c["G"][0, 0], c["G"][1, 1]])


def circuit_numbers(M, c):
    """E_C^A, E_C^B, K, G_A, G_B (GHz) of a Maxwell matrix with the resonators of circuit c."""
    k = circuit(M, c["fr"])
    return np.array([k["EC"][0], k["EC"][1], k["K"], k["G"][0, 0], k["G"][1, 1]])


def target_capacitances(M_ref, c):
    """The Maxwell matrix that hits ``goal(c)``, with every other partial capacitance (claw to ground, parasitics) from
    M_ref."""
    P_ref, GOAL = partials(M_ref), goal(c)

    def with_five(x):
        P = P_ref.copy()
        for (i, j), v in zip(FIVE, x):
            P[i, j] = P[j, i] = v
        return maxwell_from(P)

    x = fsolve(lambda x: circuit_numbers(with_five(x), c) / GOAL - 1, [P_ref[i, j] for i, j in FIVE], xtol=1e-12)
    return with_five(x)


def final_circuit(targets, M):
    """The circuit of a Maxwell matrix, with E_J chosen so that each transmon's f01 is on target."""
    EJ_EC = transmon_parameters(targets)
    c = circuit(M, [targets["f_r_A"], targets["f_r_B"]])
    EJ = []
    for i, q in enumerate("AB"):
        fit = least_squares(
            lambda x, i=i, q=q: [transmon(x[0], c["EC"][i])[0][1] - targets[f"f_q_{q}"]], [EJ_EC[q][0]], xtol=1e-14
        )
        EJ.append(fit.x[0])
    return c | {"EJ": np.array(EJ)}


def ideal_reference(c):
    """A starting Maxwell matrix (fF) for ``target_capacitances`` when no layout exists yet: only the five controlled
    partial capacitances (rough circuit estimates), no claw-to-ground and no parasitic couplings."""
    cs = e**2 / (2 * h * 1e9 * np.asarray(c["EC"])) * 1e15
    P = np.zeros((5, 5))
    for (i, j), v in zip(FIVE, [0.85 * cs[0], 0.85 * cs[1], 2.0, 0.1 * cs[0], 0.1 * cs[1]]):
        P[i, j] = P[j, i] = v
    return maxwell_from(P)


__all__ = [
    "FIVE",
    "KEYS",
    "PHI0",
    "STATES",
    "charge_matrix_elements",
    "circuit",
    "circuit_numbers",
    "compare",
    "design_circuit",
    "design_table",
    "dressed",
    "final_circuit",
    "fit_transmon",
    "goal",
    "hamiltonian",
    "h_params",
    "ideal_reference",
    "kron",
    "line_capacitance",
    "maxwell_from",
    "partials",
    "rwa_g",
    "solve_couplings",
    "target_capacitances",
    "transmon",
    "transmon_parameters",
]
