import numpy as np
import scqubits as scq


def find_a_fq(C_g, C_B, Lj):
    e = 1.602e-19
    hbar = 1.054e-34
    C_Sigma = C_g + C_B
    EJ = ((hbar / 2 / e) ** 2) / Lj * (1.5092e24)
    EC = e**2 / (2 * C_Sigma) * (1.5092e24)
    transmon = scq.Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
    a = transmon.anharmonicity() * 1000
    f_q = transmon.E01()
    return a, f_q


def find_g_a_fq(C_g, C_B, f_r, Lj, N):
    e = 1.602e-19
    hbar = 1.054e-34
    Z_0 = 50
    C_Q = C_B
    C_Sigma = C_g + C_Q
    omega_r = 2 * np.pi * f_r
    EJ = ((hbar / 2 / e) ** 2) / Lj * (1.5092e24)
    EC = e**2 / (2 * C_Sigma) * (1.5092e24)
    C_r = np.pi / (N * omega_r * Z_0)
    det_C = C_Sigma * (C_r + C_g) - C_g**2
    transmon = scq.Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
    a = transmon.anharmonicity() * 1000
    g_J = (C_g / np.sqrt(C_Sigma)) * np.sqrt(hbar * omega_r * e**2 / det_C) * (EJ / (8 * EC)) ** (1 / 4)
    g = (g_J / hbar) / 1e6 / (2 * np.pi)
    f_q = transmon.E01()
    return g, a, f_q


def find_kappa(f_rough, C_tg, C_tb):
    Z0 = 50
    w_rough = 2 * np.pi * f_rough
    C_res = np.pi / (2 * w_rough * Z0) * 1e15
    w_est = np.sqrt(C_res / (C_res + C_tg + C_tb)) * w_rough
    kappa = (1 / 2 * Z0 * (w_est**2) * (C_tb**2) / (C_res + C_tg + C_tb)) * 1e-15 / (2 * np.pi) * 1e-3
    f_est = w_est / (2 * np.pi)
    return f_est, kappa


def find_chi(alpha, f_q, g, f_r):
    omega_q = 2 * np.pi * f_q * 1e9
    omega_r = 2 * np.pi * f_r * 1e9
    g *= 1e6 * 2 * np.pi
    alpha *= 1e6 * 2 * np.pi
    delta = omega_r - omega_q
    sigma = omega_r + omega_q
    return 2 * g**2 * (alpha / (delta * (delta - alpha)) - alpha / (sigma * (sigma + alpha))) * 1e-6
