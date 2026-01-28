"""
#!TODO: Generalize the half-wave cavity method usage
"""

import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import jit
from pyEPR.calcs import Convert
from scipy.constants import Planck, e, hbar
from scipy.optimize import brentq
from scqubits.core.transmon import Transmon

from squadds.calcs.qubit import QubitHamiltonian

"""
========================================================
Constants
========================================================
"""
# constants
ϕ0 = hbar / (2 * e)  # Flux quantum (Weber)

# Cache for transmon calculations - significantly speeds up repeated calculations
# Round to 6 decimal places to increase cache hits for similar values
_TRANSMON_CACHE_PRECISION = 6


@lru_cache(maxsize=50000)
def _cached_transmon_E01_alpha(EJ_rounded: float, EC_rounded: float, ncut: int = 30) -> tuple[float, float]:
    """
    Cached calculation of E01 and anharmonicity for a transmon.

    Uses rounded EJ/EC values as cache keys to increase hit rate.
    Cache can hold up to 50,000 unique (EJ, EC) pairs.

    Returns:
        tuple: (E01 in GHz, anharmonicity in MHz)
    """
    transmon = Transmon(EJ=EJ_rounded, EC=EC_rounded, ng=0, ncut=ncut)
    E01 = transmon.E01()
    alpha = transmon.anharmonicity() * 1e3  # MHz
    return E01, alpha


def get_transmon_E01_alpha(EJ: float, EC: float, ncut: int = 30) -> tuple[float, float]:
    """
    Get E01 and anharmonicity with caching.

    Rounds values to increase cache hit rate while maintaining accuracy.
    """
    EJ_rounded = round(EJ, _TRANSMON_CACHE_PRECISION)
    EC_rounded = round(EC, _TRANSMON_CACHE_PRECISION)
    return _cached_transmon_E01_alpha(EJ_rounded, EC_rounded, ncut)


def clear_transmon_cache():
    """Clear the transmon calculation cache."""
    _cached_transmon_E01_alpha.cache_clear()


def get_transmon_cache_info():
    """Get cache statistics."""
    return _cached_transmon_E01_alpha.cache_info()


"""
========================================================
Numba decorated methods
========================================================
"""


@jit(nopython=True)
def Ec_from_Cs(Cs):
    """
    Calculate the charging energy (Ec) in GHz from the capacitance (Cs) in fF.
    """
    Cs_SI = Cs * 1e-15
    Ec_Joules = (e**2) / (2 * Cs_SI)
    Ec_Hz = Ec_Joules / Planck
    Ec_GHz = Ec_Hz * 1e-9
    return Ec_GHz


@jit(nopython=True)
def EC_numba(cross_to_claw, cross_to_ground):
    C_eff_fF = np.abs(cross_to_ground) + np.abs(cross_to_claw)
    EC = Ec_from_Cs(C_eff_fF)
    return EC


@jit(nopython=True)
def EC_numba_vectorized(cross_to_claw_arr, cross_to_ground_arr):
    """Vectorized EC calculation for arrays - much faster than loop."""
    n = len(cross_to_claw_arr)
    result = np.empty(n, dtype=np.float32)
    for i in range(n):
        C_eff_fF = np.abs(cross_to_ground_arr[i]) + np.abs(cross_to_claw_arr[i])
        result[i] = Ec_from_Cs(C_eff_fF)
    return result


@jit(nopython=True)
def g_from_cap_matrix_numba(C, C_c, EJ, f_r, res_type, Z0=50):
    """
    Calculate coupling strength 'g' using the capacitance matrix formalism (numba-accelerated).

    Uses the formula:
        g = (C_g / sqrt(C_Q + C_g)) * sqrt(hbar * omega_r * e^2 / det(C)) * (E_J / (8 * E_C,Q))^(1/4)

    where det(C) = (C_Q + C_g)(C_r + C_g) - C_g^2

    Args:
        - C (float): Qubit self-capacitance to ground, C_Q (in fF).
        - C_c (float): Coupling capacitance, C_g (in fF).
        - EJ (float): Josephson energy (in GHz).
        - f_r (float): Resonator frequency (in GHz).
        - res_type (str): 'half' or 'quarter' wave resonator.
        - Z0 (float): Characteristic impedance (ohms). Default 50.

    Returns:
        - g (float): Coupling strength in MHz.
    """
    # Keep capacitances in fF (as per original convention), convert to F only where needed
    C_Q_fF = np.abs(C)  # Qubit self-capacitance to ground (fF)
    C_g_fF = np.abs(C_c)  # Coupling capacitance (fF)
    # C_Q_fF + C_g_fF  # Total qubit capacitance (fF)

    # Convert to SI units (F) for the formula
    C_Q = C_Q_fF * 1e-15  # F
    C_g = C_g_fF * 1e-15  # F
    C_q_total = C_Q + C_g  # F

    # Angular resonator frequency
    omega_r = 2 * np.pi * f_r * 1e9  # rad/s

    # Resonator type factor: N=2 for half-wave, N=4 for quarter-wave
    res_type_factor = 2 if res_type == "half" else 4 if res_type == "quarter" else 1

    # Resonator capacitance from transmission line model: C_r = pi / (N * omega_r * Z0)
    C_r = np.pi / (res_type_factor * omega_r * Z0)  # F

    # Capacitance matrix determinant: det(C) = (C_Q + C_g)(C_r + C_g) - C_g^2
    det_C = C_q_total * (C_r + C_g) - C_g**2

    # Effective qubit capacitance from the matrix
    C_q_eff = det_C / (C_r + C_g)

    # Charging energy from effective qubit capacitance
    EC = Ec_from_Cs(C_q_eff * 1e15)

    # Coupling strength using capacitance matrix formula
    # g [J] = (C_g / sqrt(C_Sigma)) * sqrt(hbar * omega_r * e^2 / det(C)) * (EJ / (8 * EC))^(1/4)
    # Note: This formula gives g in energy units (Joules), need to divide by hbar to get rad/s
    g_J = (C_g / np.sqrt(C_q_total)) * np.sqrt(hbar * omega_r * e**2 / det_C) * (EJ / (8 * EC)) ** (1 / 4)

    # Convert from Joules to MHz: g_MHz = g_J / hbar / (2*pi) / 1e6
    return (g_J / hbar) * 1e-6 / (2 * np.pi)  # MHz


class TransmonCrossHamiltonian(QubitHamiltonian):
    """
    Class representing the Hamiltonian for a transmon qubit in a cross-coupled configuration.
    """

    def __init__(self, analysis):
        """
        Initialize the TransmonCrossHamiltonian object.

        Attributes:
            - analysis: The analysis object associated with the Hamiltonian.
        """
        import scqubits as scq

        super().__init__(analysis)
        self.selected_resonator_type = analysis.selected_resonator_type
        scq.set_units("GHz")

    def plot_data(self, data_frame):
        """
        Plot the data from the given DataFrame.

        Args:
            - data_frame: The DataFrame containing the data to be plotted.
        """
        data_frame.plot(kind="box", subplots=True, layout=(1, 3), sharex=False, sharey=False)
        plt.show()

    def EC(self, cross_to_claw, cross_to_ground):
        """
        Calculate the charging energy (EC) of the transmon qubit.

        Args:
            - cross_to_claw: Capacitance between the cross and the claw.
            - cross_to_ground: Capacitance between the cross and the ground.

        Returns:
            - EC: The charging energy of the transmon qubit.
        """
        C_eff_fF = abs(cross_to_ground) + abs(cross_to_claw)
        EC = Convert.Ec_from_Cs(C_eff_fF, units_in="fF", units_out="GHz")
        return EC

    def _calculate_target_qubit_params(self, w_q, alpha, Z_0=50):
        """
        Calculate the target qubit parameters (EJ, EC, EJEC, Lj) based on the given qubit frequency (w_q) and anharmonicity (alpha).

        Args:
            - w_q: The qubit frequency.
            - alpha: The anharmonicity of the qubit.

        Returns:
            - EJ: The Josephson energy of the qubit.
            - EC: The charging energy of the qubit.
            - EJEC: The ratio of EJ to EC.
            - Lj: The Josephson inductance of the qubit.
        """
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        EJEC = EJ / EC
        Lj = Convert.Lj_from_Ej(EJ, units_in="GHz", units_out="nH")
        self.EJ = EJ
        self.EC = EC
        self.EJEC = EJEC
        self.Lj = Lj
        return EJ, EC, EJEC, Lj

    def EJ_and_LJ(self, w_q, alpha, *args, **kwargs):
        """
        Calculate the Josephson energy (EJ) and Josephson inductance (Lj) based on the given qubit frequency (w_q) and anharmonicity (alpha).

        Args:
            - w_q: The qubit frequency.
            - alpha: The anharmonicity of the qubit.

        Returns:
            - EJ: The Josephson energy of the qubit.
            - Lj: The Josephson inductance of the qubit.
        """
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        Lj = Convert.Lj_from_Ej(EJ, units_in="GHz", units_out="nH")
        self.EJ = EJ
        self.Lj = Lj
        return EJ, Lj

    def EJ(self, w_q, alpha):
        """
        Calculate the Josephson energy (EJ) based on the given qubit frequency (w_q) and anharmonicity (alpha).

        Args:
            - w_q: The qubit frequency.
            - alpha: The anharmonicity of the qubit.

        Returns:
            - EJ: The Josephson energy of the qubit.
        """
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        self.EJ = EJ
        return EJ

    def calculate_target_quantities(self, f_res, alpha, g, w_q, res_type, Z_0=50):
        """
        Calculate the target quantities (C_q, C_c, EJ, EC, EJ_EC_ratio) based on the given parameters.

        Uses the capacitance matrix formula to solve for C_c numerically:
            g = (C_g / sqrt(C_Sigma)) * sqrt(hbar * omega_r * e^2 / det(C)) * (E_J / (8 * E_C))^(1/4)

        Args:
            - f_res: The resonator frequency (in GHz).
            - alpha: The anharmonicity of the qubit (in GHz).
            - g: The coupling strength between the qubit and the resonator (in MHz).
            - w_q: The qubit frequency (in GHz).
            - res_type: The type of resonator ('half' or 'quarter').
            - Z_0: The characteristic impedance of the resonator (in ohms). Default is 50.

        Returns:
            - C_q: The total capacitance of the qubit (in fF).
            - C_c: The coupling capacitance between the qubit and the resonator (in fF).
            - EJ: The Josephson energy of the qubit (in GHz).
            - EC: The charging energy of the qubit (in GHz).
            - EJ_EC_ratio: The ratio of EJ to EC.
        """
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        C_q_fF = Convert.Cs_from_Ec(EC, units_in="GHz", units_out="fF")  # Total qubit capacitance in fF
        C_Sigma = C_q_fF * 1e-15  # Total qubit capacitance in F

        omega_r = 2 * np.pi * f_res * 1e9  # Angular frequency in rad/s

        # Resonator type factor
        if res_type == "half" or res_type == 2:
            res_type_factor = 2
        elif res_type == "quarter" or res_type == 4:
            res_type_factor = 4
        else:
            raise ValueError("res_type must be 'half', 'quarter', 2, or 4")

        # Resonator capacitance: C_r = pi / (N * omega_r * Z_0)
        C_r = np.pi / (res_type_factor * omega_r * Z_0)  # in F

        # Common factor in the g formula (gives g in Joules)
        common_factor = np.sqrt(hbar * omega_r * e**2) * (EJ / (8 * EC)) ** (1 / 4)

        # Target g in SI units (Joules)
        # g_MHz -> g_Hz -> g_rad/s -> g_J
        g_target_J = g * 1e6 * 2 * np.pi * hbar  # Convert from MHz (linear) to Joules

        def g_residual(C_g):
            """Residual function: g_calculated - g_target = 0"""
            if C_g <= 0 or C_g >= C_Sigma:
                return float("inf")
            # Determinant: det(C) = C_Sigma * (C_r + C_g) - C_g^2
            det_C = C_Sigma * (C_r + C_g) - C_g**2
            if det_C <= 0:
                return float("inf")
            # g_calc is in Joules
            g_calc_J = (C_g / np.sqrt(C_Sigma)) * (common_factor / np.sqrt(det_C))
            return g_calc_J - g_target_J

        # Solve for C_g numerically using Brent's method
        # C_g must be positive and less than C_Sigma
        C_g_min = 1e-18  # Small positive value
        C_g_max = C_Sigma * 0.99  # Less than total capacitance

        try:
            C_c_F = brentq(g_residual, C_g_min, C_g_max, xtol=1e-20)
        except ValueError:
            # If brentq fails, fall back to a reasonable estimate
            # This can happen if the target g is not achievable with the given parameters
            raise ValueError(
                f"Could not find a valid coupling capacitance for target g={g} MHz. "
                f"The target coupling may be outside the achievable range for the given qubit parameters."
            )

        C_c_fF = C_c_F * 1e15  # Convert to fF
        EJ_EC_ratio = EJ / EC

        return C_q_fF, C_c_fF, EJ, EC, EJ_EC_ratio

    def g_and_alpha(self, C, C_c, f_q, EJ, f_r, res_type, Z0=50):
        """
        Calculate the coupling strength (g) and anharmonicity (alpha) based on the given parameters.

        Args:
            - C: The capacitance between the qubit and the ground.
            - C_c: The coupling capacitance between the qubit and the resonator.
            - f_q: The qubit frequency.
            - EJ: The Josephson energy of the qubit.
            - f_r: The resonator frequency.
            - res_type: The type of resonator.
            - Z0: The characteristic impedance of the resonator.

        Returns:
            - g: The coupling strength between the qubit and the resonator.
            - alpha: The anharmonicity of the qubit.
        """
        C, C_c = abs(C) * 1e-15, abs(C_c) * 1e-15
        C_q = C + C_c
        g = self.g_from_cap_matrix(C, C_c, EJ, f_r, res_type, Z0)
        EC = Convert.Ec_from_Cs(C_q, units_in="F", units_out="GHz")
        transmon = Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
        alpha = transmon.anharmonicity() * 1e3  # MHz
        return g, alpha

    def g_alpha_freq(self, C, C_c, EJ, f_r, res_type, Z0=50):
        """
        Calculate the coupling strength, anharmonicity, and transition frequency of a transmon qubit.

        Args:
            - C (float): Total capacitance of the transmon qubit.
            - C_c (float): Coupling capacitance between the transmon qubit and the resonator.
            - EJ (float): Josephson energy of the transmon qubit.
            - f_r (float): Resonator frequency.
            - res_type (str): Type of resonator. Must be either 'half' or 'quarter'.
            - Z0 (float, optional): Characteristic impedance of the transmission line. Defaults to 50.

        Returns:
            - (g, alpha, freq) (tuple): A tuple containing the coupling strength (g), anharmonicity (alpha), and transition frequency (freq).
        """
        import scqubits as scq

        scq.set_units("GHz")
        C_q = C + C_c
        if res_type == "half":
            res_type = 2
        elif res_type == "quarter":
            res_type = 4
        else:
            raise ValueError("res_type must be either 'half' or 'quarter'")
        g = self.g_from_cap_matrix(C, C_c, EJ, f_r, res_type, Z0)
        EC = Convert.Ec_from_Cs(C_q, units_in="fF", units_out="GHz")
        transmon = Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
        alpha = transmon.anharmonicity() * 1e3  # MHz
        freq = transmon.E01()
        return g, alpha, freq

    def g_from_cap_matrix(self, C, C_c, EJ, f_r, res_type, Z0=50):
        """
        Calculate the coupling strength 'g' between a transmon qubit and a resonator
        based on the capacitance matrix formalism.

        Uses the formula:
            g = (C_g / sqrt(C_Q + C_g)) * sqrt(hbar * omega_r * e^2 / det(C)) * (E_J / (8 * E_C,Q))^(1/4)

        where det(C) = (C_Q + C_g)(C_r + C_g) - C_g^2 is the determinant of the
        capacitance matrix:
            C = [[C_Q + C_g,    -C_g    ],
                 [   -C_g,    C_r + C_g]]

        Args:
            - C (float): Qubit self-capacitance to ground, C_Q (in femtofarads, fF).
            - C_c (float): Coupling capacitance between qubit and resonator, C_g (in femtofarads, fF).
            - EJ (float): Josephson energy of the qubit (in GHz).
            - f_r (float): Resonator frequency (in GHz).
            - res_type (str): Type of resonator. Can be 'half' or 'quarter'.
            - Z0 (float, optional): Characteristic impedance of the transmission line (in ohms). Default is 50 ohms.

        Returns:
            - g (float): Coupling strength 'g' between the qubit and the resonator (in MHz).
        """
        # Convert capacitances from fF to F
        C_Q = abs(C) * 1e-15  # Qubit self-capacitance to ground (F)
        C_g = abs(C_c) * 1e-15  # Coupling capacitance (F)

        # Total qubit capacitance
        C_q_total = C_Q + C_g

        # Angular resonator frequency
        omega_r = 2 * np.pi * f_r * 1e9  # rad/s

        # Resonator capacitance derived from transmission line model
        # C_r = pi / (N * omega_r * Z0) where N=2 for half-wave, N=4 for quarter-wave
        if res_type == "half" or res_type == 2:
            res_type_factor = 2
        elif res_type == "quarter" or res_type == 4:
            res_type_factor = 4
        else:
            raise ValueError("res_type must be 'half', 'quarter', 2, or 4")

        C_r = np.pi / (res_type_factor * omega_r * Z0)  # Resonator capacitance (F)

        # Capacitance matrix determinant: det(C) = (C_Q + C_g)(C_r + C_g) - C_g^2
        det_C = (C_q_total) * (C_r + C_g) - C_g**2

        # Effective qubit capacitance from the matrix
        C_q_eff = det_C / (C_r + C_g)

        # Charging energy from effective qubit capacitance
        EC = Convert.Ec_from_Cs(C_q_eff, units_in="F", units_out="GHz")

        # Coupling strength using capacitance matrix formula
        # g [J] = (C_g / sqrt(C_Q + C_g)) * sqrt(hbar * omega_r * e^2 / det(C)) * (E_J / (8 * E_C,Q))^(1/4)
        # Note: This formula gives g in energy units (Joules), need to divide by hbar to get rad/s
        g_J = (C_g / np.sqrt(C_q_total)) * np.sqrt(hbar * omega_r * e**2 / det_C) * (EJ / (8 * EC)) ** (1 / 4)

        # Convert from Joules to MHz: g_MHz = g_J / hbar / (2*pi) / 1e6
        return (g_J / hbar) * 1e-6 / (2 * np.pi)  # MHz

    def get_freq_alpha_fixed_LJ(self, fig4_df, LJ_target):
        """
        Calculate the frequencies and anharmonicities of transmons with a fixed LJ value.

        Args:
            - fig4_df: DataFrame containing the values of EC for different transmons.
            - LJ_target: Target value of LJ in nH.

        Returns:
            - freq: List of frequencies of the transmons.
            - alpha: List of anharmonicities of the transmons.
        """
        EJ = Convert.Ej_from_Lj(LJ_target, units_in="nH", units_out="GHz")
        EC = fig4_df["EC"].values
        transmons = [Transmon(EJ=EJ, EC=EC[i], ng=0, ncut=30) for i in range(len(EC))]
        alpha = [transmon.anharmonicity() * 1e3 for transmon in transmons]
        freq = [transmon.E01() for transmon in transmons]
        return freq, alpha

    def E01_and_anharmonicity(self, EJ, EC, ng=0, ncut=30):
        """
        Calculate the energy of the first excited state (E01) and the anharmonicity (alpha) of a transmon qubit.

        Uses cached calculations when ng=0 for significant speedup on large datasets.

        Args:
            - EJ (float): Josephson energy of the transmon qubit.
            - EC (float): Charging energy of the transmon qubit.
            - ng (float, optional): Offset charge on the transmon qubit. Defaults to 0.
            - ncut (int, optional): Truncation level for the transmon qubit's Hilbert space. Defaults to 30.

        Returns:
            - E01 (float): Energy of the first excited state (E01) in GHz.
            - alpha (float): Anharmonicity (alpha) in MHz.
        """
        # Use cached version for ng=0 (the common case)
        if ng == 0:
            return get_transmon_E01_alpha(EJ, EC, ncut)

        # Fall back to direct calculation for non-zero ng
        transmon = Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
        E01 = transmon.E01()
        alpha = transmon.anharmonicity() * 1e3  # MHz
        return E01, alpha

    def E01(self, EJ, EC, ng=0, ncut=30):
        """
        Calculate the energy of the first excited state (E01) of a transmon qubit.

        Uses cached calculations when ng=0 for significant speedup.

        Args:
            - EJ (float): Josephson energy of the transmon qubit.
            - EC (float): Charging energy of the transmon qubit.
            - ng (float, optional): Offset charge on the transmon qubit. Default is 0.
            - ncut (int, optional): Truncation level for the transmon qubit's Hilbert space. Default is 30.

        Returns:
            - E01 (float): Energy of the first excited state (E01) of the transmon qubit.
        """
        # Use cached version for ng=0
        if ng == 0:
            E01, _ = get_transmon_E01_alpha(EJ, EC, ncut)
            return E01

        transmon = Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
        E01 = transmon.E01()
        return E01

    def add_qubit_H_params(self):
        """
        Add qubit Hamiltonian parameters to the DataFrame.

        This method calculates and adds the qubit Hamiltonian parameters, such as EC, EJ, and EJEC,
        to the DataFrame.

        Args:
            None

        Returns:
            None
        """
        EJ_target = self.EJ(self.target_params["qubit_frequency_GHz"], self.target_params["anharmonicity_MHz"] * 1e-3)
        self.df["EC"] = self.df.apply(lambda row: self.EC(row["cross_to_claw"], row["cross_to_ground"]), axis=1)
        self.df["EJ"] = EJ_target
        self.df["qubit_frequency_GHz"], self.df["anharmonicity_MHz"] = zip(
            *self.df.apply(lambda row: self.E01_and_anharmonicity(row["EJ"], row["EC"]), axis=1)
        )

    def add_qubit_H_params_chunk(self, df):
        """
        Add qubit Hamiltonian parameters to the DataFrame chunk.

        This method calculates and adds the qubit Hamiltonian parameters, such as EC, EJ, and EJEC.

        Optimization: Since EJ is constant for all rows, we only compute E01/anharmonicity
        for unique EC values (rounded to reduce count), then map results back. This avoids
        redundant Transmon instantiations and matrix diagonalizations.

        Args:
            - df: The DataFrame chunk to which the parameters will be added.

        Returns:
            - df: The DataFrame chunk with the added parameters
        """
        EJ_target = self.EJ(self.target_params["qubit_frequency_GHz"], self.target_params["anharmonicity_MHz"] * 1e-3)
        EJ_target = np.float32(EJ_target)

        cross_to_claw_values = df["cross_to_claw"].values.astype(np.float64)
        cross_to_ground_values = df["cross_to_ground"].values.astype(np.float64)
        EC_values = EC_numba_vectorized(cross_to_claw_values, cross_to_ground_values)

        df["EC"] = EC_values
        df["EJ"] = EJ_target

        # Optimization: compute only for unique EC values, then map back
        # Round EC to reduce unique count while maintaining accuracy
        EC_rounded = np.round(EC_values, _TRANSMON_CACHE_PRECISION)
        unique_ECs = np.unique(EC_rounded)

        # Pre-compute E01 and alpha for all unique EC values
        unique_results = {}
        for ec in unique_ECs:
            E01, alpha = get_transmon_E01_alpha(float(EJ_target), float(ec))
            unique_results[ec] = (E01, alpha)

        # Map results back to full array
        freq_values = np.array([unique_results[ec][0] for ec in EC_rounded], dtype=np.float32)
        alpha_values = np.array([unique_results[ec][1] for ec in EC_rounded], dtype=np.float32)

        df["qubit_frequency_GHz"] = freq_values
        df["anharmonicity_MHz"] = alpha_values

        return df

    def add_cavity_coupled_H_params(self, num_chunks="auto", Z_0=50):
        """
        Add cavity-coupled Hamiltonian parameters to the dataframe.

        This method calculates the coupling strength 'g_MHz' between the transmon qubit and the cavity,
        based on the capacitance matrix, transmon parameters, cavity frequency, resonator type, and characteristic impedance.

        Optimization: Pre-computes all unique transmon parameters in the main process before
        parallel processing, avoiding redundant matrix diagonalizations across workers.

        Args:
            - num_chunks: The number of chunks to split the DataFrame into for parallel processing. Default is "auto" which sets the number of chunks to the number of logical CPUs.
            - Z_0: The characteristic impedance of the transmission line. Default is 50 ohms.

        Returns:
            None
        """
        if self.selected_resonator_type == "half":
            if num_chunks == "auto":
                num_chunks = os.cpu_count() or 4
            elif num_chunks > (os.cpu_count() or 4):
                raise ValueError(f"num_chunks must be less than or equal to {os.cpu_count() or 4}")
            else:
                num_chunks = 2
                raise UserWarning("`num_chunk`s must be an integer greater than 0. Defaulting to 2.")
            print(f"Using {num_chunks} chunks for parallel processing")

            # === OPTIMIZATION: Pre-compute all transmon values in main process ===
            # Step 1: Calculate EJ (constant for all rows)
            EJ_target = self.EJ(
                self.target_params["qubit_frequency_GHz"], self.target_params["anharmonicity_MHz"] * 1e-3
            )
            EJ_target = float(EJ_target)

            # Step 2: Calculate all EC values (vectorized, fast)
            cross_to_claw = self.df["cross_to_claw"].values.astype(np.float64)
            cross_to_ground = self.df["cross_to_ground"].values.astype(np.float64)
            EC_all = EC_numba_vectorized(cross_to_claw, cross_to_ground)

            # Step 3: Find unique EC values (rounded for efficiency)
            EC_rounded = np.round(EC_all, _TRANSMON_CACHE_PRECISION)
            unique_ECs = np.unique(EC_rounded)

            # Step 4: Pre-compute E01 and alpha for all unique ECs
            print(f"Computing transmon params for {len(unique_ECs)} unique EC values...", flush=True)
            unique_E01 = np.empty(len(unique_ECs), dtype=np.float32)
            unique_alpha = np.empty(len(unique_ECs), dtype=np.float32)
            for i, ec in enumerate(unique_ECs):
                E01, alpha = get_transmon_E01_alpha(EJ_target, float(ec))
                unique_E01[i] = E01
                unique_alpha[i] = alpha
            print(f"Pre-computed {len(unique_ECs)} unique transmon states")

            # Step 5: Create lookup arrays using numpy searchsorted for O(log n) lookup
            # This is MUCH faster than a Python loop over 16.5M values
            indices = np.searchsorted(unique_ECs, EC_rounded)

            # Step 6: Vectorized mapping (fast numpy indexing)
            freq_values = unique_E01[indices]
            alpha_values = unique_alpha[indices]

            # Step 7: Store in dataframe
            self.df["EC"] = EC_all
            self.df["EJ"] = np.float32(EJ_target)
            self.df["qubit_frequency_GHz"] = freq_values
            self.df["anharmonicity_MHz"] = alpha_values

            # Step 8: Now parallel process only the g calculation
            self.df = self.parallel_process_g_only(self.df, num_chunks, Z_0)
        else:
            self.add_qubit_H_params()
            self.df["g_MHz"] = self.df.apply(
                lambda row: self.g_from_cap_matrix(
                    C=row["cross_to_ground"],
                    C_c=row["cross_to_claw"],
                    EJ=row["EJ"],
                    f_r=row["cavity_frequency_GHz"],
                    res_type=row["resonator_type"],
                    Z0=50,
                ),
                axis=1,
            )

    def add_cavity_coupled_H_params_chunk(self, chunk, Z_0=50):
        """
        Add cavity-coupled Hamiltonian parameters to the DataFrame chunk.

        This method calculates the coupling strength 'g_MHz' between the transmon qubit and the cavity,

        Args:
            - chunk: The DataFrame chunk to which the parameters will be added.
            - Z_0: The characteristic impedance of the transmission line. Default is 50 ohms.

        Returns:
            - chunk: The DataFrame chunk with the added parameters.
        """
        chunk = self.add_qubit_H_params_chunk(chunk)

        cross_to_ground_values = chunk["cross_to_ground"].values
        cross_to_claw_values = chunk["cross_to_claw"].values
        EJ_values = chunk["EJ"].values
        cavity_frequency_values = chunk["cavity_frequency_GHz"].values

        g_values = np.array(
            [
                g_from_cap_matrix_numba(ground, claw, ej, freq, "half", Z_0)
                for ground, claw, ej, freq in zip(
                    cross_to_ground_values, cross_to_claw_values, EJ_values, cavity_frequency_values
                )
            ],
            dtype=np.float32,
        )

        chunk["g_MHz"] = g_values

        return chunk

    def parallel_process_dataframe(self, df, num_chunks, Z_0=50):
        """
        Process the DataFrame in parallel.

        This method splits the DataFrame into chunks and processes each chunk in parallel.

        Args:
            - df: The DataFrame to be processed.
            - num_chunks: The number of chunks to split the DataFrame into.
            - Z_0: The characteristic impedance of the transmission line. Default is 50

        Returns:
            - df: The DataFrame with the added parameters.

        """
        chunks = np.array_split(df, num_chunks)

        with Parallel(n_jobs=num_chunks) as parallel:
            results = parallel(delayed(self.add_cavity_coupled_H_params_chunk)(chunk, Z_0) for chunk in chunks)

        return pd.concat(results)

    def _calculate_g_chunk(self, chunk, Z_0=50):
        """Calculate g values for a chunk (EC, EJ, freq, alpha already computed)."""
        cross_to_ground_values = chunk["cross_to_ground"].values
        cross_to_claw_values = chunk["cross_to_claw"].values
        EJ_values = chunk["EJ"].values
        cavity_frequency_values = chunk["cavity_frequency_GHz"].values

        g_values = np.array(
            [
                g_from_cap_matrix_numba(ground, claw, ej, freq, "half", Z_0)
                for ground, claw, ej, freq in zip(
                    cross_to_ground_values, cross_to_claw_values, EJ_values, cavity_frequency_values
                )
            ],
            dtype=np.float32,
        )
        chunk["g_MHz"] = g_values
        return chunk

    def parallel_process_g_only(self, df, num_chunks, Z_0=50):
        """
        Parallel process only the g calculation (qubit params already computed).

        This optimized method is used when EC, EJ, qubit_frequency_GHz, and
        anharmonicity_MHz are already computed in the main process.
        """
        chunks = np.array_split(df, num_chunks)

        try:
            with Parallel(n_jobs=num_chunks) as parallel:
                results = parallel(delayed(self._calculate_g_chunk)(chunk, Z_0) for chunk in chunks)
            return pd.concat(results)
        except Exception as e:
            print(f"Parallel processing failed with error: {e}. Falling back to serial execution.")
            return self._calculate_g_chunk(df, Z_0)

    def chi(self, EJ, EC, g, f_r):
        """
        Calculate the full cavity frequency shift between |0> and |1> states of a qubit using g, f_r, f_q, and alpha. It uses the result derived using 2nd-order perturbation theory (equation 9 in SQuaDDS paper).
        Args:
            - EJ (float): Josephson energy of the transmon qubit.
            - EC (float): Charging energy of the transmon qubit.
            - g (float): The coupling strength between the qubit and the cavity.
            - f_r (float): The resonant frequency of the cavity.
            - f_q (float): The frequency spacing between the first two qubit levels.

        Returns:
            - chi (float): The full dispersive shift of the cavity
        """
        #!TODO: Speed up this calculation using numba
        omega_r = 2 * np.pi * f_r * 1e9
        transmon = Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
        alpha = transmon.anharmonicity() * 1e3  # MHz, linear or angular
        omega_q = transmon.E01()  # is this in linear or angular frequency units
        delta = omega_r - omega_q
        sigma = omega_r + omega_q

        chi = 2 * g**2 * (alpha / (delta * (delta - alpha)) - alpha / (sigma * (sigma + alpha)))

        return chi
