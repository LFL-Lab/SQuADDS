"""
#!TODO: Generalize the half-wave cavity method usage
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from numba import jit, prange
from pyEPR.calcs import Convert
from scipy.constants import Planck, e, h, hbar, pi
from scqubits.core.transmon import Transmon

from squadds.calcs.qubit import QubitHamiltonian

"""
========================================================
Constants
========================================================
"""
# constants
ϕ0 = hbar / (2 * e)   # Flux quantum (Weber)

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
    Ec_Joules = (e ** 2) / (2 * Cs_SI)
    Ec_Hz = Ec_Joules / Planck
    Ec_GHz = Ec_Hz * 1e-9
    return Ec_GHz

@jit(nopython=True)
def EC_numba(cross_to_claw, cross_to_ground):
    C_eff_fF = np.abs(cross_to_ground) + np.abs(cross_to_claw)
    EC = Ec_from_Cs(C_eff_fF)
    return EC

@jit(nopython=True, parallel=True)
def g_from_cap_matrix_numba(C, C_c, EJ, f_r, res_type, Z0=50):
    """
    !TODO: resolve the error : \"The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.\" for this method
    """
    C = np.abs(C)
    C_c = np.abs(C_c)
    C_q = C_c + C # fF

    omega_r = 2 * np.pi * f_r * 1e9

    EC = Ec_from_Cs(C_q)

    res_type_factor = 2 if res_type == "half" else 4 if res_type == "quarter" else 1

    g = (np.abs(C_c) / C_q) * omega_r * np.sqrt(res_type_factor * Z0 * e ** 2 / (hbar * np.pi)) * (EJ / (8 * EC)) ** (1 / 4)
    return (g * 1E-6) / (2 * np.pi)  # MHz


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
        data_frame.plot(kind='box', subplots=True, layout=(1, 3), sharex=False, sharey=False)
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
        EC = Convert.Ec_from_Cs(C_eff_fF, units_in='fF',units_out='GHz')
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
        Lj = Convert.Lj_from_Ej(EJ, units_in='GHz', units_out='nH')
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
        Lj = Convert.Lj_from_Ej(EJ, units_in='GHz', units_out='nH')
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

        Args:
            - f_res: The resonator frequency.
            - alpha: The anharmonicity of the qubit.
            - g: The coupling strength between the qubit and the resonator.
            - w_q: The qubit frequency.
            - res_type: The type of resonator.
            - Z_0: The characteristic impedance of the resonator.

        Returns:
            - C_q: The total capacitance of the qubit.
            - C_c: The coupling capacitance between the qubit and the resonator.
            - EJ: The Josephson energy of the qubit.
            - EC: The charging energy of the qubit.
            - EJ_EC_ratio: The ratio of EJ to EC.
        """
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        C_q = Convert.Cs_from_Ec(EC, units_in='GHz', units_out='fF')
        omega_r = 2 * np.pi * f_res
        if res_type == "half":
            res_type = 2
        elif res_type == "quarter":
            res_type = 4
        else:
            raise ValueError("res_type must be either 'half' or 'quarter'")
        prefactor = np.sqrt(res_type * Z_0 * e**2 / (hbar * np.pi)) * (EJ / (8 * EC))**(1/4)
        denominator = omega_r * prefactor
        numerator = g * C_q
        C_c = numerator / denominator
        EJ_EC_ratio = EJ / EC
        return C_q, C_c, EJ, EC, EJ_EC_ratio

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
        EC = Convert.Ec_from_Cs(C_q, units_in='F', units_out='GHz')
        transmon = Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
        alpha = transmon.anharmonicity() * 1E3  # MHz
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
        EC = Convert.Ec_from_Cs(C_q, units_in='fF', units_out='GHz')
        transmon = Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
        alpha = transmon.anharmonicity() * 1E3  # MHz
        freq = transmon.E01()
        return g, alpha, freq

    def g_from_cap_matrix(self, C, C_c, EJ, f_r, res_type, Z0=50):
        """
        Calculate the coupling strength 'g' between a transmon qubit and a resonator
        based on the capacitance matrix.

        Args:
            - C (float): Capacitance between the qubit and the resonator (in femtofarads, fF).
            - C_c (float): Coupling capacitance of the resonator (in femtofarads, fF).
            - EJ (float): Josephson energy of the qubit (in GHz).
            - f_r (float): Resonator frequency (in GHz).
            - res_type (str): Type of resonator. Can be 'half' or 'quarter'.
            - Z0 (float, optional): Characteristic impedance of the transmission line (in ohms). Default is 50 ohms.

        Returns:
            - g (float): Coupling strength 'g' between the qubit and the resonator (in MHz).
        """
        C = abs(C) * 1e-15  # F
        C_c = abs(C_c) * 1e-15  # F
        C_q = C_c + C
        omega_r = 2 * np.pi * f_r * 1e9
        EC = Convert.Ec_from_Cs(C_q, units_in='F', units_out='GHz')

        if res_type == "half":
            res_type = 2
        elif res_type == "quarter":
            res_type = 4

        g = (abs(C_c) / C_q) * omega_r * np.sqrt(res_type * Z0 * e**2 / (hbar * np.pi)) * (EJ / (8 * EC))**(1/4)
        return (g * 1E-6) / (2 * np.pi)  # MHz


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
        EJ = Convert.Ej_from_Lj(LJ_target, units_in='nH', units_out='GHz')
        EC = fig4_df["EC"].values
        transmons = [Transmon(EJ=EJ, EC=EC[i], ng=0, ncut=30) for i in range(len(EC))]
        alpha = [transmon.anharmonicity() * 1E3 for transmon in transmons]
        freq = [transmon.E01() for transmon in transmons]
        return freq, alpha

    def E01_and_anharmonicity(self, EJ, EC, ng=0, ncut=30):
        """
        Calculate the energy of the first excited state (E01) and the anharmonicity (alpha) of a transmon qubit.

        Args:
            - EJ (float): Josephson energy of the transmon qubit.
            - EC (float): Charging energy of the transmon qubit.
            - ng (float, optional): Offset charge on the transmon qubit. Defaults to 0.
            - ncut (int, optional): Truncation level for the transmon qubit's Hilbert space. Defaults to 30.

        Returns:
            - E01 (float): Energy of the first excited state (E01) in GHz.
            - alpha (float): Anharmonicity (alpha) in MHz.
        """
        transmon = Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
        E01 = transmon.E01()
        alpha = transmon.anharmonicity() * 1E3  # MHz
        return E01, alpha

    def E01(self, EJ, EC, ng=0, ncut=30):
        """
        Calculate the energy of the first excited state (E01) of a transmon qubit.

        Args:
            - EJ (float): Josephson energy of the transmon qubit.
            - EC (float): Charging energy of the transmon qubit.
            - ng (float, optional): Offset charge on the transmon qubit. Default is 0.
            - ncut (int, optional): Truncation level for the transmon qubit's Hilbert space. Default is 30.

        Returns:
            - E01 (float): Energy of the first excited state (E01) of the transmon qubit.
        """
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
        self.df['EJ'] = EJ_target
        self.df['qubit_frequency_GHz'], self.df['anharmonicity_MHz'] = zip(*self.df.apply(lambda row: self.E01_and_anharmonicity(row['EJ'], row['EC']), axis=1))

    def add_qubit_H_params_chunk(self, df):
        """
        Add qubit Hamiltonian parameters to the DataFrame chunk.

        This method calculates and adds the qubit Hamiltonian parameters, such as EC, EJ, and EJEC,

        Args:
            - df: The DataFrame chunk to which the parameters will be added.

        Returns:
            - df: The DataFrame chunk with the added parameters
        """
        EJ_target = self.EJ(self.target_params["qubit_frequency_GHz"], self.target_params["anharmonicity_MHz"] * 1e-3)
        EJ_target = np.float32(EJ_target)  # Ensure memory-efficient data type

        cross_to_claw_values = df["cross_to_claw"].values
        cross_to_ground_values = df["cross_to_ground"].values
        EC_values = np.array([EC_numba(claw, ground) for claw, ground in zip(cross_to_claw_values, cross_to_ground_values)], dtype=np.float32)

        df["EC"] = EC_values
        df['EJ'] = EJ_target
        df['qubit_frequency_GHz'], df['anharmonicity_MHz'] = np.vectorize(self.E01_and_anharmonicity)(df['EJ'].values, df['EC'].values)
        df['qubit_frequency_GHz'] = df['qubit_frequency_GHz'].astype(np.float32)
        df['anharmonicity_MHz'] = df['anharmonicity_MHz'].astype(np.float32)

        return df


    def add_cavity_coupled_H_params(self, num_chunks="auto",Z_0=50):
        """
        Add cavity-coupled Hamiltonian parameters to the dataframe.

        This method calculates the coupling strength 'g_MHz' between the transmon qubit and the cavity,
        based on the capacitance matrix, transmon parameters, cavity frequency, resonator type, and characteristic impedance.

        Args:
            - num_chunks: The number of chunks to split the DataFrame into for parallel processing. Default is "auto" which sets the number of chunks to the number of logical CPUs.
            - Z_0: The characteristic impedance of the transmission line. Default is 50 ohms.

        Returns:
            None
        """
        if self.selected_resonator_type == "half":
            if num_chunks == "auto":
                num_chunks = psutil.cpu_count(logical=True)
            elif num_chunks > psutil.cpu_count(logical=True):
                raise ValueError(f"num_chunks must be less than or equal to {psutil.cpu_count(logical=True)}")
            else:
                num_chunks = 2
                raise UserWarning("`num_chunk`s must be an integer greater than 0. Defaulting to 2.")
            print(f"Using {num_chunks} chunks for parallel processing")
            self.df = self.parallel_process_dataframe(self.df, num_chunks)
        else:
            self.add_qubit_H_params()
            self.df['g_MHz'] = self.df.apply(lambda row: self.g_from_cap_matrix(C=row['cross_to_ground'], 
                            C_c=row['cross_to_claw'], 
                            EJ=row['EJ'],
                            f_r=row['cavity_frequency_GHz'],
                            res_type=row['resonator_type'], 
                            Z0=50), axis=1)

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

        cross_to_ground_values = chunk['cross_to_ground'].values
        cross_to_claw_values = chunk['cross_to_claw'].values
        EJ_values = chunk['EJ'].values
        cavity_frequency_values = chunk['cavity_frequency_GHz'].values

        g_values = np.array([g_from_cap_matrix_numba(ground, claw, ej, freq, 'half', Z_0)
                            for ground, claw, ej, freq in zip(cross_to_ground_values, cross_to_claw_values, EJ_values, cavity_frequency_values)], dtype=np.float32)

        chunk['g_MHz'] = g_values

        return chunk

    def parallel_process_dataframe(self, df, num_chunks, Z_0 = 50):
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
        alpha = transmon.anharmonicity() * 1E3  # MHz, linear or angular
        omega_q = transmon.E01() # is this in linear or angular frequency units
        delta = omega_r - omega_q
        sigma = omega_r + omega_q

        chi = 2 * g**2 * (alpha /(delta * (delta - alpha))- alpha/(sigma * (sigma + alpha)))

        return chi
