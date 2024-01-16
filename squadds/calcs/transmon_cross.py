import matplotlib.pyplot as plt
import numpy as np
from pyEPR.calcs import Convert
from scipy.constants import e, h, hbar
from scqubits.core.transmon import Transmon

from squadds.calcs.qubit import QubitHamiltonian


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

        EJ_target = self.EJ(self.target_params["qubit_frequency_GHz"], self.target_params["anharmonicity_MHz"]*1e-3)

        self.df["EC"] = self.df.apply(lambda row: self.EC(row["cross_to_claw"], row["cross_to_ground"]), axis=1)

        self.df['EJ'] = EJ_target
        self.df["EJEC"] = self.df.apply(lambda row: row["EJ"] / row["EC"], axis=1)
        self.df['qubit_frequency_GHz'], self.df['anharmonicity_MHz'] = zip(*self.df.apply(lambda row: self.E01_and_anharmonicity(row['EJ'], row['EC']), axis=1))


    def add_cavity_coupled_H_params(self):
        """
        Add cavity-coupled Hamiltonian parameters to the dataframe.

        This method calculates the coupling strength 'g_MHz' between the transmon qubit and the cavity,
        based on the capacitance matrix, transmon parameters, cavity frequency, resonator type, and characteristic impedance.

        Args:
            None

        Returns:
            None
        """
        self.add_qubit_H_params()
        #pprint.pprint(self.df.head())
        #pprint.pprint(self.df.columns)
        self.df['g_MHz'] = self.df.apply(lambda row: self.g_from_cap_matrix(C=row['cross_to_ground'], 
                          C_c=row['cross_to_claw'], 
                          EJ=row['EJ'],
                          f_r=row['cavity_frequency_GHz'],
                          res_type=row['resonator_type'], 
                          Z0=50), axis=1)