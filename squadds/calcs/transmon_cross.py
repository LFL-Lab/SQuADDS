import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
from squadds.calcs.qubit import QubitHamiltonian
from scqubits.core.transmon import Transmon
from pyEPR.calcs import Convert
from scipy.constants import e, h, hbar
class TransmonCrossHamiltonian(QubitHamiltonian):
    #TODO: make method handling more general
    def __init__(self, analysis):
        super().__init__(analysis)
        scq.set_units("GHz")
        
    def plot_data(self, data_frame):
        data_frame.plot(kind='box', subplots=True, layout=(1, 3), sharex=False, sharey=False)
        plt.show()

    def EC(self, cross_to_claw, cross_to_ground):
        C_eff_fF = abs(cross_to_ground) + abs(cross_to_claw)
        EC = Convert.Ec_from_Cs(C_eff_fF, units_in='fF',units_out='GHz')
        return EC

    def _calculate_target_qubit_params(self, w_q, alpha, Z_0=50):
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        EJEC = EJ / EC
        Lj = Convert.Lj_from_Ej(EJ, units_in='GHz', units_out='nH')
        self.EJ = EJ
        self.EC = EC
        self.EJEC = EJEC
        self.Lj = Lj
        return EJ, EC, EJEC, Lj

    def EJ_and_LJ(self, w_q, alpha, *args, **kwargs):
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        Lj = Convert.Lj_from_Ej(EJ, units_in='GHz', units_out='nH')
        self.EJ = EJ
        self.Lj = Lj
        return EJ, Lj

    def EJ(self, w_q, alpha):
        EJ, EC = Transmon.find_EJ_EC(w_q, alpha)
        self.EJ = EJ
        return EJ

    def calculate_target_quantities(self, f_res, alpha, g, w_q, res_type, Z_0=50):
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
        C, C_c = abs(C) * 1e-15, abs(C_c) * 1e-15
        C_q = C + C_c
        g = self.g_from_cap_matrix(C, C_c, EJ, f_r, res_type, Z0)
        EC = Convert.Ec_from_Cs(C_q, units_in='F', units_out='GHz')
        transmon = Transmon(EJ=EJ, EC=EC, ng=0, ncut=30)
        alpha = transmon.anharmonicity() * 1E3  # MHz
        return g, alpha

    def g_alpha_freq(self, C, C_c, EJ, f_r, res_type, Z0=50):
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
        EJ = Convert.Ej_from_Lj(LJ_target, units_in='nH', units_out='GHz')
        EC = fig4_df["EC"].values
        transmons = [Transmon(EJ=EJ, EC=EC[i], ng=0, ncut=30) for i in range(len(EC))]
        alpha = [transmon.anharmonicity() * 1E3 for transmon in transmons]
        freq = [transmon.E01() for transmon in transmons]
        return freq, alpha

    def E01_and_anharmonicity(self, EJ, EC, ng=0, ncut=30):
        transmon = Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
        E01 = transmon.E01()
        alpha = transmon.anharmonicity() * 1E3  # MHz
        return E01, alpha

    def E01(self, EJ, EC, ng=0, ncut=30):
        transmon = Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
        E01 = transmon.E01()
        return E01

    def add_qubit_H_params(self):
        EJ_target = self.EJ(self.target_params["qubit_frequency_GHz"], self.target_params["anharmonicity_MHz"]*1e-3)

        self.df["EC"] = self.df.apply(lambda row: self.EC(row["cross_to_claw"], row["cross_to_ground"]), axis=1)

        self.df['EJ'] = EJ_target
        self.df["EJEC"] = self.df.apply(lambda row: row["EJ"] / row["EC"], axis=1)
        
        self.df['qubit_frequency_GHz'], self.df['anharmonicity_MHz'] = zip(*self.df.apply(lambda row: self.E01_and_anharmonicity(row['EJ'], row['EC']), axis=1))


    def add_cavity_coupled_H_params(self):
        self.add_qubit_H_params()
        #pprint.pprint(self.df.head())
        #pprint.pprint(self.df.columns)
        self.df['g_MHz'] = self.df.apply(lambda row: self.g_from_cap_matrix(C=row['cross_to_ground'], 
                          C_c=row['cross_to_claw'], 
                          EJ=row['EJ'],
                          f_r=row['cavity_frequency_GHz'],
                          res_type=row['resonator_type'], 
                          Z0=50), axis=1)