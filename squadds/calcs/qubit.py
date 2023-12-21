from abc import ABC, abstractmethod
import pandas as pd

class QubitHamiltonian(ABC):
    #TODO: make method names more general
    def __init__(self, analysis):
        self.analysis = analysis
        self.db = self.analysis.db
        self.df = self.analysis.df
        self.qubit_type = self.db.selected_qubit
        self.cavity_type = self.db.selected_cavity
        self.target_param_keys = self.analysis.H_param_keys
        self.target_params = self.analysis.target_params
        self.f_q = None
        self.alpha = None
        self.Lj = None
        self.EJEC = None
        self.g = None

    @abstractmethod
    def plot_data(self, data_frame):
        pass

    @abstractmethod
    def E01(self, EJ, EC):
        pass

    @abstractmethod
    def E01_and_anharmonicity(self, EJ, EC):
        pass

    @abstractmethod
    def EJ_and_LJ(self, f_q, alpha):
        pass
    
    @abstractmethod
    def EJ(self, f_q, alpha):
        pass
    
    @abstractmethod
    def EC(self, cross_to_claw, cross_to_ground):
        pass

    @abstractmethod
    def calculate_target_quantities(self, f_res, alpha, g, w_q, N, Z_0=50):
        pass

    @abstractmethod
    def g_and_alpha(self, C, C_c, f_q, EJ, f_r, res_type, Z0=50):
        pass

    @abstractmethod
    def g_alpha_freq(self, C, C_c, EJ, f_r, res_type, Z0=50):
        pass

    @abstractmethod
    def get_freq_alpha_fixed_LJ(self, fig4_df, LJ_target):
        pass

    @abstractmethod
    def g_from_cap_matrix(self, C, C_c, EJ, f_r, res_type, Z0=50):
        pass

    @abstractmethod
    def add_qubit_H_params(self):
        pass

    @abstractmethod
    def add_cavity_coupled_H_params(self):
        pass

    
