from abc import ABC, abstractmethod
import pandas as pd

        """
        Abstract method to calculate EJ.

        Parameters:
        - f_q: The qubit frequency.
        - alpha: The anharmonicity.
        """
        pass
    
    @abstractmethod
    def EC(self, cross_to_claw, cross_to_ground):
        """
        Abstract method to calculate EC.

        Parameters:
        - cross_to_claw: The cross-to-claw capacitance.
        - cross_to_ground: The cross-to-ground capacitance.
        """
        pass

    @abstractmethod
    def calculate_target_quantities(self, f_res, alpha, g, w_q, N, Z_0=50):
        """
        Abstract method to calculate target quantities.

        Parameters:
        - f_res: The resonator frequency.
        - alpha: The anharmonicity.
        - g: The coupling strength.
        - w_q: The qubit frequency.
        - N: The number of photons.
        - Z_0: The characteristic impedance. (default: 50)
        """
        pass

    @abstractmethod
    def g_and_alpha(self, C, C_c, f_q, EJ, f_r, res_type, Z0=50):
        """
        Abstract method to calculate g and alpha.

        Parameters:
        - C: The capacitance.
        - C_c: The coupling capacitance.
        - f_q: The qubit frequency.
        - EJ: The Josephson energy.
        - f_r: The resonator frequency.
        - res_type: The resonator type.
        - Z0: The characteristic impedance. (default: 50)
        """
        pass

    @abstractmethod
    def g_alpha_freq(self, C, C_c, EJ, f_r, res_type, Z0=50):
        """
        Abstract method to calculate g, alpha, and frequency.

        Parameters:
        - C: The capacitance.
        - C_c: The coupling capacitance.
        - EJ: The Josephson energy.
        - f_r: The resonator frequency.
        - res_type: The resonator type.
        - Z0: The characteristic impedance. (default: 50)
        """
        pass

    @abstractmethod
    def get_freq_alpha_fixed_LJ(self, fig4_df, LJ_target):
        """
        Abstract method to get frequency and alpha for a fixed LJ.

        Parameters:
        - fig4_df: The figure 4 data frame.
        - LJ_target: The target LJ value.
        """
        pass

    @abstractmethod
    def g_from_cap_matrix(self, C, C_c, EJ, f_r, res_type, Z0=50):
        """
        Abstract method to calculate g from capacitance matrix.

        Parameters:
        - C: The capacitance.
        - C_c: The coupling capacitance.
        - EJ: The Josephson energy.
        - f_r: The resonator frequency.
        - res_type: The resonator type.
        - Z0: The characteristic impedance. (default: 50)
        """
        pass

    @abstractmethod
    def add_qubit_H_params(self):
        """
        Abstract method to add qubit Hamiltonian parameters.
        """
        pass

    @abstractmethod
    def add_cavity_coupled_H_params(self):
        """
        Abstract method to add cavity-coupled Hamiltonian parameters.
        """
        pass