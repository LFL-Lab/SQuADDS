from abc import ABC, abstractmethod


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
        """
        Plot the given data frame.

        Args:
            data_frame (pandas.DataFrame): The data frame to be plotted.
        """
        pass

    @abstractmethod
    def E01(self, EJ, EC):
        """
        Calculate the energy difference between the ground state (|0>) and the first excited state (|1>) of a qubit.

        Args:
            - EJ (float): Josephson energy of the qubit.
            - EC (float): Charging energy of the qubit.

        Returns:
            - float: Energy difference between the ground state and the first excited state.
        """
        pass

    @abstractmethod
    def E01_and_anharmonicity(self, EJ, EC):
        """
        Calculate the energy of the first excited state (E01) and the anharmonicity of a qubit.

        Args:
            - EJ (float): Josephson energy of the qubit.
            - EC (float): Charging energy of the qubit.

        Returns:
            - E01 (float): Energy of the first excited state.
            - anharmonicity (float): Anharmonicity of the qubit.
        """
        pass

    @abstractmethod
    def EJ_and_LJ(self, f_q, alpha):
        """
        Calculate the Josephson energy (EJ) and the Josephson inductance (LJ) 
        for a given qubit frequency (f_q) and anharmonicity (alpha).

        Args:
            f_q (float): The qubit frequency in Hz.
            alpha (float): The anharmonicity of the qubit in Hz.

        Returns:
            tuple: A tuple containing the Josephson energy (EJ) and the Josephson inductance (LJ).
        """
        pass

    @abstractmethod
    def EJ(self, f_q, alpha):
        """
        Calculate the Josephson energy (EJ) of a qubit.

        Args:
            f_q (float): The qubit frequency in Hz.
            alpha (float): The anharmonicity parameter.

        Returns:
            float: The Josephson energy (EJ) in Joules.
        """
        pass

    @abstractmethod
    def EC(self, cross_to_claw, cross_to_ground):
        """
        Performs error correction on the qubit.

        Args:
            cross_to_claw (float): The cross-talk probability from the qubit to the claw.
            cross_to_ground (float): The cross-talk probability from the qubit to the ground.

        Returns:
            None
        """
        pass

    @abstractmethod
    def calculate_target_quantities(self, f_res, alpha, g, w_q, N, Z_0=50):
        """
        Calculate the target quantities for a qubit.

        Args:
            - f_res (float): The resonance frequency of the qubit.
            - alpha (float): The anharmonicity of the qubit.
            - g (float): The coupling strength between the qubit and the resonator.
            - w_q (float): The frequency of the qubit.
            - N (int): The number of photons in the resonator.
            - Z_0 (float, optional): The characteristic impedance of the resonator. Default is 50 Ohms.

        Returns:
            None
        """
        pass

    @abstractmethod
    def g_and_alpha(self, C, C_c, f_q, EJ, f_r, res_type, Z0=50):
        """
        Calculate the coupling strength (g) and anharmonicity (alpha) of a qubit.

        Args:
            - C (float): Capacitance of the qubit.
            - C_c (float): Coupling capacitance.
            - f_q (float): Frequency of the qubit.
            - EJ (float): Josephson energy of the qubit.
            - f_r (float): Resonator frequency.
            - res_type (str): Type of resonator.
            - Z0 (float, optional): Characteristic impedance of the transmission line. Default is 50 Ohms.

        Returns:
            - g (float): Coupling strength of the qubit.
            - alpha (float): Anharmonicity of the qubit.
        """
        pass

    @abstractmethod
    def g_alpha_freq(self, C, C_c, EJ, f_r, res_type, Z0=50):
        """
        Calculate the alpha and frequency of a qubit.

        Args:
            - C (float): Capacitance of the qubit.
            - C_c (float): Coupling capacitance.
            - EJ (float): Josephson energy.
            - f_r (float): Resonator frequency.
            - res_type (str): Resonator type.
            - Z0 (float, optional): Characteristic impedance. Default is 50.

        Returns:
            - alpha (float): Qubit anharmonicity.
            - freq (float): Qubit frequency.
        """
        pass

    @abstractmethod
    def get_freq_alpha_fixed_LJ(self, fig4_df, LJ_target):
        """
        Calculate the frequency alpha for a fixed LJ value.

        Args:
            fig4_df (DataFrame): The DataFrame containing the data for Fig. 4.
            LJ_target (float): The target LJ value.

        Returns:
            float: The calculated frequency alpha.
        """
        pass

    @abstractmethod
    def g_from_cap_matrix(self, C, C_c, EJ, f_r, res_type, Z0=50):
        """
        Calculate the coupling strength 'g' of a qubit from the capacitance matrix.

        Args:
            - C (numpy.ndarray): The capacitance matrix.
            - C_c (float): The coupling capacitance.
            - EJ (float): The Josephson energy.
            - f_r (float): The resonant frequency.
            - res_type (str): The type of resonator.
            - Z0 (float, optional): The characteristic impedance. Default is 50 Ohms.

        Returns:
        - g (float): The coupling strength of the qubit.

        """
        pass

    @abstractmethod
    def add_qubit_H_params(self):
        """
        Add parameters for the Hamiltonian of the qubit.
        """
        pass

    @abstractmethod
    def add_cavity_coupled_H_params(self):
        """
        Add parameters for cavity-coupled Hamiltonian.
        """
        pass

    @abstractmethod
    def chi(self):
        """
        Calculate the chi parameter for the qubit.
        """
        pass