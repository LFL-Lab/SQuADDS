from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


def merge_dfs(qubit_df_split, cavity_df, merger_terms):
    return pd.merge(qubit_df_split, cavity_df, on=merger_terms, how="inner", suffixes=('_qubit', '_cavity_claw'))

def update_ncap_parameters(cavity_df, ncap_df, merger_terms, ncap_sim_cols):
    """
    Updates the kappa and frequency of the cavity based on the results of the CapNInterdigitalTee simulations.
    """
    for term in merger_terms:
        cavity_df[f'temp_{term}'] = cavity_df['design_options'].map(lambda x: x['cplr_opts'].get(term))
        ncap_df[f'temp_{term}'] = ncap_df['design_options'].map(lambda x: x.get(term))
    
    # Perform the merge on the specific common terms
    merged_df = pd.merge(cavity_df, ncap_df, 
                         left_on=[f'temp_{term}' for term in merger_terms], 
                         right_on=[f'temp_{term}' for term in merger_terms],
                         suffixes=('_cavity_claw', '_ncap'))

    # Update the cavity resonator frequency and kappa
    cavity_frequency_updated, kappa = update_cavity_frequency_and_kappa(merged_df)
    merged_df.loc[:, "cavity_frequency"] = cavity_frequency_updated
    merged_df.loc[:, "kappa"] = kappa

    # Update the cavity_df with the ncap parameters
    def update_cpw_cplr_opts(row):
        ncap_opts = ncap_df.iloc[0]["design_options"]
        cpw_cplr_opts = row["design_options_cavity_claw"]["cplr_opts"]
        common_terms = set(ncap_opts.keys()) & set(cpw_cplr_opts.keys())
        for term in common_terms:
            cpw_cplr_opts[term] = ncap_opts[term]
        return cpw_cplr_opts

    merged_df["design_options_cavity_claw"]["cplr_opts"] = merged_df.apply(update_cpw_cplr_opts, axis=1)

    # Remove the temporary columns
    merged_df = merged_df.drop(columns=[f'temp_{term}' for term in merger_terms])
    
    # Remove all the columns with "_ncap" suffix
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith("_ncap")])

    # Remove all the columns with ncap_sim_cols
    merged_df = merged_df.drop(columns=ncap_sim_cols)
    
    # Rename the columns with "_cavity_claw" suffix to remove the suffix
    merged_df = merged_df.rename(columns={col: col.replace("_cavity_claw", "") for col in merged_df.columns})

    return merged_df

def update_cavity_frequency_and_kappa(merged_df, Z0=50):
    """
    Updates the cavity frequency and kappa based on the given merged_df DataFrame.

    Parameters:
    - merged_df: DataFrame containing the necessary simulation results.
    - Z0: Characteristic impedance of the system (default: 50 Ohms).

    Returns:
    - cavity_frequency_updated: Updated cavity frequency in Hz.
    - kappa: Updated kappa in Hz.
    """
    # Constants
    pi = np.pi
    
    # Extract the necessary simulation results from merged_df
    omega_rough = merged_df['cavity_frequency'] * 2 * pi  # Convert from Hz to rad/s

    C_tg = merged_df['top_to_ground']   # fF
    C_tb = merged_df['top_to_bottom']   # fF
    
    # Calculate the resonator capacitance C_res from the rough resonant frequency omega_rough
    C_res = (pi / (2 * omega_rough * Z0) ) * 1e15  # fF
    
    # Compute the new resonant frequency estimate omega_est
    omega_est = np.sqrt(C_res / (C_res + C_tg + C_tb)) * omega_rough
    
    # Calculate kappa (assuming the units for the frequencies are consistent)
    kappa = 0.5 * Z0 * (omega_est**2) * (C_tb**2 / (C_res + C_tg + C_tb)) * 1e-15  # Hz
    
    # Convert omega_est from rad/s back to Hz for cavity_frequency
    cavity_frequency_updated = omega_est / (2 * pi)
    
    # Return the updated cavity frequency and kappa
    return cavity_frequency_updated, kappa / (2 * pi)