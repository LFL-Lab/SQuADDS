import numpy as np
import pandas as pd


def update_ncap_parameters(cavity_df, ncap_df, merger_terms, ncap_sim_cols):
    """
    Updates the kappa and frequency of the cavity based on the results of the NCap simulations.
    """
    for term in merger_terms:
        cavity_df[f'temp_{term}'] = cavity_df['design_options'].apply(lambda x: x['cplr_opts'].get(term))
        ncap_df[f'temp_{term}'] = ncap_df['design_options'].apply(lambda x: x.get(term))
    
    # Perform the merge on the specific common terms
    merged_df = pd.merge(cavity_df, ncap_df, 
                         left_on=[f'temp_{term}' for term in merger_terms], 
                         right_on=[f'temp_{term}' for term in merger_terms],
                         suffixes=('_cavity_claw', '_ncap'))

    # Update the cavity resonator frequency and kappa
    cavity_df["cavity_frequency"], cavity_df["kappa"] = update_cavity_frequency_and_kappa(merged_df)
    return cavity_df

def update_cavity_frequency_and_kappa(merged_df, Z0=50):
    # Constants
    pi = np.pi
    
    # Extract the necessary simulation results from merged_df
    omega_rough = merged_df['cavity_frequency'] * 2 * pi  # Convert from Hz to rad/s
    # omega_rough = merged_df['cavity_frequency']

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
    # cavity_frequency_updated = omega_est
    
    # Return the updated cavity frequency and kappa
    return cavity_frequency_updated, kappa

# Add the function to update the merged_df (you would place this inside the update_ncap_parameters function)
# merged_df['cavity_frequency'], merged_df['kappa'] = update_cavity_frequency_and_kappa(merged_df)
