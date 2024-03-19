import numpy as np
import pandas as pd


def update_ncap_parameters(cavity_df, ncap_df, merger_terms, ncap_sim_cols):
    """
    Updates the kappa and frequency of the cavity based on the results of the NCap simulations
    Args:
        cavity_df (pd.DataFrame): the dataframe containing the cavity parameters
        ncap_df (pd.DataFrame): the dataframe containing the NCap simulation results
        merger_terms (list): the list of terms to use for merging the dataframes
        ncap_sim_cols (list): the list of columns to use from the NCap dataframe
    Returns:
        pd.DataFrame: the updated cavity dataframe
    """
    for term in merger_terms:
        cavity_df[f'temp_{term}'] = cavity_df['design_options'].apply(lambda x: x['cplr_opts'].get(term))
        ncap_df[f'temp_{term}'] = ncap_df['design_options'].apply(lambda x: x.get(term))
    
    # Perform the merge on the specific common terms
    merged_df = pd.merge(cavity_df, ncap_df, 
                         left_on=[f'temp_{term}' for term in merger_terms], 
                         right_on=[f'temp_{term}' for term in merger_terms],
                         suffixes=('_cavity_claw', '_ncap'))

    # Now, select only the subset of cavity_df and add additional columns from ncap_df
    merged_df = merged_df[[col for col in cavity_df.columns if col not in ['design_options']] + ncap_sim_cols]

    # TODO: Update the cavity resonator frequency and kappa

    # TODO: Update the coupler ending to the CPW

    # Drop the temporary columns used for merging
    merged_df.drop(columns=[f'temp_{term}' for term in merger_terms], inplace=True)

    # TODO: Drop the columns from the NCap dataframe that are not needed

    return merged_df

