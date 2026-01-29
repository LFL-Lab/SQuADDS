import os
import time

import pandas as pd
from pyEPR.calcs import Convert

from squadds.core.utils import create_unified_design_options


def get_design_from_ml_predictions(analyzer, test_data, y_pred_dnn):
    """
    Generate design options DataFrame using ML predictions.

    Args:
        analyzer (Analyzer): An instance of the Analyzer class.
        test_data (pd.DataFrame): DataFrame with target parameters.
        y_pred_dnn (numpy.ndarray): Array with predicted design parameters.

    Returns:
        pd.DataFrame: DataFrame with design options similar to interpolated_designs_df.
    """

    designs_list = []

    for i, row in test_data.iterrows():
        # Get target parameters
        f_q_target = row["qubit_frequency_GHz"]
        f_res_target = row["cavity_frequency_GHz"]
        alpha_target = row["anharmonicity_MHz"]
        kappa_target = row["kappa_kHz"]
        g_target = row["g_MHz"]

        # Get the corresponding predicted design parameters
        predicted_design_params = y_pred_dnn[i]
        cross_length_pred = predicted_design_params[0]
        claw_length_pred = predicted_design_params[1]
        coupling_length_pred = predicted_design_params[2]
        total_length_pred = predicted_design_params[3]
        ground_spacing_pred = predicted_design_params[4]

        # Now find the closest designs
        # Find the closest qubit design
        closest_qubit_claw_design = analyzer.find_closest(
            {"qubit_frequency_GHz": f_q_target, "anharmonicity_MHz": alpha_target, "g_MHz": g_target}, num_top=1
        )

        # Find the closest cavity design
        target_params_cavity = {"cavity_frequency_GHz": f_res_target, "kappa_kHz": kappa_target}
        closest_cavity_cpw_design = analyzer.find_closest(target_params_cavity, num_top=1)

        # Now update the design options
        # Get the design options from the closest designs
        qubit_design_options = closest_qubit_claw_design["design_options_qubit"].iloc[0].copy()
        cavity_design_options = closest_cavity_cpw_design["design_options_cavity_claw"].iloc[0].copy()

        # Update the qubit design options with predicted values
        qubit_design_options["cross_length"] = f"{cross_length_pred}um"
        qubit_design_options["connection_pads"]["readout"]["claw_length"] = f"{claw_length_pred}um"
        qubit_design_options["connection_pads"]["readout"]["ground_spacing"] = f"{ground_spacing_pred}um"

        # Update the 'Lj' parameters
        required_Lj = Convert.Lj_from_Ej(closest_qubit_claw_design["EJ"].iloc[0], units_in="GHz", units_out="nH")
        qubit_design_options["aedt_hfss_inductance"] = required_Lj * 1e-9
        qubit_design_options["aedt_q3d_inductance"] = required_Lj * 1e-9
        qubit_design_options["q3d_inductance"] = required_Lj * 1e-9
        qubit_design_options["hfss_inductance"] = required_Lj * 1e-9
        qubit_design_options["connection_pads"]["readout"]["Lj"] = f"{required_Lj}nH"

        # Set 'claw_cpw_*' parameters to zero
        qubit_design_options["connection_pads"]["readout"]["claw_cpw_length"] = "0um"
        # qubit_design_options["connection_pads"]['readout']['claw_cpw_width'] = "0um"

        # Update the cavity design options with predicted values
        cavity_design_options["cpw_opts"]["total_length"] = f"{total_length_pred}um"
        cavity_design_options["cplr_opts"]["coupling_length"] = f"{coupling_length_pred}um"

        # Update the claw of the cavity based on the one from the qubit
        cavity_design_options["claw_opts"]["connection_pads"] = qubit_design_options["connection_pads"]

        # Create the combined design options
        device_dict = {
            "coupler_type": analyzer.selected_coupler,
            "design_options_qubit": qubit_design_options,
            "design_options_cavity_claw": cavity_design_options,
            "setup_qubit": closest_qubit_claw_design["setup_qubit"].iloc[0],
            "setup_cavity_claw": closest_cavity_cpw_design["setup_cavity_claw"].iloc[0],
        }

        device_design_options = create_unified_design_options(device_dict)

        # Add the device design options to the dictionary
        device_dict["design_options"] = device_design_options
        device_dict["design_options"]["qubit_options"]["connection_pads"]["readout"]["claw_cpw_length"] = "0um"

        designs_list.append(device_dict)

    # Now create the final DataFrame
    designs_df = pd.DataFrame(designs_list)

    return designs_df


def process_dataframe(analyzer, target_params_df, index, path_to_file):
    """
    Processes the dataframe by merging, filtering columns, and converting specific string columns to floats.

    Args:
        analyzer (Analyzer): An instance of the Analyzer class.
        target_params_df (pd.DataFrame): A dataframe containing target parameters.
        path_to_file (str): The file path where the processed dataframe will be saved.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Merge with coupling data
    merged_df = analyzer.get_complete_df(target_params_df.iloc[index])

    # Define columns to keep
    columns_to_keep = [
        "cross_length",
        "cross_gap",
        "claw_length",
        "ground_spacing",
        "coupler_type",
        "resonator_type",
        "cavity_frequency_GHz",
        "kappa_kHz",
        "EC",
        "EJ",
        "EJEC",
        "qubit_frequency_GHz",
        "anharmonicity_MHz",
        "g_MHz",
        "coupling_length",
        "total_length",
    ]

    # Filter out unwanted columns
    new_df = merged_df.drop(columns=[col for col in merged_df.columns if col not in columns_to_keep])

    # Convert specific columns from string to float
    float_col_names = ["claw_length"]
    new_df[float_col_names] = new_df[float_col_names].applymap(lambda x: float(x[:-2]))

    # Apply conversions for design options using JSON-like structure
    new_df["cross_length"] = merged_df["design_options"].apply(lambda x: float(x["qubit_options"]["cross_length"][:-2]))
    new_df["cross_gap"] = merged_df["design_options"].apply(lambda x: float(x["qubit_options"]["cross_gap"][:-2]))
    new_df["ground_spacing"] = merged_df["design_options"].apply(
        lambda x: float(x["qubit_options"]["connection_pads"]["readout"]["ground_spacing"][:-2])
    )
    new_df["coupling_length"] = merged_df["design_options"].apply(
        lambda x: float(x["cavity_claw_options"]["coupler_options"]["coupling_length"][:-2])
    )
    new_df["total_length"] = merged_df["design_options"].apply(
        lambda x: float(x["cavity_claw_options"]["cpw_opts"]["left_options"]["total_length"][:-2])
    )

    # Drop the 'coupler_type' and 'resonator_type' columns
    new_df = new_df.drop(columns=["coupler_type", "resonator_type"])

    return new_df


def generate_qubit_cavity_training_data(analyzer, target_params_df, path_to_file):
    """
    Generates training data for qubit and cavity designs based on target parameters.

    Args:
        analyzer (Analyzer): An instance of the Analyzer class.
        target_params_df (pd.DataFrame): A dataframe containing target parameters.
        path_to_file (str): The file path where the processed dataframe will be saved.
    """
    if analyzer.selected_resonator_type == "quarter":
        # for each entry in the target_params_df, process the dataframe and concatenate the results
        processed_dfs = []
        for index in range(len(target_params_df)):
            processed_dfs.append(process_dataframe(analyzer, target_params_df, index, path_to_file))
        # concatenate the results
        training_df = pd.concat(processed_dfs)

        # if the file path is provided, save the processed dataframe if not create training_data foder and save the processed dataframe with unique timestamp
        if path_to_file:
            if path_to_file.endswith(".parquet"):
                training_df.to_parquet(path_to_file)
            elif path_to_file.endswith(".csv"):
                training_df.to_csv(path_to_file)
            else:
                raise ValueError("Please provide a valid file format (.parquet or .csv) for the training data.")
            print(f"Training data saved to {path_to_file}")
        else:
            if not os.path.exists("training_data"):
                os.makedirs("training_data")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            training_df.to_parquet(f"training_data/training_data_{timestamp}.parquet")
            print(f"Training data saved to training_data/training_data_{timestamp}.parquet")
        return training_df

    elif analyzer.selected_resonator_type == "half":
        raise NotImplementedError(
            "Half resonator type data generation has not been implemented yet. Please feel free to contribute!"
        )
