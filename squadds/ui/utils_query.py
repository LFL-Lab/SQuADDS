import numpy as np
import streamlit as st

from squadds import Analyzer, SQuADDS_DB


def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(v) for v in obj]
    else:
        return obj


@st.cache_data(show_spinner=False)
def find_closest_cached(
    system_type, qubit_type, cavity_type, resonator_type, params, num_results, num_cpu, skip_df_gen
):
    """
    Cached design search. If skip_df_gen is True, will not regenerate the DataFrame if config is unchanged.
    """
    db = SQuADDS_DB()
    db.unselect_all()
    if system_type == "Qubit-Cavity":
        db.select_system(["qubit", "cavity_claw"])
        db.select_qubit(qubit_type)
        db.select_cavity_claw(cavity_type)
        db.select_resonator_type(resonator_type)
    elif system_type == "Qubit Only":
        db.select_system("qubit")
        db.select_qubit(qubit_type)
    else:  # Cavity Only
        db.select_system("cavity_claw")
        db.select_cavity_claw(cavity_type)
        db.select_resonator_type(resonator_type)
    db.create_system_df()
    analyzer = Analyzer(db)
    if system_type == "Qubit-Cavity" and resonator_type == "half":
        results = analyzer.find_closest(
            target_params=params,
            num_top=num_results,
            metric="Euclidean",
            parallel=True,
            num_cpu=num_cpu,
            skip_df_gen=skip_df_gen,
        )
    else:
        results = analyzer.find_closest(
            target_params=params, num_top=num_results, metric="Euclidean", skip_df_gen=skip_df_gen
        )
    return results, analyzer


# Utility functions for extracting design parameters (can be expanded as needed)
def extract_cavity_only_params(results):
    cpw_keys = ["claw_gap", "claw_length", "claw_width", "ground_spacing", "claw_cpw_length", "claw_cpw_width"]
    coupler_keys = [
        "coupling_length",
        "coupling_space",
        "down_length",
        "orientation",
        "prime_gap",
        "prime_width",
        "second_gap",
        "second_width",
        "cap_distance",
        "cap_gap",
        "cap_gap_ground",
        "cap_width",
        "finger_count",
        "finger_length",
    ]
    cpw_vals = {k: [] for k in cpw_keys}
    coupler_vals = {k: [] for k in coupler_keys}
    for _idx, row in results.iterrows():
        opts = convert_numpy_to_list(row["design_options"])
        # CPW: from claw_opts['connection_pads']['readout']
        claw_opts = opts.get("claw_opts", {})
        readout = claw_opts.get("connection_pads", {}).get("readout", {})
        for k in cpw_keys:
            v = readout.get(k)
            if v is not None:
                cpw_vals[k].append(v)
        # Coupler: from cplr_opts
        cplr_opts = opts.get("cplr_opts", {})
        for k in coupler_keys:
            v = cplr_opts.get(k)
            if v is not None:
                coupler_vals[k].append(v)
    return cpw_vals, coupler_vals
