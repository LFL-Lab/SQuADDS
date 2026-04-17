"""Pure dataframe enrichment helpers used by the Analyzer compatibility facade."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fix_cavity_claw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and rescale cavity-claw columns using the legacy Analyzer behavior."""
    if ("cavity_frequency" in df.columns) or ("kappa" in df.columns):
        df = df.rename(columns={"cavity_frequency": "cavity_frequency_GHz", "kappa": "kappa_kHz"})
        df["cavity_frequency_GHz"] = df["cavity_frequency_GHz"] * 1e-9
        df["kappa_kHz"] = df["kappa_kHz"] * 1e-3
        try:
            df.drop(columns=["units"], inplace=True)
        except Exception:
            pass
    return df


def extract_qubit_options(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Extract qubit geometry arrays from Analyzer result rows."""
    qubit_options_dict = {
        "claw_gap": [],
        "claw_length": [],
        "claw_width": [],
        "ground_spacing": [],
        "cross_gap": [],
        "cross_length": [],
        "cross_width": [],
    }

    for idx, row in df.iterrows():
        try:
            design_options = row.get("design_options", None)
            if design_options is None:
                raise ValueError(f"Row {idx} has no 'design_options'.")
            qubit_options = design_options.get("qubit_options", {})

            claw_gap = qubit_options.get("connection_pads", {}).get("readout", {}).get("claw_gap")
            claw_length = qubit_options.get("connection_pads", {}).get("readout", {}).get("claw_length")
            claw_width = qubit_options.get("connection_pads", {}).get("readout", {}).get("claw_width")
            ground_spacing = qubit_options.get("connection_pads", {}).get("readout", {}).get("ground_spacing")
            cross_gap = qubit_options.get("cross_gap")
            cross_length = qubit_options.get("cross_length")
            cross_width = qubit_options.get("cross_width")

            if None in [claw_gap, claw_length, claw_width, ground_spacing, cross_gap, cross_length, cross_width]:
                raise ValueError(f"Row {idx} has missing qubit parameter(s).")

            qubit_options_dict["claw_gap"].append(claw_gap)
            qubit_options_dict["claw_length"].append(claw_length)
            qubit_options_dict["claw_width"].append(claw_width)
            qubit_options_dict["ground_spacing"].append(ground_spacing)
            qubit_options_dict["cross_gap"].append(cross_gap)
            qubit_options_dict["cross_length"].append(cross_length)
            qubit_options_dict["cross_width"].append(cross_width)
        except Exception as error:
            print(f"Error processing row {idx}: {str(error)}")
            for key in qubit_options_dict:
                qubit_options_dict[key].append(None)

    return {key: np.array(value) for key, value in qubit_options_dict.items()}


def extract_cpw_options(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Extract CPW geometry arrays from Analyzer result rows."""
    cpw_options_dict = {"total_length": [], "trace_gap": [], "trace_width": []}

    for idx, row in df.iterrows():
        try:
            design_options = row.get("design_options", None)
            if design_options is None:
                raise ValueError(f"Row {idx} has no 'design_options'.")
            cpw_opts = design_options.get("cavity_claw_options", {}).get("cpw_opts", {}).get("left_options", {})

            total_length = cpw_opts.get("total_length")
            trace_gap = cpw_opts.get("trace_gap")
            trace_width = cpw_opts.get("trace_width")

            if None in [total_length, trace_gap, trace_width]:
                raise ValueError(f"Row {idx} has missing CPW parameter(s).")

            cpw_options_dict["total_length"].append(total_length)
            cpw_options_dict["trace_gap"].append(trace_gap)
            cpw_options_dict["trace_width"].append(trace_width)
        except Exception as error:
            print(f"Error processing row {idx}: {str(error)}")
            for key in cpw_options_dict:
                cpw_options_dict[key].append(None)

    return {key: np.array(value) for key, value in cpw_options_dict.items()}


def extract_coupler_options(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Extract coupler geometry arrays from Analyzer result rows."""
    coupler_options_dict = {
        "coupling_length": [],
        "coupling_space": [],
        "down_length": [],
        "orientation": [],
        "prime_gap": [],
        "prime_width": [],
        "second_gap": [],
        "second_width": [],
        "cap_distance": [],
        "cap_gap": [],
        "cap_gap_ground": [],
        "cap_width": [],
        "finger_count": [],
        "finger_length": [],
    }

    for idx, row in df.iterrows():
        try:
            design_options = row.get("design_options", None)
            if design_options is None:
                raise ValueError(f"Row {idx} has no 'design_options'.")
            cavity_claw_options = design_options.get("cavity_claw_options", {})
            coupler_type = cavity_claw_options.get("coupler_type")

            if coupler_type == "CLT":
                coupler_options = cavity_claw_options.get("coupler_options", {})
                extracted_options = {
                    "coupling_length": coupler_options.get("coupling_length"),
                    "coupling_space": coupler_options.get("coupling_space"),
                    "down_length": coupler_options.get("down_length"),
                    "orientation": coupler_options.get("orientation"),
                    "prime_gap": coupler_options.get("prime_gap"),
                    "prime_width": coupler_options.get("prime_width"),
                    "second_gap": coupler_options.get("second_gap"),
                    "second_width": coupler_options.get("second_width"),
                }
            elif coupler_type in ["NCap", "CapNInterdigital"]:
                coupler_options = cavity_claw_options.get("coupler_options", {})
                extracted_options = {
                    "cap_distance": coupler_options.get("cap_distance"),
                    "cap_gap": coupler_options.get("cap_gap"),
                    "cap_gap_ground": coupler_options.get("cap_gap_ground"),
                    "cap_width": coupler_options.get("cap_width"),
                    "finger_count": coupler_options.get("finger_count"),
                    "finger_length": coupler_options.get("finger_length"),
                    "orientation": coupler_options.get("orientation"),
                }
            else:
                raise ValueError(f"Row {idx} has an unsupported coupler_type: {coupler_type}")

            for key in extracted_options:
                coupler_options_dict[key].append(extracted_options.get(key))
        except Exception as error:
            print(f"Error processing row {idx}: {str(error)}")
            for key in coupler_options_dict:
                coupler_options_dict[key].append(None)

    return {key: np.array(value) for key, value in coupler_options_dict.items()}
