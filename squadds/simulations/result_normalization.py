"""Pure helpers for shaping simulation outputs into SQuADDS records."""

from __future__ import annotations


def _require_mapping(name, payload, required_keys):
    if not payload:
        raise ValueError(f"{name} is required and must include {', '.join(required_keys)}.")

    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing_keys)}.")

    return payload


def _normalize_frequency_to_ghz(value, unit=None):
    """Normalize raw frequency values to GHz while tolerating legacy unit metadata."""
    normalized_unit = str(unit).lower() if unit is not None else None
    if normalized_unit == "mhz":
        return value / 1e3
    if normalized_unit == "khz":
        return value / 1e6
    if normalized_unit == "hz":
        return value / 1e9
    if normalized_unit == "ghz" and abs(value) < 1e6:
        return value
    if abs(value) >= 1e6:
        return value / 1e9
    return value


def _normalize_linewidth_to_khz(value, unit=None):
    """Normalize raw linewidth values to kHz while tolerating legacy unit metadata."""
    normalized_unit = str(unit).lower() if unit is not None else None
    if normalized_unit == "ghz":
        return value * 1e6
    if normalized_unit == "mhz":
        return value * 1e3
    if normalized_unit == "hz":
        return value / 1e3
    if normalized_unit == "khz" and abs(value) < 1e4:
        return value
    if abs(value) >= 1e4:
        return value / 1e3
    return value


def normalize_simulation_results(
    emode_df=None,
    lom_df=None,
    ncap_lom_df=None,
    *,
    find_g_a_fq_fn,
    find_kappa_fn,
):
    """Normalize raw eigenmode/LOM results into the legacy combined Hamiltonian payload."""
    if ncap_lom_df is None:
        ncap_lom_df = {}
    if lom_df is None:
        lom_df = {}
    if emode_df is None:
        emode_df = {}
    if emode_df == {} and lom_df == {}:
        print("No simulation results available.")
        return None

    emode_sim_results = _require_mapping(
        "emode_df['sim_results']",
        _require_mapping("emode_df", emode_df, ["sim_results"])["sim_results"],
        ["cavity_frequency", "kappa", "Q"],
    )
    lom_sim_results = _require_mapping(
        "lom_df['sim_results']",
        _require_mapping("lom_df", lom_df, ["sim_results", "design"])["sim_results"],
        ["cross_to_claw", "cross_to_ground"],
    )
    lom_design = _require_mapping("lom_df['design']", lom_df["design"], ["design_options"])
    lom_design_options = _require_mapping(
        "lom_df['design']['design_options']",
        lom_design["design_options"],
        ["aedt_q3d_inductance"],
    )

    cross2cpw = abs(lom_sim_results["cross_to_claw"]) * 1e-15
    cross2ground = abs(lom_sim_results["cross_to_ground"]) * 1e-15
    raw_frequency = emode_sim_results["cavity_frequency"]
    Lj = max(lom_design_options["aedt_q3d_inductance"], 1e-9)
    N = 2 if ncap_lom_df != {} else 4
    gg, aa, ff_q = find_g_a_fq_fn(cross2cpw, cross2ground, raw_frequency, Lj, N=N)
    raw_kappa = emode_sim_results["kappa"]
    Q = emode_sim_results["Q"]
    if ncap_lom_df != {}:
        ncap_sim_results = _require_mapping("ncap_lom_df", ncap_lom_df, ["sim_results"])["sim_results"]
        ncap_sim_results = _require_mapping(
            "ncap_lom_df['sim_results']",
            ncap_sim_results,
            ["C_top2ground", "C_top2bottom"],
        )
        raw_frequency, raw_kappa = find_kappa_fn(
            emode_sim_results["cavity_frequency"],
            ncap_sim_results["C_top2ground"],
            ncap_sim_results["C_top2bottom"],
        )

    return dict(
        cavity_frequency_GHz=_normalize_frequency_to_ghz(
            raw_frequency,
            emode_sim_results.get("cavity_frequency_unit"),
        ),
        Q=Q,
        kappa_kHz=_normalize_linewidth_to_khz(
            raw_kappa,
            emode_sim_results.get("kappa_unit"),
        ),
        g_MHz=gg,
        anharmonicity_MHz=aa,
        qubit_frequency_GHz=ff_q,
    )


def build_eigenmode_payload(coupler_type, geometry_dict, setup, renderer_options, f_rough, Q, kappa, data):
    """Build the legacy eigenmode result payload."""
    return {
        "design": {"coupler_type": coupler_type, "design_options": geometry_dict, "design_tool": "Qiskit Metal"},
        "sim_options": {
            "sim_type": "epr",
            "setup": setup,
            "renderer_options": renderer_options,
            "simulator": "Ansys HFSS",
        },
        "sim_results": {
            "cavity_frequency": f_rough,
            "cavity_frequency_unit": "GHz",
            "Q": Q,
            "kappa": kappa,
            "kappa_unit": "kHz",
        },
        "misc": data,
    }


def build_ncap_lom_payload(param, setup, cap_df, data, coupler_name):
    """Build the legacy NCap lumped-element payload."""
    return {
        "design": {"coupler_type": "NCap", "design_options": param, "design_tool": "Qiskit Metal"},
        "sim_options": {"sim_type": "lom", "setup": setup, "simulator": "Ansys HFSS"},
        "sim_results": {
            "C_top2top": abs(cap_df[f"cap_body_0_{coupler_name}"].values[0]),
            "C_top2bottom": abs(cap_df[f"cap_body_0_{coupler_name}"].values[1]),
            "C_top2ground": abs(cap_df[f"cap_body_0_{coupler_name}"].values[2]),
            "C_bottom2bottom": abs(cap_df[f"cap_body_1_{coupler_name}"].values[1]),
            "C_bottom2ground": abs(cap_df[f"cap_body_1_{coupler_name}"].values[2]),
            "C_ground2ground": abs(cap_df["ground_main_plane"].values[2]),
        },
        "misc": data,
    }


def build_xmon_lom_payload(design_options, setup, renderer_options, cap_df, qname, cname):
    """Build the legacy Xmon lumped-element payload."""
    return {
        "design": {"design_options": design_options, "design_tool": "Qiskit Metal"},
        "sim_options": {
            "sim_type": "lom",
            "setup": setup,
            "renderer_options": renderer_options,
            "simulator": "Ansys HFSS",
        },
        "sim_results": {
            "cross_to_ground": 0
            if "ground_main_plane" not in cap_df.loc[f"cross_{qname}"]
            else abs(cap_df.loc[f"cross_{qname}"]["ground_main_plane"]),
            "claw_to_ground": 0
            if "ground_main_plane" not in cap_df.loc[f"{cname}_connector_arm_{qname}"]
            else abs(cap_df.loc[f"{cname}_connector_arm_{qname}"]["ground_main_plane"]),
            "cross_to_claw": abs(cap_df.loc[f"cross_{qname}"][f"{cname}_connector_arm_{qname}"]),
            "cross_to_cross": abs(cap_df.loc[f"cross_{qname}"][f"cross_{qname}"]),
            "claw_to_claw": abs(cap_df.loc[f"{cname}_connector_arm_{qname}"][f"{cname}_connector_arm_{qname}"]),
            "ground_to_ground": 0
            if "ground_main_plane" not in cap_df.loc[f"cross_{qname}"]
            else abs(cap_df.loc["ground_main_plane"]["ground_main_plane"]),
            "units": "fF",
        },
    }
