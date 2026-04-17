import pandas as pd

from squadds.simulations.result_normalization import (
    build_eigenmode_payload,
    build_ncap_lom_payload,
    build_xmon_lom_payload,
    normalize_simulation_results,
)


def test_normalize_simulation_results_combines_eigenmode_and_lom_data():
    emode_df = {
        "sim_results": {
            "cavity_frequency": 7.1,
            "cavity_frequency_unit": "GHz",
            "kappa": 140.0,
            "kappa_unit": "kHz",
            "Q": 1.2e5,
        }
    }
    lom_df = {
        "sim_results": {"cross_to_claw": 10.0, "cross_to_ground": 20.0},
        "design": {"design_options": {"aedt_q3d_inductance": 2.0}},
    }

    result = normalize_simulation_results(
        emode_df=emode_df,
        lom_df=lom_df,
        find_g_a_fq_fn=lambda cross2cpw, cross2ground, f_r, Lj, N: (55.0, -210.0, 4.95),
        find_kappa_fn=lambda *args: (0.0, 0.0),
    )

    assert result == {
        "cavity_frequency_GHz": 7.1,
        "Q": 1.2e5,
        "kappa_kHz": 140.0,
        "g_MHz": 55.0,
        "anharmonicity_MHz": -210.0,
        "qubit_frequency_GHz": 4.95,
    }


def test_normalize_simulation_results_updates_ncap_frequency_and_kappa():
    emode_df = {
        "sim_results": {
            "cavity_frequency": 7.1,
            "cavity_frequency_unit": "GHz",
            "kappa": 140.0,
            "kappa_unit": "kHz",
            "Q": 1.2e5,
        }
    }
    lom_df = {
        "sim_results": {"cross_to_claw": 10.0, "cross_to_ground": 20.0},
        "design": {"design_options": {"aedt_q3d_inductance": 0.5}},
    }
    ncap_lom_df = {"sim_results": {"C_top2ground": 1.0, "C_top2bottom": 2.0}}

    result = normalize_simulation_results(
        emode_df=emode_df,
        lom_df=lom_df,
        ncap_lom_df=ncap_lom_df,
        find_g_a_fq_fn=lambda cross2cpw, cross2ground, f_r, Lj, N: (60.0, -220.0, 5.05),
        find_kappa_fn=lambda cavity_frequency, c_top2ground, c_top2bottom: (6.9, 180.0),
    )

    assert result["cavity_frequency_GHz"] == 6.9
    assert result["kappa_kHz"] == 180.0
    assert result["g_MHz"] == 60.0


def test_normalize_simulation_results_converts_hz_payloads_to_ghz_and_khz():
    emode_df = {
        "sim_results": {
            "cavity_frequency": 7.1e9,
            "cavity_frequency_unit": "Hz",
            "kappa": 140e3,
            "kappa_unit": "Hz",
            "Q": 1.2e5,
        }
    }
    lom_df = {
        "sim_results": {"cross_to_claw": 10.0, "cross_to_ground": 20.0},
        "design": {"design_options": {"aedt_q3d_inductance": 2.0}},
    }

    result = normalize_simulation_results(
        emode_df=emode_df,
        lom_df=lom_df,
        find_g_a_fq_fn=lambda cross2cpw, cross2ground, f_r, Lj, N: (55.0, -210.0, 4.95),
        find_kappa_fn=lambda *args: (0.0, 0.0),
    )

    assert result["cavity_frequency_GHz"] == 7.1
    assert result["kappa_kHz"] == 140.0


def test_normalize_simulation_results_handles_legacy_large_values_even_when_units_are_mislabeled():
    emode_df = {
        "sim_results": {
            "cavity_frequency": 7.1e9,
            "cavity_frequency_unit": "GHz",
            "kappa": 140e3,
            "kappa_unit": "kHz",
            "Q": 1.2e5,
        }
    }
    lom_df = {
        "sim_results": {"cross_to_claw": 10.0, "cross_to_ground": 20.0},
        "design": {"design_options": {"aedt_q3d_inductance": 2.0}},
    }

    result = normalize_simulation_results(
        emode_df=emode_df,
        lom_df=lom_df,
        find_g_a_fq_fn=lambda cross2cpw, cross2ground, f_r, Lj, N: (55.0, -210.0, 4.95),
        find_kappa_fn=lambda *args: (0.0, 0.0),
    )

    assert result["cavity_frequency_GHz"] == 7.1
    assert result["kappa_kHz"] == 140.0


def test_build_eigenmode_payload_matches_legacy_shape():
    payload = build_eigenmode_payload(
        "CLT",
        {"foo": "bar"},
        {"name": "setup"},
        {"mesh": "fine"},
        7.1,
        120000,
        140.0,
        {"raw": 1},
    )

    assert payload["design"]["coupler_type"] == "CLT"
    assert payload["sim_options"]["renderer_options"] == {"mesh": "fine"}
    assert payload["sim_results"]["cavity_frequency_unit"] == "GHz"
    assert payload["sim_results"]["kappa_unit"] == "kHz"


def test_build_ncap_lom_payload_reads_capacitance_matrix_by_legacy_labels():
    cap_df = pd.DataFrame(
        {
            "cap_body_0_test": [1.0, 2.0, 3.0],
            "cap_body_1_test": [4.0, 5.0, 6.0],
            "ground_main_plane": [7.0, 8.0, 9.0],
        }
    )

    payload = build_ncap_lom_payload({"finger_count": 6}, {"name": "setup"}, cap_df, {"raw": 1}, "test")

    assert payload["sim_results"] == {
        "C_top2top": 1.0,
        "C_top2bottom": 2.0,
        "C_top2ground": 3.0,
        "C_bottom2bottom": 5.0,
        "C_bottom2ground": 6.0,
        "C_ground2ground": 9.0,
    }


def test_build_xmon_lom_payload_handles_ground_plane_fields():
    cap_df = pd.DataFrame(
        {
            "cross_xmon": [11.0, 12.0, 13.0],
            "readout_connector_arm_xmon": [21.0, 22.0, 23.0],
            "ground_main_plane": [31.0, 32.0, 33.0],
        },
        index=["cross_xmon", "readout_connector_arm_xmon", "ground_main_plane"],
    )

    payload = build_xmon_lom_payload(
        {"cross_length": "200um"},
        {"name": "setup"},
        {"mesh": "fine"},
        cap_df,
        "xmon",
        "readout",
    )

    assert payload["sim_results"]["cross_to_ground"] == 31.0
    assert payload["sim_results"]["claw_to_ground"] == 32.0
    assert payload["sim_results"]["cross_to_claw"] == 21.0
    assert payload["sim_results"]["cross_to_cross"] == 11.0
    assert payload["sim_results"]["claw_to_claw"] == 22.0
    assert payload["sim_results"]["ground_to_ground"] == 33.0
    assert payload["sim_results"]["units"] == "fF"
