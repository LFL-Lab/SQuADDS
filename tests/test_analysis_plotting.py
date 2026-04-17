import matplotlib.pyplot as plt
import pandas as pd
import pytest

from squadds.core.analysis_plotting import build_closest_design_hspace_plot


def test_build_closest_design_hspace_plot_supports_quarter_wave():
    df = pd.DataFrame(
        {
            "cavity_frequency_GHz": [6.8, 7.0, 7.2],
            "kappa_kHz": [120.0, 140.0, 160.0],
            "anharmonicity_MHz": [-220.0, -210.0, -200.0],
            "g_MHz": [60.0, 70.0, 80.0],
        }
    )
    target_params = {
        "cavity_frequency_GHz": 7.05,
        "kappa_kHz": 150.0,
        "anharmonicity_MHz": -205.0,
        "g_MHz": 75.0,
    }
    closest_df_entry = df.iloc[1]

    fig, (ax1, ax2) = build_closest_design_hspace_plot(df, target_params, closest_df_entry, "quarter")

    assert fig is not None
    assert ax1.get_xlabel() == r"$f_{res}$ (GHz)"
    assert ax1.get_ylabel() == r"$\kappa / 2 \pi$ (kHz)"
    assert ax2.get_xlabel() == r"$\alpha / 2 \pi$ (MHz)"
    assert ax2.get_ylabel() == r"$g / 2 \pi$ (MHz)"
    plt.close(fig)


def test_build_closest_design_hspace_plot_supports_half_wave():
    df = pd.DataFrame(
        {
            "cavity_frequency_GHz": [6.8, 7.0, 7.2],
            "kappa_kHz": [120.0, 140.0, 160.0],
            "anharmonicity_MHz": [-220.0, -210.0, -200.0],
            "g_MHz": [60.0, 70.0, 80.0],
            "kappa": [120000.0, 140000.0, 160000.0],
        }
    )
    target_params = {
        "cavity_frequency_GHz": 7.05,
        "kappa_kHz": 150.0,
        "anharmonicity_MHz": -205.0,
        "g_MHz": 75.0,
    }
    closest_df_entry = df.iloc[1]

    fig, (ax1, ax2) = build_closest_design_hspace_plot(df, target_params, closest_df_entry, "half")

    assert fig is not None
    assert ax1.get_ylabel() == r"$\kappa / 2 \pi$ (Hz)"
    assert ax2.get_ylabel() == r"$g / 2 \pi$ (MHz)"
    plt.close(fig)


def test_build_closest_design_hspace_plot_rejects_unknown_resonator_type():
    df = pd.DataFrame(
        {
            "cavity_frequency_GHz": [7.0],
            "kappa_kHz": [140.0],
            "anharmonicity_MHz": [-210.0],
            "g_MHz": [70.0],
        }
    )

    with pytest.raises(ValueError, match='not supported'):
        build_closest_design_hspace_plot(df, {}, df.iloc[0], "lumped")
