"""Plotting helpers for the Analyzer compatibility facade."""

from __future__ import annotations

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import seaborn as sns


def build_closest_design_hspace_plot(df, target_params, closest_df_entry, selected_resonator_type):
    """Build the legacy closest-design H-space plot and return the figure and axes."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)

    viridis_cmap = plt.get_cmap("viridis")
    viridis_cmap(0.2)
    color_presim = viridis_cmap(0.9)
    color_database = viridis_cmap(0.6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    if selected_resonator_type == "quarter":
        ax1.scatter(
            x=df["cavity_frequency_GHz"],
            y=df["kappa_kHz"],
            color=color_presim,
            marker=".",
            s=50,
            label="Pre-Simulated",
        )
        ax1.scatter(
            x=target_params["cavity_frequency_GHz"],
            y=target_params["kappa_kHz"],
            color="red",
            s=100,
            marker="x",
            label="Target",
        )
        closest_fres = closest_df_entry["cavity_frequency_GHz"]
        closest_kappa_kHz = closest_df_entry["kappa_kHz"]
        ax1.scatter(
            closest_fres, closest_kappa_kHz, color=[color_database], s=100, marker="s", alpha=0.7, label="Closest"
        )
        ax1.set_xlabel(r"$f_{res}$ (GHz)", fontweight="bold", fontsize=24)
        ax1.set_ylabel(r"$\kappa / 2 \pi$ (kHz)", fontweight="bold", fontsize=24)
        ax1.tick_params(axis="both", which="major", labelsize=20)

        ax2.scatter(
            x=df["anharmonicity_MHz"],
            y=df["g_MHz"],
            color=color_presim,
            marker=".",
            s=50,
            label="Pre-Simulated",
        )
        ax2.scatter(
            x=target_params["anharmonicity_MHz"],
            y=target_params["g_MHz"],
            color="red",
            s=100,
            marker="x",
            label="Target",
        )
        closest_alpha = [closest_df_entry["anharmonicity_MHz"]]
        closest_g = [closest_df_entry["g_MHz"]]
        ax2.scatter(closest_alpha, closest_g, color=[color_database], s=100, marker="s", alpha=0.7, label="Closest")
        ax2.set_xlabel(r"$\alpha / 2 \pi$ (MHz)", fontweight="bold", fontsize=24)
        ax2.set_ylabel(r"$g / 2 \pi$ (MHz)", fontweight="bold", fontsize=24)
        ax2.tick_params(axis="both", which="major", labelsize=20)

    elif selected_resonator_type == "half":
        x1_range = (df["cavity_frequency_GHz"].min(), df["cavity_frequency_GHz"].max())
        y1_range = (df["kappa_kHz"].min(), df["kappa_kHz"].max())
        x2_range = (df["anharmonicity_MHz"].min(), df["anharmonicity_MHz"].max())
        y2_range = (df["g_MHz"].min(), df["g_MHz"].max())

        canvas1 = ds.Canvas(plot_width=800, plot_height=600, x_range=x1_range, y_range=y1_range)
        canvas2 = ds.Canvas(plot_width=800, plot_height=600, x_range=x2_range, y_range=y2_range)
        agg1 = canvas1.points(df, "cavity_frequency_GHz", "kappa_kHz")
        agg2 = canvas2.points(df, "anharmonicity_MHz", "g_MHz")

        cmap = plt.get_cmap("Blues")
        colors = [cmap(i) for i in range(cmap.N)]
        hex_colors = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b, _ in colors]
        img1 = tf.shade(agg1, cmap=hex_colors)
        img2 = tf.shade(agg2, cmap=hex_colors)

        ax1.imshow(img1.to_pil(), aspect="auto", extent=[*x1_range, *y1_range])
        ax1.set_xlabel(r"$f_{res}$ (GHz)", fontweight="bold", fontsize=24)
        ax1.set_ylabel(r"$\kappa / 2 \pi$ (Hz)", fontweight="bold", fontsize=24)
        ax1.tick_params(axis="both", which="major", labelsize=20)

        ax2.imshow(img2.to_pil(), aspect="auto", extent=[*x2_range, *y2_range])
        ax2.set_xlabel(r"$\alpha / 2 \pi$ (MHz)", fontweight="bold", fontsize=24)
        ax2.set_ylabel(r"$g / 2 \pi$ (MHz)", fontweight="bold", fontsize=24)
        ax2.tick_params(axis="both", which="major", labelsize=20)

        ax1.plot(target_params["cavity_frequency_GHz"], target_params["kappa_kHz"] * 1e3, "rx", label="Target")
        ax2.plot(target_params["anharmonicity_MHz"], target_params["g_MHz"], "ro", label="Target")

        ax1.plot(closest_df_entry["cavity_frequency_GHz"], closest_df_entry["kappa"], "bs", alpha=1, label="Closest")
        ax2.plot(
            closest_df_entry["anharmonicity_MHz"],
            closest_df_entry["g_MHz"],
            "bs",
            alpha=0.7,
            label="Closest",
        )
    else:
        raise ValueError(
            f'Your chosen resonator type - {selected_resonator_type} - is not supported. Please use "quarter" or "half"'
        )

    legend1 = ax1.legend(loc="upper left", fontsize=16)
    for text in legend1.get_texts():
        text.set_fontweight("bold")

    legend2 = ax2.legend(loc="lower left", fontsize=16)
    for text in legend2.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()
    return fig, (ax1, ax2)
