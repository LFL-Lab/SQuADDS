#!/usr/bin/env python
"""Render the port-complete study into a report, figures, and metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {
    "old/v0": "#B23A48",
    "new/v0": "#D1495B",
    "new/v0-etch": "#E9A44C",
    "new/v0-ports": "#E9C46A",
    "new/v1-local": "#6A4C93",
    "old/v2": "#7FB2C0",
    "new/v2": "#00798C",
    "mixed/v2 published-source + port-complete-target": "#00798C",
}
PRIMARY = {"TransmonCross": "cross_to_claw", "CapNInterdigitalTee": "top_to_bottom"}


def learning_curves(within: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, component in zip(axes, PRIMARY):
        frame = within[within.component_name == component]
        for name, group in frame.groupby("representation"):
            curve = group.groupby("train_rows", as_index=False).agg(
                rmse=("rmse_fF", "mean"), sd=("rmse_fF", "std")
            )
            axis.errorbar(
                curve.train_rows, curve.rmse, yerr=curve.sd, marker="o", capsize=2, linewidth=2,
                label=name, color=COLORS.get(name, "#888888"),
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("labeled training designs")
        axis.set_ylabel("held-out RMSE (fF)")
        axis.set_title(f"{component}  ({PRIMARY[component]})")
        axis.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7.5, loc="lower left")
    figure.suptitle("Within-family learning curves, port-complete GDS (12 grouped repeats)")
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    plt.close(figure)


def parity(raw: pd.DataFrame, predictions: Path | None, output: Path) -> None:
    if predictions is None or not predictions.is_file():
        return
    frame = pd.read_parquet(predictions)
    components = sorted(frame.component_name.unique())
    representations = ["old/v2", "new/v2", "new/v0-ports"]
    figure, axes = plt.subplots(len(components), len(representations), figsize=(12.5, 8.0))
    for row, component in enumerate(components):
        for column, name in enumerate(representations):
            axis = axes[row][column]
            subset = frame[(frame.component_name == component) & (frame.representation == name)]
            if subset.empty:
                axis.set_visible(False)
                continue
            axis.scatter(subset.expected_fF, subset.predicted_fF, s=6, alpha=0.35,
                         color=COLORS.get(name, "#888888"))
            span = [subset.expected_fF.min(), subset.expected_fF.max()]
            axis.plot(span, span, "--", color="#6B7280", linewidth=1)
            residual = subset.predicted_fF - subset.expected_fF
            axis.set_title(f"{component}\n{name}  RMSE={np.sqrt((residual**2).mean()):.3f} fF", fontsize=9)
            axis.set_xlabel("simulated (fF)")
            axis.set_ylabel("predicted (fF)")
            axis.grid(alpha=0.3)
    figure.suptitle("Predicted versus simulated on held-out designs, one representative fold")
    figure.tight_layout()
    figure.savefig(output / "parity.png", dpi=150)
    plt.close(figure)


def cross_family_figure(cross: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    summary = cross.groupby(["representation", "method", "train_rows"], as_index=False).agg(
        rmse=("rmse_fF", "mean"), r2=("r2", "mean")
    )
    for axis, method in zip(axes, ["target-only", "transfer (source prior)"]):
        for name, group in summary[summary.method == method].groupby("representation"):
            group = group[group.train_rows > 0].sort_values("train_rows")
            axis.plot(group.train_rows, group.rmse, marker="o", linewidth=2,
                      label=name, color=COLORS.get(name, "#888888"))
        for name, group in summary[summary.method == "source-only (zero-shot)"].groupby("representation"):
            axis.axhline(float(group.rmse.iloc[0]), linestyle=":", linewidth=1.4,
                         color=COLORS.get(name, "#888888"))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("labeled CapNInterdigitalTee designs")
        axis.set_ylabel("held-out RMSE (fF)")
        axis.set_title(method)
        axis.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7.5)
    figure.suptitle("GeneralizedCapNInterdigital to CapNInterdigitalTee; dotted lines are zero-shot")
    figure.tight_layout()
    figure.savefig(output / "cross_family.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    arguments = parser.parse_args()
    results = arguments.results_dir

    raw = pd.read_parquet(results / "raw_metrics.parquet")
    config = json.loads((results / "config.json").read_text())
    within = raw.query("study == 'within-family'")
    cross = raw.query("study == 'cross-family'")

    learning_curves(within, results)
    cross_family_figure(cross, results)
    parity(raw, arguments.predictions, results)

    full = within[within.fraction == 1.0]
    within_table = full.groupby(["component_name", "representation"], as_index=False).agg(
        rmse_fF=("rmse_fF", "mean"), rmse_sd=("rmse_fF", "std"),
        median_abs_error_fF=("median_abs_error_fF", "mean"),
        mae_fF=("mae_fF", "mean"), r2=("r2", "mean"), r2_sd=("r2", "std"),
        train_rows=("train_rows", "mean"), test_rows=("test_rows", "mean"),
    )
    cross_table = cross.groupby(["representation", "method", "fraction"], as_index=False).agg(
        rmse_fF=("rmse_fF", "mean"), rmse_sd=("rmse_fF", "std"),
        median_abs_error_fF=("median_abs_error_fF", "mean"), r2=("r2", "mean"),
    )

    def value(component, representation, column):
        row = within_table[
            (within_table.component_name == component) & (within_table.representation == representation)
        ]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    effects = {
        "transmon_ports_under_v2": {
            "contrast": "old/v2 vs new/v2 on TransmonCross",
            "note": "conductor geometry is byte-identical between releases, so only the ports move",
            "rmse_fF_before": value("TransmonCross", "old/v2", "rmse_fF"),
            "rmse_fF_after": value("TransmonCross", "new/v2", "rmse_fF"),
        },
        "transmon_etch_under_v0": {
            "contrast": "new/v0 vs new/v0-etch on TransmonCross",
            "note": "published v0 ignores TransmonCross etch entirely",
            "rmse_fF_before": value("TransmonCross", "new/v0", "rmse_fF"),
            "rmse_fF_after": value("TransmonCross", "new/v0-etch", "rmse_fF"),
        },
        "transmon_ports_under_v0": {
            "contrast": "new/v0-etch vs new/v0-ports on TransmonCross",
            "note": "etch already present in both, so only the ports move",
            "rmse_fF_before": value("TransmonCross", "new/v0-etch", "rmse_fF"),
            "rmse_fF_after": value("TransmonCross", "new/v0-ports", "rmse_fF"),
        },
        "capn_geometry_under_v2": {
            "contrast": "old/v2 vs new/v2 on CapNInterdigitalTee",
            "note": "corrected CPW conductor plus ordered ports",
            "rmse_fF_before": value("CapNInterdigitalTee", "old/v2", "rmse_fF"),
            "rmse_fF_after": value("CapNInterdigitalTee", "new/v2", "rmse_fF"),
        },
        "capn_ports_under_v0": {
            "contrast": "new/v0-etch vs new/v0-ports on CapNInterdigitalTee",
            "rmse_fF_before": value("CapNInterdigitalTee", "new/v0-etch", "rmse_fF"),
            "rmse_fF_after": value("CapNInterdigitalTee", "new/v0-ports", "rmse_fF"),
        },
    }
    for entry in effects.values():
        before, after = entry["rmse_fF_before"], entry["rmse_fF_after"]
        entry["delta_rmse_fF"] = after - before
        entry["relative_change"] = (after - before) / before if before else float("nan")

    zero_shot = cross_table[cross_table.method == "source-only (zero-shot)"]
    metrics = {
        "config": config,
        "within_family_full_pool": within_table.round(6).to_dict(orient="records"),
        "cross_family": cross_table.round(6).to_dict(orient="records"),
        "effect_isolation": effects,
        "cross_family_zero_shot": zero_shot.round(6).to_dict(orient="records"),
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    within_table.to_csv(results / "within_family.csv", index=False)
    cross_table.to_csv(results / "cross_family.csv", index=False)
    print(within_table.round(4).to_string(index=False))
    print()
    print(json.dumps(effects, indent=2))


if __name__ == "__main__":
    main()
