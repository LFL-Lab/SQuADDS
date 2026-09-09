#!/usr/bin/env python
"""Export deterministic publication PNGs from executed Tutorial 24 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as PolygonPatch

DPI = 360
SCHEMA_VERSION = "1.0.0"
FIGURE_NAMES = (
    "acceptance_scorecard.png",
    "physical_trajectories.png",
    "block_ablation.png",
    "qmetal_atlas.png",
    "modular_arithmetic.png",
    "native_vs_embedding.png",
    "similarity_capacitance.png",
    "transfer_learning.png",
    "generalist_growth.png",
    "unseen_composite.png",
)
PALETTE = ("#00798C", "#D1495B", "#6A4C93", "#EDA842", "#2A9D8F", "#48639C", "#8A5A44")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    # Qiskit Metal may select Qt during test collection; force a headless
    # backend again at call time so figure export remains CI-safe.
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": DPI,
        }
    )


def save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "SQuADDS Tutorial 24 deterministic exporter"},
    )
    plt.close(figure)


def plot_acceptance_scorecard(source: Path, destination: Path) -> dict[str, Any]:
    configure_style()
    frame = pd.read_csv(source)
    gates = [column for column in frame if column.startswith("pass_")]
    required = {"representation", *gates}
    if not gates or not required <= set(frame):
        raise ValueError("Acceptance scorecard requires representation and pass_* columns.")
    values = frame[gates].astype(bool).astype(int).to_numpy()
    labels = [column.removeprefix("pass_").replace("_", " ") for column in gates]
    if "arbitrary_terminal_contract" in frame:
        values = np.column_stack([values, frame["arbitrary_terminal_contract"].astype(bool).astype(int)])
        labels.append("arbitrary terminals")
    figure, axis = plt.subplots(figsize=(11.2, 4.4), constrained_layout=True)
    axis.imshow(values, aspect="auto", vmin=0, vmax=1, cmap=ListedColormap(["#D1495B", "#2A9D8F"]))
    axis.set_xticks(np.arange(len(labels)), labels, rotation=36, ha="right", rotation_mode="anchor")
    axis.set_yticks(np.arange(len(frame)), frame["representation"])
    axis.set_xlabel("Predeclared acceptance requirement")
    axis.set_ylabel("Representation")
    axis.set_title("Planar embedding acceptance scorecard")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "PASS" if values[row, column] else "FAIL",
                ha="center",
                va="center",
                color="white",
                fontsize=6.8,
                fontweight="bold",
            )
    for position in np.arange(-0.5, len(labels), 1):
        axis.axvline(position, color="white", linewidth=0.7)
    for position in np.arange(-0.5, len(frame), 1):
        axis.axhline(position, color="white", linewidth=0.7)
    save_figure(figure, destination)
    return {
        "rows": len(frame),
        "gate_count": len(labels),
        "pass_cells": int(values.sum()),
        "total_cells": int(values.size),
    }


def plot_physical_trajectories(source: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    required = {"representation", "group", "control_value", "distance_from_baseline"}
    if not required <= set(frame):
        raise ValueError(f"Physical trajectories missing columns: {sorted(required - set(frame))}")
    groups = list(dict.fromkeys(frame["group"].astype(str)))
    representations = list(dict.fromkeys(frame["representation"].astype(str)))
    columns = 2
    rows = int(np.ceil(len(groups) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10.8, 3.5 * rows), squeeze=False, constrained_layout=True)
    xlabels = {
        "uniform_scale": "Uniform scale factor",
        "terminal_gap_um": "Terminal gap (µm)",
        "facing_length_um": "Facing length (µm)",
        "ground_clearance_um": "Ground clearance (µm)",
    }
    for group_index, group in enumerate(groups):
        axis = axes.flat[group_index]
        subset = frame[frame["group"] == group]
        for rep_index, representation in enumerate(representations):
            curve = subset[subset["representation"] == representation].sort_values("control_value")
            axis.plot(
                curve["control_value"],
                curve["distance_from_baseline"],
                marker="o",
                markersize=3.2,
                linewidth=1.35,
                color=PALETTE[rep_index % len(PALETTE)],
                label=representation,
            )
        axis.set_title(group.replace("_um", "").replace("_", " ").title())
        axis.set_xlabel(xlabels.get(group, group.replace("_", " ")))
        axis.set_ylabel("Relative embedding drift")
        axis.grid(True, alpha=0.22, linewidth=0.6)
    for axis in axes.flat[len(groups) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False)
    figure.suptitle("Embedding response to controlled physical perturbations", fontweight="bold")
    save_figure(figure, destination)
    return {"rows": len(frame), "groups": groups, "representations": representations}


def plot_block_ablation(source: Path, destination: Path) -> dict[str, Any]:
    """Render long or wide ablation tables without assuming unpublished values."""
    frame = pd.read_csv(source)
    if {"mode", "selection", "nuisance", "physical", "ratio"} <= set(frame):
        figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
        for axis, mode in zip(axes, ("cumulative", "drop one")):
            subset = frame[frame["mode"] == mode].copy()
            labels = (
                subset["selection"].str.replace("global + ", "", regex=False).str.replace("without ", "− ", regex=False)
            )
            axis.barh(labels, subset["ratio"], color=PALETTE[0 if mode == "cumulative" else 3])
            axis.set_xlabel("physical response / nuisance response")
            axis.set_title("Add blocks in order" if mode == "cumulative" else "Remove one block")
            axis.grid(axis="x", alpha=0.2)
            axis.tick_params(axis="y", labelsize=7)
        figure.suptitle("v4 layout-view block ablation: every retained view stays physics-sensitive", fontweight="bold")
        save_figure(figure, destination)
        return {
            "rows": len(frame),
            "ratio_min": float(frame["ratio"].min()),
            "ratio_max": float(frame["ratio"].max()),
            "modes": list(dict.fromkeys(frame["mode"])),
        }
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in {"seed", "fold"}]
    if not numeric:
        raise ValueError("Block-ablation table has no numeric result columns.")
    label_column = (
        "block" if "block" in frame else next((column for column in frame if frame[column].dtype == object), None)
    )
    if label_column is None:
        labels = frame.index.astype(str)
    else:
        labels = frame[label_column].astype(str)
    values = frame[numeric].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(max(7.2, 0.75 * len(labels)), 4.8), constrained_layout=True)
    width = 0.8 / len(numeric)
    positions = np.arange(len(labels))
    for index, column in enumerate(numeric):
        axis.bar(
            positions - 0.4 + width / 2 + index * width,
            values[:, index],
            width=width,
            color=PALETTE[index % len(PALETTE)],
            label=column.replace("_", " "),
        )
    axis.axhline(0, color="#333333", linewidth=0.7)
    axis.set_xticks(positions, labels, rotation=32, ha="right")
    axis.set_ylabel("Reported metric value")
    axis.set_title("Embedding block ablation")
    axis.legend(frameon=False, ncol=min(3, len(numeric)))
    axis.grid(axis="y", alpha=0.2)
    save_figure(figure, destination)
    return {"rows": len(frame), "numeric_columns": numeric, "label_column": label_column}


def _gds_polygons(path: Path, layer: int = 1, datatype: int = 10) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.top_cell()
    if top is None:
        raise ValueError(f"No top cell in {path}")
    layer_index = layout.find_layer(layer, datatype)
    if layer_index is None:
        return []
    region = kdb.Region(top.begin_shapes_rec(layer_index))
    region.merge()
    dbu = float(layout.dbu)
    result = []
    for polygon in region.each():
        exterior = np.asarray([(point.x * dbu, point.y * dbu) for point in polygon.each_point_hull()])
        holes = [
            np.asarray([(point.x * dbu, point.y * dbu) for point in polygon.each_point_hole(index)])
            for index in range(polygon.holes())
        ]
        result.append((exterior, holes))
    return result


def plot_qmetal_atlas(source: Path, destination: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text())
    records = [record for record in payload["records"] if record["status"] == "generated_valid"]
    columns = 6
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13.2, 2.05 * rows), constrained_layout=True)
    categories = list(dict.fromkeys(record["component"]["category"] for record in records))
    colors = {category: PALETTE[index % len(PALETTE)] for index, category in enumerate(categories)}
    for axis, record in zip(axes.flat, records):
        path = source.parent / record["gds_path"]
        polygons = _gds_polygons(path)
        for exterior, holes in polygons:
            axis.add_patch(
                PolygonPatch(exterior, closed=True, facecolor=colors[record["component"]["category"]], edgecolor="none")
            )
            for hole in holes:
                axis.add_patch(PolygonPatch(hole, closed=True, facecolor="white", edgecolor="white"))
        if polygons:
            points = np.vstack([exterior for exterior, _ in polygons])
            left, bottom = points.min(axis=0)
            right, top = points.max(axis=0)
            span = max(right - left, top - bottom, 1e-9)
            margin = 0.08 * span
            axis.set_xlim(left - margin, right + margin)
            axis.set_ylim(bottom - margin, top + margin)
        axis.set_aspect("equal")
        axis.set_axis_off()
        title = record["component"]["class_name"]
        axis.set_title(title, fontsize=6.7, pad=1.5, color="#222222")
    for axis in axes.flat[len(records) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Qiskit Metal singleton atlas: {len(records)} validated geometry-bearing components",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(figure, destination)
    return {
        "rendered_components": len(records),
        "inventory_count": int(payload.get("inventory_count", len(payload["records"]))),
        "status_counts": payload.get("counts", {}).get("by_status", {}),
        "geometry_equivalence_classes": payload.get("counts", {}).get("geometry_equivalence_classes"),
    }


def _draw_gds(axis: Any, path: Path, title: str) -> None:
    styles = {
        (1, 0): ("#DFE6EE", "#8796A8"),
        (1, 10): ("#27689D", "#174A72"),
        (2, 0): ("#EE6A3B", "#A83D1E"),
        (3, 0): ("#F1A340", "#A8660C"),
        (4, 0): ("#9C6ADE", "#5D3B91"),
    }
    all_points = []
    for layer, (face, edge) in styles.items():
        for exterior, holes in _gds_polygons(path, *layer):
            all_points.append(exterior)
            axis.add_patch(PolygonPatch(exterior, closed=True, facecolor=face, edgecolor=edge, linewidth=0.55))
            for hole in holes:
                axis.add_patch(PolygonPatch(hole, closed=True, facecolor="white", edgecolor="white"))
    if all_points:
        points = np.vstack(all_points)
        left, bottom = points.min(axis=0)
        right, top = points.max(axis=0)
        span = max(right - left, top - bottom, 1e-9)
        margin = 0.04 * span
        axis.set_xlim(left - margin, right + margin)
        axis.set_ylim(bottom - margin, top + margin)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(title, fontweight="bold")


def plot_modular_arithmetic(source: Path, destination: Path) -> dict[str, Any]:
    summary = json.loads(source.read_text())
    manifest = pd.read_csv(source.parent / "manifest.csv").set_index("fixture")
    blocks = pd.read_csv(source.parent / "block_arithmetic.csv")
    figure = plt.figure(figsize=(12.4, 7.0), constrained_layout=True)
    grid = GridSpec(2, 3, figure=figure, height_ratios=[1.15, 0.85])
    for column, (fixture, title) in enumerate(
        (
            ("module_a", "A: left + shared"),
            ("module_b", "B: shared + right"),
            ("joined_three_terminal", "A ⊕ B: merged graph"),
        )
    ):
        _draw_gds(figure.add_subplot(grid[0, column]), source.parents[4] / manifest.loc[fixture, "gds_path"], title)
    axis = figure.add_subplot(grid[1, :2])
    axis.barh(blocks["block"], blocks["naive_addition_relative_error"], color=PALETTE[0])
    axis.set_xlabel("relative error of raw vector A + B")
    axis.set_title("Raw fixed-vector addition fails block by block")
    axis.grid(axis="x", alpha=0.2)
    axis = figure.add_subplot(grid[1, 2])
    labels = ["raw A+B\nerror", "pose-matched Δ\nerror", "rotated wider\ndrift"]
    values = [
        summary["naive_vector_addition_relative_error"],
        summary["pose_matched_delta_relative_error"],
        summary["wider_gap_rotation_relative_drift"],
    ]
    axis.bar(labels, values, color=(PALETTE[1], PALETTE[4], PALETTE[4]))
    axis.set_yscale("log")
    axis.set_ylabel("relative error or drift")
    axis.set_title("Composition versus difference diagnostics")
    axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Port-aware planar composition: merge the graph, then re-encode", fontsize=13, fontweight="bold")
    save_figure(figure, destination)
    return {
        "fixtures": int(summary["fixtures"]),
        "naive_addition_relative_error": summary["naive_vector_addition_relative_error"],
        "pose_delta_cosine": summary["pose_matched_delta_cosine"],
    }


def plot_native_vs_embedding(source: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    mean = (
        frame.groupby(["family", "representation"], sort=False)[["r2_log1p", "rmse_fF", "medae_fF"]]
        .mean()
        .reset_index()
    )
    std = (
        frame.groupby(["family", "representation"], sort=False)[["r2_log1p", "rmse_fF", "medae_fF"]].std().reset_index()
    )
    families = list(dict.fromkeys(mean["family"]))
    representations = list(dict.fromkeys(mean["representation"]))
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), constrained_layout=True)
    metrics = (("r2_log1p", "$R^2$ in log(1+C)"), ("rmse_fF", "RMSE (fF)"), ("medae_fF", "Median absolute error (fF)"))
    positions = np.arange(len(families))
    width = 0.82 / len(representations)
    for axis, (metric, label) in zip(axes, metrics):
        for index, representation in enumerate(representations):
            rows = mean[mean["representation"] == representation].set_index("family").loc[families]
            errors = std[std["representation"] == representation].set_index("family").loc[families]
            axis.bar(
                positions - 0.41 + width / 2 + index * width,
                rows[metric],
                width,
                yerr=errors[metric],
                capsize=2,
                color=PALETTE[index],
                label=representation,
            )
        axis.set_xticks(
            positions,
            [
                name.replace("GeneralizedCapNInterdigital", "Generalized IDC").replace("CapNInterdigitalTee", "Tee")
                for name in families
            ],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle("Matched within-family prediction: native options versus universal embeddings", fontweight="bold")
    save_figure(figure, destination)
    return {"rows": len(frame), "repeats": int(frame["repeat"].nunique()), "families": families}


def plot_similarity_capacitance(source: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    balanced = frame[frame["family_pair"] == "all_family_balanced"].copy()
    pairwise = frame[frame["family_pair"] != "all_family_balanced"].copy()
    pairwise[["family_a", "family_b"]] = pairwise["family_pair"].str.split("|", expand=True, regex=False)
    families = sorted(set(pairwise["family_a"]) | set(pairwise["family_b"]))
    figure, axes = plt.subplots(
        1, 4, figsize=(13.6, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1, 1, 1]}
    )
    low = balanced["spearman"] - balanced["ci95_low"]
    high = balanced["ci95_high"] - balanced["spearman"]
    axes[0].bar(
        np.arange(len(balanced)),
        balanced["spearman"],
        yerr=np.vstack([low, high]),
        capsize=4,
        color=PALETTE[: len(balanced)],
    )
    axes[0].set_xticks(np.arange(len(balanced)), balanced["representation"], rotation=25, ha="right")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_title("All families, balanced")
    image = None
    for axis, representation in zip(axes[1:], balanced["representation"]):
        matrix = np.full((len(families), len(families)), np.nan)
        for _, row in pairwise[pairwise["representation"] == representation].iterrows():
            i, j = families.index(row["family_a"]), families.index(row["family_b"])
            matrix[i, j] = matrix[j, i] = row["spearman"]
        image = axis.imshow(matrix, vmin=-0.25, vmax=0.55, cmap="RdBu")
        axis.set_xticks(range(len(families)), ["Tee", "Gen IDC", "Cross"], rotation=40, ha="right")
        axis.set_yticks(range(len(families)), ["Tee", "Gen IDC", "Cross"])
        axis.set_title(representation)
    if image is not None:
        figure.colorbar(image, ax=axes[1:], shrink=0.72, label="pairwise Spearman")
    figure.suptitle("Embedding distance is only modestly calibrated to capacitance difference", fontweight="bold")
    save_figure(figure, destination)
    return {
        "balanced_pairs": int(balanced["pairs"].iloc[0]),
        "balanced_spearman": dict(zip(balanced["representation"], balanced["spearman"])),
    }


def plot_transfer_learning(source: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    grouped = (
        frame.groupby(["representation", "relation", "target_labels", "method"])["r2_log1p"]
        .agg(["mean", "std"])
        .reset_index()
    )
    representations = list(dict.fromkeys(grouped["representation"]))
    methods = list(dict.fromkeys(grouped["method"]))
    figure, axes = plt.subplots(1, len(representations), figsize=(13.2, 4.1), sharey=True, constrained_layout=True)
    for axis, representation in zip(axes, representations):
        subset = grouped[grouped["representation"] == representation]
        for method_index, method in enumerate(methods):
            for relation, style in (("near", "-"), ("far", "--")):
                rows = subset[(subset["method"] == method) & (subset["relation"] == relation)].sort_values(
                    "target_labels"
                )
                axis.errorbar(
                    rows["target_labels"],
                    rows["mean"],
                    yerr=rows["std"],
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    linestyle=style,
                    color=PALETTE[method_index],
                    label=f"{method.replace('_', ' ')} · {relation}",
                )
        axis.set_xscale("log")
        axis.set_xticks([1, 5, 10, 50, 100], [1, 5, 10, 50, 100])
        axis.axhline(0, color="#333", linewidth=0.6)
        axis.set_xlabel("target labels")
        axis.set_title(representation)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("mean $R^2$ in log(1+C)")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    figure.suptitle("Near/far few-shot transfer: target-only remains the 100-label winner", fontweight="bold")
    save_figure(figure, destination)
    return {
        "rows": len(frame),
        "repeats": int(frame["repeat"].nunique()),
        "budgets": sorted(frame["target_labels"].unique().tolist()),
    }


def plot_generalist_growth(source: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    grouped = (
        frame.groupby(["representation", "train_family_count", "train_designs"])["r2_log1p"]
        .agg(["mean", "std"])
        .reset_index()
    )
    representations = list(dict.fromkeys(grouped["representation"]))
    figure, axes = plt.subplots(1, len(representations), figsize=(12.8, 4.0), sharey=True, constrained_layout=True)
    for axis, representation in zip(axes, representations):
        subset = grouped[grouped["representation"] == representation]
        for count in sorted(subset["train_family_count"].unique()):
            rows = subset[subset["train_family_count"] == count].sort_values("train_designs")
            axis.errorbar(
                rows["train_designs"],
                rows["mean"],
                yerr=rows["std"],
                marker="o",
                linewidth=1.4,
                color=PALETTE[count - 1],
                label=f"{count} families",
            )
        axis.set_xscale("log")
        axis.set_xticks([30, 100, 300, 600], [30, 100, 300, 600])
        axis.axhline(0, color="#333", linewidth=0.6)
        axis.set_xlabel("total training designs")
        axis.set_title(representation)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("mean cross-task $R^2$")
    axes[-1].legend(frameon=False)
    figure.suptitle("Family diversity matters more than one-family density", fontweight="bold")
    save_figure(figure, destination)
    return {
        "rows": len(frame),
        "budgets": sorted(frame["train_designs"].unique().tolist()),
        "family_counts": sorted(frame["train_family_count"].unique().tolist()),
    }


def plot_unseen_composite(source: Path, destination: Path) -> dict[str, Any]:
    matrices = np.load(source)
    matrix = matrices["joined_three_terminal"]
    manifest = pd.read_csv(source.parent / "manifest.csv").set_index("fixture")
    gds_path = source.parents[4] / manifest.loc["joined_three_terminal", "gds_path"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    _draw_gds(axes[0], gds_path, "Unseen three-terminal joined GDS")
    image = axes[1].imshow(matrix, cmap="RdBu", vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
    labels = ["left", "shared", "right", "ground"]
    axes[1].set_xticks(range(4), labels, rotation=30, ha="right")
    axes[1].set_yticks(range(4), labels)
    axes[1].set_title("4×4 deterministic low-fidelity operator")
    for row in range(4):
        for column in range(4):
            axes[1].text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes[1], shrink=0.8, label="proxy fF")
    figure.suptitle("Structural zero-shot pass; high-fidelity capacitance remains untested", fontweight="bold")
    save_figure(figure, destination)
    return {"matrix_shape": list(matrix.shape), "status": "structural_pass_predictive_untested"}


def plot_generic_csv(source: Path, destination: Path) -> dict[str, Any]:
    frame = pd.read_csv(source)
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    if len(numeric) < 2:
        raise ValueError("Generic CSV rendering requires at least two numeric columns.")
    x, ys = numeric[0], numeric[1:5]
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ordered = frame.sort_values(x)
    for index, column in enumerate(ys):
        axis.plot(
            ordered[x],
            ordered[column],
            marker="o",
            linewidth=1.3,
            markersize=3,
            color=PALETTE[index],
            label=column.replace("_", " "),
        )
    axis.set_xlabel(x.replace("_", " "))
    axis.set_ylabel("Reported value")
    axis.set_title(source.stem.replace("_", " ").title())
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, destination)
    return {"rows": len(frame), "x_column": x, "y_columns": ys}


def _record(
    figure_name: str,
    source: Path | None,
    destination: Path,
    plotter: Callable[[Path, Path], dict[str, Any]],
) -> dict[str, Any]:
    record: dict[str, Any] = {"figure": figure_name, "source": str(source) if source else None}
    if source is None or not source.is_file():
        record.update(status="skipped_missing_source", output=None)
        return record
    try:
        details = plotter(source, destination)
    except Exception as error:  # noqa: BLE001 - preserve per-figure failures in the manifest.
        record.update(status="failed", output=None, error_type=type(error).__name__, error_message=str(error))
        return record
    record.update(
        status="generated",
        output=str(destination),
        source_sha256=sha256_file(source),
        output_sha256=sha256_file(destination),
        dpi=DPI,
        details=details,
    )
    return record


def export_all(runtime: Path, output: Path) -> dict[str, Any]:
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    requested = [
        ("acceptance_scorecard.png", runtime / "acceptance_scorecard.csv", plot_acceptance_scorecard),
        ("physical_trajectories.png", runtime / "physical_trajectories.csv", plot_physical_trajectories),
        ("block_ablation.png", runtime / "block_ablation.csv", plot_block_ablation),
        ("qmetal_atlas.png", runtime / "qmetal-atlas" / "manifest.json", plot_qmetal_atlas),
        ("modular_arithmetic.png", runtime / "modularity" / "summary.json", plot_modular_arithmetic),
        ("native_vs_embedding.png", runtime / "predictive" / "family_specific_baselines.csv", plot_native_vs_embedding),
        (
            "similarity_capacitance.png",
            runtime / "predictive" / "distance_capacitance_correlation.csv",
            plot_similarity_capacitance,
        ),
        ("transfer_learning.png", runtime / "predictive" / "transfer_learning_curves.csv", plot_transfer_learning),
        ("generalist_growth.png", runtime / "predictive" / "generalist_growth_diversity.csv", plot_generalist_growth),
        ("unseen_composite.png", runtime / "modularity" / "proxy_matrices.npz", plot_unseen_composite),
    ]
    records = [_record(name, source, output / name, plotter) for name, source, plotter in requested]
    claimed = {source.resolve() for _, source, _ in requested}
    supporting = {"controlled_manifest.csv"}
    for source in sorted(runtime.glob("*.csv")):
        if source.resolve() in claimed or source.name in supporting:
            continue
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source.stem) + ".png"
        if name in FIGURE_NAMES:
            continue
        records.append(_record(name, source, output / name, plot_generic_csv))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dpi": DPI,
        "runtime": str(runtime),
        "output_directory": str(output),
        "counts": {
            status: sum(record["status"] == status for record in records)
            for status in ("generated", "skipped_missing_source", "failed")
        },
        "figures": records,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=repo / "tutorials/runtime/tutorial24")
    parser.add_argument("--output", type=Path, default=repo.parent / "embedding-transfer-paper/figures/generated")
    arguments = parser.parse_args()
    result = export_all(arguments.runtime.resolve(), arguments.output.resolve())
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    if result["counts"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
