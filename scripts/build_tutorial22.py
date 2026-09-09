#!/usr/bin/env python
"""Build Tutorial 22: what geometric similarity means in v2."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("tutorials/Tutorial-22_Geometry_Similarity_and_Capacitance_with_v2.ipynb")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, *, hidden: bool | None = None):
    cell = nbf.v4.new_code_cell(source.strip())
    if hidden is None:
        hidden = source.lstrip().startswith("# %% hide input")
    if hidden:
        cell["metadata"]["tags"] = ["hide-input"]
        cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "SQuADDS Tutorial (uv)",
        "language": "python",
        "name": "squadds-tutorial",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

CELLS = [
    markdown(
        r"""
# Tutorial 22: when do similar geometries have similar capacitances?

Tutorial 20 asked whether `universal-geometry-v2` transfers across three
component classes. It found a deliberately uncomfortable result: v2 makes
few-shot transfer excellent, but a cosine neighbor from another family is not
automatically a physically interchangeable design.

This tutorial turns that warning into a quantitative anatomy of similarity.
It uses the same unified layout release and the same three mutual-capacitance
targets:

| family | terminal pair | mutual capacitance |
| --- | --- | --- |
| `GeneralizedCapNInterdigital` | north / south | `north_to_south` |
| `CapNInterdigitalTee` | top / bottom | `top_to_bottom` |
| `TransmonCross` | cross / claw | `cross_to_claw` |

The question is not merely whether two 512-vectors have a large cosine. We will
ask, at each level of the representation, **what was made similar**, whether
that similarity predicts capacitance, and whether the answer changes when a
neighbor comes from another component family.

We will:

1. build a class-balanced three-family cohort;
2. connect interpretable physical coordinates to capacitance;
3. open the mutual and terminal-to-ground coupling spectra;
4. separate family identity from local geometric neighborhoods;
5. calibrate cosine similarity against capacitance error;
6. turn similarity into a nearest-neighbor predictor with R2, RMSE, and median
   absolute error;
7. inspect real GDS pairs that succeed or fail; and
8. replay the complete six-step encoder on all three component families; and
9. finish with a practical rule for retrieval, transfer, and inverse design.

Every figure is interactive: hover for the exact design or statistic, zoom into
dense regions, toggle families from legends, and use the dropdowns to change the
embedding block or physical question without changing the cohort.
"""
    ),
    code(
        r"""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shapely
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from squadds.layouts import canonical_design_id
from squadds.layouts.geometry_v2 import (
    COUPLING_BINS,
    COUPLING_BLOCK_SIZE,
    COUPLING_EDGES,
    GROUND_BINS,
    GROUND_EDGES,
    METRIC_BLOCK_SIZE,
    METRIC_NAMES,
    PARAMETER_BLOCK_SIZE,
    SHAPE_BLOCK_SIZE,
    TERMINAL_PAIRS,
    V2_DIMENSIONS,
)

pio.renderers.default = "notebook_connected"
pd.set_option("display.max_columns", 40)
logging.getLogger("httpx").setLevel(logging.WARNING)

SEED = 22
BALANCED_PER_CLASS = 894
PAIR_SAMPLES = 30_000
K_VALUES = [1, 5, 15]

CLASSES = {
    "GeneralizedCapNInterdigital": {
        "file": "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
        "mutual": "north_to_south",
        "grounds": ("north_to_ground", "south_to_ground"),
    },
    "CapNInterdigitalTee": {
        "file": "coupler-CapNInterdigitalTee-cap_matrix.json",
        "mutual": "top_to_bottom",
        "grounds": ("top_to_ground", "bottom_to_ground"),
    },
    "TransmonCross": {
        "file": "qubit-TransmonCross-cap_matrix.json",
        "mutual": "cross_to_claw",
        "grounds": ("cross_to_ground", "claw_to_ground"),
    },
}
CLASS_COLORS = {
    "GeneralizedCapNInterdigital": "#00798C",
    "CapNInterdigitalTee": "#D1495B",
    "TransmonCross": "#6A4C93",
}
BLOCK_COLORS = {
    "v2 full": "#00798C",
    "geometry only": "#2A9D8F",
    "physical metrics": "#F4A261",
    "coupling spectrum": "#4C86A8",
    "shape spectrum": "#6A4C93",
    "parameter statistics": "#E9C46A",
    "physics proxy": "#8AB17D",
    "terminal-ground spectrum": "#D1495B",
}

CHECKPOINTS = Path(os.getenv("SQUADDS_TUTORIAL20_CACHE", Path.home() / ".cache/squadds/tutorial20"))
PORT_COMPLETE_TABLE = CHECKPOINTS / "universal-geometry-v2-portcomplete.parquet"
V2_TABLE = Path(os.getenv("SQUADDS_V2_ALL_TABLE", PORT_COMPLETE_TABLE))
if not V2_TABLE.is_file():
    raise FileNotFoundError(
        "Tutorial 22 uses the unified three-family table built for Tutorial 20; "
        f"expected {V2_TABLE}."
    )

PORT_COMPLETE_ROOT = Path(
    os.getenv(
        "SQUADDS_PORT_COMPLETE_ROOT",
        str(Path.home() / "Documents/New project/SQuADDS-port-gds-artifacts/layout-dataset"),
    )
)
database = {
    name: Path(hf_hub_download("SQuADDS/SQuADDS_DB", spec["file"], repo_type="dataset"))
    for name, spec in CLASSES.items()
}
print("v2 table:", V2_TABLE)
print("local unified layouts:", PORT_COMPLETE_ROOT)
"""
    ),
    markdown(
        r"""
## 1. One physical target, three different geometric priors

We use `log1p(C)` for correlations and R2 because the three capacitance ranges
are very different, but always report RMSE and median absolute error back in
femtofarads. That distinction matters: a representation can rank designs
correctly in log space while still missing an engineering tolerance in fF.

The full catalogue is strongly imbalanced. To prevent the 13,683 generalized
couplers from defining the mean, scale, and neighborhood of every other class,
we take the same deterministic **894 designs per family** used by Tutorial 20's
balanced experiment.
"""
    ),
    code(
        r"""
def load_targets():
    records = []
    for component, spec in CLASSES.items():
        for row in json.loads(database[component].read_text()):
            options = row["design"]["design_options"]
            results = row["sim_results"]
            mutual = abs(float(results[spec["mutual"]]))
            ground_sum = sum(abs(float(results[name])) for name in spec["grounds"])
            records.append(
                {
                    "design_id": canonical_design_id(component, options),
                    "component_name": component,
                    "mutual_fF": mutual,
                    "log_mutual": float(np.log1p(mutual)),
                    "ground_sum_fF": ground_sum,
                }
            )
    return pd.DataFrame(records).drop_duplicates("design_id")


targets = load_targets()
v2_frame = pd.read_parquet(V2_TABLE).drop_duplicates("design_id")
joined = targets.merge(v2_frame[["design_id", "embedding"]], on="design_id")

picked = []
for component in sorted(CLASSES):
    rows = joined.index[joined.component_name == component].to_numpy()
    ordered = rows[np.argsort(joined.loc[rows, "design_id"].to_numpy())]
    picked.append(np.random.default_rng(SEED).permutation(ordered)[:BALANCED_PER_CLASS])
cohort = np.sort(np.concatenate(picked))
data = joined.loc[cohort].reset_index(drop=True)
v2 = np.vstack(data.pop("embedding").to_numpy()).astype(np.float32)
components = data.component_name.to_numpy()
mutual_fF = data.mutual_fF.to_numpy(float)
log_mutual = data.log_mutual.to_numpy(float)

assert len(data) == BALANCED_PER_CLASS * len(CLASSES)
assert v2.shape == (len(data), V2_DIMENSIONS)

summary = data.groupby("component_name").agg(
    designs=("design_id", "size"),
    minimum_fF=("mutual_fF", "min"),
    median_fF=("mutual_fF", "median"),
    maximum_fF=("mutual_fF", "max"),
)
summary.round(3)
"""
    ),
    code(
        r"""
# %% hide input
figure = go.Figure()
for component in sorted(CLASSES):
    values = data.loc[data.component_name == component, "mutual_fF"]
    figure.add_trace(
        go.Violin(
            x=[component] * len(values),
            y=values,
            name=component,
            legendgroup=component,
            line_color=CLASS_COLORS[component],
            fillcolor=CLASS_COLORS[component],
            opacity=0.55,
            box_visible=True,
            meanline_visible=True,
            points=False,
            hovertemplate=f"<b>{component}</b><br>C=%{{y:.3f}} fF<extra></extra>",
        )
    )
figure.update_layout(
    title="The same target spans three family-specific capacitance regimes",
    yaxis={"title": "mutual capacitance (fF)", "type": "log"},
    xaxis={"title": "component family"},
    template="plotly_white",
    height=520,
    showlegend=False,
)
figure.show()
"""
    ),
    markdown(
        r"""
The distributions overlap, so class identity is not a substitute for the
target. But their centers and ranges differ enough that a cross-family neighbor
must answer two questions: is the local geometry comparable, and is the mapping
from that geometry to capacitance calibrated the same way in both families?

## 2. Start with coordinates that have physical names

Before treating v2 as an abstract vector, inspect four coordinates with direct
units or physical meanings. Use the dropdown to place capacitance against the
minimum terminal gap, conductor area, primary inverse-gap integral, or terminal
area asymmetry. Marker size is conductor area and hover exposes the exact design.

These are associations, not causal fits. They show why a single scalar cannot
align all three classes: capacitance depends jointly on separation, facing
boundary length, terminal scale, and the ground environment.
"""
    ),
    code(
        r"""
metric_index = {name: index for index, name in enumerate(METRIC_NAMES)}
PHYSICAL_VIEWS = {
    "minimum terminal gap (um)": np.expm1(v2[:, metric_index["log1p_minimum_pair_gap_um"]]),
    "conductor area (um^2)": np.expm1(v2[:, metric_index["log1p_conductor_area_um2"]]),
    "primary inverse-gap integral": np.expm1(
        v2[:, metric_index["log1p_primary_inverse_gap_integral"]]
    ),
    "terminal area ratio (large / small)": np.maximum(
        np.expm1(v2[:, metric_index["log1p_terminal_0_area_um2"]]),
        np.expm1(v2[:, metric_index["log1p_terminal_1_area_um2"]]),
    )
    / np.maximum(
        np.minimum(
            np.expm1(v2[:, metric_index["log1p_terminal_0_area_um2"]]),
            np.expm1(v2[:, metric_index["log1p_terminal_1_area_um2"]]),
        ),
        1e-9,
    ),
}
data["conductor_area_um2"] = PHYSICAL_VIEWS["conductor area (um^2)"]

correlation_rows = []
for label, values in PHYSICAL_VIEWS.items():
    for component in ["pooled", *sorted(CLASSES)]:
        mask = np.ones(len(data), dtype=bool) if component == "pooled" else components == component
        correlation_rows.append(
            {
                "coordinate": label,
                "cohort": component,
                "Spearman with log C": float(spearmanr(values[mask], log_mutual[mask]).statistic),
            }
        )
pd.DataFrame(correlation_rows).pivot(
    index="coordinate", columns="cohort", values="Spearman with log C"
).round(3)
"""
    ),
    code(
        r"""
# %% hide input
view_names = list(PHYSICAL_VIEWS)
figure = go.Figure()
for view_index, (view_name, x_values) in enumerate(PHYSICAL_VIEWS.items()):
    for component in sorted(CLASSES):
        rows = np.flatnonzero(components == component)
        area = data.conductor_area_um2.to_numpy()[rows]
        size = 5 + 10 * np.sqrt(area / max(float(area.max()), 1.0))
        figure.add_trace(
            go.Scattergl(
                x=x_values[rows],
                y=mutual_fF[rows],
                mode="markers",
                name=component,
                legendgroup=component,
                visible=view_index == 0,
                marker={"size": size, "opacity": 0.55, "color": CLASS_COLORS[component]},
                customdata=np.column_stack([data.design_id.to_numpy()[rows], area]),
                hovertemplate=(
                    f"<b>{component}</b><br>{view_name}=%{{x:.4g}}<br>"
                    "C=%{y:.4g} fF<br>area=%{customdata[1]:.1f} um^2<br>"
                    "%{customdata[0]}<extra></extra>"
                ),
            )
        )

buttons = []
traces_per_view = len(CLASSES)
for index, name in enumerate(view_names):
    visible = [False] * len(figure.data)
    start = index * traces_per_view
    visible[start : start + traces_per_view] = [True] * traces_per_view
    buttons.append(
        {
            "label": name,
            "method": "update",
            "args": [
                {"visible": visible},
                {"xaxis.title": name, "title": f"Mutual capacitance versus {name}"},
            ],
        }
    )
figure.update_layout(
    title=f"Mutual capacitance versus {view_names[0]}",
    xaxis={"title": view_names[0], "type": "log"},
    yaxis={"title": "mutual capacitance (fF)", "type": "log"},
    template="plotly_white",
    height=590,
    updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.16}],
)
figure.show()
"""
    ),
    markdown(
        r"""
The family-specific correlations are stronger and more consistent than the
pooled correlations. That is the first sign that "geometrically close" is a
conditional statement: the same gap or area participates in a different shape
and scale context in a comb, a Tee, and a cross-and-claw qubit.

The primary inverse-gap integral is the most portable scalar here: its Spearman
correlation with mutual capacitance is **+0.959, +0.942, and +0.984** in the
three families and +0.914 pooled. Conductor area looks equally persuasive when
each family is studied alone (+0.909, +0.868, +0.542) but collapses to +0.250
pooled. Terminal area ratio is more severe: +0.513 in the Tee, -0.117 in the
generalized coupler, -0.469 in the transmon, and essentially zero pooled. A
shared coordinate is not automatically a shared response law.

## 3. Open the coupling spectrum

The 192-coordinate coupling block is not one opaque descriptor. Its first 144
slots hold terminal-to-terminal boundary length over absolute distance bins;
its last 48 reserve 12 terminal-to-ground bins for each of four terminals.

All layouts in this tutorial share a dynamic ground plane and two moat-bridging
ports, so terminal 0 and terminal 1 both have real ground spectra. Use the
dropdown to compare the mutual spectrum with either terminal's ground spectrum.
The y-axis is physical boundary length recovered from the stored `log1p`
coordinates; the x-axis is the frozen micrometre distance grid.
"""
    ),
    code(
        r"""
COUPLING_START = METRIC_BLOCK_SIZE
GROUND_START = COUPLING_START + len(TERMINAL_PAIRS) * COUPLING_BINS

SPECTRA = {
    "terminal 0 to terminal 1": (
        np.sqrt(COUPLING_EDGES[:-1] * COUPLING_EDGES[1:]),
        np.expm1(v2[:, COUPLING_START : COUPLING_START + COUPLING_BINS]),
    ),
    "terminal 0 to ground": (
        np.sqrt(GROUND_EDGES[:-1] * GROUND_EDGES[1:]),
        np.expm1(v2[:, GROUND_START : GROUND_START + GROUND_BINS]),
    ),
    "terminal 1 to ground": (
        np.sqrt(GROUND_EDGES[:-1] * GROUND_EDGES[1:]),
        np.expm1(v2[:, GROUND_START + GROUND_BINS : GROUND_START + 2 * GROUND_BINS]),
    ),
}

peak_rows = []
for spectrum, (centres, values) in SPECTRA.items():
    for component in sorted(CLASSES):
        median = np.median(values[components == component], axis=0)
        peak_rows.append(
            {
                "spectrum": spectrum,
                "component": component,
                "peak distance um": float(centres[int(np.argmax(median))]),
                "integrated boundary length um": float(median.sum()),
            }
        )
pd.DataFrame(peak_rows).round(3)
"""
    ),
    code(
        r"""
# %% hide input
figure = go.Figure()
for spectrum_index, (spectrum, (centres, values)) in enumerate(SPECTRA.items()):
    for component in sorted(CLASSES):
        subset = values[components == component]
        median = np.median(subset, axis=0)
        low, high = np.quantile(subset, [0.25, 0.75], axis=0)
        figure.add_trace(
            go.Scatter(
                x=centres,
                y=median,
                mode="lines+markers",
                name=component,
                legendgroup=component,
                visible=spectrum_index == 0,
                line={"color": CLASS_COLORS[component], "width": 3},
                marker={"size": 7},
                customdata=np.column_stack([low, high]),
                hovertemplate=(
                    f"<b>{component}</b><br>{spectrum}<br>distance=%{{x:.3g}} um<br>"
                    "median boundary=%{y:.3g} um<br>IQR=%{customdata[0]:.3g}–%{customdata[1]:.3g} um"
                    "<extra></extra>"
                ),
            )
        )

buttons = []
for index, spectrum in enumerate(SPECTRA):
    visible = [False] * len(figure.data)
    start = index * len(CLASSES)
    visible[start : start + len(CLASSES)] = [True] * len(CLASSES)
    buttons.append(
        {
            "label": spectrum,
            "method": "update",
            "args": [
                {"visible": visible},
                {"title": f"Median {spectrum} coupling spectrum", "xaxis.title": f"{spectrum} distance (um)"},
            ],
        }
    )
first_spectrum = next(iter(SPECTRA))
figure.update_layout(
    title=f"Median {first_spectrum} coupling spectrum",
    xaxis={"title": f"{first_spectrum} distance (um)", "type": "log"},
    yaxis={"title": "boundary length in distance bin (um)"},
    template="plotly_white",
    height=560,
    updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.16}],
)
figure.show()
"""
    ),
    markdown(
        r"""
This spectrum explains more than a bounding box can. Two designs can share a
minimum gap but allocate very different amounts of facing metal to that gap.
Conversely, a cross and an interdigital capacitor can place boundary length in
similar absolute distance bins while remaining globally dissimilar shapes.

The ground dropdowns also expose terminal asymmetry. The TransmonCross cross
and claw do not contribute interchangeable spectra, so a similarity that
averages away terminal identity throws out physics the ordered port convention
was added to preserve.

Numerically, the median mutual spectrum peaks near **12.1 um** for the
generalized coupler but **56.2 um** for both the Tee and transmon. The transmon's
two ground spectra peak in different bins—31.6 um for terminal 0 and 6.81 um
for terminal 1—and carry roughly 2,400 versus 1,030 um of integrated boundary
length. The encoder is seeing the cross/claw asymmetry, not merely counting two
ports.

## 4. Seven legitimate meanings of "similar"

v2 publishes blocks because no one metric should silently decide that contour,
gap distribution, design parameters, and an electrostatic proxy are equally
important. We compare seven views below, plus the terminal-ground sub-block.

For this analysis only, each block is standardized on the balanced cohort and
each row is normalized to unit length. Cosine then compares deviations from the
balanced catalogue mean. This is an analysis-time similarity—not a change to
the fixed, catalogue-independent v2 vector.
"""
    ),
    code(
        r"""
COUPLING_STOP = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE
SHAPE_STOP = COUPLING_STOP + SHAPE_BLOCK_SIZE
PARAMETER_STOP = SHAPE_STOP + PARAMETER_BLOCK_SIZE
GEOMETRY_COLUMNS = np.r_[0:SHAPE_STOP, PARAMETER_STOP:V2_DIMENSIONS]

BLOCKS = {
    "v2 full": v2,
    "geometry only": v2[:, GEOMETRY_COLUMNS],
    "physical metrics": v2[:, :METRIC_BLOCK_SIZE],
    "coupling spectrum": v2[:, METRIC_BLOCK_SIZE:COUPLING_STOP],
    "shape spectrum": v2[:, COUPLING_STOP:SHAPE_STOP],
    "parameter statistics": v2[:, SHAPE_STOP:PARAMETER_STOP],
    "physics proxy": v2[:, PARAMETER_STOP:V2_DIMENSIONS],
    "terminal-ground spectrum": v2[:, GROUND_START : GROUND_START + 4 * GROUND_BINS],
}


def standardized_unit(matrix):
    values = matrix.astype(np.float64)
    centre = values.mean(axis=0)
    scale = values.std(axis=0)
    keep = scale > 1e-9
    z = (values[:, keep] - centre[keep]) / scale[keep]
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


UNIT = {name: standardized_unit(matrix) for name, matrix in BLOCKS.items()}


def mean_cosine_matrix(unit):
    names = sorted(CLASSES)
    answer = np.zeros((len(names), len(names)))
    for i, first in enumerate(names):
        left = unit[components == first]
        for j, second in enumerate(names):
            right = unit[components == second]
            if i == j:
                total = left.sum(axis=0)
                answer[i, j] = (float(total @ total) - len(left)) / (len(left) * (len(left) - 1))
            else:
                answer[i, j] = float(left.mean(axis=0) @ right.mean(axis=0))
    return answer


FAMILY_COSINE = {name: mean_cosine_matrix(unit) for name, unit in UNIT.items()}
family_separation = pd.DataFrame(
    [
        {
            "block": name,
            "mean within-family cosine": float(np.diag(matrix).mean()),
            "mean cross-family cosine": float(matrix[np.triu_indices(3, 1)].mean()),
            "family separation": float(
                np.diag(matrix).mean() - matrix[np.triu_indices(3, 1)].mean()
            ),
        }
        for name, matrix in FAMILY_COSINE.items()
    ]
)
family_separation.round(3).sort_values("family separation", ascending=False)
"""
    ),
    code(
        r"""
# %% hide input
names = sorted(CLASSES)
first_block = next(iter(FAMILY_COSINE))
figure = go.Figure(
    go.Heatmap(
        z=FAMILY_COSINE[first_block],
        x=names,
        y=names,
        zmin=-0.8,
        zmax=1.0,
        colorscale="RdBu",
        reversescale=True,
        text=np.round(FAMILY_COSINE[first_block], 3),
        texttemplate="%{text:.3f}",
        colorbar={"title": "mean cosine"},
        hovertemplate="%{y}<br>%{x}<br>mean cosine=%{z:.4f}<extra></extra>",
    )
)
buttons = [
    {
        "label": block,
        "method": "update",
        "args": [{"z": [matrix], "text": [np.round(matrix, 3)]}, {"title": f"Family similarity in {block}"}],
    }
    for block, matrix in FAMILY_COSINE.items()
]
figure.update_layout(
    title=f"Family similarity in {first_block}",
    template="plotly_white",
    height=560,
    margin={"l": 205, "b": 145},
    updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.15}],
)
figure.show()
"""
    ),
    markdown(
        r"""
The dropdown reveals a crucial distinction. A block can be excellent at
recognizing a family and poor at finding equal capacitance across families.
Shape coordinates are expected to separate a comb from a cross; that does not
make them wrong. It makes them unsuitable as the only cross-family retrieval
metric.

Parameter statistics are the strongest family identifier: mean within-family
cosine is +0.952 and mean cross-family cosine is -0.469, a separation of 1.421.
The terminal-ground and complete coupling spectra are next at 1.148 and 1.033.
The physics proxy has the weakest family separation, 0.729, which helps explain
why Tutorial 20 found it to be the only individually useful zero-shot block.

## 5. A map is a view, not a metric

The next plot uses an unsupervised randomized two-dimensional sketch. Change the
block with the dropdown and hover any design for its capacitance, gap, and ID.
The projection is useful for seeing family manifolds and overlaps, but it does
not preserve every neighbor. We therefore use it to generate hypotheses, never
as the quantitative similarity test.
"""
    ),
    code(
        r"""
def sketch(matrix, seed=SEED):
    values = matrix.astype(np.float64)
    centre, scale = values.mean(axis=0), values.std(axis=0)
    keep = scale > 1e-9
    standardized = (values[:, keep] - centre[keep]) / scale[keep]
    rng = np.random.default_rng(seed)
    projected = standardized @ rng.normal(
        0, 1 / np.sqrt(32), size=(standardized.shape[1], min(32, standardized.shape[1]))
    )
    left, singular, _ = np.linalg.svd(projected, full_matrices=False)
    return left[:, :2] * singular[:2]


MAP_BLOCKS = ["v2 full", "physical metrics", "coupling spectrum", "shape spectrum", "physics proxy"]
MAPS = {name: sketch(BLOCKS[name]) for name in MAP_BLOCKS}
"""
    ),
    code(
        r"""
# %% hide input
figure = go.Figure()
gap_um = PHYSICAL_VIEWS["minimum terminal gap (um)"]
for block_index, block in enumerate(MAP_BLOCKS):
    coordinates = MAPS[block]
    for component in sorted(CLASSES):
        rows = np.flatnonzero(components == component)
        figure.add_trace(
            go.Scattergl(
                x=coordinates[rows, 0],
                y=coordinates[rows, 1],
                mode="markers",
                name=component,
                legendgroup=component,
                visible=block_index == 0,
                marker={"size": 6, "opacity": 0.55, "color": CLASS_COLORS[component]},
                customdata=np.column_stack(
                    [mutual_fF[rows], gap_um[rows], data.design_id.to_numpy()[rows]]
                ),
                hovertemplate=(
                    f"<b>{component}</b><br>C=%{{customdata[0]:.3f}} fF<br>"
                    "minimum gap=%{customdata[1]:.3f} um<br>%{customdata[2]}<extra></extra>"
                ),
            )
        )
buttons = []
for index, block in enumerate(MAP_BLOCKS):
    visible = [False] * len(figure.data)
    start = index * len(CLASSES)
    visible[start : start + len(CLASSES)] = [True] * len(CLASSES)
    buttons.append(
        {
            "label": block,
            "method": "update",
            "args": [{"visible": visible}, {"title": f"Two-dimensional sketch of {block}"}],
        }
    )
figure.update_layout(
    title=f"Two-dimensional sketch of {MAP_BLOCKS[0]}",
    xaxis={"title": "unsupervised sketch direction 1"},
    yaxis={"title": "unsupervised sketch direction 2"},
    template="plotly_white",
    height=590,
    updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.15}],
)
figure.show()
"""
    ),
    markdown(
        r"""
The shape map forms the clearest class islands; the physical and physics-proxy
maps create more cross-family overlap. Neither pattern is automatically better.
Family islands help classification and within-family interpolation. Overlap is
what a zero- or few-shot cross-family model needs, provided the overlap is also
calibrated to capacitance.

## 6. Calibrate cosine against capacitance error

We now test the metric directly. For every within- and cross-family pairing we
sample 30,000 pairs, bin them by cosine decile, and measure the median relative
capacitance difference. A useful metric should slope downward: its most similar
pairs should have the smallest capacitance discrepancy.

Use the dropdown to change blocks and toggle individual family pairings in the
legend. This plot is more informative than a single Spearman coefficient because
it shows whether only the extreme nearest-neighbor tail is trustworthy.
"""
    ),
    code(
        r"""
pair_names = []
pair_indices = {}
rng = np.random.default_rng(SEED)
class_names = sorted(CLASSES)
for i, first in enumerate(class_names):
    for second in class_names[i:]:
        left_pool = np.flatnonzero(components == first)
        right_pool = np.flatnonzero(components == second)
        left = rng.choice(left_pool, PAIR_SAMPLES)
        right = rng.choice(right_pool, PAIR_SAMPLES)
        valid = left != right
        label = f"within {first}" if first == second else f"{first} vs {second}"
        pair_names.append(label)
        pair_indices[label] = (left[valid], right[valid])

calibration_records = []
correlation_records = []
PAIR_VALUES = {}
for block, unit in UNIT.items():
    for pairing, (left, right) in pair_indices.items():
        similarity = np.sum(unit[left] * unit[right], axis=1)
        relative = 200 * np.abs(mutual_fF[left] - mutual_fF[right]) / np.maximum(
            mutual_fF[left] + mutual_fF[right], 1e-12
        )
        delta_log = np.abs(log_mutual[left] - log_mutual[right])
        PAIR_VALUES[(block, pairing)] = (left, right, similarity, relative, delta_log)
        correlation_records.append(
            {
                "block": block,
                "pairing": pairing,
                "Spearman(cosine, |delta log C|)": float(spearmanr(similarity, delta_log).statistic),
            }
        )
        quantile = pd.qcut(similarity, 10, labels=False, duplicates="drop")
        frame = pd.DataFrame({"similarity": similarity, "relative": relative, "decile": quantile})
        grouped = frame.groupby("decile", observed=True).agg(
            mean_cosine=("similarity", "mean"),
            median_relative_error=("relative", "median"),
            p90_relative_error=("relative", lambda values: float(np.quantile(values, 0.9))),
        )
        for decile, row in grouped.iterrows():
            calibration_records.append(
                {"block": block, "pairing": pairing, "decile": int(decile), **row.to_dict()}
            )

calibration = pd.DataFrame(calibration_records)
correlations = pd.DataFrame(correlation_records)
cross_correlations = correlations[~correlations.pairing.str.startswith("within")]
cross_correlations.pivot(
    index="pairing", columns="block", values="Spearman(cosine, |delta log C|)"
).round(3)
"""
    ),
    code(
        r"""
# %% hide input
figure = go.Figure()
for block_index, block in enumerate(BLOCKS):
    for pairing in pair_names:
        frame = calibration.query("block == @block and pairing == @pairing").sort_values("mean_cosine")
        figure.add_trace(
            go.Scatter(
                x=frame.mean_cosine,
                y=frame.median_relative_error,
                mode="lines+markers",
                name=pairing,
                legendgroup=pairing,
                visible=block_index == 0,
                line={"width": 2.5, "dash": "solid" if pairing.startswith("within") else "dot"},
                marker={"size": 7},
                customdata=frame.p90_relative_error,
                hovertemplate=(
                    f"<b>{pairing}</b><br>mean cosine=%{{x:.3f}}<br>"
                    "median relative difference=%{y:.1f}%<br>90th percentile=%{customdata:.1f}%"
                    "<extra></extra>"
                ),
            )
        )
buttons = []
for index, block in enumerate(BLOCKS):
    visible = [False] * len(figure.data)
    start = index * len(pair_names)
    visible[start : start + len(pair_names)] = [True] * len(pair_names)
    buttons.append(
        {
            "label": block,
            "method": "update",
            "args": [
                {"visible": visible},
                {"title": f"Capacitance calibration of {block} cosine"},
            ],
        }
    )
figure.update_layout(
    title="Capacitance calibration of v2 full cosine",
    xaxis={"title": "mean cosine in similarity decile"},
    yaxis={"title": "median pairwise capacitance difference (%)"},
    template="plotly_white",
    height=610,
    updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.16}],
)
figure.show()
"""
    ),
    markdown(
        r"""
Inside a family, increasing cosine generally means decreasing capacitance
error. Across families the slope depends on the block and pairing. In
particular, a block can assign a sensible *ordering* (negative Spearman) while
remaining poorly calibrated in absolute fF because the two families occupy
different capacitance regimes.

That is why "the nearest geometry" and "a geometry with the same capacitance"
are different retrieval requests.

The two coupler families are the successful cross-class case: full-v2 cosine
has Spearman **-0.600** against `|delta log C|`, geometry-only reaches -0.714,
and every geometry block is usefully negative. Across the qubit boundary the
answer reverses. Tee-versus-transmon full cosine is **+0.453**—more similar can
mean *less* capacitively alike—while the physics proxy is the only useful block
at -0.340. Generalized-versus-transmon is essentially unranked in every block;
even the physics proxy is only -0.068.

## 7. Make similarity predict: the k-nearest-neighbor audit

A similarity metric becomes operational when it retrieves labels. For every
design we predict `log1p(C)` from its 1, 5, or 15 nearest neighbors, then convert
back to fF. We run two scopes:

- **within-family:** candidate neighbors must share the query's component class;
- **cross-family:** every same-family candidate is forbidden.

No regressor is trained and no family label enters the vector. This is a direct
audit of what the metric itself knows. The heatmaps show R2 in log space, while
hover also reports RMSE and median absolute error in fF.
"""
    ),
    code(
        r"""
def r2_score(expected, predicted):
    residual = float(np.sum((expected - predicted) ** 2))
    total = float(np.sum((expected - expected.mean()) ** 2))
    return 1.0 - residual / max(total, 1e-12)


neighbor_records = []
NEIGHBORS = {}
for block, unit in UNIT.items():
    similarity = (unit @ unit.T).astype(np.float32)
    np.fill_diagonal(similarity, -np.inf)
    for scope in ("within-family", "cross-family"):
        allowed = components[:, None] == components[None, :]
        if scope == "cross-family":
            allowed = ~allowed
        masked = np.where(allowed, similarity, -np.inf)
        candidate = np.argpartition(masked, -max(K_VALUES), axis=1)[:, -max(K_VALUES) :]
        candidate_scores = np.take_along_axis(masked, candidate, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        neighbors = np.take_along_axis(candidate, order, axis=1)
        NEIGHBORS[(block, scope)] = neighbors
        for k in K_VALUES:
            prediction_log = log_mutual[neighbors[:, :k]].mean(axis=1)
            prediction_fF = np.expm1(prediction_log)
            for component in [*sorted(CLASSES), "macro"]:
                if component == "macro":
                    class_scores = []
                    for family in sorted(CLASSES):
                        rows = components == family
                        error = prediction_fF[rows] - mutual_fF[rows]
                        class_scores.append(
                            (
                                r2_score(log_mutual[rows], prediction_log[rows]),
                                float(np.sqrt(np.mean(error**2))),
                                float(np.median(np.abs(error))),
                            )
                        )
                    r2, rmse, medae = np.mean(class_scores, axis=0)
                else:
                    rows = components == component
                    error = prediction_fF[rows] - mutual_fF[rows]
                    r2 = r2_score(log_mutual[rows], prediction_log[rows])
                    rmse = float(np.sqrt(np.mean(error**2)))
                    medae = float(np.median(np.abs(error)))
                neighbor_records.append(
                    {
                        "block": block,
                        "scope": scope,
                        "k": k,
                        "component": component,
                        "r2_log": float(r2),
                        "rmse_fF": float(rmse),
                        "median_absolute_error_fF": float(medae),
                    }
                )

neighbors = pd.DataFrame(neighbor_records)
neighbors.query("component == 'macro' and k == 5").pivot(
    index="block", columns="scope", values=["r2_log", "rmse_fF", "median_absolute_error_fF"]
).round(3)
"""
    ),
    code(
        r"""
# %% hide input
block_order = list(BLOCKS)
family_order = sorted(CLASSES)
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["neighbors restricted within family", "same-family neighbors forbidden"],
    horizontal_spacing=0.14,
)
for column, scope in enumerate(("within-family", "cross-family"), start=1):
    frame = neighbors.query("scope == @scope and k == 5 and component != 'macro'")
    r2 = frame.pivot(index="component", columns="block", values="r2_log").reindex(
        index=family_order, columns=block_order
    )
    rmse = frame.pivot(index="component", columns="block", values="rmse_fF").reindex(
        index=family_order, columns=block_order
    )
    medae = frame.pivot(
        index="component", columns="block", values="median_absolute_error_fF"
    ).reindex(index=family_order, columns=block_order)
    custom = np.stack([rmse.to_numpy(), medae.to_numpy()], axis=-1)
    figure.add_trace(
        go.Heatmap(
            z=r2,
            x=block_order,
            y=family_order,
            zmin=-6,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(r2, 2),
            texttemplate="%{text:.2f}",
            customdata=custom,
            colorbar={"title": "R2", "x": 0.46 if column == 1 else 1.02},
            hovertemplate=(
                "%{y}<br>%{x}<br>R2(log C)=%{z:.4f}<br>"
                "RMSE=%{customdata[0]:.3f} fF<br>median |error|=%{customdata[1]:.3f} fF"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=column,
    )
figure.update_xaxes(tickangle=-32)
figure.update_layout(
    title="Five-neighbor capacitance prediction: similarity is scope dependent",
    template="plotly_white",
    height=610,
    margin={"l": 190, "b": 170},
)
figure.show()
"""
    ),
    markdown(
        r"""
Within-family nearest neighbors are a strong nonparametric predictor for most
blocks. Cross-family retrieval is much harder and strongly family dependent.
The gap between the two panels is the **family calibration penalty**: geometric
descriptors that order a local design sweep need not share one global mapping to
capacitance.

RMSE and median absolute error tell complementary stories. RMSE exposes a few
catastrophic false neighbors; the median describes the typical retrieved
design. A safe retrieval system needs both, not R2 alone.

At five neighbors the within-family macro R2 is **0.960–0.973** for every major
block except terminal-ground alone (0.870), with typical median errors around
0.4–0.5 fF. Every cross-family macro R2 is negative. The full vector reaches
-6.555 with 14.15 fF RMSE; the physics proxy is least bad at -1.048, 5.30 fF
RMSE, and 2.87 fF median absolute error. This is not a small degradation—it is a
different retrieval regime.
"""
    ),
    markdown(
        r"""
## 8. Matching mutual capacitance is not matching a capacitance matrix

Each simulation also reports the sum of the two terminal-to-ground
capacitances. For the same sampled cross-family pairs, compare relative error in
mutual capacitance with relative error in ground-capacitance sum. Dashed lines
mark 10% agreement. Toggle pairings in the legend and hover for both simulated
values.

The lower-left quadrant contains pairs that are electrostatically similar in
both summaries. The lower-right and upper-left quadrants are the warning:
matching one scalar capacitance does not make a two-terminal device equivalent.
"""
    ),
    code(
        r"""
ground_sum_fF = data.ground_sum_fF.to_numpy(float)
fingerprint_frames = []
fingerprint_summary = []
for pairing, (left, right) in pair_indices.items():
    if pairing.startswith("within"):
        continue
    mutual_relative = 200 * np.abs(mutual_fF[left] - mutual_fF[right]) / np.maximum(
        mutual_fF[left] + mutual_fF[right], 1e-12
    )
    ground_relative = 200 * np.abs(ground_sum_fF[left] - ground_sum_fF[right]) / np.maximum(
        ground_sum_fF[left] + ground_sum_fF[right], 1e-12
    )
    frame = pd.DataFrame(
        {
            "pairing": pairing,
            "left": left,
            "right": right,
            "mutual_relative": mutual_relative,
            "ground_relative": ground_relative,
        }
    )
    fingerprint_frames.append(frame)
    mutual_match = mutual_relative <= 10
    ground_match = ground_relative <= 10
    fingerprint_summary.append(
        {
            "pairing": pairing,
            "pairs": len(left),
            "mutual within 10%": float(mutual_match.mean()),
            "ground sum within 10%": float(ground_match.mean()),
            "both within 10%": float((mutual_match & ground_match).mean()),
            "P(ground within 10% | mutual within 10%)": float(
                ground_match[mutual_match].mean() if mutual_match.any() else np.nan
            ),
        }
    )
fingerprints = pd.concat(fingerprint_frames, ignore_index=True)
pd.DataFrame(fingerprint_summary).round(3)
"""
    ),
    code(
        r"""
# %% hide input
figure = go.Figure()
pair_colors = ["#00798C", "#D1495B", "#6A4C93"]
for color, pairing in zip(pair_colors, sorted(fingerprints.pairing.unique())):
    frame = fingerprints.query("pairing == @pairing")
    shown = frame.iloc[::6]
    left = shown.left.to_numpy(int)
    right = shown.right.to_numpy(int)
    figure.add_trace(
        go.Scattergl(
            x=np.maximum(shown.mutual_relative, 1e-3),
            y=np.maximum(shown.ground_relative, 1e-3),
            mode="markers",
            name=pairing,
            marker={"size": 5, "opacity": 0.35, "color": color},
            customdata=np.column_stack(
                [mutual_fF[left], mutual_fF[right], ground_sum_fF[left], ground_sum_fF[right]]
            ),
            hovertemplate=(
                f"<b>{pairing}</b><br>mutual difference=%{{x:.2f}}%<br>"
                "ground-sum difference=%{y:.2f}%<br>mutual A/B=%{customdata[0]:.3f}/%{customdata[1]:.3f} fF<br>"
                "ground A/B=%{customdata[2]:.3f}/%{customdata[3]:.3f} fF<extra></extra>"
            ),
        )
    )
figure.add_vline(x=10, line_dash="dash", line_color="#6B7280")
figure.add_hline(y=10, line_dash="dash", line_color="#6B7280")
figure.update_layout(
    title="Cross-family agreement in mutual and terminal-to-ground capacitance",
    xaxis={"title": "mutual-capacitance difference (%)", "type": "log"},
    yaxis={"title": "ground-sum capacitance difference (%)", "type": "log"},
    template="plotly_white",
    height=590,
)
figure.show()
"""
    ),
    markdown(
        r"""
This is the strongest reason not to interpret a cosine or a matched mutual
capacitance as a drop-in replacement. Mutual coupling and loading to ground are
different projections of the same field solution. Cross-family inverse design
should retrieve on a multi-output capacitance fingerprint—or refit a supervised
head—not on one scalar alone.

The sampled numbers are stark. Four percent of Tee/generalized pairs match
mutual capacitance within 10%, but only **1.7% of those mutual matches** also
match ground loading within 10%. For both cross-family pairings involving the
transmon, none of the sampled mutual matches also meet the 10% ground criterion.
The apparent lower-left overlap almost disappears when both outputs are
required simultaneously.

## 9. Which named metrics change their meaning across families?

The 48-coordinate physical block is interpretable enough to audit coordinate by
coordinate. We select the twelve metrics with the strongest correlation in any
cohort and compare Spearman correlations pooled and separately by family.

A sign change is not a bug in the encoder. It means the coordinate participates
in different geometric contexts—for example, more conductor area can mean more
facing fingers in one class and a larger but more weakly coupled cross in
another.
"""
    ),
    code(
        r"""
metric_correlations = []
for index, name in enumerate(METRIC_NAMES):
    for cohort_name in ["pooled", *sorted(CLASSES)]:
        rows = np.ones(len(data), dtype=bool) if cohort_name == "pooled" else components == cohort_name
        values = v2[rows, index]
        statistic = 0.0 if float(np.std(values)) <= 1e-12 else spearmanr(values, log_mutual[rows]).statistic
        metric_correlations.append(
            {
                "metric": name.replace("log1p_", "").replace("log_", "").replace("_", " "),
                "cohort": cohort_name,
                "spearman": float(statistic) if np.isfinite(statistic) else 0.0,
            }
        )
metric_correlations = pd.DataFrame(metric_correlations)
priority = (
    metric_correlations.groupby("metric").spearman.apply(lambda values: float(np.max(np.abs(values))))
    .sort_values(ascending=False)
    .head(12)
    .index
)
correlation_matrix = metric_correlations[metric_correlations.metric.isin(priority)].pivot(
    index="metric", columns="cohort", values="spearman"
).loc[priority, ["pooled", *sorted(CLASSES)]]
correlation_matrix.round(3)
"""
    ),
    code(
        r"""
# %% hide input
figure = go.Figure(
    go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        reversescale=True,
        text=np.round(correlation_matrix, 2),
        texttemplate="%{text:+.2f}",
        colorbar={"title": "Spearman"},
        hovertemplate="%{y}<br>%{x}<br>Spearman=%{z:+.4f}<extra></extra>",
    )
)
figure.update_layout(
    title="The same physical coordinate can carry different capacitance trends by family",
    template="plotly_white",
    height=650,
    margin={"l": 250, "b": 150},
)
figure.show()
"""
    ),
    markdown(
        r"""
The pooled column can hide or reverse family-level trends—the geometric version
of Simpson's paradox. This is why a similarity score should be validated in the
retrieval scope where it will be used. Pooled correlation is not evidence that
every cross-family neighbor is meaningful.

The clearest reversal is the normalized second moment `mu20`: Spearman is
**-0.913** in the Tee and -0.689 in the transmon, but +0.681 in the generalized
coupler and only +0.126 pooled. By contrast, the inverse-gap integrals remain
strongly positive in every family (+0.959, +0.942, +0.984). The heatmap
therefore distinguishes portable physical coordinates from family-conditioned
shape summaries.

## 10. See the success cases and the counterexamples in GDS

Aggregate metrics can make a failure sound abstract. We now select four real
pairs from the sampled audit:

- a faithful within-family neighbor;
- a cross-family pair with nearly matched capacitance despite low full-vector
  similarity;
- a cross-family false friend from the high-cosine tail; and
- a cross-family pair selected by the physics-proxy block.

Use the dropdown to change cases. The first two panels are the actual GDS layer
polygons. The third compares eight blockwise cosines for that exact pair. A pair
can be close in one physical sense and far in another; the full-vector cosine
compresses those disagreements into one number.
"""
    ),
    code(
        r"""
def all_pairs(kind):
    frames = []
    for pairing, (left, right) in pair_indices.items():
        is_within = pairing.startswith("within")
        if (kind == "within") != is_within:
            continue
        full = PAIR_VALUES[("v2 full", pairing)]
        physics = PAIR_VALUES[("physics proxy", pairing)]
        frames.append(
            pd.DataFrame(
                {
                    "left": left,
                    "right": right,
                    "pairing": pairing,
                    "full_cosine": full[2],
                    "relative_error": full[3],
                    "delta_log": full[4],
                    "physics_cosine": physics[2],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


within_pairs = all_pairs("within")
cross_pairs = all_pairs("cross")

high_within = within_pairs.full_cosine >= within_pairs.full_cosine.quantile(0.995)
faithful = within_pairs[high_within].sort_values(["relative_error", "full_cosine"], ascending=[True, False]).iloc[0]

matched = cross_pairs[cross_pairs.relative_error <= cross_pairs.relative_error.quantile(0.01)]
matched = matched.sort_values(["full_cosine", "relative_error"]).iloc[0]

high_cross = cross_pairs.full_cosine >= cross_pairs.full_cosine.quantile(0.99)
false_friend = cross_pairs[high_cross].sort_values("relative_error", ascending=False).iloc[0]

high_physics = cross_pairs.physics_cosine >= cross_pairs.physics_cosine.quantile(0.99)
physics_pair = cross_pairs[high_physics].sort_values(["relative_error", "physics_cosine"], ascending=[True, False]).iloc[0]

CASES = {
    "faithful within-family neighbor": faithful,
    "same capacitance, different family geometry": matched,
    "cross-family high-cosine false friend": false_friend,
    "physics-proxy-selected cross-family pair": physics_pair,
}

case_rows = []
for case, row in CASES.items():
    left, right = int(row.left), int(row.right)
    entry = {
        "case": case,
        "family A": components[left],
        "family B": components[right],
        "C A (fF)": mutual_fF[left],
        "C B (fF)": mutual_fF[right],
        "relative difference (%)": row.relative_error,
    }
    for block, unit in UNIT.items():
        entry[f"cosine: {block}"] = float(unit[left] @ unit[right])
    case_rows.append(entry)
case_table = pd.DataFrame(case_rows)
case_table.round(4)
"""
    ),
    code(
        r"""
from scipy import ndimage

from squadds.layouts.geometry_v2 import (
    PHYSICS_BLOCK_SIZE,
    VACUUM_PERMITTIVITY,
    _boundary_samples,
    _raster_frame,
    _rasterize,
    _role_geometry,
    _terminals,
    parameter_block,
    read_layer_geometry,
)

ROLE_COLORS = {"domain": "#C7CDD4", "conductor": "#00798C", "port": "#D1495B", "etch": "#E9C46A"}


def build_gds_index():
    published_manifest = Path(
        hf_hub_download("SQuADDS/SQuADDS_Layouts", "metadata/manifest.parquet", repo_type="dataset")
    )
    index = {
        row.design_id: ("published", row.gds_path)
        for row in pd.read_parquet(published_manifest).itertuples()
    }
    local_manifest = PORT_COMPLETE_ROOT / "metadata/manifest.parquet"
    if not local_manifest.is_file():
        raise FileNotFoundError(f"The GDS pair gallery needs {local_manifest}")
    for row in pd.read_parquet(local_manifest).itertuples():
        index[row.design_id] = ("local", PORT_COMPLETE_ROOT / row.gds_path)
    return index


GDS_INDEX = build_gds_index()


def resolve_gds(design_id):
    origin, value = GDS_INDEX[design_id]
    if origin == "local":
        return value
    return Path(
        hf_hub_download(
            "SQuADDS/SQuADDS_Layouts",
            filename=value,
            repo_type="dataset",
        )
    )


def polygon_traces(shape, role):
    traces = []
    for polygon in getattr(shape, "geoms", [shape]):
        if polygon.geom_type != "Polygon":
            continue
        x, y = polygon.exterior.xy
        traces.append(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                fill="toself" if role != "domain" else None,
                fillcolor=ROLE_COLORS[role],
                opacity=0.78 if role != "domain" else 0.35,
                line={"color": ROLE_COLORS[role], "width": 1.2},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        for interior in polygon.interiors:
            hx, hy = interior.xy
            traces.append(
                go.Scatter(
                    x=list(hx),
                    y=list(hy),
                    mode="lines",
                    line={"color": ROLE_COLORS[role], "width": 1.0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return traces


def gds_traces(design_id):
    grouped = _role_geometry(read_layer_geometry(resolve_gds(design_id)), None)
    traces = []
    for role in ("domain", "etch", "conductor", "port"):
        for _, shape in grouped[role]:
            traces.extend(polygon_traces(shape, role))
    return traces
"""
    ),
    code(
        r"""
# %% hide input
figure = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["design A", "design B", "the same pair in eight similarity spaces"],
    column_widths=[0.32, 0.32, 0.36],
    horizontal_spacing=0.06,
)
case_trace_ranges = []
case_titles = []
for case, row in CASES.items():
    left, right = int(row.left), int(row.right)
    start = len(figure.data)
    for column, design_index in ((1, left), (2, right)):
        for trace in gds_traces(data.iloc[design_index].design_id):
            figure.add_trace(trace, row=1, col=column)
    cosines = [float(UNIT[block][left] @ UNIT[block][right]) for block in BLOCKS]
    figure.add_trace(
        go.Bar(
            x=cosines,
            y=list(BLOCKS),
            orientation="h",
            marker_color=[BLOCK_COLORS[name] for name in BLOCKS],
            text=[f"{value:+.3f}" for value in cosines],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{y}<br>cosine=%{x:+.4f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    case_trace_ranges.append((start, len(figure.data)))
    case_titles.append(
        f"{case}: {components[left]} {mutual_fF[left]:.3f} fF vs "
        f"{components[right]} {mutual_fF[right]:.3f} fF; difference {row.relative_error:.1f}%"
    )

for trace in figure.data:
    trace.visible = False
for index in range(*case_trace_ranges[0]):
    figure.data[index].visible = True

buttons = []
for case_index, case in enumerate(CASES):
    visible = [False] * len(figure.data)
    for index in range(*case_trace_ranges[case_index]):
        visible[index] = True
    buttons.append(
        {
            "label": case,
            "method": "update",
            "args": [{"visible": visible}, {"title": case_titles[case_index]}],
        }
    )
for column in (1, 2):
    figure.update_yaxes(scaleanchor=f"x{'' if column == 1 else column}", scaleratio=1, row=1, col=column)
figure.update_xaxes(title_text="x (um)", row=1, col=1)
figure.update_xaxes(title_text="x (um)", row=1, col=2)
figure.update_yaxes(title_text="y (um)", row=1, col=1)
figure.update_xaxes(title_text="standardized cosine", range=[-1.05, 1.15], row=1, col=3)
figure.update_layout(
    title=case_titles[0],
    template="plotly_white",
    height=610,
    margin={"l": 90, "b": 120},
    updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.17}],
)
figure.show()
"""
    ),
    markdown(
        r"""
The counterexamples are the central lesson of this notebook. Equal capacitance
does not imply equal geometry: different shapes can trade boundary length,
distance, and terminal scale to reach the same scalar result. Conversely, a
large aggregate cosine can hide a disagreement in the block that matters for a
particular target.

The selected equal-capacitance generalized/transmon pair differs by only 1.28%
in mutual C but has full-v2 cosine **-0.726**. The high-cosine cross-family false
friend goes the other way: cosine +0.219 but 0.312 versus 3.761 fF, a 169%
difference. The physics-selected Tee/generalized pair matches at 8.7635 versus
8.7628 fF with physics cosine +0.966 even though its full cosine is only +0.038.
One scalar ranking cannot express those blockwise disagreements.

## 11. The whole v2 algorithm, now on all three component families

Tutorial 19 replayed `universal-geometry-v2` on one generalized interdigital
capacitor. The same anatomy is more informative when the three geometries are
placed side by side. For each family we choose the design nearest its median
mutual capacitance, then run the **same six encoder views**:

1. discover and order terminals from conductor and port layers;
2. measure physical geometry on an adaptive raster;
3. sample terminal boundaries to build absolute-distance coupling spectra;
4. summarize the conductor mask and contour in shape coordinates;
5. classify schema-free design parameters; and
6. solve the two-dimensional boundary-element proxy and dilation topology.

Drag the slider. The top row shows what the encoder is looking at for all three
families at that step. The bottom row shows the corresponding raw 512-vector;
the coordinates written by the active step are colored and all other
coordinates are grey. This is the Tutorial 19 visualization generalized to the
Tee and TransmonCross, not a projection or a learned explanation.
"""
    ),
    code(
        r"""
FAMILY_ANATOMY_ORDER = [
    "GeneralizedCapNInterdigital",
    "CapNInterdigitalTee",
    "TransmonCross",
]
FAMILY_SHORT = {
    "GeneralizedCapNInterdigital": "Generalized interdigital",
    "CapNInterdigitalTee": "Interdigital Tee",
    "TransmonCross": "Transmon cross / claw",
}
TERMINAL_COLORS = ["#00798C", "#D1495B", "#6A4C93", "#E9C46A"]
ANATOMY_PORT_COLOR = "#F59E0B"

representatives = {}
for component in FAMILY_ANATOMY_ORDER:
    rows = np.flatnonzero(components == component)
    median_capacitance = np.median(mutual_fF[rows])
    representatives[component] = int(rows[np.argmin(np.abs(mutual_fF[rows] - median_capacitance))])

needed_ids = {data.iloc[index].design_id for index in representatives.values()}
options_by_id = {}
for component, path in database.items():
    for database_row in json.loads(path.read_text()):
        design_options = database_row["design"]["design_options"]
        design_id = canonical_design_id(component, design_options)
        if design_id in needed_ids:
            options_by_id[design_id] = design_options


def anatomy_polygon_traces(shape, color, label, *, fill=True, opacity=0.78):
    traces = []
    for polygon in getattr(shape, "geoms", [shape]):
        if polygon.geom_type != "Polygon":
            continue
        x, y = polygon.exterior.xy
        traces.append(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                fill="toself" if fill else None,
                fillcolor=color,
                opacity=opacity,
                line={"color": color, "width": 1.3},
                showlegend=False,
                hovertemplate=f"{label}<extra></extra>",
            )
        )
        for interior in polygon.interiors:
            hx, hy = interior.xy
            traces.append(
                go.Scatter(
                    x=list(hx),
                    y=list(hy),
                    mode="lines",
                    line={"color": color, "width": 1.0},
                    showlegend=False,
                    hovertemplate=f"{label}: ground-plane hole<extra></extra>",
                )
            )
    return traces


def boundary_charge_density(terminals):
    centers, lengths, owners = [], [], []
    for index, terminal in enumerate(terminals[:4]):
        coordinates, weights = _boundary_samples(terminal, target=160)
        centers.append(coordinates)
        lengths.append(weights)
        owners.append(np.full(len(coordinates), index))
    points = np.vstack(centers)
    segment = np.concatenate(lengths)
    labels = np.concatenate(owners)
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(delta**2, axis=2)) * 1e-6
    np.fill_diagonal(distance, 1.0)
    green = -np.log(distance) / (2 * np.pi * VACUUM_PERMITTIVITY)
    np.fill_diagonal(
        green,
        -(np.log(segment * 1e-6 / 2) - 1) / (2 * np.pi * VACUUM_PERMITTIVITY),
    )
    selector = np.stack([(labels == index).astype(float) for index in range(len(terminals[:4]))], axis=1)
    try:
        charges = np.linalg.solve(green, selector)
    except np.linalg.LinAlgError:
        charges = np.linalg.lstsq(green, selector, rcond=None)[0]
    density = charges[:, 0] / np.maximum(segment, 1e-12)
    return points, density


def anatomy_views(design_index):
    design_id = data.iloc[design_index].design_id
    geometry = read_layer_geometry(resolve_gds(design_id))
    grouped = _role_geometry(geometry, None)
    conductor = shapely.union_all([shape for _, shape in grouped["conductor"]])
    terminals = _terminals(conductor, grouped["port"])
    assert len(terminals) == 2

    frame = _raster_frame(conductor.bounds)
    conductor_mask = _rasterize(conductor, frame)
    interior = ndimage.distance_transform_edt(conductor_mask) * frame[2]
    raster_x = np.linspace(frame[0], frame[0] + frame[3], conductor_mask.shape[1])
    raster_y = np.linspace(frame[1] + frame[3], frame[1], conductor_mask.shape[0])

    role_traces = []
    for _, domain in grouped["domain"]:
        role_traces.extend(
            anatomy_polygon_traces(domain, ROLE_COLORS["domain"], "ground plane", fill=False, opacity=0.42)
        )
    for terminal_index, terminal in enumerate(terminals):
        role_traces.extend(
            anatomy_polygon_traces(
                terminal,
                TERMINAL_COLORS[terminal_index],
                f"ordered terminal {terminal_index}",
            )
        )
    for port_index, (_, port) in enumerate(grouped["port"]):
        role_traces.extend(
            anatomy_polygon_traces(port, ANATOMY_PORT_COLOR, f"port marker {port_index}", opacity=0.95)
        )

    first_points, _ = _boundary_samples(terminals[0], target=256)
    second_points, _ = _boundary_samples(terminals[1], target=256)
    first_distance = shapely.distance(shapely.points(first_points), terminals[1])
    second_distance = shapely.distance(shapely.points(second_points), terminals[0])
    distance_limit = float(np.quantile(np.r_[first_distance, second_distance], 0.95))
    coupling_traces = [
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker={
                "size": 5,
                "color": distances,
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": max(distance_limit, 1e-9),
            },
            showlegend=False,
            hovertemplate=f"terminal {terminal_index}<br>distance to other terminal=%{{marker.color:.3f}} um<extra></extra>",
        )
        for terminal_index, (points, distances) in enumerate(
            ((first_points, first_distance), (second_points, second_distance))
        )
    ]

    _, parameter_metadata = parameter_block(options_by_id[design_id])
    parameter_counts = pd.Series(parameter_metadata["parameter_classes"]).value_counts()
    parameter_names = parameter_counts.index.tolist()
    parameter_trace = go.Bar(
        x=np.arange(len(parameter_counts)),
        y=parameter_counts.to_numpy(),
        marker_color="#E9C46A",
        customdata=np.asarray(parameter_names)[:, None],
        text=parameter_names,
        textposition="outside",
        showlegend=False,
        hovertemplate="%{customdata[0]}: %{y} options<extra></extra>",
    )

    charge_points, density = boundary_charge_density(terminals)
    density_limit = max(float(np.quantile(np.abs(density), 0.98)), 1e-30)
    physics_trace = go.Scatter(
        x=charge_points[:, 0],
        y=charge_points[:, 1],
        mode="markers",
        marker={
            "size": 6,
            "color": density,
            "colorscale": "RdBu",
            "cmid": 0,
            "cmin": -density_limit,
            "cmax": density_limit,
        },
        showlegend=False,
        hovertemplate="charge density=%{marker.color:.3g}<extra></extra>",
    )

    return [
        role_traces,
        [
            go.Heatmap(
                z=np.where(conductor_mask, interior, np.nan),
                x=raster_x,
                y=raster_y,
                colorscale="Magma",
                showscale=False,
                hovertemplate="interior half-width=%{z:.3f} um<extra></extra>",
            )
        ],
        coupling_traces,
        [
            go.Heatmap(
                z=conductor_mask.astype(float),
                x=raster_x,
                y=raster_y,
                colorscale=[[0, "#FFFFFF"], [1, "#6A4C93"]],
                showscale=False,
                hoverinfo="skip",
            )
        ],
        [parameter_trace],
        [physics_trace],
    ]


PIPELINE_STEPS = [
    ("1-2. roles and ordered terminals", None, None, "#C7CDD4"),
    ("3. physical metrics", 0, METRIC_BLOCK_SIZE, "#F4A261"),
    ("4. coupling spectrum", METRIC_BLOCK_SIZE, COUPLING_STOP, "#00798C"),
    ("5. shape spectrum", COUPLING_STOP, SHAPE_STOP, "#6A4C93"),
    ("6. parameter statistics", SHAPE_STOP, PARAMETER_STOP, "#E9C46A"),
    ("7. physics proxy", PARAMETER_STOP, V2_DIMENSIONS, "#2A9D8F"),
]
ANATOMY_VIEWS = {
    component: anatomy_views(representatives[component]) for component in FAMILY_ANATOMY_ORDER
}

occupancy_rows = []
for component in FAMILY_ANATOMY_ORDER:
    vector = v2[representatives[component]]
    for label, start, stop, _ in PIPELINE_STEPS[1:]:
        occupancy_rows.append(
            {
                "family": FAMILY_SHORT[component],
                "block": label.split(". ", 1)[1],
                "coordinates": stop - start,
                "nonzero coordinates": int(np.count_nonzero(np.abs(vector[start:stop]) > 1e-12)),
            }
        )
pd.DataFrame(occupancy_rows).pivot(
    index="block", columns="family", values="nonzero coordinates"
).reindex([step[0].split(". ", 1)[1] for step in PIPELINE_STEPS[1:]])
"""
    ),
    code(
        r"""
# %% hide input
subplot_titles = [
    f"{FAMILY_SHORT[component]}<br>C={mutual_fF[representatives[component]]:.3f} fF"
    for component in FAMILY_ANATOMY_ORDER
] + ["raw 512-coordinate vector"] * 3
walk = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=subplot_titles,
    vertical_spacing=0.15,
    horizontal_spacing=0.055,
    row_heights=[0.64, 0.36],
)

step_trace_ranges = []
step_titles = []
for step_index, (label, start_coordinate, stop_coordinate, color) in enumerate(PIPELINE_STEPS):
    trace_start = len(walk.data)
    for column, component in enumerate(FAMILY_ANATOMY_ORDER, start=1):
        for trace in ANATOMY_VIEWS[component][step_index]:
            walk.add_trace(trace, row=1, col=column)

        coordinate_colors = ["#E5E7EB"] * V2_DIMENSIONS
        if start_coordinate is not None:
            coordinate_colors[start_coordinate:stop_coordinate] = [color] * (stop_coordinate - start_coordinate)
        walk.add_trace(
            go.Bar(
                x=np.arange(V2_DIMENSIONS),
                y=v2[representatives[component]],
                marker_color=coordinate_colors,
                showlegend=False,
                hovertemplate="coordinate %{x}<br>raw value=%{y:.4g}<extra></extra>",
            ),
            row=2,
            col=column,
        )
    step_trace_ranges.append((trace_start, len(walk.data)))
    written = (
        "terminal identities established; no coordinates written"
        if start_coordinate is None
        else f"coordinates {start_coordinate}-{stop_coordinate - 1} ({stop_coordinate - start_coordinate} values)"
    )
    step_titles.append(f"{label} -> {written}")

for trace in walk.data:
    trace.visible = False
for trace_index in range(*step_trace_ranges[0]):
    walk.data[trace_index].visible = True

slider_steps = []
for step_index, (label, _, _, _) in enumerate(PIPELINE_STEPS):
    visible = [False] * len(walk.data)
    for trace_index in range(*step_trace_ranges[step_index]):
        visible[trace_index] = True
    slider_steps.append(
        {
            "label": label.split(".", 1)[0],
            "method": "update",
            "args": [{"visible": visible}, {"title": step_titles[step_index]}],
        }
    )

for column in range(1, 4):
    axis_suffix = "" if column == 1 else str(column)
    walk.update_yaxes(scaleanchor=f"x{axis_suffix}", scaleratio=1, row=1, col=column)
    walk.update_xaxes(title_text="coordinate", range=[0, V2_DIMENSIONS - 1], row=2, col=column)
    walk.update_yaxes(title_text="raw value", row=2, col=column)
walk.update_yaxes(title_text="y (um)", row=1, col=1)
walk.update_layout(
    title=step_titles[0],
    template="plotly_white",
    height=900,
    bargap=0,
    margin={"l": 75, "r": 35, "b": 90, "t": 120},
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "encoder step "},
            "pad": {"t": 55},
            "steps": slider_steps,
        }
    ],
)
walk.show()
"""
    ),
    markdown(
        r"""
The coordinate boundaries are identical in every column: metrics 0–47,
coupling 48–239, shape 240–367, parameters 368–463, and physics 464–511. What
changes is the geometry that populates them. The Generalized and Tee columns
turn facing fingers into dense short-range boundary samples; the Transmon
column turns the cross/claw separation and asymmetry into a very different
distance field. On the roles step, orange port markers touch the two separately
colored terminals, making the terminal ordering visible rather than implicit.

The lower-row bars also prevent a common misreading of the embedding. Equal
block width does not imply equal information density or equal importance. For
example, the parameter block describes the source design schema rather than the
GDS contour, while the physics block applies one shared, deliberately simplified
field construction to every family. The vector is a concatenation of views,
not a claim that a comb and a cross are the same object.

## 12. A block-by-pairing scorecard

The final heatmap summarizes the pairwise calibration experiment. More negative
is better: it means high cosine is associated with small `|delta log C|`.
Within-family and cross-family columns are shown together so a block cannot earn
a universal-similarity claim from one easy setting.
"""
    ),
    code(
        r"""
# %% hide input
scorecard = correlations.pivot(
    index="block", columns="pairing", values="Spearman(cosine, |delta log C|)"
).reindex(index=list(BLOCKS), columns=pair_names)
figure = go.Figure(
    go.Heatmap(
        z=scorecard,
        x=scorecard.columns,
        y=scorecard.index,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        text=np.round(scorecard, 2),
        texttemplate="%{text:+.2f}",
        colorbar={"title": "Spearman"},
        hovertemplate="%{y}<br>%{x}<br>Spearman=%{z:+.4f}<extra></extra>",
    )
)
figure.update_layout(
    title="No v2 block is a universally calibrated capacitance metric",
    template="plotly_white",
    height=620,
    margin={"l": 205, "b": 190},
)
figure.show()
"""
    ),
    markdown(
        r"""
## 13. What geometric similarity means—and does not mean

### Established

- A v2 cosine is meaningful only after naming the block, scaling reference, and
  candidate scope. "Most similar in v2" is incomplete without those choices.
- Inside a component family, local v2 neighborhoods track capacitance well. The
  calibration curves slope downward and five-neighbor macro R2 is 0.960–0.973
  for the major blocks, with typical median errors around 0.4–0.5 fF.
- Across families, geometric similarity and equal capacitance separate. The GDS
  gallery contains both equal-capacitance/different-shape pairs and
  high-cosine/different-capacitance false friends.
- Across the two coupler families, geometry-only cosine ranks capacitance
  distance well (Spearman -0.714). The same full-vector test is misleading for
  Tee versus transmon (+0.453); the physics proxy is the only useful block for
  that pairing (-0.340).
- The coupling spectrum adds information a minimum gap cannot: how much terminal
  boundary occurs at each absolute distance. Ordered ports preserve the very
  strong terminal asymmetry of `TransmonCross` rather than averaging cross and
  claw together.
- The unified ground convention is scientifically consequential. Both active
  terminals now populate terminal-to-ground spectra in every family, so those
  48 coordinates have the same physical interpretation across the cohort.
- Family identity is strongest in shape space; cross-family overlap is greater
  in physical metrics and the physics proxy. A block that separates classes is
  not defective—it answers a different similarity question.
- R2, RMSE, and median absolute error are all necessary. R2 measures variation
  explained in log space, median error describes the typical retrieval, and
  RMSE reveals rare catastrophic neighbors. At five cross-family neighbors the
  full vector scores -6.555 R2 and 14.15 fF RMSE; the physics proxy is least bad
  at -1.048 and 5.30 fF.
- Matching one capacitance does not match the capacitance matrix. Only 1.7% of
  sampled Tee/generalized pairs that match mutual C within 10% also match ground
  loading within 10%; the sampled transmon cross-family matches reach 0%.

### Not established

- High full-vector cosine does **not** establish electrostatic equivalence
  across component families.
- Equal mutual capacitance does not imply substitutable layout geometry, ground
  capacitance, fabrication footprint, or frequency behavior.
- A two-dimensional projection does not certify nearest neighbors; it is only a
  hypothesis-generating view of a much higher-dimensional space.
- These results do not validate similarity for eigenmode devices or measured
  chips. All three targets here are simulated electrostatic capacitances.
- Standardizing on this balanced cohort is useful for analysis but is not part
  of the catalogue-independent `universal-geometry-v2` definition.

### Practical rule

Use **within-family full or geometry-only similarity** for interpolation and
duplicate/design retrieval. Use **physical metrics, coupling, or the physics
proxy** to propose cross-family candidates, then calibrate them with target
labels. For a genuinely new family, follow Tutorial 20: acquire roughly ten
well-spread simulations and adapt a supervised head. Similarity is an
applicability signal and a candidate generator—not a replacement for that
calibration.
"""
    ),
]

notebook["cells"] = CELLS
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, str(OUTPUT))
print(f"Wrote {OUTPUT} with {len(CELLS)} cells.")
