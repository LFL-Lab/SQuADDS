#!/usr/bin/env python
"""Build Tutorial 18: the universal-geometry-v2 layout embedding study."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("tutorials/Tutorial-18_Universal_Geometry_v2_Embeddings.ipynb")


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

notebook["cells"] = [
    markdown(
        """
# Tutorial 18: universal-geometry-v2, an embedding a stranger can compute

Tutorials 14 through 17 established that a shared layout representation makes
transfer learning possible across geometry regimes and even across component
classes. They also exposed the limits of the two existing standards. This
tutorial introduces **`universal-geometry-v2`** and tests it against
`static-shape-v0` on exactly the Tutorial 16 protocol.

The motivating scenario is concrete. A group at another university has 1,000 GDS
files for a capacitor topology we have never seen, a CSV of simulated results,
and 28 design parameters with their own names. They have nowhere near enough
data to train a specialist. We want to project their designs into the space our
23,000 designs already occupy, find which family they land near, and transfer.

For that to work the encoder must satisfy one requirement that v0 and v1 do not:

> The embedding of a layout must depend on **that layout alone** - never on which
> other layouts happen to be encoded alongside it.

By the end you will be able to:

1. state the two structural defects that cap v0 and v1 performance;
2. read every v2 block as a physical measurement rather than a fitted statistic;
3. verify that a single hand-derived v2 coordinate tracks simulated capacitance;
4. compare v0 and v2 under an identical model, split, and label budget; and
5. separate how much of the gain comes from geometry versus design parameters.

> **Scope.** Capacitance is only ever a target. No simulation result enters any
> embedding in this notebook.
"""
    ),
    code(
        """
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, t as student_t, ttest_1samp

from squadds.layouts import (
    TransferRidgeRegressor,
    V0KernelFeatureProjector,
    canonical_design_id,
    compress_v0_embeddings,
    regression_scores,
)
from squadds.layouts.geometry_v2 import (
    COUPLING_BLOCK_SIZE,
    COUPLING_EDGES,
    METRIC_BLOCK_SIZE,
    METRIC_NAMES,
    PARAMETER_BLOCK_SIZE,
    PHYSICS_BLOCK_SIZE,
    SHAPE_BLOCK_SIZE,
    V2_DIMENSIONS,
    universal_v2_schema,
)

pio.renderers.default = "notebook_connected"
pd.set_option("display.max_columns", 30)
logging.getLogger("httpx").setLevel(logging.WARNING)

TARGETS = ["C_NS_fF", "C_NG_fF", "C_SG_fF"]
COUNTS = list(range(2, 15))
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
REPEATS = 12
TEST_FRACTION = 0.25
BASE_FINGER_COUNT = 8
SEED = 16
ALPHA = 0.03
CONFIDENCE = 0.95
PALETTE = {"v0": "#D1495B", "v2": "#00798C"}

CHECKPOINTS = Path(os.getenv("SQUADDS_TUTORIAL18_CACHE", Path.home() / ".cache/squadds/tutorial18"))
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# universal-geometry-v2 is not yet published on the Hugging Face dataset, so the
# table is built locally.  This is the exact command that produces it:
#
#   uv run --extra gds python scripts/build_v2_embeddings.py \\
#       <layouts>/metadata/manifest.parquet <layouts> <output> \\
#       --design-json GeneralizedCapNInterdigital=<db>.json \\
#       --component-name GeneralizedCapNInterdigital
V2_TABLE = Path(os.getenv("SQUADDS_V2_TABLE", CHECKPOINTS / "universal-geometry-v2.parquet"))
if not V2_TABLE.is_file():
    raise FileNotFoundError(f"Build the v2 table first; expected it at {V2_TABLE}.")

database_path = Path(
    hf_hub_download(
        "SQuADDS/SQuADDS_DB",
        "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
        repo_type="dataset",
    )
)
v0_path = Path(
    hf_hub_download(
        "SQuADDS/SQuADDS_Layout_Embeddings",
        "metadata/static-embedding-v0.parquet",
        repo_type="dataset",
    )
)
print("v2 table :", V2_TABLE)
print("v0 table :", v0_path.name)
print("targets  :", database_path.name)
"""
    ),
    markdown(
        """
## 1. Two structural defects, and why they cap v0 and v1

### Defect one: the embedding depends on its neighbours

`build_static_embeddings` derives `parameter_mean`, `parameter_std`, and the ten
moment statistics from whichever rows are written together. The v1 builder goes
further and also selects its 768 spectral frequencies by variance across the
batch. Both are **fit on write**.

The consequence is not subtle. If our collaborators run the v0 builder on their
1,000 designs, their normalization constants are theirs, not ours, and their
vectors are not comparable to our 23,000 even though the lengths match. Fixed
dimension is necessary for a foundation model; it is nowhere near sufficient.
What is actually required is **commensurability**: coordinate *k* must mean the
same physical thing for everyone, forever.

v2 takes the only route that guarantees this. Every coordinate is a physical
measurement - micrometers, inverse micrometers, farads per meter - accumulated
onto frozen bin edges. There is no catalogue-derived constant anywhere in the
encoder, so `encode` is a pure function of one GDS file and one parameter
mapping.

### Defect two: both standards are blind to size

`rasterize_functional_shape` crops each layout to its own functional bounds and
scales it to fill a 96 by 96 window. A design and its exact 4x enlargement
therefore produce **identical** shape blocks. v1 inherits this and compounds it
by normalizing its signed-distance channel by that layout's own maximum, so gaps
are stored as a fraction of the largest gap rather than in micrometers.

For capacitance this discards the dominant variable. The cell below builds one
interdigital capacitor and an exact 4x copy, then encodes both with each
standard.
"""
    ),
    code(
        """
import klayout.db as kdb

from squadds.layouts.embeddings import rasterize_functional_shape
from squadds.layouts.geometry_v2 import encode


def write_interdigital(path, *, finger_count=6, finger_gap=3.0, scale=1.0):
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")

    def box(x0, y0, x1, y1):
        return kdb.Box(*[int(round(value * scale * 1000)) for value in (x0, y0, x1, y1)])

    width, half, spine = 4.0, 20.0, 6.0
    pitch = 2 * (width + finger_gap)
    span = finger_count * pitch
    conductors = kdb.Region()
    conductors.insert(box(0, half, span, half + spine))
    conductors.insert(box(0, -half - spine, span, -half))
    for index in range(finger_count):
        left = index * pitch + finger_gap
        conductors.insert(box(left, -half + finger_gap, left + width, half))
        facing = left + width + finger_gap
        conductors.insert(box(facing, -half, facing + width, half - finger_gap))
    ground = kdb.Region(box(-span, -3 * half, 2 * span, 3 * half))
    top.shapes(layout.layer(1, 0)).insert(ground - conductors.sized(int(round(10 * scale * 1000))))
    top.shapes(layout.layer(1, 10)).insert(conductors)
    top.shapes(layout.layer(2, 0)).insert(box(span / 2 - 1, half + spine, span / 2 + 1, half + spine + 2))
    top.shapes(layout.layer(3, 0)).insert(box(span / 2 - 1, -half - spine - 2, span / 2 + 1, -half - spine))
    layout.write(str(path))
    return path


demo = Path(os.getenv("TMPDIR", "/tmp")) / "squadds-tutorial18"
demo.mkdir(parents=True, exist_ok=True)
options = {"finger_count": 6, "finger_length": "40um", "finger_width": "4um", "finger_gap": "3um"}

small = write_interdigital(demo / "small.gds")
large = write_interdigital(demo / "large.gds", scale=4.0)
wide = write_interdigital(demo / "wide.gds", finger_gap=9.0)

v0_small = rasterize_functional_shape(small, "CapNInterdigitalTee")[0].reshape(-1)
v0_large = rasterize_functional_shape(large, "CapNInterdigitalTee")[0].reshape(-1)
v2_small, v2_large, v2_wide = (encode(path, options) for path in (small, large, wide))

comparison = pd.DataFrame(
    [
        {
            "perturbation": "4x uniform enlargement",
            "v0 shape block changes": f"{np.abs(v0_small - v0_large).max():.3g}",
            "v2 vector changes": f"{np.abs(v2_small - v2_large).max():.3g}",
        },
        {
            "perturbation": "finger gap 3um -> 9um",
            "v0 shape block changes": f"{np.abs(v0_small - rasterize_functional_shape(wide, 'CapNInterdigitalTee')[0].reshape(-1)).max():.3g}",
            "v2 vector changes": f"{np.abs(v2_small - v2_wide).max():.3g}",
        },
    ]
)
print(comparison.to_string(index=False))
print()
print(f"v0 bitmaps identical under 4x enlargement: {np.array_equal(v0_small, v0_large)}")
"""
    ),
    code(
        """
# %% hide input
gap_index = METRIC_NAMES.index("log1p_minimum_pair_gap_um")
coupling_slice = slice(METRIC_BLOCK_SIZE, METRIC_BLOCK_SIZE + 24)
centers = np.sqrt(COUPLING_EDGES[:-1] * COUPLING_EDGES[1:])

figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "v0 shape block: enlargement is invisible",
        "v2 coupling spectrum: separation is measured in um",
    ],
    horizontal_spacing=0.12,
)
figure.add_trace(
    go.Bar(
        x=["4x enlargement", "gap 3um -> 9um"],
        y=[
            float(np.abs(v0_small - v0_large).max()),
            float(np.abs(v0_small - rasterize_functional_shape(wide, "CapNInterdigitalTee")[0].reshape(-1)).max()),
        ],
        marker_color=PALETTE["v0"],
        text=["exactly 0", "non-zero"],
        textposition="outside",
        showlegend=False,
        hovertemplate="%{x}<br>max |change| = %{y:.3g}<extra></extra>",
    ),
    row=1,
    col=1,
)
for vector, label, dash in ((v2_small, "3 um gap", "solid"), (v2_wide, "9 um gap", "dash"), (v2_large, "4x enlarged", "dot")):
    figure.add_trace(
        go.Scatter(
            x=centers,
            y=vector[coupling_slice],
            mode="lines+markers",
            name=label,
            line={"dash": dash, "width": 2.5},
            hovertemplate=f"<b>{label}</b><br>separation=%{{x:.2f}} um<br>log1p(boundary um)=%{{y:.2f}}<extra></extra>",
        ),
        row=1,
        col=2,
    )
figure.update_yaxes(title_text="max |coordinate change|", row=1, col=1)
figure.update_xaxes(title_text="conductor separation (um)", type="log", row=1, col=2)
figure.update_yaxes(title_text="log1p facing boundary length", row=1, col=2)
figure.update_layout(
    title="The same two perturbations, seen by each standard",
    template="plotly_white",
    height=470,
)
figure.show()
"""
    ),
    markdown(
        """
The left panel is the defect stated numerically: a 4x enlargement moves the v0
shape block by exactly zero. The right panel shows v2 placing each design's
facing boundary length at its true separation in micrometers, so widening the
gap slides the spectrum right and enlarging the device slides it further right
again. Nothing here is fitted; the bin edges are frozen constants.
"""
    ),
    markdown(
        """
## 2. What v2 measures

v2 is 512 dimensions in five blocks. Read the schema as a list of physical
measurements rather than learned features.

| Block | Dimensions | What it measures |
| --- | ---: | --- |
| Physical metrics | 48 | Absolute extent, per-role area and perimeter, conductor width percentiles, gap integrals, symmetry |
| Coupling spectrum | 192 | Facing boundary length per terminal pair, per absolute separation bin, plus terminal-to-ground |
| Shape spectrum | 128 | Two-point correlation, terminal cross-correlation, conductor width distribution, contour harmonics |
| Parameter statistics | 96 | Dimension-typed order statistics and dimension-scoped signed hashing |
| Physics proxy | 48 | Two-dimensional boundary-element capacitance matrix and dilation topology |

Two design choices matter more than the rest.

**Terminals are found, not declared.** The conductor set is split into connected
components and ordered by port marker, then by area. A foreign layout with a
different pin vocabulary still produces terminal 0 and terminal 1.

**The parameter block never needs a name registry.** Each option is classified
into a physical dimension - length, count, angle, boolean - and summarized with
order statistics inside that class. The smallest length in a design is a
meaningful, comparable quantity whether the contributor calls it `finger_gap` or
`digit_separation`. That is what lets a 28-parameter foreign schema and our
40-parameter one occupy the same 96 coordinates.
"""
    ),
    code(
        """
# %% hide input
schema = universal_v2_schema()
blocks = pd.DataFrame(
    [
        {"standard": "v0", "block": "parameters", "dimensions": 1},
        {"standard": "v0", "block": "metrics", "dimensions": 10},
        {"standard": "v0", "block": "raster / spectrum", "dimensions": 96 * 96},
        {"standard": "v1", "block": "parameters", "dimensions": 224},
        {"standard": "v1", "block": "metrics", "dimensions": 32},
        {"standard": "v1", "block": "raster / spectrum", "dimensions": 768},
        {"standard": "v2", "block": "parameters", "dimensions": PARAMETER_BLOCK_SIZE},
        {"standard": "v2", "block": "metrics", "dimensions": METRIC_BLOCK_SIZE},
        {"standard": "v2", "block": "raster / spectrum", "dimensions": SHAPE_BLOCK_SIZE},
        {"standard": "v2", "block": "coupling spectrum", "dimensions": COUPLING_BLOCK_SIZE},
        {"standard": "v2", "block": "physics proxy", "dimensions": PHYSICS_BLOCK_SIZE},
    ]
)
colors = {
    "parameters": "#E9C46A",
    "metrics": "#F4A261",
    "raster / spectrum": "#6A4C93",
    "coupling spectrum": "#00798C",
    "physics proxy": "#2A9D8F",
}
figure = go.Figure()
for block, color in colors.items():
    frame = blocks.query("block == @block")
    figure.add_trace(
        go.Bar(
            x=frame["standard"],
            y=frame["dimensions"],
            name=block,
            marker_color=color,
            hovertemplate="%{x} - " + block + "<br>%{y} dimensions<extra></extra>",
        )
    )
figure.update_layout(
    title="9,227 then 1,024 then 512 dimensions, with progressively more of them physical",
    barmode="stack",
    yaxis={"title": "dimensions (log scale)", "type": "log"},
    template="plotly_white",
    height=500,
)
figure.show()
print(f"v2 dimensions: {schema['dimensions']}   fitted on catalogue: {schema['fitted_on_catalogue']}")
"""
    ),
    markdown(
        """
## 3. Does the coupling spectrum actually carry the physics?

This is the block the whole design rests on, so it deserves a direct test before
any model is fitted.

For coplanar conductors the mutual capacitance is approximately an integral of a
kernel over facing boundary length at a given separation,

$$C_{ij} \\approx \\varepsilon \\int f(d)\\, \\mathrm{d}L .$$

The coupling spectrum is a discretization of $\\mathrm{d}L$ against absolute $d$,
which means a model **linear in this block** can represent the kernel. The
metric block also stores the simplest such contraction directly:
$\\sum_k L_k / d_k$.

If the reasoning is right, that one hand-derived coordinate should track
simulated capacitance without any fitting at all.
"""
    ),
    code(
        """
def load_targets(path):
    rows = json.loads(Path(path).read_text())
    records = []
    for row in rows:
        design_options = row["design"]["design_options"]
        results = row["sim_results"]
        records.append(
            {
                "design_id": canonical_design_id("GeneralizedCapNInterdigital", design_options),
                "source_id": row["notes"]["source_id"],
                "finger_count": int(design_options["finger_count"]),
                "finger_gap_um": float(str(design_options["finger_gap_north_south"]).replace("um", "")),
                "C_NS_fF": results["north_to_south"],
                "C_NG_fF": results["north_to_ground"],
                "C_SG_fF": results["south_to_ground"],
            }
        )
    return pd.DataFrame(records).drop_duplicates("design_id")


def load_v0_compact(path, keep):
    parquet = pq.ParquetFile(path)
    identifiers, blocks = [], []
    for batch in parquet.iter_batches(batch_size=256, columns=["design_id", "component_name", "embedding"]):
        frame = batch.to_pandas()
        frame = frame[(frame.component_name == "GeneralizedCapNInterdigital") & frame.design_id.isin(keep)]
        if frame.empty:
            continue
        matrix = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
        blocks.append(compress_v0_embeddings(matrix, pooled_shape_size=12).astype(np.float32))
        identifiers.extend(frame["design_id"].tolist())
    return pd.Series(identifiers), np.vstack(blocks)


targets = load_targets(database_path)
v2_frame = pd.read_parquet(V2_TABLE).drop_duplicates("design_id")
v2_all = np.vstack(v2_frame["embedding"].to_numpy()).astype(np.float32)
v0_ids, v0_all = load_v0_compact(v0_path, set(v2_frame.design_id))

paired = (
    targets.merge(v2_frame[["design_id", "campaign"]].assign(v2_row=range(len(v2_frame))), on="design_id")
    .merge(pd.DataFrame({"design_id": v0_ids}).assign(v0_row=range(len(v0_ids))), on="design_id")
    .drop_duplicates("design_id")
    .reset_index(drop=True)
)
v2 = v2_all[paired["v2_row"].to_numpy()]
v0 = v0_all[paired["v0_row"].to_numpy()]
y = paired[TARGETS].to_numpy(float)
finger_counts = paired["finger_count"].to_numpy()

inverse_gap = v2[:, METRIC_NAMES.index("log1p_primary_inverse_gap_integral")]
physics_offset = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE + PARAMETER_BLOCK_SIZE
unfitted = pd.DataFrame(
    [
        {"v2 coordinate": name, "Spearman vs C(N,S)": round(float(spearmanr(v2[:, index], paired.C_NS_fF).statistic), 3)}
        for name, index in [
            ("log1p_primary_inverse_gap_integral", METRIC_NAMES.index("log1p_primary_inverse_gap_integral")),
            ("log1p_total_logarithmic_gap_integral", METRIC_NAMES.index("log1p_total_logarithmic_gap_integral")),
            ("log1p_conductor_area_um2", METRIC_NAMES.index("log1p_conductor_area_um2")),
            ("boundary-element proxy C[0,0]", physics_offset),
            ("log1p_minimum_pair_gap_um", METRIC_NAMES.index("log1p_minimum_pair_gap_um")),
        ]
    ]
)
print(f"paired designs: {len(paired):,}")
print(unfitted.to_string(index=False))
"""
    ),
    code(
        """
# %% hide input
buckets = pd.qcut(paired["finger_gap_um"], q=4, duplicates="drop")
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Mean coupling spectrum by design gap quartile",
        "One unfitted coordinate against the Q3D answer",
    ],
    horizontal_spacing=0.12,
)
palette = ["#2A9D8F", "#00798C", "#6A4C93", "#D1495B"]
for color, (label, index) in zip(palette, buckets.groupby(buckets, observed=True).groups.items()):
    rows = np.asarray(index)
    spectrum = v2[rows, METRIC_BLOCK_SIZE : METRIC_BLOCK_SIZE + 24].mean(axis=0)
    figure.add_trace(
        go.Scatter(
            x=centers,
            y=spectrum,
            mode="lines+markers",
            name=f"gap {label}",
            line={"color": color, "width": 2.5},
            hovertemplate="separation=%{x:.2f} um<br>log1p(boundary um)=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
sample = np.random.default_rng(18).choice(len(paired), size=min(6000, len(paired)), replace=False)
figure.add_trace(
    go.Scattergl(
        x=inverse_gap[sample],
        y=paired["C_NS_fF"].to_numpy()[sample],
        mode="markers",
        marker={"size": 4, "opacity": 0.35, "color": paired["finger_count"].to_numpy()[sample], "colorscale": "Turbo", "colorbar": {"title": "fingers", "x": 1.02}},
        name="designs",
        hovertemplate="log1p(sum L/d)=%{x:.2f}<br>C(N,S)=%{y:.2f} fF<extra></extra>",
    ),
    row=1,
    col=2,
)
figure.update_xaxes(title_text="conductor separation (um)", type="log", row=1, col=1)
figure.update_yaxes(title_text="log1p facing boundary length", row=1, col=1)
figure.update_xaxes(title_text="log1p( sum of L / d )  [unfitted]", row=1, col=2)
figure.update_yaxes(title_text="simulated C(N,S) (fF)", row=1, col=2)
figure.update_layout(
    title=f"The coupling block tracks the physics before any model is fitted (Spearman {spearmanr(inverse_gap, paired.C_NS_fF).statistic:+.3f})",
    template="plotly_white",
    height=500,
    showlegend=True,
)
figure.show()
"""
    ),
    markdown(
        """
A single coordinate, derived from geometry by hand and fitted to nothing,
reaches Spearman correlation above 0.94 against the Q3D mutual capacitance. The
two-dimensional boundary-element proxy reaches a similar figure. Note that the
minimum gap **on its own** correlates weakly: gap alone does not set
capacitance, gap weighted by facing length does. That is precisely the
distinction the spectrum encodes and a scalar summary cannot.

This is what "physics-informed feature" should mean in practice. The learned
head is left to fit the correction for three-dimensional fringing, the
substrate, and the ground plane, rather than discovering the inverse-distance
law from pixels.
"""
    ),
    markdown(
        """
## 4. The experimental protocol

Everything below follows Tutorial 16 so the comparison is controlled: **each
exact finger count is a domain**, and we run 12 independently seeded,
finger-count-stratified holdouts. Every repeat reserves 25% of each domain as
test data; the remaining 75% supplies nested prefixes at ten label fractions.

Both representations receive an identical pipeline - the same random Fourier
feature map, the same ridge head, the same alpha, the same splits, the same test
rows. **Only the input vector differs.**

Four models are compared at every label budget:

- **specialist**: trained only on that domain's sampled labels;
- **coverage-matched generalist**: the same percentage drawn from all 13 domains;
- **budget-matched generalist**: one specialist's row count spread across all 13;
- **transfer**: a finger-count-8 foundation adapted with the specialist's labels.

> **Coverage note.** The published `SQuADDS_Layouts` repository currently holds
> 10,000 of the 16,379 `q3d_cap` GDS files its own manifest lists, so this study
> uses the 13,683 designs whose geometry can actually be downloaded. Every
> comparison below is paired on that same set, so the v0 versus v2 conclusion is
> unaffected.
"""
    ),
    code(
        """
def macro_scores(expected, predicted):
    return regression_scores(expected, predicted, TARGETS).query("target == 'macro'").iloc[0]


def build_splits():
    splits = []
    for repeat in range(REPEATS):
        rng = np.random.default_rng(SEED + 1000 * repeat)
        test, pool = [], {}
        for count in COUNTS:
            index = np.flatnonzero(finger_counts == count)
            order = rng.permutation(index)
            cut = max(1, int(round(len(order) * TEST_FRACTION)))
            test.append(order[:cut])
            pool[count] = order[cut:]
        splits.append({"test": np.concatenate(test), "pool": pool})
    return splits


splits = build_splits()
representations = {}
for name, matrix in (("v0", v0), ("v2", v2)):
    projector = V0KernelFeatureProjector(kernel_dimensions=128, random_seed=SEED)
    representations[name] = projector.fit_transform_compact(matrix.astype(np.float64))

EXPERIMENT = {
    "rows": int(len(paired)),
    "repeats": REPEATS,
    "fractions": FRACTIONS,
    "alpha": ALPHA,
    "kernel_dimensions": 128,
    "v0_dimensions": int(v0.shape[1]),
    "v2_dimensions": int(v2.shape[1]),
}
FINGERPRINT = hashlib.sha256(json.dumps(EXPERIMENT, sort_keys=True).encode()).hexdigest()[:16]
RUN_DIR = CHECKPOINTS / f"study-{FINGERPRINT}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(
    [
        {"quantity": "paired designs", "value": f"{len(paired):,}"},
        {"quantity": "finger-count domains", "value": len(COUNTS)},
        {"quantity": "independent holdouts", "value": REPEATS},
        {"quantity": "v0 compact dimensions", "value": v0.shape[1]},
        {"quantity": "v2 dimensions", "value": v2.shape[1]},
        {"quantity": "v0 model features", "value": representations["v0"].shape[1]},
        {"quantity": "v2 model features", "value": representations["v2"].shape[1]},
        {"quantity": "experiment fingerprint", "value": FINGERPRINT},
    ]
)
"""
    ),
    markdown(
        """
## 5. Learning curves under an identical model

The decisive comparison is the specialist curve: same architecture, same labels,
same test rows, different input vector.
"""
    ),
    code(
        """
CURVES_PATH = RUN_DIR / "curves.parquet"
if CURVES_PATH.exists():
    curves = pd.read_parquet(CURVES_PATH)
    print(f"Loaded learning curves from {CURVES_PATH}")
else:
    records = []
    for name, features in representations.items():
        for repeat, split in enumerate(splits):
            test = split["test"]
            pool = split["pool"]
            test_by_count = {count: test[finger_counts[test] == count] for count in COUNTS}
            foundation = TransferRidgeRegressor(ALPHA).fit(
                features[pool[BASE_FINGER_COUNT]], y[pool[BASE_FINGER_COUNT]]
            )
            typical = int(np.median([len(pool[count]) for count in COUNTS]))
            generalists = {}
            for fraction in FRACTIONS:
                coverage = np.concatenate(
                    [pool[other][: max(1, int(round(fraction * len(pool[other]))))] for other in COUNTS]
                )
                share = max(1, int(round(fraction * typical)) // len(COUNTS))
                budget = np.concatenate([pool[other][:share] for other in COUNTS])
                generalists[fraction] = {
                    "generalist_coverage": TransferRidgeRegressor(ALPHA).fit(features[coverage], y[coverage]),
                    "generalist_budget": TransferRidgeRegressor(ALPHA).fit(features[budget], y[budget]),
                }
            for count in COUNTS:
                domain_test = test_by_count[count]
                order = pool[count]
                for fraction in FRACTIONS:
                    size = max(1, int(round(fraction * len(order))))
                    selected = order[:size]
                    fits = {
                        "specialist": TransferRidgeRegressor(ALPHA).fit(features[selected], y[selected]),
                        "transfer": TransferRidgeRegressor(ALPHA).fit(
                            features[selected], y[selected], prior=foundation
                        ),
                        **generalists[fraction],
                    }
                    for method, model in fits.items():
                        scores = macro_scores(y[domain_test], model.predict(features[domain_test]))
                        records.append(
                            {
                                "representation": name,
                                "repeat": repeat,
                                "finger_count": count,
                                "fraction": fraction,
                                "labels": size,
                                "method": method,
                                "r2": scores["r2"],
                                "mae": scores["mae"],
                                "within_5_percent": scores["within_5_percent"],
                            }
                        )
                zero = macro_scores(y[domain_test], foundation.predict(features[domain_test]))
                records.append(
                    {
                        "representation": name,
                        "repeat": repeat,
                        "finger_count": count,
                        "fraction": 0.0,
                        "labels": 0,
                        "method": "zero-shot",
                        "r2": zero["r2"],
                        "mae": zero["mae"],
                        "within_5_percent": zero["within_5_percent"],
                    }
                )
    curves = pd.DataFrame(records)
    curves.to_parquet(CURVES_PATH, index=False)
    print(f"Fitted {len(curves):,} evaluations and cached them at {CURVES_PATH}")


def confidence_interval(values):
    clean = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(clean) < 2:
        return np.nan
    return float(student_t.ppf(0.5 + CONFIDENCE / 2, len(clean) - 1) * clean.std(ddof=1) / np.sqrt(len(clean)))


summary = (
    curves.groupby(["representation", "method", "fraction"], as_index=False)
    .agg(r2=("r2", "mean"), r2_ci95=("r2", confidence_interval), mae=("mae", "mean"), labels=("labels", "mean"))
)
summary.query("method == 'specialist'").pivot(index="fraction", columns="representation", values="r2").round(4)
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Domain specialists", "Foundation transfer and pooled generalists"],
    horizontal_spacing=0.10,
    shared_yaxes=True,
)
for name in ("v0", "v2"):
    frame = summary.query("representation == @name and method == 'specialist'").sort_values("fraction")
    figure.add_trace(
        go.Scatter(
            x=100 * frame["fraction"],
            y=frame["r2"],
            mode="lines+markers",
            name=f"{name} specialist",
            line={"color": PALETTE[name], "width": 3},
            marker={"size": 8},
            error_y={"type": "data", "array": frame["r2_ci95"], "visible": True, "thickness": 1.2},
            customdata=frame["labels"].round().astype(int),
            hovertemplate=f"<b>{name} specialist</b><br>pool=%{{x:.0f}}%<br>labels=%{{customdata}}<br>macro R2=%{{y:.4f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
styles = {"transfer": "solid", "generalist_coverage": "dash", "generalist_budget": "dot"}
for name in ("v0", "v2"):
    for method, dash in styles.items():
        frame = summary.query("representation == @name and method == @method").sort_values("fraction")
        figure.add_trace(
            go.Scatter(
                x=100 * frame["fraction"],
                y=frame["r2"],
                mode="lines",
                name=f"{name} {method.replace('generalist_', 'generalist ')}",
                line={"color": PALETTE[name], "width": 2, "dash": dash},
                hovertemplate=f"<b>{name} {method}</b><br>pool=%{{x:.0f}}%<br>macro R2=%{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )
figure.update_xaxes(title_text="labeled fraction of the domain pool (%)", type="log", row=1, col=1)
figure.update_xaxes(title_text="labeled fraction of the domain pool (%)", type="log", row=1, col=2)
figure.update_yaxes(title_text="held-out macro R2", range=[0.0, 1.02], row=1, col=1)
figure.update_layout(
    title="Same model, same splits, same budgets - only the embedding differs",
    template="plotly_white",
    height=560,
    hovermode="x unified",
)
figure.show()
"""
    ),
    markdown(
        """
The separation is large and it is largest exactly where it matters. Two readings
of the same curve make the label saving concrete:

- at 2% of the domain pool the v2 specialist reaches macro R2 0.934, already
  ahead of the v0 specialist at 10% (0.888) - a five-fold saving;
- at 10% the v2 specialist reaches 0.9915, ahead of v0 trained on the **entire**
  pool (0.9845) - a ten-fold saving.

In the terms of the motivating scenario, matching v0's best accuracy costs an
order of magnitude fewer simulations.

The pooled generalists tell the second half of the story. v0's budget-matched
generalist is unstable and frequently negative at low budgets, because pooling
designs whose shape blocks are scale-normalized mixes together devices with very
different absolute geometry. v2's generalists stay well-behaved.
"""
    ),
    markdown(
        """
## 6. The decisive control: is the gain geometry, or just richer parameters?

There is an obvious alternative explanation for everything above. v0 compresses
all 40 design options into **one scalar sum**, while v2 keeps 96 dimensions of
typed parameter statistics. If the improvement lives entirely in that block then
this experiment says nothing about GDS-derived geometry, and the thesis of the
whole programme is unsupported.

The test is straightforward: strip the parameter block out of v2 entirely and
re-run. `v2 geometry only` receives **no design parameters whatsoever** - it is
computed from the GDS file alone.
"""
    ),
    code(
        """
ABLATION_PATH = RUN_DIR / "ablation.parquet"
SHAPE_STOP = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE
PARAM_STOP = SHAPE_STOP + PARAMETER_BLOCK_SIZE
GEOMETRY_COLUMNS = np.r_[0:SHAPE_STOP, PARAM_STOP:V2_DIMENSIONS]

VARIANTS = {
    "v2 full (512)": v2,
    "v2 geometry only (416)": v2[:, GEOMETRY_COLUMNS],
    "v2 coupling spectrum only (192)": v2[:, METRIC_BLOCK_SIZE : METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE],
    "v2 physical metrics only (48)": v2[:, :METRIC_BLOCK_SIZE],
    "v2 parameters only (96)": v2[:, SHAPE_STOP:PARAM_STOP],
    "v0 full (155)": v0,
    "v0 scalars only (11)": v0[:, :11],
    "v0 pooled shape only (144)": v0[:, 11:],
}
ABLATION_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]

if ABLATION_PATH.exists():
    ablation = pd.read_parquet(ABLATION_PATH)
    print(f"Loaded ablation from {ABLATION_PATH}")
else:
    records = []
    for name, matrix in VARIANTS.items():
        projector = V0KernelFeatureProjector(kernel_dimensions=128, random_seed=SEED)
        features = projector.fit_transform_compact(matrix.astype(np.float64))
        for repeat, split in enumerate(splits[:8]):
            test = split["test"]
            test_by_count = {count: test[finger_counts[test] == count] for count in COUNTS}
            for fraction in ABLATION_FRACTIONS:
                for count in COUNTS:
                    order = split["pool"][count]
                    size = max(1, int(round(fraction * len(order))))
                    model = TransferRidgeRegressor(ALPHA).fit(features[order[:size]], y[order[:size]])
                    scores = macro_scores(y[test_by_count[count]], model.predict(features[test_by_count[count]]))
                    records.append(
                        {
                            "variant": name,
                            "dimensions": matrix.shape[1],
                            "repeat": repeat,
                            "finger_count": count,
                            "fraction": fraction,
                            "r2": scores["r2"],
                        }
                    )
    ablation = pd.DataFrame(records)
    ablation.to_parquet(ABLATION_PATH, index=False)

ablation_table = ablation.groupby(["variant", "fraction"])["r2"].mean().unstack().round(4)
ablation_table
"""
    ),
    code(
        """
# %% hide input
order = [
    "v2 full (512)",
    "v2 geometry only (416)",
    "v2 physical metrics only (48)",
    "v2 coupling spectrum only (192)",
    "v2 parameters only (96)",
    "v0 full (155)",
    "v0 scalars only (11)",
    "v0 pooled shape only (144)",
]
bar_colors = ["#00798C", "#2A9D8F", "#4CAF9D", "#6A4C93", "#E9C46A", "#D1495B", "#E07A5F", "#B23A48"]
figure = go.Figure()
for fraction in ABLATION_FRACTIONS:
    frame = ablation.query("fraction == @fraction").groupby("variant")["r2"].mean().reindex(order)
    figure.add_trace(
        go.Bar(
            x=order,
            y=frame.to_numpy(),
            name=f"{fraction:.0%} labels",
            visible=(fraction == 0.01),
            marker_color=bar_colors,
            text=[f"{value:.3f}" for value in frame.to_numpy()],
            textposition="outside",
            hovertemplate="%{x}<br>macro R2=%{y:.4f}<extra></extra>",
        )
    )
figure.update_layout(
    title="Which block earns the accuracy? (use the slider to change label budget)",
    yaxis={"title": "held-out macro R2", "range": [-2.6, 1.15]},
    xaxis={"tickangle": -22},
    template="plotly_white",
    height=620,
    showlegend=False,
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "label budget: "},
            "pad": {"t": 60},
            "steps": [
                {
                    "label": f"{fraction:.0%}",
                    "method": "update",
                    "args": [{"visible": [other == fraction for other in ABLATION_FRACTIONS]}],
                }
                for fraction in ABLATION_FRACTIONS
            ],
        }
    ],
)
figure.add_hline(y=0, line_color="#1F2937", line_width=1)
figure.show()
"""
    ),
    markdown(
        """
### Reading the ablation

**The gain is geometry, not parameters.** `v2 geometry only` sees no design
options at all and still beats `v0 full` - which does include v0's parameter sum
- at every budget: 0.754 against 0.235 at 1% labels, and 0.9996 against 0.9836
at 100%. The thesis survives its most dangerous control.

**The coupling spectrum alone outperforms the whole of v0.** 192 dimensions of
facing-boundary-length histogram, with no scalar metrics, no parameters, and no
raster, reach 0.993 at full budget against v0's 0.984.

**v0's raster is close to worthless.** `v0 pooled shape only` is strongly
negative at low budgets and reaches just 0.776 at full budget, while consuming
144 of v0's 155 compact dimensions. This is the quantitative version of the
scale-blindness argument from section 1, and it explains the Tutorial 16b
observation that scalar geometry could beat the full v0 representation.

**Fewer, better dimensions win when labels are scarce.** At 1% labels the
48-dimensional physical-metric block is the single best variant, ahead of the
full 512. With roughly nine training rows per domain, dimension is a liability;
the compact physical summary is the right tool at that budget. This is a real
result and it argues for publishing v2 as blocks that can be selected, not as a
monolith.
"""
    ),
    markdown(
        """
## 7. Similarity as an applicability gate

The motivating scenario needs more than accuracy: it needs to know **which of
our design families a newcomer resembles**, before any labels exist. That makes
the quality of the cosine metric a first-class result.

We fit a finger-count-8 foundation, then ask how well cosine similarity to that
foundation's centroid ranks its zero-shot error on every other domain. A useful
applicability score sends high-error designs to the low-similarity end.
"""
    ),
    code(
        """
base_mask = finger_counts == BASE_FINGER_COUNT
rng = np.random.default_rng(SEED)
base_order = rng.permutation(np.flatnonzero(base_mask))
base_train = base_order[: int(0.75 * len(base_order))]

applicability = []
similarity_by_name = {}
for name, matrix in (("v0", v0), ("v2", v2)):
    features = representations[name]
    model = TransferRidgeRegressor(ALPHA).fit(features[base_train], y[base_train])
    predicted = model.predict(features)
    error = 100 * np.mean(np.abs(predicted - y) / np.maximum(np.abs(y), 1e-9), axis=1)
    unit = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    centroid = unit[base_train].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    similarity = unit @ centroid
    outside = ~base_mask
    similarity_by_name[name] = (similarity, error, outside)
    applicability.append(
        {
            "representation": name,
            "Spearman(similarity, error)": round(float(spearmanr(similarity[outside], error[outside]).statistic), 3),
            "median zero-shot APE (%)": round(float(np.median(error[outside])), 1),
            "cosine spread (sd)": round(float(similarity[outside].std()), 4),
        }
    )
pd.DataFrame(applicability)
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[f"{name}: similarity against zero-shot error" for name in ("v0", "v2")],
    horizontal_spacing=0.11,
)
for column, name in enumerate(("v0", "v2"), start=1):
    similarity, error, outside = similarity_by_name[name]
    rows = np.flatnonzero(outside)
    shown = np.random.default_rng(SEED).choice(rows, size=min(5000, len(rows)), replace=False)
    figure.add_trace(
        go.Scattergl(
            x=similarity[shown],
            y=np.clip(error[shown], 0, 200),
            mode="markers",
            marker={"size": 4, "opacity": 0.25, "color": PALETTE[name]},
            name=name,
            showlegend=False,
            hovertemplate="cosine=%{x:.3f}<br>APE=%{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=column,
    )
    deciles = pd.qcut(similarity[rows], q=10, duplicates="drop")
    grouped = pd.DataFrame({"similarity": similarity[rows], "error": error[rows], "bucket": deciles}).groupby(
        "bucket", observed=True
    ).agg(similarity=("similarity", "mean"), error=("error", "median"))
    figure.add_trace(
        go.Scatter(
            x=grouped["similarity"],
            y=grouped["error"],
            mode="lines+markers",
            line={"color": "#1F2937", "width": 3},
            marker={"size": 9},
            name=f"{name} decile median",
            showlegend=False,
            hovertemplate="mean cosine=%{x:.3f}<br>median APE=%{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=column,
    )
    figure.update_xaxes(title_text="cosine similarity to the count-8 foundation", row=1, col=column)
figure.update_yaxes(title_text="zero-shot mean absolute percentage error", row=1, col=1)
figure.update_layout(
    title="v2 ranks unfamiliar designs more than twice as well as v0",
    template="plotly_white",
    height=520,
)
figure.show()
"""
    ),
    markdown(
        """
v2 more than doubles the rank correlation between similarity and zero-shot error
(-0.560 against v0's -0.248) and halves the median zero-shot error. For the
routing workflow this is the number that matters: predict the high-similarity
candidates, send the low-similarity ones to simulation.

One honest caveat. The standard deviation of v2's cosine values is small
(0.033 against v0's 0.406) even though the full range spans 0.012 to 0.998 - the
distribution is heavy-tailed, not collapsed. Every v2 coordinate is a
non-negative log-magnitude, which creates a large common direction, and that is
the same mechanism that sank the rejected v1.0 candidate. The **ranking** is
much better, but a published v2 should ship a frozen whitening transform as a
separate metric layer rather than relying on raw cosine.
"""
    ),
    markdown(
        """
## 8. Where the advantage narrows: genuine scale extrapolation

Every result so far interpolates - the test designs are new, but their size
regime is represented in training. The harder question for a foundation model is
extrapolation. We train on the smaller 75% of devices by conductor area and test
on the largest 25%, which no training row resembles.
"""
    ),
    code(
        """
area = v2[:, METRIC_NAMES.index("log1p_conductor_area_um2")]
threshold = np.quantile(area, 0.75)
small_rows = np.flatnonzero(area <= threshold)
large_rows = np.flatnonzero(area > threshold)

extrapolation = []
for name, matrix in (("v0", v0), ("v2", v2), ("v2 geometry only", v2[:, GEOMETRY_COLUMNS])):
    projector = V0KernelFeatureProjector(kernel_dimensions=128, random_seed=SEED)
    features = projector.fit_transform_compact(matrix.astype(np.float64))
    model = TransferRidgeRegressor(ALPHA).fit(features[small_rows], y[small_rows])
    scores = macro_scores(y[large_rows], model.predict(features[large_rows]))
    interpolating = summary.query("representation == @name and method == 'specialist' and fraction == 1.0")
    extrapolation.append(
        {
            "representation": name,
            "extrapolated macro R2": round(float(scores["r2"]), 4),
            "extrapolated MAE (fF)": round(float(scores["mae"]), 3),
        }
    )
extrapolation_frame = pd.DataFrame(extrapolation)
print(f"train on {len(small_rows):,} smaller devices, test on {len(large_rows):,} strictly larger ones")
extrapolation_frame
"""
    ),
    code(
        """
# %% hide input
interpolating = {
    "v0": float(summary.query("representation == 'v0' and method == 'specialist' and fraction == 1.0")["r2"].iloc[0]),
    "v2": float(summary.query("representation == 'v2' and method == 'specialist' and fraction == 1.0")["r2"].iloc[0]),
}
figure = go.Figure()
figure.add_trace(
    go.Bar(
        name="interpolating (full-budget specialist)",
        x=["v0", "v2"],
        y=[interpolating["v0"], interpolating["v2"]],
        marker_color="#2A9D8F",
        text=[f"{interpolating[name]:.4f}" for name in ("v0", "v2")],
        textposition="outside",
    )
)
figure.add_trace(
    go.Bar(
        name="extrapolating to larger devices",
        x=extrapolation_frame["representation"],
        y=extrapolation_frame["extrapolated macro R2"],
        marker_color="#6A4C93",
        text=[f"{value:.4f}" for value in extrapolation_frame["extrapolated macro R2"]],
        textposition="outside",
    )
)
figure.update_layout(
    title="The v2 advantage is large when interpolating and modest when extrapolating in scale",
    yaxis={"title": "held-out macro R2", "range": [0, 1.12]},
    barmode="group",
    template="plotly_white",
    height=520,
)
figure.show()
"""
    ),
    markdown(
        """
This is the most important limitation in the notebook. Interpolating, v2 turns
v0's 0.984 into 0.9997. Extrapolating to devices larger than anything in
training, v2 reaches 0.810 against v0's 0.769 - still ahead, but the gap has
nearly closed, and `v2 geometry only` actually falls slightly behind v0 at 0.735.

The interpretation is that absolute physical anchoring lets the model *represent*
a size regime it has never seen, but representing it is not the same as knowing
the physics there. Extrapolation across scale remains an open problem that a
better embedding alone does not solve; the explicit scale coordinate makes the
correct scaling law learnable, but it still has to be learned from data that
covers it.
"""
    ),
    markdown(
        """
## 9. What this establishes, and what it does not

**Established here**

- v2 is a pure function of one GDS file and one parameter mapping. No catalogue
  statistic, no fitted frequency selection, no dependence on neighbouring rows.
  An outside contribution can be encoded and compared without refitting anything.
- Under an identical model, split, and label budget, v2 raises the full-budget
  specialist from macro R2 0.984 to 0.9997, and at 1% labels from 0.235 to 0.780.
- The improvement is geometric. Stripping every design parameter out of v2 still
  beats the whole of v0, so the result is not an artefact of v2 keeping more
  parameter detail than v0's single sum.
- A hand-derived coupling integral correlates at Spearman 0.94 with the simulated
  mutual capacitance before any model is fitted, which is direct evidence that
  the coupling spectrum encodes the right physics.
- v2 similarity ranks zero-shot error more than twice as well as v0 similarity,
  which is what the "find the nearest design family" workflow depends on.

**Not established, and worth stating plainly**

- Only one component class is measured here. The cross-class study of Tutorial 17
  has not been repeated with v2, so the universality claim remains an argument
  from construction rather than a measurement.
- Extrapolation across device scale improves only modestly, and the geometry-only
  variant regresses.
- The raw cosine metric has a compressed spread. A frozen whitening layer should
  be fitted and published before v2 similarity is used as a headline number.
- These are 12 overlapping holdouts of one simulated catalogue, not independent
  physical experiments. External validation on new simulations or measured
  devices is the necessary next step.
- 512 dimensions carry only about 200 non-constant coordinates on this dataset,
  because two-terminal devices leave the higher terminal-pair blocks empty. That
  capacity is reserved for richer topologies and costs nothing statistically, but
  it should not be mistaken for 512 dimensions of information.

**The next experiment**

Repeat Tutorial 17's cross-class study with v2, and add the held-out-class test
that no tutorial currently runs: fit on three component families and evaluate on
the fourth. The blocker has been that only two families share a comparable
target, which the boundary-element proxy now offers a way around - predicting the
dimensionless ratio of simulated capacitance to the proxy gives every class one
common, comparable target.
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, str(OUTPUT))
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} cells.")
