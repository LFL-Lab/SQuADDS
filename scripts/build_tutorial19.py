#!/usr/bin/env python
"""Build Tutorial 19: a worked example of the universal-geometry-v2 encoder."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("tutorials/Tutorial-19_How_Universal_Geometry_v2_Works.ipynb")


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
# Tutorial 19: how `universal-geometry-v2` works, one design at a time

Tutorial 18 showed **that** v2 outperforms `static-shape-v0`. This tutorial shows
**how** it is computed. We take a single real capacitor out of the catalogue and
follow it all the way from GDS polygons to a 512-number vector, drawing every
intermediate quantity.

The encoder is deliberately simple enough to audit. Every coordinate is a
physical measurement, so each step below can be checked against a picture of the
geometry rather than against a training loss.

```
GDS polygons
  -> layer roles and terminals
  -> block 1  physical metrics        (48)
  -> block 2  coupling spectrum      (192)
  -> block 3  shape spectrum         (128)
  -> block 4  parameter statistics    (96)
  -> block 5  physics proxy           (48)
  -> 512-dimensional vector
```

Along the way we will **re-derive the coupling spectrum by hand** with plain
`shapely` and `numpy`, and check that it reproduces the library output exactly.
The final section runs the scenario the whole design exists for: a stranger's
capacitor, with its own shape and its own 28 parameter names, projected into the
space our catalogue already occupies.
"""
    ),
    code(
        """
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shapely
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots
from scipy import ndimage

from squadds.layouts.geometry_v2 import (
    COUPLING_BINS,
    COUPLING_BLOCK_SIZE,
    COUPLING_EDGES,
    GROUND_BINS,
    METRIC_BLOCK_SIZE,
    METRIC_NAMES,
    PARAMETER_BLOCK_SIZE,
    PARAMETER_DIMENSION_CLASSES,
    PHYSICS_BLOCK_SIZE,
    SHAPE_BINS,
    SHAPE_BLOCK_SIZE,
    SHAPE_EDGES,
    TOPOLOGY_RADII_UM,
    V2_DIMENSIONS,
    VACUUM_PERMITTIVITY,
    classify_parameter,
    encode,
    parameter_block,
    read_layer_geometry,
    soft_histogram,
)
from squadds.layouts.geometry_v2 import (
    _boundary_samples,
    _raster_frame,
    _rasterize,
    _role_geometry,
    _terminals,
)

pio.renderers.default = "notebook_connected"
pd.set_option("display.max_columns", 30)

INK = "#1F2937"
ROLE_COLORS = {
    "conductor": "#00798C",
    "etch": "#E9C46A",
    "port": "#D1495B",
    "domain": "#C7CDD4",
}
TERMINAL_COLORS = ["#00798C", "#D1495B", "#6A4C93", "#E9C46A"]

SOURCE_ID = "exp6/cap_0000"
GDS_PATH = Path(
    hf_hub_download(
        "SQuADDS/SQuADDS_Layouts",
        f"raw/GeneralizedCapNInterdigital/{SOURCE_ID}.gds",
        repo_type="dataset",
    )
)
DATABASE = Path(
    hf_hub_download(
        "SQuADDS/SQuADDS_DB",
        "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
        repo_type="dataset",
    )
)
row = next(item for item in json.loads(DATABASE.read_text()) if item["notes"]["source_id"] == SOURCE_ID)
DESIGN_OPTIONS = row["design"]["design_options"]
MEASURED = row["sim_results"]

print(f"worked example : {SOURCE_ID}")
print(f"design options : {len(DESIGN_OPTIONS)} entries")
print(f"simulated C(N,S) = {MEASURED['north_to_south']:.4f} fF")
"""
    ),
    markdown(
        """
## 1. The input contract

v2 takes exactly two things: a GDS file whose `(layer, datatype)` pairs follow
the published semantics, and the native design-parameter mapping from whatever
tool produced it. Nothing else. No catalogue, no fitted statistics, no
simulation results.

The four roles the encoder recognizes are `conductor`, `etch`, `port`, and
`domain`. For this component family the mapping is:

| layer | datatype | role |
| --- | --- | --- |
| 1 | 10 | conductor - the signal metal |
| 1 | 0 | domain - the ground plane, with a cutout around the device |
| 2 | 0 | port - marks the north terminal |
| 3 | 0 | port - marks the south terminal |
"""
    ),
    code(
        """
geometry = read_layer_geometry(GDS_PATH)
grouped = _role_geometry(geometry, None)

layer_table = pd.DataFrame(
    [
        {
            "layer": key[0],
            "datatype": key[1],
            "role": role,
            "polygons": len(list(getattr(shape, "geoms", [shape]))),
            "area_um2": round(shape.area, 3),
            "perimeter_um": round(shape.length, 3),
        }
        for role, entries in grouped.items()
        for key, shape in entries
    ]
).sort_values(["layer", "datatype"])
print(layer_table.to_string(index=False))
print()
print("design options (first 8 of %d):" % len(DESIGN_OPTIONS))
for name in sorted(DESIGN_OPTIONS)[:8]:
    print(f"  {name:32s} {DESIGN_OPTIONS[name]}")
"""
    ),
    code(
        """
# %% hide input
def polygon_traces(shape, name, color, *, opacity=0.75, paper="#FFFFFF", show=True):
    traces = []
    first = True
    for polygon in getattr(shape, "geoms", [shape]):
        if polygon.geom_type != "Polygon":
            continue
        x, y = polygon.exterior.xy
        traces.append(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                fill="toself",
                fillcolor=color,
                opacity=opacity,
                line={"color": color, "width": 1.2},
                name=name,
                legendgroup=name,
                showlegend=show and first,
                hoverinfo="skip",
            )
        )
        first = False
        for interior in polygon.interiors:
            hx, hy = interior.xy
            traces.append(
                go.Scatter(
                    x=list(hx),
                    y=list(hy),
                    mode="lines",
                    fill="toself",
                    fillcolor=paper,
                    line={"color": color, "width": 0.8},
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return traces


figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Full GDS, all four roles", "Functional geometry the encoder measures"],
    horizontal_spacing=0.09,
)
for role in ("domain", "etch", "conductor", "port"):
    for key, shape in grouped[role]:
        for trace in polygon_traces(shape, role, ROLE_COLORS[role]):
            figure.add_trace(trace, row=1, col=1)
for role in ("conductor", "port"):
    for key, shape in grouped[role]:
        for trace in polygon_traces(shape, role, ROLE_COLORS[role], show=False):
            figure.add_trace(trace, row=1, col=2)
figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
figure.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
figure.update_xaxes(title_text="x (um)")
figure.update_yaxes(title_text="y (um)", row=1, col=1)
figure.update_layout(
    title=f"{SOURCE_ID}: a two-terminal interdigital capacitor",
    template="plotly_white",
    height=520,
)
figure.show()
"""
    ),
    markdown(
        """
The left panel is everything in the file. The ground plane dominates it, and
notice that it is **not** a solid rectangle: it has a cutout around the device.
That cutout is a real physical gap, and the encoder measures it.

The right panel is what remains after the ground plane is set aside as context.
This is the functional geometry.
"""
    ),
    markdown(
        """
## 2. Step one: terminals are discovered, not declared

A design tool knows that this component has a "north pad" and a "south pad". The
encoder must not, because a foreign contributor will use different words.

Instead v2 merges the conductor layer and splits it into **connected
components**. Each component is a terminal. Ordering is then fixed by two
deterministic rules:

1. if port markers exist, order terminals by the port layer nearest to each;
2. otherwise, order by descending area, breaking ties by first appearance.

This is what makes `terminal 0` and `terminal 1` mean the same thing in a
28-parameter foreign layout as in this one.
"""
    ),
    code(
        """
conductor = shapely.union_all([shape for _, shape in grouped["conductor"]])
terminals = _terminals(conductor, grouped["port"])

terminal_table = pd.DataFrame(
    [
        {
            "terminal": index,
            "area_um2": round(terminal.area, 3),
            "perimeter_um": round(terminal.length, 3),
            "nearest port layer": min(
                (key[0] for key, port in grouped["port"] if terminal.distance(port) < 1e-6),
                default=None,
            ),
        }
        for index, terminal in enumerate(terminals)
    ]
)
print(terminal_table.to_string(index=False))
print()
print(f"minimum separation between terminal 0 and terminal 1: {terminals[0].distance(terminals[1]):.4f} um")
"""
    ),
    code(
        """
# %% hide input
figure = go.Figure()
for key, shape in grouped["domain"]:
    for trace in polygon_traces(shape, "ground plane", ROLE_COLORS["domain"], opacity=0.35):
        figure.add_trace(trace)
for index, terminal in enumerate(terminals):
    for trace in polygon_traces(terminal, f"terminal {index}", TERMINAL_COLORS[index]):
        figure.add_trace(trace)
for key, port in grouped["port"]:
    for trace in polygon_traces(port, f"port layer {key[0]}", ROLE_COLORS["port"], opacity=1.0):
        figure.add_trace(trace)
figure.update_yaxes(scaleanchor="x", scaleratio=1)
figure.update_layout(
    title="Connected components become terminals; port markers fix their order",
    xaxis_title="x (um)",
    yaxis_title="y (um)",
    template="plotly_white",
    height=560,
)
figure.show()
"""
    ),
    markdown(
        """
## 3. Step two: the coupling spectrum

This is the block the design rests on, so we derive it from scratch.

For coplanar conductors the mutual capacitance is approximately an integral of
some kernel over facing boundary length at a given separation,

$$C_{ij} \\;\\approx\\; \\varepsilon \\int f(d)\\, \\mathrm{d}L .$$

We do not know $f$, and we do not need to. If we store $\\mathrm{d}L$ **binned
against absolute separation $d$**, then any model that is linear in those bins
can represent any $f$. The learned head fits the kernel; the encoder supplies
the integral's measure.

Concretely, for terminal $i$ against terminal $j$:

1. walk terminal $i$'s boundary at uniform arclength, taking 1,024 samples;
2. for each sample, compute the exact distance to terminal $j$;
3. accumulate each sample's arclength share into a frozen log-spaced grid of 24
   bins spanning 0.1 um to 1000 um;
4. symmetrize over both directions and take `log1p`.

The bin edges are constants of the standard, never fitted, which is exactly what
makes one contributor's bin 12 the same physical quantity as another's.
"""
    ),
    code(
        """
north, south = terminals[0], terminals[1]


def sample_boundary(shape, count=1024):
    \"\"\"Uniform-arclength samples with the boundary length each one represents.\"\"\"
    boundary = shape.boundary
    length = boundary.length
    positions = (np.arange(count) + 0.5) * length / count
    coordinates = shapely.get_coordinates(shapely.line_interpolate_point(boundary, positions))
    return coordinates, np.full(count, length / count)


north_points, north_weights = sample_boundary(north)
south_points, south_weights = sample_boundary(south)

north_distances = shapely.distance(shapely.points(north_points), south)
south_distances = shapely.distance(shapely.points(south_points), north)

forward = soft_histogram(north_distances, north_weights, COUPLING_EDGES)
backward = soft_histogram(south_distances, south_weights, COUPLING_EDGES)
hand_derived = np.log1p(0.5 * (forward + backward))

vector = encode(GDS_PATH, DESIGN_OPTIONS)
library = vector[METRIC_BLOCK_SIZE : METRIC_BLOCK_SIZE + COUPLING_BINS]

print(f"largest disagreement between the hand derivation and encode(): {np.abs(hand_derived - library).max():.2e}")
assert np.allclose(hand_derived, library, atol=1e-5)

centers = np.sqrt(COUPLING_EDGES[:-1] * COUPLING_EDGES[1:])
spectrum_table = pd.DataFrame(
    {
        "bin": range(COUPLING_BINS),
        "separation_um": centers.round(3),
        "facing_boundary_um": (0.5 * (forward + backward)).round(4),
        "stored_value": hand_derived.round(4),
    }
)
print(spectrum_table[spectrum_table.facing_boundary_um > 0].to_string(index=False))
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Every boundary sample, coloured by distance to the facing terminal",
        "The same distances, binned into the coupling spectrum",
    ],
    horizontal_spacing=0.11,
    column_widths=[0.46, 0.54],
)
for trace in polygon_traces(south, "terminal 1", TERMINAL_COLORS[1], opacity=0.25, show=False):
    figure.add_trace(trace, row=1, col=1)
for trace in polygon_traces(north, "terminal 0", TERMINAL_COLORS[0], opacity=0.18, show=False):
    figure.add_trace(trace, row=1, col=1)
figure.add_trace(
    go.Scatter(
        x=north_points[:, 0],
        y=north_points[:, 1],
        mode="markers",
        marker={
            "size": 5,
            "color": north_distances,
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": float(np.quantile(north_distances, 0.95)),
            "colorbar": {"title": "distance<br>to facing<br>terminal (um)", "x": 0.43, "len": 0.85},
        },
        name="terminal 0 boundary",
        showlegend=False,
        hovertemplate="x=%{x:.2f} um<br>y=%{y:.2f} um<br>distance=%{marker.color:.2f} um<extra></extra>",
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Bar(
        x=centers,
        y=0.5 * (forward + backward),
        marker_color="#00798C",
        name="facing boundary length",
        showlegend=False,
        hovertemplate="separation=%{x:.2f} um<br>boundary length=%{y:.3f} um<extra></extra>",
    ),
    row=1,
    col=2,
)
figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
figure.update_xaxes(title_text="x (um)", row=1, col=1)
figure.update_yaxes(title_text="y (um)", row=1, col=1)
figure.update_xaxes(title_text="conductor separation (um)", type="log", row=1, col=2)
figure.update_yaxes(title_text="facing boundary length (um)", row=1, col=2)
figure.update_layout(
    title="From geometry to spectrum: where the metal faces metal, and how far away",
    template="plotly_white",
    height=520,
)
figure.show()
"""
    ),
    markdown(
        """
The left panel is the whole idea in one picture. The dark points along the inner
edges of the fingers sit a couple of micrometers from the facing comb and carry
almost all of the capacitance; the bright points on the outer edges are tens of
micrometers away and contribute little. The right panel is that same information
with the geometry thrown away and only the physics kept.

This also explains why the minimum gap on its own is a poor predictor. It tells
you where the leftmost bar sits, but not how tall it is - and the height, the
facing boundary length, is what multiplies the kernel.

### Why the bins are soft

A hard histogram assigns each sample entirely to one bin. A sample sitting a
hair from a bin edge then jumps its whole weight to the neighbour when the
geometry moves by a nanometre, which makes the embedding discontinuous in the
thing it is supposed to measure. v2 splits each sample linearly between the two
nearest bin centres in log-distance, so the spectrum is Lipschitz in the
geometry.
"""
    ),
    code(
        """
# %% hide input
probe_edges = COUPLING_EDGES
probe_centers = centers
shift_grid = np.linspace(0.0, 0.35, 60)
hard_track, soft_track = [], []
for shift in shift_grid:
    shifted = north_distances * (1.0 + shift)
    hard, _ = np.histogram(np.clip(shifted, probe_edges[0], probe_edges[-1]), bins=probe_edges, weights=north_weights)
    soft = soft_histogram(shifted, north_weights, probe_edges)
    hard_track.append(hard)
    soft_track.append(soft)
hard_track = np.asarray(hard_track)
soft_track = np.asarray(soft_track)
busiest = int(np.argmax(soft_track[0]))

figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f"Bin {busiest} as the geometry is stretched",
        "Total change in the whole spectrum",
    ],
    horizontal_spacing=0.12,
)
figure.add_trace(
    go.Scatter(
        x=100 * shift_grid,
        y=hard_track[:, busiest],
        mode="lines",
        name="hard bins",
        line={"color": "#D1495B", "width": 3},
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(
        x=100 * shift_grid,
        y=soft_track[:, busiest],
        mode="lines",
        name="soft bins (v2)",
        line={"color": "#00798C", "width": 3},
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(
        x=100 * shift_grid[1:],
        y=np.abs(np.diff(hard_track, axis=0)).sum(axis=1),
        mode="lines",
        name="hard bins",
        line={"color": "#D1495B", "width": 3},
        showlegend=False,
    ),
    row=1,
    col=2,
)
figure.add_trace(
    go.Scatter(
        x=100 * shift_grid[1:],
        y=np.abs(np.diff(soft_track, axis=0)).sum(axis=1),
        mode="lines",
        name="soft bins (v2)",
        line={"color": "#00798C", "width": 3},
        showlegend=False,
    ),
    row=1,
    col=2,
)
figure.update_xaxes(title_text="all separations stretched by (%)")
figure.update_yaxes(title_text="boundary length in the bin (um)", row=1, col=1)
figure.update_yaxes(title_text="step-to-step change (um)", row=1, col=2)
figure.update_layout(
    title="Hard bins jump; soft bins move smoothly with the geometry",
    template="plotly_white",
    height=470,
)
figure.show()
"""
    ),
    markdown(
        """
The staircase on the left is a hard histogram handing an entire bin's worth of
boundary to its neighbour in one step. The right panel shows the same effect as
total spectrum movement: the hard version spikes whenever a population crosses
an edge, while the soft version stays bounded.

The remaining 168 coordinates of this block cover the other five terminal pairs
and each terminal's spectrum against the ground plane. For a two-terminal device
most stay zero, and that reserved capacity is what lets a three- or four-terminal
component use the same 512 coordinates.
"""
    ),
    markdown(
        """
## 4. Step three: physical metrics

Forty-eight scalars, each a named measurement in absolute units. The point of
this block is that a human can read it, and every entry has a dimension. Below
are the twelve that matter most for this device, next to their raw values.
"""
    ),
    code(
        """
frame = _raster_frame(conductor.bounds)
pixel = frame[2]
conductor_mask = _rasterize(conductor, frame)
interior = ndimage.distance_transform_edt(conductor_mask) * pixel
widths = 2.0 * interior[conductor_mask]

highlights = [
    "log1p_bbox_width_um",
    "log1p_bbox_height_um",
    "log1p_conductor_area_um2",
    "log1p_conductor_perimeter_um",
    "log1p_minimum_pair_gap_um",
    "log1p_primary_inverse_gap_integral",
    "log1p_conductor_width_p05_um",
    "log1p_conductor_width_p50_um",
    "terminal_count",
    "horizontal_symmetry",
    "vertical_symmetry",
    "conductor_fill_fraction",
]
metric_table = pd.DataFrame(
    [
        {
            "metric": name,
            "stored value": round(float(vector[METRIC_NAMES.index(name)]), 4),
            "physical value": (
                round(float(np.expm1(vector[METRIC_NAMES.index(name)])), 4)
                if name.startswith("log1p_")
                else round(float(vector[METRIC_NAMES.index(name)]), 4)
            ),
        }
        for name in highlights
    ]
)
print(metric_table.to_string(index=False))
print()
print(f"conductor width percentiles from the distance transform (um): {np.percentile(widths, [5, 50, 95]).round(3)}")
"""
    ),
    code(
        """
# %% hide input
extent_x = [frame[0], frame[0] + frame[3]]
extent_y = [frame[1], frame[1] + frame[3]]
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Distance to the nearest conductor edge", "Conductor width distribution"],
    horizontal_spacing=0.13,
)
figure.add_trace(
    go.Heatmap(
        z=np.where(conductor_mask, interior, np.nan),
        x=np.linspace(extent_x[0], extent_x[1], conductor_mask.shape[1]),
        y=np.linspace(extent_y[1], extent_y[0], conductor_mask.shape[0]),
        colorscale="Magma",
        colorbar={"title": "half-width<br>(um)", "x": 0.43, "len": 0.85},
        hovertemplate="x=%{x:.2f} um<br>y=%{y:.2f} um<br>half-width=%{z:.2f} um<extra></extra>",
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Bar(
        x=np.sqrt(SHAPE_EDGES[:-1] * SHAPE_EDGES[1:]),
        y=soft_histogram(widths, np.full(len(widths), pixel * pixel), SHAPE_EDGES),
        marker_color="#6A4C93",
        showlegend=False,
        hovertemplate="width=%{x:.2f} um<br>area=%{y:.2f} um^2<extra></extra>",
    ),
    row=1,
    col=2,
)
figure.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed", row=1, col=1)
figure.update_xaxes(title_text="x (um)", row=1, col=1)
figure.update_xaxes(title_text="conductor width (um)", type="log", row=1, col=2)
figure.update_yaxes(title_text="conductor area at that width (um^2)", row=1, col=2)
figure.update_layout(
    title="Feature size is measured, not inferred from a parameter name",
    template="plotly_white",
    height=500,
)
figure.show()
"""
    ),
    markdown(
        """
The bright ridge running down each finger is its half-width. Reducing that map
to percentiles gives minimum feature size, typical linewidth, and pad size as
three absolute numbers, none of which require knowing what the design tool
called them. A contributor whose parameter is named `digit_thickness` produces
the same coordinates as ours named `finger_width`.

## 5. Step four: the shape spectrum

Four channels of 32 bins. Two are correlation functions computed by FFT on a
raster whose pixel size is recorded, so the radial bins stay in micrometers:

- **two-point correlation** of the conductor set, the probability that two points
  separated by $r$ are both metal, which reveals the finger pitch;
- **cross-correlation** between terminal 0 and terminal 1, which reveals the
  interleaving;
- the **width distribution** shown above;
- **contour harmonics**, the Fourier transform of the boundary traversed at
  uniform arclength, taken on exact polygon vertices with no rasterization.
"""
    ),
    code(
        """
# %% hide input
def autocorrelate(first, second):
    spectrum = np.fft.rfft2(first.astype(float))
    other = spectrum if second is first else np.fft.rfft2(second.astype(float))
    return np.fft.fftshift(np.fft.irfft2(spectrum * np.conj(other), s=first.shape)) / first.size


def radial(correlation, pixel_size, edges):
    size = correlation.shape[0]
    grid = np.arange(size) - size // 2
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    radius = np.hypot(xx, yy) * pixel_size
    index = np.digitize(radius.reshape(-1), edges) - 1
    profile = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    values = correlation.reshape(-1)
    valid = (index >= 0) & (index < len(profile))
    np.add.at(profile, index[valid], values[valid])
    np.add.at(counts, index[valid], 1.0)
    return profile / np.maximum(counts, 1.0)


masks = [_rasterize(terminal, frame) for terminal in terminals]
shape_centers = np.sqrt(SHAPE_EDGES[:-1] * SHAPE_EDGES[1:])
two_point = radial(autocorrelate(conductor_mask, conductor_mask), pixel, SHAPE_EDGES) / max(conductor_mask.mean(), 1e-12)
cross = radial(autocorrelate(masks[0], masks[1]), pixel, SHAPE_EDGES) / np.sqrt(
    max(masks[0].mean(), 1e-12) * max(masks[1].mean(), 1e-12)
)

contour, _ = _boundary_samples(terminals[0], target=256)
signal = contour[:, 0] + 1j * contour[:, 1]
spectrum = np.fft.fft(signal - signal.mean())

figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Correlation functions in absolute micrometers", "Contour rebuilt from N harmonics"],
    horizontal_spacing=0.12,
)
figure.add_trace(
    go.Scatter(
        x=shape_centers,
        y=two_point,
        mode="lines+markers",
        name="conductor with itself",
        line={"color": "#00798C", "width": 3},
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(
        x=shape_centers,
        y=cross,
        mode="lines+markers",
        name="terminal 0 with terminal 1",
        line={"color": "#D1495B", "width": 3},
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(
        x=contour[:, 0],
        y=contour[:, 1],
        mode="lines",
        name="exact contour",
        line={"color": INK, "width": 3},
    ),
    row=1,
    col=2,
)
for harmonics, color in ((4, "#E9C46A"), (8, "#6A4C93"), (16, "#00798C")):
    filtered = np.zeros_like(spectrum)
    filtered[:harmonics] = spectrum[:harmonics]
    filtered[-harmonics:] = spectrum[-harmonics:]
    rebuilt = np.fft.ifft(filtered) + signal.mean()
    figure.add_trace(
        go.Scatter(
            x=rebuilt.real,
            y=rebuilt.imag,
            mode="lines",
            name=f"{harmonics} harmonics",
            line={"color": color, "width": 2, "dash": "dash"},
        ),
        row=1,
        col=2,
    )
figure.update_xaxes(title_text="separation r (um)", type="log", row=1, col=1)
figure.update_yaxes(title_text="normalized correlation", row=1, col=1)
figure.update_xaxes(title_text="x (um)", row=1, col=2)
figure.update_yaxes(title_text="y (um)", scaleanchor="x2", scaleratio=1, row=1, col=2)
figure.update_layout(
    title="Shape measured as correlation length and as boundary harmonics",
    template="plotly_white",
    height=520,
)
figure.show()
"""
    ),
    markdown(
        """
On the right, four harmonics give the pad outline, eight begin to suggest the
comb, and sixteen resolve individual fingers. Storing the harmonic magnitudes
gives a compact, orientation-free description of exactly that progression - and
because it runs on polygon vertices, it never pays the resolution cost that sank
v0's 96 by 96 raster.

## 6. Step five: parameter statistics, without a name registry

This design has 40 options. A stranger's design will have a different number
with different names. v2 handles both with the same 96 coordinates by asking
what each parameter **is** rather than what it is called.

Each option is parsed into a physical dimension class - length, count, angle,
boolean, or other - using its unit suffix first and its name only as a fallback.
Within each class the encoder stores summary statistics plus **order
statistics**: the sorted smallest and largest values. `min(lengths)` is the
minimum feature size in the design, which is physically comparable across every
contributor, whatever they call it.
"""
    ),
    code(
        """
classified = []
for name, value in sorted(DESIGN_OPTIONS.items()):
    result = classify_parameter(name, value)
    if result is not None:
        classified.append({"parameter": name, "raw": str(value), "dimension": result[0], "canonical": result[1]})
classified_frame = pd.DataFrame(classified)

print(classified_frame["dimension"].value_counts().to_string())
print()
lengths = np.sort(classified_frame.query("dimension == 'length'")["canonical"].to_numpy())
print(f"the eight smallest lengths in this design (um): {lengths[:8].round(3)}")
print(f"the three largest  lengths in this design (um): {lengths[-3:].round(3)}")
print()
block, metadata = parameter_block(DESIGN_OPTIONS)
print(f"parameter block: {len(block)} dimensions from {metadata['parameter_count']} options")
print(f"identical to the block inside encode(): {np.allclose(block, vector[METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE : METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE + PARAMETER_BLOCK_SIZE], atol=1e-6)}")
"""
    ),
    code(
        """
# %% hide input
counts = classified_frame["dimension"].value_counts().reindex(PARAMETER_DIMENSION_CLASSES).fillna(0)
widths = {"length": 24, "count": 12, "angle": 8, "boolean": 4, "other": 12}
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Options sorted into dimension classes", "The 24 length coordinates"],
    horizontal_spacing=0.13,
    specs=[[{"type": "bar"}, {"type": "bar"}]],
)
figure.add_trace(
    go.Bar(
        x=list(PARAMETER_DIMENSION_CLASSES),
        y=counts.to_numpy(),
        name="options",
        marker_color="#00798C",
        text=[int(value) for value in counts.to_numpy()],
        textposition="outside",
        hovertemplate="%{x}: %{y} options<extra></extra>",
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Bar(
        x=list(PARAMETER_DIMENSION_CLASSES),
        y=[widths[name] for name in PARAMETER_DIMENSION_CLASSES],
        name="coordinates",
        marker_color="#E9C46A",
        text=[widths[name] for name in PARAMETER_DIMENSION_CLASSES],
        textposition="outside",
        hovertemplate="%{x}: %{y} coordinates<extra></extra>",
    ),
    row=1,
    col=1,
)
labels = ["count", "sum", "mean", "min", "max", "median", "sd", "range"] + [
    f"{index + 1}-smallest" for index in range(8)
] + [f"{index + 1}-largest" for index in range(8)]
figure.add_trace(
    go.Bar(
        x=labels,
        y=block[:24],
        marker_color="#6A4C93",
        showlegend=False,
        hovertemplate="%{x}<br>stored value=%{y:.3f}<extra></extra>",
    ),
    row=1,
    col=2,
)
figure.update_yaxes(title_text="number", row=1, col=1)
figure.update_xaxes(tickangle=-40, row=1, col=2)
figure.update_yaxes(title_text="stored value (signed log scale)", row=1, col=2)
figure.update_layout(
    title="Any parameter schema, any size, becomes the same 96 coordinates",
    template="plotly_white",
    barmode="group",
    height=520,
)
figure.show()
"""
    ),
    markdown(
        """
## 7. Step six: a physics proxy the model does not have to rediscover

The last block runs an actual, if crude, electrostatics calculation. The
conductor boundaries are discretized into segments, each pair gets the
two-dimensional free-space Green function

$$G_{mn} = -\\frac{1}{2\\pi\\varepsilon_0}\\ln\\lVert r_m - r_n \\rVert ,$$

and the linear system is solved for the charge that holds each terminal at unit
potential. Integrating that charge gives an approximate capacitance matrix.

This is **not** a simulation. It is two-dimensional, it ignores the substrate,
the finite metal thickness, and every three-dimensional fringing effect. That is
the point: it captures the part of the map that is identical for every component
class, so the learned head only has to fit a smoother, more transferable
correction instead of rediscovering the inverse-distance law from pixels.
"""
    ),
    code(
        """
segment_points, segment_lengths, owners = [], [], []
for index, terminal in enumerate(terminals[:4]):
    coordinates, weights = _boundary_samples(terminal, target=160)
    segment_points.append(coordinates)
    segment_lengths.append(weights)
    owners.append(np.full(len(coordinates), index))
points_matrix = np.vstack(segment_points)
segment = np.concatenate(segment_lengths)
labels_vector = np.concatenate(owners)

delta = points_matrix[:, None, :] - points_matrix[None, :, :]
separation = np.sqrt((delta**2).sum(axis=2)) * 1e-6
np.fill_diagonal(separation, 1.0)
green = -np.log(separation) / (2 * np.pi * VACUUM_PERMITTIVITY)
np.fill_diagonal(green, -(np.log(segment * 1e-6 / 2) - 1) / (2 * np.pi * VACUUM_PERMITTIVITY))
selector = np.stack([(labels_vector == index).astype(float) for index in range(len(terminals[:4]))], axis=1)
charges = np.linalg.solve(green, selector)
capacitance = selector.T @ charges

proxy_ff = np.abs(capacitance[0, 1]) * 1e15
print(f"two-dimensional proxy  |C(0,1)| = {proxy_ff:.4f} fF per metre of depth")
print(f"simulated (Q3D, 3D)     C(N,S)  = {MEASURED['north_to_south']:.4f} fF")
print()
print("These are different quantities in different units, so the numbers are not")
print("comparable and the proxy is not a prediction. What makes it a useful")
print("feature is that it is monotone in the simulated value across the whole")
print("catalogue: Tutorial 18 measures Spearman +0.920 between this coordinate")
print("and the Q3D mutual capacitance over all 13,683 designs.")
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Solved surface charge for terminal 0 held at unit potential",
        "Connected components and holes as the metal is dilated",
    ],
    horizontal_spacing=0.12,
)
density = charges[:, 0] / segment
limit = float(np.quantile(np.abs(density), 0.98))
figure.add_trace(
    go.Scatter(
        x=points_matrix[:, 0],
        y=points_matrix[:, 1],
        mode="markers",
        marker={
            "size": 7,
            "color": density,
            "colorscale": "RdBu",
            "cmid": 0,
            "cmin": -limit,
            "cmax": limit,
            "colorbar": {"title": "charge<br>density", "x": 0.43, "len": 0.85},
        },
        showlegend=False,
        hovertemplate="x=%{x:.2f} um<br>y=%{y:.2f} um<br>density=%{marker.color:.3g}<extra></extra>",
    ),
    row=1,
    col=1,
)
physics = vector[V2_DIMENSIONS - PHYSICS_BLOCK_SIZE :]
components = np.expm1(physics[16:32])
holes = np.expm1(physics[32:48])
figure.add_trace(
    go.Scatter(
        x=TOPOLOGY_RADII_UM,
        y=components,
        mode="lines+markers",
        name="connected components",
        line={"color": "#00798C", "width": 3},
    ),
    row=1,
    col=2,
)
figure.add_trace(
    go.Scatter(
        x=TOPOLOGY_RADII_UM,
        y=holes,
        mode="lines+markers",
        name="enclosed holes",
        line={"color": "#D1495B", "width": 3},
    ),
    row=1,
    col=2,
)
figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
figure.update_xaxes(title_text="x (um)", row=1, col=1)
figure.update_yaxes(title_text="y (um)", row=1, col=1)
figure.update_xaxes(title_text="dilation radius (um)", type="log", row=1, col=2)
figure.update_yaxes(title_text="count", row=1, col=2)
figure.update_layout(
    title="A crude electrostatic solve, and the scale at which the combs merge",
    template="plotly_white",
    height=520,
)
figure.show()
"""
    ),
    markdown(
        """
Charge piles up on the facing finger edges and on the outer corners, which is
exactly where a textbook says it should. The right panel is the topology
signature: the two terminals stay separate until the dilation radius reaches
half the finger gap, then merge into one component. The radius at which that
happens **is** the gap, read off without ever naming it.

## 8. Assembling the vector

Five blocks, concatenated in a fixed order. No normalization against a
catalogue, no whitening, no learned projection - just the measurements.
"""
    ),
    code(
        """
# %% hide input
bounds = [
    ("physical metrics", 0, METRIC_BLOCK_SIZE, "#F4A261"),
    ("coupling spectrum", METRIC_BLOCK_SIZE, METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE, "#00798C"),
    (
        "shape spectrum",
        METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE,
        METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE,
        "#6A4C93",
    ),
    (
        "parameter statistics",
        METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE,
        V2_DIMENSIONS - PHYSICS_BLOCK_SIZE,
        "#E9C46A",
    ),
    ("physics proxy", V2_DIMENSIONS - PHYSICS_BLOCK_SIZE, V2_DIMENSIONS, "#2A9D8F"),
]
figure = go.Figure()
for name, start, stop, color in bounds:
    figure.add_trace(
        go.Bar(
            x=np.arange(start, stop),
            y=vector[start:stop],
            name=f"{name} ({stop - start})",
            marker_color=color,
            hovertemplate=f"<b>{name}</b><br>coordinate %{{x}}<br>value=%{{y:.3f}}<extra></extra>",
        )
    )
figure.update_layout(
    title=f"The complete universal-geometry-v2 vector for {SOURCE_ID}",
    xaxis_title="coordinate",
    yaxis_title="value",
    bargap=0,
    template="plotly_white",
    height=470,
)
figure.show()

occupancy = pd.DataFrame(
    [
        {
            "block": name,
            "dimensions": stop - start,
            "non-zero here": int(np.count_nonzero(vector[start:stop])),
        }
        for name, start, stop, _ in bounds
    ]
)
print(occupancy.to_string(index=False))
print(f"\\ntotal: {V2_DIMENSIONS} dimensions, {int(np.count_nonzero(vector))} non-zero for this two-terminal device")
"""
    ),
    markdown(
        """
Roughly a third of the coordinates are non-zero. The empty ones are the
terminal-pair spectra for terminals this device does not have, and the parameter
classes it does not use. That reserved capacity costs nothing statistically - a
constant column contributes nothing to a standardized regression - and it is
what lets a four-terminal qubit-coupler share the vector layout with this
two-terminal capacitor.

## 9. Checking the invariances

The encoder claims a specific set of symmetries. Each one is checkable in a few
lines, so here they are, checked.
"""
    ),
    code(
        """
import klayout.db as kdb


def transformed_copy(path, destination, *, shift=(0.0, 0.0), scale=1.0, mirror=False, rotate=False):
    layout = kdb.Layout()
    layout.read(str(path))
    transform = kdb.DCplxTrans(scale, 270.0 if rotate else 0.0, mirror, shift[0], shift[1])
    for cell in layout.each_cell():
        for index in layout.layer_indices():
            cell.shapes(index).transform(transform)
    layout.write(str(destination))
    return destination


work = Path(os.getenv("TMPDIR", "/tmp")) / "squadds-tutorial19"
work.mkdir(parents=True, exist_ok=True)

coupling_span = slice(METRIC_BLOCK_SIZE, METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE)
metric_span = slice(0, METRIC_BLOCK_SIZE)

checks = []
for label, kwargs in [
    ("translate by (4000, -2500) um", {"shift": (4000.0, -2500.0)}),
    ("mirror in x", {"mirror": True}),
    ("rotate by 270 degrees", {"rotate": True}),
    ("scale by 3x", {"scale": 3.0}),
]:
    variant = transformed_copy(GDS_PATH, work / f"{abs(hash(label))}.gds", **kwargs)
    moved = encode(variant, DESIGN_OPTIONS)
    checks.append(
        {
            "transform": label,
            "whole vector": round(float(np.abs(vector - moved).max()), 6),
            "coupling spectrum": round(float(np.abs(vector[coupling_span] - moved[coupling_span]).max()), 6),
            "physical metrics": round(float(np.abs(vector[metric_span] - moved[metric_span]).max()), 6),
        }
    )
invariance = pd.DataFrame(checks)
print(invariance.to_string(index=False))
print()
print("Read the last two columns separately: the coupling spectrum is a set of")
print("distances and is orientation free, while the metric block deliberately")
print("stores bbox width and height apart, so rotating a 11 x 31 um device swaps")
print("them. Only translation is invariant everywhere.")
"""
    ),
    markdown(
        """
Translation is exact to the last bit, because the encoder re-origins the layout
on its conductor bounds before measuring anything.

Mirroring barely moves the vector. The residual is not a modelling choice but a
sampling artefact: reflecting the boundary changes which of the 1,024 samples
land near a bin centre, and soft binning keeps that perturbation small.

**Rotation deserves care, and the table corrects a tempting overstatement.** The
coupling spectrum is built from distances, so it is essentially untouched by a
270-degree rotation. The metric block is not, and should not be: it stores
`log1p_bbox_width_um` and `log1p_bbox_height_um` as separate coordinates, so
rotating an 11 by 31 micrometre device swaps them and moves those entries a long
way. v2 is orientation-free in the blocks that describe shape and coupling, and
orientation-aware in the block that describes extent. If a downstream task needs
full rotational invariance, drop or symmetrize the four extent coordinates rather
than assuming the whole vector already has it.

Scale is the last row and the important one. v0 and v1 are invariant here, which
sounds like a virtue and is actually the defect Tutorial 18 traced: a design and
its 3x copy have very different capacitance, so an encoder that maps them to the
same vector has destroyed the answer. v2 moves, and it moves in a structured way
- the whole coupling spectrum slides up by exactly $\\log 3$ in separation.

## 10. The scenario this was all built for

Now the payoff. A group we have never met sends us one capacitor. It has a
different outline, a different finger style, a surrounding guard ring, and **28
parameters with names none of our tooling recognizes**.

We have never seen this design, never fitted anything to it, and have no
simulation result for it. We encode it with the same frozen function and ask the
catalogue which of our 13,683 designs it most resembles.
"""
    ),
    code(
        """
def build_foreign_capacitor(path):
    \"\"\"A deliberately different two-terminal capacitor from an imaginary group.\"\"\"
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")

    def box(x0, y0, x1, y1):
        return kdb.Box(*[int(round(value * 1000)) for value in (x0, y0, x1, y1)])

    left = kdb.Region()
    right = kdb.Region()
    digits, pitch, digit_length = 9, 5.5, 26.0
    for index in range(digits):
        y = index * pitch
        # Tapered horizontal digits interleaved from opposite spines.
        target = left if index % 2 == 0 else right
        if index % 2 == 0:
            target.insert(box(-digit_length, y, 2.0, y + 2.2))
        else:
            target.insert(box(-2.0, y, digit_length, y + 2.2))
    left.insert(box(-digit_length - 4.0, -3.0, -digit_length, digits * pitch + 3.0))
    right.insert(box(digit_length, -3.0, digit_length + 4.0, digits * pitch + 3.0))
    conductors = left + right

    guard = kdb.Region(box(-digit_length - 16, -15, digit_length + 16, digits * pitch + 15))
    ground = guard - conductors.sized(9000)
    top.shapes(layout.layer(1, 0)).insert(ground)
    top.shapes(layout.layer(1, 10)).insert(conductors)
    top.shapes(layout.layer(2, 0)).insert(box(-digit_length - 4.0, digits * pitch + 3.0, -digit_length, digits * pitch + 5.0))
    top.shapes(layout.layer(3, 0)).insert(box(digit_length, -5.0, digit_length + 4.0, -3.0))
    layout.write(str(path))
    return path


FOREIGN_OPTIONS = {
    "digit_pitch": "5.5um",
    "digit_extent": "26um",
    "digit_thickness": "2.2um",
    "digit_population": 9,
    "spine_thickness": "4um",
    "guard_clearance": "9um",
    "guard_extent_x": "84um",
    "guard_extent_y": "74um",
    "substrate_index": 11.45,
    "metal_layer": 1,
    "is_symmetric": True,
    "launch_rotation": "0deg",
    **{f"aux_length_{index}": f"{1.0 + 0.7 * index}um" for index in range(10)},
    **{f"aux_flag_{index}": bool(index % 2) for index in range(6)},
}

foreign_path = build_foreign_capacitor(work / "foreign.gds")
foreign_vector = encode(foreign_path, FOREIGN_OPTIONS)
print(f"foreign design: {len(FOREIGN_OPTIONS)} parameters, none of which appear in our catalogue")
print(f"shared parameter names with our design: {sorted(set(FOREIGN_OPTIONS) & set(DESIGN_OPTIONS))}")
print(f"encoded to {foreign_vector.shape[0]} dimensions with no refitting")
"""
    ),
    code(
        """
CACHE = Path(os.getenv("SQUADDS_TUTORIAL18_CACHE", Path.home() / ".cache/squadds/tutorial18"))
V2_TABLE = Path(os.getenv("SQUADDS_V2_TABLE", CACHE / "universal-geometry-v2.parquet"))
if not V2_TABLE.is_file():
    raise FileNotFoundError(f"Build the v2 catalogue first (see Tutorial 18); expected {V2_TABLE}.")

catalogue = pd.read_parquet(V2_TABLE).drop_duplicates("design_id").reset_index(drop=True)
catalogue_matrix = np.vstack(catalogue["embedding"].to_numpy()).astype(np.float32)
targets = {
    item["notes"]["source_id"]: item["sim_results"]["north_to_south"] for item in json.loads(DATABASE.read_text())
}

# Standardize in float64: in float32 the spread of a constant coordinate such as
# terminal_count does not evaluate to exactly zero, and the column survives the
# filter only to divide by zero a moment later.
reference = catalogue_matrix.astype(np.float64)
center = reference.mean(axis=0)
scale = reference.std(axis=0)
keep = scale > 1e-8
center, scale = center[keep], scale[keep]
standardized = (reference[:, keep] - center) / scale
query = (foreign_vector.astype(np.float64)[keep] - center) / scale
print(f"{int(keep.sum())} of {V2_DIMENSIONS} coordinates vary across the catalogue and define the metric")

unit = standardized / np.maximum(np.linalg.norm(standardized, axis=1, keepdims=True), 1e-12)
query_unit = query / max(np.linalg.norm(query), 1e-12)
similarity = unit @ query_unit
assert np.isfinite(similarity).all(), "similarity must be finite for every catalogue row"
best = np.argsort(similarity)[::-1][:8]

neighbours = pd.DataFrame(
    {
        "source_id": catalogue.loc[best, "source_id"].to_numpy(),
        "cosine": similarity[best].round(4),
        "min gap (um)": catalogue.loc[best, "minimum_pair_gap_um"].to_numpy().round(3),
        "simulated C(N,S) (fF)": [round(targets.get(name, float("nan")), 4) for name in catalogue.loc[best, "source_id"]],
    }
)
print(f"foreign design minimum conductor gap: {float(np.expm1(foreign_vector[METRIC_NAMES.index('log1p_minimum_pair_gap_um')])):.3f} um")
print()
print(neighbours.to_string(index=False))
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["The stranger's capacitor", "Where it lands in our catalogue"],
    horizontal_spacing=0.12,
    column_widths=[0.44, 0.56],
)
foreign_geometry = read_layer_geometry(foreign_path)
foreign_grouped = _role_geometry(foreign_geometry, None)
for role in ("domain", "conductor", "port"):
    for key, shape in foreign_grouped[role]:
        for trace in polygon_traces(shape, role, ROLE_COLORS[role], show=False):
            figure.add_trace(trace, row=1, col=1)

rng = np.random.default_rng(19)
shown = rng.choice(len(catalogue), size=min(4000, len(catalogue)), replace=False)
gap_values = catalogue["minimum_pair_gap_um"].to_numpy()
figure.add_trace(
    go.Scattergl(
        x=similarity[shown],
        y=gap_values[shown],
        mode="markers",
        marker={"size": 4, "opacity": 0.3, "color": "#C7CDD4"},
        name="catalogue",
        hovertemplate="cosine=%{x:.3f}<br>min gap=%{y:.2f} um<extra></extra>",
    ),
    row=1,
    col=2,
)
figure.add_trace(
    go.Scattergl(
        x=similarity[best],
        y=gap_values[best],
        mode="markers",
        marker={"size": 11, "color": "#D1495B", "symbol": "star"},
        name="nearest eight",
        hovertemplate="cosine=%{x:.3f}<br>min gap=%{y:.2f} um<extra></extra>",
    ),
    row=1,
    col=2,
)
figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
figure.update_xaxes(title_text="x (um)", row=1, col=1)
figure.update_yaxes(title_text="y (um)", row=1, col=1)
figure.update_xaxes(title_text="cosine similarity to the foreign design", row=1, col=2)
figure.update_yaxes(title_text="minimum conductor gap (um)", row=1, col=2)
figure.update_layout(
    title="An unseen design, an unseen parameter schema, and the neighbours it selects",
    template="plotly_white",
    height=530,
)
figure.show()
"""
    ),
    markdown(
        """
The nearest neighbours are not chosen because a parameter called `finger_gap`
matched something - no name in the foreign schema matches ours. They are chosen
because the two designs put comparable amounts of metal at comparable
separations, which is the quantity that sets the capacitance.

That is the whole proposition. The stranger did not have to adopt our design
tool, our parameter names, or our component library. They had to emit a GDS file
with documented layer roles, and the rest is measurement.

## 11. Summary

| Step | Input | Output | Fitted to anything? |
| --- | --- | --- | --- |
| Roles and terminals | GDS polygons | ordered terminal list | no |
| Physical metrics | terminals, raster | 48 named scalars in um | no |
| Coupling spectrum | boundary samples, exact distances | 192 log-spaced bins | no |
| Shape spectrum | raster FFT, contour FFT | 128 coefficients | no |
| Parameter statistics | any option mapping | 96 typed coordinates | no |
| Physics proxy | boundary-element solve | 48 coordinates | no |

Nothing in that column changes, which is the property the whole design exists to
guarantee. `encode` is a pure function: run it on one design or on a hundred
thousand, in our lab or in someone else's, and coordinate 61 is always the
facing boundary length between terminals 0 and 1 at a separation of about 1.7
micrometers.

Two honest caveats carried over from Tutorial 18. The raw cosine metric has a
compressed spread, so the standardization used in section 10 should eventually be
replaced by a published, frozen whitening transform rather than statistics taken
from whatever catalogue happens to be loaded. And every measurement here is of
one component family; the cross-class study remains to be run.

**Where to go next**

- Tutorial 18 for the quantitative comparison against `static-shape-v0`.
- Tutorial 17 for the cross-component transfer study that v2 has not yet repeated.
- `squadds/layouts/geometry_v2.py` for the encoder itself, and
  `universal_v2_schema()` for the machine-readable contract.
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, str(OUTPUT))
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} cells.")
