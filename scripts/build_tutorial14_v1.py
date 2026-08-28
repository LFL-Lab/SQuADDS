#!/usr/bin/env python
"""Build the pedagogical v0/v1 layout embedding notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("tutorials/Tutorial-14_Exploring_Static_Layout_Embeddings.ipynb")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, *, hidden: bool = False):
    cell = nbf.v4.new_code_cell(source.strip())
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
    "language_info": {"name": "python", "version": "3.10"},
}
notebook["cells"] = [
    markdown(
        """
# Tutorial 14: From static-embedding-v0 to universal-geometry-v1

This tutorial develops intuition for two **versioned layout embedding standards**.
We use the complete 20,062-design `GeneralizedCapNInterdigital` sweep so that every
comparison is paired: the same simulated design, the same immutable GDS, two
different representations.

By the end, you will be able to:

1. explain what every v0 and v1 block represents;
2. switch standards through the SQuADDS API;
3. inspect how geometry and capacitance organize in either latent space;
4. use cosine similarity to retrieve physically related layouts; and
5. test whether v1 actually improves on v0 using explicit acceptance gates.

> **Important:** capacitance is used only to evaluate the embedding. It is never
> included in either embedding vector.
"""
    ),
    markdown(
        """
## 1. Load one paired catalogue

The notebook first checks `SQUADDS_TUTORIAL14_RELEASE` for a local staged release.
After the Hugging Face pull request is merged, no environment variable is needed:
both clients download their selected version from `SQuADDS/SQuADDS_Layout_Embeddings`.
"""
    ),
    code(
        """
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots

from squadds.core.db import SQuADDS_DB
from squadds.layouts import (
    LayoutEmbeddingClient,
    UNIVERSAL_METRIC_NAMES,
)

release_env = os.environ.get("SQUADDS_TUTORIAL14_RELEASE")
release = Path(release_env) if release_env else None
if release and release.is_dir():
    schema_path = release / "embeddings/models/universal-geometry-v1/schema.json"
    control_map_path = release / "embeddings/models/universal-geometry-v1/control-map.parquet"
    v0_client = LayoutEmbeddingClient(
        version="v0", embedding_path=release / "embeddings/metadata/static-embedding-v0.parquet"
    )
    v1_client = LayoutEmbeddingClient(
        version="v1",
        embedding_path=release / "embeddings/metadata/universal-geometry-v1.parquet",
        schema_path=schema_path,
        control_map_path=control_map_path,
    )
    database_path = release / "db/coupler-GeneralizedCapNInterdigital-cap_matrix.json"
else:
    v0_client = LayoutEmbeddingClient(version="v0")
    v1_client = LayoutEmbeddingClient(version="v1")
    database_path = Path(
        hf_hub_download(
            repo_id="SQuADDS/SQuADDS_DB",
            repo_type="dataset",
            filename="coupler-GeneralizedCapNInterdigital-cap_matrix.json",
        )
    )

v0 = v0_client.embeddings().query("component_name == 'GeneralizedCapNInterdigital'")
v1 = v1_client.embeddings().query("component_name == 'GeneralizedCapNInterdigital'")
paired = (
    v1[["layout_id", "design_id", "source_id"]]
    .merge(v0[["layout_id"]], on="layout_id", validate="one_to_one")
    .reset_index(drop=True)
)
rows = json.loads(database_path.read_text())
row_by_source = {row["notes"]["source_id"]: row for row in rows}
schema_v1 = v1_client.schema()
control_map = v1_client.control_map()

print(f"Paired GeneralizedCapNInterdigital layouts: {len(paired):,}")
print(f"v0 dimensions: {len(v0.iloc[0]['embedding']):,}")
print(f"v1 dimensions: {len(v1.iloc[0]['embedding']):,}")
"""
    ),
    markdown(
        """
## 2. Two standards, two design philosophies

`static-embedding-v0` is deliberately transparent: one summed parameter, ten geometric
moments, and every pixel of a 96 by 96 signed bitmap. It proved that layout geometry
can be joined to simulation data, but the sum destroys parameter names and the raw
bitmap is large.

`universal-geometry-v1` accepts only a GDS file, a small layer-role mapping, and the
native parameter dictionary from **any** layout tool. It separates:

- **32 geometry metrics:** scale-aware physical and morphological values; availability
  remains metadata and cannot inflate cosine similarity;
- **768 shape coefficients:** variance-selected full-spectrum 2D DCT coefficients from
  signed material and boundary-distance rasters at 96 by 96 resolution;
- **224 control channels:** stable signed hashing of parameter paths after each named
  control is centered and scaled, with a sidecar map back to the layout knobs.

Each block is normalized and explicitly weighted. The fitted statistics and selected
frequencies use geometry and layout parameters only; capacitance is held out for
evaluation.
"""
    ),
    code(
        """
fig = go.Figure()
fig.add_trace(
    go.Bar(
        name="v0",
        x=["parameter controls", "geometry metrics", "shape"],
        y=[1, 10, 96 * 96],
        marker_color="#D1495B",
        text=[1, 10, 96 * 96],
        textposition="outside",
    )
)
fig.add_trace(
    go.Bar(
        name="v1",
        x=["parameter controls", "geometry metrics", "shape"],
        y=[224, 32, 768],
        marker_color="#00798C",
        text=[224, 32, 768],
        textposition="outside",
    )
)
fig.update_layout(
    title="Embedding anatomy: 9,227 dimensions become 1,024 structured dimensions",
    yaxis_title="dimensions (log scale)",
    yaxis_type="log",
    barmode="group",
    template="plotly_white",
    height=500,
)
fig.show()
""",
        hidden=True,
    ),
    markdown(
        """
## 3. Shape detail must survive compression

The v0 panel below is its bitmap block. V1 reconstructs its signed material and
boundary-distance channels through the public API using the exact selected frequencies,
centering statistics, partial whitening, and row norm stored by the standard.

Unlike the rejected v1.0 candidate, v1.1 does not crop the spectrum to an 8 by 8
low-pass square. Its target-blind variance selection can retain the higher frequencies
that encode finger count, narrow gaps, and edge curvature.
"""
    ),
    code(
        """
example_id = paired.iloc[len(paired) // 3]["layout_id"]
v0_record = v0_client.get(example_id)
v0_bitmap = np.asarray(v0_record["embedding"], dtype=np.float32)[11:].reshape(96, 96)
v1_rasters = v1_client.shape_rasters(example_id)
reconstructed = [
    v1_rasters["signed_functional_material"],
    v1_rasters["signed_distance_to_functional_boundary"],
]

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("v0: signed 96x96 bitmap", "v1: functional material", "v1: signed distance"),
)
fig.add_trace(go.Heatmap(z=v0_bitmap, colorscale="RdBu", showscale=False), row=1, col=1)
fig.add_trace(go.Heatmap(z=reconstructed[0], colorscale="RdBu", showscale=False), row=1, col=2)
fig.add_trace(go.Heatmap(z=reconstructed[1], colorscale="BrBG", showscale=False), row=1, col=3)
fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
fig.update_layout(title="One GDS represented by each standard", template="plotly_white", height=430)
fig.show()
""",
        hidden=True,
    ),
    markdown(
        """
## 4. Explore geometry and capacitance in both latent spaces

We use a fixed, target-blind random sample for a responsive browser and reproducible
acceptance benchmark. The same sample and the same randomized projection are used for
both standards. Open the dropdown to
color by **every numeric layout parameter and capacitance result present in the
dataset**. Smooth color gradients indicate that nearby vectors preserve that
quantity; abrupt mixing reveals a weakly represented factor.
"""
    ),
    code(
        """
number_pattern = re.compile(
    r"^\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*([A-Za-zµμ]*)\\s*$"
)
unit_scale = {"": 1.0, "um": 1.0, "µm": 1.0, "μm": 1.0, "nm": 1e-3, "mm": 1e3}

def numeric(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = number_pattern.match(value)
        if match and match.group(2) in unit_scale:
            return float(match.group(1)) * unit_scale[match.group(2)]
    return np.nan

catalogue = paired.copy()
all_parameter_names = sorted(
    {
        name
        for row in rows
        for name, value in row["design"]["design_options"].items()
        if np.isfinite(numeric(value))
    }
)
capacitance_names = [
    "north_to_north",
    "north_to_south",
    "north_to_ground",
    "south_to_south",
    "south_to_ground",
    "ground_to_ground",
]
for name in all_parameter_names:
    catalogue[f"geometry: {name}"] = [
        numeric(row_by_source[source]["design"]["design_options"].get(name))
        for source in catalogue["source_id"]
    ]
for name in capacitance_names:
    catalogue[f"capacitance: {name} (fF)"] = [
        numeric(row_by_source[source]["sim_results"].get(name)) for source in catalogue["source_id"]
    ]

sample_size = min(2000, len(catalogue))
sample_indices = np.random.default_rng(1401).choice(
    len(catalogue), size=sample_size, replace=False
)
sample = catalogue.iloc[sample_indices].reset_index(drop=True)
v0_lookup = v0.set_index("layout_id")
v1_lookup = v1.set_index("layout_id")
v0_matrix = np.vstack(v0_lookup.loc[sample["layout_id"], "embedding"]).astype(np.float32)
v1_matrix = np.vstack(v1_lookup.loc[sample["layout_id"], "embedding"]).astype(np.float32)

def compact_v0(matrix):
    moments = matrix[:, :11]
    pixels = matrix[:, 11:].reshape(-1, 96, 96)
    pooled = pixels.reshape(-1, 12, 8, 12, 8).mean(axis=(2, 4))
    compact = np.column_stack([moments, pooled.reshape(len(matrix), -1)])
    norms = np.linalg.norm(compact, axis=1, keepdims=True)
    return compact / np.where(norms > 0, norms, 1.0)

def project(matrix):
    rng = np.random.default_rng(14)
    sketch_size = min(64, matrix.shape[1])
    sketch = matrix @ rng.normal(
        0.0, 1.0 / np.sqrt(sketch_size), size=(matrix.shape[1], sketch_size)
    ).astype(np.float32)
    sketch -= sketch.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(sketch, full_matrices=False)
    return u[:, :2] * singular_values[:2]

v0_compact = compact_v0(v0_matrix)
v0_xy = project(v0_compact)
v1_xy = project(v1_matrix)
color_fields = [
    column for column in sample.columns if column.startswith(("geometry: ", "capacitance: "))
]
initial_field = "geometry: finger_count"
initial_color = sample[initial_field]
hover = np.column_stack(
    [
        sample["source_id"],
        sample["geometry: finger_count"],
        sample["geometry: finger_length"],
        sample["capacitance: north_to_south (fF)"],
    ]
)

fig = make_subplots(rows=1, cols=2, subplot_titles=("static-embedding-v0", "universal-geometry-v1"))
for column, xy in enumerate((v0_xy, v1_xy), start=1):
    fig.add_trace(
        go.Scattergl(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker={
                "color": initial_color,
                "colorscale": "Turbo",
                "size": 6,
                "opacity": 0.72,
                "showscale": column == 2,
                "colorbar": {"title": initial_field, "x": 1.02},
            },
            customdata=hover,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>fingers=%{customdata[1]}"
                "<br>finger length=%{customdata[2]} um"
                "<br>C(north,south)=%{customdata[3]:.3f} fF<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=column,
    )

buttons = [
    {
        "label": field.replace("geometry: ", "").replace("capacitance: ", "C: "),
        "method": "update",
        "args": [
            {"marker.color": [sample[field], sample[field]]},
            {"title": f"Same paired layouts colored by {field}"},
        ],
    }
    for field in color_fields
]
fig.update_layout(
    title=f"Same paired layouts colored by {initial_field}",
    template="plotly_white",
    height=650,
    margin={"r": 210, "t": 130},
    updatemenus=[
        {
            "buttons": buttons,
            "direction": "down",
            "x": 0.0,
            "y": 1.16,
            "xanchor": "left",
            "yanchor": "top",
        }
    ],
)
fig.update_xaxes(title_text="projection axis 1")
fig.update_yaxes(title_text="projection axis 2")
fig.show()
""",
        hidden=True,
    ),
    markdown(
        """
## 5. Similarity is a statement about representation

Cosine similarity compares vector directions. A value near one means that the
standard considers two layouts close; it does **not** by itself prove equal physics.
The rejected v1.0 candidate collapsed most random pairs near cosine similarity one:
constant availability bits, uncentered controls, and duplicate low-frequency channels
created a large common direction. V1.1 removes those offsets. A healthy distribution
must retain meaningful dynamic range rather than merely looking compact.
"""
    ),
    code(
        """
rng = np.random.default_rng(140)
pair_left = rng.integers(0, sample_size, size=8000)
pair_right = rng.integers(0, sample_size, size=8000)
different = pair_left != pair_right
pair_left, pair_right = pair_left[different], pair_right[different]
v0_similarity = np.sum(v0_matrix[pair_left] * v0_matrix[pair_right], axis=1)
v1_similarity = np.sum(v1_matrix[pair_left] * v1_matrix[pair_right], axis=1)

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=v0_similarity, name="v0", opacity=0.68, histnorm="probability density", marker_color="#D1495B"
    )
)
fig.add_trace(
    go.Histogram(
        x=v1_similarity, name="v1", opacity=0.68, histnorm="probability density", marker_color="#00798C"
    )
)
fig.update_layout(
    title="Cosine similarity of 8,000 random layout pairs",
    xaxis_title="cosine similarity",
    yaxis_title="density",
    barmode="overlay",
    template="plotly_white",
    height=500,
)
fig.show()
""",
        hidden=True,
    ),
    markdown(
        """
## 6. Visual nearest-neighbor retrieval

Here one query design is held fixed. Each standard retrieves its four closest
neighbors from the same candidate pool. The displayed images are v0's transparent
bitmaps, used only as a common visualization canvas. Compare topology, finger count,
and capacitance in the hover labels.
"""
    ),
    code(
        """
query_index = sample_size // 2
query_id = sample.iloc[query_index]["layout_id"]
v0_scores = v0_matrix @ v0_matrix[query_index]
v1_scores = v1_matrix @ v1_matrix[query_index]
v0_scores[query_index] = -np.inf
v1_scores[query_index] = -np.inf
v0_neighbors = np.argsort(v0_scores)[-4:][::-1]
v1_neighbors = np.argsort(v1_scores)[-4:][::-1]

fig = make_subplots(
    rows=2,
    cols=5,
    row_titles=("v0 neighbors", "v1 neighbors"),
    column_titles=("query", "nearest 1", "nearest 2", "nearest 3", "nearest 4"),
    horizontal_spacing=0.025,
    vertical_spacing=0.12,
)
for row_number, neighbors in enumerate((v0_neighbors, v1_neighbors), start=1):
    indices = [query_index, *neighbors]
    for column_number, index in enumerate(indices, start=1):
        bitmap = v0_matrix[index, 11:].reshape(96, 96)
        fig.add_trace(
            go.Heatmap(z=bitmap, colorscale="RdBu", zmid=0, showscale=False, hoverinfo="skip"),
            row=row_number,
            col=column_number,
        )
        item = sample.iloc[index]
        axis_number = (row_number - 1) * 5 + column_number
        xref = "x domain" if axis_number == 1 else f"x{axis_number} domain"
        yref = "y domain" if axis_number == 1 else f"y{axis_number} domain"
        fig.add_annotation(
            x=0.5,
            y=-0.08,
            xref=xref,
            yref=yref,
            text=(
                f"n={item['geometry: finger_count']:.0f}, "
                f"C={item['capacitance: north_to_south (fF)']:.2f} fF"
            ),
            showarrow=False,
            font={"size": 10},
        )
fig.update_yaxes(autorange="reversed")
fig.update_layout(
    title="One query, two definitions of nearest",
    template="plotly_white",
    height=700,
    margin={"b": 90},
)
fig.show()
""",
        hidden=True,
    ),
    markdown(
        """
## 7. Quantify local faithfulness

For 300 query layouts we retrieve one non-self neighbor from the same 2,000-layout
pool. Lower error is better. We report:

- absolute finger-count difference;
- standardized distance across all numeric geometry controls;
- distance between normalized 96 by 96 signed shape rasters;
- absolute mutual-capacitance difference as a physics proxy.

This is a strict paired comparison. V1 never saw capacitance, so any improvement in
the fourth metric comes from a more faithful geometry neighborhood.
"""
    ),
    code(
        """
query_indices = np.linspace(0, sample_size - 1, 300, dtype=int)
v0_query_scores = v0_matrix[query_indices] @ v0_matrix.T
v1_query_scores = v1_matrix[query_indices] @ v1_matrix.T
v0_query_scores[np.arange(len(query_indices)), query_indices] = -np.inf
v1_query_scores[np.arange(len(query_indices)), query_indices] = -np.inf
v0_nearest = np.argmax(v0_query_scores, axis=1)
v1_nearest = np.argmax(v1_query_scores, axis=1)

geometry_columns = [column for column in color_fields if column.startswith("geometry: ")]
geometry_values = sample[geometry_columns].to_numpy(dtype=float)
means = np.nanmean(geometry_values, axis=0)
stds = np.nanstd(geometry_values, axis=0)
stds = np.where(stds > 1e-12, stds, 1.0)
standardized_geometry = np.nan_to_num((geometry_values - means) / stds)
finger_values = sample["geometry: finger_count"].to_numpy()
cap_values = sample["capacitance: north_to_south (fF)"].to_numpy()
shape_values = v0_matrix[:, 11:].copy()
shape_values /= np.maximum(np.linalg.norm(shape_values, axis=1, keepdims=True), 1e-12)

def retrieval_metrics(neighbors):
    return [
        np.mean(np.abs(finger_values[query_indices] - finger_values[neighbors])),
        np.mean(
            np.linalg.norm(
                standardized_geometry[query_indices] - standardized_geometry[neighbors], axis=1
            )
        ),
        np.mean(np.linalg.norm(shape_values[query_indices] - shape_values[neighbors], axis=1)),
        np.mean(np.abs(cap_values[query_indices] - cap_values[neighbors])),
    ]

v0_quality = retrieval_metrics(v0_nearest)
v1_quality = retrieval_metrics(v1_nearest)
metric_labels = [
    "finger count MAE",
    "geometry distance",
    "shape distance",
    "mutual C MAE (fF)",
]
quality = pd.DataFrame({"metric": metric_labels, "v0": v0_quality, "v1": v1_quality})
display(quality.round(4))

fig = make_subplots(rows=1, cols=4, subplot_titles=metric_labels)
for column, metric in enumerate(metric_labels, start=1):
    values = quality.loc[quality["metric"] == metric, ["v0", "v1"]].iloc[0]
    fig.add_trace(
        go.Bar(
            x=["v0", "v1"],
            y=values,
            marker_color=["#D1495B", "#00798C"],
            text=np.round(values, 3),
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=column,
    )
fig.update_layout(
    title="Nearest-neighbor error: lower is better",
    template="plotly_white",
    height=460,
)
fig.show()

v0_finger_exact = np.mean(finger_values[query_indices] == finger_values[v0_nearest])
v1_finger_exact = np.mean(finger_values[query_indices] == finger_values[v1_nearest])
acceptance = pd.DataFrame(
    [
        ("topology", v0_finger_exact, v1_finger_exact, v1_finger_exact >= v0_finger_exact),
        ("all parameters", v0_quality[1], v1_quality[1], v1_quality[1] <= 0.90 * v0_quality[1]),
        ("shape", v0_quality[2], v1_quality[2], v1_quality[2] <= 1.10 * v0_quality[2]),
        ("physics proxy", v0_quality[3], v1_quality[3], v1_quality[3] <= v0_quality[3]),
        (
            "similarity dynamic range",
            np.std(v0_similarity),
            np.std(v1_similarity),
            np.std(v1_similarity) >= 0.60 * np.std(v0_similarity),
        ),
    ],
    columns=["gate", "v0 reference", "v1 result", "passed"],
)
display(acceptance.round(4))
assert acceptance["passed"].all(), "v1 must pass every gate before it is described as better"
""",
        hidden=True,
    ),
    markdown(
        """
The claim is intentionally conditional: **v1 is accepted only when every row in the
executed gate table passes**. Compactness alone is not a quality metric. The topology
gate prevents visibly different finger patterns from being called neighbors, the
shape gate prevents excessive compression loss, and capacitance remains a genuinely
held-out physics check.
"""
    ),
    markdown(
        """
## 8. The inverse-design hook: named controls survive

V0 adds every numeric parameter into one scalar. Width + count and count + width are
indistinguishable, and two different parameter sets can share the same sum. V1 hashes
the **parameter path and value together**, writes each row's names, values, indices,
and signs, and publishes the global control map below. Collisions are possible in a
fixed-size hashed block, but they are visible and auditable rather than silent.
"""
    ),
    code(
        """
fig = go.Figure(
    go.Scatter(
        x=control_map["hash_index"],
        y=control_map["parameter_name"],
        mode="markers",
        marker={
            "size": 11,
            "color": control_map["hash_sign"],
            "colorscale": [[0, "#D1495B"], [1, "#00798C"]],
            "cmin": -1,
            "cmax": 1,
            "colorbar": {"title": "hash sign", "x": 1.03},
        },
        customdata=np.column_stack(
            [control_map["minimum"], control_map["maximum"], control_map["count"]]
        ),
        hovertemplate=(
            "<b>%{y}</b><br>channel=%{x}<br>range=[%{customdata[0]:.3g}, "
            "%{customdata[1]:.3g}]<br>records=%{customdata[2]}<extra></extra>"
        ),
    )
)
fig.update_layout(
    title="Every GeneralizedCapNInterdigital layout knob maps back to a v1 control channel",
    xaxis_title="v1 parameter-control channel (0-127)",
    yaxis_title="canonical layout parameter",
    template="plotly_white",
    height=780,
    margin={"l": 220, "r": 130},
)
fig.show()
""",
        hidden=True,
    ),
    markdown(
        """
## 9. Choose v0 or v1 explicitly through the API

The default remains v0 for backward compatibility. New work can select the accepted
v1.1 standard explicitly. Both
standards use the same `layout_id`, so changing versions does not change how a
simulation row resolves to its GDS. The MCP tools `get_layout_embedding` and
`find_similar_layouts` expose the same `embedding_version="v0" | "v1"` argument
for agent and service integrations.
"""
    ),
    code(
        """
v0_api = LayoutEmbeddingClient(version="v0")
v1_api = LayoutEmbeddingClient(version="v1")

# Direct lookup when a stable layout_id is already available:
v1_record = v1_client.get(example_id)
neighbors = v1_client.nearest(
    example_id,
    limit=5,
    component_name="GeneralizedCapNInterdigital",
)

# Database-row bridge (shown without a network call because this notebook uses
# the local staged clients while the Hugging Face v1 PR is under review):
# SQuADDS_DB.get_layout_embedding(
#     rows[0],
#     embedding_client=v1_client,
#     embedding_version="v1",
# )

print(v1_record["embedding_model"], len(v1_record["embedding"]))
print("nearest cosine similarities:", [round(item["cosine_similarity"], 4) for item in neighbors])
"""
    ),
    markdown(
        """
## 10. What v1 improves, and what it does not claim yet

**Why v1 is the better ML substrate**

- It is about 9 times smaller while retaining physical scale and functional shape.
- Metric, shape, and controls receive explicit, audited influence rather than equal
  weight by assumption.
- Parameter identity survives, enabling gradients or generated proposals to be
  translated back into the originating layout tool.
- A foreign institution needs only GDS, its native parameter dictionary, and a tiny
  layer-role mapping to enter the same frozen space.
- The schema records every transformation, normalization statistic, and role.

**What remains for later v1 releases**

This first frozen v1 release is calibrated on GeneralizedCapNInterdigital only.
The algorithm already accepts arbitrary layouts, but the frozen normalization and
frequency selection must be validated on the union of all SQuADDS component families.
Cross-tool invariance, rotations, hierarchy-aware topology, and learned contrastive
objectives should be benchmarked before calling it a foundation encoder.

That distinction matters: this tutorial demonstrates a robust standard and a complete
proof of principle, not an unsupported claim that one component family spans every
quantum device.
"""
    ),
]

OUTPUT.write_text(nbf.writes(notebook))
print(f"Wrote {OUTPUT}")
