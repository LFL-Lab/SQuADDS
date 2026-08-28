# Layout Embeddings v2 release

`universal-geometry-v2` is additive. It does not replace or rewrite
`static-embedding-v0` or `universal-geometry-v1`, and the SQuADDS API still
defaults to `v0`.

## What changes relative to v1

v0 and v1 are **fit on write**. Their normalization statistics, and in v1 the
selected spectral frequencies, are derived from whichever rows are written
together. A contributor who runs either builder on their own designs therefore
lands in a different space, even though the vectors share a length.

v2 removes that coupling. Every coordinate is a physical measurement in
micrometers, inverse micrometers, or farads per meter, accumulated onto frozen
bin edges. `squadds.layouts.encode` is a pure function of one GDS file and one
design-option mapping, so an outside contribution can be encoded and compared
without refitting anything.

v2 is also deliberately **not** scale invariant. v0 and v1 crop each layout to
its own functional bounds, so a design and its exact enlargement produce
identical shape blocks. Because conductor separation in micrometers is the
dominant variable for capacitance, v2 measures distances absolutely.

## Hugging Face file layout

The release belongs in the existing `SQuADDS/SQuADDS_Layout_Embeddings`
dataset and adds only new paths:

```text
metadata/
  universal-geometry-v2.parquet
models/
  universal-geometry-v2/
    schema.json
    release-manifest.json
```

The 512 dimensions are:

- 48 named physical metrics: extent, per-role area and perimeter, conductor
  width percentiles, gap integrals, and symmetry;
- 192 coupling-spectrum bins: facing boundary length per terminal pair per
  absolute separation, plus each terminal against the ground plane;
- 128 shape-spectrum coefficients: two-point correlation, terminal
  cross-correlation, conductor width distribution, and contour harmonics;
- 96 parameter statistics: dimension-typed order statistics with
  dimension-scoped signed hashing, so any parameter count or naming convention
  maps to a fixed width;
- 48 physics-proxy coordinates: a two-dimensional boundary-element capacitance
  matrix and dilation topology.

Simulation targets are never embedding inputs.

## Coverage

The first v2 table contains **13,683** `GeneralizedCapNInterdigital` designs.

This is fewer than the 20,062 rows in the v0 and v1 tables for the same family.
The reason is upstream: `SQuADDS/SQuADDS_Layouts` currently publishes 10,000 of
the 16,379 `q3d_cap` GDS artifacts that its own `metadata/manifest.parquet`
lists, so the remaining 6,379 layouts cannot be encoded from the released
geometry. v0 and v1 were generated before that gap appeared. The count in
`release-manifest.json` records it as `layouts_without_downloadable_gds`.

Regenerating the missing artifacts into `SQuADDS_Layouts` and re-running the
builder is the only step required to bring v2 to full coverage; nothing in the
encoder changes.

## Reproduce and publish

The builder performs no catalogue-wide fitting, so two runs over disjoint
contributions produce directly concatenable output:

```bash
uv run --extra gds python scripts/build_v2_embeddings.py \
  /path/to/SQuADDS_Layouts/metadata/manifest.parquet \
  /path/to/SQuADDS_Layouts \
  /path/to/staging \
  --design-json GeneralizedCapNInterdigital=/path/to/coupler-GeneralizedCapNInterdigital-cap_matrix.json \
  --component-name GeneralizedCapNInterdigital
```

Two complete runs produce byte-identical parquet, schema, and release-manifest
files. Verify the checksums in `release-manifest.json` before uploading:

```bash
hf upload SQuADDS/SQuADDS_Layout_Embeddings \
  /path/to/staging . \
  --repo-type dataset \
  --include "metadata/universal-geometry-v2.parquet" \
  --include "models/universal-geometry-v2/*" \
  --commit-message "Add GeneralizedCapNInterdigital universal-geometry-v2 embeddings"
```

## Selecting the standard

The API default remains `v0`. Users opt into v2 explicitly:

```python
from squadds.layouts import LayoutEmbeddingClient

v2 = LayoutEmbeddingClient(version="v2")
record = v2.get("layout:sha256:<layout hash>")
neighbors = v2.nearest(record["layout_id"], limit=10)
```

The database bridge and the MCP tools expose the same choice:

```python
embedding = db.get_layout_embedding(row, embedding_version="v2")
```

Encoding a layout that is not in the catalogue needs no download at all:

```python
from squadds.layouts import encode

vector = encode("my_capacitor.gds", {"digit_pitch": "5.5um", "digit_population": 9})
```

## Evidence

Tutorial 18 compares v0 and v2 under an identical model, split policy, and label
budget over the 13,683 paired designs. Tutorial 19 walks one design through the
encoder and checks the explanation against the shipped implementation.

| held-out macro R2 | 1% labels | 10% labels | 100% labels |
| --- | ---: | ---: | ---: |
| `static-shape-v0` (155 compact features) | 0.235 | 0.888 | 0.984 |
| `universal-geometry-v2` | 0.780 | 0.9915 | 0.9998 |
| `universal-geometry-v2`, geometry block only | 0.754 | 0.9896 | 0.9996 |

The third row is the control that matters. It receives no design parameters at
all and still beats the whole of v0, which includes v0's parameter sum, so the
improvement is geometric rather than an artifact of v2 retaining more parameter
detail.

A single unfitted coordinate, the facing-boundary integral
`log1p_primary_inverse_gap_integral`, reaches Spearman +0.941 against the
simulated mutual capacitance, and the boundary-element proxy reaches +0.920.

## Known limits

- Only one component class is measured. The cross-class study of Tutorial 17 has
  not been repeated with v2.
- Extrapolating to devices larger than any in training, v2 reaches macro R2
  0.810 against v0's 0.769; the advantage narrows sharply outside the
  interpolation regime.
- Raw cosine similarity has a compressed spread. A frozen whitening transform
  should be fitted and published as a separate metric layer before v2 similarity
  is used as a headline number.
- On two-terminal devices roughly 204 of the 512 coordinates vary; the rest are
  reserved for richer terminal topologies and cost nothing statistically.
