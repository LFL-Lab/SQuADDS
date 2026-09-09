# Layout Embeddings v1 release

`universal-geometry-v1` is additive. It does not replace or rewrite
`static-embedding-v0`.

The reviewed release is staged in
[SQuADDS_Layout_Embeddings PR #2](https://huggingface.co/datasets/SQuADDS/SQuADDS_Layout_Embeddings/discussions/2).
It combines the expanded v0 catalogue with the first v1 configuration so the
dataset card and both versioned tables can be reviewed as one atomic release.

## Hugging Face file layout

The prepared release belongs in the existing
`SQuADDS/SQuADDS_Layout_Embeddings` dataset:

```text
metadata/
  static-embedding-v0.parquet
  static-embedding-v0.schema.json
  universal-geometry-v1.parquet
models/
  universal-geometry-v1/
    schema.json
    control-map.parquet
    release-manifest.json
README.md
```

The first v1 table contains only `GeneralizedCapNInterdigital`. Its stable
`layout_id` values join directly to both `SQuADDS_Layouts` and `SQuADDS_DB`.
Simulation targets are intentionally excluded from the embedding inputs.

The accepted v1.1 schema has 1,024 dimensions:

- 32 centered physical and morphological metrics;
- 768 variance-selected full-spectrum shape coefficients at 96 by 96 resolution;
- 224 centered, named parameter-control channels.

The rejected 512-dimensional v1.0 candidate is not a release. It lost finger
detail through an 8 by 8 low-pass DCT crop and collapsed cosine similarities
through constant availability and uncentered-control offsets.

## Acceptance benchmark

Run the paired benchmark before publishing any regenerated v1 table:

```bash
uv run python scripts/benchmark_universal_embeddings.py \
  /path/to/static-embedding-v0.parquet \
  /path/to/universal-geometry-v1.parquet \
  /path/to/coupler-GeneralizedCapNInterdigital-cap_matrix.json
```

V1 must pass every gate: topology retrieval must match or beat v0, full
parameter distance must improve by at least 10%, normalized shape distance may
regress by no more than 10%, held-out mutual-capacitance locality must improve,
and random-pair cosine similarity must retain at least 60% of v0's standard
deviation. Simulation targets are evaluated only after the geometry-only model
is frozen.

## Review and publish

Generate into a staging directory, verify the files, and open a dataset pull
request:

```bash
uv run python scripts/build_universal_embeddings.py \
  /path/to/SQuADDS_Layouts/metadata/manifest.parquet \
  /path/to/SQuADDS_Layout_Embeddings \
  --design-json GeneralizedCapNInterdigital=/path/to/coupler-GeneralizedCapNInterdigital-cap_matrix.json \
  --gds-source raw=/path/to/SQuADDS_Layouts

hf upload SQuADDS/SQuADDS_Layout_Embeddings \
  /path/to/SQuADDS_Layout_Embeddings \
  . \
  --type dataset \
  --include "README.md" \
  --include "metadata/universal-geometry-v1.parquet" \
  --include "models/universal-geometry-v1/*" \
  --commit-message "Add GeneralizedCapNInterdigital universal-geometry-v1 embeddings" \
  --create-pr
```

## Merge order

Merge the Hugging Face dataset pull request before the SQuADDS code pull
request. Then smoke-test `LayoutEmbeddingClient(version="v1")` against the
dataset's `main` revision before merging the code. This order prevents a window
where the released API advertises v1 but the default dataset revision does not
yet contain its parquet table and model metadata.

The SQuADDS API defaults to `v0`; users opt into v1 explicitly:

```python
from squadds.layouts import LayoutEmbeddingClient

v0 = LayoutEmbeddingClient(version="v0")
v1 = LayoutEmbeddingClient(version="v1")
```

The database bridge exposes the same choice:

```python
embedding = db.get_layout_embedding(row, embedding_version="v1")
```
