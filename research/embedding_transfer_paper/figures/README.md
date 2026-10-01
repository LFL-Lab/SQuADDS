# Transcript-driven figure and analysis map

This is an evidence map, not a six-figure quota. Each item below corresponds to
an argument explicitly requested in the recordings or the subsequent scope
decisions. Once results are stable, related items may be consolidated into the
display-item limit of the selected journal. Do not drop an acceptance test just
to meet a figure count; move detailed panels to Supplementary Information.

## Must-have main-story candidates

### P1. Why a shared layout representation is needed

- Simulation-cost/data-fragmentation schematic.
- Incompatible parameter dictionaries for geometrically/physically related
  component families.
- Quantitative tabular-baseline failure belongs here only after the controlled
  comparison exists; until then it is a labeled placeholder.

**Claim supported:** GDS plus explicit semantics offers a common interface where
family-specific tables do not.

### P2. Static planar embedding and representation contract

- GDS processing from polygons/layers to conductors, ground, terminals,
  connectivity, geometry blocks, and capacitance proxy.
- Multi-terminal representation/output contract and an example showing how a
  galvanic join changes conductor count and capacitance-matrix dimension.
- Pose nuisance versus physical-change table.
- Future stack/material sidecar shown only as an extension point, not as a
  validated flip-chip result.

**Claim supported:** the method has explicit physical semantics and is not a
bitmap or family-parameter encoding.

### P3. Embedding figures of merit and v2+ block ablation

- Narrow and broad trajectories for small- and large-scale parameter changes.
- Translation/rotation/reflection/domain-crop drift.
- Smoothness/discontinuity diagnostics.
- Cumulative and drop-one-block v2+ ablations, showing which blocks create
  desired sensitivity or invariance.
- Include ground, topology, terminal relabeling, and scale controls.

**Claim supported:** detectable, smooth physical signal exists at relevant
scales while nuisance changes remain controlled.

### P4. Component atlas and structured sampling

- One GDS instance for every feasible Qiskit Metal component, with every point
  labeled or interactively traceable and all failures/exclusions reported.
- Uniqueness/collision audit of the design vectors. Do not equate a visually
  separated 2D projection with uniqueness in the full vector.
- Global family structure across Qiskit Metal and relevant SQuADDS components.
- `GeneralizedCapNInterdigital` as the worked within-family trajectory.
- The newly supplied sweep families as additional family/coverage tests.

**Claim supported:** the frozen representation spans heterogeneous components
and organizes controlled in-family variation.

### P5. Representation arithmetic and planar modularity

- Vector differences along controlled changes and whether the changed blocks
  are interpretable/repeatable.
- Port-aware composition diagram.
- Embedding composed from modules versus embedding of the directly generated
  joined GDS.
- Two two-conductor modules joined into a three-conductor example, followed by
  at least one higher-terminal-count stress test.

**Claim supported:** composition respects placement, ground, ports, and
galvanic topology. A simple raw vector sum is not assumed in advance.

### P6. Embedding versus native tabular predictors

- Identical-label, identical-split comparison on each swept family.
- Accuracy and residuals for embedding and CSV/parameter baselines.
- Held-out-family panel demonstrating that incompatible native schemas do not
  provide a direct shared transfer interface.

**Claim supported:** the shared representation improves prediction and enables
comparisons/transfer that native dictionaries cannot express naturally.

### P7. Does embedding distance carry physical meaning?

- Family-by-family average similarity matrix with a predeclared weighting rule.
- Layout/capacitance examples from the closest and most distant family pairs;
  within each, show high- and lower-similarity design pairs.
- Embedding distance or cosine similarity versus normalized capacitance
  difference, with uncertainty and explicit counterexamples.

**Claim supported:** representation proximity contains a transfer-relevant
capacitance signal; similarity alone is not treated as proof.

### P8. Transfer learning for nearby and distant families

- Fixed-target-test learning curves at 1, 5, 10, 50, and 100 target
  simulations.
- Source-pretrained/fine-tuned, target-only, no-transfer, and relevant
  generalist baselines.
- Repeat for the high-similarity and low-similarity source--target pairs.
- Report the target-label count needed to reach frozen error tolerances.

**Claim supported:** transfer reduces new simulations, and embedding proximity
predicts when it helps.

### P9. Controlled path toward a generalist model

- Add samples within one family, then nearby families, distant families, and
  mixtures while controlling sample count.
- Evaluate every stage on the same validation sets, including a wholly unseen
  family.
- Leave-one-cluster-out, interpolation/extrapolation, cluster-edge, and
  heavy-tail panels to expose global-mean collapse.

**Claim supported:** coverage/diversity, not only row count, governs generalist
performance.

### P10. Final unseen composite-system demonstration

- Previously modeled modules connected into a planar multi-terminal device not
  present in training.
- Automatically inferred conductor topology and matrix dimension.
- Zero-/one-shot matrix prediction, expected failure analysis if applicable,
  and rapid adaptation on a small new sweep versus target-only learning.

**Claim supported:** the complete workflow moves from reusable modules to an
unseen system and learns it with fewer new simulations.

## Optional / supplementary analyses

- Required fine-tuning labels versus neighborhood distance.
- Additional unusual layout-pair case studies.
- Projection-method stability (PCA/UMAP/etc.); full-space metrics remain
  primary.
- Complete feature-block correlation matrices and per-coordinate trajectories.
- Additional terminal-count/composition stress tests.
- v0/v1/v1* history, only if already inexpensive and reproducible.
- Hyperparameter, seed, split, weighting, and confidence-interval diagnostics.

## Deferred figures

- Planar-versus-flip-chip embedding maps.
- Same-XY/different-plane separation tests.
- Material/thickness/layer-stack held-out generalization.
- Semiconductor or photonic demonstrations.

These should be drafted only after a stack sidecar convention and crossed
geometry--stack datasets exist. The planar algorithm should be architected so
these features can be appended without changing the planar core.

## Export convention

Use stable semantic names while the editorial grouping is unsettled, for
example `representation_contract`, `v2plus_block_ablation`,
`qmetal_component_atlas`, `similarity_capacitance`, `transfer_learning_curves`,
and `unseen_composite`. Retain vector PDF/SVG plus 300--600 dpi raster previews.
Assign final `fig01_...` numbering only after main-versus-supplementary panels
are selected.
