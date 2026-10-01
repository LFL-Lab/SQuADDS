# Claims and evidence gate

This map follows the two recordings. It is not a commitment to an arbitrary
number of figures. The final journal display-item count will be handled by
combining panels only after the results exist.

No headline claim enters the abstract, title, or conclusion until its stated
test passes on held-out data. Planned outcomes below are hypotheses, not
results.

## Scope fixed for the present paper

- Validate the embedding on **planar chips first**.
- Design the representation interface so a future GDS-layer-to-material-stack
  JSON/XML sidecar can add plane, material, and thickness information without
  redefining the planar geometry core.
- Support **arbitrary multi-terminal layouts**, conductor connectivity, and the
  corresponding capacitance-matrix structure. The scientific scope is not
  restricted to an ordered two-terminal scalar query.
- Use v2 and later embeddings. Include v0/v1 only if they can be reproduced at
  negligible cost; they are not required baselines.
- Use controlled simple geometries alongside realistic Qiskit Metal and
  SQuADDS-derived layouts.

## Must-have evidence: representation

| Claim under test | Required evidence | Controls / split | Gate |
| --- | --- | --- | --- |
| A static GDS-derived representation can encode heterogeneous planar components without catalogue fitting | Generate one valid GDS instance for every feasible Qiskit Metal component and plot every vector; report encoder failures, invalid components, exact duplicates, and non-identical-vector collisions explicitly | Embed each file independently with a frozen encoder; audit geometry and semantic validity before interpreting uniqueness | Pending |
| The representation detects meaningful changes at small and large scales | Narrow and broad sweeps of gap, width, overlap/length, spacing, finger count, ground clearance, and uniform scale | Paired one-factor controls plus selected multivariate sweeps; report vector shift by block and in aggregate | Pending |
| The response is smooth and learnable rather than a binning artefact | Parameter trajectories, local finite differences, discontinuity checks, rank/monotonic trend summaries where physics warrants monotonicity | Repeat at multiple step sizes and near feature-bin boundaries | Pending |
| Nuisance pose does not dominate the representation | Translation, in-plane rotation, reflection, and irrelevant outer-domain/ground-crop controls | Paired base/transformed GDS; relative drift for each v2+ block | Pending |
| Each proposed feature block earns its place | v2, later versions, and cumulative/drop-one-block ablations for shape, scale, topology/ground, proximity, and capacitance-proxy blocks | Identical generated layouts and normalization for every ablation | Pending |
| Component families and within-family sweeps have organized structure | Atlas of all feasible Qiskit Metal singleton components; dense `GeneralizedCapNInterdigital` sweeps; the newly supplied sweep families; any relevant SQuADDS base components | Label family, generator parameters, and sweep coverage; show both global map and within-family trajectories | Pending |
| Ground and galvanic topology are represented correctly | Ground/moat controls, conductor joins/splits, terminal relabeling, and missing/invalid-terminal cases | Check conductor graph and required capacitance-matrix dimension before comparing vectors | Pending |
| Arbitrary terminal count is handled without silently reducing the problem to two terminals | Valid planar examples with increasing terminal/conductor count; permutation-equivariance tests; full matrix target bookkeeping | Compare relabeled copies and joined layouts; report all independent matrix entries, not only one mutual term | Pending |
| Representation differences and composition have physical meaning | Controlled vector-difference tests; compose port-annotated modules and compare with direct embedding of the physically joined GDS | Preserve relative placement, scale, ports, ground, and galvanic merges; compare both representation and inferred conductor graph | Pending |

The first notebook is the acceptance harness for this entire block. Passing
predictive accuracy cannot compensate for a failed physical or topological
contract.

## Must-have evidence: prediction and transfer

| Claim under test | Required evidence | Controls / split | Gate |
| --- | --- | --- | --- |
| Embeddings are more useful than family-specific tables | Same simulations, model class/capacity, preprocessing budget, and splits for embedding versus native CSV/parameter dictionaries | In-family and held-out-family evaluation; explain that schemas differ rather than manufacturing a misleading shared table | Pending |
| Embedding proximity carries capacitance information | Class-level similarity map, design-pair examples from high- and low-similarity family pairs, and distance-versus-normalized-capacitance-difference analysis | Family-balanced weighting; bootstrap intervals; include counterexamples and avoid selecting examples on the test labels | Pending |
| Proximity predicts transfer usefulness | Source specialist adapted to a fixed target test set with 1, 5, 10, 50, and 100 target simulations for both nearby and distant family pairs | Nested target subsets, repeated seeds, target-only and no-transfer baselines | Pending |
| A generalist benefits from controlled diversity rather than merely more rows | Add within-family samples, nearby families, distant families, and combinations in a controlled sequence | Fixed validation sets including a wholly unseen family; match total sample count where required | Pending |
| The generalist does not collapse to a global mean | Leave-one-cluster-out, within-cluster interpolation, nearby extrapolation, cluster-edge, and heavy-tail tests | Per-family residual plots and calibration, not aggregate error alone | Pending |
| A new composite system can be recognized and adapted rapidly | Join previously modeled modules, infer the changed conductor count/matrix shape, make a zero-/one-shot prediction, then fine-tune on a small composite sweep | Direct composite GDS as ground truth representation; fixed test set; target-only baseline | Pending |

Primary predictive endpoints are full capacitance-matrix entry errors (with
matrix-aware aggregation), per-entry RMSE/MAE in physical units, per-target
$R^2$ where defined, calibration/uncertainty if used, and target simulations
needed to cross predeclared engineering tolerances. Thresholds must be frozen
before the final runs.

## Optional or supplementary evidence

- Fine-tuning data required as a continuous function of embedding-neighborhood
  distance, if the relationship is stable enough to interpret.
- Visually unusual or counterintuitive layout pairs that illuminate aggregate
  metrics.
- Additional Qiskit Metal instantiations beyond the one-per-component atlas.
- Full pairwise block-correlation, hyperparameter, seed, and projection-method
  robustness plots.
- v0/v1/v1* comparisons if they are already reproducible; otherwise omit them.
- Cross-domain discussion for semiconductor and photonic workflows, clearly
  framed as an implication rather than demonstrated evidence.

## Claims that should wait

- **Flip-chip or layer-stack generalization.** The algorithm should expose a
  future stack-aware interface, but this paper's first validated embedding is
  planar. Do not show or claim multi-plane performance without crossed
  geometry--stack datasets and a versioned material/thickness map.
- **A foundation model.** Controlled generalist scaling may motivate this term,
  but it should not be used as a headline without broad task, family, and
  distributional evidence.
- **Universal or unlimited modular composition.** Demonstrate finite planar
  multi-terminal compositions first. Do not infer unbounded composition from
  one vector-addition example.
- **Transfer to semiconductor or photonic devices.** This is future scope until
  tested on those data.
- **Simulation replacement.** A low-fidelity capacitance proxy is a feature or
  mean function; it is not high-fidelity truth.

## Immediate acquisition dependencies

1. Inventory which Qiskit Metal components can be instantiated and exported
   under the planar GDS/terminal standard; log exclusions rather than hiding
   them.
2. Locate and schema-audit the new supplied sweep families and the
   `GeneralizedCapNInterdigital` sweeps.
3. Freeze the planar GDS semantic contract for ground, conductors, ordered or
   named terminals, and galvanic connectivity.
4. Freeze v2+ block definitions and acceptance thresholds before comparison.
5. Use the representation results to specify any additional multivariate,
   cross-family, and composite-system simulations required for the paper.
