# Executable plan: representation-first embedding acceptance

## Decision rule

The notebook is an acceptance harness, not a showcase assembled after seeing
the results. Thresholds are declared before scoring. A candidate may be useful
for prediction while still failing the representation contract; failures remain
visible and motivate an additive successor rather than silently changing v0--v3.

## Phase 1 — Standardized planar GDS benchmark

1. Use the existing `GeneralizedCapNInterdigital` sweep and every new swept
   family in `gds_json_exports`, preserving stable source/design identities.
2. Discover the installed Qiskit Metal component catalogue and attempt one
   default or minimally valid instance of every concrete planar component.
   Export every successful instance to the same standardized GDS contract and
   record every exclusion with an explicit reason. The embedding map must show
   that distinct valid layouts produce distinct vectors; collisions are
   inspected geometrically rather than hidden by jitter.
3. Generate paired pose controls (translation, 90-degree and arbitrary
   rotation, reflection) without changing the underlying device.
4. Generate physical perturbation paths (uniform scale, terminal gap, facing
   overlap, conductor width, ground clearance and outer-domain crop).
5. Include controlled two-, three-, four-, and higher-terminal layouts. A
   fixed-size design representation must be permutation invariant, while a
   terminal-pair query representation must be equivariant to relabelling and
   retain the influence of all unqueried conductors.
6. Validate ports, ground holes, conductor attachment, deterministic file
   identity and manifest completeness.

Flip-chip data are deliberately deferred. The planar schema will nonetheless
reserve a versioned layer-stack input contract: a future JSON/XML sidecar maps
GDS layer/datatype pairs to material, thickness and physical plane. No specific
layer numbers (for example `60/0`) are frozen until that contract is designed
against real stack files.

## Phase 2 — Predeclared figure of merit suite

For v2, v2 with its published conditioned-scale view, v3 fixed-stack core, v3
complete and each new candidate, compute:

- deterministic repeatability;
- translation/rotation/reflection and outer-crop drift;
- uniform-scale factorization;
- local physical sensitivity and monotonicity;
- trajectory continuity and bin-edge stability;
- terminal relabel equivariance;
- ground/topology sensitivity;
- arbitrary-terminal resolvability and relabelling equivariance;
- nearest-neighbour stability and family mixing;
- correlation of distance with normalized capacitance difference;
- held-out-family R2, RMSE and median absolute error as a downstream audit.

Primary thresholds:

| Requirement | Pass gate |
| --- | --- |
| Determinism | bitwise equality on repeated encoding |
| Translation | relative drift <= `1e-7` |
| Right-angle rotation/reflection | relative drift <= `5e-4` |
| Arbitrary rotation | relative drift <= `3e-3` |
| Outer ground crop | relative drift <= `1e-5` |
| Physical sensitivity | median physical drift >= `10 ×` median nuisance drift |
| Smooth sweep | no single step > `5 ×` median adjacent step |
| Arbitrary terminals | no terminal-count ceiling; all pair queries retain the other-conductor context |
| Relabelling | global view invariant and pair-query view equivariant under a common port permutation |
| No catalogue fitting | encoding one layout never consults other rows or labels |

Thresholds describe numerical representation behavior; predictive endpoints are
reported rather than converted into arbitrary pass/fail claims.

## Phase 3 — Diagnose failures by block

The notebook reports full-vector and block-level scores. A failure is assigned
to one of four causes: pose leakage, scale entanglement, topology/role loss, or
loss of other-terminal context. Counterexample GDS files and their
interactive overlays are shown next to the affected feature blocks.

## Phase 4 — Additive improvement

Implement an additive planar `capacitance-operator-v4` without altering v0--v3:

- retain the v3 invariant planar core;
- provide a permutation-invariant global design vector for visualization;
- provide a fixed-size, ordered pair-query vector for every capacitance-matrix
  entry, regardless of terminal count;
- aggregate all unqueried conductors as an explicit electrostatic context;
- append graph/operator spectra whose size does not depend on terminal count;
- make translation, in-plane rotation and reflection nuisance transformations;
- retain absolute scale, local ground and conductor topology as physical signals;
- define an extensible layer-stack hook but do not claim flip-chip validation.

Rerun the same untouched acceptance harness. Iterate only on failures with a
physical explanation; do not optimize directly against a single downstream R2.

## Phase 5 — Paper and acquisition handoff

1. Export the figures required by the recorded paper narrative; only after the
   story is stable should related panels be consolidated to match a target
   journal's display-item limit.
2. Update the separate `embedding-transfer-paper/` package only with claims that
   pass the evidence matrix.
3. Specify Saikat's crossed sweep: topology × in-family geometry × scale ×
   stack, with pose duplicates and one standardized fidelity ladder.
4. Freeze target splits and engineering tolerances before simulations arrive.

## Concrete deliverables

- `Tutorial-24_Embedding_Representation_Acceptance.ipynb` with live Plotly views.
- Reproducible GDS/sidecar generator and manifest under ignored notebook runtime.
- v2/v3/v4 comparison table and machine-readable acceptance report.
- Contract tests for every planar invariance plus explicit rejection of
  unsupported multi-plane stacks until the stack-aware extension is validated.
- Separate Overleaf-ready manuscript directory with claims/evidence gate.
