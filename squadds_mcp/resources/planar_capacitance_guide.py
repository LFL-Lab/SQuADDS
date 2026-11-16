"""
Planar capacitance guide resources.
===================================

- ``squadds://planar-capacitance-guide``: models (QCap, QCap-Fast), accuracy (median % with MAE and RMSE in fF), time,
  conventions, the MCP workflows (predict -> Hamiltonian -> SQuADDS, design, retrain, contribute) and the
  access note.
- ``squadds://planar-capacitance-tutorials``: Tutorials 11-16 and which tools reproduce each.

The numbers are those of ``docs/source/ml_models.rst`` (QCap has no held-out score of its own: its in-sample fit and
one never-seen check, and the held-out numbers of the paper's versions of the models, labelled as such);
``tests/test_mcp_capacitance_workflows.py`` checks that every number quoted here appears there.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

DOCS_BASE = "https://lfl-lab.github.io/SQuADDS/source/tutorials/"

TUTORIALS = [
    {
        "number": 11,
        "title": "Predicting Capacitance Matrices from GDS",
        "file": "Tutorial-11_Predicting_Capacitance_Matrices_from_GDS",
        "covers": "a multipad star and a two-claw Xmon to the full Maxwell matrix; physical checks and timing; "
        "QCap vs QCap-Fast; physics priors vs the model; batch prediction and threads; matrix -> E_C; a prediction as a "
        "SQuADDS_DB row; the squadds-ml CLI; the geometric embedding; ground and conductors on one layer (detected "
        "ground, ground_pieces)",
        "tools": [
            "generate_component_gds",
            "inspect_capacitance_layout",
            "predict_capacitance_from_gds",
            "compute_capacitance_prior",
            "capacitance_to_hamiltonian",
        ],
    },
    {
        "number": 12,
        "title": "Planar Capacitance Families in SQuADDS",
        "file": "Tutorial-12_Planar_Capacitance_Families_in_SQuADDS",
        "covers": "the SQuADDS_PlanarCapacitance dataset: families, sizes, splits; raw tables (simulations and QC flags, "
        "convergence, solved GDS); processed tables (embedding, physics prior and extrapolated (optimized) prior, model input); nearest "
        "designs; the same data in SQuADDS_DB (a concentric capacitor with a capacitance target, an Xmon with a target "
        "frequency); public vs held-out labels, licence and contributors",
        "tools": [
            "query_planar_capacitance",
            "get_planar_capacitance_design",
            "nearest_planar_capacitance_designs",
            "get_capacitance_component_info",
            "enable_squadds_db_source",
            "find_closest_designs",
            "get_capacitance_data",
        ],
    },
    {
        "number": 13,
        "title": "From Layout to Hamiltonian with ML",
        "file": "Tutorial-13_From_Layout_to_Hamiltonian_with_ML",
        "covers": "find_closest for a TransmonCross + cavity; render, predict with MLSimulator, compare with Q3D; "
        "capacitance -> E_C, f_q, alpha, g; an interpolated design nobody simulated; Xmon, TransmonPocket and "
        "GeneralizedCapStar targets from the planar capacitance families; QCap-Fast",
        "tools": [
            "find_closest_designs",
            "predict_closest_designs_ml",
            "interpolate_design",
            "predict_capacitance_from_design",
            "capacitance_to_hamiltonian",
        ],
    },
    {
        "number": 14,
        "title": "Training on Your Own Designs",
        "file": "Tutorial-14_Training_on_Your_Own_Designs",
        "covers": "Q3D labels and label QC; building training shards; zero-shot or from scratch (fit_in_family); "
        "retraining the collective model; retraining on your own cluster (SLURM)",
        "tools": [
            "check_capacitance_label",
            "build_capacitance_training_shards",
            "plan_capacitance_training",
            "run_capacitance_fit_smoke",
            "score_capacitance_model",
        ],
    },
    {
        "number": 15,
        "title": "Contributing Capacitance Data and Models",
        "file": "Tutorial-15_Contributing_Capacitance_Data_and_Models",
        "covers": "reading the Q3D export; filling the contribution template; validating; catching broken labels "
        "(flipped sign, pF / fF mix-up); what the pull request adds; validating and contributing a model; why test "
        "scores are maintainer-run, once",
        "tools": [
            "init_capacitance_contribution",
            "validate_capacitance_contribution",
            "contribute_capacitance_data",
            "validate_capacitance_model",
            "contribute_capacitance_model",
        ],
    },
    {
        "number": 16,
        "title": "Inverse Design by ML Screening",
        "file": "Tutorial-16_Inverse_Design_by_ML_Screening",
        "covers": "sizing a search; generating a pool of random Xmon layouts; predicting every matrix; picking the best "
        "candidate per target E_C; a TransmonPocket with three couplers; comparing the picks with Q3D",
        "tools": [
            "generate_component_gds",
            "inspect_capacitance_layout",
            "predict_capacitance_from_design",
            "predict_capacitance_from_gds",
            "capacitance_to_hamiltonian",
        ],
    },
]

GUIDE = """# Planar capacitance models: guide for agents

> **Access.** The models download from `SQuADDS/QCap` and the dataset is
> `SQuADDS/SQuADDS_PlanarCapacitance` on Hugging Face. Local copies work too (`SQUADDS_CAPML_MODELS`,
> `SQUADDS_PLANAR_CAP_DATA`). The test labels stay in the private `SQuADDS/SQuADDS_PlanarCapacitance-heldout`. Opening a
> contribution pull request needs a Hugging Face token. Running a model needs the optional
> extra: `pip install "squadds[ml]"` (torch, gdstk, numba, threadpoolctl, shapely, pyyaml); without it the tools return an
> install hint.

## What the models do
GDS layout -> the **full N x N Maxwell capacitance matrix** in fF over every conductor, ground included, for any N.
A thin-film method-of-moments physics prior is corrected by graph networks; every result carries physical checks
(symmetric, diagonal positive, off-diagonal negative, closure, positive-definite signal block) and its timing with machine
and thread count.

## Conventions (always)
- Units fF. Ground net `G` is row/column 0; conductors follow in GDS file order (`C1..Cn`), or SQuADDS net names
  (`cross`, `claw`, `pad_top`, `north`, ...) when predicted from a component.
- Diagonal > 0 (self capacitance), off-diagonal < 0 (= -mutual). `<a>_to_<b>` keys are positive magnitudes.
- **Never merge nets**: the conductor count is read from the layout; predict and report the full matrix.
- Layers: ground 1/0, conductors 1/10 (configurable). Training stack: Si, eps_r 11.45, 350 um, zero-thickness metal.
  Other stacks (`eps_r`, `thickness_um`): `QCap` solves the physics prior at that stack and the unchanged
  networks correct it (calibrated route: 0.21-1.02 % (MAE 0.07-0.33, RMSE 0.09-0.45 fF) on 15 held-out designs at every
  tested stack vs 0.23 % (0.080 / 0.111 fF) on the training stack; self terms up to 1.22 % (0.79 / 0.86 fF); a bounded
  error, not training-stack accuracy). `QCap-Fast` cannot solve a prior: it rescales by (1 + eps_r) / (1 + 11.45) and warns (measured off the stack:
  6.36 % (MAE 1.74, RMSE 2.39 fF), 13.1 % (3.43 / 4.67 fF) across thickness; use `QCap` there).
  Tested: Si at 10-1000 um, or eps_r 3.8-24 at 350 um; other combinations warn.
- Ground and conductors on **one layer**: pass that layer as both `ground_layers` and `conductor_layers`. The metal is
  split into numbered pieces and the ground is detected (`ground_detection`: pieces, reason, confidence, plus a warning).
  `inspect_capacitance_layout` shows the pieces (and a numbered PNG) before predicting; if the ground is wrong, pass
  `ground_pieces` (e.g. `"1"` or `"1,4"`) to the prediction or prior.
- Every accuracy number: median % **with** MAE and RMSE in fF, **and** the wall time with machine and threads.

## Models
| model | what it is | use it for |
|---|---|---|
| `QCap` (**default**) | optimized physics prior (two solves, Richardson-extrapolated) + 3-seed residual graph-network ensemble, trained on all 45,771 on-stack Q3D layouts | design work |
| `QCap-Fast` | improved release of the paper's fast model (same architecture; surrogate trained on more generated layouts, networks retrained on all 45,771 labelled layouts): no field solve, a learned surrogate predicts the prior | quick screening of trained families only |

**QCap-Fast warning.** On new designs of a trained family `QCap-Fast` was worse than the physics prior alone: 3.175 % (MAE 0.583 fF /
RMSE 0.824 fF) against 1.765 % (MAE 0.275 fF / RMSE 0.376 fF); the paper's version of it, on a held-out family, 3.11 % (MAE
0.533 fF / RMSE 0.633 fF) against 1.55 % (MAE 0.166 fF / RMSE 0.205 fF). Use `QCap` for anything that matters.

**Physical checks.** When a check fails (symmetric, diagonal positive, off-diagonal negative, closure, positive-definite
signal block), `checks.passed` is false and the first warning reads "physical check failed ... This matrix is not
physical: do not use it". Do not use that matrix; a model that solves the physics (`QCap`) is the safer
choice, especially outside the trained component families.

## Accuracy (QCap and QCap-Fast have no held-out score of their own)
Per layout the median relative error over mutual entries against Q3D (0.1 fF floor), averaged over layouts. "Paper's version"
rows belong to the paper's version of each model (networks trained on the 35,354 TRAIN layouts only), not to the released models; quote them as such.

| set (layouts) | QCap: median % (MAE / RMSE) | QCap-Fast: median % (MAE / RMSE) |
|---|---|---|
| this model, in-sample (45,771; trained on them) | 0.207 % (0.0679 fF / 0.077 fF) | 0.671 % (0.353 fF / 0.395 fF) |
| this model, B5 never seen (39) | 1.05 % (0.132 fF / 0.188 fF) | 3.175 % (0.583 fF / 0.824 fF) |
| paper's version, validation (4,608) | 0.229 % (0.0674 fF / 0.0792 fF) | 0.846 % (0.491 fF / 0.549 fF) |
| paper's version, test, in family (4,009) | 0.177 % (0.0203 fF / 0.025 fF) | 0.63 % (0.113 fF / 0.1414 fF) |
| paper's version, test, zero-shot macro (1,878) | 0.587 % (1.092 fF / 1.152 fF) | 3.576 % (4.608 fF / 5.058 fF) |
| paper's version, held-out family (4,071) | 0.408 % (0.049 fF / 0.059 fF) | 3.113 % (0.533 fF / 0.633 fF) |

## Time per layout
- QCap: 0.845 / 5.96 s (median / p90) in family on an x86 cluster node with 2 CPUs. This package: 1.88 s for N <= 4
  and 23.0 s for 10 <= N <= 20 on a GH200 node with 2 threads; 3.76 s and 46.2 s on an x86 Xeon Silver 4116 with 2 threads.
- QCap-Fast: 0.091 / 0.274 s in family (x86, 2 CPUs); this package 0.093 s for N <= 4 on a GH200 node, 0.17 s on the Xeon
  Silver 4116 (2 threads).
- First call of a session: about 40 s (weights download + sha256 check + solver compile); repeats hit a cache.

## Workflows (tools in order)
### A. GDS -> matrix -> Hamiltonian -> SQuADDS
0. `inspect_capacitance_layout(gds_path)`: the conductors and ground the model will read (count, areas, ground area in
   the file vs read; pieces and the detected ground for a one-layer file; optional PNG).
1. `predict_capacitance_from_gds(gds_path)` (or `predict_capacitance_from_design(component_class, design_options)` for
   SQuADDS net names; both take `eps_r` / `thickness_um`). Optional baseline: `compute_capacitance_prior(gds_path)`.
2. `capacitance_to_hamiltonian(matrix_fF, names, qubit_type="xmon" | "transmon_pocket", ...)` or explicit
   `qubit_nodes` / `coupler`; give `EJ_GHz` or the target `qubit_frequency_GHz` + `anharmonicity_MHz`, and
   `resonator_frequency_GHz` for g. Returns C_sigma, E_C, E_J/E_C, L_j, f_q, alpha, g.
3. Compare with the database (the `planar_capacitance` source is on by default): `find_closest_designs`
   (`qubit="Xmon"` / `"TransmonPocket"`, or `system_type="coupler"` with capacitance targets in fF).
   `predict_closest_designs_ml` predicts those rows with the models next to their stored simulation (the
   `Analyzer.predict_capacitance` route); `nearest_planar_capacitance_designs` / `get_planar_capacitance_design` show the
   geometrically closest dataset designs and one design's options, Q3D QC and convergence, label and stored priors.

### B. Design a component of a new family to targets
`get_capacitance_component_info(component_class)` (default options) -> `find_closest_designs` for a starting point ->
`predict_capacitance_from_design` -> adjust options -> repeat -> `generate_component_gds` for the final GDS. Prompt:
`design_planar_capacitor_to_target`.

### C. Retrain on your own data and compute
`check_capacitance_label` (Q3D `cap_matrix.csv`) -> `build_capacitance_training_shards` (<= 16 designs here) or
`plan_capacitance_training(mode="build")` -> `run_capacitance_fit_smoke` (<= 500 steps, pipeline check) ->
`plan_capacitance_training(mode="fit" | "in_family" | "collective")` writes the driver and a generic SLURM script (run
it yourself) -> `score_capacitance_model(model=<run dir>, data=<shards>, split="val")`. The released recipe is 18,000
steps per seed (about 240 s per seed on a GH200 CPU with 2 threads). `in_family` keeps the zero-shot model unless a
from-scratch fit is materially better (zero-shot wins ties). Test scoring is maintainer-only.

### D. Contribute data or models (dry run by default)
`init_capacitance_contribution(dest)` -> fill it -> `validate_capacitance_contribution(path)` ->
`contribute_capacitance_data(path)` (dry run) -> show the user -> `contribute_capacitance_data(path, dry_run=False,
confirm=True)`. Models: `validate_capacitance_model(repo_id)` -> `contribute_capacitance_model(repo_id)` (dry run) ->
`dry_run=False, confirm=True`. Never self-report a test score.

### E. Two coupled transmons with half-wave readout (the two-transmon example)
`solve_coupled_transmon_hamiltonian(targets)` (full Hamiltonian, no RWA: E_J, E_C, L_J, g, J, target Maxwell matrix) ->
`generate_two_transmon_gds(p, variant, out_path, predict=True)` -> `solve_coupled_transmon_hamiltonian(targets,
reference_matrix_fF=M, layout_matrix_fF=M)` -> iterate (QCap to confirm, QCap-Fast only for step directions) ->
`find_closest_designs(system_type="qubit_cavity", qubit="TransmonCross", resonator_type="half", ...)` for the half-wave
resonators -> `correct_half_wave_resonator` (length and kappa for your claw). The layout is not a trained family. Tutorial
16 checked its two final layouts against Q3D: the coupling entries agreed to 0.573 % (MAE 0.128 / RMSE 0.209 fF) gap and
0.337 % (MAE 0.101 / RMSE 0.171 fF) interdigital (Q3D about 31 min on 16 cores of an Intel Xeon Silver 4116 node; the
prediction about 3.4 s with 2 threads on an Intel Xeon E5-2640 v4). Other layouts are unvalidated: run a field solve
before fabrication. Prompt: `design_coupled_transmons`.

## Reference
- Dataset: `query_planar_capacitance` (filters family / N / split / collection / config; `include_labels` for Q3D
  matrices; test labels are held out); `get_planar_capacitance_design` (one design in detail) and
  `nearest_planar_capacitance_designs`. `get_planar_capacitance_conventions` for the machine-readable conventions.
- Tutorials 11-16: `squadds://planar-capacitance-tutorials`. Docs: `docs/source/ml_models.rst` ("Planar capacitance
  models").
"""


def tutorials_markdown() -> str:
    """The tutorials index as markdown."""
    lines = [
        "# Planar capacitance tutorials (SQuADDS Tutorials 11-16)",
        "",
        "Notebooks in `tutorials/` of the SQuADDS repository; rendered docs links below. Each lists the MCP tools that "
        "reproduce its steps.",
        "",
    ]
    for t in TUTORIALS:
        lines += [
            f"## Tutorial {t['number']}: {t['title']}",
            f"- Notebook: `tutorials/{t['file']}.ipynb`",
            f"- Docs: {DOCS_BASE}{t['file']}.html",
            f"- Covers: {t['covers']}",
            f"- MCP tools: {', '.join(f'`{x}`' for x in t['tools'])}",
            "",
        ]
    return "\n".join(lines)


def register_planar_capacitance_resources(mcp: FastMCP) -> None:
    """Register the planar capacitance guide and tutorials-index resources."""

    @mcp.resource("squadds://planar-capacitance-guide")
    async def planar_capacitance_guide() -> str:
        """Guide to the planar capacitance models (QCap, QCap-Fast): accuracy (median % with MAE and RMSE in fF), time, conventions,
        the predict -> Hamiltonian -> SQuADDS, design, retrain and contribute workflows, and access."""
        return GUIDE

    @mcp.resource("squadds://planar-capacitance-tutorials")
    async def planar_capacitance_tutorials() -> str:
        """Index of SQuADDS Tutorials 11-16 (planar capacitance) with the MCP tools that reproduce each."""
        return tutorials_markdown()
