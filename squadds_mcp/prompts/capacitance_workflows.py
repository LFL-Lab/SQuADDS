"""
Planar capacitance workflow prompts.
====================================

Step-by-step instructions that chain the planar capacitance tools (``capacitance_ml``, ``capacitance_design``,
``capacitance_training``, ``capacitance_contribution``) with the classic SQuADDS tools. Each prompt names the tools to call
in order; new workflows add a prompt here.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from squadds_mcp.tools.capacitance_design import TWO_TRANSMON_CONVENTIONS, TWO_TRANSMON_VALIDITY

REPORTING_RULES = """## Reporting rules
- Capacitance in fF, Maxwell convention, ground first; report the full matrix (never merge nets).
- Any accuracy figure: median % with MAE and RMSE in fF, plus wall time with machine and thread count.
- Say which model produced a number (default `QCap`; `QCap-Fast` only for screening trained families).
- On a missing `[ml]` extra, give the install command the tool returns."""


def register_capacitance_workflow_prompts(mcp: FastMCP) -> None:
    """Register the planar capacitance workflow prompts."""

    @mcp.prompt()
    def gds_to_hamiltonian(
        gds_path: str = "chip.gds",
        qubit_type: str = "xmon",
        qubit_frequency_GHz: float = 5.0,
        anharmonicity_MHz: float = -200.0,
        resonator_frequency_GHz: float = 7.0,
        resonator_type: str = "quarter",
    ) -> str:
        """Predict the Maxwell capacitance matrix of a GDS and turn it into transmon Hamiltonian parameters."""
        return f"""# From a GDS to Hamiltonian parameters

Layout: `{gds_path}`; qubit kind: `{qubit_type}`; target f_q = {qubit_frequency_GHz} GHz, alpha = {anharmonicity_MHz} MHz;
resonator {resonator_frequency_GHz} GHz ({resonator_type}-wave).

Read `squadds://planar-capacitance-guide` first if you have not.

### Step 0: Look at the metal
`inspect_capacitance_layout(gds_path="{gds_path}")`: conductor count and areas, ground area in the file vs read by the
model. If ground and conductors share one layer, pass that layer as both `ground_layers` and `conductor_layers`: check the
detected `ground_pieces` (and the PNG with `png_path`); if the ground is wrong, use `ground_pieces` from here on.

### Step 1: Predict the matrix
Call `predict_capacitance_from_gds(gds_path="{gds_path}")` (model `QCap`; same layers / `ground_pieces` as Step 0;
`eps_r` / `thickness_um` for another substrate). If `checks.passed` is false, do not use the matrix: say so and stop. Read
`warnings`, note `timing_s`, `threads` and `machine`. Conductors are `G, C1..Cn` in GDS file order. If the layout was
built from a SQuADDS component, prefer `predict_capacitance_from_design(component_class, design_options)`: it names the
nets (`cross`, `claw`, `pad_top`, ...).

### Step 2 (optional): Physics baseline
`compute_capacitance_prior(gds_path="{gds_path}")` gives the physics-only matrix the model corrects; quote both if the user
asks how much the network changes.

### Step 3: Identify the qubit conductors
Show the user the conductor names and the self / mutual capacitances. With `G, C1..Cn` names, ask (or infer from sizes
and mutuals) which conductor is the island (grounded transmon) or which two are the pads (floating transmon), and which
conductor is the readout coupler.

### Step 4: Hamiltonian
Call `capacitance_to_hamiltonian(matrix_fF=..., names=..., qubit_type="{qubit_type}")` (presets: `xmon` /
`transmon_cross` = island `cross` + coupler `claw`; `transmon_pocket` = pads `pad_top`, `pad_bot` + coupler
`readout_pad`), or pass `qubit_nodes` and `coupler` explicitly. Add `qubit_frequency_GHz={qubit_frequency_GHz}`,
`anharmonicity_MHz={anharmonicity_MHz}` (E_J from the target, as the SQuADDS Analyzer does) or an `EJ_GHz` the user gives,
and `resonator_frequency_GHz={resonator_frequency_GHz}`, `resonator_type="{resonator_type}"` for g.

### Step 5: Compare with SQuADDS
With the `planar_capacitance` source (on by default), call `find_closest_designs(system_type="qubit",
qubit="Xmon" or "TransmonPocket", target_params={{"qubit_frequency_GHz": {qubit_frequency_GHz}, "anharmonicity_MHz":
{anharmonicity_MHz}}})` to place the layout among simulated designs.

### Step 6: Report
C_sigma and C_g (fF), E_C (MHz), E_J (GHz) and E_J/E_C, L_j (nH), f_q (GHz), alpha (MHz), g (MHz), the model,
the prediction time with machine and threads, and any warnings.

{REPORTING_RULES}
"""

    @mcp.prompt()
    def design_planar_capacitor_to_target(
        component_class: str = "GeneralizedCapConcentric",
        target_capacitances_fF: str = '{"north_to_south": 25.0}',
        tolerance_pct: float = 2.0,
        out_dir: str = "designs",
        max_iterations: int = 6,
    ) -> str:
        """Design a component of a planar capacitance family to target capacitances: GDS -> predict -> iterate."""
        return f"""# Design a {component_class} to target capacitances

Targets (fF, `<a>_to_<b>` = positive mutual or self term): {target_capacitances_fF}; tolerance {tolerance_pct} %;
at most {max_iterations} prediction rounds; GDS files under `{out_dir}/`.

### Step 1: Know the component
`get_capacitance_component_info(component_class="{component_class}")`: default options, the net names (which keys the
targets refer to) and the dataset family. For a qubit (`Xmon`, `TransmonPocket`) turn frequency / anharmonicity targets
into capacitance targets with `capacitance_to_hamiltonian` on candidate matrices.

### Step 2: Start from data
With the `planar_capacitance` source (on by default), call `find_closest_designs(system_type="coupler",
coupler="{component_class}", target_params={target_capacitances_fF}, num_results=3)` (qubits: `system_type="qubit"`,
`qubit=...`). Take the best row's `design_options` as the starting point; these are Q3D-simulated values.

### Step 3: Predict and iterate
Loop: `predict_capacitance_from_design(component_class="{component_class}", design_options=...)` -> compare the
`sim_results` keys with the targets -> change ONE geometric option at a time in the direction the physics suggests
(longer overlap / smaller gap -> larger mutual) -> repeat. Keep a table per round: options changed, predicted values,
error vs target (% and fF), prediction time. Stop when every target is within {tolerance_pct} % or after
{max_iterations} rounds. Use model `QCap` (`QCap-Fast` is for screening trained families only and was worse than the physics
prior on new designs).

### Step 4: Write the GDS
`generate_component_gds(component_class="{component_class}", design_options=<final>, out_path="{out_dir}/<name>.gds")`
(dataset export: the convention the model was trained on). Optionally confirm with `predict_capacitance_from_gds` on that
file and `compute_capacitance_prior` as a physics cross-check.

### Step 5: Report
Final options, the full predicted Maxwell matrix (fF, ground first), every target with its error (% and fF), the GDS path,
and a note that the prediction is ML (model `QCap`; the paper's version of it scores 0.229 % (MAE 0.0674 fF /
RMSE 0.0792 fF) on validation, per `squadds://planar-capacitance-guide`): recommend a Q3D check before fabrication.

{REPORTING_RULES}
"""

    @mcp.prompt()
    def retrain_capacitance_model(
        data_description: str = "my Q3D-simulated layouts (GDS + cap_matrix.csv)",
        mode: str = "in_family",
        cluster: str = "SLURM",
    ) -> str:
        """Retrain the capacitance networks on your own data, on your own compute."""
        return f"""# Retrain on your own data ({mode}) on your own compute ({cluster})

Data: {data_description}.

### Step 1: QC every label
`check_capacitance_label(path=<cap_matrix.csv>)` per design (symmetry, Maxwell signs, closure, positive definiteness,
units, convergence). Fix or drop failures before anything else; never merge nets to make a label fit a layout.

### Step 2: Describe the designs
Write a JSON list, one entry per design: `gds`, `labels` (the `cap_matrix.csv` path or a matrix with `label_names`),
`key`, `family`, `split` (`train` / `val`), `names` (`auto` unless the layout needs an explicit net map).

### Step 3: Build shards
Up to 16 designs: `build_capacitance_training_shards(designs, out_dir)`. More: `plan_capacitance_training(mode="build",
designs_json=..., shard_dir=..., out_dir=..., script_dir=...)` and run the script it writes on your compute.

### Step 4: Smoke test
`run_capacitance_fit_smoke(shard_dir, out_dir, steps=200)`: checks the pipeline in seconds (not a usable model).

### Step 5: Plan the real run
`plan_capacitance_training(mode="{mode}", ...)`:
- `fit`: the paper recipe on the shard dir, one seed per SLURM array task (18,000 steps per seed).
- `in_family`: k-fold CV of a from-scratch fit against the zero-shot QCap; zero-shot wins ties.
- `collective`: your shards added to the all-data training set (`base_shards`), 3 seeds.
Fill `partition`, `account`, `time`, `mem`, `gpus` and `setup` (module loads, venv) with the user's cluster values; nothing
is site-specific. Give the user `run_on_slurm` / `run_locally`; do not run long jobs inside the MCP server.

### Step 6: Score
After the run: `score_capacitance_model(model=<run dir>, data=<shard dir>, split="val")`, next to
`score_capacitance_model(model="QCap", data=<shard dir>)` as the zero-shot reference (zero-shot only for designs
outside the SQuADDS dataset: QCap was trained on all of its labelled layouts). Report median % with MAE and RMSE
in fF, per family, with time, machine and threads. Test scoring is maintainer-only.

### Step 7: Use or share
Predict with the run (local model path) or share it via the `contribute_capacitance_data_or_model` prompt.

{REPORTING_RULES}
"""

    @mcp.prompt()
    def contribute_capacitance_data_or_model(kind: str = "data", path_or_repo: str = "my_contribution/") -> str:
        """Contribute planar capacitance data (GDS + Q3D matrices) or a trained model via a Hugging Face pull request."""
        return f"""# Contribute {kind}: `{path_or_repo}`

Pull requests are outward-facing: every contribution tool defaults to `dry_run=True`. Open a real pull request only after
the user has seen the dry-run result and said yes, with `dry_run=False` AND `confirm=True`.

## Data
1. `init_capacitance_contribution(dest_dir)` copies the template (manifest.yaml, designs.csv, README; gds/, sims/).
2. Fill it: manifest (contributor, licence CC-BY-4.0 or CC0-1.0, solver, stack, setup, layer map, units, family),
   one designs.csv row per design (`net_order` = `G` first, then conductors in GDS file order), the GDS as solved, the
   Q3D `cap_matrix.csv` as exported. Use `check_capacitance_label` on each matrix while filling it.
3. `validate_capacitance_contribution(path)`: fix every error (schema, GDS readback, conductor count vs matrix size,
   net names, symmetry, signs, positive definiteness, licence, duplicates of held-out test designs); read the warnings.
   `check_prior=True` (needs `[ml]`) compares each label with the physics prior and flags factor-2 outliers.
4. `contribute_capacitance_data(path)` (dry run): show the user the target folder, the files and the report.
5. On the user's go-ahead: `contribute_capacitance_data(path, dry_run=False, confirm=True)`; report the PR URL.

## Model
1. The model repo holds `*.pt` checkpoints (`squadds-ml train` run dirs) and `squadds_model.yaml` (name, title, licence,
   contributor, training_data sha256 + split used, val_score with median %, MAE and RMSE in fF, layouts, seconds,
   machine, threads). Score on validation with `score_capacitance_model`; never self-report a test score.
2. `validate_capacitance_model(model_repo_id, revision)` (or `local_dir=`); `val_data=<val shard dir>` recomputes the score.
3. `contribute_capacitance_model(model_repo_id, revision)` (dry run): show the zoo entry and report.
4. On the user's go-ahead: `contribute_capacitance_model(..., dry_run=False, confirm=True)`; report the PR URL.

A real PR needs a Hugging Face token (the contributor setup of Tutorial 3).

{REPORTING_RULES}
"""

    @mcp.prompt()
    def design_coupled_transmons(
        targets_GHz: str = (
            '{"f_q_A": 4.3, "eta_A": -0.214, "f_q_B": 4.6, "eta_B": -0.209, "f_r_A": 6.6, "f_r_B": 6.5, "J": 0.045, '
            '"chi_A": 0.0008, "chi_B": 0.0008, "kappa_A": 0.0016, "kappa_B": 0.0011}'
        ),
        variant: str = "gap",
        tolerance_pct: float = 1.0,
        out_dir: str = "two_transmon",
        max_iterations: int = 10,
    ) -> str:
        """Design two coupled transmons with half-wave readout resonators from Hamiltonian targets to a GDS (the two-transmon example)."""
        conventions = "\n".join(f"- {x}" for x in TWO_TRANSMON_CONVENTIONS)
        validity = "\n".join(f"- {x}" for x in TWO_TRANSMON_VALIDITY)
        return f"""# Two coupled transmons from Hamiltonian targets to a layout ({variant} coupler)

Targets (GHz): {targets_GHz}. Tolerance {tolerance_pct} % on every Hamiltonian parameter; at most {max_iterations}
geometry steps; files under `{out_dir}/`. This is the SQuADDS two-transmon design example (`squadds://planar-capacitance-tutorials`).

## Conventions (state them in the report)
{conventions}

## Validity caveats (carry them into the report, in substance)
{validity}

### Step 1: Solve the Hamiltonian
`solve_coupled_transmon_hamiltonian(targets={targets_GHz})`: E_J, E_C, E_J/E_C, L_J, I_c, n01, g, |chi| per transmon,
K and J, a target check (every miss should be ~0), and the target capacitances from an ideal reference. Show the transmon
table to the user.

### Step 2: Draw the first layout and predict it
`generate_two_transmon_gds(p=[160, 150, 0.8, 0.8, {25 if variant == "gap" else 15}], variant="{variant}",
out_path="{out_dir}/{variant}_0.gds", predict=True, model="QCap")` (p = [L_A, L_B, claw fraction A, claw fraction B,
coupling], um). Keep L_A != L_B (mirror-symmetric layouts can lose ground when read). Then
`solve_coupled_transmon_hamiltonian(targets, reference_matrix_fF=<its matrix_fF>, layout_matrix_fF=<its matrix_fF>)`: the
real target capacitances (with this layout's claw-to-ground and parasitics) and the layout's misses.

### Step 3: Iterate the geometry
Arm lengths set C_sigma (E_C, so eta), claw fractions set C_g (g, so chi), the coupling knob sets C_AB (J). Newton steps
on the five controlled capacitances: estimate the 5 x 5 Jacobian by finite differences with `model="QCap-Fast"` (directions
only), take the step, and confirm the new layout with `model="QCap"` before accepting it. If a QCap step does not
reduce the misses, recompute the Jacobian with QCap (in the two-transmon example's interdigital variant QCap-Fast pointed the coupler
the wrong way). Re-run `solve_coupled_transmon_hamiltonian(targets, reference_matrix_fF=M, layout_matrix_fF=M)` on each
accepted layout (E_J is re-solved so f01 is exact). Stop when `worst_miss_pct` < {tolerance_pct} or after
{max_iterations} steps. Log per step: p, the five capacitances (fF), the Hamiltonian misses (% and MHz), prediction time.

### Step 4: The final layout
`generate_two_transmon_gds(p=<final>, variant="{variant}", out_path="{out_dir}/{variant}_final.gds", predict=True)`.
Report the final E_J / L_J (from the layout matrix), every Hamiltonian target vs achieved, and the full 5 x 5 Maxwell
matrix (fF, ground first).

### Step 5: Resonators from SQuADDS
For each qubit, call `find_closest_designs` with the "find_closest_designs" arguments in the `resonators` part of the
Step 1 result (`system_type="qubit_cavity"`, `qubit="TransmonCross"`, `resonator_type="half"`: the half-wave search runs
on qubit-cavity systems; use only the resonator part of the row, its length and NCap feedline coupler). The row was
simulated with its own claw: `correct_half_wave_resonator(...)` (the `correction` text of the Step 1 result says which
values to pass; line frequency and node capacitance come from the final `layout` result) gives the resonator length and
kappa for your claw (the two-transmon example's resonator step). kappa is the SQuADDS row value scaled, not re-simulated: say so.

### Step 6: Report
Geometry, junctions, resonators, every target vs achieved (% and MHz), prediction model / time / machine /
threads, and the validity caveats above. Recommend one field solve (Q3D or Palace) of the final layout before
fabrication with the Q3D runner `scripts/planar_capacitance/two_transmon_q3d/run_q3d.py` (PyAEDT; the two-transmon example ran it on both of its final
layouts, about 31 min each on 16 cores; the stored solves are in `scripts/planar_capacitance/two_transmon_q3d/q3d/`). Those stored solves
describe the two-transmon example's default targets only, not a new layout.

{REPORTING_RULES}
"""
