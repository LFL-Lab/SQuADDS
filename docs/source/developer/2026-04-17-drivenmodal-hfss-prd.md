# Driven-Modal HFSS API PRD

**Status:** Proposed  
**Date:** April 17, 2026  
**Branch:** `codex/drivenmodal-api-prd`

## Overview

SQuADDS currently exposes Ansys-backed eigenmode and lumped-element workflows through [`AnsysSimulator`](/Users/shanto/LFL/SQuADDS/SQuADDS/squadds/simulations/ansys_simulator.py), but it does not provide a production-ready API for HFSS driven-modal simulations. This feature adds a new driven-modal workflow for three geometry families already present in SQuADDS:

- qubit-coupled-resonator-feedline systems, for both quarter-wave and half-wave resonators
- qubit-claw geometries
- NCap geometries

The first release is design- and API-complete, but it intentionally stops short of live HFSS validation in CI. The implementation must be structured so that once Ansys-enabled validation is ready, the runtime and dataset flows can be turned on without re-architecting the feature.

Implementation will be tracked through a single repo-native work log at [2026-04-17-drivenmodal-hfss-worklog.md](/Users/shanto/LFL/SQuADDS/SQuADDS/docs/source/developer/2026-04-17-drivenmodal-hfss-worklog.md). That file is the source of truth for current execution status, active decisions, restart instructions, verification evidence, and handoff notes for future agents.

## Problem Statement

Users can already search SQuADDS for pre-simulated devices and can run some Ansys-backed workflows, but they cannot yet:

- run HFSS driven-modal simulations from SQuADDS geometry objects through a stable public API
- extract multi-frequency capacitance matrices from driven-modal Y-parameters for qubit-claw and NCap structures
- export and preserve Touchstone files as first-class artifacts in the contribution flow
- post-process driven-modal coupled-system data into SQuADDS-native Hamiltonian quantities such as `f_r`, `kappa`, `chi`, and `g`
- inspect or control the simulated layer stack in a user-visible way

This feature closes that gap while staying aligned with the current SQuADDS narrative: users begin from existing SQuADDS geometries or SQuADDS target parameters, then run an HFSS workflow through SQuADDS instead of dropping into an ad hoc Ansys script.

## Goals

### Product goals

- Provide a stable, production-ready SQuADDS API for HFSS driven-modal simulations.
- Preserve the existing `AnsysSimulator` entrypoint so the new workflow feels native to the package.
- Make layer-stack assumptions explicit and inspectable.
- Support artifact-rich data outputs, especially Touchstone and frequency-dependent capacitance data.
- Keep extracted scalar results easy to query and compare against the existing SQuADDS dataset.
- Ship two tutorials that validate the new workflow against existing SQuADDS/Q3D/HFSS-derived records before talking about new dataset population.

### Technical goals

- Use Qiskit Metal as the geometry-to-HFSS frontend.
- Use HFSS driven-modal setups with lumped ports for all supported systems in this phase.
- Keep materials fixed by preset in v1, while allowing user overrides for thickness-related layer-stack parameters.
- Separate raw solver artifacts from query-friendly extracted results without splitting the user API.

## Non-Goals

- No live HFSS execution in automated CI for this phase.
- No user-defined material editing in the public API for this phase.
- No support for arbitrary custom multiport networks beyond the named SQuADDS systems.
- No wave-port or driven-terminal workflow in this phase.
- No automatic upload of new driven-modal datasets until the result schema and artifact packaging are implemented and reviewed.

## Supported Systems In Scope

### 1. Capacitance extraction workflows

- qubit-claw geometries already represented in SQuADDS
- NCap geometries already represented in SQuADDS

For these systems, the HFSS driven-modal run will use lumped ports and export a frequency-dependent port admittance matrix. SQuADDS will post-process that admittance data into:

- capacitance matrix at each sweep frequency
- selected reference-frequency capacitance matrices for summary storage
- raw Touchstone and/or tabulated Y-parameter artifacts

### 2. Coupled-system workflows

- quarter-wave qubit-cavity-feedline systems
- half-wave qubit-cavity-feedline systems

For these systems, the HFSS driven-modal run will export a 3-port network. SQuADDS will then use scqubits and scikit-rf to compute:

- `LJ_g` and `LJ_e`
- `f_q` and `alpha`
- embedded `f_r` and `f_e`
- `chi`
- `Q_L` and `kappa`
- `g`

## Integration Constraints

### Qiskit Metal requirement

The implementation must use Qiskit Metal as the primary frontend to HFSS rather than bypassing it with raw AEDT scripting.

Local package inspection confirms that the installed `quantum-metal` stack already exposes the key renderer hooks needed for this feature:

- `QHFSSRenderer.add_drivenmodal_setup(...)`
- `QHFSSRenderer.add_sweep(...)`
- `QHFSSRenderer.render_design(..., port_list=..., jj_to_port=...)`
- `QHFSSRenderer.activate_ansys_design(..., solution_type="drivenmodal")`

The design should use the non-deprecated generic Ansys/HFSS activation surface where available, even if older `activate_drivenmodal_*` helpers still exist.

### Layer stack and materials

The new flow must make the simulated layer stack explicit.

Qiskit Metal already exposes `LayerStackHandler` with the canonical columns:

- `chip_name`
- `layer`
- `datatype`
- `material`
- `thickness`
- `z_coord`
- `fill`

This feature will adopt a fixed material preset, exposed as a named SQuADDS layer-stack profile, for example `squadds_hfss_v1`. Users may override thickness-related values, but not the material identities.

### Allowed overrides in v1

- substrate thickness
- metal thickness
- layer `z_coord` values when needed to preserve a consistent physical stack
- airbox / padding values if these are part of the simulation-space configuration rather than the material stack itself

### Disallowed overrides in v1

- changing substrate material
- changing metal material
- introducing arbitrary new layers from the public API

Every driven-modal request and result must expose the resolved layer stack so users can see exactly what was simulated.

## User Experience

### Entry paths

Users should be able to start from either of these flows:

1. **Geometry-first flow**
   - start from a SQuADDS design row, interpolated design, or explicit design options
   - build a driven-modal request
   - run the simulation
   - inspect raw and processed outputs

2. **Target-parameter-assisted flow**
   - use the existing `SQuADDS_DB` and `Analyzer` workflow to find or interpolate a candidate design from target parameters
   - hand that geometry into the driven-modal request builder
   - run the simulation and compare the extracted results to the pre-simulated record

The HFSS runner itself should remain geometry-driven under the hood. The target-parameter path is a convenience layer, not a different simulation backend.

## Public API

The public API should remain anchored on `AnsysSimulator`, with a dedicated internal driven-modal package under `squadds/simulations/drivenmodal/`.

### Public request models

- `DrivenModalLayerStackSpec`
- `DrivenModalSweepSpec`
- `DrivenModalArtifactPolicy`
- `CapacitanceExtractionRequest`
- `CoupledSystemDrivenModalRequest`

### Public result models

- `DrivenModalRunManifest`
- `CapacitanceExtractionResult`
- `CoupledSystemDrivenModalResult`

### Public facade methods on `AnsysSimulator`

- `build_capacitance_extraction_request(...)`
- `build_coupled_system_drivenmodal_request(...)`
- `run_drivenmodal(request, *, checkpoint_dir=None, export_artifacts=True)`
- `load_drivenmodal_result(path_or_manifest)`
- `plot_capacitance_vs_frequency(result, ...)`
- `plot_sparameter_response(result, ...)`

### Optional SQuADDS-native convenience helpers

- `build_request_from_dataframe_row(...)`
- `build_request_from_interpolated_design(...)`
- `build_request_from_target_params(...)`

These convenience helpers should live at the SQuADDS integration boundary, not inside the HFSS runner itself.

## API Behavior

### Capacitance extraction request

A capacitance extraction request must include:

- system kind: `qubit_claw` or `ncap`
- design payload from SQuADDS
- layer-stack preset plus thickness overrides
- driven-modal setup settings
- sweep specification
- port-node mapping
- artifact export policy

### Coupled-system request

A coupled-system request must include:

- resonator type: `quarter_wave` or `half_wave`
- design payload from SQuADDS
- layer-stack preset plus thickness overrides
- driven-modal setup settings
- sweep specification
- feedline input/output port assignment
- JJ port assignment
- qubit post-processing parameters or enough information to derive them from the design
- artifact export policy

## Internal Architecture

The implementation should be split into focused units:

### `models.py`

Typed request/result models and validation helpers.

### `layer_stack.py`

Layer-stack presets, override normalization, and conversion into the shape expected by Qiskit Metal’s `LayerStackHandler`.

### `ports.py`

System-specific port placement logic:

- qubit-claw driven-modal ports
- NCap driven-modal ports
- coupled-system feedline ports
- coupled-system JJ port

### `hfss_runner.py`

The HFSS/Qiskit Metal execution layer:

- activate driven-modal design
- render geometry
- apply ports
- create driven-modal setup
- insert sweep
- export Touchstone/Y-parameter artifacts

### `capacitance.py`

Capacitance extraction from frequency-dependent Y-parameters.

### `coupled_postprocess.py`

Coupled-system post-processing using scqubits and scikit-rf.

### `artifacts.py`

Checkpoint manifests, artifact path conventions, and resumable file export bookkeeping.

### `dataset_records.py`

Serialization into SQuADDS/Hugging Face-ready summary rows and artifact references.

## Porting Methodology

### Capacitance workflows

For `qubit_claw` and `ncap`, the solver output of interest is the multiport admittance response across frequency. The post-processing contract is:

1. Export admittance information or a Touchstone representation from HFSS.
2. Recover the complex Y-matrix at each frequency sample.
3. Compute the capacitance matrix from the imaginary part of Y:
   - `C(f) = Im(Y(f)) / (2πf)`
4. Apply node ordering and naming metadata so the resulting matrix is physically meaningful and reproducible.
5. Store:
   - the dense frequency-dependent matrix trace as an artifact
   - one or more extracted reference-frequency matrices inline in the summary record

The extracted capacitance matrices should be symmetrized and validated against basic passivity and naming invariants before they are serialized.

### Coupled-system workflows

The coupled-system workflow follows the four-stage process already outlined for this feature:

1. HFSS driven-modal 3-port extraction with lumped ports on feedline input, feedline output, and the JJ location
2. scqubits-based transmon analysis to compute `LJ_g`, `LJ_e`, `f_q`, and `alpha`
3. scikit-rf embedding of the JJ port to obtain `f_r`, `f_e`, `chi`, `Q_L`, and `kappa`
4. dispersive back-calculation of `g`, using the full expression that includes the non-RWA correction term

The public API should expose both the derived final values and the intermediate quantities needed for auditability and re-analysis.

## Checkpointing And Run Manifests

Driven-modal jobs are expensive and HFSS can fail partway through a run. Every simulation run must be checkpointed.

Each run should produce a manifest directory with:

- normalized request payload
- resolved layer-stack table
- geometry metadata
- setup and sweep metadata
- raw export file inventory
- processed result inventory
- per-stage status markers
- timestamps and version metadata

The minimum stage markers are:

- `prepared`
- `rendered`
- `setup_created`
- `sweep_completed`
- `artifacts_exported`
- `postprocessed`
- `serialized`

This allows reruns to resume from exported artifacts rather than repeating the entire HFSS solve whenever possible.

The same restart semantics must also be reflected in the implementation work log so another agent can determine, without reading git history, which checkpoint stages already exist and which code paths are expected to be resume-safe.

## Dataset Design

The driven-modal dataset should stay within the SQuADDS Hugging Face repository model, but summary rows and heavy artifacts should be separated logically.

### Summary records in `SQuADDS/SQuADDS_DB`

Each new driven-modal config should store compact, query-friendly summary rows. Proposed config families:

- `qubit_claw-drivenmodal-capacitance`
- `ncap-drivenmodal-capacitance`
- `half-wave-cavity_claw-drivenmodal`
- `quarter-wave-cavity_claw-drivenmodal`

Each summary row should include:

- `design`
- `sim_options`
- `sim_results`
- `layer_stack`
- `artifacts`
- `provenance`

### Artifact storage in the same dataset repo

Heavy artifacts should be stored in structured sidecar paths in the same Hugging Face dataset repository so that:

- a user can discover everything from one repo
- the summary rows remain compact
- the artifact location remains stable and easy to dereference

Proposed artifact path pattern:

`artifacts/drivenmodal/<config>/<run_id>/...`

This design keeps the user-facing data model simple now while still allowing a future move to a dedicated artifact repo if volume grows too large. The summary schema should therefore reference artifacts by URI/path, not by hardcoded assumption about repo boundaries.

### Summary record content for capacitance configs

- reference-frequency capacitance matrix
- node ordering
- sweep bounds and count
- optional sparse preview trace metadata
- artifact URIs for:
  - Touchstone or Y-parameter export
  - dense capacitance-vs-frequency table
  - checkpoint manifest

### Summary record content for coupled-system configs

- `f_r`
- `f_e`
- `chi`
- `Q_L`
- `kappa`
- `f_q`
- `alpha`
- `g`
- port definitions
- layer stack
- artifact URIs for:
  - `.s3p`
  - embedded resonance traces
  - post-processing tables
  - checkpoint manifest

## API Support For Visualization

The data model must make it easy to visualize:

- capacitance matrix entries as a function of frequency
- `S21` magnitude and phase
- embedded ground- and excited-state resonances
- comparison between driven-modal results and existing SQuADDS/Q3D-derived values

The plotting helpers should load dense artifact data transparently from the stored artifact URIs so users do not need to manually chase files in order to make standard comparisons.

## Tutorial Requirements

This feature ships with two tutorials.

### Tutorial 10: Driven-modal capacitance extraction for qubit-claw and NCap

This tutorial should:

1. load a qubit-claw geometry from SQuADDS
2. run the driven-modal capacitance extraction API
3. compare the extracted capacitance matrix against the Q3D-backed capacitance values already in SQuADDS
4. repeat the workflow for an NCap geometry
5. plot capacitance-vs-frequency data
6. end with a documentation section describing how this workflow will populate new datasets, what those datasets contain, and what the public API exposes

### Tutorial 11: Driven-modal coupled-system extraction and post-processing

This tutorial should:

1. load an existing quarter-wave or half-wave coupled-system design from SQuADDS
2. run the driven-modal coupled-system workflow
3. compute `f_r`, `kappa`, `chi`, and `g`
4. compare the extracted values against the pre-simulated SQuADDS record
5. visualize the embedded resonance response and the final scalar comparison
6. end with a documentation section describing how this workflow will populate new datasets, what those datasets contain, and what the public API exposes

## Testing Strategy

The first implementation wave should emphasize deterministic unit and fixture-based tests:

- request validation
- layer-stack preset resolution and override rules
- port-spec generation
- artifact manifest creation and resume semantics
- Y-to-capacitance post-processing
- coupled-system post-processing from fixture Touchstone data
- summary-row serialization
- `AnsysSimulator` facade wiring

What is explicitly deferred:

- live HFSS execution in CI
- golden comparisons that require Ansys on the runner

The implementation should still be written so that local Windows/Ansys validation can be added later as opt-in integration coverage.

## New Dependencies

The coupled-system post-processing plan requires `scikit-rf`, which is not currently declared in [`pyproject.toml`](/Users/shanto/LFL/SQuADDS/SQuADDS/pyproject.toml:1). The implementation plan should therefore add `scikit-rf` as a core simulation dependency unless it is intentionally isolated behind an optional extra.

## Risks And Mitigations

### Risk: port placement ambiguity

Mitigation:

- encode port placement as explicit typed specs
- persist port geometry metadata in the manifest
- keep node ordering stable and serialized

### Risk: layer-stack drift between runs

Mitigation:

- fixed preset names
- resolved layer-stack export in every result
- thickness overrides normalized into a canonical table before execution

### Risk: result rows become too large for Hugging Face summary loading

Mitigation:

- keep dense traces in artifact sidecars
- keep only compact extracted summaries inline

### Risk: users cannot tell what HFSS actually simulated

Mitigation:

- print and serialize the resolved layer stack
- print and serialize the port map
- keep setup and sweep parameters in every manifest and summary row

## Release Criteria

The feature is ready for implementation review when the following are true:

- the public request/result shapes are agreed upon
- the layer-stack contract is explicit
- the dataset and artifact schema are explicit
- the tutorial requirements are explicit
- the implementation plan is detailed enough to execute incrementally without revisiting product shape
