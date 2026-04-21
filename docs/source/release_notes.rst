Release Notes
=============

Version 0.4.4 (2026-04-19)
--------------------------

* **Alpha Version 0.4.4**

**Driven-Modal HFSS Simulations**

- Added a new ``squadds.simulations.drivenmodal`` subpackage that provides a stable, production-ready API for HFSS driven-modal workflows on three SQuADDS geometry families: NCap interdigital couplers, qubit-claw structures, and quarter/half-wave qubit-cavity-feedline coupled systems.
- Added typed request/result/spec models (``CapacitanceExtractionRequest``, ``CapacitanceExtractionResult``, ``CoupledSystemDrivenModalRequest``, ``CoupledSystemDrivenModalResult``, ``DrivenModalSetupSpec``, ``DrivenModalSweepSpec``, ``DrivenModalLayerStackSpec``, ``DrivenModalArtifactPolicy``, ``DrivenModalPortSpec``, ``DrivenModalRunManifest``) so driven-modal jobs can be specified, serialized, and resumed without ad-hoc dictionaries.
- Added an explicit, user-visible Qiskit-Metal layer-stack profile (``squadds_hfss_v1``) with thickness-only overrides; substrate/metal materials remain fixed by preset. Layer stacks are emitted as CSV alongside every run and re-attached on restart.
- Added an ``AnsysSimulator.run_drivenmodal(request, *, checkpoint_dir=None, export_artifacts=True)`` entrypoint that initializes a checkpointed driven-modal run from a typed request without bypassing the existing ``AnsysSimulator`` lifecycle.
- Added resumable artifact/checkpoint manifests (``squadds.simulations.drivenmodal.artifacts``) so a crash or restart does not require re-solving sweep points whose Touchstone/parquet artifacts are already on disk.
- Added port-spec builders (``build_capacitance_port_specs``, ``build_coupled_system_port_specs``, ``split_rendered_ports``) that produce Qiskit-Metal-compatible ``port_list`` / ``jj_to_port`` payloads from declarative port mappings, with strict validation of ``component`` / ``pin`` / ``metadata`` inputs.
- Added pure post-processing helpers for coupled systems (``calculate_chi_hz``, ``calculate_g_from_chi``, ``calculate_kappa_hz``, ``calculate_loaded_q``, ``y_to_s``, ``terminate_port_y``) that operate on raw Y-parameter tensors and return SQuADDS-native Hamiltonian quantities.
- Added admittance-to-capacitance helpers (``capacitance_matrix_from_y``, ``capacitance_dataframe_from_y_sweep``, ``maxwell_capacitance_dataframe``) for NCap and qubit-claw extraction.
- Added a JJ-port admittance toolkit (``squadds.simulations.drivenmodal.qubit_admittance``) with parallel ``R || L || C`` JJ models, terminated-port admittance reduction (Schur-complement based, with explicit short-circuit handling), zero-crossing extraction of the linear qubit pole, and ``scqubits``-backed transmon ``f_q`` / ``alpha`` extraction from saved ``Y33`` data.
- Added HFSS-data plumbing (``parameter_dataframe_to_tensor``, ``network_from_parameter_dataframe``, ``write_touchstone_from_dataframe``) that converts pyEPR parameter tables into dense complex tensors and exports first-class Touchstone artifacts via ``scikit-rf``.
- Added live-design helpers (``create_multiplanar_design``, ``write_qiskit_layer_stack_csv``, ``apply_cryo_silicon_material_properties``) that align driven-modal HFSS with the existing eigenmode/Q3D convention of cryogenic silicon (``epsilon_r = 11.45``, ``loss_tangent = 1e-7``) instead of the AEDT room-temperature default.
- Exposed the new request/result/spec models as top-level ``squadds.simulations`` exports so users can compose driven-modal runs without importing private modules.

**AnsysSimulator Improvements**

- Hardened ``AnsysSimulator.update_simulation_setup`` against ``None``, empty, and non-``dict`` setup payloads in both the unknown-parameter discovery loop and the final update loop, so malformed entries are skipped with a warning instead of raising.
- Made ``update_simulation_setup`` non-interactive friendly: when ``input()`` raises ``EOFError`` (e.g. CI, scripts, MCP), unknown parameters are auto-accepted with a clear console message instead of crashing.
- Replaced non-ASCII glyphs in the ``AnsysSimulator`` console output (simulation plan banners, status checks, parameter prompts) with ASCII so Windows ``cp1252`` consoles no longer mangle the logs.
- Refactored ``setMaterialProperties`` to prefer the live HFSS renderer's already-open project/design when patching cryogenic silicon, with safe ``ansys.aedt.core`` fallbacks; added DI-friendly ``materials_factory`` / ``hfss_factory`` hooks for testability.
- Hardened material lookups against PyAEDT versions where ``Materials.exists_material(...)`` returns a boolean rather than the material object: helpers now try ``checkifmaterialexists`` first and only mutate values that actually expose ``permittivity`` / ``dielectric_loss_tangent``.

**MCP Server**

- Added a built-in `Model Context Protocol (MCP) <https://modelcontextprotocol.io>`_ server that lets AI coding assistants (Claude, Cursor, VS Code Copilot, Gemini, Codex) interact with the SQuADDS database through the standardized MCP protocol.
- **16 MCP Tools**: Database browsing, design search (``find_closest_designs``), physics-based interpolation (``interpolate_design``), Qiskit-Metal code snippet generation (``get_qiskit_metal_snippet``), contributor info, and more.
- **6 MCP Resources**: Version info, citation, component lists, dataset summaries, and a comprehensive CPW layout guide (``squadds://layout-guide``) specifying impedance matching, feedline topology, and airbridge generation.
- **3 MCP Prompts**: Guided workflow templates for designing fab-ready qubit-cavity chips (``design_fab_ready_chip``), exploring the database, and finding optimal designs.
- Run with ``uv run squadds-mcp`` (stdio) or ``SQUADDS_MCP_TRANSPORT=streamable-http uv run squadds-mcp`` (HTTP).
- Works with Claude Desktop, Claude Code, Cursor, VS Code, Antigravity, Gemini CLI, and OpenAI Codex.
- Full documentation: `MCP_README.md <https://github.com/LFL-Lab/SQuADDS/blob/master/MCP_README.md>`_ and `MCP_DEVELOPER_GUIDE.md <https://github.com/LFL-Lab/SQuADDS/blob/master/MCP_DEVELOPER_GUIDE.md>`_.
- Added CI workflows for MCP testing (nightly compatibility checks against latest PyPI SQuADDS) and automated sync notifications on new releases.

**Refactor and Maintainability**

- Refactored large internal modules into smaller helper modules while preserving the public SQuADDS APIs.
- Kept `squadds.core.*`, `squadds.simulations.*`, and contributor/database entrypoints as compatibility facades over the refactored internals.
- Added characterization tests around the extracted helper modules to make future cleanup safer.

**Testing and CI**

- Added a dedicated `integration-smoke` workflow job that runs `tests/mvp_test.py` on GitHub Actions with live Hugging Face access enabled.
- Restored visible MVP smoke-test output in CI logs so collaborators can inspect the merged dataframe, closest-match result, interpolated design, and setup payloads directly.
- Improved test logging for integration debugging by using explicit pytest tracebacks/output settings in CI.

**Documentation**

- Added runnable driven-modal tutorial scripts to ``tutorials/``:
   - ``Tutorial-10_DrivenModal_Capacitance_Extraction.py`` (NCap and qubit-claw capacitance extraction from driven-modal Y-parameters).
   - ``Tutorial-12_DrivenModal_Qubit_Port_Admittance.py`` (qubit ``Y33`` extraction with model-sweep / feedline-termination sensitivity analysis).
   - ``Tutorial-13_DrivenModal_Combined_Hamiltonian_Extraction.py`` (single render, segmented qubit/bridge/resonator sweeps, end-to-end Hamiltonian comparison table).
- Added a ``tutorials/export_docsite_notebooks.py`` utility that converts the ``# %%`` script tutorials into ``.ipynb`` notebooks for both the ``tutorials/`` directory and the public docsite.
- Published the two production-ready notebooks to the docsite tutorial index:
   - ``Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb``.
   - ``Tutorial-11_DrivenModal_Combined_Hamiltonian_Extraction.ipynb`` (docsite-facing renumbering of Tutorial 13).
- Added a regression test (``tests/test_drivenmodal_tutorial_docs.py``) that fails if the shipped notebook artifacts disappear or lose the Tutorial 5-style title/license/footer cell structure.
- Added MCP server unit tests (43 tests covering schemas, utilities, and server creation).

**Dependencies**

- Added ``scikit-rf>=1.2`` as a runtime dependency to support Touchstone export and ``Network``-based post-processing for driven-modal coupled systems.
- Updated the `datasets` floor and lockfile so live Hugging Face metadata with `Json` feature types remains compatible during MVP smoke tests.
- Added optional ``mcp`` dependency group: ``mcp[cli]>=1.9``, ``pydantic>=2.0``.

**Bug Fixes**

- ``SQuADDS_DB.create_system_df`` (and therefore any ``Analyzer``-derived dataframe for the qubit+cavity coupled system) now inflates JSON-string sub-payloads in ``design_options_qubit`` / ``design_options_cavity_claw`` / ``design_options`` into real dicts. Previously, datasets that stored ``cplr_opts`` / ``lead`` / ``meander`` as JSON strings (current HuggingFace dataset schema) broke downstream workflows that mutate those fields via nested dict access (e.g. ``pred_df.design_options_cavity_claw.iloc[0]["cplr_opts"][key] = value``).
- ``Analyzer.find_closest`` now recomputes when any of the requested ``target_params`` columns are absent from the cached dataframe instead of raising ``KeyError`` mid-search.
- ``QubitCavity.make_cpws`` now preserves caller-provided ``lead`` and ``meander`` overrides (using ``setdefault`` instead of unconditional assignment), and actually applies the previously-discarded ``min(fillet, spacing/2.1)`` clamp so meander geometries stop self-intersecting on the Windows/Ansys validation machine.
- ``QubitCavity.make_qubit`` / ``make_cavity`` / ``make_cpws`` now use explicit key checks for ``cpw_opts`` / ``cpw_options`` and ``coupler_options`` / ``cplr_opts`` instead of bare ``except`` blocks, so the underlying ``KeyError`` is raised cleanly when neither alias is present.
- Tutorial driven-modal flows now work around a ``pyEPR.load_ansys_project(...)`` path-duplication bug when ``QHFSSRenderer`` is initialized with both ``project_path`` and ``project_name``.
- Fixed a `ScalingInterpolator` regression where NumPy was not imported during interpolation scaling.
- Fixed half-wave parquet output writing so reruns overwrite expected artifacts even when the destination directory already exists.
- Fixed helper edge cases found during refactor review, including option-key detection, sweep chunk key preservation, optional merger-term handling, contributor token fallback behavior, and live dataset setup/design payload normalization.

**Backwards-Compatibility Notes**

- ``QubitCavity.make_cpws`` default ``lead.start_straight`` changed from ``"75um"`` to ``"150um"`` (carried over from the QubitCavity geometry fix). Users who previously did not specify ``lead.start_straight`` in ``cavity_claw_options.cpw_opts.left_options`` will see a longer initial straight section in the meandered CPW. Pass an explicit ``lead.start_straight`` override to restore the prior geometry.
- The fillet on the meandered CPW is now actually clamped to ``min(provided_fillet, spacing/2.1)``; previously this calculation was performed but its result was discarded. Geometries that relied on the unclamped fillet may render slightly differently.

---

Version 0.4.3 (2026-01-28)
--------------------------

* **Alpha Version 0.4.3**

**Performance Improvements**

- Great speedup of halfwave cavity workflows (**seconds now instead of minutes*!**)
- Replaced `joblib` with NumPy vectorization in `Analyzer.find_closest`, making database queries instant and eliminating overhead.
- Added `numba.prange` support for true multi-core CPU utilization during parameter extraction.

**New Features**

- `AnsysSimulator` is now stateful (stores `device_dict`).
- Added `update_simulation_setup(target, **kwargs)` for granular hyperparameter updates:
    - Supports `target="qubit"`, `"cavity_claw"`, `"coupler"`, `"generic"`, or `"all"`.
    - Intelligently maps targets to correct setup dictionaries based on system type.
    - Interactive confirmation for unknown parameters to prevent typos and for specifying more hyperparameters.
- Added `get_simulation_setup(target)` to view current setup parameters in formatted tables.
- `simulate()` now uses valid internal state if no argument is provided.
- **Transparency:** Prints simulation hyperparameters securely before execution.
- All three setups (`setup_qubit`, `setup_cavity_claw`, `setup_coupler`) now properly included in half-wave cavity dataframes.
- Added `update_design_parameters(**kwargs)` for direct geometry modification.

**Bug Fixes**

- Corrected `N=4` (Quarter-Wave) hardcoding to `N=2` (Half-Wave) in `objects.py`.
- Resolved `TerminatedWorkerError` by removing process-based parallelism in favor of vectorization.
- Fixed `KeyError` in `run_eigenmode` for NCap simulations.
- Fixed `SettingWithCopyWarning` and linting errors in `analysis.py`.

**Improvements**

- Cleaned up unused parallel processing methods and dependencies.
- Improved code stability across operating systems (macOS, Windows, Linux).
- Added `rich` for beautiful, colored terminal status outputs during simulations.

---

Version 0.4.2 (2026-01-27)
--------------------------

* **Alpha Version 0.4.2**

**Bug Fixes**

- Fixed a critical `Numba` compilation error in half-wave cavity calculations by replacing incompatible `Convert` utility calls with JIT-compatible implementations.

---

Version 0.4.1 (2026-01-23)
--------------------------

* **Alpha Version 0.4.1**

**Improvements**

- A more accurate coupling strength calculation, see :download:`g_derivation.pdf <resources/g_derivation.pdf>`

**Bug Fixes**

- Fixed a critical issue in Ansys simulations by pinning the ``pyaedt`` dependency to a compatible version (0.23).

---

Version 0.4.0 (2025-01-09)
--------------------------

* **Alpha Version 0.4.0 [MAJOR INFRASTRUCTURE RELEASE]**

**Breaking Changes**

- Migrated from ``setup.py`` to modern ``pyproject.toml`` (PEP 621)
- Switched from ``qiskit-metal`` to ``quantum-metal>=0.5.0`` (ARM64 compatible)
- Removed Conda-based installation; now uses ``uv`` for package management
- Python 3.10+ required (dropped 3.9 support)
- NumPy pinned to ``<2.0`` for quantum-metal compatibility

**New Features**

- Full native Apple Silicon (ARM64) support via quantum-metal
- Modern ``uv`` package management for faster, more reliable installs
- Added ``ruff`` linting to development dependencies
- Added PySide6 for modern Qt GUI support
- Cross-platform CI/CD testing on Python 3.10, 3.11, 3.12

**Performance Improvements**

- **~7x speedup** for half-wave cavity Hamiltonian parameter calculations
- Added LRU caching for transmon E01/anharmonicity calculations
- Pre-compute unique EC values before parallel processing to avoid redundant matrix diagonalizations
- Vectorized EC calculation using numba for array processing
- Optimized value mapping using numpy ``searchsorted`` (O(log n) instead of O(n))
- Reduced transmon calculations from ~16.5M to ~1934 unique values for typical datasets

**Infrastructure**

- Replaced all CI/CD workflows with ``astral-sh/setup-uv`` action
- Added ``uv.lock`` for reproducible builds
- Updated Docker image to use uv instead of Conda
- Modernized PyPI publishing with trusted publishing (OIDC)
- Added ruff configuration in pyproject.toml

**Dependencies Removed (Bloatware)**

- ``memory_profiler`` (unused)
- ``addict`` (unused)
- ``dask`` (transitive)
- ``pyarrow`` (transitive)
- ``cython`` (build-time only)
- ``qutip`` (transitive via scqubits)

**Dependencies Added**

- ``matplotlib`` (was missing)
- ``shapely`` (was missing)
- ``pyside6`` (for quantum-metal)
- ``ruff`` (dev dependency)

**Documentation**

- Completely rewrote installation instructions for uv workflow
- Updated developer notes with new development setup
- Simplified getting started guide
- Removed all Conda/environment.yml references

**Bug Fixes**

- Fixed deprecated ``HfFolder`` import (replaced with ``get_token()``)
- Fixed version consistency checks in prepare-release workflow

---

Version 0.3.7 (2024-03-19)
--------------------------

* **Alpha Version 0.3.7**

**New Features**

- Added automated version bumping workflow
- Added Palace tutorial with EPR calculations
- Added ML interpolation notebook
- Added SQuADDS WebUI tutorial
- Added GDS post-processing files for DVK foundry
- Added support for airbridges
- Added DeepWiki integration for codebase chat
- Added contribution portal integration

**Documentation**

- Updated Tutorial 7 with EPR calculations and improved formatting
- Updated Tutorial 8 with ML interpolation examples
- Added Apple Silicon installation instructions
- Fixed broken links in tutorials
- Added Qiskit Metal Fall Fest ML interpolation tutorial link
- Updated contribution guidelines and documentation
- Added issue templates and improved issue management

**Bug Fixes**

- Fixed branch name inconsistency in Docker workflow
- Updated outdated GitHub Actions versions
- Fixed GUI/design bug in Tutorial 7 leading to 4 ports on TL
- Fixed KLayout version compatibility issues
- Fixed broken links in documentation

**Infrastructure**

- Added automated contributor list updates
- Improved CI/CD pipeline with version bumping
- Added automated release notes generation

---

Version 0.3.6 (2024-10-13)
--------------------------

* **Alpha Version 0.3.6**

**Bug Fixes**

- Fixed bugs in the `AnsysSimulator` code
- Added non local rst dependencies for contributor list
- Added some more db access methods

---

Version 0.3.5 (2024-10-11)
--------------------------

* **Alpha Version 0.3.5 [MAJOR RELEASE]**

**New Features**

- Added methods to `Analysis` to retrieve design parameters easily from any dataframe with `design_options` column
- Added custom TransmonCross and JJ elements
- Added methods for getting `design_df` from ML interpolations
- Added some gds post-processing methods for gds files

**Bug Fixes**

- Fixed bugs in the `AnsysSimulator` code
- Minor bug fixes in string inconsistencies and better error messaging

**Change Log**

https://github.com/LFL-Lab/SQuADDS/compare/v0.3.4...v0.3.5

---

Version 0.3.4 (2024-09-27)
--------------------------

* **Alpha Version 0.3.4**

**New Features**

- Fixed unicode error in local builds on Windows
- Added measured device data API
- Added more data columns to experimental device


---

Version 0.3.3 (2024-09-20)
--------------------------

* **Alpha Version 0.3.3**

**New Features**

- Patched some bugs in simulation and contribution info query

---

Version 0.3.2 (2024-08-30)
--------------------------

* **Alpha Version 0.3.2**

**New Features**

- Patched a bug in simulation of some half-wave cavity resonator systems

---

Version 0.3.1 (2024-08-16)
--------------------------

* **Alpha Version 0.3.1**

**New Features**

- Added added methods for showing better contributions and measured device information (includes fabublox recipes)
- Updated tutorials (1,3 and 3p5)

---

Version 0.3.0 (2024-08-12)
--------------------------

* **Alpha Version 0.3.0**

**New Features**

- Added support for half-wave cavity resonators

---

Version 0.2.36 (2024-07-06)
--------------------------

* **Alpha Version 0.2.36**

**Breaking Changes**

- support for `NCap` and `CapNInterdigitalTee` strings everywhere in HuggingFace and the codebase (**`Ncap` will be dropped in the future**)
- `select_coupler` would no longer be supported. Use `select_resonator_type` instead
-  `Analyzer` object no longer requires `SQuADDS_DB()` in its constructor

**New Features**

- Added the option to create lambda/2 resonators
- Interpolation support for lambda/2 resonators

**Improvements**

- Added `chi` as a query parameter
- Better and more intuitive API
- "hot reload" of `Analyzer` object
- Updated documentation and tutorials
- Added `release-drafter` for automated release notes

**Bug Fixes**

- Fixed bugs in Simulation Code (simulation of CapNInterdigitalTee and half feature-half_wave_cavity)

---

Version 0.2.35 (2024-07-04)
--------------------------

* **Alpha Version 0.2.35**

**Bug Fixes**

- Bug in accessing simulation data from HuggingFace has been patched

**Documentation**

- Added `FAQ <https://lfl-lab.github.io/SQuADDS/source/getting_started.html#accessing-the-database>`_ for issue with accessing HF data on SQuADDS releases < 0.2.35

Version 0.2.34 (2024-07-04)
--------------------------

* **Alpha Version 0.2.34**

**Bug Fixes**

- Addressed API to only show sim data configs
- Fixed a bug in AnsysSimulator for setup_dict for qubit-cavity systems
- Fixed bugs for geometries extracted from interpolator (and utils.py)

**Documentation**

- Added tutorial3p5 files
- Added README.md, wish_list.md, and docs/source/developer/index.rst from feature-half_wave_cavity (latest)

---

Version 0.2.33 (2024-03-14)
--------------------------

* **Alpha Version 0.2.33**

- Bug fixes in ansys_simulator code for whole device `sweep` functionality
- Added multiple helper/utility methods for ansys simulations
- Methods added to clulate chi, full dispersive shift of the cavity
- Updated `requirements.txt` and documentation
- Added method to set `GITHUB_TOKEN`

Version 0.2.32 (2024-02-02)
--------------------------

* **Alpha Version 0.2.32**

- Bug fix in ansys_simulator code
- Fixed hyperlinks

Version 0.2.31 (2024-01-17)
--------------------------

* **Alpha Version 0.2.31**

- Bug fix in contributor validation function


Version 0.2.3 (2024-01-17)
--------------------------

* **Alpha Version 0.2.3**

- Bug fixes in simulator engine

- Added sweep functionality to simulator

- Updated Tutorial 2 to reflect changes in simulator code

- Added functionality for adding to existing configurations

- Completed Tutorial 3


Version 0.2.2 (2024-01-10)
--------------------------

* **Alpha Version 0.2.2**

- Documentation added to the entire codebase


Version 0.2.1 (2024-01-10)
--------------------------

* **Alpha Version 0.2.1**

- Bug fixes:

  - change `"c"` to `"readout"` in both code and database entries

- Handled Warnings from pyaedt

Version 0.2 (2023-12-24)
--------------------------

* **Alpha Version 0.2**

- Simulator functionalities added

- Tutorial-2_Simulate_interpolated_designs added

- Issues with automated docsite generator persist


Version 0.1.7 (2023-12-23)
--------------------------

* **Alpha Version 0.1.7**

- Fixed issues with automated docsite generator

- Standardized path imports in all files

- Version to merge with simulator functionalities

Version 0.1.6 (2023-12-20)
--------------------------

* **Alpha Version**

  - Database hosted on `HuggingFace <https://huggingface.co/datasets/SQuADDS/SQuADDS_DB>`_

  - Pre-simulated data on TransmonCross, Cavity with Claw and Couplers only.

  - Closest pre-simulated design and interpolated design retrieval implemented

  - Interpolation logic based on our `paper <https://arxiv.org/>`_

  - Tutorials on basic usage, contribution, and simulation added

  - pypi package created
