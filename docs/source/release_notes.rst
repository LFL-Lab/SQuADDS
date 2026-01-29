Release Notes
=============

Version 0.4.3 (2026-01-28)
--------------------------

* **Alpha Version 0.4.3**

**Performance Improvements**

- **Great speedup of halfwave cavity workflows (seconds now instead of minutes)**
- Replaced `joblib` with **NumPy vectorization** in `Analyzer.find_closest`, making database queries instant and eliminating overhead.
- Added `numba.prange` support for true multi-core CPU utilization during parameter extraction.

**Bug Fixes**

- Corrected `N=4` (Quarter-Wave) hardcoding to `N=2` (Half-Wave) in `objects.py`.
- Resolved `TerminatedWorkerError` by removing process-based parallelism in favor of vectorization.
- Fixed `KeyError` in `run_eigenmode` for NCap simulations.
- Fixed `SettingWithCopyWarning` and linting errors in `analysis.py`.

**Improvements**

- Cleaned up unused parallel processing methods and dependencies.
- Improved code stability across operating systems (macOS, Windows, Linux).

**New Features**

- Added **Asynchronous Ansys Simulation** support (`simulate(..., run_async=True)`).
- Added `rich` for beautiful, colored terminal status outputs during simulations.
- Added `wait_for_all()` helper to synchronize async simulations.
- **Improved Simulation UX:**
    - `AnsysSimulator` is now stateful (stores `device_dict`).
    - Added `update_simulation_setup(target, **kwargs)` for granular hyperparameter updates:
        - Supports `target="qubit"`, `"cavity_claw"`, `"coupler"`, `"generic"`, or `"all"`.
        - Intelligently maps targets to correct setup dictionaries based on system type.
        - Interactive confirmation for unknown parameters to prevent typos.
    - Added `get_simulation_setup(target)` to view current setup parameters in formatted tables.
    - Added `update_design_parameters(**kwargs)` for direct geometry modification.
    - `simulate()` now uses valid internal state if no argument is provided.
    - **Transparency:** Prints simulation hyperparameters securely before execution.
    - **Half-Wave Cavity Support:** All three setups (`setup_qubit`, `setup_cavity_claw`, `setup_coupler`) now properly included in HWC dataframes.

**Known Issues**

- **Async Simulation Limitation:** `simulate(run_async=True)` may fail with "signal only works in main thread" error due to Ansys/Qiskit Metal limitations. Use `run_async=False` (default) for reliable simulations.

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
