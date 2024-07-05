Release Notes
=============

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


