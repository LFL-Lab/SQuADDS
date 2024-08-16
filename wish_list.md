# To Do List for SQuADDS:

Refer to [contribution guidelines](CONTRIBUTING.md) for more information on how to contribute code.

## Bug fixes:

- addressing issues on GitHub
- addressing TODOs in the code
- addressing any bugs in the code
- robustly handling caching and environment variables set up for all OS (some windows users had issues)

## Simulations:

- Generalize [SQDMetal](https://github.com/sqdlab/SQDMetal) Palace simulations to work with ANY HPC and make PR
- Do hyperparameter optimization study of Palace sims and test for reliability/repeatability
- Integrate SQDMetal (fixed version) as a dependency of SQuADDS
- Add easy Palace simulation API on SQuADDS (with parallel processing)
- Write a comprehensive tutorial on how to use Palace for simulations
- Stress-test and report any bugs for Ansys Simulations using SQuADDS package
- Providing compute to run simulations to cover sparse regions of the Hamiltonian space

## Core:

- **speed up half-wave cavity operations + decreate memory usage** (e.g. speed up both one time cost methods such as generating the dataset + local operations done at the edge)
- **stupid simple API for users to contribute experimental data to SQuADDS_DB**
- handle cases where user does not **wish** specify a resonator_type
- Better system design for both SQuADDS package and SQuADDS_DB?
- a system to "metalize" any .gds/.dxf file (i.e. from the CAD file generate the corresponding Qiskit Metal file)
- add support to handle designs generated via other tools (explicitly not qiskit metal e.g. [KQCircuits](https://github.com/iqm-finland/KQCircuits), [DXFWriter](https://github.com/SchusterLab/maskLib), [gdsfactory](https://github.com/gdsfactory/gplugins), [pyhfss](https://github.com/QW-QubitDesign/pyHFSS), [phidl](https://github.com/amccaugh/phidl))
- API access to KQCircuits for placing and rendering JJs
- API (modules/methods) to easily allow for adding more data columns to existing simulation entries in the database (e.g. someone with a lot of compute may want to rerun all the geometries in our database and add the participation ratio of various interfaces to the database -> lets make it easy for them?)
- changing datasets to `SQLite` or some other format to handle larger-than-memory datasets as we scale?
- refactor code to implement faster way with lower memory usage for handling dataframe operations?

## Contribution:

- Use HuggingFace Hub API for handling contributions to the database in a more streamlined/automated way (e.g. create clone, branch, PR, etc)
- Implement and Deploy an Acceptance Server for handling contributions to the database (i.e. calculates simulation and measured value discrepancies, automatically simulates some representative data points for reliability, notifies maintainers for approval, etc) [**Not Needed in the immediate future at all**]
- Automated Integration of CAD files along with their measured Hamiltonian parameters [**Not Needed in the immediate future at all**]

## ML:

- Architecture/Framework for how to deploy ML models on SQuADDS (via HuggingFace endpoints/spaces, in code, etc?)
- API (modules/methods) for incorporating ML interpolation feature into SQuADDS
- Use of HF Tasks for ML applications?
- Finding relevant design space variables for any system for a given set of $\hat{H}$ parameters using encoders
- Finding analytical dependence of $\hat{H}$ parameters on design space variables using Kolmogorov-Arnold Networks (KANs)[[kan_paper](https://arxiv.org/abs/2404.19756), [sym_res](https://kindxiaoming.github.io/pykan/Examples/Example_4_symbolic_regression.html)] [can be combined with above]
- Expanding on training datasets using cGANs/VAEs/PINNs (i.e. try to simulate the simulator?)
- Expand on this work [Elie Genois et al 2021](https://arxiv.org/pdf/2106.13126)

## Workflows:

- Addition of any other workflow that adds to/helps developers contribute
- an automated build check with comprehensive unit tests

## Feature Requests:

- add/show pictures of QComponents in `SQuADDS_DB`
- script to autoupdate a contributor thank-you list on both the github repo, docsite, and HuggingFace dataset
- letting users add methods with computation and append to `merged_df` for search in `Analyzer` module
- allowing users to pass on a circuit from SQCircuits and SQuADDS to provide them a first guess physical layout in Qiskit Metal (helpful resources: [sqcircuits](https://github.com/stanfordLINQS/SQcircuit/), [circuit_interpret_1](https://github.com/mahmut-aksakalli/circuit_recognizer), [circuit_interpret_2](https://github.com/aaanthonyyy/CircuitNet))
- incorporate [SCILLA](https://github.com/aspuru-guzik-group/scilla?tab=readme-ov-file) and/or its applications?

## Boring but Necessary:

- **Check for breaking changes in the latest version of dependencies and update the package accordingly**
- Add messaging for unconfident results (i.e. outside of weak coupling regime, etc)
- create unit tests for each feature/file
- create proper train/test/splits and changing `SQuADDS_DB()` to always return all data
- Standardize the way we handle units for simulated results + impmelent necessary changes in backend
- More tutorials on how to use the package + various applications of the package
- Check to see if precision of design parameters are being handled correctly + fix as needed
- Change all `NCap` to `CapNInterdigital`

## Fancy/For Fun:

- Implement LLM (support for OpenAI and local llama models) based queries for SQuADDS using [pandasai](https://docs.pandas-ai.com/intro)

---

# To Do List for SQuADDS:

Refer to [contribution guidelines](CONTRIBUTING.md) for more information on how to contribute code.

## Bug fixes:

- addressing issues on GitHub
- addressing TODOs in the code
- addressing any bugs in the code
- robustly handling caching and environment variables set up for all OS (some windows users had issues)

## Simulations:

- Generalize [SQDMetal](https://github.com/sqdlab/SQDMetal) Palace simulations to work with ANY HPC and make PR
- Do hyperparameter optimization study of Palace sims and test for reliability/repeatability
- Integrate SQDMetal (fixed version) as a dependency of SQuADDS
- Add easy Palace simulation API on SQuADDS (with parallel processing)
- Write a comprehensive tutorial on how to use Palace for simulations
- Stress-test and report any bugs for Ansys Simulations using SQuADDS package
- Providing compute to run simulations to cover sparse regions of the Hamiltonian space

## Core:

- **speed up half-wave cavity operations + decreate memory usage** (e.g. speed up both one time cost methods such as generating the dataset + local operations done at the edge)
- **stupid simple API for users to contribute experimental data to SQuADDS_DB**
- handle cases where user does not **wish** specify a resonator_type
- Better system design for both SQuADDS package and SQuADDS_DB?
- a system to "metalize" any .gds/.dxf file (i.e. from the CAD file generate the corresponding Qiskit Metal file)
- add support to handle designs generated via other tools (explicitly not qiskit metal e.g. [KQCircuits](https://github.com/iqm-finland/KQCircuits), [DXFWriter](https://github.com/SchusterLab/maskLib), [gdsfactory](https://github.com/gdsfactory/gplugins), [pyhfss](https://github.com/QW-QubitDesign/pyHFSS), [phidl](https://github.com/amccaugh/phidl))
- API (modules/methods) to easily allow for adding more data columns to existing simulation entries in the database (e.g. someone with a lot of compute may want to rerun all the geometries in our database and add the participation ratio of various interfaces to the database -> lets make it easy for them?)
- changing datasets to `SQLite` or some other format to handle larger-than-memory datasets as we scale?
- refactor code to implement faster way with lower memory usage for handling dataframe operations?

## Contribution:

- Use local llms/free secure LLM to create/update the measured_device dataset using the github repo
- Use HuggingFace Hub API for handling contributions to the database in a more streamlined/automated way (e.g. create clone, branch, PR, etc)
- Implement and Deploy an Acceptance Server for handling contributions to the database (i.e. calculates simulation and measured value discrepancies, automatically simulates some representative data points for reliability, notifies maintainers for approval, etc) [**Not Needed in the immediate future at all**]
- Automated Integration of CAD files along with their measured Hamiltonian parameters [**Not Needed in the immediate future at all**]

## ML:

- Architecture/Framework for how to deploy ML models on SQuADDS (via HuggingFace endpoints/spaces, in code, etc?)
- API (modules/methods) for incorporating ML interpolation feature into SQuADDS
- Use of HF Tasks for ML applications?
- Finding relevant design space variables for any system for a given set of $\hat{H}$ parameters using encoders
- Finding analytical dependence of $\hat{H}$ parameters on design space variables using Kolmogorov-Arnold Networks (KANs)[[kan_paper](https://arxiv.org/abs/2404.19756), [sym_res](https://kindxiaoming.github.io/pykan/Examples/Example_4_symbolic_regression.html)] [can be combined with above]
- Expanding on training datasets using cGANs/VAEs/PINNs (i.e. try to simulate the simulator?)
- Expand on this work [Elie Genois et al 2021](https://arxiv.org/pdf/2106.13126)

## Workflows:

- Addition of any other workflow that adds to/helps developers contribute
- an automated build check with comprehensive unit tests

## Feature Requests:

- add/show pictures of QComponents in `SQuADDS_DB`
- script to autoupdate a contributor thank-you list on both the github repo, docsite, and HuggingFace dataset
- letting users add methods with computation and append to `merged_df` for search in `Analyzer` module
- allowing users to pass on a circuit from SQCircuits and SQuADDS to provide them a first guess physical layout in Qiskit Metal (helpful resources: [sqcircuits](https://github.com/stanfordLINQS/SQcircuit/), [circuit_interpret_1](https://github.com/mahmut-aksakalli/circuit_recognizer), [circuit_interpret_2](https://github.com/aaanthonyyy/CircuitNet))
- incorporate [SCILLA](https://github.com/aspuru-guzik-group/scilla?tab=readme-ov-file) and/or its applications?

## Boring but Necessary:

- **Check for breaking changes in the latest version of dependencies and update the package accordingly**
- create unit tests for each feature/file
- create proper train/test/splits and changing `SQuADDS_DB()` to always return all data
- Standardize the way we handle units for simulated results + impmelent necessary changes in backend
- More tutorials on how to use the package + various applications of the package
- Check to see if precision of design parameters are being handled correctly + fix as needed
- Change all `NCap` to `CapNInterdigital`

## Fancy/For Fun:

- Implement LLM (support for OpenAI and local llama models) based queries for SQuADDS using [pandasai](https://docs.pandas-ai.com/intro)

---
