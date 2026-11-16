MCP Server (AI Agent Integration)
=================================

SQuADDS includes a built-in `Model Context Protocol (MCP) <https://modelcontextprotocol.io>`_ server that lets AI coding assistants interact with the entire database — searching designs, interpolating parameters, and exploring components — through a standardized protocol.

.. contents:: On this page
   :local:
   :depth: 2


Agent Setup (Copy-Paste Prompt)
-------------------------------

If you're using an AI coding assistant (Claude, Cursor, Copilot, Gemini, Codex), just paste this prompt to have it set up SQuADDS MCP for you:

.. code-block:: text

   I need you to set up the SQuADDS MCP server so I can access the superconducting
   qubit design database through you. Here's what to do:

   1. Clone the repo and install:
      git clone https://github.com/LFL-Lab/SQuADDS.git
      cd SQuADDS
      uv sync --extra mcp

   2. Add the MCP server to your config. The command to run the server is:
      uv run --directory /path/to/SQuADDS squadds-mcp

   3. Once connected, read the `squadds://guide` resource for a quick overview
      of available tools.

   The server exposes these key tools:
   - `list_components` / `list_datasets` — explore the database
   - `find_closest_designs` — find designs matching target Hamiltonian parameters
   - `interpolate_design` — get physics-interpolated designs
   - `get_hamiltonian_param_keys` — discover valid search parameters

   Typical target parameter ranges:
   - qubit_frequency_GHz: 3–8
   - anharmonicity_MHz: −500 to −50
   - cavity_frequency_GHz: 5–12
   - kappa_kHz: 10–1000
   - g_MHz: 10–200

   Please set this up and confirm you can access the SQuADDS tools.


Manual Setup
------------

Install
^^^^^^^

.. code-block:: bash

   git clone https://github.com/LFL-Lab/SQuADDS.git
   cd SQuADDS
   uv sync --extra mcp

Run
^^^

.. code-block:: bash

   # stdio mode (for local AI assistants)
   uv run squadds-mcp

   # HTTP mode (for networked/remote usage)
   SQUADDS_MCP_TRANSPORT=streamable-http uv run squadds-mcp


AI Client Configuration
------------------------

Claude Desktop
^^^^^^^^^^^^^^

Add to ``claude_desktop_config.json``:

.. code-block:: json

   {
     "mcpServers": {
       "squadds": {
         "command": "uv",
         "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
       }
     }
   }

Claude Code
^^^^^^^^^^^

.. code-block:: bash

   claude mcp add squadds -- uv run --directory /path/to/SQuADDS squadds-mcp

Cursor
^^^^^^

Add to ``.cursor/mcp.json`` in your project:

.. code-block:: json

   {
     "mcpServers": {
       "squadds": {
         "command": "uv",
         "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
       }
     }
   }

VS Code (Copilot)
^^^^^^^^^^^^^^^^^

Add to ``.vscode/settings.json``:

.. code-block:: json

   {
     "mcp": {
       "servers": {
         "squadds": {
           "command": "uv",
           "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
         }
       }
     }
   }

Antigravity / Gemini CLI
^^^^^^^^^^^^^^^^^^^^^^^^^

Add to ``~/.gemini/settings.json`` (or project-level ``.gemini/settings.json``):

.. code-block:: json

   {
     "mcpServers": {
       "squadds": {
         "command": "uv",
         "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
       }
     }
   }

OpenAI Codex CLI
^^^^^^^^^^^^^^^^

.. code-block:: bash

   codex --mcp-config mcp.json

With ``mcp.json``:

.. code-block:: json

   {
     "mcpServers": {
       "squadds": {
         "command": "uv",
         "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
       }
     }
   }


Available Tools
---------------

Database Tools
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``list_components``
     - List supported component types
     - —
   * - ``list_component_names``
     - List component names for a type
     - ``component``
   * - ``list_configs``
     - List all dataset configurations
     - —
   * - ``list_datasets``
     - Overview of all datasets
     - —
   * - ``get_dataset_info``
     - Dataset metadata (features, size)
     - ``component``, ``component_name``, ``data_type``
   * - ``get_dataset``
     - Load dataset rows (paginated)
     - ``component``, ``component_name``, ``data_type``, ``limit``, ``offset``
   * - ``list_measured_devices``
     - All experimental devices
     - —
   * - ``get_simulation_results``
     - Simulation results for a device
     - ``device_name``
   * - ``list_squadds_db_sources``
     - Enabled / available ``SQuADDS_DB`` sources and the configs each adds
     - —
   * - ``enable_squadds_db_source``
     - Enable an extra source (``planar_capacitance``: Xmon, TransmonPocket, concentric, star, shielded configs) for the following calls
     - ``source``, ``revision``, ``local_dir``
   * - ``disable_squadds_db_source``
     - Disable an extra source
     - ``source``

Analysis Tools
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``get_hamiltonian_param_keys``
     - Valid target parameter keys
     - ``system_type``
   * - ``find_closest_designs``
     - **Primary search tool** — find closest designs (``system_type="coupler"`` with capacitance targets in fF for the planar capacitance couplers)
     - ``system_type``, ``target_params``, ``num_results``, ``metric``, ``qubit``, ``coupler``
   * - ``predict_closest_designs_ml``
     - Closest single-component designs (``qubit`` / ``coupler``) predicted with the planar capacitance models next to their stored simulation (``Analyzer.predict_capacitance``; needs ``squadds[ml]``)
     - ``system_type``, ``target_params``, ``num_results``, ``qubit``, ``coupler``, ``model``, ``eps_r``, ``thickness_um``

Interpolation Tools
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``interpolate_design``
     - Physics-scaled interpolated design
     - ``target_params``, ``qubit``, ``cavity``, ``resonator_type``

Contribution Tools
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``get_reference_device``
     - Reference experimental device info
     - ``component``, ``component_name``, ``data_type``
   * - ``get_fabrication_recipe``
     - Fabrication recipe for a device
     - ``device_name``
   * - ``list_contributors``
     - All data contributors
     - —


Hosted ML API tools
^^^^^^^^^^^^^^^^^^^

``list_hosted_ml_models`` lists the models the SQuADDS ML API serves (``GET /v1/models``, with each model's inputs and
units) and ``predict_with_hosted_ml_model`` runs one with JSON ``inputs`` and ``options``. No ``[ml]`` extra is needed. See
:ref:`hosted-inference`.

Planar Capacitance Tools
^^^^^^^^^^^^^^^^^^^^^^^^

Running a model needs the optional ML extra (``pip install "squadds[ml]"``); without it the tools return an install hint.
Start with
the resource ``squadds://planar-capacitance-guide``; details in :doc:`ml_models`.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``list_capacitance_models``
     - Released planar capacitance models (``QCap`` is the default; ``QCap-Fast`` is for screening trained families)
     - —
   * - ``predict_capacitance_from_gds``
     - Full N x N Maxwell capacitance matrix (fF, ground first) from a GDS layout, with physical checks and timing; other
       substrates solve the prior at the stack (QCap; QCap-Fast rescales and warns), see :doc:`ml_models`; ground and
       conductors on one layer: the ground is detected automatically (``ground_detection``), ``ground_pieces`` overrides it
     - ``gds_path``, ``model``, ``threads``, ``eps_r``, ``thickness_um``, ``ground_pieces``
   * - ``query_planar_capacitance``
     - Query the planar capacitance dataset by family, conductor count or split
     - ``family``, ``n_conductors``, ``split``, ``collection``
   * - ``get_planar_capacitance_conventions``
     - Units, Maxwell signs, layer map, training stack and the off-stack routes, models and physics priors
     - —
   * - ``validate_capacitance_contribution``
     - Validate a local capacitance data contribution folder before opening a pull request
     - ``path``, ``check_prior``, ``reference``
   * - ``compute_capacitance_prior``
     - Physics-only Maxwell matrix of a GDS (optimized or coarse thin-film prior), a baseline for the model; returns each solve, the ground detection and the warnings
     - ``gds_path``, ``prior``, ``threads``, ``eps_r``, ``thickness_um``, ``ground_pieces``
   * - ``inspect_capacitance_layout``
     - The conductors and ground a prediction would read, without predicting: count, areas, ground area in the file vs read, numbered pieces and the detected ground for one-layer files, optional PNG
     - ``gds_path``, ``ground_layers``, ``conductor_layers``, ``ground_pieces``, ``png_path``
   * - ``get_planar_capacitance_design``
     - One dataset design in detail: options, Q3D solve with QC flags and convergence, label, adaptive passes, stored priors, GDS path
     - ``design_id``, ``parts``
   * - ``nearest_planar_capacitance_designs``
     - Dataset designs geometrically closest to a design (pooled embedding; not a capacitance distance)
     - ``design_id``, ``k``, ``family``, ``split``

Design: component, GDS and Hamiltonian
""""""""""""""""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``get_capacitance_component_info``
     - Families, Qiskit Metal classes, SQuADDS configs, net names; default options of one class
     - ``component_class``
   * - ``generate_component_gds``
     - Component class + options -> GDS in the dataset export (returns the path and a layout readback)
     - ``component_class``, ``design_options``, ``out_path``, ``export``
   * - ``predict_capacitance_from_design``
     - Component class + options -> Maxwell matrix with SQuADDS net names (``MLSimulator``)
     - ``component_class``, ``design_options``, ``model``, ``save_gds_dir``, ``eps_r``, ``thickness_um``
   * - ``capacitance_to_hamiltonian``
     - Maxwell matrix (or a GDS) -> C_sigma, C_g, E_C, E_J/E_C, L_j, f_q, alpha, g (grounded or floating transmon)
     - ``matrix_fF``, ``names`` or ``gds_path`` (with ``model``, layers, ``eps_r``, ``thickness_um``, ``ground_pieces``), ``qubit_type``, ``qubit_nodes``, ``coupler``, ``EJ_GHz``
   * - ``solve_coupled_transmon_hamiltonian``
     - Two coupled transmons + two half-wave resonators (the two-transmon example): targets (f_q, eta, f_r, kappa, J, chi) -> E_J, E_C, L_J, g, J and the target 5 x 5 Maxwell matrix (full Hamiltonian, no RWA); evaluates a predicted layout matrix; resonator lines 50 ohm (fixed)
     - ``targets``, ``reference_matrix_fF``, ``layout_matrix_fF``
   * - ``generate_two_transmon_gds``
     - The two-Xmon layout (gap or interdigital coupler) -> GDS, optionally with its predicted Maxwell matrix (outside the trained families) and a ground-area check
     - ``p``, ``variant``, ``out_path``, ``predict``, ``model``, ``eps_r``, ``thickness_um``
   * - ``correct_half_wave_resonator``
     - A SQuADDS half-wave row adapted to your claw (the two-transmon example's resonator step): the row's claw capacitance (predicted), resonator length and kappa
     - ``row_frequency_GHz``, ``row_kappa_kHz``, ``row_length_um``, ``f_r_GHz``, ``line_frequency_GHz``, ``node_capacitance_fF``, ``claw_options``

Training, scoring and label QC
""""""""""""""""""""""""""""""

Long runs never execute inside the server: ``plan_capacitance_training`` writes a driver and a generic SLURM script for your own compute. In-process work is bounded (16 designs for shards, 500 steps for a smoke fit, 20 layouts for end-to-end scoring).

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``check_capacitance_label``
     - Label QC of a Q3D ``cap_matrix.csv`` or a matrix (symmetry, signs, closure, PD, units, passes)
     - ``path`` or ``matrix_fF`` + ``names``
   * - ``build_capacitance_training_shards``
     - A few GDS + labels -> training shards (embedding + physics priors)
     - ``designs``, ``out_dir``
   * - ``plan_capacitance_training``
     - Driver + SLURM scripts for ``build`` / ``fit`` / ``in_family`` / ``collective`` retraining
     - ``mode``, ``out_dir``, ``script_dir``, ``shard_dir``, ``designs_json``, ``partition``, ``account``
   * - ``run_capacitance_fit_smoke``
     - Short in-process fit (<= 500 steps) to check a shard dir end to end
     - ``shard_dir``, ``out_dir``, ``steps``
   * - ``score_capacitance_model``
     - Validation score: median % with MAE and RMSE in fF, per family, with time, machine, threads (test scoring refused)
     - ``model``, ``data`` or ``layouts``, ``split``

Contributing data and models
""""""""""""""""""""""""""""

Every outward-facing tool is a dry run by default; a pull request is opened only with ``dry_run=False`` **and** ``confirm=True``.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Tool
     - Description
     - Key Parameters
   * - ``init_capacitance_contribution``
     - Copy the contribution template (manifest.yaml, designs.csv, README) to a folder
     - ``dest_dir``
   * - ``contribute_capacitance_data``
     - Validate a contribution folder and plan (or, with confirm, open) the dataset pull request
     - ``path``, ``dry_run``, ``confirm``
   * - ``validate_capacitance_model``
     - Validate a model repository: checkpoints, sha256, metadata, validation score
     - ``model_repo_id``, ``revision``, ``local_dir``, ``val_data``
   * - ``contribute_capacitance_model``
     - Validate and plan (or, with confirm, open) the model-zoo pull request
     - ``model_repo_id``, ``dry_run``, ``confirm``

Available Resources
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - URI
     - Description
   * - ``squadds://version``
     - SQuADDS + MCP server versions
   * - ``squadds://citation``
     - BibTeX citation
   * - ``squadds://components``
     - Supported component types
   * - ``squadds://configs``
     - Dataset configuration strings
   * - ``squadds://datasets``
     - Dataset summary table
   * - ``squadds://guide``
     - Quick reference for AI agents
   * - ``squadds://planar-capacitance-guide``
     - Planar capacitance models (QCap, QCap-Fast): accuracy (median % with MAE and RMSE in fF), time, conventions, the predict -> Hamiltonian, design, retrain and contribute workflows
   * - ``squadds://planar-capacitance-tutorials``
     - Tutorials 11-15 with the MCP tools that reproduce each


Available Prompts
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Prompt
     - Description
     - Parameters
   * - ``design_qubit_cavity``
     - Step-by-step coupled system design
     - Target H-params
   * - ``explore_database``
     - Database exploration guide
     - —
   * - ``find_optimal_design``
     - Natural-language design search
     - ``parameter_description``
   * - ``gds_to_hamiltonian``
     - GDS -> Maxwell matrix -> transmon Hamiltonian parameters, compared with SQuADDS
     - ``gds_path``, ``qubit_type``, target f_q / alpha, resonator
   * - ``design_planar_capacitor_to_target``
     - Design a component of a planar capacitance family to target capacitances (GDS -> predict -> iterate)
     - ``component_class``, ``target_capacitances_fF``, ``tolerance_pct``
   * - ``retrain_capacitance_model``
     - Retrain on your own data on your own compute (label QC, shards, smoke fit, SLURM plan, scoring)
     - ``data_description``, ``mode``, ``cluster``
   * - ``contribute_capacitance_data_or_model``
     - Contribute data or a model through a Hugging Face pull request (dry run first)
     - ``kind``, ``path_or_repo``
   * - ``design_coupled_transmons``
     - Two coupled transmons with half-wave readout, targets -> Hamiltonian -> GDS -> prediction -> iteration -> SQuADDS resonators, with the validity caveats (the two-transmon example)
     - ``targets_GHz``, ``variant``, ``tolerance_pct``


Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``HF_TOKEN``
     - —
     - HuggingFace API token
   * - ``SQUADDS_MCP_TRANSPORT``
     - ``stdio``
     - Transport: ``stdio`` or ``streamable-http``
   * - ``SQUADDS_MCP_HOST``
     - ``0.0.0.0``
     - HTTP host (only for HTTP transport)
   * - ``SQUADDS_MCP_PORT``
     - ``8000``
     - HTTP port (only for HTTP transport)
   * - ``SQUADDS_MCP_SOURCES``
     - ``default,planar_capacitance``
     - ``SQuADDS_DB`` sources enabled at startup (``default`` = ``SQuADDS/SQuADDS_DB`` only)
   * - ``SQUADDS_PLANAR_CAP_DATA``
     - —
     - Local copy of the planar capacitance dataset
   * - ``SQUADDS_CAPML_MODELS``
     - —
     - Local planar capacitance model bundle (instead of the Hugging Face download)


Testing with MCP Inspector
--------------------------

.. code-block:: bash

   # Start the server in HTTP mode
   SQUADDS_MCP_TRANSPORT=streamable-http uv run squadds-mcp &

   # Connect with the Inspector
   npx -y @modelcontextprotocol/inspector
   # Then connect to http://localhost:8000/mcp


Further Documentation
---------------------

- Full MCP documentation: `MCP_README.md <https://github.com/LFL-Lab/SQuADDS/blob/master/MCP_README.md>`_
- Developer guide for extending the MCP server: `MCP_DEVELOPER_GUIDE.md <https://github.com/LFL-Lab/SQuADDS/blob/master/MCP_DEVELOPER_GUIDE.md>`_
- Model Context Protocol specification: `modelcontextprotocol.io <https://modelcontextprotocol.io>`_
