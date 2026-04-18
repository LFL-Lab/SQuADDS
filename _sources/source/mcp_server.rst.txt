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
     - **Primary search tool** — find closest designs
     - ``system_type``, ``target_params``, ``num_results``, ``metric``

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
