# SQuADDS MCP Server

> Model Context Protocol (MCP) server for the [SQuADDS](https://github.com/LFL-Lab/SQuADDS) superconducting quantum device design database.

This MCP server wraps the SQuADDS Python library, making the database of pre-simulated superconducting quantum device designs accessible to AI agents (Claude, Cursor, VS Code Copilot, Gemini, Codex, and custom LLM agents) and human developers through the standardized [Model Context Protocol](https://modelcontextprotocol.io).

## Agent Setup (Copy-Paste This to Your AI Agent)

> **If you're using an AI coding assistant**, just paste this prompt into it to have it set up everything for you:

```
I need you to set up the SQuADDS MCP server so I can access the superconducting
qubit design database through you. Here's what to do:

1. Clone the repo and install:
   git clone https://github.com/LFL-Lab/SQuADDS.git
   cd SQuADDS
   uv sync --extra mcp --extra ml
   (drop `--extra ml` if you don't need the planar capacitance models)

2. Add the MCP server to your config. The command to run the server is:
   uv run --directory /path/to/SQuADDS squadds-mcp

3. Once connected, read the `squadds://guide` resource for a quick overview
   of available tools.

The server exposes these key tools:
- `list_components` / `list_datasets` — explore the database
- `find_closest_designs` — find designs matching target Hamiltonian parameters
- `interpolate_design` — get physics-interpolated designs
- `get_hamiltonian_param_keys` — discover valid search parameters
- `predict_capacitance_from_gds` — full Maxwell capacitance matrix of a GDS layout
  (read `squadds://planar-capacitance-guide` first)
- `capacitance_to_hamiltonian` — E_C, qubit frequency, anharmonicity and g from a matrix
- `query_planar_capacitance` / `nearest_planar_capacitance_designs` — the planar
  capacitance dataset
- `list_hosted_ml_models` / `predict_with_hosted_ml_model` — models on the hosted API

Typical target parameter ranges:
- qubit_frequency_GHz: 3–8
- anharmonicity_MHz: −500 to −50
- cavity_frequency_GHz: 5–12
- kappa_kHz: 10–1000
- g_MHz: 10–200

Please set this up and confirm you can access the SQuADDS tools.
```

---

## Manual Setup

### 1. Install dependencies

```bash
cd SQuADDS
uv sync --extra mcp
```

### 2. Set up HuggingFace authentication

```bash
# Option A: environment variable
export HF_TOKEN=your_token_here

# Option B: huggingface CLI login
uv run huggingface-cli login
```

### 3. Run the server

```bash
# Run via the CLI entrypoint (stdio mode — for local AI assistants)
uv run squadds-mcp

# Or run the module directly
uv run python -m squadds_mcp.server

# HTTP mode (for networked/remote usage)
SQUADDS_MCP_TRANSPORT=streamable-http SQUADDS_MCP_PORT=8000 uv run squadds-mcp
```

### 4. Connect from an AI client

<details>
<summary><strong>Claude Desktop</strong></summary>

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "squadds": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"],
      "env": {
        "HF_TOKEN": "your_token_here"
      }
    }
  }
}
```
</details>

<details>
<summary><strong>Claude Code</strong></summary>

```bash
claude mcp add squadds -- uv run --directory /path/to/SQuADDS squadds-mcp
```
</details>

<details>
<summary><strong>Cursor</strong></summary>

Add to `.cursor/mcp.json` in your project:
```json
{
  "mcpServers": {
    "squadds": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
    }
  }
}
```
</details>

<details>
<summary><strong>VS Code (Copilot)</strong></summary>

Add to `.vscode/settings.json`:
```json
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
```
</details>

<details>
<summary><strong>Antigravity (Gemini)</strong></summary>

Add to `~/.gemini/settings.json` (or project-level `.gemini/settings.json`):
```json
{
  "mcpServers": {
    "squadds": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
    }
  }
}
```
</details>

<details>
<summary><strong>Gemini CLI</strong></summary>

Add to `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "squadds": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
    }
  }
}
```
</details>

<details>
<summary><strong>OpenAI Codex CLI</strong></summary>

```bash
codex --mcp-config mcp.json
```

With `mcp.json`:
```json
{
  "mcpServers": {
    "squadds": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/SQuADDS", "squadds-mcp"]
    }
  }
}
```
</details>

<details>
<summary><strong>HTTP Transport (for any client)</strong></summary>

```bash
SQUADDS_MCP_TRANSPORT=streamable-http SQUADDS_MCP_PORT=8000 uv run squadds-mcp
```

Then connect from any MCP client to `http://localhost:8000/mcp`.
</details>

---

## Architecture

```
┌─────────────┐
│  AI Client   │  Claude Desktop, Cursor, Custom Agent
└──────┬──────┘
       │ MCP Protocol (stdio or HTTP)
┌──────▼──────────────────────────────┐
│         SQuADDS MCP Server          │
│  ┌────────┐ ┌──────────┐ ┌───────┐ │
│  │ Tools  │ │Resources │ │Prompts│ │
│  └────┬───┘ └────┬─────┘ └───┬───┘ │
│       │          │            │     │
│  ┌────▼──────────▼────────────▼───┐ │
│  │        SQuADDS Library         │ │
│  │  (SQuADDS_DB, Analyzer, etc.)  │ │
│  └──────────────┬─────────────────┘ │
└─────────────────┼───────────────────┘
                  │
         ┌────────▼────────┐
         │  HuggingFace Hub │
         │  (SQuADDS_DB)    │
         └─────────────────┘
```

---

## Available Tools

### Database Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_components` | List supported component types | — |
| `list_component_names` | List component names for a type | `component` |
| `list_configs` | List all dataset configurations | — |
| `list_datasets` | Overview of all datasets | — |
| `get_dataset_info` | Dataset metadata (features, size) | `component`, `component_name`, `data_type` |
| `get_dataset` | Load dataset rows (paginated) | `component`, `component_name`, `data_type`, `limit`, `offset` |
| `list_measured_devices` | All experimental devices | — |
| `get_simulation_results` | Simulation results for a device | `device_name` |
| `list_squadds_db_sources` | Enabled / available `SQuADDS_DB` sources and the configs each adds | — |
| `enable_squadds_db_source` | Enable an extra source (`planar_capacitance`: Xmon, TransmonPocket, concentric, star, shielded configs) for the following calls | `source`, `revision`, `local_dir` |
| `disable_squadds_db_source` | Disable an extra source | `source` |
| `list_data_types` | The data types (`cap_matrix`, `eigenmode`) and what each holds | — |
| `get_capacitance_data` | Capacitance-matrix rows of a component (shortcut for `get_dataset` with `cap_matrix`) | `component`, `component_name`, `limit`, `offset` |
| `get_resonator_info` | Quarter-wave vs half-wave resonators and the `resonator_type` to pass to the search tools | — |

### Analysis Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_hamiltonian_param_keys` | Valid target param keys | `system_type` |
| `find_closest_designs` | **Primary search tool** — find closest designs (`system_type="coupler"` with capacitance targets in fF for the planar capacitance couplers) | `system_type`, `target_params`, `num_results`, `metric`, `qubit`, `coupler` |
| `predict_closest_designs_ml` | Closest single-component designs (`qubit` / `coupler`) predicted with the planar capacitance models next to their stored simulation (`Analyzer.predict_capacitance`; needs `squadds[ml]`) | `system_type`, `target_params`, `num_results`, `qubit`, `coupler`, `model`, `eps_r`, `thickness_um` |

### Interpolation Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `interpolate_design` | Physics-scaled interpolated design | `target_params`, `qubit`, `cavity`, `resonator_type` |

### Contribution Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_reference_device` | Reference experimental device info | `component`, `component_name`, `data_type` |
| `get_fabrication_recipe` | Fabrication recipe for a device | `device_name` |
| `list_contributors` | All data contributors | — |

### Simulation and Layout Helpers

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_squadds_simulation_playbook` | JSON playbook for Q3D, HFSS eigenmode, sweeps and driven-modal workflows | `playbook_variant` (`summary` or `full`) |
| `get_drivenmodal_playbook_json` | JSON playbook for HFSS driven-modal workflows only | — |
| `get_maxwell_capacitance_conventions` | How to read an HFSS Maxwell matrix as pair capacitances without double counting | — |
| `get_qiskit_metal_snippet` | Qiskit Metal boilerplate for a component (qubit, CLT, cavity, ...) | `component` |

**Hosted ML API:** `list_hosted_ml_models` lists the models the SQuADDS ML API serves (`GET /v1/models`, with each model's inputs and units) and `predict_with_hosted_ml_model` runs one with JSON `inputs` and `options`. No `[ml]` extra is needed, and `SQUADDS_ML_API_URL` overrides the API URL.

### Planar Capacitance ML Tools

Need `pip install "squadds[ml]"` to run a model; without it every tool returns a clear message with the install command.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `list_capacitance_models` | Released models (`QCap` default, `QCap-Fast`) with reference accuracy (median %, MAE, RMSE in fF), time and warnings | — |
| `predict_capacitance_from_gds` | GDS file -> full N x N Maxwell matrix (fF, ground first), names, checks, timing, `sim_results` view; other substrates solve the prior at that stack (QCap; QCap-Fast rescales and warns); ground and conductors on one layer: the ground is detected (`ground_detection`), `ground_pieces` overrides it | `gds_path`, `model` (`QCap`), `threads`, `ground_layers`, `conductor_layers`, `eps_r`, `thickness_um`, `ground_pieces` |
| `query_planar_capacitance` | Designs of `SQuADDS/SQuADDS_PlanarCapacitance` via `PlanarCapacitanceDB` (optionally with labels); the same families appear in `SQuADDS_DB()` (the `planar_capacitance` source, on by default) | `family`, `n_conductors`, `split`, `collection`, `config`, `limit`, `offset`, `include_labels` |
| `get_planar_capacitance_conventions` | Units, Maxwell signs, ground first, never-merged nets, layer map, served stack, models and the QCap-Fast warning, priors, metric | — |
| `validate_capacitance_contribution` | Dataset validator on a local contribution folder (read-only) | `path`, `check_prior`, `reference` |
| `compute_capacitance_prior` | Physics-only Maxwell matrix of a GDS (optimized or coarse thin-film prior), a baseline for the model; returns each solve, the ground detection and the warnings | `gds_path`, `prior`, `threads`, `eps_r`, `thickness_um`, `ground_pieces` |
| `inspect_capacitance_layout` | The conductors and ground a prediction would read, without predicting: count, areas, ground area in the file vs read, numbered pieces and the detected ground for one-layer files, optional PNG | `gds_path`, `ground_layers`, `conductor_layers`, `ground_pieces`, `png_path` |
| `get_planar_capacitance_design` | One dataset design in detail: options, Q3D solve with QC flags and convergence, label, adaptive passes, stored priors, GDS path | `design_id`, `parts` |
| `nearest_planar_capacitance_designs` | Dataset designs geometrically closest to a design (pooled embedding; not a capacitance distance) | `design_id`, `k`, `family`, `split` |

Start with the resource `squadds://planar-capacitance-guide` (models, accuracy with MAE and RMSE, time, conventions, workflows).

#### Design: component, GDS and Hamiltonian

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_capacitance_component_info` | Families, Qiskit Metal classes, SQuADDS configs, net names; default options of one class | `component_class` |
| `generate_component_gds` | Component class + options -> GDS in the dataset export (path + layout readback) | `component_class`, `design_options`, `out_path`, `export` |
| `predict_capacitance_from_design` | Component class + options -> Maxwell matrix with SQuADDS net names (`MLSimulator`) | `component_class`, `design_options`, `model`, `save_gds_dir`, `eps_r`, `thickness_um` |
| `capacitance_to_hamiltonian` | Maxwell matrix (or a GDS) -> C_sigma, C_g, E_C, E_J/E_C, L_j, f_q, alpha, g (grounded or floating transmon) | `matrix_fF`, `names` or `gds_path` (with `model`, layers, `eps_r`, `thickness_um`, `ground_pieces`), `qubit_type`, `qubit_nodes`, `coupler`, `EJ_GHz` |
| `solve_coupled_transmon_hamiltonian` | Two coupled transmons + two half-wave resonators (the two-transmon example): targets -> E_J, E_C, L_J, g, J and the target 5 x 5 Maxwell matrix (full Hamiltonian, no RWA); evaluates a predicted layout matrix; resonator lines 50 ohm (fixed) | `targets`, `reference_matrix_fF`, `layout_matrix_fF` |
| `generate_two_transmon_gds` | The two-Xmon layout (gap or interdigital coupler) -> GDS, optionally with its predicted Maxwell matrix (outside the trained families) and a ground-area check | `p`, `variant`, `out_path`, `predict`, `model`, `eps_r`, `thickness_um` |
| `correct_half_wave_resonator` | A SQuADDS half-wave row adapted to your claw (the two-transmon example's resonator step): the row's claw capacitance (predicted), resonator length and kappa | `row_frequency_GHz`, `row_kappa_kHz`, `row_length_um`, `f_r_GHz`, `line_frequency_GHz`, `node_capacitance_fF`, `claw_options` |

#### Training, scoring and label QC

Long runs never execute inside the server: `plan_capacitance_training` writes a driver and a generic SLURM script for your own compute. In-process work is bounded (16 designs for shards, 500 steps for a smoke fit, 20 layouts for end-to-end scoring).

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `check_capacitance_label` | Label QC of a Q3D `cap_matrix.csv` or a matrix (symmetry, signs, closure, PD, units, passes) | `path` or `matrix_fF` + `names` |
| `build_capacitance_training_shards` | A few GDS + labels -> training shards (embedding + physics priors) | `designs`, `out_dir` |
| `plan_capacitance_training` | Driver + SLURM scripts for `build` / `fit` / `in_family` / `collective` retraining | `mode`, `out_dir`, `script_dir`, `shard_dir`, `designs_json`, `partition`, `account` |
| `run_capacitance_fit_smoke` | Short in-process fit (<= 500 steps) to check a shard dir end to end | `shard_dir`, `out_dir`, `steps` |
| `score_capacitance_model` | Validation score: median % with MAE and RMSE in fF, per family, with time, machine, threads (test scoring refused) | `model`, `data` or `layouts`, `split` |

#### Contributing data and models

Dry run by default: a pull request is opened only with `dry_run=False` **and** `confirm=True`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `init_capacitance_contribution` | Copy the contribution template (manifest.yaml, designs.csv, README) to a folder | `dest_dir` |
| `contribute_capacitance_data` | Validate a contribution folder and plan (or, with confirm, open) the dataset pull request | `path`, `dry_run`, `confirm` |
| `validate_capacitance_model` | Validate a model repository: checkpoints, sha256, metadata, validation score | `model_repo_id`, `revision`, `local_dir`, `val_data` |
| `contribute_capacitance_model` | Validate and plan (or, with confirm, open) the model-zoo pull request | `model_repo_id`, `dry_run`, `confirm` |

---

## Available Resources

| URI | Description |
|-----|-------------|
| `squadds://version` | SQuADDS + MCP server versions |
| `squadds://citation` | BibTeX citation |
| `squadds://components` | Supported component types |
| `squadds://configs` | Dataset configuration strings |
| `squadds://datasets` | Dataset summary table |
| `squadds://guide` | Quick reference for AI agents |
| `squadds://planar-capacitance-guide` | Planar capacitance models (QCap, QCap-Fast), accuracy (median % with MAE and RMSE in fF), time, conventions, workflows |
| `squadds://planar-capacitance-tutorials` | Tutorials 11-16 with the MCP tools that reproduce each |

---

## Available Prompts

| Prompt | Description | Parameters |
|--------|-------------|------------|
| `design_qubit_cavity` | Step-by-step coupled system design | Target H-params |
| `explore_database` | Database exploration guide | — |
| `find_optimal_design` | Natural-language design search | `parameter_description` |
| `gds_to_hamiltonian` | GDS -> Maxwell matrix -> transmon Hamiltonian parameters, compared with SQuADDS | `gds_path`, `qubit_type`, target f_q / alpha, resonator |
| `design_planar_capacitor_to_target` | Design a planar capacitance component to target capacitances (GDS -> predict -> iterate) | `component_class`, `target_capacitances_fF`, `tolerance_pct` |
| `retrain_capacitance_model` | Retrain on your own data on your own compute | `data_description`, `mode`, `cluster` |
| `contribute_capacitance_data_or_model` | Contribute data or a model via a Hugging Face pull request (dry run first) | `kind`, `path_or_repo` |
| `design_coupled_transmons` | Two coupled transmons with half-wave readout: targets -> Hamiltonian -> GDS -> prediction -> iteration -> SQuADDS resonators, with the validity caveats (the two-transmon example) | `targets_GHz`, `variant`, `tolerance_pct` |

---

## Example: Finding a Qubit-Cavity Design

Here's what an AI agent interaction looks like:

```
User: Design a transmon qubit coupled to a quarter-wave resonator.
      Target: qubit at 4.5 GHz, anharmonicity -220 MHz,
      cavity at 9.0 GHz, kappa 100 kHz, coupling 75 MHz.

Agent: I'll search the SQuADDS database for matching designs.
       [calls find_closest_designs with target_params]

Agent: Found 3 matching designs. The closest has:
       - Qubit frequency: 4.48 GHz (target: 4.5)
       - Anharmonicity: -218 MHz (target: -220)
       - Cross length: 310 um
       - Claw length: 86 um
       - Resonator length: 4200 um
       ...
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token |
| `SQUADDS_MCP_TRANSPORT` | `stdio` | Transport: `stdio` or `streamable-http` |
| `SQUADDS_MCP_HOST` | `0.0.0.0` | HTTP host (only for HTTP transport) |
| `SQUADDS_MCP_PORT` | `8000` | HTTP port (only for HTTP transport) |
| `SQUADDS_MCP_SOURCES` | `default,planar_capacitance` | `SQuADDS_DB` sources enabled at startup (`default` = `SQuADDS/SQuADDS_DB` only) |
| `SQUADDS_PLANAR_CAP_DATA` | — | Local copy of the planar capacitance dataset |
| `SQUADDS_CAPML_MODELS` | — | Local planar capacitance model bundle (instead of the Hugging Face download) |

---

## Testing with MCP Inspector

```bash
# Start the server
uv run squadds-mcp &

# Connect with the Inspector
npx -y @modelcontextprotocol/inspector
```

Then connect to `http://localhost:8000/mcp` (if HTTP) or use stdio mode in the Inspector.

---

## Project Structure

```
squadds_mcp/
├── __init__.py           # Package version
├── server.py             # FastMCP server, lifespan, CLI entrypoint
├── schemas.py            # Pydantic models for structured I/O
├── utils.py              # Serialization & formatting helpers
├── tools/
│   ├── database.py       # DB browsing & query tools
│   ├── analysis.py       # Design search tools
│   ├── interpolation.py  # Scaling interpolation tools
│   ├── contribution.py   # Contributor & device info tools
│   ├── capacitance_ml.py           # Planar capacitance models, priors, dataset, contribution validator
│   ├── capacitance_design.py       # Component -> GDS -> Maxwell matrix -> Hamiltonian
│   ├── capacitance_training.py     # Label QC, shards, retraining plans (SLURM), smoke fit, scoring
│   └── capacitance_contribution.py # Data / model pull requests (dry run by default)
├── resources/
│   ├── metadata.py                 # Read-only data resources
│   └── planar_capacitance_guide.py # Planar capacitance guide + tutorials index
└── prompts/
    ├── workflows.py                # Guided workflow prompts
    └── capacitance_workflows.py    # Planar capacitance workflow prompts
```

---

## Citation

If you use SQuADDS in your research, please cite:

```bibtex
@article{Shanto2024squaddsvalidated,
  doi = {10.22331/q-2024-09-09-1465},
  title = {{SQuADDS}: A validated design database and simulation workflow for superconducting qubit design},
  author = {Shanto, Sadman and Kuo, Andre and Miyamoto, Clark and Zhang, Haimeng and Maurya, Vivek and Vlachos, Evangelos and Hecht, Malida and Shum, Chung Wa and Levenson-Falk, Eli},
  journal = {Quantum},
  volume = {8},
  pages = {1465},
  year = {2024}
}
```
