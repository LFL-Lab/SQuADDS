# SQuADDS MCP Server

> Model Context Protocol (MCP) server for the [SQuADDS](https://github.com/LFL-Lab/SQuADDS) superconducting quantum device design database.

This MCP server wraps the SQuADDS Python library, making the database of pre-simulated superconducting quantum device designs accessible to AI agents (Claude, Cursor, VS Code Copilot, Gemini, Codex, and custom LLM agents) and human developers through the standardized [Model Context Protocol](https://modelcontextprotocol.io).

## 🤖 Agent Setup (Copy-Paste This to Your AI Agent)

> **If you're using an AI coding assistant**, just paste this prompt into it to have it set up everything for you:

```
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
```

---

## 🧑‍💻 Manual Setup

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

### Analysis Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_hamiltonian_param_keys` | Valid target param keys | `system_type` |
| `find_closest_designs` | **Primary search tool** — find closest designs | `system_type`, `target_params`, `num_results`, `metric` |

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

---

## Available Prompts

| Prompt | Description | Parameters |
|--------|-------------|------------|
| `design_qubit_cavity` | Step-by-step coupled system design | Target H-params |
| `explore_database` | Database exploration guide | — |
| `find_optimal_design` | Natural-language design search | `parameter_description` |

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
│   └── contribution.py   # Contributor & device info tools
├── resources/
│   └── metadata.py       # Read-only data resources
└── prompts/
    └── workflows.py      # Guided workflow prompts
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
