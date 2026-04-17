# SQuADDS MCP Server

> Model Context Protocol (MCP) server for the [SQuADDS](https://github.com/LFL-Lab/SQuADDS) superconducting quantum device design database.

This MCP server wraps the SQuADDS Python library, making the database of pre-simulated superconducting quantum device designs accessible to AI agents (Claude, Cursor, custom LLM agents) and human developers through the standardized [Model Context Protocol](https://modelcontextprotocol.io).

## Quick Start

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
# Run via the CLI entrypoint
uv run squadds-mcp

# Or run the module directly
uv run python -m squadds_mcp.server
```

### 4. Connect from an AI client

#### Claude Desktop / Claude Code

Add to your MCP settings (`claude_desktop_config.json` or via `claude mcp add`):

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

Or with Claude Code:
```bash
claude mcp add squadds -- uv run --directory /path/to/SQuADDS squadds-mcp
```

#### Cursor

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

#### HTTP Transport (for networked/remote usage)

```bash
SQUADDS_MCP_TRANSPORT=streamable-http SQUADDS_MCP_PORT=8000 uv run squadds-mcp
```

Then connect from any MCP client to `http://localhost:8000/mcp`.

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
