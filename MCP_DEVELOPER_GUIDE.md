# SQuADDS MCP Server — Developer Guide

This guide explains how to extend the MCP server with new tools, resources, and prompts. It's written for both **humans** and **AI agents** contributing to the codebase.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Adding a New Tool](#adding-a-new-tool)
- [Adding a New Resource](#adding-a-new-resource)
- [Adding a New Prompt](#adding-a-new-prompt)
- [Working with the Database Context](#working-with-the-database-context)
- [Serialization Rules](#serialization-rules)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Common Patterns](#common-patterns)

---

## Architecture Overview

```
squadds_mcp/
├── server.py          ← Central orchestrator (lifespan, registration)
├── schemas.py         ← Pydantic models (input/output contracts)
├── utils.py           ← Shared helpers (serialization, formatting)
├── tools/             ← MCP Tools (actions that do work)
│   ├── database.py    ← DB browsing & queries
│   ├── analysis.py    ← Design search
│   ├── interpolation.py ← Physics interpolation
│   └── contribution.py  ← Contributor info
├── resources/         ← MCP Resources (read-only data)
│   └── metadata.py
└── prompts/           ← MCP Prompts (workflow templates)
    └── workflows.py
```

**Key principles:**
1. **Tools** = actions (like POST endpoints). They take parameters and do work.
2. **Resources** = data (like GET endpoints). Read-only, addressed by URI.
3. **Prompts** = templates. Return instruction strings for AI agents.
4. **Schemas** = contracts. Pydantic models for type safety and documentation.
5. **Lifespan** = setup/teardown. The DB is initialized once and shared.

---

## Adding a New Tool

### Step 1: Define the output schema (if needed)

In `schemas.py`, add a Pydantic model:

```python
class MyToolResult(BaseModel):
    """Description of what this result contains."""

    field_name: str = Field(description="What this field means.")
    data: dict[str, Any] = Field(description="The actual data.")
```

### Step 2: Create the tool function

In the appropriate `tools/*.py` file (or create a new one), add your tool inside the `register_*_tools()` function:

```python
def register_my_tools(mcp: FastMCP) -> None:
    """Register my new tools."""

    @mcp.tool()
    async def my_new_tool(
        ctx: Context,
        param1: str,
        param2: int = 10,
    ) -> MyToolResult:
        """Clear description of what this tool does.

        This docstring becomes the tool's description in MCP.
        AI agents read this to decide when to use the tool.
        Be specific about:
        - What the tool does
        - What parameters mean
        - What the output contains
        - Any side effects or caveats

        Args:
            param1: Description of param1.
            param2: Description of param2 (default: 10).
        """
        # Access the database via lifespan context
        db = ctx.request_context.lifespan_context.db

        # Do your work...
        result = db.some_method(param1)

        # Return a structured result
        return MyToolResult(
            field_name=param1,
            data=sanitize_for_json(result),
        )
```

### Step 3: Register in server.py

In `server.py` → `create_server()`, add:

```python
from squadds_mcp.tools.my_module import register_my_tools
register_my_tools(mcp)
```

### Step 4: Write a test

In `tests/test_mcp_tools.py`, add a test for your tool.

---

## Adding a New Resource

Resources are simpler than tools — they're addressed by URI and take no parameters (except URI template variables).

In `resources/metadata.py`:

```python
@mcp.resource("squadds://my_data")
async def get_my_data(ctx: Context) -> str:
    """Description of this resource."""
    db = ctx.request_context.lifespan_context.db
    data = db.some_read_only_method()
    return json.dumps(data, indent=2)
```

For URI templates with variables:

```python
@mcp.resource("squadds://components/{component}/names")
async def get_component_names(component: str, ctx: Context) -> str:
    """Get component names for a given type."""
    db = ctx.request_context.lifespan_context.db
    names = db.get_component_names(component)
    return json.dumps(names or [], indent=2)
```

---

## Adding a New Prompt

Prompts return instruction strings. They can accept parameters.

In `prompts/workflows.py`:

```python
@mcp.prompt()
def my_workflow(target_frequency: float = 5.0) -> str:
    """Brief description of this workflow prompt.

    Detailed explanation of when an agent should use this.
    """
    return f"""# My Workflow

## Target: {target_frequency} GHz

### Step 1: ...
Call `tool_name(...)` with ...

### Step 2: ...
Review the output and ...
"""
```

---

## Working with the Database Context

Every tool and resource function can access the shared `SQuADDS_DB` instance:

```python
async def my_tool(ctx: Context) -> ...:
    db = ctx.request_context.lifespan_context.db

    # Now use db methods:
    db.supported_components()
    db.get_component_names("qubit")
    db.get_dataset(data_type="cap_matrix", component="qubit", component_name="TransmonCross")
    # etc.
```

**Important**: The `SQuADDS_DB` is a singleton with mutable state (selected_system, selected_qubit, etc.). When building multi-step tools (like `find_closest_designs`), always call `db.unselect_all()` first to reset state.

---

## Serialization Rules

SQuADDS data contains numpy arrays, pandas objects, and nested dicts that aren't JSON-safe. Always use the utilities in `utils.py`:

```python
from squadds_mcp.utils import sanitize_for_json, dataframe_to_records

# For arbitrary objects with numpy types:
safe_data = sanitize_for_json(raw_data)

# For DataFrames with pagination:
rows = dataframe_to_records(df, limit=50, offset=0)
```

What `sanitize_for_json` handles:
- `numpy.ndarray` → `list`
- `numpy.int64` → `int`
- `numpy.float64` → `float`
- `NaN` / `Inf` → `None`
- `pd.Timestamp` → ISO string
- `bytes` → UTF-8 string
- `set` → `list`

---

## Error Handling

Tools should handle errors gracefully and return meaningful messages:

```python
@mcp.tool()
async def my_tool(ctx: Context, component: str) -> ComponentListResult:
    db = ctx.request_context.lifespan_context.db

    # Validate inputs
    supported = db.supported_components()
    if component not in supported:
        raise ValueError(
            f"Component '{component}' not supported. "
            f"Available: {supported}"
        )

    # If a non-critical operation fails, return a useful default
    try:
        names = db.get_component_names(component)
    except Exception as e:
        return ComponentListResult(items=[], count=0)

    return ComponentListResult(items=names, count=len(names))
```

---

## Testing

### Unit tests

```bash
# Run all MCP tests
uv run pytest tests/test_mcp_tools.py tests/test_mcp_schemas.py -v
```

### MCP Inspector (interactive)

```bash
# Start server in HTTP mode
SQUADDS_MCP_TRANSPORT=streamable-http uv run squadds-mcp &

# Launch inspector
npx -y @modelcontextprotocol/inspector
# Connect to http://localhost:8000/mcp
```

### Programmatic testing

```python
from squadds_mcp.server import create_server

server = create_server()
# The server object can be introspected for registered tools, etc.
```

---

## Common Patterns

### Pattern 1: Paginated dataset access

Always paginate large datasets to avoid overwhelming AI context windows:

```python
@mcp.tool()
async def get_data(ctx: Context, limit: int = 50, offset: int = 0) -> DatasetResult:
    limit = min(limit, 200)  # Safety cap
    df = load_data()
    rows = dataframe_to_records(df, limit=limit, offset=offset)
    return DatasetResult(rows=rows, total_rows=len(df), offset=offset, limit=limit, ...)
```

### Pattern 2: Database state reset

Always reset DB state before configuring for a new search:

```python
db.unselect_all()
db.select_system(...)
db.select_qubit(...)
db.create_system_df()
```

### Pattern 3: Fresh Analyzer per search

The `Analyzer` reads from the singleton DB. Create a fresh one for each search to avoid stale state:

```python
from squadds.core.analysis import Analyzer

db.unselect_all()
# ... configure db ...
db.create_system_df()
analyzer = Analyzer(db)  # Fresh instance
result = analyzer.find_closest(...)
```

### Pattern 4: Structured output with Pydantic

Always define a Pydantic model for tool results. This enables:
- Automatic JSON Schema for MCP clients
- Type validation
- Clear documentation

```python
class MyResult(BaseModel):
    """What this result contains."""
    value: float = Field(description="The computed value.")

@mcp.tool()
async def compute(ctx: Context, x: float) -> MyResult:
    return MyResult(value=x * 2)
```

---

## File-by-File Reference

| File | Purpose | When to modify |
|------|---------|----------------|
| `server.py` | Server factory + CLI | Adding new tool modules |
| `schemas.py` | Pydantic models | Adding new tool I/O types |
| `utils.py` | Shared helpers | Adding serialization logic |
| `tools/database.py` | DB browsing tools | Adding dataset-related tools |
| `tools/analysis.py` | Design search tools | Adding search/analysis tools |
| `tools/interpolation.py` | Interpolation tools | Adding interpolation methods |
| `tools/contribution.py` | Contributor tools | Adding contributor/device tools |
| `resources/metadata.py` | Read-only resources | Adding static data resources |
| `prompts/workflows.py` | Prompt templates | Adding guided workflows |
