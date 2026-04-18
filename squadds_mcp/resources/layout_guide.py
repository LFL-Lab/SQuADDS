"""
Layout knowledge resources.
============================

MCP Resources that provide domain knowledge about superconducting quantum
device layout best practices. These help AI agents produce physically
correct chip designs rather than geometrically broken ones.

These resources encode the hard-won layout heuristics from Tutorial 5
("Designing a fab-ready chip with SQuADDS") and standard CPW microwave
engineering knowledge.

Adding a New Resource
---------------------
1. Add a ``@mcp.resource("squadds://your_uri")`` decorated function
   inside ``register_layout_resources()``.
2. Resources should return static text — no heavy computation.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def register_layout_resources(mcp: FastMCP) -> None:
    """Register all layout knowledge resources on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.resource("squadds://layout-guide")
    async def get_layout_guide() -> str:
        """Comprehensive CPW layout best practices for superconducting quantum devices.

        This resource encodes the critical layout rules that prevent common
        geometry errors when building chips from SQuADDS design parameters.
        AI agents MUST read this before generating any qiskit-metal layout code.
        """
        return """# Superconducting Quantum Device Layout Best Practices

## Critical Rule: DO NOT Use QubitCavity for Production Designs

The `QubitCavity` class in `squadds.components.coupled_systems` is a convenience
wrapper for quick visualization. It has known geometry issues:
- Trace width mismatches at the claw-CPW connection
- Uncontrolled meander kinks
- No charge line or feedline support

**For any real design, build each component individually** using qiskit-metal
primitives (TransmonCross, CoupledLineTee, RouteMeander, RouteAnchors, etc.)
as shown in the `design_fab_ready_chip` prompt.

---

## 1. Trace Width Matching (CRITICAL)

Every CPW connection point must have matching trace widths on both sides.
Mismatches create impedance discontinuities that cause reflections and
unpredictable electromagnetic behavior.

**Key connections to check:**
- **Claw → Resonator**: The qubit claw's `claw_cpw_width` MUST equal the
  RouteMeander's `trace_width`. In the SQuADDS database, the resonator
  trace_width is typically `10um` and the claw_cpw_width should be `10um`.
- **Coupler → Resonator**: The CoupledLineTee's `second_width` should match
  the RouteMeander's `trace_width`.
- **Feedline → Launchpad**: The feedline's `trace_width` and `trace_gap`
  must match the LaunchpadWirebond's `trace_width` and `trace_gap`.

**Example (correct):**
```python
# Qubit claw CPW width matches resonator trace width
qubit_opts = dict(
    connection_pads=dict(
        c=dict(claw_cpw_width='10um', ...)  # Must match resonator
    ), ...
)
cpw_opts = Dict(trace_width='10um', ...)  # Same as claw_cpw_width
```

## 2. No Sharp Turns

Sharp 90° bends in CPW traces cause:
- **Radiation loss**: Microwave energy radiates at discontinuities
- **TLS loss**: Sharp corners concentrate electric fields in surface defects
- **Current crowding**: Reduces critical current and creates hot spots

**Rules:**
- Always set a `fillet` parameter on RouteMeander (typically `30–50um`)
- For charge lines and non-meander routes, use `RouteAnchors` with `fillet`
  parameter instead of `RouteMeander` — it avoids forced 90° turns
- Never use `RouteStraight` for paths that need bends

**Example (charge line with smooth turns):**
```python
route = RouteAnchors(design, 'charge_line', Dict(
    fillet='80um',
    anchors=anchor_dict,
    pin_inputs=Dict(
        start_pin=Dict(component='LP1', pin='tie'),
        end_pin=Dict(component='otg_1', pin='open')
    ),
    lead=Dict(start_straight='15um', end_straight='50um')
))
```

## 3. Meander Design Rules

RouteMeander creates meandering CPW lines to achieve a target total_length.
Poor meander parameters cause self-intersecting geometry.

**Rules:**
- `meander.spacing` should be ≥ 3× `trace_width` (minimum 100um typical)
- `fillet` must be < `meander.spacing / 2` (otherwise curves overlap)
- `meander.asymmetry` should be ~1/3 of the CLT `coupling_length` to
  avoid kinks at the coupler connection
- If kinks still appear, increase `lead.start_straight` and
  `lead.end_straight` (try 50–150um)

**Safe defaults:**
```python
cpw_opts = Dict(
    total_length='4000um',
    trace_width='10um',
    trace_gap='6um',
    fillet='30.9um',
    meander=Dict(spacing='100um', asymmetry='-50um'),
    lead=Dict(start_straight='75um', end_straight='50um'),
    pin_inputs=Dict(
        start_pin=Dict(component='clt1', pin='second_end'),
        end_pin=Dict(component='Q1', pin='c')
    )
)
```

## 4. Lead Straight Parameters (Kink Fix)

Kinks at meander endpoints are the #1 geometry issue. They happen when
the meander algorithm doesn't have enough straight-line runway before
the first bend.

**Fix:** Set `lead.start_straight` and `lead.end_straight`:
```python
for cpw in cpw_objects:
    cpw.options.lead.start_straight = '75um'
    cpw.options.lead.end_straight = '50um'
```

After changing, always `gui.rebuild()` and visually inspect.

## 5. Feedlines & Impedance Matching

- **Impedance Matching (50Ω)**: Feedline traces are usually matched to 50Ω.
  The required `trace_width` and `trace_gap` depend entirely on the user's
  layer stack (e.g., Si/Al vs Sapphire/Nb).
  👉 **Agent Action**: ALWAYS ask the user for their 50Ω `trace_width` and `trace_gap` dimensions for their specific substrate before creating a feedline.
- **Port Topology**: Feedlines can be configuration in different ways.
  👉 **Agent Action**: Ask the user:
  1. How many readout/feedline launchpads do they want?
  2. What are the rough positions (e.g., edge of the chip)?
  3. Are the readout lines pass-through (2 ports per line) or reflective (1 port and 1 `OpenToGround` or short)?
- **Resonator CPWs**: These are typically higher impedance (70–100Ω). The SQuADDS database options handle this. Do not mix matched feedline dimensions with resonator CPW dimensions.

## 6. Component Placement

**Qubit orientation:**
- Qubits on opposite sides of the feedline should have mirrored
  orientations (e.g., '-90' on left, '90' on right)
- JJ flip should match: `JJ_flip=False` for left-side, `True` for
  right-side, so all JJs face the same direction for single-write
  angle evaporation

**Coupler placement:**
- CoupledLineTee (CLT) couplers sit on the feedline
- Their `pos_y` should match the qubit's `pos_y`
- Orientation should match the qubit's orientation

## 7. GDS Export Checklist

Before exporting:
1. Assign correct layer numbers per your foundry:
   ```python
   for name, comp in design.components.items():
       comp.options.layer = str(metal_layer)
   ```
2. Enable ground plane holes (cheese pattern):
   ```python
   a_gds.options.cheese.cheese_0_x = '25um'
   a_gds.options.cheese.cheese_0_y = '25um'
   a_gds.options.cheese.edge_nocheese = '200um'
   ```
3. Set keepout (no-cheese) buffer around traces:
   ```python
   a_gds.options.no_cheese.buffer = '100um'
   ```
4. Run DRC before sending to fab

## 8. Airbridges

Airbridges are critical for preventing slot-line modes on CPW meanders and feedlines.
- Use `AirbridgeGenerator` from `squadds.components.airbridge.airbridge_generator`.
- Calculate `crossover_length` = `trace_width` + (2 × `trace_gap`).
- Call `AirbridgeGenerator(design, target_comps=[meander_comp], crossover_length=[crossover_length], pitch=0.070, add_curved_ab=True)`

## 9. Junction Geometry

When computing Josephson junction dimensions (specifically for **Dolan style JJs**):
- $L_J = \\Phi_0 / (2\\pi I_c)$ where $I_c = J_c \\times l_{JJ} \\times w_{JJ}$
- Round $w_{JJ}$ to your foundry's minimum resolution (often 10nm steps)
- Enforce minimum width: `jj_width = max(computed_width, foundry_min)`
"""

    @mcp.resource("squadds://chip-design-reference")
    async def get_chip_design_reference() -> str:
        """Step-by-step chip design reference based on the SQuADDS Tutorial 5 workflow.

        This encodes the complete design flow from SQuADDS parameters
        to a fab-ready GDS file, using individual qiskit-metal components.
        """
        return """# Chip Design Reference (Tutorial 5 Workflow)

## Overview

This reference describes the complete workflow for going from target
Hamiltonian parameters to a fab-ready GDS file. It follows the approach
in SQuADDS Tutorial 5 ("Designing a fab-ready chip with SQuADDS").

**Key principle:** Build each component individually with qiskit-metal
primitives. Do NOT use the `QubitCavity` convenience class for production.

---

## Phase 1: Get Design Parameters from SQuADDS

```python
from squadds import Analyzer, SQuADDS_DB

db = SQuADDS_DB()
db.select_system(["cavity_claw", "qubit"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("quarter")
df = db.create_system_df()

analyzer = Analyzer(db)
results = analyzer.find_closest(target_params=target_params, num_top=1)

# Extract structured options
data_qubit = analyzer.get_qubit_options(results)   # claw_gap, claw_length, etc.
data_cpw = analyzer.get_cpw_options(results)       # trace_width, trace_gap, total_length
data_coupler = analyzer.get_coupler_options(results) # coupling_length, coupling_space, etc.
LJs = analyzer.get_Ljs(results)                    # Josephson inductance in nH
```

## Phase 2: Compute Junction Geometry

Given your foundry's constraints (JJ length, current densities, min width) for a **Dolan style** junction:

```python
import numpy as np

def compute_JJ_width(Lj_nH, current_density_uA_um2, JJ_length_nm):
    Lj = Lj_nH * 1e-9
    Jc = current_density_uA_um2 * 1e-6
    l_jj = JJ_length_nm * 1e-3  # nm to um
    phi_0 = 2.067833848e-15
    Ic = phi_0 / (2 * np.pi * Lj)
    return (Ic / (Jc * l_jj)) * 1e3  # in nm
```

Round to foundry resolution and enforce minimum width.

## Phase 3: Build Design in Qiskit-Metal

### Required imports:
```python
from collections import OrderedDict
import qiskit_metal as metal
from qiskit_metal import Dict
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.tlines.anchored_path import RouteAnchors
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from squadds.components.qubits import TransmonCross
```

### Build order:
1. Create `DesignPlanar` with chip dimensions
2. Place feedline launchpads + RouteStraight feedline
3. Place charge line launchpads
4. For each qubit-resonator pair:
   a. Create `TransmonCross` with SQuADDS qubit options
   b. Create `CoupledLineTee` on feedline with SQuADDS coupler options
   c. Create `RouteMeander` connecting CLT to qubit with SQuADDS CPW options
   d. Fix kinks: set `lead.start_straight='75um'`, `lead.end_straight='50um'`
5. Add charge lines with `RouteAnchors` (NOT RouteMeander)
6. `gui.rebuild()` and visually inspect

### Critical trace-width rule:
The qubit connection pad's `claw_cpw_width` must equal the RouteMeander's
`trace_width`. Both should typically be `10um` in SQuADDS designs.

## Phase 4: Add Airbridges

Add airbridges to CPW meanders to suppress parasitic modes.
```python
from squadds.components.airbridge.airbridge_generator import AirbridgeGenerator

cpw_width = float(design.parse_value(cpw_opts.trace_width))
cpw_gap = float(design.parse_value(cpw_opts.trace_gap))
crossover_length = cpw_width + (2 * cpw_gap)

AirbridgeGenerator(
    design=design,
    target_comps=cpw_objects,
    crossover_length=[crossover_length],
    min_spacing=0.005,
    pitch=0.070,
    add_curved_ab=True
)
gui.rebuild()
```

## Phase 5: Export GDS

1. Assign metal layers per foundry spec
2. Configure cheese (ground plane holes) and keepout regions
3. Export: `a_gds.export_to_gds("filename.gds")`
4. Run DRC checks
"""
