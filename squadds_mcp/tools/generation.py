"""
Qiskit-Metal Code Generation Tools.
================================

MCP tools that return standard boilerplate Python code snippets
for building components in Qiskit-Metal.

This allows agents to inject robust, predefined component creation logic
into the user's workspace, rather than writing error-prone code from scratch.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# Define the dictionary of code snippets
SNIPPETS = {
    "qubit": '''def create_qubit(name: str, design, opts: dict):
    from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
    opts["orientation"] = "-90"
    return TransmonCross(design, name, options=opts)
''',
    
    "clt": '''def create_clt_coupler(name: str, design, opts: dict):
    from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
    opts["orientation"] = "-90"
    return CoupledLineTee(design, name, options=opts)
''',
    
    "ncap": '''def create_ncap_coupler(name: str, design, opts: dict):
    from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee
    opts["orientation"] = "-90"
    return CapNInterdigitalTee(design, name, options=opts)
''',
    
    "cpw": '''def create_cpw(name: str, design, cplr_name: str, qubit_name: str, opts: dict):
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
    from qiskit_metal import Dict
    
    # Improve lead geometry to avoid kinks
    opts.update({"lead": Dict(start_straight="75um", end_straight="50um")})
    
    opts.update({
        "pin_inputs": Dict(
            start_pin=Dict(component=cplr_name, pin="second_end"),
            end_pin=Dict(component=qubit_name, pin="readout"), # Replace 'readout' with claw connection name if needed
        )
    })
    opts.update({"meander": Dict(spacing="100um")})
    return RouteMeander(design, name, options=opts)
''',

    "wirebond": '''def create_wirebond_port(name: str, design, pos_x: str, pos_y: str, orientation: str, trace_width: str, trace_gap: str):
    from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
    return LaunchpadWirebond(design, name, options=dict(
        pos_x=pos_x,
        pos_y=pos_y,
        orientation=orientation,
        trace_width=trace_width,
        trace_gap=trace_gap
    ))
''',

    "feedline": '''def create_feedline(name: str, design, start_pin: dict, end_pin: dict, trace_width: str, trace_gap: str):
    from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
    return RouteStraight(design, name, options=dict(
        pin_inputs=dict(
            start_pin=start_pin,
            end_pin=end_pin
        ),
        trace_width=trace_width,
        trace_gap=trace_gap
    ))
''',

    "charge_line_otg": '''def create_charge_line_otg(name: str, design, pos_x: str, pos_y: str, orientation: str):
    from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
    return OpenToGround(design, name, options=dict(
        pos_x=pos_x,
        pos_y=pos_y,
        orientation=orientation
    ))
''',

    "charge_line_route": '''def create_charge_line_route(name: str, design, start_pin: dict, end_pin: dict, anchors: dict):
    from qiskit_metal.qlibrary.tlines.anchored_path import RouteAnchors
    return RouteAnchors(design, name, options=dict(
        pin_inputs=dict(
            start_pin=start_pin,
            end_pin=end_pin
        ),
        anchors=anchors,
        fillet='80um' # Large fillet to avoid 90-degree corners
    ))
''',

    "airbridge": '''def add_airbridges(design, cpw_objects: list, cpw_width_um: float, cpw_gap_um: float):
    from squadds.components.airbridge.airbridge_generator import AirbridgeGenerator
    
    # Calculate crossover length
    crossover_length = f"{cpw_width_um + (2 * cpw_gap_um)}um"
    
    AirbridgeGenerator(
        design=design,
        target_comps=cpw_objects,
        crossover_length=[crossover_length],
        min_spacing=0.005,
        pitch=0.070,
        add_curved_ab=True
    )
'''
}

def register_generation_tools(mcp: FastMCP) -> None:
    """Register all code generation tools on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.tool()
    async def get_qiskit_metal_snippet(component: str) -> str:
        """Returns standard Qiskit-Metal boilerplate Python code for a requested component.
        
        The code is provided as a raw Python string that you can inject
        into the user's workspace.
        
        Use this tool repeatedly to assemble all necessary component methods 
        for a layout workflow.

        Args:
            component: The component to generate code for. Valid options:
                - "qubit": TransmonCross
                - "clt": CoupledLineTee
                - "ncap": CapNInterdigitalTee
                - "cpw": RouteMeander
                - "wirebond": LaunchpadWirebond
                - "feedline": RouteStraight
                - "charge_line_otg": OpenToGround
                - "charge_line_route": RouteAnchors
                - "airbridge": AirbridgeGenerator

        Returns:
            A string containing the Python definition for creating the component.
        """
        component = component.lower()
        if component not in SNIPPETS:
            valid_keys = ", ".join(SNIPPETS.keys())
            return f"Error: Unknown component '{component}'. Valid components are: {valid_keys}"
        
        return SNIPPETS[component]
