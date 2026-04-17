"""
Physics-based interpolation tools.
===================================

These tools wrap the ``ScalingInterpolator`` to produce scaled designs
that match target Hamiltonian parameters more closely than nearest-neighbor
lookup alone.

The interpolation workflow:
    1. Find the closest designs via ``find_closest_designs``
    2. Scale qubit/cavity/coupler dimensions based on physics relationships
    3. Return an interpolated design that better matches targets

Adding a New Interpolation Tool
-------------------------------
1. Add your ``@mcp.tool()`` function inside ``register_interpolation_tools()``.
2. Follow the pattern: configure DB → create Analyzer → run interpolation.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from squadds_mcp.schemas import InterpolatedDesignResult
from squadds_mcp.tools.analysis import _configure_db_for_search
from squadds_mcp.utils import sanitize_for_json


def register_interpolation_tools(mcp: FastMCP) -> None:
    """Register interpolation tools on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.tool()
    async def interpolate_design(
        ctx: Context,
        target_params: dict[str, Any],
        qubit: str = "TransmonCross",
        cavity: str = "RouteMeander",
        resonator_type: str = "quarter",
    ) -> InterpolatedDesignResult:
        """Get a physics-interpolated design for exact target Hamiltonian parameters.

        Unlike ``find_closest_designs`` which returns the nearest pre-simulated entry,
        this tool **scales** geometric dimensions (cross length, claw length,
        resonator length, coupling length) based on physics relationships to
        produce a design that should more closely match your target parameters.

        **Important**: This tool only works for coupled qubit-cavity systems.
        The scaling relationships are derived from transmon cross + meander cavity physics.

        **Note**: This calls ``find_closest_designs`` internally, so the first
        invocation may be slow if the dataset hasn't been loaded yet.

        Args:
            target_params: Target Hamiltonian parameters. Required keys:
                - qubit_frequency_GHz: Target qubit frequency (GHz)
                - anharmonicity_MHz: Target anharmonicity (MHz, usually negative)
                - cavity_frequency_GHz: Target cavity frequency (GHz)
                - kappa_kHz: Target linewidth (kHz)
                - g_MHz: Target coupling strength (MHz)
                - resonator_type: 'quarter' or 'half'
            qubit: Qubit component name (default 'TransmonCross').
            cavity: Cavity component name (default 'RouteMeander').
            resonator_type: 'quarter' or 'half' (default 'quarter').
        """
        db = ctx.request_context.lifespan_context.db

        from squadds.core.analysis import Analyzer
        from squadds.interpolations.physics import ScalingInterpolator

        # Configure DB for coupled system search
        _configure_db_for_search(db, "qubit_cavity", qubit, cavity, resonator_type)
        db.create_system_df()

        # Create analyzer with H-params computed
        analyzer = Analyzer(db)

        # Ensure resonator_type is in target_params
        if "resonator_type" not in target_params:
            target_params["resonator_type"] = resonator_type

        # Run interpolation
        interpolator = ScalingInterpolator(analyzer, target_params)
        result_df = interpolator.get_design()
        row = result_df.iloc[0]

        # Extract results
        design_options = sanitize_for_json(row.get("design_options", {}))
        qubit_options = sanitize_for_json(row.get("design_options_qubit", {}))
        cavity_options = sanitize_for_json(row.get("design_options_cavity_claw", {}))
        coupler_type = row.get("coupler_type", None)

        return InterpolatedDesignResult(
            design_options=design_options,
            qubit_options=qubit_options,
            cavity_options=cavity_options,
            coupler_type=coupler_type,
        )
