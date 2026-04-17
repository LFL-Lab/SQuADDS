"""
Design search & analysis tools.
================================

These tools wrap the SQuADDS ``Analyzer`` class to find the closest
pre-simulated designs matching target Hamiltonian parameters.

Workflow reminder (for AI agents):
    1. Use ``get_hamiltonian_param_keys`` to discover valid target params
    2. Use ``find_closest_designs`` to search for matching designs
    3. Use ``get_design_options`` / ``get_qubit_options`` / etc. to extract details

Adding a New Analysis Tool
--------------------------
1. Define an ``async def`` function with ``@mcp.tool()`` decorator.
2. Access the DB via ``ctx.request_context.lifespan_context.db``.
3. Create a fresh ``Analyzer`` and configure it (the Analyzer should NOT
   be cached across calls since its state depends on search params).
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from squadds_mcp.schemas import (
    ClosestDesignsResult,
    DesignResult,
    HamiltonianKeysResult,
)
from squadds_mcp.utils import extract_h_params_from_row, sanitize_for_json


def _configure_db_for_search(
    db: Any,
    system_type: str,
    qubit: str = "TransmonCross",
    cavity: str = "RouteMeander",
    resonator_type: str = "quarter",
) -> None:
    """Configure a SQuADDS_DB instance for a design search.

    Resets all prior selections and sets up the system, qubit, cavity,
    and resonator type. This is a stateful mutation of the singleton DB instance.
    """
    db.unselect_all()

    if system_type == "qubit_cavity":
        db.select_system(["qubit", "cavity_claw"])
        db.select_qubit(qubit)
        db.select_cavity_claw(cavity)
        db.select_resonator_type(resonator_type)
    elif system_type == "qubit":
        db.select_system("qubit")
        db.select_qubit(qubit)
    elif system_type == "cavity_claw":
        db.select_system("cavity_claw")
        db.select_cavity_claw(cavity)
        db.select_resonator_type(resonator_type)
    else:
        raise ValueError(
            f"Invalid system_type '{system_type}'. Must be one of: 'qubit_cavity', 'qubit', 'cavity_claw'."
        )


def register_analysis_tools(mcp: FastMCP) -> None:
    """Register all design-search/analysis tools on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.tool()
    async def get_hamiltonian_param_keys(
        ctx: Context,
        system_type: str,
    ) -> HamiltonianKeysResult:
        """Get the valid target Hamiltonian parameter keys for a system type.

        Use the returned keys when constructing ``target_params`` for
        ``find_closest_designs``.

        Args:
            system_type: One of 'qubit_cavity', 'qubit', 'cavity_claw'.
        """
        key_map = {
            "qubit": ["qubit_frequency_GHz", "anharmonicity_MHz"],
            "cavity_claw": ["resonator_type", "cavity_frequency_GHz", "kappa_kHz"],
            "qubit_cavity": [
                "qubit_frequency_GHz",
                "anharmonicity_MHz",
                "cavity_frequency_GHz",
                "kappa_kHz",
                "g_MHz",
                "resonator_type",
            ],
        }
        if system_type not in key_map:
            raise ValueError(
                f"Invalid system_type '{system_type}'. Must be one of: 'qubit_cavity', 'qubit', 'cavity_claw'."
            )
        return HamiltonianKeysResult(keys=key_map[system_type], system_type=system_type)

    @mcp.tool()
    async def find_closest_designs(
        ctx: Context,
        system_type: str,
        target_params: dict[str, Any],
        num_results: int = 3,
        metric: str = "Euclidean",
        qubit: str = "TransmonCross",
        cavity: str = "RouteMeander",
        resonator_type: str = "quarter",
    ) -> ClosestDesignsResult:
        """Find the closest pre-simulated designs matching target Hamiltonian parameters.

        This is the **primary design search tool**. It:
        1. Configures the SQuADDS database for the specified system
        2. Loads the merged dataset
        3. Computes Hamiltonian parameters for all entries
        4. Ranks entries by distance to your targets using the chosen metric

        **Note**: First call may be slow (10–60s) as it downloads and processes
        the dataset. Subsequent calls with the same system config are faster.

        Args:
            system_type: Type of system. One of:
                - 'qubit_cavity': Coupled qubit + resonator (requires g_MHz, qubit & cavity params)
                - 'qubit': Standalone qubit (requires qubit_frequency_GHz, anharmonicity_MHz)
                - 'cavity_claw': Standalone cavity (requires cavity_frequency_GHz, kappa_kHz)
            target_params: Dict of Hamiltonian target values. Keys depend on system_type:
                - qubit: {qubit_frequency_GHz, anharmonicity_MHz}
                - cavity_claw: {cavity_frequency_GHz, kappa_kHz, resonator_type}
                - qubit_cavity: all of the above + {g_MHz}
            num_results: Number of closest designs to return (default 3, max 20).
            metric: Distance metric. One of: 'Euclidean', 'Manhattan', 'Chebyshev',
                    'WeightedEuclidean', 'Custom' (default: 'Euclidean').
            qubit: Qubit component name (default 'TransmonCross').
            cavity: Cavity component name (default 'RouteMeander').
            resonator_type: 'quarter' or 'half' (default 'quarter').
        """
        db = ctx.request_context.lifespan_context.db
        num_results = min(num_results, 20)  # Safety cap

        # Import Analyzer fresh (it reads from the singleton DB)
        from squadds.core.analysis import Analyzer

        _configure_db_for_search(db, system_type, qubit, cavity, resonator_type)

        # Build the system DataFrame
        db.create_system_df()

        # Create analyzer and search
        analyzer = Analyzer(db)
        closest_df = analyzer.find_closest(
            target_params=target_params,
            num_top=num_results,
            metric=metric,
            display=False,
        )

        # Extract results
        h_param_keys = analyzer.H_param_keys or []
        designs = []
        for rank, (_idx, row) in enumerate(closest_df.iterrows(), start=1):
            # Design options
            design_opts = {}
            if "design_options" in row.index and row["design_options"] is not None:
                design_opts = sanitize_for_json(row["design_options"])

            # H-params
            h_params = extract_h_params_from_row(row, h_param_keys)

            # Metadata
            meta = {}
            for col in ["coupler_type", "contributor"]:
                if col in row.index:
                    meta[col] = sanitize_for_json(row[col])

            designs.append(
                DesignResult(
                    rank=rank,
                    design_options=design_opts,
                    hamiltonian_params=h_params,
                    metadata=meta,
                )
            )

        return ClosestDesignsResult(
            designs=designs,
            num_results=len(designs),
            target_params=sanitize_for_json(target_params),
            system_config={
                "system_type": system_type,
                "qubit": qubit,
                "cavity": cavity,
                "resonator_type": resonator_type,
                "metric": metric,
            },
        )
