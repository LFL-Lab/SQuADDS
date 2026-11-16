"""
Design search & analysis tools.
================================

These tools wrap the SQuADDS ``Analyzer`` class to find the closest
pre-simulated designs matching target Hamiltonian parameters.

Workflow reminder (for AI agents):
    1. Use ``get_hamiltonian_param_keys`` to discover valid target params
    2. Use ``find_closest_designs`` to search for matching designs
    3. Use ``get_design_options`` / ``get_qubit_options`` / etc. to extract details
    4. ``predict_closest_designs_ml``: the closest single-component designs predicted with the planar capacitance models

Adding a New Analysis Tool
--------------------------
1. Define an ``async def`` function with ``@mcp.tool()`` decorator.
2. Access the DB via ``ctx.request_context.lifespan_context.db``.
3. Create a fresh ``Analyzer`` and configure it (the Analyzer should NOT
   be cached across calls since its state depends on search params).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from mcp.server.fastmcp import Context, FastMCP

from squadds_mcp.schemas import (
    ClosestDesignsResult,
    DesignResult,
    HamiltonianKeysResult,
)
from squadds_mcp.utils import extract_h_params_from_row, sanitize_for_json

ML_SYSTEMS = ("qubit", "coupler")


def closest_designs_ml(
    db: Any,
    system_type: str,
    target_params: dict[str, Any],
    num_results: int = 3,
    metric: str = "Euclidean",
    qubit: str = "TransmonCross",
    coupler: str = "GeneralizedCapConcentric",
    model: str = "QCap",
    threads: int | None = None,
    eps_r: float = 11.45,
    thickness_um: float = 350.0,
) -> dict[str, Any]:
    """Synchronous body of ``predict_closest_designs_ml``: ``Analyzer.find_closest`` then ``Analyzer.predict_capacitance``."""
    from squadds_mcp.tools.capacitance_ml import _ml_missing, _predictor

    if system_type not in ML_SYSTEMS:
        return {
            "error": f"system_type must be one of {ML_SYSTEMS}: the ML models predict single components. For a "
            "qubit-cavity row predict its qubit with predict_capacitance_from_design(component_class='TransmonCross', "
            "design_options=<the row's qubit options>)",
        }
    err = _ml_missing()
    if err:
        return err
    from squadds.core.analysis import Analyzer

    try:
        _configure_db_for_search(db, system_type, qubit, coupler=coupler)
        db.create_system_df()
        analyzer = Analyzer(db)
        closest = analyzer.find_closest(
            target_params=target_params, num_top=min(int(num_results), 20), metric=metric, display=False
        )
        pred = analyzer.predict_capacitance(
            model=model,
            threads=threads,
            predictor=_predictor(model, threads),
            eps_r=float(eps_r),
            thickness_um=float(thickness_um),
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    designs, rel, diff = [], [], []
    for rank, (idx, row) in enumerate(closest.iterrows(), start=1):
        p = pred.loc[idx]
        r = p["ml_result"]
        compare = {}
        for col, ml in p.items():
            key = col[:-3] if isinstance(col, str) and col.endswith("_ml") else None
            if key is None or key == "maxwell_matrix" or key not in row.index:
                continue
            try:
                stored, ml = float(row[key]), float(ml)
            except (TypeError, ValueError):
                continue
            compare[key] = {"stored_fF": stored, "ml_fF": ml, "diff_fF": ml - stored}
            if stored:
                compare[key]["diff_pct"] = 100 * (ml / stored - 1)
                rel.append(abs(ml / stored - 1))
            diff.append(ml - stored)
        designs.append(
            {
                "rank": rank,
                "design_id": row.get("design_id"),
                "design_options": row.get("design_options"),
                "stored_vs_ml": compare,
                "fidelity": p["fidelity"],
                "ml_total_s": p["ml_total_s"],
                "warnings": r["warnings"],
                "threads": r["prediction"].threads,
                "machine": r["prediction"].machine,
            }
        )
    out: dict[str, Any] = {
        "designs": designs,
        "model": model,
        "system_type": system_type,
        "target_params": target_params,
    }
    if diff:
        d = np.asarray(diff, float)
        out["agreement"] = {
            "entries": len(d),
            "median_abs_pct": 100 * float(np.median(rel)) if rel else None,
            "mae_fF": float(np.mean(np.abs(d))),
            "rmse_fF": float(np.sqrt(np.mean(d**2))),
            "note": "ML vs the stored simulation of the same rows; QCap was trained on all labelled data, which can "
            "include these rows: a consistency check, not a held-out score",
        }
    return sanitize_for_json(out)


def _configure_db_for_search(
    db: Any,
    system_type: str,
    qubit: str = "TransmonCross",
    cavity: str = "RouteMeander",
    resonator_type: str = "quarter",
    coupler: str = "GeneralizedCapConcentric",
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
    elif system_type == "coupler":
        db.select_system("coupler")
        db.select_coupler(coupler)
    else:
        raise ValueError(
            f"Invalid system_type '{system_type}'. Must be one of: 'qubit_cavity', 'qubit', 'cavity_claw', 'coupler'."
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
            system_type: One of 'qubit_cavity', 'qubit', 'cavity_claw', 'coupler'. For 'coupler' (the planar
                capacitance families, planar_capacitance source) the targets are capacitances in fF named
                ``<a>_to_<b>`` after the coupler's nets, e.g. ``north_to_south`` (concentric, interdigital),
                ``star_to_pad_0`` (star), ``inner_to_outer`` (shielded concentric).
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
            "coupler": ["north_to_south", "north_to_ground", "south_to_ground"],
        }
        if system_type not in key_map:
            raise ValueError(
                f"Invalid system_type '{system_type}'. Must be one of: 'qubit_cavity', 'qubit', 'cavity_claw', 'coupler'."
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
        coupler: str = "GeneralizedCapConcentric",
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
                - 'coupler': Standalone coupler of the planar capacitance families (targets: capacitances in fF,
                  e.g. {"north_to_south": 25.0}); uses the planar_capacitance source (on by default)
            target_params: Dict of Hamiltonian target values. Keys depend on system_type:
                - qubit: {qubit_frequency_GHz, anharmonicity_MHz}
                - cavity_claw: {cavity_frequency_GHz, kappa_kHz, resonator_type}
                - qubit_cavity: all of the above + {g_MHz}
            num_results: Number of closest designs to return (default 3, max 20).
            metric: Distance metric. One of: 'Euclidean', 'Manhattan', 'Chebyshev',
                    'WeightedEuclidean', 'Custom' (default: 'Euclidean').
            qubit: Qubit component name (default 'TransmonCross'; 'Xmon' and 'TransmonPocket' with the
                planar_capacitance source, using their Maxwell-matrix Hamiltonians).
            cavity: Cavity component name (default 'RouteMeander').
            resonator_type: 'quarter' or 'half' (default 'quarter').
            coupler: coupler component name for system_type='coupler' (e.g. 'GeneralizedCapConcentric',
                'GeneralizedCapStar', 'ShieldedCapConcentric', 'GeneralizedCapNInterdigital').
        """
        db = ctx.request_context.lifespan_context.db
        num_results = min(num_results, 20)  # Safety cap

        # Import Analyzer fresh (it reads from the singleton DB)
        from squadds.core.analysis import Analyzer

        _configure_db_for_search(db, system_type, qubit, cavity, resonator_type, coupler)

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
        if system_type == "coupler":
            h_param_keys = [k for k in target_params if k not in h_param_keys] + list(h_param_keys)
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
            for col in ["coupler_type", "contributor", "design_id"]:
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
                **({"coupler": coupler} if system_type == "coupler" else {}),
            },
        )

    @mcp.tool()
    async def predict_closest_designs_ml(
        ctx: Context,
        system_type: str,
        target_params: dict[str, Any],
        num_results: int = 3,
        metric: str = "Euclidean",
        qubit: str = "TransmonCross",
        coupler: str = "GeneralizedCapConcentric",
        model: str = "QCap",
        threads: int | None = None,
        eps_r: float = 11.45,
        thickness_um: float = 350.0,
    ) -> dict[str, Any]:
        """Find the closest pre-simulated designs and predict each with the planar capacitance models, next to the stored
        simulation (``Analyzer.find_closest`` then ``Analyzer.predict_capacitance``; needs ``squadds[ml]``).

        Single components only: ``system_type`` ``qubit`` (``qubit`` = ``TransmonCross``, or ``Xmon`` /
        ``TransmonPocket`` with the planar_capacitance source, on by default) or ``coupler`` (capacitance targets in
        fF); arguments as in ``find_closest_designs``. Each row is rebuilt
        in quantum-metal, exported and predicted with ``model`` (``QCap`` default or ``QCap-Fast``).

        Returns per row: design_id, design_options, every ``<a>_to_<b>`` capacitance stored vs ML (fF, difference in fF
        and %), fidelity, ML time with machine and threads; and the agreement over all rows (median |difference| % with
        MAE and RMSE in fF). For a design nobody simulated use ``predict_capacitance_from_design``.
        """
        db = ctx.request_context.lifespan_context.db
        return await asyncio.to_thread(
            closest_designs_ml,
            db,
            system_type,
            target_params,
            num_results,
            metric,
            qubit,
            coupler,
            model,
            threads,
            eps_r,
            thickness_um,
        )
