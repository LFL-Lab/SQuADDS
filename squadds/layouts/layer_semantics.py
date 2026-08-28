"""Published GDS layer-role contract for the SQuADDS layout registry."""

from __future__ import annotations

import json
from pathlib import Path

LAYER_SEMANTICS_SCHEMA_VERSION = "1.3.0"
PUBLISHED_ROLE_PROFILE = "published-v0"
PORT_COMPLETE_ROLE_PROFILE = "layer-semantics-v1.3"

LAYER_SEMANTICS = {
    "schema_version": LAYER_SEMANTICS_SCHEMA_VERSION,
    "units": "um",
    "components": {
        "GeneralizedCapNInterdigital": [
            {"layer": 1, "datatype": 10, "semantic": "signal_conductors"},
            {"layer": 1, "datatype": 0, "semantic": "ground_simulation_domain"},
            {"layer": 2, "datatype": 0, "semantic": "north_port"},
            {"layer": 3, "datatype": 0, "semantic": "south_port"},
        ],
        "CapNInterdigitalTee": [
            {"layer": 1, "datatype": 10, "semantic": "signal_conductors"},
            {"layer": 1, "datatype": 0, "semantic": "ground_simulation_domain"},
            {"layer": 2, "datatype": 0, "semantic": "prime_top_port"},
            {"layer": 3, "datatype": 0, "semantic": "second_bottom_port"},
        ],
        "CavityClawRouteMeander": [
            {"layer": 1, "datatype": 10, "semantic": "signal_conductors"},
            {"layer": 1, "datatype": 11, "semantic": "etch_cutout"},
        ],
        "TransmonCross": [
            {"layer": 1, "datatype": 10, "semantic": "signal_conductors"},
            {"layer": 1, "datatype": 0, "semantic": "ground_simulation_domain"},
            {"layer": 2, "datatype": 0, "semantic": "cross_junction_port"},
            {"layer": 3, "datatype": 0, "semantic": "readout_claw_port"},
        ],
    },
}

_FUNCTIONAL_SEMANTIC_ROLES = {
    "signal_conductors": "conductor",
    "etch_cutout": "etch",
    "north_port": "port",
    "south_port": "port",
    "prime_top_port": "port",
    "second_bottom_port": "port",
    "cross_junction_port": "port",
    "readout_claw_port": "port",
}


def functional_layer_roles(component_name: str) -> dict[tuple[int, int], str]:
    """Return conductor/etch/port roles from the versioned dataset contract.

    The simulation-domain ground is deliberately omitted because v0/v1 crop
    and rasterize only functional component geometry.
    """
    try:
        entries = LAYER_SEMANTICS["components"][component_name]
    except KeyError as exc:
        choices = ", ".join(sorted(LAYER_SEMANTICS["components"]))
        raise ValueError(f"No layer semantics for {component_name!r}; choose one of: {choices}.") from exc
    return {
        (int(entry["layer"]), int(entry["datatype"])): _FUNCTIONAL_SEMANTIC_ROLES[entry["semantic"]]
        for entry in entries
        if entry["semantic"] in _FUNCTIONAL_SEMANTIC_ROLES
    }


def write_layer_semantics(path: Path) -> None:
    """Write the layer contract atomically using the registry filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(LAYER_SEMANTICS, indent=2) + "\n")
    temporary.replace(path)
