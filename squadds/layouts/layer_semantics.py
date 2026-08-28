"""Published GDS layer-role contract for the SQuADDS layout registry."""

from __future__ import annotations

import json
from pathlib import Path

LAYER_SEMANTICS_SCHEMA_VERSION = "1.2.0"

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
            {"layer": 1, "datatype": 11, "semantic": "etch_cutout"},
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
            {"layer": 1, "datatype": 11, "semantic": "etch_cutout"},
            {"layer": 2, "datatype": 0, "semantic": "cross_junction_port"},
            {"layer": 3, "datatype": 0, "semantic": "readout_claw_port"},
        ],
    },
}


def write_layer_semantics(path: Path) -> None:
    """Write the layer contract atomically using the registry filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(LAYER_SEMANTICS, indent=2) + "\n")
    temporary.replace(path)
