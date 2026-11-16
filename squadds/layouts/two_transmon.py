"""A two-Xmon layout: two Xmon crosses with a claw each, coupled across a gap ("gap") or through an
interdigital capacitor ("interdigital"), and the ordering of its Maxwell matrix (ground, cross A, cross B, claw A, claw B).

The geometry vector is ``p = [L_A, L_B, claw fraction A, claw fraction B, coupling]`` (um; claw fraction 1 = the claw runs
along the whole arm up to 15 um from the perpendicular arms; coupling = the gap in "gap", the capacitor's lead length in
"interdigital"). Fixed parameters are ``FIXED`` / ``IDC``. Used by the MCP tool ``generate_two_transmon_gds``.

Validity: this layout is not a family of the planar capacitance dataset; ``ground_spacing = 0`` and the 11.7 um claw lead are
outside the trained Xmon ranges.
"""

from __future__ import annotations

import numpy as np
from qiskit_metal import designs

from squadds.components.generalized_ncap_interdigital import GeneralizedCapNInterdigital
from squadds.components.xmon import Xmon

VARIANTS = ("gap", "interdigital")
P_START = {"gap": [160.0, 150.0, 0.8, 0.8, 25.0], "interdigital": [160.0, 150.0, 0.8, 0.8, 15.0]}
NETS = ["G", "Q_A:cross", "Q_B:cross", "Q_A:readout_connector_arm", "Q_B:readout_connector_arm"]
BODIES = ["ground", "cross A", "cross B", "claw A", "claw B"]

FIXED = {  # um; the claw parameters are those of quantum-metal's TransmonCross connection pads
    "cross_width": 30,
    "cross_gap": 20,
    "claw_width": 19.5,
    "claw_gap": 4.5,
    "ground_spacing": 0,
    "claw_cpw_length": 40,
    "claw_cpw_width": 11.7,
}
IDC = {  # um, the interdigital capacitor of variant (b)
    "finger_count": 2,
    "finger_length": 5,
    "finger_width": 5,
    "finger_gap": 6,
    "spine_width": 5,
    "lead_width": 10,
    "lead_gap": 2,
    "gap_ground": 9,
}
um = "{}um".format


def xmon_options(x, cross_length, claw_length, location):
    f = FIXED
    claw = {
        "connector_location": location,
        "connector_type": "0",
        "claw_length": um(claw_length),
        "claw_width": um(f["claw_width"]),
        "claw_gap": um(f["claw_gap"]),
        "ground_spacing": um(f["ground_spacing"]),
        "claw_cpw_length": um(f["claw_cpw_length"]),
        "claw_cpw_width": um(f["claw_cpw_width"]),
    }
    return {
        "pos_x": um(x),
        "cross_length": um(cross_length),
        "cross_width": um(f["cross_width"]),
        "cross_gap": um(f["cross_gap"]),
        "connection_pads": {"readout": claw},
    }


def idc_options(lead, x_north):
    """Interdigital capacitor rotated by 90 degrees: its north lead starts at x_north and it runs towards +x."""
    p = IDC
    o = {
        "finger_count": str(p["finger_count"]),
        "finger_length": um(p["finger_length"]),
        "finger_width": um(p["finger_width"]),
        "finger_gap_north_south": um(p["finger_gap"]),
        "finger_gap_east_west": um(p["finger_gap"]),
        "finger_radius": um(p["finger_width"] / 2),
        "finger_etch_radius": um(p["finger_width"] / 2),
        "finger_edge_gap_radius": um(1),
        "orientation": "90",
        "pos_x": um(x_north),
    }
    for s in ("north", "south"):
        o |= {
            f"{s}_cpw_length": um(lead),
            f"{s}_cpw_width": um(p["lead_width"]),
            f"{s}_cpw_gap": um(p["lead_gap"]),
            f"{s}_cpw_radius": um(1),
            f"{s}_cpw_etch_radius": um(1),
            f"{s}_spine_width": um(p["spine_width"]),
        }
    for s in ("north", "south", "east", "west"):
        o[f"{s}_gap_ground"] = um(p["gap_ground"])
    for s in ("north_west", "north_east", "south_west", "south_east"):
        o |= {f"{s}_spine_radius": um(1), f"{s}_spine_etch_radius": um(1)}
    return o


def idc_span(lead):
    """Length (um) of the interdigital capacitor from lead end to lead end."""
    p = IDC
    return 2 * lead + 2 * p["spine_width"] + p["finger_length"] + p["finger_gap"]


def geometry(p):
    L_A, L_B, frac_A, frac_B, coupling = p
    return {"L_A": L_A, "L_B": L_B, "claw_A": frac_A * (L_A - 15), "claw_B": frac_B * (L_B - 15), "coupling": coupling}


def two_transmons(p, variant):
    """The two-Xmon layout of variant "gap" or "interdigital" (a fresh Qiskit Metal design)."""
    g = geometry(p)
    d = designs.DesignPlanar({}, overwrite_enabled=True)
    overlap = 1.0  # the capacitor's leads reach 1 um into the arms, so they join the crosses
    gap = g["coupling"] if variant == "gap" else idc_span(g["coupling"]) - 2 * overlap
    Xmon(d, "Q_A", options=xmon_options(-(g["L_A"] + gap / 2), g["L_A"], g["claw_A"], "0"))
    Xmon(d, "Q_B", options=xmon_options(g["L_B"] + gap / 2, g["L_B"], g["claw_B"], "180"))
    if variant == "interdigital":
        GeneralizedCapNInterdigital(d, "C_AB", options=idc_options(g["coupling"], -gap / 2 - overlap))
    d.rebuild()
    return d


def ordered_maxwell(result):
    """Full Maxwell matrix (fF; ground, cross A, cross B, claw A, claw B) of an ``MLSimulator.predict(two_transmons(...))``
    result."""
    names = list(result["net_names"])
    if len(names) != 5:
        raise ValueError(f"expected ground + 4 bodies, got {names}")
    idx = [next(i for i, n in enumerate(names) if net in n.split("+")) for net in NETS]
    return np.asarray(result["prediction"].matrix_fF)[np.ix_(idx, idx)]
