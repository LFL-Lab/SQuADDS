"""SQuADDS_DB view of the planar capacitance dataset: canonical rows -> ``SQuADDS_DB`` row schema.

One canonical store (``SQuADDS/SQuADDS_PlanarCapacitance``, long-format N x N tables); this module converts its rows on
load into the row contract of the existing SQuADDS configs (``generalized_ncap_dataset.py``): ``design`` (Qiskit Metal
options with units, component class and module), ``sim_options`` (setup, ``layer_stack``, simulator ``Ansys Q3D``),
``sim_results`` (positive pairwise magnitudes ``<a>_to_<b>``, the signed ``maxwell_matrix`` keyed ``C_<net>_<net>`` with the
dataset net labels, ``units`` fF), ``notes`` and ``contributor``. Nothing is stored twice.

Only designs with a public label enter the view (test-split labels stay in the private heldout repo). Rows of
``coupler-GeneralizedCapNInterdigital-cap_matrix`` that are already in SQuADDS_DB (``squadds_db_ref`` set) are skipped,
and every config is deduplicated by ``geom_hash``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from squadds.database.generalized_ncap_dataset import CONTRIBUTOR as SAIKAT_CONTRIBUTOR

INTERDIGITAL_CONFIG = "coupler-GeneralizedCapNInterdigital-cap_matrix"
VIEW_CONFIGS = (
    INTERDIGITAL_CONFIG,
    "coupler-GeneralizedCapConcentric-cap_matrix",
    "coupler-GeneralizedCapStar-cap_matrix",
    "coupler-ShieldedCapConcentric-cap_matrix",
    "qubit-Xmon-cap_matrix",
    "qubit-TransmonPocket-cap_matrix",
)
SIMULATOR = "Ansys Q3D"

# dataset net label -> SQuADDS terminal name, per component class; keys of ``sim_results`` follow this order (ground last)
NET_NAMES: dict[str, dict[str, str]] = {
    "GeneralizedCapNInterdigital": {"N": "north", "S": "south"},
    "GeneralizedCapConcentric": {"N": "north", "S": "south"},
    "GeneralizedCapStar": {"S": "star", "N": "pad_0", **{f"P{k}": f"pad_{k}" for k in range(1, 12)}},
    "ShieldedCapConcentric": {"S": "inner", "N": "outer", **{f"P{k}": f"guard_{k}" for k in range(1, 9)}},
    "Xmon": {"S": "cross", "N": "claw", "P1": "bus_claw"},
    "TransmonCross": {"S": "cross", "N": "claw", "P1": "bus_claw"},
    "TransmonPocket": {"S": "pad_top", "P1": "pad_bot", "N": "readout_pad", "P2": "bus_pad"},
}
# Qiskit Metal qgeometry element name -> dataset net label (used to name the bodies of a rendered component)
QGEOMETRY_NETS: dict[str, dict[str, str]] = {
    "GeneralizedCapNInterdigital": {"cap_north_body": "N", "cap_south_body": "S"},
    "GeneralizedCapConcentric": {"cap_outer_body": "N", "cap_inner_body": "S"},
    "GeneralizedCapStar": {"star": "S", "pad_0": "N", **{f"pad_{k}": f"P{k}" for k in range(1, 12)}},
    "ShieldedCapConcentric": {"inner": "S", "outer_ring": "N", **{f"guard_{k}": f"P{k}" for k in range(1, 9)}},
    "Xmon": {"cross": "S", "readout_connector_arm": "N", "bus_connector_arm": "P1"},
    "TransmonCross": {"cross": "S", "readout_connector_arm": "N", "bus_connector_arm": "P1"},
    "TransmonPocket": {
        "pad_top": "S",
        "pad_bot": "P1",
        "readout_connector_pad": "N",
        "readout_wire": "N",
        "bus_connector_pad": "P2",
        "bus_wire": "P2",
    },
}
COMPONENT_MODULES = {"TransmonPocket": "qiskit_metal.qlibrary.qubits.transmon_pocket"}
PIN_NAMES = {
    "GeneralizedCapNInterdigital": (["north_end", "south_end"], {"top": "north_end", "bottom": "south_end"}),
}
# exp6 / exp7 were drawn with the first squadds2-helpers spine-ledge rule (components report, 2026-10-29)
TANGENT_LEDGE_COLLECTIONS = frozenset({"exp6", "exp7"})
# exp11 designs whose stored options do not describe the solved GDS (the label belongs to the GDS); exp12: whole collection
# plus 4 re-exported exp11 designs whose export merge clipped a pad differently (star export report, 2026-10-30)
NOT_REPRODUCIBLE = frozenset(
    f"ref/exp11/q3d_cap_{i}"
    for i in "1330 1419 1459 1481 1514 1556 1589 1591 1679 1695 1707 1835 1847 1882 1986 0326 1209 1583 1854".split()
)
NOT_REPRODUCIBLE_COLLECTIONS = frozenset({"exp12"})
# export style that rebuilds each star layout (squadds.components.export_fixups.EXPORT_STYLES): exp12 used the gdspy-era
# re-export (exact for 1,281 of 1,833 under shapely 2.0 / GEOS 3.11, approximate otherwise); these exp11 designs the re-export
EXP11_REEXPORT = frozenset(
    """
    0004 0012 0023 0093 0097 0099 0107 0110 0116 0136 0154 0159 0168 0175 0181 0184 0186 0235 0239 0257 0283
    0288 0293 0299 0301 0321 0335 0340 0344 0348 0383 0392 0406 0413 0442 0460 0470 0481 0482 0486 0491 0493
    0506 0511 0518 0523 0546 0566 0587 0601 0624 0627 0638 0639 0649 0650 0667 0674 0676 0682 0707 0711 0715
    0724 0725 0732 0740 0748 0777 0780 0800 0804 0824 0860 0876 0880 0932 0936 0950 0952 0959 0960 0961 0962
    0967 0971 0976 0977 0983 0986 1007 1010 1018 1019 1035 1043 1058 1063 1080 1090 1117 1119 1151 1169 1198
    1219 1225 1241 1247 1249 1259 1284 1304 1307 1308 1311 1374 1389 1394 1409 1430 1455 1477 1486 1498 1512
    1515 1517 1533 1546 1557 1569 1573 1590 1618 1642 1670 1693 1721 1744 1747 1761 1777 1793 1803 1859 1872
    1888 1904 1919 1962 1989 1994 1995
    """.split()
)
CONTRIBUTORS = {
    "saikat_das": dict(SAIKAT_CONTRIBUTOR),
    "sadman_shanto": {
        "group": "Levenson-Falk Lab",
        "PI": "Eli Levenson-Falk",
        "institution": "USC",
        "uploader": "Sadman Ahmed Shanto",
        "date_created": "2026-10-29",
    },
}
DESIGN_VIEW_COLUMNS = [
    "design_id",
    "geom_hash",
    "collection",
    "paper_family",
    "component_class",
    "squadds_config",
    "squadds_db_ref",
    "net_order",
    "split",
    "params",
    "contributor_id",
    "licence_id",
]
SIM_VIEW_COLUMNS = [
    "simulation_id",
    "design_id",
    "role",
    "solver",
    "substrate_material",
    "eps_r",
    "substrate_um",
    "tan_delta",
    "metal_model",
    "metal_t_um",
    "setup",
    "passes_completed",
    "qc_convergence",
    "qc_ok",
]


def component_class_of(config: str) -> str:
    return config.split("-")[1]


def net_name(component_class: str, label: str) -> str:
    if label == "G":
        return "ground"
    return NET_NAMES.get(component_class, {}).get(label, label)


def sim_results_from_matrix(labels: list[str], C, component_class: str) -> dict[str, Any]:
    """``sim_results`` of one design: ``<a>_to_<b>`` positive magnitudes for every pair and every self term (ground
    included, signal names before ground), the signed Maxwell matrix keyed with the dataset labels, and ``units``."""
    C = np.asarray(C, float)
    order = list(NET_NAMES.get(component_class, {}))
    sig = [i for i, lab in enumerate(labels) if lab != "G"]
    sig.sort(key=lambda i: (order.index(labels[i]) if labels[i] in order else len(order), i))
    g = labels.index("G")
    idx = [*sig, g]
    names = [net_name(component_class, labels[i]) for i in idx]
    out: dict[str, Any] = {}
    for p, i in enumerate(idx):
        for q in range(p, len(idx)):
            j = idx[q]
            out[f"{names[p]}_to_{names[q]}"] = float(abs(C[i, j]))
    if component_class == "GeneralizedCapNInterdigital":
        alias = {"north": "top", "south": "bottom", "ground": "ground"}
        for k in [k for k in out if k.split("_to_")[0] in ("north", "south")]:
            a, b = k.split("_to_")
            out[f"{alias[a]}_to_{alias[b]}"] = out[k]
    out["maxwell_matrix"] = {
        f"C_{labels[i]}_{labels[j]}": float(C[i, j]) for i in range(len(labels)) for j in range(len(labels))
    }
    out["units"] = "fF"
    return out


def design_options_for(collection: str, component_class: str, params: dict) -> dict:
    """Qiskit Metal options (strings with units) that rebuild the design with its SQuADDS component class."""
    if component_class == "ShieldedCapConcentric":
        from squadds.components.shielded_cap_concentric import shielded_options_from_generator

        return shielded_options_from_generator(params["generator"]["params"])
    opts = json.loads(json.dumps(params["cap_options"]))
    if collection in TANGENT_LEDGE_COLLECTIONS:
        opts["spine_ledge_rule"] = "tangent"
    return opts


def _sim_options(sim: dict, params: dict) -> dict:
    setup = json.loads(sim["setup"]) if sim.get("setup") else {}
    q3d = setup.get("ansys_gds", {}).get("q3d_signal", {})
    stack = dict(setup.get("layer_stack") or {})
    stack.update(
        substrate_material=sim.get("substrate_material"),
        eps_r=_num(sim.get("eps_r")),
        substrate_thickness_um=_num(sim.get("substrate_um")),
        tan_delta=_num(sim.get("tan_delta")),
        metal_model=sim.get("metal_model"),
    )
    if _num(sim.get("metal_t_um")) is not None:
        stack["metal_thickness_um"] = _num(sim.get("metal_t_um"))
    renderer = {
        "margin_factor_east_west": params.get("margin_factor_ew"),
        "margin_factor_north_south": params.get("margin_factor_ns"),
        "gds_north_port_length": params.get("gds_north_port_length"),
        "gds_north_port_layer": params.get("gds_north_port_layer"),
        "gds_south_port_length": params.get("gds_south_port_length"),
        "gds_south_port_layer": params.get("gds_south_port_layer"),
        "mesh": setup.get("mesh"),
        "gds_import": q3d or None,
    }
    return {
        "sim_type": "cap_matrix",
        "setup": q3d.get("setup", {}),
        "renderer_options": renderer,
        "layer_stack": stack,
        "simulator": SIMULATOR,
    }


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def export_style(collection: str, design_id: str) -> str | None:
    """Export style that rebuilds a star layout from its options; None for the other families."""
    if collection == "exp12":
        return "reexport_gdspy"
    if collection == "exp11":
        return "reexport" if design_id.rsplit("_", 1)[-1] in EXP11_REEXPORT else "original"
    return None


def _ground_gap_um(component_class: str, params: dict):
    """Unrounded generator ground gap of the exp13-15 families (the chip margin of their dataset GDS), else None."""
    gen = (params.get("generator") or {}).get("params") if isinstance(params, dict) else None
    if not gen:
        return None
    from squadds.layouts.dataset_export import FAMILY_OF_CLASS, generator_ground_gap_um

    try:
        return generator_ground_gap_um(FAMILY_OF_CLASS.get(component_class, ""), gen)
    except (KeyError, TypeError):
        return None


def convert_row(design: dict, sim: dict, labels: list[str], C) -> dict:
    """One SQuADDS_DB row from a canonical design row, its reference simulation row and its label matrix."""
    cls = design["component_class"]
    params = json.loads(design["params"]) if isinstance(design["params"], str) else design["params"]
    design_block = {
        "design_options": design_options_for(design["collection"], cls, params),
        "design_tool": "qiskit-metal",
        "coupler_type": cls,
        "component_name": cls,
        "component_module": COMPONENT_MODULES.get(cls, "squadds.components"),
        "component_class": cls,
    }
    if cls in PIN_NAMES:
        design_block["pin_names"], design_block["pin_aliases"] = PIN_NAMES[cls]
    reproducible = not (design["design_id"] in NOT_REPRODUCIBLE or design["collection"] in NOT_REPRODUCIBLE_COLLECTIONS)
    ref = design.get("squadds_db_ref")
    notes = {
        "design_id": design["design_id"],
        "family": design.get("paper_family"),
        "collection": design["collection"],
        "split": design.get("split"),
        "simulation_id": sim["simulation_id"],
        "convergence": sim.get("qc_convergence"),
        "label_qc_ok": bool(sim["qc_ok"]) if sim.get("qc_ok") is not None else None,
        "geometry_reproducible": reproducible,
        "export_style": export_style(design["collection"], design["design_id"]),
        "ground_gap_um": _ground_gap_um(cls, params),
        "net_order": list(labels),
        "net_names": {lab: net_name(cls, lab) for lab in labels},
        "squadds_db_ref": json.loads(ref) if isinstance(ref, str) and ref else None,
        "source_dataset": "SQuADDS/SQuADDS_PlanarCapacitance",
        "licence": design.get("licence_id"),
    }
    contributor = CONTRIBUTORS.get(design.get("contributor_id"), {"uploader": design.get("contributor_id")})
    return {
        "sim_options": _sim_options(sim, params),
        "sim_results": sim_results_from_matrix(list(labels), C, cls),
        "design": design_block,
        "notes": notes,
        "contributor": dict(contributor),
    }


def _pad_sim_results(rows: list[dict]) -> None:
    """Give every row the same ``sim_results`` keys (``None`` where a design has fewer conductors), so the flattened
    SQuADDS DataFrame (keys taken from the first row) keeps every pairwise column."""
    keys: dict[str, None] = {}
    for r in rows:
        keys.update(dict.fromkeys(k for k in r["sim_results"] if k not in ("maxwell_matrix", "units")))
    if any(len(r["sim_results"]) != len(keys) + 2 for r in rows):
        order = [*NET_NAMES.get(rows[0]["design"]["component_class"], {}).values(), "ground"]

        def rank(key):
            return tuple(order.index(n) if n in order else len(order) for n in key.split("_to_"))

        keys = dict.fromkeys(sorted(keys, key=rank))
    for r in rows:
        sr = r["sim_results"]
        r["sim_results"] = {**{k: sr.get(k) for k in keys}, "maxwell_matrix": sr["maxwell_matrix"], "units": "fF"}


def build_view(db, config: str) -> pd.DataFrame:
    """SQuADDS_DB rows (nested dict columns, as ``load_dataset(...).to_pandas()``) of one config from a
    ``PlanarCapacitanceDB``. Empty DataFrame when the config has no public labelled design."""
    if config not in VIEW_CONFIGS:
        raise KeyError(f"{config!r} is not a planar capacitance config; configs: {list(VIEW_CONFIGS)}")
    d = db.designs(config=config, columns=[c for c in DESIGN_VIEW_COLUMNS if c != "params"])
    if config == INTERDIGITAL_CONFIG:
        d = d[d.squadds_db_ref.isna()]
    d = d[d.split != "test"].sort_values("design_id").drop_duplicates("geom_hash", keep="first")
    columns = ["sim_options", "sim_results", "design", "notes", "contributor"]
    if d.empty:
        return pd.DataFrame(columns=columns)
    ids = d.design_id.tolist()
    sims = db.simulations(ids, role="reference", columns=SIM_VIEW_COLUMNS).drop_duplicates("design_id")
    mats = db.labels(ids)
    params = db.params(ids)
    sims = sims.set_index("design_id")
    rows = []
    for rec in d.to_dict("records"):
        did = rec["design_id"]
        if did not in mats or did not in sims.index:
            continue
        M = mats[did]
        rec["params"] = params[did]
        sim = {"simulation_id": sims.at[did, "simulation_id"], **sims.loc[did].to_dict()}
        rows.append(convert_row(rec, sim, list(M.index), M.to_numpy()))
    if not rows:
        return pd.DataFrame(columns=columns)
    _pad_sim_results(rows)
    return pd.DataFrame(rows, columns=columns)
