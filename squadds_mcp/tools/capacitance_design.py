"""
Planar capacitance design tools.
================================

Component -> GDS -> Maxwell matrix -> Hamiltonian, for the component families of the planar capacitance dataset
(interdigital / paddle, concentric, star / multipad star, Xmon, TransmonPocket, shielded concentric).

Tools:
    - ``get_capacitance_component_info``     the families, their Qiskit Metal classes, net names, configs, default options
    - ``generate_component_gds``             component class + options -> GDS in the dataset export (returns the path)
    - ``predict_capacitance_from_design``    component class + options -> named Maxwell matrix (``MLSimulator``)
    - ``capacitance_to_hamiltonian``         Maxwell matrix (or a GDS) -> C_sigma, E_C, E_J, f_q, alpha, g, L_j
    - ``solve_coupled_transmon_hamiltonian``  two transmons + two lambda/2 resonators: targets -> E_J, E_C, L_J, g, J and
                                             the target Maxwell matrix (full Hamiltonian, no RWA; the two-transmon example)
    - ``generate_two_transmon_gds``          The two-Xmon layout (gap or interdigital coupler) -> GDS (+ prediction)
    - ``correct_half_wave_resonator``        a SQuADDS half-wave row adapted to your claw: resonator length and kappa

The Hamiltonian step needs only numpy / scqubits / pyEPR (no ``[ml]`` extra). Qubit kinds are a registry
(``QUBIT_PRESETS``); the coupled two-transmon design shares its code with the two-transmon example (``squadds.calcs.coupled_transmons``,
``squadds.layouts.two_transmon``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from squadds_mcp.tools.capacitance_ml import _ml_missing, _predictor, ground_area_in_file, predict_gds
from squadds_mcp.utils import sanitize_for_json

GROUND_NAMES = ("G", "ground")
QUBIT_PRESETS: dict[str, dict[str, Any]] = {
    "xmon": {
        "nodes": ["cross"],
        "coupler": "claw",
        "note": "grounded island: C_sigma = the cross's Maxwell diagonal (every claw counts)",
    },
    "transmon_cross": {
        "nodes": ["cross"],
        "coupler": "claw",
        "note": "grounded island: C_sigma = Maxwell diagonal (with one claw this equals the SQuADDS TransmonCross "
        "convention cross_to_ground + cross_to_claw)",
    },
    "transmon_pocket": {
        "nodes": ["pad_top", "pad_bot"],
        "coupler": "readout_pad",
        "note": "floating pair: the pad-to-pad mode, C_eff = C12 + C1s C2s / (C1s + C2s), coupler pads held at ground",
    },
}


def _family_table() -> list[dict[str, Any]]:
    from squadds.database.planar_capacitance_view import NET_NAMES, VIEW_CONFIGS
    from squadds.layouts.dataset_export import CLASS_OF_FAMILY, MARGIN_GAPS

    rows = []
    for family, cls in CLASS_OF_FAMILY.items():
        config = next((c for c in VIEW_CONFIGS if f"-{cls}-" in c), None)
        rows.append(
            {
                "family": family,
                "component_class": cls,
                "squadds_db_config": config,
                "net_names": {"G": "ground", **NET_NAMES.get(cls, {})},
                "margin_gaps": MARGIN_GAPS[family],
            }
        )
    return rows


def _new_component(component_class: str, design_options: dict | None):
    from qiskit_metal import designs

    from squadds.components import get_component_class

    design = designs.DesignPlanar({}, overwrite_enabled=True)
    comp = get_component_class(component_class)(design, "component", options=dict(design_options or {}))
    design.rebuild()
    return comp


def component_info(component_class: str | None = None) -> dict[str, Any]:
    """Synchronous body of ``get_capacitance_component_info``."""
    try:
        table = _family_table()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out: dict[str, Any] = {
        "families": table,
        "export": "generate_component_gds writes the dataset convention: ground (1, 0), one polygon per conductor on "
        "(1, 10), 2 um port stubs on (2, 0) / (3, 0), 1 nm grid; the models were trained on this export",
        "options_units": "Qiskit Metal strings with units, e.g. '20um'; nested dicts as in the component's default_options",
    }
    if component_class:
        row = next((r for r in table if r["component_class"].lower() == component_class.lower()), None)
        if row is None:
            return {"error": f"{component_class!r} is not a planar capacitance family", "families": table}
        try:
            comp = _new_component(row["component_class"], None)
        except Exception as e:  # noqa: BLE001
            return {"error": f"cannot build {row['component_class']} ({type(e).__name__}: {e})", **row}
        out = {
            **row,
            "default_options": sanitize_for_json(dict(comp.options)),
            "doc": (type(comp).__doc__ or "").strip(),
        }
    return out


def generate_gds(
    component_class: str,
    design_options: dict | None,
    out_path: str,
    export: str = "dataset",
    export_style: str | None = None,
    margin_gaps: float | None = None,
    ground_gap_um: float | None = None,
) -> dict[str, Any]:
    """Synchronous body of ``generate_component_gds``."""
    err = _ml_missing()
    if err:
        return err
    out = Path(out_path).expanduser()
    if out.suffix.lower() != ".gds":
        return {"error": f"out_path must end in .gds: {out_path}"}
    try:
        comp = _new_component(component_class, design_options)
        if export == "dataset":
            from squadds.layouts.dataset_export import dataset_export_settings, export_dataset_gds

            settings = dataset_export_settings(comp, None, margin_gaps, export_style, ground_gap_um)
            export_dataset_gds(comp, out, None, margin_gaps, export_style, ground_gap_um)
        elif export == "qgeometry":
            from squadds.ml.capacitance.bridge import default_clearance_um, render_gds

            clearance = default_clearance_um(comp.design)
            render_gds(comp.design, out, clearance)
            settings = {"convention": "qgeometry", "clearance_um": clearance}
        else:
            return {"error": "export must be 'dataset' or 'qgeometry'"}
        from squadds.ml.capacitance.qc import gds_qc

        rep = gds_qc(out)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    keep = ("n_conductors", "n_polygons", "layers", "min_gap_um", "geom_hash", "unit_m")
    return sanitize_for_json(
        {
            "gds_path": str(out.resolve()),
            "component_class": type(comp).__name__,
            "design_options": dict(comp.options),
            "export": settings,
            "layout": {k: rep.info[k] for k in keep if k in rep.info},
            "warnings": rep.warnings,
            "errors": rep.errors,
            "next": "predict_capacitance_from_gds(gds_path) for the Maxwell matrix (conductors named C1..Cn in file "
            "order), or predict_capacitance_from_design for SQuADDS net names",
        }
    )


def predict_design(
    component_class: str,
    design_options: dict | None,
    model: str = "QCap",
    threads: int | None = None,
    export: str = "auto",
    save_gds_dir: str | None = None,
    eps_r: float = 11.45,
    thickness_um: float = 350.0,
) -> dict[str, Any]:
    """Synchronous body of ``predict_capacitance_from_design``."""
    err = _ml_missing()
    if err:
        return err
    try:
        from squadds.ml.capacitance.bridge import MLSimulator

        workdir = Path(save_gds_dir).expanduser() if save_gds_dir else None
        predictor = _predictor(model, threads)
        sim = MLSimulator(
            threads=threads,
            predictor=predictor,
            export=export,
            workdir=workdir,
            eps_r=float(eps_r),
            thickness_um=float(thickness_um),
        )
        r = sim.predict({"design_options": dict(design_options or {}), "component_class": component_class})
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    res = r["prediction"]
    labels = list(r["net_names"])
    names = [r["net_names"][lab] for lab in labels]
    return sanitize_for_json(
        {
            "names": names,
            "labels": labels,
            "matrix_fF": res.matrix_fF,
            "units": "fF",
            "convention": "Maxwell matrix, ground first; diagonal > 0, off-diagonal = -mutual",
            "sim_results": r["sim_results"],
            "design_options": r["design"]["design_options"],
            "component_class": r["design"]["component_class"],
            "gds_path": r["design"]["gds"],
            "export": r["export"],
            "fidelity": r["fidelity"],
            "model": res.model,
            "checks": r["checks"],
            "warnings": r["warnings"],
            "substrate": res.substrate,
            "timing_s": r["timing"],
            "threads": res.threads,
            "machine": res.machine,
        }
    )


def _matrix_and_names(matrix_fF, names) -> tuple[np.ndarray, list[str]]:
    C = np.asarray(matrix_fF, float)
    names = [str(n) for n in names]
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != len(names):
        raise ValueError(f"matrix_fF must be square with one row per name ({len(names)} names, shape {C.shape})")
    if len(set(names)) != len(names):
        raise ValueError(f"names must be unique: {names}")
    return C, names


def hamiltonian_from_maxwell(
    matrix_fF,
    names,
    qubit_type: str | None = None,
    qubit_nodes: list[str] | None = None,
    coupler: str | None = None,
    EJ_GHz: float | None = None,
    qubit_frequency_GHz: float | None = None,
    anharmonicity_MHz: float | None = None,
    resonator_frequency_GHz: float | None = None,
    resonator_type: str = "quarter",
    Z0: float = 50.0,
) -> dict[str, Any]:
    """Transmon parameters of one qubit mode of a Maxwell matrix (fF, ground included, every other conductor at ground).

    E_J: ``EJ_GHz`` if given, else the E_J of the target ``(qubit_frequency_GHz, anharmonicity_MHz)`` (the SQuADDS Analyzer
    rule: E_J from the target, f_q and alpha from this layout's E_C). g: to ``coupler`` through the equivalent grounded
    two-node circuit (``g_from_cap_matrix``), when ``resonator_frequency_GHz`` is given."""
    from pyEPR.calcs import Convert
    from scipy.constants import Planck, e
    from scqubits.core.transmon import Transmon

    from squadds.calcs.floating_transmon import floating_transmon_capacitances
    from squadds.calcs.transmon_cross import g_from_cap_matrix_numba, get_transmon_E01_alpha

    C, names = _matrix_and_names(matrix_fF, names)
    preset = QUBIT_PRESETS.get(qubit_type or "", {})
    if qubit_type and not preset:
        raise ValueError(f"qubit_type must be one of {sorted(QUBIT_PRESETS)} (or pass qubit_nodes)")
    nodes = list(qubit_nodes or preset.get("nodes") or [])
    coupler = coupler if coupler is not None else preset.get("coupler")
    if not nodes or len(nodes) > 2:
        raise ValueError("qubit_nodes: one conductor (grounded transmon) or two (floating transmon)")
    missing = [n for n in [*nodes, *([coupler] if coupler else [])] if n not in names]
    if missing:
        if coupler in missing and coupler == preset.get("coupler") and not set(nodes) & set(missing):
            coupler = None
        else:
            raise ValueError(f"{missing} not in names {names}")
    if any(n in GROUND_NAMES for n in nodes):
        raise ValueError("the ground net cannot be a qubit node")
    if len(nodes) == 1:
        i = names.index(nodes[0])
        c_sigma = float(C[i, i])
        c_g = float(-C[i, names.index(coupler)]) if coupler else 0.0
        mode = "grounded"
    else:
        c_sigma, c_g = floating_transmon_capacitances(C, names, tuple(nodes), coupler)
        mode = "floating"
    EC_GHz = e**2 / (2 * c_sigma * 1e-15) / Planck * 1e-9
    out: dict[str, Any] = {
        "mode": mode,
        "qubit_nodes": nodes,
        "coupler": coupler,
        "C_sigma_fF": c_sigma,
        "C_g_fF": c_g,
        "EC_MHz": 1e3 * EC_GHz,
        "note": preset.get("note")
        or ("grounded island: C_sigma = Maxwell diagonal" if mode == "grounded" else "floating pair: pad-to-pad mode"),
    }
    ej_source = None
    if EJ_GHz is None and qubit_frequency_GHz is not None and anharmonicity_MHz is not None:
        EJ_GHz = float(Transmon.find_EJ_EC(qubit_frequency_GHz, anharmonicity_MHz * 1e-3)[0])
        ej_source = "from the target (f_q, alpha), as the SQuADDS Analyzer does"
    elif EJ_GHz is not None:
        ej_source = "given"
    if EJ_GHz is not None:
        f_q, alpha = get_transmon_E01_alpha(float(EJ_GHz), EC_GHz)
        out.update(
            EJ_GHz=float(EJ_GHz),
            EJ_source=ej_source,
            EJ_over_EC=float(EJ_GHz) / EC_GHz,
            Lj_nH=float(Convert.Lj_from_Ej(float(EJ_GHz), "GHz", "nH")),
            qubit_frequency_GHz=float(f_q),
            anharmonicity_MHz=float(alpha),
        )
        if resonator_frequency_GHz is not None and coupler and c_g > 0:
            out["g_MHz"] = float(
                g_from_cap_matrix_numba(
                    c_sigma - c_g, c_g, float(EJ_GHz), float(resonator_frequency_GHz), resonator_type, Z0
                )
            )
            out["g_inputs"] = {
                "C_Q_fF": c_sigma - c_g,
                "C_g_fF": c_g,
                "resonator_frequency_GHz": resonator_frequency_GHz,
                "resonator_type": resonator_type,
                "Z0_ohm": Z0,
            }
    else:
        out["hint"] = "pass EJ_GHz, or qubit_frequency_GHz and anharmonicity_MHz, for f_q, alpha, L_j and g"
    return out


def hamiltonian(
    matrix_fF=None,
    names=None,
    gds_path: str | None = None,
    threads: int | None = None,
    model: str = "QCap",
    ground_layers: str = "1/0",
    conductor_layers: str = "1/10",
    eps_r: float = 11.45,
    thickness_um: float = 350.0,
    ground_pieces: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Synchronous body of ``capacitance_to_hamiltonian``."""
    pred = None
    if gds_path:
        pred = predict_gds(
            gds_path,
            model=model,
            threads=threads,
            ground_layers=ground_layers,
            conductor_layers=conductor_layers,
            eps_r=eps_r,
            thickness_um=thickness_um,
            ground_pieces=ground_pieces,
        )
        if "error" in pred:
            return pred
        matrix_fF, names = pred["matrix_fF"], pred["names"]
    if matrix_fF is None or names is None:
        return {"error": "pass matrix_fF and names (ground included), or gds_path"}
    try:
        out = hamiltonian_from_maxwell(matrix_fF, names, **kwargs)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "names": list(names)}
    if pred is not None:
        keep = (
            "names",
            "matrix_fF",
            "model",
            "checks",
            "warnings",
            "substrate",
            "timing_s",
            "threads",
            "machine",
        )
        out["prediction"] = {k: pred[k] for k in keep if k in pred}
    return sanitize_for_json(out)


BODIES = ["ground", "cross A", "cross B", "claw A", "claw B"]  # = squadds.layouts.two_transmon.BODIES
TWO_TRANSMON_TARGET_KEYS = ("f_q_A", "eta_A", "f_q_B", "eta_B", "f_r_A", "f_r_B", "J", "chi_A", "chi_B")
TWO_TRANSMON_CONVENTIONS = [
    "frequencies f = omega / 2 pi in GHz; eta < 0; chi and kappa as magnitudes",
    "targets are the parameters of the Hamiltonian (exact transmon f01 and eta with E_C loaded by the whole circuit, bare "
    "f_r, J = K n01_A n01_B, g = G n01 with the exact charge matrix elements), not dressed transitions: with typical "
    "targets |1_A 1_B> and |0_A 2_B> lie close and sqrt(2) J mixes them, so a dressed eta is not a well-posed target "
    "(the two-transmon example shows parameter vs dressed for every quantity)",
    "chi = half the full dispersive shift, (f_r with the qubit in |1> - f_r with it in |0>) / 2, computed from the full "
    "Hamiltonian (6 levels per transmon, 5 photons per resonator, counter-rotating terms kept); reading chi as the full "
    "shift gives g about sqrt(2) smaller",
    "kappa is not part of the Hamiltonian: it is set by the resonator's feedline coupler (SQuADDS half-wave rows)",
]
TWO_TRANSMON_VALIDITY = [
    "two Xmons with claws (and an interdigital capacitor) on one chip is NOT a training family of the capacitance models. "
    "the two-transmon example checked its two final layouts (its default targets) against Q3D: median relative error of QCap "
    "model over the coupling entries 0.573 % (MAE 0.128 / RMSE 0.209 fF) gap and 0.337 % (MAE 0.101 / RMSE 0.171 fF) "
    "interdigital, over all 15 entries 0.568 % (MAE 0.395 / RMSE 0.967 fF) and 0.423 % (MAE 0.645 / RMSE 1.882 fF); with "
    "the Q3D matrix every Hamiltonian parameter stayed within 2.2 % of its target (J 45.9 MHz vs 45 MHz, f_q +2 to +12 "
    "MHz). Time: Q3D 2025.2 about 31 min per layout on 16 cores (Intel Xeon Silver 4116), the prediction about 3.4 s "
    "(2 threads, Intel Xeon E5-2640 v4). Stored solves: scripts/planar_capacitance/two_transmon_q3d/q3d/. Any other layout is "
    "unvalidated: run one field solve (Q3D / Palace) of the final layout before fabrication",
    "out of the trained ranges: ground_spacing = 0 (claw inside the qubit pocket, needed for g ~ 120 MHz with a lambda/2 "
    "resonator) and the 11.7 um claw lead (dataset 10 um)",
    "closest validated cases (paper recipe vs Q3D): unseen qubits and couplers 0.672 % (MAE 0.2103 fF / RMSE 0.3522 fF); "
    "lattices, pairs within 100 um, 0.61-1.19 % (MAE 0.21-0.33 fF, RMSE not reported); in-family 0.177 % (MAE 0.0203 / "
    "RMSE 0.0250 fF)",
    "sensitivity (two-transmon example): +1 % cross-to-ground C moves f_q by -17.1 MHz, eta -1.9 MHz, chi -2.9 %, J -0.6 %; "
    "+1 % claw C moves chi +1.5 %",
    "QCap-Fast vs QCap on the final layouts differed by 7.76 % (MAE 1.39 / RMSE 1.97 fF) gap and 14.27 % (MAE 3.25 / RMSE "
    "4.47 fF) interdigital, and QCap-Fast derivatives pointed the coupler the wrong way: confirm every step with QCap, use "
    "QCap-Fast only for step directions",
    "avoid exactly mirror-symmetric layouts (L_A = L_B): the layout reader can drop part of a keyhole ground; compare the "
    "ground area read by the model with the file (generate_two_transmon_gds returns ground_area_check; "
    "inspect_capacitance_layout does it for any GDS)",
]
RESONATOR_STEP = (
    "correct_half_wave_resonator(row_frequency_GHz=<row cavity_frequency_GHz>, row_kappa_kHz=<row kappa_kHz>, "
    "row_length_um=<the row resonator's cpw_opts total_length, um>, f_r_GHz=<target f_r>, line_frequency_GHz and "
    "node_capacitance_fF=<this qubit's entries of layout.line_frequency_GHz / layout.node_capacitance_fF from "
    "solve_coupled_transmon_hamiltonian with your layout matrix>, claw_options=<the row's claw_opts>) predicts the row's "
    "claw, then scales the length and kappa (the two-transmon example's resonator step); kappa is the row value scaled, not re-simulated"
)


def _check_targets(targets: dict) -> list[str]:
    missing = [k for k in TWO_TRANSMON_TARGET_KEYS if k not in targets]
    errors = [f"missing targets {missing}"] if missing else []
    errors += [f"{k} must be negative (anharmonicity, GHz)" for k in ("eta_A", "eta_B") if targets.get(k, -1) >= 0]
    errors += [
        f"{k} must be positive (GHz)" for k in TWO_TRANSMON_TARGET_KEYS if "eta" not in k and targets.get(k, 1) <= 0
    ]
    return errors


def _five(M) -> dict[str, float]:
    M = np.asarray(M, float)
    return {
        "C_sigma_A (cross A diagonal)": M[1, 1],
        "C_sigma_B (cross B diagonal)": M[2, 2],
        "C_AB (cross A - cross B)": -M[1, 2],
        "C_g_A (cross A - claw A)": -M[1, 3],
        "C_g_B (cross B - claw B)": -M[2, 4],
    }


def coupled_transmon_design(
    targets: dict,
    reference_matrix_fF=None,
    layout_matrix_fF=None,
) -> dict[str, Any]:
    """Synchronous body of ``solve_coupled_transmon_hamiltonian`` (resonator lines Z0 = 50 ohm, fixed)."""
    import time

    from squadds.calcs import coupled_transmons as ct

    errors = _check_targets(targets)
    if errors:
        return {"error": "; ".join(errors), "target_keys": list(TWO_TRANSMON_TARGET_KEYS)}
    t0 = time.perf_counter()
    try:
        c = ct.design_circuit(targets, ct.solve_couplings(targets, abs(targets["chi_A"]), abs(targets["chi_B"])))
        table = ct.design_table(c)
        achieved = ct.compare(targets, c)
        solve_s = time.perf_counter() - t0
        ref = np.asarray(reference_matrix_fF, float) if reference_matrix_fF is not None else ct.ideal_reference(c)
        if ref.shape != (5, 5):
            return {"error": f"reference_matrix_fF must be 5 x 5 ({BODIES}), got {ref.shape}"}
        M = ct.target_capacitances(ref, c)
        resid = ct.circuit_numbers(M, c) / ct.goal(c) - 1
        k = ct.circuit(M, c["fr"])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out: dict[str, Any] = {
        "transmons": table.to_dict(),
        "circuit_GHz": {"EJ": c["EJ"], "EC": c["EC"], "K": c["K"], "G": np.diag(c["G"]), "fr": c["fr"]},
        "hamiltonian_check": achieved.to_dict(orient="index"),
        "target_capacitances": {
            "bodies": BODIES,
            "matrix_fF": M,
            "five_controlled_fF": _five(M),
            "reference": "given"
            if reference_matrix_fF is not None
            else "ideal (no claw-to-ground, no parasitics): pass "
            "the predicted matrix of your first layout as reference_matrix_fF for the real targets",
            "solve_residual_pct": 100 * resid,
            "note": "the five controlled capacitances hit E_C^A, E_C^B, K, G_A, G_B exactly; the small cross couplings the "
            "full inverse adds (qubit A to resonator B and vice versa, resonator-resonator) move chi by ~0.1 % even at the "
            "target matrix itself",
            "line_capacitance_fF": k["C_line_fF"],
            "line_frequency_GHz": k["f_line"],
            "Z0_ohm": 50.0,
        },
        "timing_s": {"hamiltonian": solve_s, "total": time.perf_counter() - t0},
        "conventions": TWO_TRANSMON_CONVENTIONS,
        "validity": TWO_TRANSMON_VALIDITY,
        "resonators": {
            q: {
                "find_closest_designs": {
                    "system_type": "qubit_cavity",
                    "qubit": "TransmonCross",
                    "cavity": "RouteMeander",
                    "resonator_type": "half",
                    "target_params": {
                        "qubit_frequency_GHz": targets[f"f_q_{q}"],
                        "anharmonicity_MHz": 1e3 * targets[f"eta_{q}"],
                        "cavity_frequency_GHz": targets[f"f_r_{q}"],
                        **({"kappa_kHz": 1e6 * targets[f"kappa_{q}"]} if f"kappa_{q}" in targets else {}),
                    },
                },
                "note": "the half-wave search runs on qubit-cavity systems; use only the resonator part of the row (its "
                "length and NCap feedline coupler). The claw loads the resonator, so the row's length and kappa must be "
                "corrected for your claw",
                "correction": RESONATOR_STEP,
            }
            for q in "AB"
        },
    }
    if layout_matrix_fF is not None:
        L = np.asarray(layout_matrix_fF, float)
        if L.shape != (5, 5):
            return {"error": f"layout_matrix_fF must be 5 x 5 ({BODIES}), got {L.shape}"}
        try:
            fc = ct.final_circuit(targets, L)
            check = ct.compare(targets, fc)
            five_t, five_l = _five(M), _five(L)
            out["layout"] = {
                "E_J_GHz (f01 on target)": fc["EJ"],
                "L_J_nH": ct.design_table(fc).loc["L_J (nH)"].to_dict(),
                "hamiltonian_check": check.to_dict(orient="index"),
                "worst_miss_pct": float(check["miss (%)"].abs().max()),
                "line_frequency_GHz": fc["f_line"],
                "line_capacitance_fF": fc["C_line_fF"],
                "node_capacitance_fF": fc["C_node_fF"],
                "capacitance_miss": {
                    k: {
                        "target_fF": five_t[k],
                        "layout_fF": five_l[k],
                        "miss_fF": five_l[k] - five_t[k],
                        "miss_pct": 100 * (five_l[k] / five_t[k] - 1),
                    }
                    for k in five_t
                },
            }
        except Exception as e:  # noqa: BLE001
            out["layout"] = {"error": f"{type(e).__name__}: {e}"}
    return sanitize_for_json(out)


def line_frequency(f_loaded: float, C_end_fF: float) -> float:
    """Frequency (GHz) of the bare 50 ohm lambda/2 line whose lumped capacitance plus C_end resonates at f_loaded."""
    from squadds.calcs.coupled_transmons import line_capacitance

    f = f_loaded
    for _ in range(60):
        C = 1e15 * line_capacitance(f)
        f = f_loaded * np.sqrt((C + C_end_fF) / C)
    return float(f)


def claw_capacitance(
    claw_options: dict, model: str = "QCap", threads: int | None = None, eps_r=11.45, thickness_um=350.0
):
    """Self capacitance (fF) of a SQuADDS row's claw on its own (``TransmonClaw``), predicted by ``model`` (default ``QCap``)."""
    from qiskit_metal import designs

    from squadds.components.claw_coupler import TransmonClaw
    from squadds.ml.capacitance.bridge import MLSimulator

    d = designs.DesignPlanar({}, overwrite_enabled=True)
    opts = {k: v for k, v in dict(claw_options).items() if k not in ("pos_x", "pos_y", "orientation")}
    TransmonClaw(d, "claw", options=opts)
    d.rebuild()
    sim = MLSimulator(
        threads=threads,
        predictor=_predictor(model, threads),
        eps_r=float(eps_r),
        thickness_um=float(thickness_um),
    )
    r = sim.predict(d)
    return float(r["prediction"].matrix_fF[1, 1]), r


def resonator_correction(
    row_frequency_GHz: float,
    row_kappa_kHz: float,
    row_length_um: float,
    f_r_GHz: float,
    line_frequency_GHz: float,
    node_capacitance_fF: float,
    row_claw_capacitance_fF: float | None = None,
    claw_options: dict | None = None,
    model: str = "QCap",
    threads: int | None = None,
    eps_r: float = 11.45,
    thickness_um: float = 350.0,
) -> dict[str, Any]:
    """Synchronous body of ``correct_half_wave_resonator``."""
    from squadds.calcs.coupled_transmons import line_capacitance

    pred = None
    if row_claw_capacitance_fF is None:
        if not claw_options:
            return {"error": "pass row_claw_capacitance_fF, or claw_options (the row's claw_opts) to predict it"}
        err = _ml_missing()
        if err:
            return err
        try:
            row_claw_capacitance_fF, r = claw_capacitance(claw_options, model, threads, eps_r, thickness_um)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        p = r["prediction"]
        pred = {
            "model": p.model,
            "warnings": r["warnings"],
            "timing_s": r["timing"],
            "threads": p.threads,
            "machine": p.machine,
        }
    c_end = float(row_claw_capacitance_fF)
    f_line_row = line_frequency(float(row_frequency_GHz), c_end)
    c_tot_row = 1e15 * float(line_capacitance(f_line_row)) + c_end
    length = float(row_length_um) * f_line_row / float(line_frequency_GHz)
    kappa = (
        float(row_kappa_kHz) * (float(f_r_GHz) / float(row_frequency_GHz)) ** 2 * c_tot_row / float(node_capacitance_fF)
    )
    out: dict[str, Any] = {
        "resonator_length_um": length,
        "kappa_kHz": kappa,
        "row_claw_capacitance_fF": c_end,
        "row_line_frequency_GHz": f_line_row,
        "row_total_capacitance_fF": c_tot_row,
        "line_frequency_GHz": float(line_frequency_GHz),
        "model": "lumped lambda/2 line (Z0 = 50 ohm) loaded by the claw, as squadds.core.processing."
        "update_cavity_frequency_and_kappa: length scaled by f_line_row / f_line, kappa by (f_r / f_row)^2 * "
        "C_total_row / C_node (fixed feedline coupler)",
        "note": "kappa is the SQuADDS row value scaled, not re-simulated. the two-transmon example tested the length scaling on 12 "
        "simulated siblings of one claw and coupler: a few MHz within a couple of mm of the reference length, drifting "
        "far from it (MAE 82.7 MHz, RMSE 172.1 MHz over all 12, one outlier sibling included)",
    }
    if pred:
        out["claw_prediction"] = pred
    return sanitize_for_json(out)


def two_transmon_gds(
    p: list[float],
    variant: str,
    out_path: str,
    predict: bool = False,
    model: str = "QCap",
    threads: int | None = None,
    eps_r: float = 11.45,
    thickness_um: float = 350.0,
) -> dict[str, Any]:
    """Synchronous body of ``generate_two_transmon_gds``."""
    err = _ml_missing()
    if err:
        return err
    out = Path(out_path).expanduser()
    if out.suffix.lower() != ".gds":
        return {"error": f"out_path must end in .gds: {out_path}"}
    if len(p) != 5:
        return {"error": "p = [L_A, L_B, claw fraction A, claw fraction B, coupling] (um, um, -, -, um)"}
    try:
        from squadds.layouts import two_transmon as T
        from squadds.ml.capacitance import prepare_layout
        from squadds.ml.capacitance.bridge import render_gds

        if variant not in T.VARIANTS:
            return {"error": f"variant must be one of {T.VARIANTS}"}
        design = T.two_transmons(np.asarray(p, float), variant)
        out.parent.mkdir(parents=True, exist_ok=True)
        render_gds(design, out)
        in_file, read = ground_area_in_file(out, [(1, 0)]), float(prepare_layout(str(out)).ground.area)
        res: dict[str, Any] = {
            "gds_path": str(out.resolve()),
            "variant": variant,
            "geometry_um": T.geometry(p),
            "fixed_um": {"FIXED": T.FIXED, **({"IDC": T.IDC} if variant == "interdigital" else {})},
            "bodies": T.BODIES,
            "ground_area_check": {
                "in_file_um2": in_file,
                "read_by_model_um2": read,
                "ok": abs(read - in_file) <= 1e-4 * in_file,
            },
            "validity": TWO_TRANSMON_VALIDITY,
        }
        if p[0] == p[1]:
            res["warning"] = (
                "L_A = L_B: an exactly mirror-symmetric layout can lose part of the keyhole ground when read"
            )
        if not res["ground_area_check"]["ok"]:
            res["ground_warning"] = "the model reads less ground than the file holds: change the geometry slightly"
        if predict:
            from squadds.ml.capacitance.bridge import MLSimulator

            sim = MLSimulator(
                threads=threads,
                predictor=_predictor(model, threads),
                eps_r=float(eps_r),
                thickness_um=float(thickness_um),
            )
            r = sim.predict(design)
            pred = r["prediction"]
            res |= {
                "matrix_fF": T.ordered_maxwell(r),
                "model": pred.model,
                "checks": r["checks"],
                "warnings": r["warnings"],
                "substrate": pred.substrate,
                "timing_s": r["timing"],
                "threads": pred.threads,
                "machine": pred.machine,
                "next": "solve_coupled_transmon_hamiltonian(targets, layout_matrix_fF=matrix_fF) for the Hamiltonian misses",
            }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return sanitize_for_json(res)


def register_capacitance_design_tools(mcp: FastMCP) -> None:
    """Register the planar capacitance design tools (component -> GDS -> matrix -> Hamiltonian)."""

    @mcp.tool()
    async def get_capacitance_component_info(component_class: str | None = None) -> dict[str, Any]:
        """The component families of the planar capacitance dataset and how to build them.

        Without an argument: one row per family (``interdigital``, ``paddle``, ``concentric``, ``star``,
        ``multipad_star``, ``xmon``, ``pocket``, ``shielded_concentric``) with its Qiskit Metal class
        (``GeneralizedCapNInterdigital``, ``GeneralizedCapConcentric``, ``GeneralizedCapStar``, ``Xmon``,
        ``TransmonPocket``, ``ShieldedCapConcentric``), its SQuADDS_DB config, the dataset net labels -> SQuADDS net names
        (e.g. ``S`` -> ``cross``) and the export margin. With ``component_class``: that row plus the class's full
        ``default_options`` (start from these, change only what you need) and its docstring.
        """
        return await asyncio.to_thread(component_info, component_class)

    @mcp.tool()
    async def generate_component_gds(
        component_class: str,
        out_path: str,
        design_options: dict[str, Any] | None = None,
        export: str = "dataset",
        export_style: str | None = None,
        margin_gaps: float | None = None,
        ground_gap_um: float | None = None,
    ) -> dict[str, Any]:
        """Build a planar capacitance component in Qiskit Metal and write its GDS (needs the ``[ml]`` extra for gdstk).

        Args:
            component_class: e.g. ``Xmon``, ``GeneralizedCapStar``, ``GeneralizedCapConcentric``, ``TransmonPocket``,
                ``ShieldedCapConcentric``, ``GeneralizedCapNInterdigital`` (see ``get_capacitance_component_info``).
            out_path: the GDS file to write (``.gds``; parent folders are created).
            design_options: Qiskit Metal options (strings with units, e.g. ``{"cross_length": "180um"}``); missing keys
                take the class defaults.
            export: ``dataset`` (default: the convention the models were trained on, chip = component + margin) or
                ``qgeometry`` (square ground domain around the rendered qgeometry).
            export_style: star layouts only (default ``original`` for one pad, ``reexport_gdspy`` for more).
            margin_gaps: chip margin in ground gaps (default per family). ground_gap_um: margin gap for Xmon /
                TransmonPocket / ShieldedCapConcentric (default from the options).

        Returns the absolute ``gds_path``, the resolved options and export settings, and a layout readback
        (conductor count including ground, layers, smallest gap, ``geom_hash``).
        """
        return await asyncio.to_thread(
            generate_gds, component_class, design_options, out_path, export, export_style, margin_gaps, ground_gap_um
        )

    @mcp.tool()
    async def predict_capacitance_from_design(
        component_class: str,
        design_options: dict[str, Any] | None = None,
        model: str = "QCap",
        threads: int | None = None,
        export: str = "auto",
        save_gds_dir: str | None = None,
        eps_r: float = 11.45,
        thickness_um: float = 350.0,
    ) -> dict[str, Any]:
        """Predict the Maxwell capacitance matrix of a component from its class and options (``MLSimulator``).

        The component is rebuilt, exported exactly as the dataset was (for the dataset families; ``export="qgeometry"``
        for anything else, e.g. quantum-metal ``TransmonCross``) and predicted. Conductors carry SQuADDS net names
        (``cross``, ``claw``, ``pad_top``, ``north``, ...), so the result feeds ``capacitance_to_hamiltonian`` and the
        SQuADDS ``sim_results`` keys (``cross_to_claw``, ``north_to_south``, ...) directly.

        Args:
            component_class: a SQuADDS / Qiskit Metal component class name.
            design_options: its options (class defaults for missing keys); a SQuADDS row's ``design_options`` works.
            model: ``QCap`` (default) or ``QCap-Fast`` (screening of trained families only).
            threads: CPU threads. export: ``auto`` / ``dataset`` / ``qgeometry``.
            save_gds_dir: keep the exported GDS in this folder (``gds_path`` in the result); default: temporary.
            eps_r / thickness_um: the substrate (default the training stack, Si 11.45, 350 um); off it ``QCap``
                solves the physics prior at that stack, ``QCap-Fast`` rescales and warns (see
                ``predict_capacitance_from_gds``).

        Returns ``names`` (ground first, SQuADDS net names), ``labels`` (dataset labels G, S, N, P1, ...),
        ``matrix_fF``, ``sim_results``, checks, warnings, ``substrate`` and timing with machine and threads.
        """
        return await asyncio.to_thread(
            predict_design,
            component_class,
            design_options,
            model,
            threads,
            export,
            save_gds_dir,
            eps_r,
            thickness_um,
        )

    @mcp.tool()
    async def capacitance_to_hamiltonian(
        matrix_fF: list[list[float]] | None = None,
        names: list[str] | None = None,
        gds_path: str | None = None,
        qubit_type: str | None = None,
        qubit_nodes: list[str] | None = None,
        coupler: str | None = None,
        EJ_GHz: float | None = None,
        qubit_frequency_GHz: float | None = None,
        anharmonicity_MHz: float | None = None,
        resonator_frequency_GHz: float | None = None,
        resonator_type: str = "quarter",
        Z0: float = 50.0,
        threads: int | None = None,
        model: str = "QCap",
        ground_layers: str = "1/0",
        conductor_layers: str = "1/10",
        eps_r: float = 11.45,
        thickness_um: float = 350.0,
        ground_pieces: str = "",
    ) -> dict[str, Any]:
        """Transmon Hamiltonian parameters from a Maxwell capacitance matrix (fF, ground included) or from a GDS.

        Pass ``matrix_fF`` + ``names`` (e.g. from ``predict_capacitance_from_design``, ``predict_capacitance_from_gds`` or
        a dataset label), or ``gds_path`` (predicted first with ``model``, ``ground_layers``,
        ``conductor_layers``, ``eps_r``, ``thickness_um`` and ``ground_pieces`` exactly as in
        ``predict_capacitance_from_gds``; conductors are then ``C1..Cn`` in file order). Every conductor that is not part
        of the qubit mode is held at ground.

        Qubit mode: ``qubit_type`` preset ``xmon`` / ``transmon_cross`` (island ``cross``, coupler ``claw``) or
        ``transmon_pocket`` (pads ``pad_top`` + ``pad_bot``, coupler ``readout_pad``), or explicit ``qubit_nodes``: one
        conductor (grounded transmon, C_sigma = Maxwell diagonal) or two (floating transmon, the pad-to-pad mode), and
        ``coupler`` (the readout conductor for g).

        E_J: ``EJ_GHz``, or the target ``qubit_frequency_GHz`` + ``anharmonicity_MHz`` (E_J from the target, f_q and alpha
        from this E_C: the SQuADDS Analyzer rule). g (MHz) needs ``resonator_frequency_GHz`` (``resonator_type``
        ``quarter`` / ``half``, ``Z0``).

        Returns C_sigma and C_g (fF), E_C (MHz), and with E_J: E_J/E_C, L_j (nH), f_q (GHz), alpha (MHz) and g (MHz).
        """
        return await asyncio.to_thread(
            hamiltonian,
            matrix_fF,
            names,
            gds_path,
            threads,
            model,
            ground_layers,
            conductor_layers,
            eps_r,
            thickness_um,
            ground_pieces,
            qubit_type=qubit_type,
            qubit_nodes=qubit_nodes,
            coupler=coupler,
            EJ_GHz=EJ_GHz,
            qubit_frequency_GHz=qubit_frequency_GHz,
            anharmonicity_MHz=anharmonicity_MHz,
            resonator_frequency_GHz=resonator_frequency_GHz,
            resonator_type=resonator_type,
            Z0=Z0,
        )

    @mcp.tool()
    async def solve_coupled_transmon_hamiltonian(
        targets: dict[str, float],
        reference_matrix_fF: list[list[float]] | None = None,
        layout_matrix_fF: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Two coupled transmons, each read out by a half-wave resonator: solve the full Hamiltonian inverse problem.

        Targets (GHz, f = omega / 2 pi; the keys of the two-transmon example): ``f_q_A``, ``eta_A`` (< 0), ``f_q_B``, ``eta_B``,
        ``f_r_A``, ``f_r_B``, ``J``, ``chi_A``, ``chi_B`` (magnitudes) and optionally ``kappa_A``, ``kappa_B``. They are the
        parameters of the Hamiltonian (exact charge-basis transmons, bare resonators, J = K n01_A n01_B, g = G n01 with the
        exact matrix elements); chi = half the dispersive shift from the full Hamiltonian (6 levels per transmon, 5 photons
        per resonator, counter-rotating terms kept: no RWA). Takes a few seconds.

        Returns E_J, E_C, E_J/E_C, L_J, I_c, n01, G, g, |chi| per transmon, K and J; the check of every target; the target
        Maxwell matrix (fF; ground, cross A, cross B, claw A, claw B) with its five controlled capacitances (C_sigma A/B,
        C_AB, C_g A/B); ``find_closest_designs`` arguments for the resonators with the by-hand length / kappa
        correction for your claw; the conventions and validity caveats. The resonator lines are 50 ohm (fixed).

        Args:
            reference_matrix_fF: a predicted 5 x 5 matrix of your layout in that body order (claw-to-ground and
                parasitics are taken from it); default: an ideal reference without them.
            layout_matrix_fF: a predicted 5 x 5 matrix to evaluate: E_J re-solved so f01 is on target, then every
                Hamiltonian miss (% and MHz), the miss of each controlled capacitance, and the line frequency and node
                capacitance each resonator needs (inputs of the resonator correction).
        """
        return await asyncio.to_thread(coupled_transmon_design, targets, reference_matrix_fF, layout_matrix_fF)

    @mcp.tool()
    async def generate_two_transmon_gds(
        p: list[float],
        out_path: str,
        variant: str = "gap",
        predict: bool = False,
        model: str = "QCap",
        threads: int | None = None,
        eps_r: float = 11.45,
        thickness_um: float = 350.0,
    ) -> dict[str, Any]:
        """Draw The two-Xmon layout and write its GDS (``[ml]`` extra); optionally predict its Maxwell matrix.

        Args:
            p: ``[L_A, L_B, claw fraction A, claw fraction B, coupling]``: arm lengths (um; set C_sigma), claw fractions
                (0-1 of the arm; set G), coupling = the gap between the arm tips (``gap``) or the capacitor's lead length
                (``interdigital``), um. the two-transmon example starts from [160, 150, 0.8, 0.8, 25] (gap) / [..., 15] (interdigital).
            out_path: the GDS to write (qgeometry export, ground 1/0, conductors 1/10).
            variant: ``gap`` or ``interdigital``.
            predict: also predict the 5 x 5 Maxwell matrix (ground, cross A, cross B, claw A, claw B) with ``model``:
                ``QCap`` to confirm a design (~seconds), ``QCap-Fast`` only for step directions.
            eps_r / thickness_um: the substrate of the prediction (default the training stack, Si 11.45, 350 um).

        Also returns ``ground_area_check`` (ground area in the file vs read by the model; they must agree). The layout is
        outside the trained families (see ``validity`` in the result).
        """
        return await asyncio.to_thread(
            two_transmon_gds, p, variant, out_path, predict, model, threads, eps_r, thickness_um
        )

    @mcp.tool()
    async def correct_half_wave_resonator(
        row_frequency_GHz: float,
        row_kappa_kHz: float,
        row_length_um: float,
        f_r_GHz: float,
        line_frequency_GHz: float,
        node_capacitance_fF: float,
        row_claw_capacitance_fF: float | None = None,
        claw_options: dict[str, Any] | None = None,
        model: str = "QCap",
        threads: int | None = None,
        eps_r: float = 11.45,
        thickness_um: float = 350.0,
    ) -> dict[str, Any]:
        """Adapt a SQuADDS half-wave resonator row to your claw: the resonator length and kappa (the two-transmon example's resonator step).

        A row's frequency includes the loading of its own claw. With the lumped lambda/2 model SQuADDS uses for half-wave
        resonators (Z0 = 50 ohm): the row's claw capacitance (``row_claw_capacitance_fF``, or predicted from
        ``claw_options`` with ``model``, default ``QCap``) gives the row's bare line frequency; the length is scaled to the line
        frequency your circuit needs, and kappa by (f_r / f_row)^2 * C_total_row / C_node (fixed feedline coupler).

        Args:
            row_frequency_GHz / row_kappa_kHz / row_length_um: the row from ``find_closest_designs`` (half-wave,
                ``system_type="qubit_cavity"``; length = the resonator's ``cpw_opts`` ``total_length``).
            f_r_GHz: your resonator target. line_frequency_GHz / node_capacitance_fF: this qubit's entries of
                ``layout.line_frequency_GHz`` / ``layout.node_capacitance_fF`` from ``solve_coupled_transmon_hamiltonian``
                with your layout matrix.
            claw_options: the row's ``claw_opts`` (needs the ``[ml]`` extra), or pass ``row_claw_capacitance_fF``.
        """
        return await asyncio.to_thread(
            resonator_correction,
            row_frequency_GHz,
            row_kappa_kHz,
            row_length_um,
            f_r_GHz,
            line_frequency_GHz,
            node_capacitance_fF,
            row_claw_capacitance_fF,
            claw_options,
            model,
            threads,
            eps_r,
            thickness_um,
        )


__all__ = ["QUBIT_PRESETS", "register_capacitance_design_tools", "hamiltonian_from_maxwell"]
