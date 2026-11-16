"""Bridges from the planar capacitance models into the existing SQuADDS workflows.

- ``MLSimulator``: the call and result shape of the capacitance (Q3D / LOM) path of ``AnsysSimulator``
  (``simulate(device_dict)`` -> ``{"design", "sim_options", "sim_results"}``) plus ``fidelity="ml:<model>"`` and ``timing``.
  Input: a Qiskit Metal design or component, a SQuADDS row (rebuilt with the component registry) or a GDS path.
- ``predict_dataframe``: predictions for the rows of an Analyzer result (``Analyzer.predict_capacitance`` calls it).
- ``transmon_cross_capacitances`` / ``EC_MHz`` / ``g_MHz`` / ``EC_MHz_from_matrix``: feed ``squadds.calcs``.

Export (``export=``): ``"auto"`` (default) writes a component of a dataset family (``squadds.layouts.dataset_export``:
interdigital / paddle, concentric, star, Xmon, TransmonPocket, shielded concentric) exactly as the dataset was exported,
so the model sees the input it was trained on; any other layout (e.g. quantum-metal ``TransmonCross``, several
components) goes through ``export_qgeometry_gds`` (square ground domain, 169 um padding). ``"dataset"`` / ``"qgeometry"``
force one. The result records it in ``export``. Star rows of the SQuADDS view carry their style in
``notes.export_style``. Conductor bodies are named from the qgeometry they overlap (``planar_capacitance_view.QGEOMETRY_NETS``), so the
``sim_results`` keys match the SQuADDS configs (``north_to_south``, ``cross_to_claw``, ...).
"""

from __future__ import annotations

import importlib
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from squadds.database.planar_capacitance_view import QGEOMETRY_NETS, net_name, sim_results_from_matrix

QISKIT_METAL_CLASSES = {
    "TransmonCross": "qiskit_metal.qlibrary.qubits.transmon_cross",
    "TransmonPocket": "qiskit_metal.qlibrary.qubits.transmon_pocket",
}
GAP_OPTION_KEYS = ("cpw_gap", "claw_gap", "ground_spacing", "gap_ground", "ground_gap", "min_trace_gap", "cross_gap")
DEFAULT_CLEARANCE_UM = 5.0
EXPORTS = ("auto", "dataset", "qgeometry")


# ------------------------------------------------------------------------------------------------ rows -> components
def _clean(value):
    """Drop ``None`` / NaN entries (union-schema padding of HF rows) and numpy scalars from an options tree."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not _is_missing(v)}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def row_options_and_class(row, component_name: str | None = None) -> tuple[dict, str]:
    """(design_options, component class name) of a SQuADDS row: nested HF record, flattened DataFrame row (Series or
    dict) or an ``AnsysSimulator`` device dict."""
    from squadds.core.json_utils import deserialize_json_like

    if hasattr(row, "to_dict"):
        row = row.to_dict()
    rec = row["design"] if isinstance(row.get("design"), dict) else row
    opts = deserialize_json_like(rec.get("design_options", row.get("design_options")))
    if not isinstance(opts, dict):
        raise ValueError("row has no design_options")
    cls = component_name
    for key in ("component_class", "component_name", "coupler_type"):
        if cls is None and isinstance(rec.get(key), str) and rec.get(key):
            cls = rec[key]
    if cls is None and "cross_length" in opts:
        cls = "TransmonCross"
    if cls is None:
        raise ValueError("cannot tell the component class of this row: pass component_name=")
    return _clean(opts), cls


def component_class(name: str):
    """Qiskit Metal class of a dataset component name (SQuADDS registry first, then quantum-metal qubits)."""
    from squadds.components import get_component_class

    try:
        return get_component_class(name)
    except ValueError:
        if name in QISKIT_METAL_CLASSES:
            return getattr(importlib.import_module(QISKIT_METAL_CLASSES[name]), name)
        raise


def build_component(row, component_name: str | None = None, design=None, name: str = "component"):
    """Rebuild the component of a SQuADDS row in ``design`` (a new ``DesignPlanar`` by default)."""
    from qiskit_metal import designs

    opts, cls = row_options_and_class(row, component_name)
    design = design if design is not None else designs.DesignPlanar({}, overwrite_enabled=True)
    comp = component_class(cls)(design, name, options=opts)
    design.rebuild()
    return comp


# ------------------------------------------------------------------------------------------------ rendering and naming
def _length_um(design, value) -> float | None:
    try:
        v = float(design.parse_value(value)) * 1e3
    except Exception:  # noqa: BLE001
        return None
    return v if v > 0 else None


def default_clearance_um(design) -> float:
    """Signal-to-ground clearance for the exported ground: the smallest lead / claw / ground gap option of the components
    (the native etch stays authoritative; the clearance only closes open CPW ends)."""
    found = []

    def scan(opts):
        for k, v in opts.items():
            if isinstance(v, dict):
                scan(v)
            elif any(k.endswith(g) for g in GAP_OPTION_KEYS) and isinstance(v, (str, int, float)):
                for part in str(v).split(","):
                    um = _length_um(design, part.strip()) if part.strip() else None
                    if um:
                        found.append(um)

    for comp in design.components.values():
        scan(dict(comp.options))
    return min(found) if found else DEFAULT_CLEARANCE_UM


def render_gds(design, path, clearance_um: float | None = None) -> Path:
    """Export ``design`` to a GDS in the training layer convention (ground (1, 0), conductors (1, 10))."""
    from squadds.layouts.qmetal_gds import export_qgeometry_gds

    path = Path(path)
    clearance = float(clearance_um or default_clearance_um(design))
    export_qgeometry_gds(design, path, include_ground_domain=True, minimum_ground_clearance_um=clearance)
    return path


def _qgeometry_elements(design) -> list[tuple[str, str, str, Any]]:
    """(component class, component name, element name, geometry in µm) of every conductor qgeometry element."""
    from shapely import affinity

    out = []
    for table in ("poly", "path"):
        for r in design.qgeometry.tables[table].itertuples(index=False):
            if bool(r.helper) or bool(r.subtract):
                continue
            g = r.geometry
            if table == "path":
                g = g.buffer(float(r.width) / 2, cap_style="flat", join_style="round")
            comp = design._components[r.component]
            out.append((type(comp).__name__, comp.name, r.name, affinity.scale(g, 1e3, 1e3, origin=(0, 0))))
    return out


def body_labels(conductors, design) -> tuple[list[str], str | None]:
    """Dataset net label (``N``, ``S``, ``P1``, ...) of each conductor body, and the component class when the layout has a
    single component. Unmapped bodies keep the predictor's name ``C<k>``; with several components the labels are
    ``<component>:<element>``."""
    elems = _qgeometry_elements(design)
    classes = {e[0] for e in elems}
    single = next(iter(classes)) if len({e[1] for e in elems}) == 1 else None
    labels = []
    for k, body in enumerate(conductors):
        hits = [e for e in elems if body.intersection(e[3]).area > 1e-6 * max(e[3].area, 1e-12)]
        lab = None
        if single and hits:
            cands = {QGEOMETRY_NETS.get(single, {}).get(e[2]) for e in hits}
            cands.discard(None)
            lab = cands.pop() if len(cands) == 1 else None
        elif hits:
            lab = "+".join(sorted({f"{e[1]}:{e[2]}" for e in hits}))
        lab = lab or f"C{k + 1}"
        while lab in labels:
            lab = f"{lab}'"
        labels.append(lab)
    return labels, single


# ------------------------------------------------------------------------------------------------ simulator
class MLSimulator:
    """Capacitance from the planar capacitance models, in the result shape of ``AnsysSimulator``'s capacitance path.

        sim = MLSimulator(analyzer, device_dict)                       # like AnsysSimulator(analyzer, device_dict)
        res = sim.simulate()                                           # {"design", "sim_options", "sim_results", ...}
        MLSimulator(model="QCap-Fast").predict("chip.gds")                  # or a component, a design or a SQuADDS row

    model: ``"QCap"`` (default), ``"QCap-Fast"`` or a loaded ``CapacitanceModel``. ``predictor``: any object with ``predict(layout) -> CapacitanceResult`` (tests).
    export: ``"auto"`` (dataset export for the dataset families, else ``"qgeometry"``), ``"dataset"`` or
    ``"qgeometry"``; ``export_style`` / ``margin_gaps`` / ``ground_gap_um`` override the dataset export
    (``squadds.layouts.dataset_export.export_dataset_gds``; a row's ``notes.export_style`` is used when not set).
    """

    def __init__(
        self,
        analyzer=None,
        design_options=None,
        model="QCap",
        threads: int | None = None,
        predictor=None,
        clearance_um: float | None = None,
        workdir=None,
        export: str = "auto",
        export_style: str | None = None,
        margin_gaps: float | None = None,
        ground_gap_um: float | None = None,
        tier=None,
        **predict_kwargs,
    ):
        from .models import no_tier

        no_tier(tier)
        self.analyzer = analyzer
        self.device_dict = deepcopy(design_options) if design_options is not None else None
        self.model, self.threads = model or "QCap", threads
        self.clearance_um = clearance_um
        if export not in EXPORTS:
            raise ValueError(f"export must be one of {EXPORTS}")
        self.export, self.export_style = export, export_style
        self.margin_gaps, self.ground_gap_um = margin_gaps, ground_gap_um
        self.workdir = Path(workdir) if workdir is not None else None
        self.predict_kwargs = predict_kwargs
        self._predictor = predictor

    def __repr__(self) -> str:
        return f"MLSimulator(model={getattr(self.model, 'name', self.model)!r})"

    @property
    def predictor(self):
        if self._predictor is None:
            from squadds.ml.capacitance import CapacitancePredictor

            self._predictor = CapacitancePredictor(self.model, threads=self.threads)
        return self._predictor

    @property
    def fidelity(self) -> str:
        from .models import DISPLAY_NAMES, is_model_name, model_key

        m = getattr(self._predictor, "model", None) or self.model
        m = getattr(m, "name", m)
        return f"ml:{DISPLAY_NAMES[model_key(m)] if is_model_name(m) else m}"

    def simulate(self, device_dict=None, run_async: bool = False) -> dict:
        """``AnsysSimulator.simulate`` for single components (capacitance only)."""
        if run_async:
            raise NotImplementedError("MLSimulator predicts synchronously (about a second per design).")
        if self.analyzer is not None and isinstance(getattr(self.analyzer, "selected_system", None), list):
            raise NotImplementedError(
                "MLSimulator predicts capacitance matrices of single components; the eigenmode part of a qubit-cavity "
                "system stays with AnsysSimulator."
            )
        device_dict = device_dict if device_dict is not None else self.device_dict
        if device_dict is None:
            raise ValueError("no device dict: pass design_options= to MLSimulator or device_dict= to simulate()")
        name = None
        if self.analyzer is not None and not _row_declares_class(device_dict):
            name = getattr(self.analyzer, "selected_coupler", None) or getattr(
                self.analyzer, "selected_component_name", None
            )
        return self.predict(device_dict, component_name=name)

    def predict(self, obj, component_name: str | None = None) -> dict:
        """Prediction for a GDS path, a Qiskit Metal design or component, or a SQuADDS row / device dict."""
        from qiskit_metal.designs import QDesign
        from qiskit_metal.qlibrary.core import QComponent

        t0 = time.perf_counter()
        design, options, cls_name, gds, notes = None, None, None, None, {}
        if isinstance(obj, (str, os.PathLike)):
            gds = Path(obj)
        elif isinstance(obj, QComponent):
            design, options, cls_name = obj.design, dict(obj.options), type(obj).__name__
        elif isinstance(obj, QDesign):
            design = obj
        else:
            comp = build_component(obj, component_name)
            design, options, cls_name = comp.design, dict(comp.options), type(comp).__name__
            notes = row_notes(obj)
        warnings = []
        if notes.get("geometry_reproducible") is False:
            warnings.append(
                f"{notes.get('design_id', 'this row')}: the stored options do not rebuild the solved GDS exactly; the "
                "prediction is for the rebuilt layout"
            )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            export = {"convention": "gds", "path": str(gds)} if gds is not None else None
            if gds is None:
                out = (self.workdir or Path(tmp)) / f"{cls_name or 'design'}_{int(time.time() * 1e6)}.gds"
                gds, design, export = self._export(obj if isinstance(obj, QComponent) else design, out, notes)
            render_s = time.perf_counter() - t0
            from squadds.ml.capacitance._runtime import prepare_layout

            lay = prepare_layout(str(gds))
            res = self.predictor.predict(lay, **self.predict_kwargs)
        if design is not None:
            bodies, single = body_labels(lay.conductors, design)
            labels = ["G", *bodies]
        else:
            labels, single = list(res.names), None
        sim_results = sim_results_from_matrix(labels, res.matrix_fF, single or "")
        timing = {"render_s": render_s, **dict(res.timing_s), "wall_s": time.perf_counter() - t0}
        return {
            "design": {
                "design_options": _clean(options) if options is not None else None,
                "design_tool": "qiskit-metal",
                "component_class": cls_name,
                "gds": str(gds) if self.workdir is not None or export["convention"] == "gds" else None,
                "export": export,
            },
            "sim_options": {
                "sim_type": "cap_matrix",
                "setup": {"model": res.model, "threads": res.threads, "prior": res.prior_version},
                "layer_stack": dict(res.substrate),
                "simulator": f"squadds.ml.capacitance ({res.model})",
            },
            "sim_results": sim_results,
            "fidelity": f"ml:{res.model}",
            "timing": timing,
            "net_names": {lab: net_name(single or "", lab) for lab in labels},
            "checks": dict(res.checks),
            "warnings": warnings + list(res.warnings),
            "export": export,
            "prediction": res,
        }

    def _export(self, obj, out: Path, notes: dict):
        """Write ``obj`` (a component or a design) to ``out``: ``(gds, design whose qgeometry names the bodies, export
        record)``."""
        from qiskit_metal.designs import QDesign

        from squadds.layouts.dataset_export import dataset_export_layout, family_of, write_dataset_gds

        out.parent.mkdir(parents=True, exist_ok=True)
        family = family_of(obj)
        if self.export == "dataset" and family is None:
            raise ValueError("export='dataset' needs one component of a dataset family; use export='qgeometry'")
        if self.export == "qgeometry" or family is None:
            design = obj if isinstance(obj, QDesign) else obj.design
            clearance = float(self.clearance_um or default_clearance_um(design))
            gds = render_gds(design, out, clearance)
            return gds, design, {"convention": "qgeometry", "clearance_um": clearance}
        style = self.export_style or (notes.get("export_style") if family == "star" else None)
        gap = self.ground_gap_um if self.ground_gap_um is not None else notes.get("ground_gap_um")
        cell, design, settings = dataset_export_layout(obj, None, self.margin_gaps, style, gap)
        return write_dataset_gds(cell, out), design, settings


def row_notes(row) -> dict:
    """``notes`` of a SQuADDS row (nested record or flattened DataFrame row); {} for a device dict without notes."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    if not isinstance(row, dict):
        return {}
    notes = row.get("notes")
    if isinstance(notes, dict):
        return notes
    keys = ("design_id", "export_style", "geometry_reproducible", "ground_gap_um")
    return {k: row[k] for k in keys if k in row and not _is_missing(row[k])}


def _row_declares_class(row) -> bool:
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    rec = row["design"] if isinstance(row.get("design"), dict) else row
    return any(
        isinstance(rec.get(k), str) and rec.get(k) for k in ("component_class", "component_name", "coupler_type")
    )


def predict_dataframe(df, model="QCap", component_name=None, simulator=None, **kwargs):
    """One prediction per row of a SQuADDS DataFrame (closest designs, interpolated designs): a DataFrame with the same
    index, the predicted ``sim_results`` columns (suffix ``_ml``), ``fidelity``, ``ml_total_s`` and ``ml_result``."""
    import pandas as pd

    sim = simulator or MLSimulator(model=model, **kwargs)
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    out = []
    for _, row in df.iterrows():
        name = component_name if not _row_declares_class(row) else None
        r = sim.predict(row, component_name=name)
        rec = {f"{k}_ml": v for k, v in r["sim_results"].items() if k not in ("maxwell_matrix", "units")}
        rec.update(
            maxwell_matrix_ml=r["sim_results"]["maxwell_matrix"],
            fidelity=r["fidelity"],
            ml_total_s=r["timing"].get("wall_s"),
            ml_result=r,
        )
        out.append(rec)
    return pd.DataFrame(out, index=df.index)


# ------------------------------------------------------------------------------------------------ calcs helpers
def _sim_results_of(x) -> dict:
    if hasattr(x, "to_sim_results"):
        return x.to_sim_results()
    if hasattr(x, "to_dict"):
        x = x.to_dict()
    return x["sim_results"] if isinstance(x.get("sim_results"), dict) else x


def transmon_cross_capacitances(x) -> dict:
    """``cross_to_claw`` and ``cross_to_ground`` (fF) of a TransmonCross / Xmon prediction or row, the inputs of
    ``squadds.calcs.transmon_cross`` (EC, g). x: an MLSimulator result, a ``sim_results`` dict or a DataFrame row."""
    sr = _sim_results_of(x)
    missing = [k for k in ("cross_to_claw", "cross_to_ground") if sr.get(k) is None]
    if missing:
        raise KeyError(f"not a TransmonCross / Xmon result (missing {missing})")
    return {"cross_to_claw": float(sr["cross_to_claw"]), "cross_to_ground": float(sr["cross_to_ground"])}


def EC_MHz(x) -> float:
    """Charging energy (MHz) with the existing SQuADDS convention ``C_sum = cross_to_ground + cross_to_claw``
    (``squadds.calcs.transmon_cross.EC_numba``). A second claw is not included: use ``EC_MHz_from_matrix`` for it."""
    from squadds.calcs.transmon_cross import EC_numba

    c = transmon_cross_capacitances(x)
    return 1e3 * float(EC_numba(c["cross_to_claw"], c["cross_to_ground"]))


def g_MHz(x, EJ_GHz: float, f_r_GHz: float, res_type: str = "quarter", Z0: float = 50.0) -> float:
    """Qubit-claw coupling g (MHz) from ``squadds.calcs.transmon_cross.g_from_cap_matrix_numba`` with the predicted
    ``cross_to_ground`` (C_Q) and ``cross_to_claw`` (C_g)."""
    from squadds.calcs.transmon_cross import g_from_cap_matrix_numba

    c = transmon_cross_capacitances(x)
    return float(g_from_cap_matrix_numba(c["cross_to_ground"], c["cross_to_claw"], EJ_GHz, f_r_GHz, res_type, Z0))


def EC_MHz_from_matrix(C, names, nodes) -> float:
    """Charging energy (MHz) of a qubit mode from a full Maxwell matrix (fF, every other conductor held at ground).

    nodes: one conductor name (grounded transmon: C_sum = C_ii, every coupling included) or two (floating transmon, the
    differential mode: 1 / C_eff = (C^-1)_aa + (C^-1)_bb - 2 (C^-1)_ab of the two-pad block)."""
    from scipy.constants import Planck, e

    C = np.asarray(C, float)
    names = list(names)
    nodes = [nodes] if isinstance(nodes, str) else list(nodes)
    idx = [names.index(n) for n in nodes]
    if len(idx) == 1:
        c_eff = C[idx[0], idx[0]]
    elif len(idx) == 2:
        inv = np.linalg.inv(C[np.ix_(idx, idx)])
        c_eff = 1.0 / (inv[0, 0] + inv[1, 1] - 2 * inv[0, 1])
    else:
        raise ValueError("nodes: one conductor (grounded) or two (floating pair)")
    return e**2 / (2 * c_eff * 1e-15) / Planck * 1e-6
