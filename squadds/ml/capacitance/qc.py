"""Label and layout QC for planar capacitance data: one set of rules for contributions, dataset builds and training shards.

Every check reports instead of raising; ``QCReport.ok`` is False when any check failed (an error). The rules are those of the
dataset (``PLAN_DATASET_HF.md`` §4.2):

- reject: matrix not square over ``net_order``, missing / duplicate / misplaced net names, non-finite entries, asymmetry
  (rel > 1e-3; Q3D exports are exactly symmetric, other solvers such as Palace are not; > 1e-6 warns, the matrix is then
  symmetrized), diagonal <= 0, off-diagonal >= 0 (a zero only when flagged), signal block not positive definite, a conductor
  count from the GDS that differs from the matrix size (nets are never merged), unreadable GDS, a duplicate of a held-out
  test design (it would leak the test label);
- warn: signal-row closure ``|sum_j C_ij| / C_ii`` > 5 %, not converged, non-monotone pass history, unusual stack, duplicates
  of non-test designs, label entries far from ``optimized_physics_prior`` (physics sanity; never a rejection).

Only numpy is imported at module level; gdstk / shapely are imported by the GDS checks, the physics prior by
``prior_sanity`` (needs ``pip install 'squadds[ml]'``).
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GROUND = "G"
TO_FF = {"F": 1e15, "nF": 1e6, "pF": 1e3, "fF": 1.0, "aF": 1e-3}
SYM_RTOL = 1e-3
SYM_WARN = 1e-6
CLOSURE_WARN = 0.05
PRIOR_FLOOR_FF = 0.1
PRIOR_RATIO_BAND = (0.5, 2.0)
TRAINING_STACK = {"eps_r": 11.45, "substrate_um": 350.0}
STACK_RANGES = {"eps_r": (1.0, 30.0), "substrate_um": (5.0, 2000.0), "freq_GHz": (0.1, 50.0)}
GDS_UNITS_M = (1e-6, 1e-9)
DEFAULT_LAYERS = {"ground": [(1, 0)], "conductor": [(1, 10)]}
_HEADER = re.compile(r"\s*C\(([^,]+),([^)]+)\)(\*1e15)?\s*\[(\w+)\]")


@dataclass
class QCReport:
    """Result of a QC run: ``ok`` (no error), ``errors``, ``warnings``, ``fixes`` (applied automatically) and the measured
    values in ``info``. ``to_markdown()`` renders it for a validation report."""

    subject: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(c.ok for c in self.children)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def extend(self, other: QCReport, prefix: str = "") -> QCReport:
        """Fold another report's errors / warnings / fixes / info into this one (messages prefixed)."""
        p = f"{prefix}: " if prefix else ""
        self.errors += [p + m for m in other.errors]
        self.warnings += [p + m for m in other.warnings]
        self.fixes += [p + m for m in other.fixes]
        self.info.update(other.info)
        return self

    def all_errors(self) -> list[str]:
        return self.errors + [f"{c.subject}: {m}" for c in self.children for m in c.all_errors()]

    def all_warnings(self) -> list[str]:
        return self.warnings + [f"{c.subject}: {m}" for c in self.children for m in c.all_warnings()]

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "fixes": self.fixes,
            "info": _jsonable(self.info),
            "children": [c.to_dict() for c in self.children],
        }

    def to_markdown(self, title: str | None = None) -> str:
        n_err, n_warn = len(self.all_errors()), len(self.all_warnings())
        out = [f"# {title or 'QC report: ' + (self.subject or '')}".rstrip(), ""]
        out.append(f"**Result: {'PASS' if self.ok else 'FAIL'}** ({n_err} errors, {n_warn} warnings)")
        out.append("")
        out += _md_block(self, level=2)
        for c in self.children:
            out.append(f"## {c.subject}: {'pass' if c.ok else 'FAIL'}")
            out.append("")
            out += _md_block(c, level=3)
        return "\n".join(out).rstrip() + "\n"

    def __repr__(self) -> str:
        return (
            f"QCReport({self.subject!r}, ok={self.ok}, errors={len(self.all_errors())}, "
            f"warnings={len(self.all_warnings())})"
        )


def _md_block(r: QCReport, level: int) -> list[str]:
    out = []
    for name, items in (("Errors", r.errors), ("Warnings", r.warnings), ("Fixed automatically", r.fixes)):
        if items:
            out.append(f"{'#' * level} {name}")
            out += [f"- {m}" for m in items]
            out.append("")
    if r.info:
        out.append(f"{'#' * level} Measured")
        out += [f"- `{k}`: {_fmt(v)}" for k, v in r.info.items()]
        out.append("")
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(_jsonable(v))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ---------------------------------------------------------------------------------------------------------------- units
def to_fF(values, unit: str) -> np.ndarray:
    """Values in ``unit`` (F, nF, pF, fF, aF; ``"pF*1e15"`` = the exp12 re-export header ``C(a,b)*1e15 [pF]``, whose values
    are pF x 1e15) converted to fF."""
    u = str(unit).strip()
    x15 = u.endswith("*1e15")
    u = u.replace("*1e15", "").strip("[] ")
    if u not in TO_FF:
        raise ValueError(f"unknown capacitance unit {unit!r}; use one of {sorted(TO_FF)}")
    return np.asarray(values, float) * TO_FF[u] * (1e-15 if x15 else 1.0)


def unit_plausibility(C_fF, declared: str) -> str | None:
    """A warning when a matrix in fF looks like it was given in another unit (entries of ~1e-15 or ~1e3 x the usual size)."""
    a = np.abs(np.asarray(C_fF, float))
    a = a[np.isfinite(a) & (a > 0)]
    if not a.size:
        return None
    m = float(np.median(a))
    if m < 1e-9:
        return f"median |C| = {m:.3g} fF after conversion from {declared}: the values look like farads"
    if m > 1e5:
        return f"median |C| = {m:.3g} fF after conversion from {declared}: implausibly large for a planar component"
    return None


# ---------------------------------------------------------------------------------------------------------------- readers
def read_cap_matrix(path) -> dict:
    """Q3D ``cap_matrix.csv`` (one row per adaptive pass; ``C(a,b) [unit]`` columns in any unit, also the ``*1e15`` header)
    -> ``{"names" (G first, then first appearance), "passes" [(pass index, N x N fF)], "units", "freq_GHz"}``. Incomplete rows
    (pre-allocated passes) are kept as NaN matrices and skipped by the QC."""
    rows = [r for r in csv.reader(open(path, encoding="utf-8")) if r]
    if not rows:
        raise ValueError(f"{path}: empty file")
    hdr, data = rows[0], rows[1:]
    cols, order, units = {}, [], set()
    i_pass = i_freq = None
    for j, h in enumerate(hdr):
        hs = h.strip().strip('"')
        if hs.lower().startswith("pass"):
            i_pass = j
        elif hs.lower().startswith("freq"):
            i_freq = j
        m = _HEADER.match(hs)
        if not m:
            continue
        a, b = m.group(1).strip(), m.group(2).strip()
        unit = m.group(4) + ("*1e15" if m.group(3) else "")
        units.add(unit)
        cols[(a, b)] = (j, float(to_fF(1.0, unit)))
        for x in (a, b):
            if x not in order:
                order.append(x)
    if not cols:
        raise ValueError(f"{path}: no C(a,b) [unit] columns")
    names = ([GROUND] if GROUND in order else []) + [x for x in order if x != GROUND]
    passes, freq = [], None
    for k, r in enumerate(data):
        C = np.full((len(names), len(names)), np.nan)
        for ia, a in enumerate(names):
            for ib, b in enumerate(names):
                jj = cols.get((a, b)) or cols.get((b, a))
                if jj is None:
                    continue
                try:
                    C[ia, ib] = float(r[jj[0]]) * jj[1]
                except (ValueError, IndexError):
                    pass
        try:
            p = int(float(r[i_pass])) if i_pass is not None else k + 1
        except (ValueError, IndexError):
            p = k + 1
        if i_freq is not None:
            try:
                freq = float(r[i_freq])
            except (ValueError, IndexError):
                pass
        passes.append((p, C))
    return {"names": names, "passes": passes, "units": sorted(units), "freq_GHz": freq}


def read_conv(path) -> dict:
    """Q3D ``.conv`` convergence file -> pass history, passes completed, delta target and final delta (%)."""
    t = Path(path).read_text(encoding="utf-8")

    def g(k, f=float):
        m = re.search(rf"{k}\s*:\s*(\S+)", t)
        try:
            return f(m.group(1)) if m else None
        except ValueError:
            return None

    hist = []
    for m in re.finditer(r"^\s*(\d+)\|\s*(\d+)\|\s*([^|]+)\|", t, re.M):
        d = m.group(3).strip()
        hist.append(
            {"pass": int(m.group(1)), "n_triangles": int(m.group(2)), "delta_pct": None if d == "N/A" else float(d)}
        )
    return {
        "pass_history": hist,
        "passes_completed": g("Completed", int),
        "delta_target_pct": g("Target"),
        "delta_pct_final": g("Current"),
    }


# ---------------------------------------------------------------------------------------------------------------- labels
def label_qc(
    C,
    names,
    net_order=None,
    unit: str = "fF",
    passes=None,
    convergence: dict | None = None,
    flagged_zero=(),
    sym_rtol: float = SYM_RTOL,
    closure_warn: float = CLOSURE_WARN,
    subject: str = "",
) -> QCReport:
    """QC of one Maxwell matrix ``C`` (``names`` = its row order; any unit, converted to fF).

    net_order: the declared net order (G first); the matrix must be square over exactly these names. passes: optional
    ``[(pass index, matrix in the same names / unit)]``, the last complete pass must equal ``C``. convergence: optional
    ``{"delta_pct_final", "delta_target_pct", "passes_completed"}``. flagged_zero: pairs ``(a, b)`` whose exact zero is
    explained (e.g. a lattice entry below the solver floor). ``info["matrix_fF"]`` holds the matrix in ``net_order``."""
    r = QCReport(subject)
    names = [str(x) for x in names]
    try:
        C = to_fF(C, unit)
    except ValueError as e:
        r.error(str(e))
        return r
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        r.error(f"matrix is not square: shape {C.shape}")
        return r
    if C.shape[0] != len(names):
        r.error(f"matrix size {C.shape[0]} does not match {len(names)} net names {names}")
        return r
    if len(set(names)) != len(names):
        r.error(f"duplicate net names {names}")
        return r
    order = [str(x) for x in (net_order if net_order is not None else names)]
    if GROUND not in order:
        r.error(f"ground net {GROUND!r} missing from the net order {order}")
        return r
    if order[0] != GROUND:
        r.error(f"ground net {GROUND!r} must be first in the net order (got {order})")
    if set(order) != set(names) or len(order) != len(names):
        r.error(f"matrix nets {sorted(names)} differ from the declared net order {order}")
        return r
    ix = [names.index(n) for n in order]
    C = C[np.ix_(ix, ix)]
    names = order
    n = len(names)
    r.info["n_conductors"] = n
    r.info["net_order"] = names
    if not np.isfinite(C).all():
        r.error("non-finite entries in the matrix")
        return r
    w = unit_plausibility(C, unit)
    if w:
        r.warn(w)
    scale = float(np.abs(C).max()) or 1.0
    sym = float(np.abs(C - C.T).max() / scale)
    r.info["symmetry_rel"] = sym
    if sym > sym_rtol:
        r.error(f"asymmetric: max |C - C^T| / max |C| = {sym:.3g} > {sym_rtol:g}")
    elif sym > SYM_WARN:
        r.warn(f"slightly asymmetric (max |C - C^T| / max |C| = {sym:.3g}); symmetrized as (C + C^T) / 2")
    Cs = 0.5 * (C + C.T)
    d = np.diag(Cs)
    if not np.all(d > 0):
        r.error(f"non-positive diagonal entries on {[names[i] for i in np.flatnonzero(d <= 0)]}")
    flagged = {frozenset((str(a), str(b))) for a, b in flagged_zero}
    pos, zero = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if Cs[i, j] > 0:
                pos.append(f"{names[i]}-{names[j]} ({Cs[i, j]:.4g} fF)")
            elif Cs[i, j] == 0 and frozenset((names[i], names[j])) not in flagged:
                zero.append(f"{names[i]}-{names[j]}")
    if pos:
        r.error(f"positive off-diagonal entries (Maxwell signs need C_ij < 0): {pos[:5]}")
    if zero:
        r.error(f"exactly-zero off-diagonal entries not flagged as explained: {zero[:5]}")
    sig = np.array([x != GROUND for x in names])
    if sig.any() and np.all(d[sig] > 0):
        clo = np.abs(Cs.sum(1)[sig]) / d[sig]
        r.info["closure_max_rel"] = float(clo.max())
        if clo.max() > closure_warn:
            worst = names[int(np.flatnonzero(sig)[int(np.argmax(clo))])]
            r.warn(f"signal-row closure |sum_j C_ij| / C_ii = {clo.max():.3g} on {worst} (> {closure_warn:.0%})")
        eig = float(np.linalg.eigvalsh(Cs[np.ix_(sig, sig)]).min())
        r.info["signal_block_min_eig_fF"] = eig
        if eig <= 0:
            r.error(f"signal block not positive definite (min eigenvalue {eig:.4g} fF)")
    if passes is not None:
        _pass_history(r, passes, C, unit, names, ix)
    if convergence:
        _convergence(r, convergence)
    r.info["matrix_fF"] = Cs
    return r


def _pass_history(r: QCReport, passes, C, unit, order, ix) -> None:
    idx = [int(p) for p, _ in passes]
    r.info["passes"] = len(idx)
    if idx != sorted(idx) or len(set(idx)) != len(idx):
        r.warn(f"pass indices are not strictly increasing: {idx}")
    complete = [(p, to_fF(M, unit)) for p, M in passes if np.isfinite(np.asarray(M, float)).all()]
    if not complete:
        r.error("no complete adaptive pass in the pass history")
        return
    last = complete[-1][1][np.ix_(ix, ix)]
    if not np.allclose(last, C, rtol=1e-9, atol=1e-12):
        r.error("the label is not the last complete pass of the pass history")
    if len(complete) >= 2:
        prev = complete[-2][1][np.ix_(ix, ix)]
        big = np.abs(last) >= 0.1
        if big.any():
            r.info["last_pass_change_max_pct"] = float(np.max(np.abs(last - prev)[big] / np.abs(last)[big]) * 100)


def _convergence(r: QCReport, conv: dict) -> None:
    fin, tgt = conv.get("delta_pct_final"), conv.get("delta_target_pct")
    r.info["delta_pct_final"], r.info["delta_target_pct"] = fin, tgt
    if conv.get("passes_completed") is not None:
        r.info["passes_completed"] = conv["passes_completed"]
    if fin is None or tgt is None:
        r.warn("convergence not recorded (final delta or target missing)")
    elif fin > tgt:
        r.warn(f"not converged: final delta {fin:g} % > target {tgt:g} %")
    r.info["converged"] = None if fin is None or tgt is None else bool(fin <= tgt)


# ---------------------------------------------------------------------------------------------------------------- layouts
def geom_hash(path) -> str:
    """``"sha256:<hex>"`` of the canonical polygon multiset of a GDS (layer, datatype, vertices rounded to 1 nm, sorted): the
    dataset's duplicate key, identical to the ``geom_hash`` column of ``SQuADDS/SQuADDS_PlanarCapacitance``."""
    import gdstk

    lib = gdstk.read_gds(str(path))
    s = lib.unit / 1e-9
    polys = [p for c in lib.top_level() for p in c.polygons]
    key = sorted((p.layer, p.datatype, np.round(p.points * s).astype(np.int64).tobytes()) for p in polys)
    h = hashlib.sha256()
    for la, dt, b in key:
        h.update(f"{la}/{dt}:".encode())
        h.update(b)
    return "sha256:" + h.hexdigest()


def _components(polys) -> list[list[int]]:
    import shapely

    n = len(polys)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n > 1:
        ii, jj = shapely.STRtree(polys).query(polys, predicate="intersects")
        for a, b in zip(ii, jj):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _layers(x) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in la) for la in x]


def gds_qc(path, layer_map: dict | None = None, nets: dict | None = None, subject: str = "") -> QCReport:
    """GDS readback: readable by gdstk, one top cell, unit 1 um or 1 nm, declared layers present, every polygon valid, and the
    conductor count (``info["n_conductors"]``, ground included) inferred from the layout.

    Conductors: the connected bodies of the conductor-layer polygons (touching polygons are one metal body, hence one net), in
    file order, unless ``nets`` = ``{net name: [conductor-layer polygon indices in file order]}`` assigns polygons explicitly.
    Bodies touching the ground metal are at ground potential (warned). ``info["geom_hash"]`` is the dataset duplicate key."""
    r = QCReport(subject or str(path))
    try:
        import gdstk
        import shapely
        from shapely.geometry import Polygon
    except ImportError as e:
        r.error(f"GDS checks need gdstk and shapely ({e}); install them with pip install 'squadds[ml]'")
        return r
    lm = layer_map or DEFAULT_LAYERS
    gl, cl = set(_layers(lm.get("ground", []))), set(_layers(lm.get("conductor", [])))
    try:
        lib = gdstk.read_gds(str(path))
    except Exception as e:  # noqa: BLE001
        r.error(f"gdstk cannot read the file: {e}")
        return r
    tops = lib.top_level()
    r.info["top_cells"] = [c.name for c in tops]
    if len(tops) != 1:
        r.error(f"{len(tops)} top cells ({[c.name for c in tops][:5]}); exactly one is required")
        if not tops:
            return r
    r.info["unit_m"], r.info["precision_m"] = lib.unit, lib.precision
    if not any(math.isclose(lib.unit, u, rel_tol=1e-9) for u in GDS_UNITS_M):
        r.error(f"GDS user unit {lib.unit:g} m: use 1 um or 1 nm")
    if not (1e-12 <= lib.precision <= 1e-9):
        r.warn(f"GDS precision {lib.precision:g} m (expected 1 nm or finer)")
    top = tops[0]
    raw = (
        top.get_polygons(apply_repetitions=True, include_paths=True) if (top.references or top.paths) else top.polygons
    )
    u = lib.unit / 1e-6
    present = sorted({(p.layer, p.datatype) for p in raw})
    r.info["layers"] = present
    missing = sorted((gl | cl) - set(present))
    if missing:
        r.error(f"declared layers {missing} have no polygons (layers present: {present})")
    gpolys, cpolys, invalid = [], [], 0
    for p in raw:
        key = (p.layer, p.datatype)
        if key not in gl and key not in cl:
            continue
        if len(p.points) < 3:
            invalid += 1
            continue
        g = Polygon(p.points * u)
        if g.area <= 0 or (not g.is_valid and abs(shapely.make_valid(g).area - g.area) > 1e-6 * g.area):
            invalid += 1  # keyhole polygons (ground planes with holes) touch themselves without changing area: accepted
        (gpolys if key in gl else cpolys).append(g)
    if invalid:
        r.error(
            f"{invalid} invalid polygon(s) on the ground / conductor layers (self-intersecting, degenerate or < 3 vertices)"
        )
    r.info["n_polygons"] = len(gpolys) + len(cpolys)
    if not gpolys or not cpolys:
        if not gpolys:
            r.error("no ground polygons")
        if not cpolys:
            r.error("no conductor polygons")
        r.info["n_conductors"] = 1 if gpolys else 0
        return r
    cvalid = [shapely.make_valid(g) if not g.is_valid else g for g in cpolys]
    if nets:
        idx = sorted(i for v in nets.values() for i in v)
        if idx != list(range(len(cvalid))):
            r.error(f"nets map must assign every conductor-layer polygon exactly once (0..{len(cvalid) - 1})")
        bodies = [shapely.union_all([cvalid[i] for i in v if 0 <= i < len(cvalid)]) for v in nets.values()]
        r.info["conductor_names"] = list(nets)
    else:
        bodies = [shapely.union_all([cvalid[i] for i in comp]) for comp in _components(cvalid)]
        if len(bodies) < len(cvalid):
            r.info["touching_polygons_joined"] = len(cvalid) - len(bodies)
    G = shapely.union_all([shapely.make_valid(g) if not g.is_valid else g for g in gpolys])
    shorted = [i for i, b in enumerate(bodies) if shapely.intersects(b, G)]
    if shorted:
        r.warn(
            f"{len(shorted)} conductor body(ies) touch the ground metal (bodies {shorted}): they are at ground potential"
        )
    r.info["n_conductors"] = 1 + len(bodies) - len(shorted)
    r.info["n_conductors_raw"] = 1 + len(bodies)
    metals = [G] + [b for i, b in enumerate(bodies) if i not in set(shorted)]
    if len(metals) > 1:
        gap = min(shapely.distance(metals[i], metals[j]) for i in range(len(metals)) for j in range(i + 1, len(metals)))
        r.info["min_gap_um"] = float(gap)
    try:
        r.info["geom_hash"] = geom_hash(path)
    except Exception as e:  # noqa: BLE001
        r.warn(f"geom_hash not computed: {e}")
    return r


def conductor_count_qc(n_layout: int, net_order, subject: str = "") -> QCReport:
    """The matrix must cover every conductor of the layout: N from the GDS (ground included) == len(net_order)."""
    r = QCReport(subject)
    if int(n_layout) != len(net_order):
        r.error(
            f"the layout has {n_layout} conductors (ground included) but the matrix has {len(net_order)} nets {list(net_order)}: "
            "every conductor must be its own net (nets are never merged)"
        )
    return r


# ---------------------------------------------------------------------------------------------------------------- dataset
def reference_hashes(db=None, local_dir=None, revision: str | None = None):
    """``design_id`` / ``geom_hash`` / ``split`` of every design of the dataset (``PlanarCapacitanceDB``; HF or a local copy)."""
    if db is None:
        from .dataset import PlanarCapacitanceDB

        db = PlanarCapacitanceDB(revision=revision, local_dir=local_dir)
    return db.read("designs", columns=["design_id", "geom_hash", "split"])


def duplicate_qc(hashes: dict, reference=None, subject: str = "duplicates") -> QCReport:
    """hashes: ``{design: geom_hash}`` of a batch; reference: a DataFrame with ``design_id``, ``geom_hash``, ``split`` (the
    dataset), or None (only duplicates inside the batch are checked; warned). A duplicate of a ``test`` design is an error."""
    r = QCReport(subject)
    seen: dict = {}
    for d, h in hashes.items():
        if h is None:
            continue
        if h in seen:
            r.warn(f"{d} has the same geometry as {seen[h]} in this batch")
        else:
            seen[h] = d
    if reference is None:
        r.warn("no dataset reference given: duplicates against SQuADDS_PlanarCapacitance were not checked")
        return r
    ref = reference[reference["geom_hash"].isin(set(seen))]
    test_dups, other_dups = [], []
    for h, sub in ref.groupby("geom_hash"):
        d = seen[h]
        ids = sorted(map(str, sub["design_id"]))
        if (sub["split"] == "test").any():
            test_dups.append(f"{d} = {ids[:3]}")
        else:
            other_dups.append(f"{d} = {ids[:3]} ({', '.join(sorted(set(map(str, sub['split']))))})")
    if test_dups:
        r.error(f"duplicate of a held-out test design (would leak its label): {test_dups}")
    if other_dups:
        r.warn(f"exact duplicates of dataset designs (kept, same group): {other_dups}")
    r.info["dataset_duplicates"] = len(test_dups) + len(other_dups)
    return r


def stack_qc(stack: dict, freq_GHz: float | None = None, subject: str = "stack") -> QCReport:
    """eps_r in [1, 30], substrate in [5, 2000] um, frequency in [0.1, 50] GHz; a stack other than the training stack
    (Si 11.45, 350 um) is accepted and recorded (warned: the released models are served for the training stack)."""
    r = QCReport(subject)
    vals = {"eps_r": stack.get("eps_r"), "substrate_um": stack.get("substrate_um"), "freq_GHz": freq_GHz}
    for k, (lo, hi) in STACK_RANGES.items():
        v = vals.get(k)
        if v is None:
            if k != "freq_GHz":
                r.error(f"stack field {k!r} missing")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            r.error(f"stack field {k!r} is not a number: {v!r}")
            continue
        r.info[k] = v
        if not lo <= v <= hi:
            r.warn(f"{k} = {v:g} outside the usual range [{lo:g}, {hi:g}]")
    eps, t = r.info.get("eps_r"), r.info.get("substrate_um")
    if eps is not None and t is not None:
        on = math.isclose(eps, TRAINING_STACK["eps_r"], rel_tol=1e-3) and math.isclose(
            t, TRAINING_STACK["substrate_um"], rel_tol=1e-3
        )
        r.info["training_stack"] = on
        if not on:
            r.warn(
                f"stack eps_r {eps:g}, {t:g} um differs from the training stack (Si 11.45, 350 um): recorded as off-stack"
            )
    return r


def prior_sanity(
    C_fF,
    net_order,
    gds,
    layer_map: dict | None = None,
    band=PRIOR_RATIO_BAND,
    floor_fF: float = PRIOR_FLOOR_FF,
    threads: int = 2,
    prior=None,
    subject: str = "physics sanity",
) -> QCReport:
    """Ratio of each label entry to ``optimized_physics_prior`` of the layout (report only, never an error).

    The label's conductors (``net_order[1:]``) are matched to the layout's in GDS file order. Entries with ``|C| >= floor_fF``
    whose ratio falls outside ``band`` are warned. prior: a precomputed ``PriorResult`` (else computed here, needs [ml])."""
    r = QCReport(subject)
    C = np.asarray(C_fF, float)
    if prior is None:
        try:
            from . import priors

            lm = layer_map or DEFAULT_LAYERS
            prior = priors.optimized_physics_prior(
                str(gds),
                threads=threads,
                ground_layers=_layers(lm.get("ground", [])),
                conductor_layers=_layers(lm.get("conductor", [])),
            )
        except ImportError as e:
            r.warn(f"physics sanity skipped (needs pip install 'squadds[ml]'): {e}")
            return r
        except Exception as e:  # noqa: BLE001
            r.warn(f"physics sanity skipped: the prior could not be computed ({e})")
            return r
    P = np.asarray(prior.C_fF, float)
    if P.shape != C.shape:
        r.warn(f"physics sanity skipped: prior is {P.shape[0]} x {P.shape[0]}, label {C.shape[0]} x {C.shape[0]}")
        return r
    names = list(net_order)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = C / P
    iu = np.triu_indices(len(C))
    far = []
    for i, j in zip(*iu):
        if abs(C[i, j]) < floor_fF or not np.isfinite(ratio[i, j]):
            continue
        if not band[0] <= ratio[i, j] <= band[1]:
            far.append(f"{names[i]}-{names[j]}: label {C[i, j]:.4g} fF, prior {P[i, j]:.4g} fF (x{ratio[i, j]:.3g})")
    mut = [(i, j) for i, j in zip(*iu) if i != j and abs(C[i, j]) >= floor_fF and np.isfinite(ratio[i, j])]
    if mut:
        rel = np.array([abs(ratio[i, j] - 1) for i, j in mut]) * 100
        r.info["prior_median_rel_diff_pct"] = float(np.median(rel))
        r.info["prior_mae_fF"] = float(np.mean([abs(C[i, j] - P[i, j]) for i, j in mut]))
        r.info["prior_rmse_fF"] = float(np.sqrt(np.mean([(C[i, j] - P[i, j]) ** 2 for i, j in mut])))
    r.info["prior_version"] = getattr(prior, "prior_version", "")
    r.info["prior_s"] = (getattr(prior, "timing_s", {}) or {}).get("total_s")
    r.info["prior_machine"] = getattr(prior, "machine", "")
    r.info["prior_threads"] = getattr(prior, "threads", None)
    if far:
        r.warn(
            f"{len(far)} label entries outside x{band[0]:g}..x{band[1]:g} of the physics prior (check nets / units): {far[:5]}"
        )
    return r
