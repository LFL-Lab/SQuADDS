"""GDS export in the convention of the planar capacitance dataset (``SQuADDS/SQuADDS_PlanarCapacitance``).

The networks were trained on this export, so a component exported here gives the model the input it was trained on.
Layers: ground with its etched pocket on (1, 0), one polygon per signal body on (1, 10), 2 um port stubs on (2, 0) /
(3, 0); 1 nm database unit. The chip is the component extent plus ``m`` ground gaps on each side:

- ``helpers`` convention (squadds2-helpers ``generate_gds.py``; interdigital / paddle exp6-9, concentric exp10, star
  exp11-12): interdigital chip = fingers + 2 m max(gap_ground) with m = 20, component centred; concentric chip =
  2 R_ground (+ CPW lengths in y) + 2 m min_trace_gap, star 2 R_pocket (+ cpw_length) + 2 m gap, m = 8; the Metal GDS
  renderer output, bodies merged per conductor; star layouts post-processed by ``squadds.components.export_fixups``
  (``export_style``).
- ``families`` convention (exp13-15 generator): chip = qgeometry extent + 2 m g, m = 8 (Xmon: g = max(cross_gap,
  ground_spacing); shielded concentric: g = ground_gap) and m = 4 for TransmonPocket (g = the pad-to-pocket gap, floored
  at 20 um); nets unioned from qgeometry, renderer ground, a port stub at the ``readout`` pin and end slots at the others.

``export_dataset_gds`` rebuilds the component from its class and options in a fresh design at the origin (orientation 0)
and reproduces the dataset GDS bit-exactly (``scripts/planar_capacitance/verify_component_gds.py``), except where the
ground gap was set from unrounded generator parameters (Xmon, TransmonPocket: pass ``ground_gap_um``) and for the star
styles under another GEOS version (see ``export_fixups``).
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

FAMILY_OF_CLASS = {
    "GeneralizedCapNInterdigital": "interdigital",
    "GeneralizedCapConcentric": "concentric",
    "GeneralizedCapStar": "star",
    "Xmon": "xmon",
    "TransmonPocket": "pocket",
    "ShieldedCapConcentric": "shielded_concentric",
}
CLASS_OF_FAMILY = {
    "interdigital": "GeneralizedCapNInterdigital",
    "paddle": "GeneralizedCapNInterdigital",
    "concentric": "GeneralizedCapConcentric",
    "star": "GeneralizedCapStar",
    "multipad_star": "GeneralizedCapStar",
    "xmon": "Xmon",
    "pocket": "TransmonPocket",
    "shielded_concentric": "ShieldedCapConcentric",
}
HELPERS_FAMILIES = ("interdigital", "paddle", "concentric", "star", "multipad_star")
MARGIN_GAPS = {
    "interdigital": 20.0,
    "paddle": 20.0,
    "concentric": 8.0,
    "star": 8.0,
    "multipad_star": 8.0,
    "xmon": 8.0,
    "pocket": 4.0,
    "shielded_concentric": 8.0,
}
POCKET_MIN_GROUND_GAP_UM = 20.0
PORT_LENGTH_UM = 2.0
FEED_PIN = "readout"
DATASET_LAYERS = {"ground": (1, 0), "signal": (1, 10), "port_north": (2, 0), "port_south": (3, 0)}


# ----------------------------------------------------------------------------------------------- resolution
def _component_of(obj):
    from qiskit_metal.designs import QDesign

    if isinstance(obj, QDesign):
        comps = list(obj.components.values())
        if len(comps) != 1:
            raise ValueError(f"the dataset export takes one component per layout; this design has {len(comps)}")
        return comps[0]
    return obj


def family_of(obj) -> str | None:
    """Dataset family of a component, a design with one component, or a class name; None outside the dataset."""
    if isinstance(obj, str):
        return FAMILY_OF_CLASS.get(obj)
    try:
        comp = _component_of(obj)
    except ValueError:
        return None
    return FAMILY_OF_CLASS.get(type(comp).__name__)


def default_star_style(options) -> str:
    """Export style of a star by its enabled pads: one pad = exp11 (``original``), more = exp12 (``reexport_gdspy``)."""
    pads = sum(int(x) for x in str(options.get("pad_enables", "1")).split(",") if x.strip())
    return "original" if pads <= 1 else "reexport_gdspy"


def _um(design, value) -> float:
    return float(design.parse_value(value)) * 1e3


def default_ground_gap_um(family: str, options, design) -> float | None:
    """Ground gap g of the ``families`` chip margin (um) from component options; None for the ``helpers`` families."""
    if family == "xmon":
        pads = options.get("connection_pads", {}) or {}
        spacing = [_um(design, p["ground_spacing"]) for p in pads.values() if "ground_spacing" in p]
        return max([_um(design, options["cross_gap"]), *spacing])
    if family == "pocket":
        pw, ph = _um(design, options["pocket_width"]), _um(design, options["pocket_height"])
        w, h, g = _um(design, options["pad_width"]), _um(design, options["pad_height"]), _um(design, options["pad_gap"])
        return max(min((pw - w) / 2, (ph - g) / 2 - h), POCKET_MIN_GROUND_GAP_UM)
    if family == "shielded_concentric":
        return _um(design, options["ground_gap"])
    return None


def generator_ground_gap_um(family: str, params: dict) -> float | None:
    """Ground gap (um) the exp13-15 generator used for the chip margin, from its unrounded ``generator.params``; the
    component options are rounded to 1 nm, so pass this as ``ground_gap_um`` to reproduce the dataset GDS exactly."""
    if family == "xmon":
        return float(max(params["cross_gap"], params["ground_spacing"]))
    if family == "pocket":
        p = params
        g = min((p["pocket_width"] - p["pad_width"]) / 2, (p["pocket_height"] - p["pad_gap"]) / 2 - p["pad_height"])
        return float(max(g, POCKET_MIN_GROUND_GAP_UM))
    if family == "shielded_concentric":
        return float(params["gap_g"])
    return None


def dataset_export_settings(obj, family=None, margin_gaps=None, export_style=None, ground_gap_um=None) -> dict:
    """The resolved export settings of a component: ``family``, ``convention`` (helpers / families), ``margin_gaps``,
    ``export_style`` (stars) and ``ground_gap_um`` (families)."""
    comp = _component_of(obj)
    family = family or FAMILY_OF_CLASS.get(type(comp).__name__)
    if family not in CLASS_OF_FAMILY:
        raise ValueError(
            f"{type(comp).__name__} is not a dataset family ({sorted(FAMILY_OF_CLASS)}); export it with "
            "squadds.layouts.export_qgeometry_gds"
        )
    options = dict(comp.options)
    star = family in ("star", "multipad_star")
    if export_style is not None and not star:
        raise ValueError(f"export_style applies to star layouts only, not {family}")
    if star:
        from squadds.components.export_fixups import EXPORT_STYLES

        export_style = export_style or default_star_style(options)
        if export_style not in EXPORT_STYLES:
            raise ValueError(f"export_style must be one of {EXPORT_STYLES}")
    helpers = family in HELPERS_FAMILIES
    if ground_gap_um is None and not helpers:
        ground_gap_um = default_ground_gap_um(family, options, comp.design)
    return {
        "convention": "dataset",
        "family": family,
        "exporter": "helpers" if helpers else "families",
        "margin_gaps": float(margin_gaps if margin_gaps is not None else MARGIN_GAPS[family]),
        "export_style": export_style,
        "ground_gap_um": None if helpers else float(ground_gap_um),
    }


# ----------------------------------------------------------------------------------------------- rendering
def _fake_junction(tmp: Path) -> Path:
    import gdstk

    path = tmp / "Fake_Junctions.GDS"
    if not path.is_file():
        lib = gdstk.Library(unit=1e-3, precision=1e-9)
        lib.new_cell("dummy_junction").add(gdstk.rectangle((0, 0), (1, 1), layer=0))
        lib.write_gds(str(path))
    return path


@contextlib.contextmanager
def _single_ground_strip():
    """Ground plane as one gdstk "not" (quantum-metal < 0.9); 0.9 subtracts it in vertical strips, so the dataset GDS
    (one ground polygon) would not be reproduced."""
    try:
        import gdstk
        from qiskit_metal.renderers.renderer_gds import gds_renderer
    except ImportError:
        yield
        return
    original = getattr(gds_renderer, "subtract_in_strips", None)
    if original is None:
        yield
        return

    def whole(base, cuts, layer, datatype, precision, strips=1):
        return gdstk.boolean(base, cuts, "not", layer=layer, datatype=datatype, precision=precision)

    gds_renderer.subtract_in_strips = whole
    try:
        yield
    finally:
        gds_renderer.subtract_in_strips = original


def _render(design, tmp: Path):
    import gdstk
    from qiskit_metal import Dict

    fd, name = tempfile.mkstemp(suffix=".gds", dir=tmp)
    os.close(fd)  # Windows cannot write or delete a file another handle keeps open
    raw = Path(name)
    r = design.renderers.gds
    r.options.path_filename = str(_fake_junction(tmp))
    off = Dict({"main": {int(L): False for L in design.qgeometry.get_all_unique_layers("main")}})
    r.options.cheese.view_in_file = off
    r.options.no_cheese.view_in_file = off
    with _single_ground_strip():
        if not r.export_to_gds(str(raw)):
            raise RuntimeError("Metal GDS export failed")
    cell = gdstk.read_gds(str(raw), 1e-9).top_level()[0].flatten()
    raw.unlink()
    return cell


def _polys_connected(a, b) -> bool:
    import gdstk

    if {tuple(p) for p in a.points.tolist()} & {tuple(p) for p in b.points.tolist()}:
        return True
    (ax0, ay0), (ax1, ay1) = a.bounding_box()
    (bx0, by0), (bx1, by1) = b.bounding_box()
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return False
    return bool(gdstk.boolean(a, b, "and", layer=a.layer, datatype=a.datatype))


def _or_group(group, layer, datatype):
    import gdstk

    pieces = [group[0]]
    for poly in group[1:]:
        nxt = []
        for piece in pieces:
            res = gdstk.boolean(piece, poly, "or", layer=layer, datatype=datatype)
            nxt.extend(res if res else [piece])
        pieces = nxt
    return pieces


def _merge_connected(polys, helpers_fallback: bool = False):
    import gdstk

    n = len(polys)
    if n <= 1:
        return list(polys)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _polys_connected(polys[i], polys[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i, p in enumerate(polys):
        groups.setdefault(find(i), []).append(p)
    merged = []
    for g in groups.values():
        out = _or_group(g, g[0].layer, g[0].datatype)
        if helpers_fallback and len(out) > 1:
            # the OR bug split this body; the dataset has the body as OR-ed over every piece in reverse order
            every = _or_group(polys[::-1], g[0].layer, g[0].datatype)
            out = [max((q for q in every if gdstk.boolean(q, g, "and")), key=lambda q: q.area())]
        merged.extend(out)
    return merged


def _collapse_north_south(polys):
    north, south = [], []
    for poly in polys:
        (_, y0), (_, y1) = poly.bounding_box()
        (north if 0.5 * (y0 + y1) >= 0 else south).append(poly)
    if not north or not south:
        raise RuntimeError(f"expected north and south bodies, got {len(north)} / {len(south)}")
    n = _or_group(north, north[0].layer, north[0].datatype)
    s = _or_group(south, south[0].layer, south[0].datatype)
    if len(n) != 1 or len(s) != 1:
        raise RuntimeError(f"expected one polygon per side, got {len(n)} / {len(s)}")
    return [n[0], s[0]]


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _helpers_geometry(family: str, design, o: dict, m: float):
    """Chip size, component position and port-stub geometry (um) of the squadds2-helpers convention."""
    pv = lambda k: _um(design, o[k])  # noqa: E731
    ports = []  # (layer, x_offset, width, gap, y_edge, direction)
    if family in ("interdigital", "paddle"):
        fc = int(design.parse_value(o["finger_count"]))
        cap_w = fc * pv("finger_width") + (fc - 1) * pv("finger_gap_east_west")
        cap_h = pv("south_spine_width") + pv("north_spine_width") + pv("finger_length") + pv("finger_gap_north_south")
        west, east = pv("west_gap_ground"), pv("east_gap_ground")
        sx = 2 * m * max(west, east) + cap_w + west + east
        sy = 2 * m * max(pv("north_gap_ground"), pv("south_gap_ground")) + cap_h
        pos = (0.0, pv("north_cpw_length") + cap_h / 2)
        north_y, south_y = cap_h / 2 + pv("north_cpw_length"), -cap_h / 2 - pv("south_cpw_length")
        ports.append((2, pv("north_cpw_xpos_offset"), pv("north_cpw_width"), pv("north_cpw_gap"), north_y, 1))
        ports.append((3, pv("south_cpw_xpos_offset"), pv("south_cpw_width"), pv("south_cpw_gap"), south_y, -1))
        return sx, sy, pos, ports
    if family == "concentric":
        feed = _truthy(o.get("inner_feed", True)) and pv("south_cpw_length") > 0
        rx, ry, g = pv("ground_inner_radius_x"), pv("ground_inner_radius_y"), pv("min_trace_gap")
        ncl, scl = pv("north_cpw_length"), pv("south_cpw_length") if feed else 0.0
        sx, sy = 2 * rx + 2 * m * g, 2 * ry + ncl + scl + 2 * m * g
        if ncl > 0:
            ports.append((2, pv("north_cpw_xpos_offset"), pv("north_cpw_width"), pv("north_cpw_gap"), ry + ncl, 1))
        if feed:
            ports.append((3, pv("south_cpw_xpos_offset"), pv("south_cpw_width"), pv("south_cpw_gap"), -ry - scl, -1))
        return sx, sy, (0.0, 0.0), ports
    if family in ("star", "multipad_star"):
        rp, g, cl = pv("pocket_radius"), pv("gap"), pv("cpw_length")
        sx, sy = 2 * rp + 2 * m * g, 2 * rp + cl + 2 * m * g
        ports.append((2, 0.0, pv("cpw_width"), pv("cpw_gap"), rp + cl, 1))
        return sx, sy, (0.0, 0.0), ports
    raise ValueError(family)


def _fresh_component(comp, design, pos=(0.0, 0.0)):
    opts = {k: v for k, v in dict(comp.options).items() if k not in ("pos_x", "pos_y", "orientation")}
    opts.update(pos_x=f"{pos[0] / 1e3}mm", pos_y=f"{pos[1] / 1e3}mm", orientation="0")
    return type(comp)(design, comp.name, options=opts)


def _export_helpers(comp, s: dict, tmp: Path):
    import gdstk
    from qiskit_metal.designs.design_planar import DesignPlanar

    from squadds.components.export_fixups import gdspy_ground_nm, star_dataset_bodies, star_dataset_ground

    family, style = s["family"], s["export_style"]
    design = DesignPlanar()
    sx, sy, pos, ports = _helpers_geometry(family, design, dict(comp.options), s["margin_gaps"])
    size = design.chips["main"]["size"]
    size["size_x"], size["size_y"] = f"{sx / 1e3}mm", f"{sy / 1e3}mm"
    _fresh_component(comp, design, pos)
    design.rebuild()
    cell = _render(design, tmp)
    grounds = [p for p in cell.polygons if (p.layer, p.datatype) == (1, 0)]
    bodies = [p for p in cell.polygons if (p.layer, p.datatype) != (1, 0)]
    if family in ("interdigital", "paddle"):
        ground = grounds[-1]
        bodies = _collapse_north_south(_merge_connected(bodies))
    elif family == "concentric" or style == "original":
        # the renderer fractures polygons over 199 points: connected pieces are OR-ed back, whole bodies kept as rendered
        ground = gdstk.boolean(grounds[0], grounds[1:], "or", layer=1, datatype=0)[0]
        bodies = _merge_connected(bodies, helpers_fallback=True)
    else:
        ground = gdstk.boolean(grounds[0], grounds[1:], "or", layer=1, datatype=0)[0]
        t = design.qgeometry.tables["poly"]
        metal = list(t[~t["subtract"].astype(bool)].geometry)
        bodies = star_dataset_bodies(metal if style == "reexport_gdspy" else bodies, style, len(metal))
        if style == "reexport_gdspy":
            etch = t[t["subtract"].astype(bool)].geometry.iloc[0]
            ground = gdspy_ground_nm((-sx / 2e3, -sy / 2e3, sx / 2e3, sy / 2e3), etch.exterior.coords)
    lib = gdstk.Library(unit=1e-9, precision=1e-9)
    out = lib.new_cell("main")
    for poly in bodies:
        out.add(poly)
    port_polys = []
    for layer, x0, w, gap, y, direction in ports:
        x0, w, gap, y = x0 * 1e3, w * 1e3, gap * 1e3, y * 1e3
        y1 = y + direction * PORT_LENGTH_UM * 1e3
        port_polys.append(gdstk.rectangle((x0 - w / 2, y), (x0 + w / 2, y1), layer=layer))
        etch = gdstk.rectangle((x0 - w / 2 - gap, y), (x0 + w / 2 + gap, y1), layer=layer, datatype=1)
        ground = gdstk.boolean(ground, etch, "not", layer=1, datatype=0)[0]
    if family in ("star", "multipad_star"):
        ground = star_dataset_ground(ground, bodies, style)
    out.add(ground)
    for p in port_polys:
        out.add(p)
    return out, design


def _family_nets(family: str, comp) -> dict[str, str]:
    """qgeometry element-name prefix -> dataset net label (feed pad N, signal S, the rest P1, P2, ...)."""
    if family == "shielded_concentric":
        return {"inner": "S", "outer_ring": "N"} | {f"guard_{k}": f"P{k}" for k in range(1, 9)}
    pads = list((dict(comp.options).get("connection_pads") or {}).keys())
    if FEED_PIN not in pads:
        raise ValueError(f"the dataset export feeds the {FEED_PIN!r} connection pad; {comp.name} has {pads}")
    others = [p for p in pads if p != FEED_PIN]
    if family == "xmon":
        nets = {"cross": "S", f"{FEED_PIN}_connector_arm": "N"}
        nets |= {f"{p}_connector_arm": f"P{k}" for k, p in enumerate(others, 1)}
        return nets
    nets = {"pad_top": "S", "pad_bot": "P1", f"{FEED_PIN}_connector_pad": "N", f"{FEED_PIN}_wire": "N"}
    for k, p in enumerate(others, 2):
        nets |= {f"{p}_connector_pad": f"P{k}", f"{p}_wire": f"P{k}"}
    return nets


def _gds_polygon(g, layer, datatype):
    import gdstk
    import numpy as np

    nm = lambda ring: [tuple(c) for c in np.asarray(ring.coords)[:-1] * 1e3]  # noqa: E731
    if g.geom_type != "Polygon":
        raise RuntimeError(f"expected one polygon, got {g.geom_type}")
    outer = gdstk.Polygon(nm(g.exterior), layer=layer, datatype=datatype)
    if not g.interiors:
        return outer
    res = gdstk.boolean(outer, [gdstk.Polygon(nm(h)) for h in g.interiors], "not", layer=layer, datatype=datatype)
    if len(res) != 1:
        raise RuntimeError(f"polygon with holes became {len(res)} pieces")
    return res[0]


def _export_families(comp, s: dict, tmp: Path):
    import gdstk
    import numpy as np
    import shapely
    from qiskit_metal import designs
    from shapely.geometry import Polygon

    family = s["family"]
    d = designs.DesignPlanar({}, overwrite_enabled=True)
    instance = _fresh_component(comp, d).name
    net_of_poly = _family_nets(family, comp)
    d.rebuild()
    t = d.qgeometry.tables
    geoms = list(t["poly"].geometry) + [g.buffer(w / 2) for g, w in zip(t["path"].geometry, t["path"].width)]
    x0, y0, x1, y1 = shapely.union_all(geoms).bounds
    m = s["margin_gaps"] * s["ground_gap_um"] / 1e3
    chip = d.chips["main"]["size"]
    chip["size_x"], chip["size_y"] = f"{(x1 - x0) + 2 * m}mm", f"{(y1 - y0) + 2 * m}mm"
    chip["center_x"], chip["center_y"] = f"{(x0 + x1) / 2}mm", f"{(y0 + y1) / 2}mm"
    d.rebuild()
    d.qgeometry.tables["junction"] = d.qgeometry.tables["junction"].iloc[0:0]
    cell = _render(d, tmp)
    ground = [shapely.make_valid(Polygon(p.points / 1e3)) for p in cell.polygons if (p.layer, p.datatype) == (1, 0)]
    ground_sh = shapely.union_all(ground).buffer(1e-4).buffer(-1e-4)
    if ground_sh.geom_type != "Polygon":
        raise RuntimeError(f"ground is not one connected body ({ground_sh.geom_type})")
    t = d.qgeometry.tables
    metal = []
    for kind in ("poly", "path"):
        df = t[kind][~t[kind]["subtract"].astype(bool)]
        for _, row in df.iterrows():
            g = row.geometry if kind == "poly" else row.geometry.buffer(row.width / 2, cap_style=2, join_style=2)
            metal.append((row["name"], shapely.affinity.scale(g, 1e3, 1e3, origin=(0, 0))))
    prefixes = sorted(net_of_poly, key=len, reverse=True)
    nets = {}
    for name, g in metal:
        net = next((net_of_poly[k] for k in prefixes if name.startswith(k)), None)
        if net is None:
            raise RuntimeError(f"metal shape without a net: {name}")
        nets.setdefault(net, []).append(g)
    nets = {k: shapely.make_valid(shapely.union_all(v)) for k, v in nets.items()}
    order = ["N", "S"] + sorted([k for k in nets if k not in ("N", "S")], key=lambda s_: int(s_[1:]))
    pins = d.components[instance].pins
    pin = pins[FEED_PIN]
    mid, nrm = np.array(pin["middle"]) * 1e3, np.array(pin["normal"])
    w, gap = pin["width"] * 1e3, pin["gap"] * 1e3
    tan = np.array([-nrm[1], nrm[0]])

    def rect(half_w):
        a, b = mid + tan * half_w, mid - tan * half_w
        return Polygon([a, b, b + nrm * PORT_LENGTH_UM, a + nrm * PORT_LENGTH_UM])

    port, slot = rect(w / 2), rect(w / 2 + gap)
    ground_sh = ground_sh.difference(slot)
    for name, pn in pins.items():
        if name == FEED_PIN:
            continue
        m2, n2 = np.array(pn["middle"]) * 1e3, np.array(pn["normal"])
        t2, hw, g2 = np.array([-n2[1], n2[0]]), pn["width"] * 1e3 / 2 + pn["gap"] * 1e3, pn["gap"] * 1e3
        a, b = m2 + t2 * hw, m2 - t2 * hw
        ground_sh = ground_sh.difference(Polygon([a, b, b + n2 * g2, a + n2 * g2]))
    lib = gdstk.Library(unit=1e-9, precision=1e-9)
    out = lib.new_cell("main")
    for k in order:
        out.add(_gds_polygon(nets[k], 1, 10))
    out.add(_gds_polygon(ground_sh, 1, 0))
    out.add(_gds_polygon(port, 2, 0))
    return out, d


# ----------------------------------------------------------------------------------------------- public
def write_dataset_gds(cell, path) -> Path:
    """Write a cell as the dataset files are written (1 nm unit and precision, polygons not fractured)."""
    import gdstk

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lib = gdstk.Library(unit=1e-9, precision=1e-9)
    lib.add(cell)
    lib.write_gds(str(path), max_points=1_000_000)
    return path


def dataset_export_layout(obj, family=None, margin_gaps=None, export_style=None, ground_gap_um=None):
    """``(cell, design, settings)``: the dataset-convention layout of a component (or a design holding one) as a gdstk
    cell (nm, snapped to 1 nm as written), the fresh design it was rendered from (its qgeometry names the bodies) and the
    resolved ``dataset_export_settings``."""
    import gdstk

    comp = _component_of(obj)
    s = dataset_export_settings(comp, family, margin_gaps, export_style, ground_gap_um)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cell, design = (_export_helpers if s["exporter"] == "helpers" else _export_families)(comp, s, Path(tmp))
        path = write_dataset_gds(cell, Path(tmp) / "out.gds")
        return gdstk.read_gds(str(path), 1e-9).top_level()[0], design, s


def dataset_export_cell(obj, family=None, margin_gaps=None, export_style=None, ground_gap_um=None):
    """The dataset-convention layout of a component (or a design holding one) as a gdstk cell (nm, 1 nm grid)."""
    return dataset_export_layout(obj, family, margin_gaps, export_style, ground_gap_um)[0]


def export_dataset_gds(component_or_design, path, family=None, margin_gaps=None, export_style=None, ground_gap_um=None):
    """Write ``component_or_design`` as the dataset exported it and return the path.

    family: inferred from the component class (``FAMILY_OF_CLASS``); ``paddle`` / ``multipad_star`` export like
    ``interdigital`` / ``star``. margin_gaps: m of the chip margin (default ``MARGIN_GAPS[family]``). export_style: star
    layouts only (``squadds.components.export_fixups.EXPORT_STYLES``; default ``original`` for one pad,
    ``reexport_gdspy`` for more; SQuADDS rows carry it in ``notes.export_style``). ground_gap_um: the margin's ground
    gap for Xmon / TransmonPocket / ShieldedCapConcentric (default from the options; the dataset used the unrounded
    generator value)."""
    cell = dataset_export_cell(component_or_design, family, margin_gaps, export_style, ground_gap_um)
    return write_dataset_gds(cell, path)
