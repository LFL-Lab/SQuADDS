"""GDS reading and layout preparation: layers -> ground net + conductor bodies (full ground by default; optional crop window)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import gdstk
import numpy as np
import shapely
from shapely.geometry import Polygon

from ._tlsq.uge.scene import _valid

GROUND = (1, 0)
CONDUCTOR = (1, 10)
N_VALIDATED_SIGNALS = 7  # largest conductor count (plus ground) seen in training and validation
N_MAX_SIGNALS = 30  # practical cap for the free Space (MoM cost); the networks themselves have no limit
AUTO_GAPS, AUTO_FRAC, AUTO_MIN_UM = 100.0, 0.5, 150.0  # margin="auto" (see deploy/LIMITS.md: crop study)  # the networks are dense over at most 8 conductors including ground

# Figure colours = the paper's (MetBrewer Archambault, Hokusai2, OKeeffe1; paper_v4/ml_paper_2026/figures/scripts/_style.py)
GROUND_COLOR = "#D9D9D9"
CONDUCTOR_COLORS = ("#AB3329", "#F9D14A", "#381A61", "#88A0DC", "#E78429", "#7C4B73", "#ED968C")
LIGHT_GOLD = "#F9D14A"  # large conductors only: thin ones skip it
WINDOW_COLOR = "#381A61"
UNMODELLED_COLOR = "#6B200C"
INK = "#111827"


def conductor_colors(areas) -> list[str]:
    """The paper's per-conductor cycle; a conductor smaller than the median area skips light gold."""
    med = float(np.median(areas)) if len(areas) else 0.0
    out, i = [], 0
    for a in areas:
        if CONDUCTOR_COLORS[i % len(CONDUCTOR_COLORS)] == LIGHT_GOLD and a < med:
            i += 1
        out.append(CONDUCTOR_COLORS[i % len(CONDUCTOR_COLORS)])
        i += 1
    return out


def _luminance(hex_color: str) -> float:
    lin = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4) for c in (int(hex_color[j:j + 2], 16) / 255 for j in (1, 3, 5))]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def text_color(hex_color: str) -> str:
    """White or ink, whichever has the higher WCAG contrast on the fill."""
    lum = _luminance(hex_color)
    return "white" if 1.05 / (lum + 0.05) >= (lum + 0.05) / (_luminance(INK) + 0.05) else INK


class LayoutError(ValueError):
    """The layout cannot be turned into a valid capacitance problem (message says why and what to change)."""


def _top_cell(lib, cell: str | None):
    if cell is not None:
        for c in lib.cells:
            if c.name == cell:
                return c
        raise LayoutError(f"cell {cell!r} not in the file (cells: {[c.name for c in lib.cells][:20]})")
    tops = lib.top_level()
    if not tops:
        raise LayoutError("the GDS file has no cells")
    if len(tops) == 1:
        return tops[0]
    # several top cells: take the one with the most (flattened) polygons
    return max(tops, key=lambda c: len(c.get_polygons()))


def read_polygons(path, cell: str | None = None) -> tuple[dict, float, str]:
    """{(layer, datatype): [shapely Polygon in µm, ...]} in file order, with references and paths flattened."""
    try:
        lib = gdstk.read_gds(str(path))
    except Exception as e:  # noqa: BLE001
        raise LayoutError(f"not a readable GDS file: {e}") from e
    top = _top_cell(lib, cell)
    u = lib.unit / 1e-6
    # flat cells (the training files): direct polygons in file order, which fixes the conductor order of the paper;
    # otherwise everything is flattened (references, repetitions, paths)
    polys = list(top.polygons) if not (top.references or top.paths) else top.get_polygons(apply_repetitions=True, include_paths=True)
    out: dict = {}
    for p in polys:
        if len(p.points) < 3:
            continue
        out.setdefault((p.layer, p.datatype), []).append(Polygon(p.points * u))
    return out, u, top.name


def load_gds(path, layer_map: dict | None = None, cell: str | None = None) -> list[dict]:
    """Detected (layer, datatype) pairs with polygon count, area (µm²) and bounding box (µm), plus a suggested role
    (ground / conductor / ground+conductor / ignore) from `layer_map` ({"ground": [(1, 0)], "conductor": [(1, 10)]} by default;
    a single-layer file without them gets "ground+conductor")."""
    layer_map = layer_map or {"ground": [GROUND], "conductor": [CONDUCTOR]}
    lm = {tuple(k): role for role, ks in layer_map.items() for k in ks}
    polys, _, top = read_polygons(path, cell)
    if len(polys) == 1 and not set(lm) <= set(polys):
        lm = {k: "ground+conductor" for k in polys}  # one metal layer: ground detected among its pieces (prepare)
    rows = []
    for (l, d), ps in sorted(polys.items()):
        b = shapely.union_all(ps).bounds if ps else (np.nan,) * 4
        rows.append({"layer": l, "datatype": d, "n_polygons": len(ps), "area_um2": float(sum(p.area for p in ps)),
                     "bbox_um": tuple(float(x) for x in b), "role": lm.get((l, d), "ignore"), "cell": top})
    return rows


def _components(polys: list) -> list[list[int]]:
    """Connected components (touching or overlapping polygons) as lists of polygon indices, ordered by first index."""
    n = len(polys)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n > 1:
        tree = shapely.STRtree(polys)
        ii, jj = tree.query(polys, predicate="intersects")
        for a, b in zip(ii, jj):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _core_bounds(G, bodies: list, warnings: list) -> tuple:
    """Bounds of the conductors together with the etched pocket(s) that contain them, so that cropping never cuts into the
    ground edge facing the conductors. Open regions (the etch reaches the outer boundary of the metal) and very large
    pockets (e.g. a CPW gap running across the chip) fall back to the conductors' own bounds."""
    S = shapely.union_all(bodies)
    sb = S.bounds
    outer = shapely.box(*shapely.union_all([G, S]).bounds)
    etched = outer.difference(G)
    parts = [p for p in getattr(etched, "geoms", [etched]) if p.intersects(S)]
    if not parts:
        return sb
    P = shapely.union_all(parts)
    pb = P.bounds
    ob = outer.bounds
    open_ = any(abs(a - b) < 1e-6 for a, b in zip(pb, ob))
    big = (pb[2] - pb[0]) * (pb[3] - pb[1]) > 25 * max((sb[2] - sb[0]) * (sb[3] - sb[1]), 1e-9)
    if open_ or big:
        if big and not open_:
            warnings.append("the etched region around the conductors is large (> 25x their bounding box); window built "
                            "around the conductors only: check the crop in the preview")
        return sb
    return (min(sb[0], pb[0]), min(sb[1], pb[1]), max(sb[2], pb[2]), max(sb[3], pb[3]))


RECT_FILL, AREA_RATIO_LOW = 0.95, 2.0  # ground detection: near-rectangular outline; top-2 piece areas within this ratio = low confidence


def _shell(g):
    """The piece with its openings filled (outer boundary only)."""
    return shapely.union_all([Polygon(p.exterior) for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"])


def detect_ground(geoms: list) -> tuple[list[int], str, str]:
    """(ground piece indices, reason, confidence) for metal pieces that share one layer. Ground = the piece whose openings
    contain every other piece; else the largest near-rectangular piece touching the layout bounding box; else the largest."""
    n = len(geoms)
    areas = [float(g.area) for g in geoms]
    shells = [_shell(g) for g in geoms]
    enc = [i for i in range(n) if all(shells[i].contains(geoms[j]) for j in range(n) if j != i)]
    if len(enc) == 1:
        k, why = enc[0], "it encloses every other piece (they sit in its openings)"
    else:
        B = shapely.union_all(geoms).bounds
        tol = 1e-6 * max(B[2] - B[0], B[3] - B[1])

        def rect_edge(i):
            with np.errstate(divide="ignore", invalid="ignore"):  # GEOS warns on axis-aligned rectangles; the envelope is still right
                fill = shells[i].area / max(shells[i].minimum_rotated_rectangle.area, 1e-30)
            return fill >= RECT_FILL and any(abs(a - b) <= tol for a, b in zip(shells[i].bounds, B))

        cand = [i for i in range(n) if rect_edge(i)]
        if cand:
            k = max(cand, key=lambda i: areas[i])
            why = "no piece encloses all the others; it is the largest near-rectangular piece on the layout boundary"
        else:
            k = max(range(n), key=lambda i: areas[i])
            why = "no piece encloses all the others and none is a near-rectangular piece on the layout boundary; it is the largest"
    top = sorted(areas, reverse=True)
    close = n > 1 and top[0] < AREA_RATIO_LOW * top[1]
    conf = "high" if len(enc) == 1 and not close else "low"
    if close:
        why += f"; the two largest pieces are within {AREA_RATIO_LOW:g}x in area"
    return [k], why, conf


@dataclass
class PreparedLayout:
    ground: object  # shapely geometry (µm), cropped
    conductors: list  # shapely geometries (µm), one per net, file order
    names: list[str]
    window: tuple | None  # (x0, y0, x1, y1) µm, or None when not cropped
    min_gap_um: float
    source: str = ""
    warnings: list[str] = field(default_factory=list)
    extra_metal: dict = field(default_factory=dict)  # label -> geometry of metal inside the window that is not modelled
    info: dict = field(default_factory=dict)
    pieces: list = field(default_factory=list)  # metal pieces (shapely, µm) when ground and conductors share a layer; piece #k = pieces[k - 1] (conductor Ck)

    @property
    def n_conductors(self) -> int:
        return len(self.conductors)

    def extent_um(self) -> float:
        """Lateral size L of the conductors: diagonal of their joint bounding box (µm)."""
        x0, y0, x1, y1 = shapely.union_all(self.conductors).bounds
        return float(np.hypot(x1 - x0, y1 - y0))

    def to_gds(self, path=None) -> bytes:
        """Simplified GDS in the training convention: ground on (1, 0), one or more polygons per conductor on (1, 10).
        Returns the bytes (and writes them when `path` is given)."""
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        cell = lib.new_cell("PREPARED")

        def add(g, layer, dt):
            for poly in getattr(g, "geoms", [g]):
                if poly.is_empty:
                    continue
                if len(poly.interiors) == 0:
                    cell.add(gdstk.Polygon(np.asarray(poly.exterior.coords)[:-1], layer=layer, datatype=dt))
                else:  # polygons with holes: boolean in gdstk keeps them as keyhole polygons
                    outer = gdstk.Polygon(np.asarray(poly.exterior.coords)[:-1])
                    holes = [gdstk.Polygon(np.asarray(r.coords)[:-1]) for r in poly.interiors]
                    for p in gdstk.boolean(outer, holes, "not", layer=layer, datatype=dt):
                        cell.add(p)

        add(self.ground, *GROUND)
        for g in self.conductors:
            add(g, *CONDUCTOR)
        if path is None:
            with tempfile.TemporaryDirectory() as d:
                lib.write_gds(Path(d) / "prepared.gds")
                return (Path(d) / "prepared.gds").read_bytes()
        lib.write_gds(str(path))
        return Path(path).read_bytes()

    def plot(self, ax=None, **kwargs):
        """Draw the layout: ground grey, conductors coloured and named as the predictions name them. ``prepare_layout(gds).plot()``;
        keyword arguments as ``render`` (labels, show_window, title, figsize, compact). Returns the matplotlib Axes."""
        fig = self.render(ax=ax, **kwargs)
        return ax if ax is not None else fig.axes[0]

    def render(self, ax=None, show_window: bool = True, title: str | None = None, figsize=(6.4, 6.4), compact: bool = False,
               labels=None):
        """Matplotlib figure: ground (grey), conductors (coloured, labelled), unmodelled metal (hatched), window. When ground and
        conductors share a layer, conductor piece #k is labelled Ck and ground pieces "G #k" (the numbers ground_pieces takes);
        compact = smaller labels; labels = {name: text} renames conductors on the figure (e.g. {"C1": "star"}), False hides them."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch, Rectangle
        from matplotlib.path import Path as MPath

        def patch(g, **kw):
            for poly in getattr(g, "geoms", [g]):
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                verts, codes = [], []
                for ring in [poly.exterior, *poly.interiors]:
                    xy = np.asarray(ring.coords)
                    verts += list(xy)
                    codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), **kw))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        patch(self.ground, facecolor=GROUND_COLOR, edgecolor="#8c8c8c", lw=0.4)
        gd = self.info.get("ground_detection")
        fs = 8 if compact else 11
        for g, nm, col in zip(self.conductors, self.names, conductor_colors([g.area for g in self.conductors])):
            patch(g, facecolor=col, edgecolor="black", lw=0.4)
            p = g.representative_point() if g.area > 0 else g.centroid
            if labels is not False:
                ax.annotate((labels or {}).get(nm, nm), (p.x, p.y), ha="center", va="center", fontsize=fs, weight="bold",
                            color=text_color(col), bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="black", lw=0.6))
        for r in (gd or {}).get("pieces", []):
            if r["role"] != "conductor":
                x, y = r["label_xy"]
                ax.annotate(f"{'G' if r['role'] == 'ground' else 'not modelled'} #{r['id']}", (x, y), ha="center", va="center",
                            fontsize=fs, weight="bold", color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="#e5e7eb", ec="#374151", lw=0.6))
        for lab, g in self.extra_metal.items():
            patch(g, facecolor="none", edgecolor=UNMODELLED_COLOR, hatch="///", lw=0.6)
        if show_window and self.window is not None:
            x0, y0, x1, y1 = self.window
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", ec=WINDOW_COLOR, lw=1.0))
        b = shapely.union_all([self.ground, *self.conductors]).bounds
        pad = 0.03 * max(b[2] - b[0], b[3] - b[1])
        ax.set_xlim(b[0] - pad, b[2] + pad)
        ax.set_ylim(b[1] - pad, b[3] + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        if title is None and gd:
            title = (f"ground = piece {', '.join(f'#{i}' for i in gd['ground_pieces'])} ({gd['method']}, confidence {gd['confidence']}); "
                     f"{len(self.conductors)} conductors")
        ax.set_title(title or f"{len(self.conductors)} conductors + ground; min gap {self.min_gap_um:.2f} µm", fontsize=10)
        return fig


def prepare(path, ground_layers=(GROUND,), conductor_layers=(CONDUCTOR,), margin_gaps: float | str | None = None,
            margin_um: float | None = None, ignore_layers=(), cell: str | None = None, select=None, select_box=None,
            max_conductors: int = N_MAX_SIGNALS, ground_pieces=None) -> PreparedLayout:
    """Extract the capacitance problem from a GDS file.

    - conductors: connected bodies of the conductor-layer polygons (touching/overlapping polygons are one net), in file
      order; bodies that touch the ground metal are shorted to it and merged into the ground net (warning).
    - ground: union of the ground-layer polygons. Default (margin_gaps=None, margin_um=None): no crop, the training
      geometry (tlsq.uge.scene.scene_from_gds; TRAINING_PLAN Addendum R5; default since 2026-10-14). Optional crop to the
      window = bbox(conductors + the etched pocket that contains them) expanded by `margin_um` if given, else `margin_gaps`
      x (smallest conductor-to-ground gap), or ("auto") by max(100 gaps, half the conductor extent, 150 µm) (deploy/LIMITS.md).
    - select: optional list of conductor-body indices (0-based, file order), or select_box = (x0, y0, x1, y1) µm (bodies
      fully inside); the other bodies are not modelled and are flagged as extra metal when they fall inside the window.
    - metal on layers that are neither ground, conductor nor in `ignore_layers` is flagged when inside the window.
    - ground and conductors on one layer (ground_layers == conductor_layers, or a file with a single metal layer while the default
      layers 1/0 and 1/10 are not both present): the layer's metal is split into connected pieces and the ground is detected
      (detect_ground; result and confidence in info["ground_detection"], plus a warning). Pieces are numbered with the
      automatically detected ground last and the others in file order, and conductor piece #k is named Ck, so the automatic
      split names its conductors as the two-layer file does and names stay put when roles change. ground_pieces = piece
      numbers (as in render()) overrides the pick; several pieces make one ground net.
    """
    gl = {tuple(x) for x in ground_layers}
    cl = {tuple(x) for x in conductor_layers}
    il = {tuple(x) for x in ignore_layers}
    if gl & cl and gl != cl:
        raise LayoutError(f"layers {sorted(gl & cl)} are assigned both ground and conductor (assign the same layers to both to detect "
                          "the ground automatically)")
    polys, _, top = read_polygons(path, cell)
    warnings: list[str] = []
    metal = [k for k in polys if k not in il]
    shared = None
    if gl == cl:
        shared = gl
    elif gl == {GROUND} and cl == {CONDUCTOR} and len(metal) == 1 and not (GROUND in polys and CONDUCTOR in polys):
        shared = set(metal)
    if shared is None and ground_pieces is not None:
        raise LayoutError("ground_pieces applies only when ground and conductors share one layer (the same ground and conductor "
                          "layers, or a single-layer file)")
    pieces, gdet = [], None
    if shared is not None:
        gl = cl = shared
        allp = [p for k in polys if k in shared for p in polys[k]]
        if not allp:
            raise LayoutError(f"no polygons on {sorted(shared)}; detected layers: {sorted(polys)}")
        val = [_valid(p) for p in allp]
        ok = [i for i, g in enumerate(val) if not g.is_empty and g.area > 0]
        comps = [[ok[j] for j in c] for c in _components([val[i] for i in ok])]
        pieces = [val[c[0]] if len(c) == 1 else _valid(shapely.union_all([val[i] for i in c])) for c in comps]
        if len(pieces) < 2:
            raise LayoutError(f"the metal on {sorted(shared)} is one connected piece: no floating conductor to compute")
        auto, why, conf = detect_ground(pieces)
        order = [i for i in range(len(pieces)) if i not in auto] + auto
        pieces, comps = [pieces[i] for i in order], [comps[i] for i in order]
        if ground_pieces is None:
            gids, method = list(range(len(pieces) - len(auto), len(pieces))), "automatic"
        else:
            try:
                gp = [x for x in ground_pieces.split(",") if x.strip()] if isinstance(ground_pieces, str) else list(ground_pieces)
                gids = sorted({int(i) - 1 for i in gp})
            except (TypeError, ValueError) as e:
                raise LayoutError(f"ground_pieces must be piece numbers such as [1] or '1,4' (got {ground_pieces!r})") from e
            bad = [i + 1 for i in gids if not 0 <= i < len(pieces)]
            if bad or not gids:
                raise LayoutError(f"ground_pieces {bad or list(ground_pieces)} not valid: this layout has pieces 1..{len(pieces)} "
                                  "(numbers as in the labelled figure)")
            if len(gids) == len(pieces):
                raise LayoutError("ground_pieces lists every piece: at least one piece must be a conductor")
            why, conf, method = f"set by ground_pieces={[i + 1 for i in gids]}", "user", "user"
        gset = {i for g in gids for i in comps[g]}
        gpolys = [allp[i] for i in sorted(gset)]
        cpolys = [allp[i] for i in range(len(allp)) if i not in gset]
        lay_s = ", ".join(f"{k[0]}/{k[1]}" for k in sorted(shared))
        ids = ", ".join(f"#{g + 1}" for g in gids)
        if method == "automatic":
            warnings.append(f"ground and conductors share layer {lay_s}: the ground was detected automatically as piece {ids} ({why}; "
                            f"confidence {conf}). Check the numbered figure of the pieces and, if it is wrong, resubmit with "
                            "ground_pieces=[piece numbers].")
        else:
            warnings.append(f"ground and conductors share layer {lay_s}: ground = piece {ids} (set by ground_pieces)")
        gdet = {"layers": [list(k) for k in sorted(shared)], "method": method, "ground_pieces": [g + 1 for g in gids], "reason": why,
                "confidence": conf, "n_pieces": len(pieces)}
    else:
        gpolys = [p for k in polys if k in gl for p in polys[k]]
        cpolys = [p for k in polys if k in cl for p in polys[k]]
    if not gpolys:
        raise LayoutError(f"no ground polygons on {sorted(gl)}; detected layers: {sorted(polys)}")
    if not cpolys:
        raise LayoutError(f"no conductor polygons on {sorted(cl)}; detected layers: {sorted(polys)}")
    cvalid = [_valid(p) for p in cpolys]
    keep = [i for i, g in enumerate(cvalid) if not g.is_empty and g.area > 0]
    if len(keep) < len(cvalid):
        warnings.append(f"{len(cvalid) - len(keep)} degenerate conductor polygon(s) (zero area) dropped")
    cvalid = [cvalid[i] for i in keep]
    G = _valid(shapely.union_all([_valid(g) for g in gpolys]))
    bodies = []
    for comp in _components(cvalid):
        bodies.append(cvalid[comp[0]] if len(comp) == 1 else _valid(shapely.union_all([cvalid[i] for i in comp])))
    n_merged = len(cvalid) - len(bodies)
    if n_merged:
        warnings.append(f"{len(cvalid)} conductor polygons form {len(bodies)} connected bodies (touching polygons merged into one net)")
    # bodies in contact with ground metal are at ground potential
    shorted = [i for i, b in enumerate(bodies) if shapely.intersects(b, G)]
    if shorted:
        warnings.append(f"{len(shorted)} conductor body(ies) touch the ground metal and were merged into the ground net")
        G = _valid(shapely.union_all([G] + [bodies[i] for i in shorted]))
        bodies = [b for i, b in enumerate(bodies) if i not in set(shorted)]
    if not bodies:
        raise LayoutError("every conductor body touches the ground: nothing floating to compute")
    all_bodies = bodies
    others = []
    if select is not None or select_box is not None:
        if select_box is not None:
            box = shapely.box(*select_box)
            idx = [i for i, b in enumerate(bodies) if box.contains(b)]
        else:
            idx = [int(i) for i in select]
            bad = [i for i in idx if i < 0 or i >= len(bodies)]
            if bad:
                raise LayoutError(f"select indices {bad} out of range (0..{len(bodies) - 1})")
        if not idx:
            raise LayoutError("the selection matched no conductor body")
        bodies = [all_bodies[i] for i in idx]
        others = [all_bodies[i] for i in range(len(all_bodies)) if i not in set(idx)]
    if len(bodies) > max_conductors:
        raise LayoutError(f"{len(bodies)} conductors found; this server is configured for at most {max_conductors} signal conductors "
                          "plus ground (a cost cap of the demo, not a limit of the models: pass prepare(max_conductors=...) "
                          "locally). Use `select` (indices or a box) to pick the conductors of interest.")
    if len(bodies) > N_VALIDATED_SIGNALS:
        warnings.append(f"{len(bodies)} conductors: beyond the validated range ({N_VALIDATED_SIGNALS} conductors plus ground). "
                        "The model applies the same pair function to every pair, but accuracy at this size has not been verified.")
    metals = [G] + bodies
    gap = min(shapely.distance(metals[i], metals[j]) for i in range(len(metals)) for j in range(i + 1, len(metals)))
    if not np.isfinite(gap) or gap <= 0:
        raise LayoutError("zero gap between two nets (overlapping metal on different nets)")
    window = None
    if margin_um is not None or margin_gaps is not None:
        gap_cg = min(shapely.distance(G, b) for b in bodies)  # conductor-to-ground gap sets the scale of the ground's field
        x0, y0, x1, y1 = _core_bounds(G, bodies, warnings)
        if margin_um is not None:
            m = float(margin_um)
        elif margin_gaps == "auto":
            # a training-like neighbourhood: the training chips extend ~150-330 µm (30-170 gaps) beyond the conductors
            m = max(AUTO_GAPS * gap_cg, AUTO_FRAC * max(x1 - x0, y1 - y0), AUTO_MIN_UM)
        else:
            m = float(margin_gaps) * gap_cg
        window = (x0 - m, y0 - m, x1 + m, y1 + m)
        box = shapely.box(*window)
        Gc = _valid(shapely.intersection(G, box))
        if Gc.is_empty:
            raise LayoutError("no ground metal inside the window; increase the margin")
        G = Gc
        metals = [G] + bodies
        gap = min(shapely.distance(metals[i], metals[j]) for i in range(len(metals)) for j in range(i + 1, len(metals)))
        if G.geom_type == "MultiPolygon" and len(G.geoms) > 1:
            warnings.append(f"cropped ground consists of {len(G.geoms)} separate pieces (all treated as one net)")
    box = shapely.box(*window) if window is not None else shapely.box(*shapely.union_all(metals).bounds)
    extra = {}
    for k, ps in polys.items():
        if k in gl or k in cl or k in il:
            continue
        inside = [p for p in ps if shapely.intersects(p, box)]
        if inside:
            extra[f"layer {k[0]}/{k[1]}"] = shapely.union_all([_valid(p) for p in inside])
    for i, b in enumerate(others):
        if shapely.intersects(b, box):
            extra.setdefault("unselected conductors", [])
            extra["unselected conductors"].append(b)
    if "unselected conductors" in extra:
        n_un = len(extra["unselected conductors"])
        extra["unselected conductors"] = shapely.union_all(extra["unselected conductors"])
        warnings.append(f"{n_un} unselected conductor body(ies) lie inside the window and are NOT modelled (they would screen the selected ones)")
    for lab in extra:
        if lab.startswith("layer"):
            warnings.append(f"metal on {lab} inside the window is not assigned (ground / conductor) and is ignored")
    holes = sum(len(p.interiors) for b in bodies for p in getattr(b, "geoms", [b]))
    if holes:
        warnings.append(f"conductors contain {holes} hole(s): supported, but outside the training families")
    size = max(np.subtract(shapely.union_all(metals).bounds[2:], shapely.union_all(metals).bounds[:2]))
    if gap < 3 * size / 4096:
        warnings.append(f"smallest gap {gap:.3g} µm is below the MoM resolution floor for a {size:.0f} µm layout "
                        f"(~{3 * size / 4096:.3g} µm): crop tighter or expect larger errors")
    names = [f"C{i + 1}" for i in range(len(bodies))]
    if gdet is not None:
        gset = set(gdet["ground_pieces"])
        cond = sorted((k for k in range(1, len(pieces) + 1) if k not in gset), key=lambda k: comps[k - 1][0])  # file order = body order
        piece_of = dict(zip(map(id, all_bodies), cond)) if len(cond) == len(all_bodies) else {}
        names = [f"C{piece_of[id(b)]}" if piece_of else nm for b, nm in zip(bodies, names)]
    info = {"cell": top, "margin_um": m if window else None, "n_conductor_polygons": len(cvalid), "n_bodies_total": len(all_bodies), "layout_size_um": float(size),
            "extent_um": float(np.hypot(*np.subtract(shapely.union_all(bodies).bounds[2:], shapely.union_all(bodies).bounds[:2])))}
    if gdet is not None:
        info["ground_detection"] = _detection_info(gdet, pieces, comps, {piece_of[id(b)]: nm for b, nm in zip(bodies, names)} if piece_of else {})
    return PreparedLayout(G, bodies, names, window, float(gap), str(path), warnings, extra, info, pieces)


def _detection_info(gdet: dict, pieces: list, comps: list, name_of: dict) -> dict:
    """JSON-ready record of the pieces and their roles (name_of: piece number -> conductor name of the modelled conductors)."""
    gset = set(gdet["ground_pieces"])
    rows = []
    for i, g in enumerate(pieces, 1):
        p = g.representative_point()
        role = "ground" if i in gset else "conductor" if i in name_of else "not modelled"
        rows.append({"id": i, "role": role, "name": "G" if i in gset else name_of.get(i), "area_um2": round(float(g.area), 3),
                     "bbox_um": [round(float(x), 3) for x in g.bounds], "n_polygons": len(comps[i - 1]), "label_xy": [float(p.x), float(p.y)]})
    return {**gdet, "pieces": rows}
