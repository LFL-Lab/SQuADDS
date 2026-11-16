"""Post-processing that reproduces the tl-squadds star dataset GDS export (exp11 / exp12).

The star layouts of the tl-squadds dataset were written by two versions of the squadds2-helpers export, the second in
two environments. ``style`` selects one:

``"original"`` (exp10, and exp11 as first written 2026-09-03..09)
    Metal GDS renderer output (gdstk; polygons over 199 points fractured on write), pieces OR-ed back per body.

``"reexport"`` (173 exp11 designs rewritten 2026-09-24..28; quantum-metal gdstk renderer, shapely 2.1 / GEOS 3.13) and ``"reexport_gdspy"`` (all of exp12, written 2026-09-29; gdspy-based qiskit-metal 0.1.x renderer, shapely 2.0 / GEOS 3.11)
    Reconstructed from the dataset files (the export script is not in squadds2-helpers); each step is checked vertex
    by vertex against the dataset GDS:

    1. polygons are fractured at 199 points: by the gdstk renderer (``reexport``), or as the gdspy renderer does it
       (``reexport_gdspy``): ``gdspy.Polygon(exterior incl. closing point).fracture(199, 1e-9)`` for the bodies and
       ``gdspy.boolean(chip, cap_etch, "not", precision=1e-9, max_points=199)`` for the ground, snapped to 1 nm and
       the ground pieces OR-ed back. gdspy and gdstk place the cuts (and 1 nm boolean vertices) differently.
    2. pieces are re-assembled with the helpers' ``_merge_cap_polys_by_shared_points``: grouped by shared vertices or
       overlap, then OR-ed one at a time into every partial result. When the first two pieces of a group only touch,
       their OR returns both and the next piece is OR-ed into each, so the group yields two overlapping polygons.
       Of all polygons the export kept the largest ``n_bodies`` (star + enabled pads): this drops the smaller
       duplicate, i.e. the part of a pad beyond a fracture line (a straight chord, up to ~3,300 um^2), and any sliver
       piece that touched nothing.
    3. each body is reoriented clockwise (first vertex kept) and simplified by Douglas-Peucker at 10 nm (shapely,
       ``preserve_topology=False``; if that splits a sliver off, the largest part is kept), which removes ~70 % of the
       pad vertices (the erode/dilate near-duplicates).
    4. every body is isolated from the ground: ground minus ``body.buffer(50 nm, join_style="bevel")``. An unfed pad's
       lead ends on the ground cut-out, so this leaves a 50 nm slit (35 nm across the bevelled corners) at its end;
       the fed pad (port etch) and the star are not affected.

    The vertices also depend on the GEOS version that built the component (1 nm vertex choices, hence fracture cuts):
    exp12 reproduces exactly only under shapely 2.0.x (GEOS 3.11), exp11 only under shapely 2.1 (GEOS 3.13).

The component geometry itself (``GeneralizedCapStar``) is unchanged. Polygons are gdstk polygons in nm.
"""

from __future__ import annotations

EXPORT_STYLES = ("original", "reexport", "reexport_gdspy")
REEXPORT_SIMPLIFY_NM = 10.0
REEXPORT_PAD_GROUND_GAP_NM = 50.0


def _connected(a, b) -> bool:
    import gdstk

    if {tuple(p) for p in a.points.tolist()} & {tuple(p) for p in b.points.tolist()}:
        return True
    (ax0, ay0), (ax1, ay1) = a.bounding_box()
    (bx0, by0), (bx1, by1) = b.bounding_box()
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return False
    return bool(gdstk.boolean(a, b, "and", layer=a.layer, datatype=a.datatype))


def helpers_or_group(group: list) -> list:
    """squadds2-helpers ``_boolean_or_group``: OR pieces one at a time into every partial result (kept verbatim)."""
    import gdstk

    layer, datatype = group[0].layer, group[0].datatype
    pieces = [group[0]]
    for poly in group[1:]:
        nxt = []
        for piece in pieces:
            res = gdstk.boolean(piece, poly, "or", layer=layer, datatype=datatype)
            nxt.extend(res if res else [piece])
        pieces = nxt
    return pieces


def helpers_body_groups(polys: list) -> list[list]:
    """Group fractured pieces by connectivity (union-find, input order) and OR each group the helpers' way."""
    n = len(polys)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _connected(polys[i], polys[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, list] = {}
    for i, p in enumerate(polys):
        groups.setdefault(find(i), []).append(p)
    return [helpers_or_group(g) for g in groups.values()]


def simplify_clockwise(poly, tol_nm: float = REEXPORT_SIMPLIFY_NM):
    """Clockwise ring (first vertex kept) simplified by Douglas-Peucker (shapely, ``preserve_topology=False``); the
    largest part if the simplification splits the ring."""
    import gdstk
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    ring = orient(Polygon(poly.points), sign=-1.0).simplify(tol_nm, preserve_topology=False)
    if ring.geom_type != "Polygon":
        ring = max(ring.geoms, key=lambda g: g.area)
    return gdstk.Polygon(np.asarray(ring.exterior.coords[:-1]), layer=poly.layer, datatype=poly.datatype)


def isolate_from_ground(ground, bodies: list, gap_nm: float = REEXPORT_PAD_GROUND_GAP_NM):
    """Ground minus every body grown by ``gap_nm`` with bevelled corners (one polygon expected)."""
    import gdstk
    import numpy as np
    from shapely.geometry import Polygon

    cut = [
        gdstk.Polygon(np.asarray(Polygon(p.points).buffer(gap_nm, join_style="bevel").exterior.coords[:-1]))
        for p in bodies
    ]
    res = gdstk.boolean(ground, cut, "not", layer=ground.layer, datatype=ground.datatype)
    if len(res) != 1:
        raise RuntimeError(f"ground split into {len(res)} pieces by the pad isolation")
    return res[0]


def gdspy_fracture_nm(coords_mm, max_points: int = 199, precision: float = 1e-9) -> list:
    """Pieces (nm, 1 nm grid) of a polygon fractured like the gdspy Metal renderer writes it."""
    import gdspy
    import gdstk
    import numpy as np

    pieces = gdspy.Polygon(list(map(tuple, coords_mm))).fracture(max_points=max_points, precision=precision)
    return [gdstk.Polygon(np.round(q * 1e6), layer=1, datatype=10) for q in pieces.polygons]


def gdspy_ground_nm(chip_box_mm, etch_coords_mm, max_points: int = 199, precision: float = 1e-9):
    """Ground (nm) of the gdspy renderer: chip box minus the cap etch, fractured, snapped to 1 nm and OR-ed back."""
    import gdspy
    import gdstk
    import numpy as np

    x0, y0, x1, y1 = chip_box_mm
    res = gdspy.boolean(
        gdspy.Rectangle((x0, y0), (x1, y1)),
        gdspy.Polygon(list(map(tuple, etch_coords_mm))),
        "not",
        precision=precision,
        max_points=max_points,
    )
    pieces = [gdstk.Polygon(np.round(q * 1e6)) for q in res.polygons]
    return gdstk.boolean(pieces[0], pieces[1:], "or", layer=1, datatype=0)[0]


def star_dataset_bodies(geoms, style: str = "reexport", n_bodies: int | None = None) -> list:
    """Signal bodies of a ``GeneralizedCapStar`` as the dataset export wrote them, one polygon per conductor.

    ``geoms``: the rendered (gdstk-fractured) pieces in nm for ``original`` / ``reexport``; the component's metal
    qgeometry polygons (shapely, mm, table order) for ``reexport_gdspy``. ``n_bodies``: star + enabled pads (re-export
    styles; defaults to ``len(geoms)`` for ``reexport_gdspy``, else to the number of connected groups).
    """
    import gdstk

    if style not in EXPORT_STYLES:
        raise ValueError(f"style must be one of {EXPORT_STYLES}, got {style!r}")
    if style == "original":
        return gdstk.boolean(geoms[0], geoms[1:], "or", layer=geoms[0].layer, datatype=geoms[0].datatype)
    if style == "reexport_gdspy":
        n_bodies = n_bodies or len(geoms)
        geoms = [pc for g in geoms for pc in gdspy_fracture_nm(g.exterior.coords)]
    groups = helpers_body_groups(geoms)
    merged = [p for g in groups for p in g]
    keep = sorted(range(len(merged)), key=lambda i: -merged[i].area())[: n_bodies or len(groups)]
    return [simplify_clockwise(merged[i]) for i in sorted(keep)]


def star_dataset_ground(ground, bodies: list, style: str = "reexport"):
    """Ground as the dataset export wrote it. ``bodies``: the exported signal bodies (the star, >= ``gap`` from the
    ground, is unaffected by the isolation)."""
    if style not in EXPORT_STYLES:
        raise ValueError(f"style must be one of {EXPORT_STYLES}, got {style!r}")
    return ground if style == "original" else isolate_from_ground(ground, bodies)
