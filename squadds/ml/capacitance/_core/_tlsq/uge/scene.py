"""Scene: the metal of a layout as a list of conductors (one ground net + any number of signal bodies), plus the stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import gdstk
import numpy as np
import shapely
from shapely.geometry import Polygon

DEFAULT_LAYERS = {"ground": [(1, 0)], "signal": [(1, 10)]}
DEFAULT_STACK = {"n_metal_layers": 1, "metal_thickness_um": 0.25, "substrate_thickness_um": 350.0, "substrate_eps_r": 11.45}


@dataclass
class Conductor:
    name: str
    role: str  # "ground" | "signal"
    geom: object  # shapely (Multi)Polygon, µm


@dataclass
class Scene:
    conductors: list[Conductor]  # index 0 is the ground net
    chip: Polygon
    pocket: object  # chip minus ground metal: the etched region (its edge is the ground's active edge)
    stack: dict = field(default_factory=lambda: dict(DEFAULT_STACK))
    markers: dict = field(default_factory=dict)  # non-metal helper shapes by (layer, datatype), e.g. port stubs; naming only

    @property
    def n(self) -> int:
        return len(self.conductors)

    def names(self) -> list[str]:
        return [c.name for c in self.conductors]


def _valid(g):
    g = shapely.make_valid(g)
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts)


def scene_from_gds(path, layers: dict = DEFAULT_LAYERS, stack: dict | None = None,
                   namer: Callable[[list, object], list[str]] | None = None) -> Scene:
    """Read only metal layers. Each signal polygon is one conductor; all ground polygons form one net.
    `namer(signal_geoms, markers)` returns a name per signal body (dataset convention); default S0, S1, ..."""
    lib = gdstk.read_gds(str(path))
    cell = lib.top_level()[0]
    u = lib.unit / 1e-6
    lay = {k: {tuple(x) for x in v} for k, v in layers.items()}
    sig, gnd, other = [], [], {}
    for p in cell.polygons:
        key = (p.layer, p.datatype)
        g = Polygon(p.points * u)
        if key in lay["signal"]:
            sig.append(g)
        elif key in lay["ground"]:
            gnd.append(g)
        else:
            other.setdefault(key, []).append(g)
    G = _valid(shapely.union_all([_valid(g) for g in gnd]))
    S = [_valid(g) for g in sig]
    chip = shapely.box(*G.bounds)
    markers = {k: shapely.union_all(v) for k, v in other.items()}
    names = namer(S, markers) if namer else [f"S{i}" for i in range(len(S))]
    conds = [Conductor("G", "ground", G)] + [Conductor(n, "signal", g) for n, g in zip(names, S)]
    return Scene(conds, chip, chip.difference(G), {**DEFAULT_STACK, **(stack or {})}, markers)


def tlsq_namer(signals: list, markers: dict, n_slots: int = 5) -> list[str]:
    """This dataset's convention: N = body fed by the north port stub (2,0); S = body fed by the south stub (3,0) if
    present, else the body containing the origin (star, inner pad); remaining bodies P1, P2, ... ordered by angle."""
    north, south = markers.get((2, 0)), markers.get((3, 0))
    n = min(range(len(signals)), key=lambda i: signals[i].distance(north))
    rest = [i for i in range(len(signals)) if i != n]
    if south is not None:
        s = min(rest, key=lambda i: signals[i].distance(south))
    else:
        s = next((i for i in rest if signals[i].contains(shapely.Point(0, 0))), max(rest, key=lambda i: signals[i].area))
    names = [""] * len(signals)
    names[s], names[n] = "S", "N"
    a0 = np.degrees(np.arctan2(signals[n].centroid.y, signals[n].centroid.x))
    for i in rest:
        if i != s:
            a = np.degrees(np.arctan2(signals[i].centroid.y, signals[i].centroid.x))
            names[i] = f"P{int(round(((a - a0) % 360) / (360 / n_slots))) % n_slots}"
    return names
