"""Scene graph (nodes = conductors, edges = all conductor pairs) and its permutation-invariant pooled vector."""

from __future__ import annotations

import numpy as np

from .blocks import EdgeTable, _cache, body_keys, boundary_points, edge_features_all, global_features, node_features_all
from .proxies import Proxy
from .scene import Scene


def scene_graph(scene: Scene, proxy: Proxy | None = None) -> dict:
    keys = body_keys(scene) if _cache() else None
    bpts, widths = boundary_points(scene)
    nodes = node_features_all(scene, keys)
    edges = edge_features_all(scene, bpts, widths, keys)
    # shielding: share of each endpoint's total geometric coupling carried by this edge
    table = isinstance(edges, EdgeTable)
    if table:
        edges = _shares_table(scene.n, edges)
        if proxy is not None:
            edges = {p: edges[p] for p in edges}
    tot = None if table else np.zeros(scene.n)
    for (i, j), f in (edges.items() if tot is not None else ()):
        tot[i] += f["_cps"]
        tot[j] += f["_cps"]
    for (i, j), f in (edges.items() if tot is not None else ()):
        si, sj = f["_cps"] / max(tot[i], 1e-12), f["_cps"] / max(tot[j], 1e-12)
        f["share_mean"], f["share_absdiff"] = 0.5 * (si + sj), abs(si - sj)
    g = {"names": scene.names(), "global": global_features(scene), "nodes": nodes, "edges": edges}
    if proxy is not None:
        px = proxy.compute(scene, edges)
        for i, v in px["node"].items():
            nodes[i].update({f"proxy_{k}": x for k, x in v.items()})
        for e, v in px["edge"].items():
            edges[e].update({f"proxy_{k}": x for k, x in v.items()})
        g["global"].update({f"proxy_{k}": x for k, x in px["global"].items()})
        if "matrix" in px:
            g["proxy_matrix"] = px["matrix"]
    for f in (edges.values() if tot is not None else ()):
        f.pop("_cps")
    return g


def _shares_table(n: int, E: EdgeTable) -> EdgeTable:
    """The shares of scene_graph on an EdgeTable: the same additions in the same order (np.add.at is sequential)."""
    I, J = np.triu_indices(n, 1)
    cps = E.cols["_cps"]
    tot = np.zeros(n)
    np.add.at(tot, np.c_[I, J].ravel(), np.repeat(cps, 2))
    si, sj = cps / np.maximum(tot[I], 1e-12), cps / np.maximum(tot[J], 1e-12)
    return E.with_cols({"share_mean": 0.5 * (si + sj), "share_absdiff": np.abs(si - sj)}, drop=("_cps",),
                       npcols=("share_mean", "share_absdiff"))


def pooled_vector(g: dict) -> dict:
    """Fixed-length universal vector for any number of conductors: global block, the ground node, sum/mean/max over
    signal nodes, and sum/mean/max over signal-signal and signal-ground edges."""
    out = {f"global.{k}": v for k, v in g["global"].items()}
    roles = ["ground" if n["role_ground"] else "signal" for n in g["nodes"]]
    keys = [k for k in g["nodes"][0] if k != "role_ground"]
    gi = roles.index("ground")
    out.update({f"ground.{k}": g["nodes"][gi][k] for k in keys})
    S = np.array([[n[k] for k in keys] for n, r in zip(g["nodes"], roles) if r == "signal"])
    for agg, fn in (("mean", np.mean), ("max", np.max), ("min", np.min)):
        out.update({f"signal.{agg}.{k}": v for k, v in zip(keys, fn(S, 0))})
    ekeys = [k for k in next(iter(g["edges"].values())) if k != "is_ground_pair"]
    for typ, flag in (("ss", 0.0), ("sg", 1.0)):
        E = np.array([[f[k] for k in ekeys] for f in g["edges"].values() if f["is_ground_pair"] == flag])
        if len(E) == 0:
            E = np.full((1, len(ekeys)), np.nan)
        for agg, fn in (("mean", np.nanmean), ("max", np.nanmax), ("min", np.nanmin)):
            out.update({f"edge_{typ}.{agg}.{k}": v for k, v in zip(ekeys, fn(E, 0))})
    return out
