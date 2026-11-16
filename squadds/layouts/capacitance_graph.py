"""Layout embedding ``planar-capacitance-graph-v1``: the geometric vectors the planar-capacitance models read.

One vector per conductor (60 shape and context features; ground is conductor 0), one per conductor pair (42 features: 17
geometric pair features, the ground-pair flag, 9 local-context ring fractions and 15 ground-block features) and one global
vector (4: log chip area, log pocket area, log substrate thickness, effective permittivity). Exact embedding (the paper
model's; no far-sample decimation). Needs the ``[ml]`` extra.

    from squadds.layouts.capacitance_graph import encode
    emb = encode("chip.gds")
    emb["node_features"].shape, emb["pair_features"].shape, emb["global_features"].shape   # (N, 60), (N(N-1)/2, 42), (4,)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

EMBEDDING_NAME = "planar-capacitance-graph"
EMBEDDING_VERSION = "planar-capacitance-graph-v1"
_DROP_NODE = ("missing_log_min_halfwidth",)  # indicator the networks append; derivable from log_min_halfwidth


def _schema() -> dict:
    import json

    from squadds.ml.capacitance._core import lp_surrogate as S

    return json.loads(json.dumps(S.SCHEMA))


def feature_names() -> dict:
    """Names of the node (60), pair (42) and global (4) features."""
    from squadds.ml import require_ml

    require_ml()
    s = _schema()
    return {
        "node": [k for k in s["nodekeys"] if k not in _DROP_NODE],
        "pair": list(s["edgekeys"]),
        "global": list(s["globkeys"]),
    }


def code_sha256() -> str:
    """sha256 of the ported stage's SOURCES.json (the sha256 of every file of the embedding and prior code)."""
    p = Path(__file__).resolve().parents[1] / "ml" / "capacitance" / "_core" / "SOURCES.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def encode(layout, threads: int = 2, **prepare_kwargs) -> dict:
    """Embedding of a GDS path, gdstk Library / Cell or PreparedLayout: arrays (float32), feature names, conductor names
    (ground ``G`` first; pairs i < j in that order), version and code sha256."""
    from squadds.ml import require_ml

    require_ml()
    import numpy as np

    from squadds.ml.capacitance import _runtime as RT
    from squadds.ml.capacitance._core import api as A
    from squadds.ml.capacitance._core import x1_core as X
    from squadds.ml.capacitance._core._tlsq.uge import blocks as B
    from squadds.ml.capacitance._core._tlsq.uge import localtok as lt
    from squadds.ml.capacitance._core._tlsq.uge.graph import scene_graph
    from squadds.ml.capacitance.predictor import _EMB, _get, _put, quiet

    lay = RT.prepare_layout(layout, **prepare_kwargs)
    lhash = RT.layout_hash(lay)
    s = _schema()
    with RT.LOCK, quiet():
        threads = RT.configure(threads)
        old = dict(B.FAR_DECIMATION)
        B.FAR_DECIMATION.update(k=0, d_far=100.0)
        try:
            key = (lhash, 0, 100.0, B.EMBED_MODE["mode"], B.EDGE_TABLE["on"], threads)
            row = _get(_EMB, key)
            if row is None:
                sc = A.scene_from_prepared(lay)
                row = lt.add_tokens(sc, scene_graph(sc))
                _put(_EMB, key, row)
            mode = B.EMBED_MODE["mode"]
        finally:
            B.FAR_DECIMATION.clear()
            B.FAR_DECIMATION.update(old)
    Xa, E, G, pairs = X.raw_arrays(row, s["nodekeys"], s["edgekeys"], s["globkeys"])
    keep = [i for i, k in enumerate(s["nodekeys"]) if k not in _DROP_NODE]
    names = feature_names()
    return {
        "version": EMBEDDING_VERSION,
        "names": list(row["names"]),
        "node_features": np.ascontiguousarray(Xa[:, keep]),
        "pair_features": E,
        "pairs": pairs.astype(np.int64),
        "global_features": G,
        "node_feature_names": names["node"],
        "pair_feature_names": names["pair"],
        "global_feature_names": names["global"],
        "embedding_mode": mode,
        "layout_hash": lhash,
        "code_sha256": code_sha256(),
        "source": str(lay.source),
    }
