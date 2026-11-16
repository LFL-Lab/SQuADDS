"""Query the ``SQuADDS/SQuADDS_PlanarCapacitance`` dataset (raw and processed tables) with pandas / pyarrow only.

No ``[ml]`` dependency: this module needs pandas, pyarrow and huggingface_hub (all core SQuADDS dependencies), so
``from squadds.ml.capacitance.dataset import PlanarCapacitanceDB`` works on a plain ``pip install squadds``.

    db = PlanarCapacitanceDB()                                  # HF repo
    db = PlanarCapacitanceDB(local_dir="/path/to/SQuADDS_PlanarCapacitance")   # or env SQUADDS_PLANAR_CAP_DATA
    ids = db.designs(family="interdigital", split="val").design_id[:5]
    db.labels(ids)        # {design_id: N x N Maxwell matrix (pandas DataFrame, fF, ground first)}

Tables are read file by file with pyarrow column / row filters, so a query never loads more of the dataset than it asks for.
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from squadds.datasets.planar_capacitance_schema import (
    GLOBAL_FEATURES,
    NODE_FEATURES,
    REPO_ID,
    TABLES,
)

ENV_VAR = "SQUADDS_PLANAR_CAP_DATA"
# pinned HF commit of REPO_ID (2026-11-14: coarse prior table renamed physics_prior); revision="main" for the latest
DATASET_REVISION = "c79953313874e9c116ca2f39a414d92025d18484"

TABLE_GLOBS = {
    "designs": "data/designs/*.parquet",
    "layouts": "data/layouts/*.parquet",
    "simulations": "data/simulations/*.parquet",
    "labels": "data/labels/*.parquet",
    "label_passes": "data/label_passes/*.parquet",
    "geometric_embedding_nodes": "data/geometric_embedding/nodes-*.parquet",
    "geometric_embedding_edges": "data/geometric_embedding/edges-*.parquet",
    "geometric_embedding_globals": "data/geometric_embedding/globals-*.parquet",
    "physics_prior": "data/physics_prior/*.parquet",
    "optimized_physics_prior": "data/optimized_physics_prior/*.parquet",
    "model_input": "data/model_input/*.parquet",
}
DESIGN_COLUMNS = [c for c, _ in TABLES["designs"] if c != "params"]
NEAREST_FEATURES = (
    [*GLOBAL_FEATURES, "log_n_conductors"] + [f"mean_{f}" for f in NODE_FEATURES] + [f"max_{f}" for f in NODE_FEATURES]
)


def long_to_matrix(
    df: pd.DataFrame, value: str, i: str = "a_idx", j: str = "b_idx", a: str = "net_a", b: str = "net_b"
):
    """Symmetric N x N DataFrame (index = columns = net names, in index order) from long-format entries ``i <= j``.

    A table that stores mutual entries only (no diagonal) gets the Maxwell diagonal from closure:
    ``C_ii = -sum_{j != i} C_ij`` (the ground net is one of the nets, so this is exact for a closed system).
    """
    n = int(max(df[i].max(), df[j].max())) + 1
    names = [None] * n
    for ii, jj, na, nb in zip(df[i], df[j], df[a], df[b]):
        names[int(ii)], names[int(jj)] = na, nb
    C = np.full((n, n), np.nan)
    ii, jj, v = df[i].to_numpy(int), df[j].to_numpy(int), df[value].to_numpy(float)
    C[ii, jj] = v
    C[jj, ii] = v
    diag = np.isnan(np.diag(C))
    if diag.any():
        off = np.where(np.eye(n, dtype=bool), 0.0, np.nan_to_num(C))
        C[diag, diag] = -off.sum(1)[diag]
    return pd.DataFrame(C, index=names, columns=names)


class PlanarCapacitanceDB:
    """Read access to the planar capacitance dataset.

    Args:
        revision: HF revision (commit sha, tag or branch) of ``repo_id``; default ``DATASET_REVISION`` (the pinned commit),
            ``"main"`` for the latest; ignored with a local directory.
        local_dir: a local copy of the repo (same layout as on HF). Default: env ``SQUADDS_PLANAR_CAP_DATA``, else HF.
        repo_id: HF dataset repo (default ``SQuADDS/SQuADDS_PlanarCapacitance``).
        token: HF token (default: the logged-in token, if any).
    """

    def __init__(self, revision: str | None = None, local_dir=None, repo_id: str = REPO_ID, token: str | None = None):
        local_dir = local_dir or os.environ.get(ENV_VAR)
        self.local_dir = Path(local_dir).expanduser() if local_dir else None
        if self.local_dir is not None and not self.local_dir.is_dir():
            raise FileNotFoundError(f"planar capacitance data directory not found: {self.local_dir}")
        self.repo_id, self.revision, self.token = repo_id, revision or DATASET_REVISION, token
        self._repo_files: list[str] | None = None
        self._nearest_cache: dict = {}

    def __repr__(self) -> str:
        where = str(self.local_dir) if self.local_dir else f"hf://datasets/{self.repo_id}@{self.revision}"
        return f"PlanarCapacitanceDB({where})"

    # ------------------------------------------------------------------------------------------------ files
    def _list_repo_files(self) -> list[str]:
        if self._repo_files is None:
            from huggingface_hub import HfApi

            self._repo_files = HfApi(token=self.token).list_repo_files(
                self.repo_id, repo_type="dataset", revision=self.revision
            )
        return self._repo_files

    def _download(self, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(self.repo_id, filename, repo_type="dataset", revision=self.revision, token=self.token)
        )

    def tables(self) -> dict[str, list[str]]:
        """Available tables -> their files (repo-relative paths); tables with no file are left out."""
        if self.local_dir is not None:
            rel = [p.relative_to(self.local_dir).as_posix() for p in self.local_dir.glob("data/**/*.parquet")]
        else:
            rel = self._list_repo_files()
        out = {t: sorted(f for f in rel if fnmatch.fnmatch(f, g)) for t, g in TABLE_GLOBS.items()}
        return {t: f for t, f in out.items() if f}

    def files(self, table: str) -> list[Path]:
        if table not in TABLE_GLOBS:
            raise KeyError(f"unknown table {table!r}; tables: {sorted(TABLE_GLOBS)}")
        if self.local_dir is not None:
            return sorted(self.local_dir.glob(TABLE_GLOBS[table]))
        return [self._download(f) for f in self.tables().get(table, [])]

    def read(self, table: str, columns=None, design_ids=None, filters=None) -> pd.DataFrame:
        """Rows of one table (all its split files) with optional column selection and pyarrow row filters
        (``[("split", "=", "val")]``); ``design_ids`` adds ``design_id in design_ids``."""
        import pyarrow.parquet as pq

        flt = list(filters or [])
        if design_ids is not None:
            ids = list(dict.fromkeys(design_ids))
            if not ids:
                return pd.DataFrame(columns=columns or [c for c, _ in TABLES.get(table, ())])
            flt.append(("design_id", "in", ids))
        parts = []
        for f in self.files(table):
            names = set(pq.read_schema(f).names)
            if any(c not in names for c, _, _ in flt):
                continue
            cols = [c for c in columns if c in names] if columns else None
            t = pq.read_table(f, columns=cols, filters=flt or None)
            if t.num_rows:
                parts.append(t.to_pandas())
        if not parts:
            return pd.DataFrame(columns=columns or [c for c, _ in TABLES.get(table, ())])
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------------------------------------ raw tables
    def designs(
        self,
        family=None,
        split=None,
        n_conductors=None,
        collection=None,
        config=None,
        columns=None,
        design_ids=None,
    ) -> pd.DataFrame:
        """Design rows (without the ``params`` JSON unless asked for in ``columns``). ``family`` matches ``family``,
        ``paper_family`` or ``model_family``; ``config`` is the SQuADDS config name; each filter takes a value or a list."""

        def as_list(v):
            return [v] if isinstance(v, (str, int, np.integer)) else list(v)

        flt = []
        for col, v in (("split", split), ("collection", collection), ("squadds_config", config)):
            if v is not None:
                flt.append((col, "in", as_list(v)))
        df = self.read("designs", columns=columns or DESIGN_COLUMNS, design_ids=design_ids, filters=flt)
        if family is not None:
            fam = set(as_list(family))
            mask = np.zeros(len(df), bool)
            for col in ("family", "paper_family", "model_family"):
                if col in df:
                    mask |= df[col].isin(fam).to_numpy()
            df = df[mask]
        if n_conductors is not None:
            df = df[df["n_conductors"].isin([float(x) for x in as_list(n_conductors)])]
        return df.reset_index(drop=True)

    def params(self, design_ids) -> dict[str, dict]:
        """Generator / component options of each design (parsed ``params`` JSON)."""
        df = self.read("designs", columns=["design_id", "params"], design_ids=design_ids)
        return {d: json.loads(p) if p else {} for d, p in zip(df.design_id, df.params)}

    def simulations(self, design_ids=None, role: str | None = None, columns=None) -> pd.DataFrame:
        flt = [("role", "=", role)] if role else None
        return self.read("simulations", columns=columns, design_ids=design_ids, filters=flt)

    def label_passes(self, design_ids, columns=None) -> pd.DataFrame:
        return self.read("label_passes", columns=columns, design_ids=design_ids)

    def labels(self, design_ids, role: str | None = "reference") -> dict[str, pd.DataFrame]:
        """Final Maxwell matrices (fF, ground first) keyed by design_id (``role`` = simulation role, default the reference
        Q3D solve); ``role=None`` returns every simulation keyed by simulation_id. Designs without a public label (test
        split) are absent."""
        sims = self.simulations(design_ids, role=role, columns=["simulation_id", "design_id", "role"])
        if sims.empty:
            return {}
        lab = self.read(
            "labels",
            columns=["simulation_id", "design_id", "net_a", "net_b", "a_idx", "b_idx", "C_fF"],
            filters=[("simulation_id", "in", sims.simulation_id.tolist())],
        )
        key = "design_id" if role else "simulation_id"
        return {k: long_to_matrix(g, "C_fF") for k, g in lab.groupby(key, sort=False)}

    def gds_path(self, design_id: str) -> Path:
        """Local path of the design's GDS (downloaded from the HF repo when needed)."""
        d = self.designs(design_ids=[design_id], columns=["design_id", "layout_id"])
        if d.empty:
            raise KeyError(f"unknown design_id {design_id!r}")
        lay = self.read("layouts", columns=["layout_id", "path"], filters=[("layout_id", "=", d.layout_id.iloc[0])])
        if lay.empty:
            raise KeyError(f"no layout row for {design_id!r} ({d.layout_id.iloc[0]})")
        rel = lay.path.iloc[0]
        return self.local_dir / rel if self.local_dir is not None else self._download(rel)

    # ------------------------------------------------------------------------------------------------ processed tables
    def _prior(self, table: str, design_ids, value: str) -> dict[str, pd.DataFrame]:
        df = self.read(table, design_ids=design_ids)
        return {k: long_to_matrix(g, value, "i", "j") for k, g in df.groupby("design_id", sort=False)}

    def physics_prior(self, design_ids, value: str = "C_fF") -> dict[str, pd.DataFrame]:
        """Single thin-film MoM solve at p = 0.375 as N x N Maxwell matrices (fF, ground first)."""
        return self._prior("physics_prior", design_ids, value)

    def optimized_physics_prior(self, design_ids, value: str = "C_R_fF") -> dict[str, pd.DataFrame]:
        """C^R = R(1.5, 3) on the corrected stack as N x N Maxwell matrices (``value="mutual_1p5_fF"`` / ``"mutual_3_fF"``
        for the two solves, diagonal from closure)."""
        return self._prior("optimized_physics_prior", design_ids, value)

    def model_input(self, design_ids) -> pd.DataFrame:
        return self.read("model_input", design_ids=design_ids)

    def embedding(self, design_ids) -> dict[str, dict]:
        """``planar-capacitance-graph-v1`` per design: ``nodes`` (one row per conductor, ground = node 0), ``edges`` (pairs
        i < j) and ``globals`` (dict)."""
        nodes = self.read("geometric_embedding_nodes", design_ids=design_ids)
        edges = self.read("geometric_embedding_edges", design_ids=design_ids)
        glob = self.read("geometric_embedding_globals", design_ids=design_ids)
        out = {}
        for d, g in glob.groupby("design_id", sort=False):
            out[d] = {
                "nodes": nodes[nodes.design_id == d].sort_values("node_idx").reset_index(drop=True),
                "edges": edges[edges.design_id == d].sort_values(["i", "j"]).reset_index(drop=True),
                "globals": g.iloc[0].to_dict(),
            }
        return out

    # ------------------------------------------------------------------------------------------------ nearest neighbours
    def pooled_embedding(self, design_ids=None, family=None, split=None) -> pd.DataFrame:
        """One vector per design (index design_id): the 4 global features, log N, and the mean and the max of each of the
        61 node features over the signal conductors (ground node excluded); 127 values, not standardized."""
        if design_ids is None and (family is not None or split is not None):
            design_ids = self.designs(
                family=family, split=split, columns=["design_id", "family", "paper_family", "model_family", "split"]
            ).design_id
        cols = ["design_id", "node_idx", *NODE_FEATURES]
        nodes = self.read("geometric_embedding_nodes", columns=cols, design_ids=design_ids)
        glob = (
            self.read(
                "geometric_embedding_globals",
                columns=["design_id", "n_conductors", *GLOBAL_FEATURES],
                design_ids=design_ids,
            )
            .drop_duplicates("design_id")
            .set_index("design_id")
        )
        sig = nodes[nodes.node_idx > 0].drop(columns="node_idx").groupby("design_id")
        mean = sig.mean().add_prefix("mean_")
        mx = sig.max().add_prefix("max_")
        g = glob[list(GLOBAL_FEATURES)].astype(float)
        g["log_n_conductors"] = np.log(glob["n_conductors"].astype(float))
        out = g.join(mean, how="inner").join(mx, how="inner")
        return out[NEAREST_FEATURES]

    def nearest(self, query, k: int = 5, family=None, split=None) -> pd.DataFrame:
        """The k designs closest to ``query`` (a design_id, or a 127-value vector in ``pooled_embedding`` order) in the
        pooled embedding space, standardized per feature over the candidate set (Euclidean distance).

        Candidates: designs of ``family`` (default: the query design's ``model_family``; all designs when the query is a
        vector and no family is given) and ``split``. This is a simple geometric similarity, not a capacitance distance.
        """
        qvec = None
        if isinstance(query, str):
            if family is None:
                d = self.designs(design_ids=[query], columns=["design_id", "model_family"])
                if d.empty:
                    raise KeyError(f"unknown design_id {query!r}")
                family = d.model_family.iloc[0]
        else:
            qvec = np.asarray(query, float).ravel()
            if qvec.size != len(NEAREST_FEATURES):
                raise ValueError(f"query vector must have {len(NEAREST_FEATURES)} values (pooled_embedding order)")
        key = (repr(family), repr(split))
        if key not in self._nearest_cache:
            self._nearest_cache[key] = self.pooled_embedding(family=family, split=split)
        P = self._nearest_cache[key]
        if isinstance(query, str):
            if query not in P.index:
                P = pd.concat([P, self.pooled_embedding(design_ids=[query])])
            qvec = P.loc[query].to_numpy(float)
        X = P.to_numpy(float)
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        sd[~(sd > 0)] = 1.0
        dist = np.sqrt(np.nansum(((X - mu) / sd - (qvec - mu) / sd) ** 2, 1))
        res = pd.DataFrame({"design_id": P.index, "distance": dist})
        if isinstance(query, str):
            res = res[res.design_id != query]
        res = res.nsmallest(k, "distance").reset_index(drop=True)
        meta = self.designs(design_ids=res.design_id, columns=["design_id", "collection", "model_family", "split"])
        return res.merge(meta, on="design_id", how="left")
