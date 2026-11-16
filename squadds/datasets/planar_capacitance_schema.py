"""Schema and conventions of the ``SQuADDS/SQuADDS_PlanarCapacitance`` dataset (pure Python; pyarrow imported lazily).

Conventions shared by every table:

- Capacitance in fF, Maxwell (short-circuit) signs: diagonal > 0, off-diagonal < 0; ``mutual_fF = -C_fF`` off the diagonal.
- Matrices are long format, one row per entry ``(net_a, net_b)`` with ``a_idx <= b_idx`` in the design's ``net_order``.
  Ground (``G``) is always the first net. Nets are never merged: N is inferred from the layout.
- ``design_id = "ref/<collection>/<design>"``; ``layout_id = "sha256:<file bytes>"``; ``geom_hash`` = sha256 of the canonical
  polygon multiset (layer, datatype, vertices rounded to 1 nm, sorted) = the duplicate-detection key across SQuADDS.
- Every processed row carries ``prior_version`` / ``embedding_version`` and the sha256 of the code that produced it.
- Test-set labels (``split == "test"``) live in the private ``SQuADDS_PlanarCapacitance-heldout``, except the interdigital
  in-family test designs (exp6-8), whose labels were already public in ``SQuADDS/SQuADDS_DB`` and are released here too.

    from squadds.datasets.planar_capacitance_schema import TABLES, arrow_schema
    arrow_schema("labels")
"""

from __future__ import annotations

REPO_ID = "SQuADDS/SQuADDS_PlanarCapacitance"
HELDOUT_REPO_ID = "SQuADDS/SQuADDS_PlanarCapacitance-heldout"
SCHEMA_VERSION = "2.0.0"
DATASET_VERSION = "2.0.0-dev"
LICENSE = "CC-BY-4.0"
UNITS = {"capacitance": "fF", "length": "um", "time": "s"}
SPLITS = ("train", "val", "test", "excluded", "none")
GROUND = "G"

EMBEDDING_VERSION = "planar-capacitance-graph-v1"
COARSE_PRIOR_VERSION = "v1_frozen_micro_mom_loc_p0.375"
OPTIMIZED_PRIOR_VERSION = "v2m_sf_fc_dipfill_margin_slabfar_R1.5_3"
COARSE_PPG = 0.375
OPTIMIZED_PPG = (1.5, 3.0)

NODE_FEATURES = (
    "compactness",
    *(f"fourier_{i:02d}" for i in range(33)),
    *(f"gran_{i:02d}" for i in range(16)),
    "log_Imajor_over_A2",
    "log_area",
    "log_aspect",
    "log_elongation",
    "log_min_halfwidth",
    "log_perimeter",
    "log_width",
    "n_holes",
    "role_ground",
    "solidity",
    "missing_log_min_halfwidth",
)
EDGE_FEATURES = (
    "is_ground_pair",
    "lg_log_cps",
    "lg_log_dmin",
    "lg_mom_0.5",
    "lg_mom_1",
    "lg_mom_2",
    *(f"lg_sep_{i:02d}" for i in range(10)),
    "log_cps",
    "log_dmin",
    "lt_e150",
    "lt_e400",
    "lt_e50",
    "lt_g150",
    "lt_g400",
    "lt_g50",
    "lt_o150",
    "lt_o400",
    "lt_o50",
    "mom_0.5",
    "mom_1",
    "mom_2",
    *(f"sep_{i:02d}" for i in range(10)),
    "share_absdiff",
    "share_mean",
)
GLOBAL_FEATURES = ("log_chip_area", "log_pocket_area", "log_substrate_t", "eps_eff")

SQUADDS_CONFIGS = {
    "exp6": ("coupler", "GeneralizedCapNInterdigital"),
    "exp7": ("coupler", "GeneralizedCapNInterdigital"),
    "exp8": ("coupler", "GeneralizedCapNInterdigital"),
    "exp9": ("coupler", "GeneralizedCapNInterdigital"),
    "exp10": ("coupler", "GeneralizedCapConcentric"),
    "exp11": ("coupler", "GeneralizedCapStar"),
    "exp12": ("coupler", "GeneralizedCapStar"),
    "exp13": ("qubit", "Xmon"),
    "exp14": ("qubit", "TransmonPocket"),
    "exp15": ("coupler", "ShieldedCapConcentric"),
}


def squadds_config(collection: str) -> str | None:
    """SQuADDS_DB-style config name (``component-ComponentClass-cap_matrix``) of a collection, or None (canonical store only)."""
    c = SQUADDS_CONFIGS.get(collection)
    return f"{c[0]}-{c[1]}-cap_matrix" if c else None


_S, _I8, _I16, _I32, _I64, _F32, _F64, _B = "string", "int8", "int16", "int32", "int64", "float32", "float64", "bool"
_LS, _LF32, _LF64 = "list<string>", "list<float32>", "list<float64>"
_PROV = (("source", _S), ("contributor_id", _S), ("licence_id", _S), ("dataset_version_added", _S))
_PROC = (("build_code_sha256", _S),)

TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "designs": (
        ("design_id", _S),
        ("layout_id", _S),
        ("geom_hash", _S),
        ("collection", _S),
        ("family", _S),
        ("paper_family", _S),
        ("model_family", _S),
        ("generator", _S),
        ("generator_seed", _I64),
        ("params", _S),
        ("component_class", _S),
        ("squadds_config", _S),
        ("squadds_db_ref", _S),
        ("n_conductors", _I16),
        ("net_order", _LS),
        ("chip_w_um", _F64),
        ("chip_h_um", _F64),
        ("min_gap_um", _F64),
        ("metal_area_um2", _F64),
        ("split", _S),
        ("split_v1", _S),
        ("group", _S),
        ("labelled", _B),
        ("notes", _S),
        *_PROV,
    ),
    "layouts": (
        ("layout_id", _S),
        ("path", _S),
        ("v1_path", _S),
        ("geom_hash", _S),
        ("sha256", _S),
        ("unit_m", _F64),
        ("precision_m", _F64),
        ("top_cell", _S),
        ("layer_map", _S),
        ("n_polygons", _I32),
        ("n_vertices", _I32),
        ("bbox_um", _LF64),
        ("file_bytes", _I32),
        ("n_conductors_layout", _I16),
        *_PROV,
    ),
    "simulations": (
        ("simulation_id", _S),
        ("design_id", _S),
        ("sim_tag", _S),
        ("solver", _S),
        ("solver_version", _S),
        ("role", _S),
        ("substrate_material", _S),
        ("eps_r", _F64),
        ("substrate_um", _F64),
        ("tan_delta", _F64),
        ("metal_model", _S),
        ("metal_t_um", _F64),
        ("freq_GHz", _F64),
        ("setup", _S),
        ("status", _S),
        ("passes_completed", _I16),
        ("converged", _B),
        ("delta_pct_final", _F64),
        ("delta_target_pct", _F64),
        ("pass_history", _S),
        ("cores", _I16),
        ("solve_s", _F64),
        ("wall_s", _F64),
        ("peak_mem_GB", _F64),
        ("machine", _S),
        ("qc_sym_rel", _F64),
        ("qc_symmetric", _B),
        ("qc_signs", _B),
        ("qc_closure_max_rel", _F64),
        ("qc_closure", _B),
        ("qc_min_eig_signal_fF", _F64),
        ("qc_pd", _B),
        ("qc_last_pass_change_max_pct", _F64),
        ("qc_convergence", _S),
        ("qc_ok", _B),
        ("provenance", _S),
        ("split", _S),
        *_PROV,
    ),
    "labels": (
        ("simulation_id", _S),
        ("design_id", _S),
        ("net_a", _S),
        ("net_b", _S),
        ("a_idx", _I16),
        ("b_idx", _I16),
        ("C_fF", _F64),
        ("mutual_fF", _F64),
        ("kind", _S),
        ("scored", _B),
        ("dist_um", _F32),
        ("split", _S),
        *_PROV,
    ),
    "label_passes": (
        ("simulation_id", _S),
        ("design_id", _S),
        ("net_a", _S),
        ("net_b", _S),
        ("a_idx", _I16),
        ("b_idx", _I16),
        ("pass", _I16),
        ("C_fF", _F64),
        ("split", _S),
    ),
    "geometric_embedding_nodes": (
        ("design_id", _S),
        ("node_idx", _I16),
        ("net", _S),
        *((f, _F32) for f in NODE_FEATURES),
        ("embedding_version", _S),
        ("embedding_code_sha256", _S),
        ("source", _S),
        ("split", _S),
        *_PROC,
    ),
    "geometric_embedding_edges": (
        ("design_id", _S),
        ("i", _I16),
        ("j", _I16),
        ("net_a", _S),
        ("net_b", _S),
        *((f, _F32) for f in EDGE_FEATURES),
        ("embedding_version", _S),
        ("embedding_code_sha256", _S),
        ("source", _S),
        ("split", _S),
        *_PROC,
    ),
    "geometric_embedding_globals": (
        ("design_id", _S),
        ("n_conductors", _I16),
        *((f, _F32) for f in GLOBAL_FEATURES),
        ("embedding_version", _S),
        ("embedding_code_sha256", _S),
        ("source", _S),
        ("split", _S),
        *_PROC,
    ),
    "physics_prior": (
        ("design_id", _S),
        ("net_a", _S),
        ("net_b", _S),
        ("i", _I16),
        ("j", _I16),
        ("ppg", _F32),
        ("mutual_fF", _F64),
        ("C_fF", _F64),
        ("prior_version", _S),
        ("prior_code_sha256", _S),
        ("machine", _S),
        ("source", _S),
        ("split", _S),
        *_PROC,
    ),
    "optimized_physics_prior": (
        ("design_id", _S),
        ("net_a", _S),
        ("net_b", _S),
        ("i", _I16),
        ("j", _I16),
        ("mutual_1p5_fF", _F32),
        ("mutual_3_fF", _F32),
        ("mutual_R_fF", _F64),
        ("C_R_fF", _F64),
        ("prior_version", _S),
        ("prior_code_sha256", _S),
        ("source", _S),
        ("split", _S),
        *_PROC,
    ),
    "model_input": (
        ("design_id", _S),
        ("n_conductors", _I16),
        ("net_order", _LS),
        ("model_family", _S),
        ("X", _LF32),
        ("E", _LF32),
        ("G", _LF32),
        ("pri", _LF32),
        ("sig", "list<bool>"),
        ("embedding_version", _S),
        ("prior_version", _S),
        ("source", _S),
        ("split", _S),
        *_PROC,
    ),
}

DESCRIPTIONS = {
    "designs": "one row per design: options JSON, family, generator, SQuADDS component class / config / DB reference, split",
    "layouts": "one row per distinct GDS file (copied byte-identical under gds/<collection>/), sha256, geom_hash, N",
    "simulations": "one row per solver run: solver, setup, stack, convergence, passes, timing, recomputed label QC flags",
    "labels": "final Maxwell matrix (last complete adaptive pass), long format, fF; test rows only in the heldout repo",
    "label_passes": "every adaptive pass of every simulation with a pass history, long format, fF",
    "geometric_embedding_nodes": f"{EMBEDDING_VERSION}: 61 conductor features (60 + missing flag), ground = node 0",
    "geometric_embedding_edges": f"{EMBEDDING_VERSION}: 42 pair features per conductor pair i < j",
    "geometric_embedding_globals": f"{EMBEDDING_VERSION}: 4 global features",
    "physics_prior": "single thin-film MoM solve at p = 0.375 (mutual entries i < j)",
    "optimized_physics_prior": "C^R = R(1.5, 3) on the corrected stack (v2m + substrate far field) with its two solves",
    "model_input": "packed per-layout input of QCap (X, E, G, pri, sig)",
}


def _arrow_type(t: str):
    import pyarrow as pa

    base = {
        "string": pa.string(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    if t.startswith("list<"):
        return pa.list_(base[t[5:-1]])
    return base[t]


def arrow_schema(table: str):
    """pyarrow schema of one table (column order and types as written)."""
    import pyarrow as pa

    return pa.schema([pa.field(n, _arrow_type(t)) for n, t in TABLES[table]])


def column_names(table: str) -> list[str]:
    return [n for n, _ in TABLES[table]]
