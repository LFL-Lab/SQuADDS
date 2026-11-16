"""Prediction result: the full Maxwell capacitance matrix (fF, ground first) with checks, timing and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

CONVENTION = "Maxwell matrix: diagonal = self, off-diagonal = -mutual (negative); row/column 0 = ground (G)"


def physical_checks(C) -> dict:
    """Symmetry, Maxwell signs, closure (row sums of the signal rows >= 0) and positive definiteness of the signal block."""
    C = np.asarray(C, float)
    n = len(C)
    off = ~np.eye(n, dtype=bool)
    scale = max(1.0, float(np.abs(C).max()))
    sym = float(np.max(np.abs(C - C.T)))
    rows = C.sum(1)[1:]
    eig = float(np.linalg.eigvalsh(C[1:, 1:]).min()) if n > 1 else None
    chk = {
        "symmetric_max_abs": sym,
        "symmetric": sym <= 1e-6 * scale,
        "diagonal_positive": bool(np.all(np.diag(C) > 0)),
        "offdiag_all_negative": bool(np.all(C[off] < 0)),
        "closure_min_row_sum": float(rows.min()) if n > 1 else None,
        "closure": bool(n == 1 or rows.min() >= -1e-9 * scale),
        "signal_block_min_eig": eig,
        "positive_definite": eig is None or eig > 0,
    }
    chk["passed"] = all(
        chk[k] for k in ("symmetric", "diagonal_positive", "offdiag_all_negative", "closure", "positive_definite")
    )
    return chk


def physical_warning(chk: dict) -> str | None:
    """Readable warning when a check of ``physical_checks`` fails (None when the matrix passes)."""
    if chk["passed"]:
        return None
    bad = [
        k.replace("_", " ")
        for k in ("symmetric", "diagonal_positive", "offdiag_all_negative", "closure", "positive_definite")
        if not chk[k]
    ]
    return (
        f"physical check failed: {', '.join(bad)} (smallest eigenvalue of the signal block {chk['signal_block_min_eig']:.4g} fF, "
        f"smallest signal row sum {chk['closure_min_row_sum']:.4g} fF). This matrix is not physical: do not use it. A model that "
        "solves the physics (QCap) is the safer choice, especially outside the trained component families."
    )


@dataclass
class CapacitanceResult:
    names: list[str]
    matrix_fF: np.ndarray
    model: str
    title: str = ""
    prior_version: str | None = None
    prior_stack: dict | None = None
    weights_sha256: list[str] = field(default_factory=list)
    learned_prior_sha256: str | None = None
    timing_s: dict = field(default_factory=dict)
    threads: int = 2
    machine: str = ""
    checks: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    substrate: dict = field(default_factory=dict)
    ground_detection: dict | None = (
        None  # pieces and roles when ground and conductors share a layer (see inspect_layout)
    )
    source: str = ""
    layout_hash: str = ""
    cached: bool = False
    package: str = ""
    units: str = "fF"
    convention: str = CONVENTION

    def __post_init__(self):
        self.matrix_fF = np.asarray(self.matrix_fF, float)
        if not self.checks:
            self.checks = physical_checks(self.matrix_fF)

    @property
    def n_conductors(self) -> int:
        return len(self.names)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.matrix_fF, index=list(self.names), columns=list(self.names))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matrix_fF"] = self.matrix_fF.tolist()
        return d

    def to_sim_results(self, names: dict | None = None) -> dict:
        """The SQuADDS_DB ``sim_results`` shape: pairwise positive magnitudes ``<a>_to_<b>`` (self terms on the diagonal,
        ground named ``ground``), the signed ``maxwell_matrix`` as ``C_<a>_<b>`` (ground ``G``) and ``units``. ``names`` renames
        conductors (e.g. ``{"C1": "north", "C2": "south"}``)."""
        names = names or {}
        lab = [names.get(n, n) for n in self.names]
        C = self.matrix_fF
        sig = list(range(1, len(lab)))
        out: dict = {}
        for a in sig:
            out[f"{lab[a]}_to_{lab[a]}"] = float(C[a, a])
            for b in sig:
                if b > a:
                    out[f"{lab[a]}_to_{lab[b]}"] = float(abs(C[a, b]))
            out[f"{lab[a]}_to_ground"] = float(abs(C[a, 0]))
        out["ground_to_ground"] = float(C[0, 0])
        out["maxwell_matrix"] = {
            f"C_{lab[i]}_{lab[j]}": float(C[i, j]) for i in range(len(lab)) for j in range(len(lab))
        }
        out["units"] = "fF"
        return out

    def __repr__(self) -> str:
        return (
            f"CapacitanceResult({self.model}, N={self.n_conductors}, names={self.names}, "
            f"total {self.timing_s.get('total_s', float('nan')):.3g} s, checks {'passed' if self.checks.get('passed') else 'FAILED'})"
        )
