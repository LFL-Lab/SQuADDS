#!/usr/bin/env python
"""Build the artifact-driven TopoCap transfer-learning report notebook.

The generated notebook is intentionally executable before experiment artifacts exist.
Every empirical figure checks its compact input table and renders an explicit waiting
panel rather than substituting example values. Once a result directory is available,
the same notebook becomes the executed research report without changing its narrative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
from pathlib import Path

import nbformat as nbf
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "tutorials/Tutorial-18_Topology_General_Transfer_Learning.ipynb"
RESULT_ARTIFACTS = (
    "data_audit",
    "learning_curves",
    "uncertainty",
    "ablations",
    "topology_checks",
    "diffusion_decision",
)
REPORT_ARTIFACT_SCHEMA_VERSION = "topocap-report-artifacts-1.0.0"
FULL_STUDY_STATUSES = {"COMPLETE", "EXPLORATORY_COMPLETE"}


def _source(value: str) -> str:
    return textwrap.dedent(value).strip()


def _markdown(value: str):
    return nbf.v4.new_markdown_cell(_source(value))


def _code(value: str, *, hidden: bool = False):
    source = _source(value)
    if hidden and not source.startswith("# %% hide input"):
        source = "# %% hide input\n" + source
    cell = nbf.v4.new_code_cell(source)
    if hidden:
        cell["metadata"]["tags"] = ["hide-input"]
        cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


def _set_deterministic_cell_ids(notebook) -> None:
    for index, cell in enumerate(notebook["cells"]):
        payload = f"{index}:{cell['cell_type']}:{cell['source']}".encode()
        cell["id"] = hashlib.sha256(payload).hexdigest()[:16]


def _validate_result_directory(results_dir: Path) -> list[str]:
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return [f"manifest.json is unreadable: {error}"]
    if not isinstance(manifest, dict):
        return ["manifest.json must contain an object"]

    problems: list[str] = []
    if manifest.get("schema_version") != REPORT_ARTIFACT_SCHEMA_VERSION:
        problems.append("manifest schema_version is unsupported")
    if manifest.get("study_status") not in FULL_STUDY_STATUSES:
        problems.append(f"study_status is not complete: {manifest.get('study_status')!r}")
    state_digest = manifest.get("state_digest")
    if not isinstance(state_digest, str) or len(state_digest) != 64:
        problems.append("state_digest is missing or malformed")

    root = results_dir.resolve()
    descriptors = manifest.get("artifacts")
    if not isinstance(descriptors, dict):
        return [*problems, "manifest artifacts must be an object"]
    for logical_name in RESULT_ARTIFACTS:
        descriptor = descriptors.get(logical_name)
        if not isinstance(descriptor, dict):
            problems.append(f"{logical_name}: descriptor is missing")
            continue
        if descriptor.get("schema_version") != REPORT_ARTIFACT_SCHEMA_VERSION:
            problems.append(f"{logical_name}: schema_version is unsupported")
        relative_path = descriptor.get("path")
        if not isinstance(relative_path, str):
            problems.append(f"{logical_name}: path is missing")
            continue
        path = (results_dir / relative_path).resolve()
        if path == root or root not in path.parents:
            problems.append(f"{logical_name}: path escapes the result directory")
            continue
        if not path.is_file():
            problems.append(f"{logical_name}: artifact is missing")
            continue
        expected_hash = descriptor.get("sha256")
        if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
            problems.append(f"{logical_name}: SHA-256 does not match")
        try:
            observed_rows = len(_read_result_table(path))
        except Exception as error:
            problems.append(f"{logical_name}: unreadable ({type(error).__name__}: {error})")
            continue
        if descriptor.get("rows") != observed_rows:
            problems.append(f"{logical_name}: row count does not match manifest")
    return problems


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_result_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported artifact type: {path.suffix}")


def build_notebook(results_dir: Path):
    embedded_results_dir = json.dumps(os.fspath(results_dir))
    notebook = nbf.v4.new_notebook()
    notebook["nbformat_minor"] = 5
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "SQuADDS Tutorial (uv)",
            "language": "python",
            "name": "squadds-tutorial",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
        "topocap_report": {
            "contract_version": "topocap-report-results-v1",
            "results_directory": os.fspath(results_dir),
            "empirical_placeholders": bool(_validate_result_directory(results_dir)),
        },
    }

    notebook["cells"] = [
        _markdown(
            r"""
            # Tutorial 18: Topology-General Transfer Learning for Capacitance

            This tutorial is an **artifact-driven research report** for TopoCap-EGRA
            (Evidence-Gated Residual Adaptation): a
            topology-aware model that reads GDS-derived conductor geometry, native layout
            controls, and an explicit electrical-net map, then predicts a physically valid
            Maxwell capacitance matrix of any supported size.

            The report has two jobs:

            1. explain and evaluate whether the model transfers from
               `GeneralizedCapNInterdigital` to a distinct `CapNInterdigitalTee` geometry;
            2. turn what we learn into a precise next-data collection brief for **Saikat Das**.

            Empirical panels are populated only from versioned result artifacts. If an artifact
            is absent, the notebook displays a labeled waiting panel. It never inserts synthetic
            scores or treats a conceptual diagram as experimental evidence.

            > **Evidence tier:** when the loaded manifest reports `EXPLORATORY_COMPLETE`, this is
            > a complete **exploratory** study. The protocol was developed while the public CapN
            > target release was available, so its intervals do not constitute a preregistered
            > confirmatory claim. A quick-smoke manifest suppresses empirical conclusions. The
            > decisive confirmation must use the new, untouched campaign specified in Part II.
            > Only ten held-out finger-count domains are independent resampling units here, so
            > alternate small-sample interval choices can be materially wider than the reported
            > domain bootstrap.
            """
        ),
        _markdown(
            r"""
            ## What you will learn

            By the end of the tutorial, you will be able to:

            - read a layout as an electrical conductor graph rather than a fixed bitmap;
            - understand how positive mutual and shunt terms reconstruct a physical Maxwell matrix;
            - distinguish test-row-leakage-resistant exploratory evidence from a random row split;
            - interpret learning curves, uncertainty diagnostics, ablations, and permutation tests;
            - decide when a diffusion model is warranted for inverse design; and
            - prepare a solver-ready, tool-neutral dataset that tests transfer across topology,
              conductor count, solver, process, institution, and layout tool.
            """
        ),
        _markdown(
            r"""
            ## 1. Load a compact, immutable result bundle

            The experiment runner writes one optional `manifest.json` and six compact tables.
            Parquet is preferred, while CSV, JSON Lines, and JSON are accepted for inspection.
            Set `SQUADDS_TOPOCAP_RESULTS` to relocate the bundle without rebuilding this notebook.

            | Logical artifact | Minimum columns | Purpose |
            |---|---|---|
            | `data_audit` | `dataset_family`, `records`, `node_count` | GDS/simulation coverage and topology sizes |
            | `learning_curves` | `method`, `support_size`, `metric`, `estimate`, `ci_low`, `ci_high` | Leakage-resistant target-label efficiency |
            | `uncertainty` | `method`, `curve`, `x`, `y` | Calibration and selective-risk behavior |
            | `ablations` | `variant`, `metric`, `estimate`, `ci_low`, `ci_high` | Which representation blocks matter |
            | `topology_checks` | `check`, `node_count`, `value`, `tolerance`, `passed` | Physical and permutation invariants |
            | `diffusion_decision` | `method`, `solver_budget`, `metric`, `estimate`, `ci_low`, `ci_high` | Solver-verified inverse-design gate |

            Confidence limits must be produced by the experiment pipeline. The report does not
            estimate a confidence interval from a single run.
            """
        ),
        _code(
            r"""
            import hashlib
            import json
            import os
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.io as pio
            from IPython.display import Markdown, display
            from plotly.subplots import make_subplots

            pio.renderers.default = "notebook_connected"
            pd.set_option("display.max_columns", 40)

            EMBEDDED_RESULTS_DIR = Path(__RESULTS_DIR__)
            RESULTS_DIR = Path(os.environ.get("SQUADDS_TOPOCAP_RESULTS", EMBEDDED_RESULTS_DIR)).expanduser()
            SUPPORTED_SUFFIXES = (".parquet", ".csv", ".jsonl", ".json")
            RESULT_SPECS = {
                "data_audit": {"dataset_family", "records", "node_count"},
                "learning_curves": {
                    "method", "support_size", "metric", "estimate", "ci_low", "ci_high"
                },
                "uncertainty": {"method", "curve", "x", "y"},
                "ablations": {"variant", "metric", "estimate", "ci_low", "ci_high"},
                "topology_checks": {"check", "node_count", "value", "tolerance", "passed"},
                "diffusion_decision": {
                    "method", "solver_budget", "metric", "estimate", "ci_low", "ci_high"
                },
            }

            manifest_path = RESULTS_DIR / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            except (json.JSONDecodeError, OSError) as error:
                manifest = {}
                print(f"Ignoring unreadable manifest: {error}")

            study_status = manifest.get("study_status", "MISSING")
            full_study_ready = study_status in {"COMPLETE", "EXPLORATORY_COMPLETE"}


            def _safe_artifact_path(logical_name):
                descriptor = manifest.get("artifacts", {}).get(logical_name)
                relative_path = None
                if isinstance(descriptor, str):
                    relative_path = descriptor
                elif isinstance(descriptor, dict):
                    relative_path = descriptor.get("path")
                if relative_path:
                    candidate = (RESULTS_DIR / relative_path).resolve()
                    root = RESULTS_DIR.resolve()
                    if candidate != root and root not in candidate.parents:
                        raise ValueError(f"Artifact path escapes result directory: {relative_path}")
                    return candidate if candidate.is_file() else None
                for suffix in SUPPORTED_SUFFIXES:
                    candidate = RESULTS_DIR / f"{logical_name}{suffix}"
                    if candidate.is_file():
                        return candidate
                return None


            def _read_table(path):
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                if path.suffix == ".csv":
                    return pd.read_csv(path)
                if path.suffix == ".jsonl":
                    return pd.read_json(path, lines=True)
                if path.suffix == ".json":
                    return pd.read_json(path)
                raise ValueError(f"Unsupported artifact type: {path.suffix}")


            def _sha256(path):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()


            tables = {}
            artifact_status = []
            for logical_name, required_columns in RESULT_SPECS.items():
                path = _safe_artifact_path(logical_name)
                status = "missing"
                detail = "waiting for experiment runner"
                frame = None
                if path is not None:
                    try:
                        frame = _read_table(path)
                        missing_columns = sorted(required_columns - set(frame.columns))
                        expected_hash = manifest.get("artifacts", {}).get(logical_name, {})
                        expected_hash = expected_hash.get("sha256") if isinstance(expected_hash, dict) else None
                        if expected_hash and _sha256(path) != expected_hash:
                            status = "invalid"
                            detail = "SHA-256 does not match manifest"
                            frame = None
                        elif missing_columns:
                            status = "invalid"
                            detail = "missing columns: " + ", ".join(missing_columns)
                            frame = None
                        else:
                            status = "ready"
                            detail = f"{len(frame):,} rows from {path.name}"
                    except Exception as error:
                        status = "invalid"
                        detail = f"{type(error).__name__}: {error}"
                        frame = None
                tables[logical_name] = frame
                artifact_status.append(
                    {"artifact": logical_name, "status": status, "detail": detail}
                )

            if not manifest:
                manifest_status = "missing"
                manifest_detail = "a checksummed manifest is required"
            elif not full_study_ready:
                manifest_status = "invalid"
                manifest_detail = f"non-publishable study status: {study_status}"
            else:
                manifest_status = "ready"
                manifest_detail = f"study_status={study_status}; state={manifest.get('state_digest', 'missing')[:12]}"
            artifact_status.append(
                {"artifact": "study_manifest", "status": manifest_status, "detail": manifest_detail}
            )

            artifact_status = pd.DataFrame(artifact_status)
            print(f"Result directory: {RESULTS_DIR}")
            print(f"Study status: {study_status}")
            artifact_status
            """.replace("__RESULTS_DIR__", embedded_results_dir)
        ),
        _code(
            r"""
            # %% hide input
            STATUS_COLORS = {"ready": "#2A9D8F", "missing": "#E9C46A", "invalid": "#D1495B"}
            status_counts = (
                artifact_status.groupby("status", as_index=False).size().sort_values("status")
            )
            fig = go.Figure(
                go.Bar(
                    x=status_counts["status"],
                    y=status_counts["size"],
                    marker_color=[STATUS_COLORS[value] for value in status_counts["status"]],
                    text=status_counts["size"],
                    textposition="outside",
                    customdata=[
                        "<br>".join(
                            artifact_status.loc[artifact_status["status"] == value, "artifact"]
                        )
                        for value in status_counts["status"]
                    ],
                    hovertemplate="<b>%{x}</b>: %{y}<br>%{customdata}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Research artifact readiness",
                xaxis_title="artifact status",
                yaxis_title="tables",
                template="plotly_white",
                height=420,
                showlegend=False,
            )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Failure audit: why EGRA has two experts

            The monolithic alternative is deliberately retained as an ablation. A rich GDS model
            can fit either family well, but cross-family pretraining can transfer solver-domain,
            ground-window, and layout-export correlations that are absent in the target family.
            Increasing the source row count does not by itself remove that acquisition shift.

            EGRA therefore does not assume that more source features are always safer. Its source
            path has low capacity and uses only canonical active controls plus electrical topology;
            its geometry path fits target-native active-region descriptors; and its evidence gate
            is allowed to reject source transfer. A successful result is not "transfer always wins."
            It is lower target-label cost when transfer is supported, followed by a safe handoff to
            the specialist when target evidence becomes stronger. The ablation intervals determine
            whether this behavior is repeatable rather than a story imposed on one split.
            """
        ),
        _markdown(
            r"""
            A green bar means the table exists, has the required columns, and matches its
            manifest hash when one is supplied. Yellow means the corresponding empirical panel
            will wait. Red means the bundle is present but should not be trusted until its schema
            or checksum is repaired.
            """
        ),
        _markdown(
            r"""
            ## 2. From layout files to a variable-size physical matrix

            A fixed image vector does not say which metal island is electrical node 0, which is
            node 1, or whether a new device has two, four, or eight conductors. TopoCap instead
            uses three tool-neutral inputs:

            - **GDS geometry:** polygons, layers, pairwise gaps, overlaps, boundary statistics,
              and local reference-ground context;
            - **electrical net sidecar:** immutable GDS selectors, node order, and reference roles;
            - **native layout controls:** named values tokenized by physical role while preserving
              the original dictionary for feedback into the source layout tool.

            Shared node and edge functions process any number of conductors. EBRA updates a compact
            source foundation with a small target support set, while a target-native active-GDS
            specialist becomes available as labels accumulate. EGRA compares those paths using only
            support-set cross-validation and refits the selected path on all support labels. The
            untouched target domain never participates in that decision.
            """
        ),
        _code(
            r"""
            # %% hide input
            labels = [
                "GDS polygons",
                "net sidecar",
                "native controls",
                "conductor graph",
                "topology-control view",
                "active-GDS view",
                "source foundation + EBRA",
                "target GDS specialist",
                "support-only evidence gate",
                "positive shunts s_i",
                "positive mutuals m_ij",
                "physical Maxwell C",
                "prediction + uncertainty",
            ]
            source = [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11]
            target = [3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12]
            values = [1] * len(source)
            fig = go.Figure(
                go.Sankey(
                    arrangement="snap",
                    node={
                        "label": labels,
                        "pad": 18,
                        "thickness": 18,
                        "color": [
                            "#264653", "#264653", "#264653", "#2A9D8F", "#E9C46A",
                            "#F4A261", "#6A4C93", "#D1495B", "#00798C", "#F4A261",
                            "#F4A261", "#D1495B", "#00798C",
                        ],
                    },
                    link={"source": source, "target": target, "value": values, "color": "rgba(0,121,140,0.18)"},
                )
            )
            fig.update_layout(
                title="TopoCap-EGRA information flow (schematic, not a performance result)",
                template="plotly_white",
                height=620,
            )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Why the output is physical by construction

            For each unordered conductor pair, the model predicts a positive mutual magnitude
            $m_{ij}$. For each conductor, it predicts a positive shunt $s_i$. The signed Maxwell
            matrix is reconstructed as

            $$C_{ij}=-m_{ij}\quad(i\ne j), \qquad
            C_{ii}=s_i+\sum_{j\ne i}m_{ij}.$$

            This makes $C$ symmetric, gives non-positive off-diagonal entries, positive diagonal
            entries, and diagonal dominance for any conductor count $N$. It also makes conductor
            relabeling testable: permuting the graph nodes must only permute rows and columns of
            the prediction.
            """
        ),
        _code(
            r"""
            from squadds.ml import components_to_maxwell, maxwell_diagnostics

            # Three conductor nodes have three shunts and three unordered pair interactions.
            example_shunts_ff = [12.0, 1.8, 2.1]
            example_mutuals_ff = [3.2, 2.7, 0.9]  # pairs (0,1), (0,2), (1,2)
            example_matrix_ff = components_to_maxwell(example_shunts_ff, example_mutuals_ff)
            example_diagnostics = maxwell_diagnostics(example_matrix_ff)

            display(pd.DataFrame(example_matrix_ff).round(3).style.set_caption("Example signed Maxwell matrix (fF)"))
            print(f"Physical by construction: {example_diagnostics.is_physical}")
            """
        ),
        _code(
            r"""
            # %% hide input
            maximum_nodes = 8
            labels = [f"net {index}" for index in range(maximum_nodes)]


            def matrix_structure(node_count):
                z = np.full((maximum_nodes, maximum_nodes), np.nan)
                text = np.full((maximum_nodes, maximum_nodes), "", dtype=object)
                for row in range(node_count):
                    for column in range(node_count):
                        if row == column:
                            z[row, column] = 2
                            text[row, column] = "s_i + sum m_ik"
                        else:
                            z[row, column] = 1
                            text[row, column] = "-m_ij"
                return z, text


            initial_z, initial_text = matrix_structure(3)
            frames = []
            for node_count in range(2, maximum_nodes + 1):
                z, text_values = matrix_structure(node_count)
                frames.append(
                    go.Frame(
                        name=str(node_count),
                        data=[go.Heatmap(z=z, text=text_values, texttemplate="%{text}")],
                        layout={"title": f"Symbolic {node_count} x {node_count} Maxwell structure"},
                    )
                )

            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=initial_z,
                        x=labels,
                        y=labels,
                        text=initial_text,
                        texttemplate="%{text}",
                        colorscale=[[0, "#F4A261"], [0.5, "#F4A261"], [0.5, "#2A9D8F"], [1, "#2A9D8F"]],
                        showscale=False,
                        hovertemplate="row=%{y}<br>column=%{x}<br>%{text}<extra></extra>",
                    )
                ],
                frames=frames,
            )
            fig.update_layout(
                title="Symbolic 3 x 3 Maxwell structure",
                template="plotly_white",
                height=650,
                xaxis={"side": "top"},
                yaxis={"autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
                sliders=[
                    {
                        "active": 1,
                        "currentvalue": {"prefix": "conductor count N = "},
                        "pad": {"t": 45},
                        "steps": [
                            {
                                "label": str(node_count),
                                "method": "animate",
                                "args": [
                                    [str(node_count)],
                                    {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                                ],
                            }
                            for node_count in range(2, maximum_nodes + 1)
                        ],
                    }
                ],
                annotations=[
                    {
                        "text": "Structure only: colors and symbols are not capacitance values.",
                        "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.16,
                        "showarrow": False,
                    }
                ],
            )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ## 3. Data audit before model evaluation

            A trustworthy transfer claim starts with data lineage. The audit should account for
            every simulation row, immutable GDS hash, net sidecar, matrix unit, conductor count,
            campaign, and split group. Counts alone are not enough: near-duplicate morphology
            profiles must stay in one split, and preprocessing statistics must be fit only on the
            corresponding training rows.
            """
        ),
        _code(
            r"""
            # %% hide input
            audit = tables["data_audit"]
            if audit is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        "<b>Waiting for data_audit</b><br>"
                        "Write dataset_family, records, and node_count; optional columns include "
                        "campaign, gds_files, matrix_unit, and split_group."
                    ),
                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, align="center",
                )
                fig.update_layout(title="Dataset and topology audit", template="plotly_white", height=480)
            else:
                audit = audit.copy()
                audit["campaign"] = audit.get("campaign", pd.Series("all", index=audit.index)).fillna("all")
                audit["series"] = audit["dataset_family"].astype(str) + " / " + audit["campaign"].astype(str)
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Simulation records by source", "Conductor-count coverage"),
                    horizontal_spacing=0.14,
                )
                for series_name, frame in audit.groupby("series", sort=True):
                    record_count = frame["records"].sum()
                    unique_gds = frame.get("unique_gds", frame["records"]).sum()
                    overlap_count = frame.get(
                        "cross_family_identity_overlap_total", pd.Series([np.nan])
                    ).max()
                    matrix_units = ", ".join(sorted(frame.get("matrix_unit", pd.Series(["unknown"])).astype(str).unique()))
                    fig.add_trace(
                        go.Bar(
                            x=[series_name], y=[record_count], name=series_name,
                            text=[f"{record_count:,.0f}"], textposition="outside",
                            customdata=[[unique_gds, overlap_count, matrix_units]],
                            hovertemplate=(
                                "%{x}<br>records=%{y:,.0f}<br>unique GDS=%{customdata[0]:,.0f}"
                                "<br>cross-family identity overlaps=%{customdata[1]}"
                                "<br>matrix unit=%{customdata[2]}<extra></extra>"
                            ),
                        ),
                        row=1, col=1,
                    )
                    topology = frame.groupby("node_count", as_index=False)["records"].sum()
                    fig.add_trace(
                        go.Scatter(
                            x=topology["node_count"], y=topology["records"],
                            mode="lines+markers", name=series_name, legendgroup=series_name,
                            showlegend=False,
                            hovertemplate="N=%{x}<br>records=%{y:,.0f}<extra></extra>",
                        ),
                        row=1, col=2,
                    )
                fig.update_xaxes(title_text="dataset / campaign", tickangle=-25, row=1, col=1)
                fig.update_yaxes(title_text="records", row=1, col=1)
                fig.update_xaxes(title_text="conductor count N", dtick=1, row=1, col=2)
                fig.update_yaxes(title_text="records", row=1, col=2)
                fig.update_layout(
                    title="Dataset and topology audit (loaded artifacts only)",
                    template="plotly_white", height=570, barmode="group",
                    legend={"orientation": "h", "y": -0.28, "x": 0.5, "xanchor": "center"},
                    margin={"b": 150},
                )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Leakage-resistant evaluation protocol

            The primary study must keep the two component families distinct:

            1. pretrain the foundation model on `GeneralizedCapNInterdigital` only;
            2. form outer `CapNInterdigitalTee` test domains by held-out morphology or finger count;
            3. reserve a different validation domain and keep model settings fixed before test evaluation;
            4. draw nested target support sets for each label budget and repeat seed;
            5. compare zero-shot, target-control and target-GDS specialists, EBRA transfer through
               topology/control, support-conditioned source retrieval, active-GDS,
               all-cached-descriptor, and v0 views, plus exactly matched shuffled-source controls,
               on the same test rows;
            6. report cluster-bootstrap intervals, paired differences, negative-transfer frequency,
               physical violations, and uncertainty coverage.

            Randomly splitting individual rows is not evidence of cross-family generalization when
            nearby sweep points or repeated geometry profiles can land on both sides of the split.

            **Reproducibility contract.** Source foundations are fit once. Every outer-domain,
            repeat, and support-size trial is written atomically and can be resumed. The study
            fingerprint covers the graph-cache manifest and bytes, v0 bytes and projection settings
            when enabled, model/adaptation configurations, the runner, and every TopoCap module.
            Changing any of those inputs requires a new result directory instead of silently mixing
            stale checkpoints with a new experiment.

            **Current alignment limitation.** The legacy CapN GDS files do not carry port markers;
            their two signal polygons are aligned to matrix rows through the recorded generator
            order contract. Hashes make that assumption reproducible, but they do not independently
            prove its electrical semantics. The next schema pilot must include an asymmetric golden
            design with explicit net markers and verified matrix-row labels before this result is
            used as confirmatory evidence.
            """
        ),
        _markdown(
            r"""
            ## 4. Learning curves: does source knowledge save target simulations?

            Each point must summarize repeated, paired trials at one target support size. The
            decisive comparison is each transfer or evidence-gated path versus target-only at the
            same support rows. The `paired_gain_vs_target_control` view is signed so that positive
            values favor the evaluated method and intervals crossing zero are inconclusive.
            Error bars are loaded from the experiment bundle; they are never reconstructed from a
            single point estimate in this notebook.
            """
        ),
        _code(
            r"""
            # %% hide input
            method_ledger = pd.DataFrame(
                [
                    ["source_control_foundation", "canonical controls + net roles", "GeneralizedNCap", "none", "zero-shot relevance"],
                    ["source_control_ebra", "canonical controls + net roles", "GeneralizedNCap", "EBRA", "compact transfer"],
                    ["source_retrieval_2048_ebra", "support-matched canonical controls", "2,048 retrieved GeneralizedNCap labels", "EBRA", "retrieval transfer"],
                    ["shuffled_source_retrieval_2048_ebra", "identical retrieved source geometries", "same 2,048 labels shuffled", "EBRA", "matched retrieval control"],
                    ["source_active_gds_ebra", "active-region GDS graph", "GeneralizedNCap", "EBRA", "geometry transfer"],
                    ["source_all_cached_descriptors_ebra", "all cached cropped descriptors", "GeneralizedNCap", "EBRA", "rich but shift-prone transfer"],
                    ["v0_gaussian64_ebra", "fixed sketch of all 9,227 v0 values", "GeneralizedNCap", "EBRA", "static-embedding baseline"],
                    ["shuffled_v0_gaussian64_ebra", "same fixed v0 sketch; shuffled labels", "negative control", "EBRA", "v0 regularization control"],
                    ["target_v0_gaussian64_scratch", "same fixed source-defined v0 sketch", "none", "target support", "v0 same-budget baseline"],
                    ["shuffled_source_ebra", "canonical controls; shuffled labels", "negative control", "EBRA", "regularization control"],
                    ["target_control_scratch", "canonical controls + net roles", "none", "target support", "same-budget baseline"],
                    ["target_active_gds_specialist", "active-region GDS graph", "none", "target support", "target-native geometry expert"],
                    ["evidence_gated_topocap", "source control transfer or target GDS", "GeneralizedNCap", "support-only CV", "reject unsafe transfer"],
                ],
                columns=["method", "model input", "source labels", "target-label use", "scientific role"],
            )
            fig = go.Figure(
                go.Table(
                    columnwidth=[0.22, 0.25, 0.16, 0.15, 0.22],
                    header={
                        "values": list(method_ledger.columns),
                        "fill_color": "#264653", "font": {"color": "white", "size": 12},
                        "align": "left", "height": 34,
                    },
                    cells={
                        "values": [method_ledger[column] for column in method_ledger],
                        "fill_color": [
                            ["#F7FAFA" if index % 2 == 0 else "#EDF6F5" for index in range(len(method_ledger))]
                        ],
                        "align": "left", "height": 30,
                    },
                )
            )
            fig.update_layout(title="What each learning-curve trace actually tests", height=590, margin={"l": 15, "r": 15})
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Metric sign conventions

            - `macro_component_log_mae`: mean absolute error after the physical shunt/mutual
              components are mapped to log space; **lower is better** and each test design has
              equal macro weight.
            - `macro_relative_frobenius`: per-design matrix error normalized by the true matrix
              Frobenius norm; **lower is better**.
            - `physical_valid_rate`: fraction satisfying the signed-Maxwell checks; **higher is
              better**, and TopoCap should achieve one by construction.
            - `paired_gain_vs_target_control`: target-control error minus evaluated-method error
              on identical test designs; **positive favors the evaluated method**, negative means
              transfer hurt, and an interval crossing zero is inconclusive.
            - `paired_gain_vs_shuffled_source`: shuffled-source error minus correctly labeled-source
              error on identical test designs; **positive is evidence that source labels carry
              useful cross-family information beyond generic prior regularization**.

            The support-conditioned retriever is frozen before the primary rerun. At every
            positive target-label budget it selects exactly 2,048 source designs using only the
            target **support** set's canonical active length, width, and gap. Distances use a
            source-only median/IQR normalization, and the source subset is balanced across source
            finger counts. The matched negative control keeps those exact source geometries and
            shuffles only their capacitance labels. Target test geometry and labels never enter
            retrieval. The selected source IDs and their digest are checkpointed per trial.

            The v0 baseline source-standardizes all 9,227 coordinates before applying a
            deterministic Gaussian sketch to 64 dimensions, then source-standardizes the sketch.
            Every original coordinate contributes without fitting a target-aware projection; no
            target row enters either normalization pass. It is a compute-controlled compressed
            baseline, not the full 9,227-dimensional v0 model, and the report makes no unmeasured
            distance-preservation claim for the sketch.
            """
        ),
        _markdown(
            r"""
            ### Inspect the frozen retrieval API

            `SupportConditionedSourceRetriever` is public through `squadds.ml`. Its protocol can
            be inspected without loading a dataset. Calling `retrieve(support_graphs)` then returns
            immutable source indices, ordered source IDs, and a digest. An empty support set raises
            instead of silently falling back to a source-only model.
            """
        ),
        _code(
            r"""
            from squadds.ml import SupportConditionedSourceRetriever

            retrieval_protocol = pd.Series(
                SupportConditionedSourceRetriever.protocol(), name="frozen value"
            ).rename_axis("protocol field").to_frame()
            retrieval_protocol
            """
        ),
        _code(
            r"""
            # %% hide input
            curves = tables["learning_curves"]
            if curves is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        "<b>Waiting for learning_curves</b><br>"
                        "One row per method, support size, metric, and validated confidence interval."
                    ),
                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                )
                fig.update_layout(title="Cross-family target-label efficiency", template="plotly_white", height=550)
            else:
                curves = curves.copy().sort_values(["metric", "method", "support_size"])
                curves["target_domain"] = curves.get(
                    "target_domain", pd.Series("all target domains", index=curves.index)
                ).fillna("all target domains")
                target_domains = curves["target_domain"].astype(str).unique()
                if len(target_domains) == 1 and target_domains[0].startswith("macro across"):
                    curves["series"] = curves["method"].astype(str)
                else:
                    curves["series"] = (
                        curves["method"].astype(str) + " | " + curves["target_domain"].astype(str)
                    )
                metrics = sorted(curves["metric"].astype(str).unique())
                series_names = sorted(curves["series"].astype(str).unique())
                palette = px.colors.qualitative.Safe + px.colors.qualitative.Bold
                color_map = {name: palette[index % len(palette)] for index, name in enumerate(series_names)}
                fig = go.Figure()
                trace_metrics = []
                for metric in metrics:
                    metric_frame = curves.loc[curves["metric"].astype(str) == metric]
                    for series_name, frame in metric_frame.groupby("series", sort=True):
                        frame = frame.sort_values("support_size")
                        fig.add_trace(
                            go.Scatter(
                                x=frame["support_size"], y=frame["estimate"],
                                mode="lines+markers", name=series_name,
                                visible=metric == metrics[0],
                                line={"width": 3, "color": color_map[series_name]},
                                marker={"size": 7},
                                error_y={
                                    "type": "data",
                                    "array": np.maximum(frame["ci_high"] - frame["estimate"], 0),
                                    "arrayminus": np.maximum(frame["estimate"] - frame["ci_low"], 0),
                                    "visible": True,
                                },
                                customdata=np.column_stack([frame["ci_low"], frame["ci_high"]]),
                                hovertemplate=(
                                    f"<b>{series_name}</b><br>support=%{{x}}<br>{metric}=%{{y:.4g}}"
                                    "<br>interval=[%{customdata[0]:.4g}, %{customdata[1]:.4g}]<extra></extra>"
                                ),
                            )
                        )
                        trace_metrics.append(metric)
                buttons = []
                for metric in metrics:
                    buttons.append(
                        {
                            "label": metric,
                            "method": "update",
                            "args": [
                                {"visible": [trace_metric == metric for trace_metric in trace_metrics]},
                                {"title": f"Cross-family target-label efficiency: {metric}", "yaxis.title": metric},
                            ],
                        }
                    )
                fig.update_layout(
                    title=f"Cross-family target-label efficiency: {metrics[0]}",
                    xaxis_title="labeled target support designs",
                    yaxis_title=metrics[0],
                    template="plotly_white", height=720, hovermode="x unified",
                    updatemenus=[
                        {
                            "buttons": buttons, "x": 0.98, "y": 1.12,
                            "xanchor": "right", "yanchor": "top",
                        }
                    ],
                    legend={
                        "orientation": "v", "y": 1.0, "x": 1.02,
                        "xanchor": "left", "yanchor": "top", "font": {"size": 10},
                    },
                    margin={"b": 90, "r": 330},
                )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            Read the curves in this order:

            1. **zero-shot:** is source knowledge relevant before target labels are seen?
            2. **target-only versus transfer:** is the paired transfer gain positive at scarce-label budgets?
            3. **confidence intervals and negative-transfer rate:** is that gain repeatable across held-out domains?
            4. **specialist convergence:** does the evidence gate hand off to a target-native model rather than remain source-biased?
            5. **negative controls:** do shuffled-source and compressed-v0 baselines rule out a generic regularization story?

            A single attractive curve is exploratory. A claim requires repeated splits, paired
            uncertainty, and stable behavior across target domains.
            """
        ),
        _markdown(
            r"""
            ## 5. Uncertainty diagnostics: does the model know when transfer is unsafe?

            EBRA returns marginal component uncertainty, not only a point prediction. The current
            implementation propagates componentwise bounds into matrix entries; because diagonal
            entries sum correlated log-normal components, these are **not guaranteed joint nominal
            matrix intervals**. Treat both panels as descriptive diagnostics until a held-out-domain
            conformal or joint-posterior calibration layer is available:

            - a **coverage curve** compares requested marginal-envelope coverage with observed coverage;
            - a **risk-coverage curve** rejects the most uncertain predictions first and checks
              whether retained error decreases.

            A useful transfer model should be accurate when it is confident and cautious under
            topology, process, or geometry shift.
            """
        ),
        _code(
            r"""
            # %% hide input
            uncertainty = tables["uncertainty"]
            if uncertainty is None:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Calibration", "Risk coverage"))
                fig.add_annotation(
                    text="Waiting for uncertainty: curve=calibration",
                    x=0.22, y=0.5, xref="paper", yref="paper", showarrow=False,
                )
                fig.add_annotation(
                    text="Waiting for uncertainty: curve=risk_coverage",
                    x=0.79, y=0.5, xref="paper", yref="paper", showarrow=False,
                )
                fig.update_layout(title="Predictive uncertainty diagnostics", template="plotly_white", height=520)
            else:
                uncertainty = uncertainty.copy()
                uncertainty["curve"] = uncertainty["curve"].astype(str).str.lower()
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Marginal-envelope coverage", "Selective risk versus retained coverage"),
                    horizontal_spacing=0.13,
                )
                methods = sorted(uncertainty["method"].astype(str).unique())
                palette = px.colors.qualitative.Safe
                colors = {method: palette[index % len(palette)] for index, method in enumerate(methods)}
                calibration = uncertainty.loc[uncertainty["curve"] == "calibration"]
                risk = uncertainty.loc[uncertainty["curve"].isin(["risk_coverage", "risk-coverage"])]
                if not calibration.empty:
                    extent = [float(calibration["x"].min()), float(calibration["x"].max())]
                    fig.add_trace(
                        go.Scatter(
                            x=extent, y=extent, mode="lines", name="identity reference",
                            line={"dash": "dash", "color": "#777777"}, legendgroup="reference",
                        ),
                        row=1, col=1,
                    )
                for method in methods:
                    frame = calibration.loc[calibration["method"].astype(str) == method].sort_values("x")
                    if not frame.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=frame["x"], y=frame["y"], mode="lines+markers", name=method,
                                line={"color": colors[method], "width": 3}, legendgroup=method,
                                hovertemplate="requested=%{x:.3f}<br>observed=%{y:.3f}<extra></extra>",
                            ),
                            row=1, col=1,
                        )
                    frame = risk.loc[risk["method"].astype(str) == method].sort_values("x")
                    if not frame.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=frame["x"], y=frame["y"], mode="lines+markers", name=method,
                                showlegend=False, line={"color": colors[method], "width": 3}, legendgroup=method,
                                hovertemplate="retained fraction=%{x:.3f}<br>selective risk=%{y:.4g}<extra></extra>",
                            ),
                            row=1, col=2,
                        )
                fig.update_xaxes(title_text="requested marginal-envelope coverage", row=1, col=1)
                fig.update_yaxes(title_text="observed entry coverage", row=1, col=1)
                fig.update_xaxes(title_text="retained prediction fraction", row=1, col=2)
                fig.update_yaxes(title_text="error among retained predictions", row=1, col=2)
                fig.update_layout(
                    title="Predictive uncertainty diagnostics (loaded artifacts only)",
                    template="plotly_white", height=680,
                    legend={
                        "orientation": "v", "y": 1.0, "x": 1.02,
                        "xanchor": "left", "yanchor": "top", "font": {"size": 10},
                    },
                    margin={"b": 90, "r": 350},
                )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ## 6. Ablations: what actually carries transfer?

            This study compares representation paths rather than pretending to isolate every raw
            descriptor independently: canonical topology/control, active-region GDS, all cached
            cropped descriptors, compressed v0, target-native specialists, and an evidence-gated combination.
            A shuffled-source control tests whether an apparent benefit is merely regularization
            rather than source-target knowledge transfer.
            """
        ),
        _code(
            r"""
            # %% hide input
            ablations = tables["ablations"]
            if ablations is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        "<b>Waiting for ablations</b><br>"
                        "Store one validated estimate and interval per variant and metric."
                    ),
                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                )
                fig.update_layout(title="Representation and transfer ablations", template="plotly_white", height=540)
            else:
                ablations = ablations.copy().sort_values(["metric", "estimate", "variant"])
                metrics = sorted(ablations["metric"].astype(str).unique())
                fig = go.Figure()
                trace_metrics = []
                for metric in metrics:
                    frame = ablations.loc[ablations["metric"].astype(str) == metric]
                    fig.add_trace(
                        go.Bar(
                            x=frame["estimate"], y=frame["variant"], orientation="h",
                            visible=metric == metrics[0], name=metric,
                            marker_color="#00798C",
                            error_x={
                                "type": "data",
                                "array": np.maximum(frame["ci_high"] - frame["estimate"], 0),
                                "arrayminus": np.maximum(frame["estimate"] - frame["ci_low"], 0),
                            },
                            customdata=np.column_stack([frame["ci_low"], frame["ci_high"]]),
                            hovertemplate=(
                                "<b>%{y}</b><br>estimate=%{x:.4g}"
                                "<br>interval=[%{customdata[0]:.4g}, %{customdata[1]:.4g}]<extra></extra>"
                            ),
                        )
                    )
                    trace_metrics.append(metric)
                buttons = [
                    {
                        "label": metric,
                        "method": "update",
                        "args": [
                            {"visible": [trace_metric == metric for trace_metric in trace_metrics]},
                            {"title": f"Representation ablations: {metric}", "xaxis.title": metric},
                        ],
                    }
                    for metric in metrics
                ]
                fig.update_layout(
                    title=f"Representation ablations: {metrics[0]}", xaxis_title=metrics[0],
                    yaxis_title="model variant", template="plotly_white", height=690,
                    updatemenus=[
                        {
                            "buttons": buttons, "x": 0.98, "y": 1.12,
                            "xanchor": "right", "yanchor": "top",
                        }
                    ],
                    margin={"l": 330}, showlegend=False,
                )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ## 7. Topology and permutation checks

            Empirical source and target datasets may both happen to use three conductors. That
            does **not** empirically prove arbitrary-size generalization. The architecture can be
            tested at $N=2,\ldots,N_{max}$ with synthetic graph structure, but a scientific claim
            across matrix sizes requires new simulated layouts with those conductor counts.

            The checks below should include at least permutation equivariance, symmetry,
            off-diagonal sign, positive diagonal, diagonal dominance, positive semidefiniteness,
            checkpoint round-trip, and matrix reconstruction error.
            """
        ),
        _code(
            r"""
            # %% hide input
            checks = tables["topology_checks"]
            if checks is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        "<b>Waiting for topology_checks</b><br>"
                        "Architecture tests are not a substitute for variable-N simulation data."
                    ),
                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                )
                fig.update_layout(title="Physical and permutation invariants", template="plotly_white", height=540)
            else:
                checks = checks.copy()
                positive_values = np.abs(checks["value"].to_numpy(dtype=float))
                positive_values = positive_values[positive_values > 0]
                floor = positive_values.min() / 10 if len(positive_values) else 1e-18
                checks["display_value"] = np.maximum(np.abs(checks["value"].astype(float)), floor)
                checks["display_tolerance"] = np.maximum(np.abs(checks["tolerance"].astype(float)), floor)
                checks["status"] = np.where(checks["passed"].astype(bool), "pass", "fail")
                fig = px.scatter(
                    checks,
                    x="node_count", y="display_value", color="check", symbol="status",
                    symbol_map={"pass": "circle", "fail": "x"},
                    hover_data={
                        "value": ":.4g", "tolerance": ":.4g", "display_value": False,
                        "node_count": True, "status": True,
                    },
                    log_y=True,
                    title="Physical and permutation invariant residuals",
                )
                for check_name, frame in checks.groupby("check", sort=True):
                    frame = frame.sort_values("node_count")
                    fig.add_trace(
                        go.Scatter(
                            x=frame["node_count"], y=frame["display_tolerance"],
                            mode="lines", name=f"{check_name} tolerance",
                            line={"dash": "dot", "width": 1}, opacity=0.55,
                            hovertemplate="N=%{x}<br>tolerance=%{y:.4g}<extra></extra>",
                        )
                    )
                fig.update_layout(
                    xaxis={"title": "conductor count N", "dtick": 1},
                    yaxis_title="absolute residual (log scale)",
                    template="plotly_white", height=650,
                    legend={"orientation": "h", "y": -0.3, "x": 0.5, "xanchor": "center"},
                    margin={"b": 170},
                )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ## 8. Should inverse design use diffusion?

            Diffusion is relevant because many geometries can realize a similar capacitance matrix.
            It is **not** the first choice for the forward surrogate. A disciplined sequence is:

            1. train and calibrate the topology-general forward model;
            2. establish retrieval-seeded, uncertainty-aware trust-region optimization as a strong
               non-generative inverse-design baseline;
            3. train a net-conditioned latent geometry diffusion model only when the dataset contains
               enough topology diversity and valid alternatives for the same target;
            4. decode through tool-specific layout adapters, then apply DRC and true-solver validation;
            5. accept diffusion only if it improves true-solver success or valid diversity at the
               same solver budget without sacrificing calibration or manufacturability.

            Raw bitmap diffusion is not recommended. The generative state should be a net-aware
            geometry representation such as per-net signed-distance fields or conductor-graph plus
            boundary latent, conditioned on matrix structure and process metadata.

            This proposal combines three established ideas rather than claiming a validated
            quantum-layout diffusion result: denoising diffusion models
            ([Ho, Jain, and Abbeel, 2020](https://arxiv.org/abs/2006.11239)), compressed latent-space
            generation ([Rombach et al., 2021](https://arxiv.org/abs/2112.10752)), and
            permutation-aware discrete graph diffusion
            ([Vignac et al., 2022](https://arxiv.org/abs/2209.14734)). Conditional diffusion has also
            been explored for electromagnetic inverse design in metasurfaces
            ([Hen et al., 2025](https://arxiv.org/abs/2506.21748)), but that does not establish
            effectiveness for superconducting-device GDS. The equal-solver-budget benchmark below
            is therefore a required future experiment, not a formality.
            """
        ),
        _code(
            r"""
            # %% hide input
            labels = [
                "target C + topology",
                "retrieval seeds",
                "trust-region baseline",
                "latent diffusion candidates",
                "layout-tool adapter",
                "DRC",
                "calibrated forward ranker",
                "true EM solver",
                "accept / active-learn",
            ]
            fig = go.Figure(
                go.Sankey(
                    node={
                        "label": labels, "pad": 18, "thickness": 18,
                        "color": ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#6A4C93", "#D1495B", "#00798C", "#264653", "#2A9D8F"],
                    },
                    link={
                        "source": [0, 1, 0, 3, 2, 4, 5, 6, 7],
                        "target": [1, 2, 3, 4, 5, 5, 6, 7, 8],
                        "value": [1] * 9,
                        "color": "rgba(106,76,147,0.18)",
                    },
                )
            )
            fig.update_layout(
                title="Equal-budget inverse-design decision path (schematic)",
                template="plotly_white", height=560,
            )
            fig.show()
            """,
            hidden=True,
        ),
        _code(
            r"""
            # %% hide input
            diffusion = tables["diffusion_decision"]
            if diffusion is None:
                fig = go.Figure()
                fig.add_annotation(
                    text=(
                        "<b>Waiting for diffusion_decision</b><br>"
                        "Do not claim a generative advantage before equal-budget true-solver evaluation."
                    ),
                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                )
                fig.update_layout(title="Diffusion go/no-go benchmark", template="plotly_white", height=540)
            else:
                diffusion = diffusion.copy()
                diffusion["estimate"] = pd.to_numeric(diffusion["estimate"], errors="coerce")
                if not np.isfinite(diffusion["estimate"]).any():
                    status = ", ".join(sorted(diffusion.get("status", pd.Series(["deferred"])).astype(str).unique()))
                    reason = diffusion.get(
                        "reason",
                        pd.Series(["Variable-topology and one-to-many inverse-design data have not been collected."]),
                    ).iloc[0]
                    fig = go.Figure()
                    fig.add_annotation(
                        text=(
                            f"<b>Diffusion benchmark {status}</b><br>{reason}<br>"
                            "The notebook records a no-go decision rather than fabricating a score."
                        ),
                        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, align="center",
                    )
                    fig.update_layout(title="Diffusion go/no-go benchmark", template="plotly_white", height=540)
                else:
                    diffusion = diffusion.sort_values(["metric", "method", "solver_budget"])
                    metrics = sorted(diffusion["metric"].astype(str).unique())
                    fig = go.Figure()
                    trace_metrics = []
                    for metric in metrics:
                        frame_for_metric = diffusion.loc[diffusion["metric"].astype(str) == metric]
                        for method, frame in frame_for_metric.groupby("method", sort=True):
                            frame = frame.sort_values("solver_budget")
                            fig.add_trace(
                                go.Scatter(
                                    x=frame["solver_budget"], y=frame["estimate"],
                                    mode="lines+markers", name=str(method), visible=metric == metrics[0],
                                    error_y={
                                        "type": "data",
                                        "array": np.maximum(frame["ci_high"] - frame["estimate"], 0),
                                        "arrayminus": np.maximum(frame["estimate"] - frame["ci_low"], 0),
                                    },
                                    hovertemplate=(
                                        f"<b>{method}</b><br>solver budget=%{{x}}<br>"
                                        f"{metric}=%{{y:.4g}}<extra></extra>"
                                    ),
                                )
                            )
                            trace_metrics.append(metric)
                    buttons = [
                        {
                            "label": metric,
                            "method": "update",
                            "args": [
                                {"visible": [trace_metric == metric for trace_metric in trace_metrics]},
                                {"title": f"Equal-budget inverse design: {metric}", "yaxis.title": metric},
                            ],
                        }
                        for metric in metrics
                    ]
                    fig.update_layout(
                        title=f"Equal-budget inverse design: {metrics[0]}",
                        xaxis_title="true-solver evaluations", yaxis_title=metrics[0],
                        template="plotly_white", height=610,
                        updatemenus=[
                            {
                                "buttons": buttons, "x": 0.98, "y": 1.12,
                                "xanchor": "right", "yanchor": "top",
                            }
                        ],
                        legend={"orientation": "h", "y": -0.22, "x": 0.5, "xanchor": "center"},
                        margin={"b": 125},
                    )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ## 9. Evidence summary

            The experiment manifest may include a `conclusions` list, but every item should point
            to a metric, interval, held-out domain, and artifact. The notebook deliberately does
            not derive a headline claim from partial files.
            """
        ),
        _code(
            r"""
            conclusions = manifest.get("conclusions", [])
            if conclusions and full_study_ready:
                lines = ["### Manifest-backed conclusions"]
                for item in conclusions:
                    if isinstance(item, dict):
                        statement = item.get("statement", "Unlabeled conclusion")
                        evidence = item.get("evidence", "evidence reference not supplied")
                        lines.append(f"- **{statement}**  \n  Evidence: `{evidence}`")
                    else:
                        lines.append(f"- {item}")
                display(Markdown("\n".join(lines)))
            else:
                display(
                    Markdown(
                        "> **No publishable empirical conclusion is available.** A full checksummed "
                        "study bundle is required; quick-smoke artifacts are intentionally suppressed."
                    )
                )
            """
        ),
        _markdown(
            r"""
            ### Mechanism audit: improvement is not automatically transfer

            A system-level gain can come from a source foundation, a better target representation,
            regularization, or the evidence gate selecting between experts. The signed comparisons
            below separate those mechanisms. Positive values favor the named method; an interval
            crossing zero is inconclusive. The source-label comparisons are especially important:
            they ask whether correctly paired source labels add information beyond the exact same
            pipeline trained on shuffled source labels.
            """
        ),
        _code(
            r"""
            # %% hide input
            learning = tables["learning_curves"]
            primary_support = int(manifest.get("scientific_scope", {}).get("primary_support_size", 16))
            mechanism_specs = [
                (
                    "source_retrieval_2048_ebra",
                    "paired_gain_vs_shuffled_source",
                    "retrieved source labels - matched shuffled labels",
                ),
                (
                    "source_retrieval_2048_ebra",
                    "paired_gain_vs_target_control",
                    "retrieved-source transfer - target parameter control",
                ),
                (
                    "evidence_gated_topocap",
                    "paired_gain_vs_target_control",
                    "EGRA - target parameter control",
                ),
                (
                    "target_active_gds_specialist",
                    "paired_gain_vs_target_control",
                    "target active-GDS - target parameter control",
                ),
                (
                    "source_active_gds_ebra",
                    "paired_gain_vs_target_control",
                    "source active-GDS transfer - target parameter control",
                ),
                (
                    "source_control_ebra",
                    "paired_gain_vs_shuffled_source",
                    "correct control-source labels - shuffled labels",
                ),
                (
                    "v0_gaussian64_ebra",
                    "paired_gain_vs_shuffled_source",
                    "correct v0-source labels - shuffled labels",
                ),
            ]
            mechanism_rows = []
            if learning is not None and full_study_ready:
                for method, metric, label in mechanism_specs:
                    frame = learning.loc[
                        (learning["method"].astype(str) == method)
                        & (learning["metric"].astype(str) == metric)
                        & (pd.to_numeric(learning["support_size"], errors="coerce") == primary_support)
                    ]
                    if not frame.empty:
                        row = frame.iloc[0]
                        mechanism_rows.append(
                            {
                                "comparison": label,
                                "estimate": float(row["estimate"]),
                                "ci_low": float(row["ci_low"]),
                                "ci_high": float(row["ci_high"]),
                            }
                        )
            mechanism = pd.DataFrame(mechanism_rows)
            if mechanism.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="Waiting for a complete mechanism-audit bundle",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
                fig.update_layout(title="What carries the K=16 gain?", template="plotly_white", height=520)
            else:
                colors = [
                    "#2A9D8F" if low > 0 else "#D1495B" if high < 0 else "#E9C46A"
                    for low, high in zip(mechanism["ci_low"], mechanism["ci_high"], strict=True)
                ]
                fig = go.Figure(
                    go.Scatter(
                        x=mechanism["estimate"],
                        y=mechanism["comparison"],
                        mode="markers",
                        marker={"size": 13, "color": colors, "line": {"color": "#264653", "width": 1}},
                        error_x={
                            "type": "data",
                            "array": mechanism["ci_high"] - mechanism["estimate"],
                            "arrayminus": mechanism["estimate"] - mechanism["ci_low"],
                            "thickness": 2,
                            "width": 7,
                        },
                        customdata=np.column_stack([mechanism["ci_low"], mechanism["ci_high"]]),
                        hovertemplate=(
                            "%{y}<br>paired gain=%{x:.4f}"
                            "<br>outer-domain 95% interval=[%{customdata[0]:.4f}, %{customdata[1]:.4f}]"
                            "<extra></extra>"
                        ),
                    )
                )
                fig.add_vline(x=0, line_dash="dash", line_color="#6C757D")
                fig.update_layout(
                    title=f"What carries the K={primary_support} gain?",
                    xaxis_title="paired reduction in macro component log-MAE (positive is better)",
                    yaxis_title="",
                    template="plotly_white",
                    height=590,
                    margin={"l": 330, "r": 45, "b": 100},
                )
            fig.show()
            """,
            hidden=True,
        ),
        _code(
            r"""
            # %% hide input
            if mechanism.empty:
                display(Markdown("> Mechanism interpretation is waiting for the complete result bundle."))
            else:
                indexed = mechanism.set_index("comparison")
                gate_rows = tables["ablations"]
                gate_note = ""
                best_method_note = ""
                if gate_rows is not None:
                    choices = gate_rows.loc[gate_rows["metric"].astype(str) == "gate_choice_rate"]
                    rates = {
                        str(row["variant"]): float(row["estimate"])
                        for _, row in choices.iterrows()
                    }
                    transfer_rate = rates.get("evidence gate chose transfer")
                    specialist_rate = rates.get("evidence gate chose specialist")
                    if transfer_rate is not None and specialist_rate is not None:
                        gate_note = (
                            f" The gate selected its transfer path in {100 * transfer_rate:.1f}% of trials "
                            f"and its target-GDS specialist in {100 * specialist_rate:.1f}%."
                        )
                    primary_losses = gate_rows.loc[
                        gate_rows["metric"].astype(str) == "macro_component_log_mae"
                    ].dropna(subset=["estimate"])
                    if not primary_losses.empty:
                        best = primary_losses.sort_values("estimate").iloc[0]
                        best_method_note = (
                            f" At K={primary_support}, the lowest loss point estimate belongs to "
                            f"`{best['variant']}` ({float(best['estimate']):.4f}); this ranking is "
                            "descriptive unless a paired interval directly supports the contrast."
                        )

                def _mechanism_classification(label):
                    if label not in indexed.index:
                        return "not available"
                    row = indexed.loc[label]
                    if row["ci_low"] <= 0 <= row["ci_high"]:
                        return "inconclusive"
                    return "positive" if row["ci_low"] > 0 else "negative"

                source_label_text = _mechanism_classification(
                    "correct control-source labels - shuffled labels"
                )
                retrieval_label_text = _mechanism_classification(
                    "retrieved source labels - matched shuffled labels"
                )
                retrieval_target_text = _mechanism_classification(
                    "retrieved-source transfer - target parameter control"
                )
                v0_label_text = _mechanism_classification(
                    "correct v0-source labels - shuffled labels"
                )
                if retrieval_label_text == "positive":
                    retrieval_interpretation = (
                        "The frozen retrieval comparison is **positive** against its matched "
                        "shuffled-label control. This is benchmark-internal exploratory evidence "
                        "that support-conditioned retrieval exposes useful source-label structure; "
                        "it is not yet evidence of universal topology transfer, and the ten-domain "
                        "small-sample sensitivity remains material."
                    )
                else:
                    retrieval_interpretation = (
                        f"The frozen retrieval comparison is **{retrieval_label_text}** against "
                        "its matched shuffled-label control, so this release does not establish "
                        "retrieval-mediated source-label transfer."
                    )
                display(
                    Markdown(
                        "### Exploratory interpretation\n\n"
                        f"- The compact-control source-label test is **{source_label_text}** relative "
                        "to shuffled source labels.\n"
                        f"- {retrieval_interpretation}\n"
                        f"- Retrieved-source transfer versus the same-budget target parameter "
                        f"control is **{retrieval_target_text}** at K={primary_support}.\n"
                        f"- The compressed-v0 source-label test is **{v0_label_text}** relative to "
                        "shuffled source labels.\n"
                        "- The original full-source EGRA gain must **not** be described as evidence "
                        "that source capacitance labels transferred electrostatic knowledge. Its "
                        "supported interpretation is narrower: target-native geometry plus "
                        "evidence-gated regularization improved over a parameter-only target "
                        f"control in this exploratory release.{gate_note}{best_method_note}\n"
                        "- The topology-bridge campaign in Part II is required to test true "
                        "source-label transfer across geometry, matrix size, solver, process, and tool."
                    )
                )
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            # Part II: Dataset collection plan for Saikat Das

            The current cross-family experiment can test geometry transfer, but if both datasets
            have the same number of electrical nodes it cannot validate transfer across matrix
            size. The next release should therefore be designed as a **topology bridge dataset**,
            not simply a larger NCap sweep.

            The objective is to disentangle five shifts:

            1. geometry family;
            2. conductor count and capacitance-matrix structure;
            3. layout tool and parameter vocabulary;
            4. process stack and institution;
            5. EM solver, meshing policy, and boundary conditions.

            Controlled anchors that repeat across these axes are as important as novel layouts.

            **Node-count convention:** throughout this report, $N$ is the dimension of the stored
            Maxwell matrix and counts every modeled conductor, including the conductor flagged as
            ground/reference. The current datasets therefore have $N=3$: two signal conductors
            plus one modeled ground/reference conductor.
            """
        ),
        _markdown(
            r"""
            ## 10. What one simulation record must contain

            Each design is a small immutable package. The GDS is necessary but not sufficient:
            the net sidecar defines the matrix topology, and the solver metadata makes labels
            scientifically comparable.
            """
        ),
        _code(
            r"""
            # %% hide input
            labels = [
                "native layout project",
                "layout.gds",
                "design.json",
                "nets.json",
                "process.json",
                "solver.json",
                "simulation.json",
                "record manifest + hashes",
                "canonical conductor graph",
                "train / validate / audit",
            ]
            fig = go.Figure(
                go.Sankey(
                    arrangement="snap",
                    node={
                        "label": labels, "pad": 16, "thickness": 18,
                        "color": ["#264653", "#2A9D8F", "#2A9D8F", "#E9C46A", "#E9C46A", "#F4A261", "#D1495B", "#6A4C93", "#00798C", "#264653"],
                    },
                    link={
                        "source": [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8],
                        "target": [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 8, 9, 9],
                        "value": [1] * 15,
                        "color": "rgba(42,157,143,0.18)",
                    },
                )
            )
            fig.update_layout(
                title="One immutable simulation record: files, identity, and use",
                template="plotly_white", height=600,
            )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Required fields

            - **`layout.gds`**: final flattened or reproducibly referenced geometry, top-cell name,
              database unit, and SHA-256.
            - **`design.json`**: layout tool, tool version, component class, complete native design
              options, random/sweep seed, and exact code revision. Keep parameter names unchanged.
              For every editable knob also record its physical role, dimension, unit, legal bounds,
              discreteness, coupled constraints, and the native option path that must be updated.
              This is the round-trip contract that lets an ML proposal return to the originating
              layout tool instead of ending as an uninterpretable latent vector.
            - **`nets.json`**: ordered `net_id` list; reference flag; conductor role; GDS layer,
              datatype, polygon, or marker selectors; and any intentionally excluded metal.
            - **`process.json`**: layer stack, material, substrate and dielectric thicknesses,
              permittivity, metal thickness, etch/bias assumptions, and fabrication process ID.
            - **`solver.json`**: solver and version, electrostatic setup, boundaries, ports/nets,
              meshing controls, convergence history, final residual, warnings, host, and runtime.
            - **`simulation.json`**: full signed Maxwell matrix, explicit unit, ordered net IDs,
              raw solver export, normalized matrix, and QA flags. Never store only selected entries.
            - **`manifest.json`**: stable design ID plus hashes of every file and provenance links.

            Use one canonical capacitance unit in the published table, preferably fF, but retain the
            raw solver unit and export. Conversion must be explicit and testable.
            """
        ),
        _markdown(
            r"""
            A minimal record has this shape; `null` means the producer must fill a value before
            submission rather than allowing the consumer to guess:

            ```json
            {
              "schema_version": "squadds-topology-capacitance-v1",
              "design_id": null,
              "files": {
                "gds": {"path": "layout.gds", "sha256": null, "top_cell": null},
                "native_design": {"path": "design.json", "sha256": null},
                "net_map": {"path": "nets.json", "sha256": null},
                "process": {"path": "process.json", "sha256": null},
                "solver": {"path": "solver.json", "sha256": null},
                "simulation": {"path": "simulation.json", "sha256": null}
              },
              "matrix": {
                "convention": "signed_maxwell",
                "unit": "fF",
                "node_order": ["net_0", "net_1", "reference"],
                "values": null
              },
              "provenance": {
                "contributor": "Saikat Das",
                "group": "Levenson-Falk Lab",
                "institution": "USC",
                "code_revision": null,
                "created_utc": null
              }
            }
            ```
            """
        ),
        _markdown(
            r"""
            ## 11. Which devices should be simulated next?

            Do **not** optimize only for the number of rows. Optimize coverage of conductor count,
            topology, pairwise interaction, process, and acquisition domain. Start with simple
            devices whose electrical nets are unambiguous, then expand to composite structures.

            Dragging across conductor count in the conceptual coverage map below shows the intended
            progression. Labels are collection priorities, not measured coverage.
            """
        ),
        _code(
            r"""
            # %% hide input
            families = [
                "single-ended pad / line",
                "two-signal capacitor / NCap",
                "multi-pad transmon or coupler",
                "resonator-feedline junction",
                "filter / crossover / electrode array",
                "cross-tool bridge geometries",
            ]
            node_counts = list(range(2, 9))
            priority_text = np.array(
                [
                    ["core", "bridge", "stretch", "", "", "", ""],
                    ["anchor", "core", "bridge", "stretch", "", "", ""],
                    ["anchor", "core", "core", "core", "stretch", "stretch", "stretch"],
                    ["anchor", "core", "core", "core", "core", "stretch", "stretch"],
                    ["", "anchor", "core", "core", "core", "core", "stretch"],
                    ["core", "core", "core", "bridge", "bridge", "bridge", "bridge"],
                ],
                dtype=object,
            )
            priority_score = np.vectorize({"": 0, "stretch": 1, "bridge": 2, "anchor": 3, "core": 4}.get)(priority_text)
            fig = go.Figure(
                go.Heatmap(
                    z=priority_score,
                    x=node_counts,
                    y=families,
                    text=priority_text,
                    texttemplate="%{text}",
                    colorscale=[
                        [0.0, "#F7F7F7"], [0.249, "#F7F7F7"],
                        [0.25, "#E9C46A"], [0.499, "#E9C46A"],
                        [0.5, "#F4A261"], [0.749, "#F4A261"],
                        [0.75, "#2A9D8F"], [1.0, "#00798C"],
                    ],
                    showscale=False,
                    hovertemplate="family=%{y}<br>N=%{x}<br>priority=%{text}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Recommended topology-coverage map (collection plan, not observed data)",
                xaxis={"title": "electrical conductor count N", "dtick": 1, "side": "top"},
                yaxis={"title": "device family", "autorange": "reversed"},
                template="plotly_white", height=570, margin={"l": 240},
            )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Recommended first families

            - **N=2:** one signal conductor plus one modeled reference conductor. Use single-ended
              pads, lines, and shielded islands to anchor signal-to-reference interactions.
            - **N=3:** two signal conductors plus one modeled reference. Retain both current NCap
              families, then add straight, curved, asymmetric, and non-interdigital two-signal
              capacitors so matrix size is not confounded with NCap geometry.
            - **N=4 to N=6:** multi-pad transmons, tunable-coupler electrode systems, resonator plus
              feedline plus nearby pads, and small filter/crossover structures.
            - **N=7 to N=8:** use a smaller stretch set to test scaling and memory rather than
              dominating the initial budget.
            - **Bridge geometries:** simulate the exact same GDS and net order in each solver,
              process stack, layout tool export path, and participating institution. These anchors
              separate solver/process bias from geometry shift.

            Include several topology families at the same $N$, and the same topology family at
            several $N$. Otherwise node count and component identity remain inseparable.
            """
        ),
        _markdown(
            r"""
            ### Concrete first collection target

            The following is a **planning proposal**, not a measured optimum. It is large enough to
            test the next scientific claim while remaining batchable. Start with a 50--100 record
            schema pilot, validate ingestion end to end, and only then launch the approximately
            10,650-simulation first release below.

            - reserve every design's split assignment before simulation;
            - keep 15% of morphology/topology blocks as a permanently untouched challenge set;
            - devote about 10% of each stratum to boundary/weak-coupling stress cases;
            - repeat at least 100 geometries byte-for-byte across two solver configurations or
              institutions as acquisition-domain anchors;
            - never count a re-meshed or re-solved anchor as an independent geometry in ML splits.
            """
        ),
        _code(
            r"""
            # %% hide input
            collection_budget = pd.DataFrame(
                [
                    {"N": 2, "family": "single-ended pad / line", "core": 1200, "stress": 200, "anchors": 100},
                    {"N": 3, "family": "non-NCap two-signal capacitor", "core": 1800, "stress": 300, "anchors": 150},
                    {"N": 4, "family": "three-signal pad / coupler", "core": 1800, "stress": 300, "anchors": 150},
                    {"N": 5, "family": "resonator-feedline-pad junction", "core": 1500, "stress": 250, "anchors": 100},
                    {"N": 6, "family": "filter / electrode array", "core": 1200, "stress": 200, "anchors": 100},
                    {"N": 7, "family": "scaling challenge", "core": 500, "stress": 100, "anchors": 50},
                    {"N": 8, "family": "scaling challenge", "core": 500, "stress": 100, "anchors": 50},
                ]
            )
            long_budget = collection_budget.melt(
                id_vars=["N", "family"], var_name="batch role", value_name="planned simulations"
            )
            fig = px.bar(
                long_budget,
                x="N",
                y="planned simulations",
                color="batch role",
                custom_data=["family"],
                barmode="stack",
                color_discrete_map={"core": "#00798C", "stress": "#D1495B", "anchors": "#E9C46A"},
                title="Proposed first variable-topology release (planning counts)",
            )
            fig.update_traces(
                hovertemplate="N=%{x}<br>%{customdata[0]}<br>%{fullData.name}=%{y:,}<extra></extra>"
            )
            fig.update_layout(
                template="plotly_white", height=590,
                xaxis={"title": "matrix dimension N (reference included)", "dtick": 1},
                yaxis_title="planned solver runs",
                legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"},
                margin={"b": 115},
            )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Tool-independent sweep coordinates

            Saikat should express geometry bounds relative to the process design rules rather than
            copying NCap-specific micrometer ranges. Let $w_{min}$ be the minimum legal conductor
            width, $g_{min}$ the minimum legal gap, and $L$ the chosen active-region span for a
            topology. Within every topology/$N$ stratum, generate Sobol points over:

            - conductor widths: $[1, 8]w_{min}$;
            - pair gaps: $[1, 20]g_{min}$, sampled approximately log-uniformly;
            - facing/overlap lengths: $[0.05, 0.9]L$;
            - individual conductor area ratios: $[0.25, 4]$ relative to the stratum median;
            - ground clearance: $[1, 50]g_{min}$;
            - bend radius: from the process minimum to $0.25L$ where the topology permits it;
            - relative offsets and rotations across the full DRC-valid range;
            - dielectric thickness/permittivity and metal thickness as explicit process-domain
              variables, not unlabeled geometric noise.

            Reject DRC-invalid Sobol points, record the rejection reason, and replenish the stratum
            rather than clipping values to a boundary. Include deliberately weakly coupled pairs,
            strongly shielded pairs, and highly unequal conductor areas because those cases test the
            positive shunt/mutual decomposition most strongly.
            """
        ),
        _markdown(
            r"""
            ## 12. How to sweep geometry without rebuilding Cartesian grids

            Use a staged design of experiments. Every stage has a different scientific purpose:

            1. **schema pilot:** manually inspect a small set spanning all topology families and
               verify net order, units, matrix signs, hashes, and GDS selectors;
            2. **space-filling core:** Sobol or Latin-hypercube sampling over physically valid
               controls, generated separately within each topology and conductor-count stratum;
            3. **boundary stress:** minimum gaps, weak coupling, large area ratios, narrow traces,
               strong shielding, and near-DRC limits;
            4. **repeated anchors:** identical designs repeated across solver versions, machines,
               process stacks, and institutions to estimate acquisition variance;
            5. **adaptive batches:** add points where calibrated model uncertainty or validation
               error is high, while preserving a permanently untouched challenge set.

            A practical starting allocation is 60% space-filling core, 15% boundary stress,
            15% repeated/bridge anchors, and 10% adaptive reserve. Treat these as planning ratios,
            not a statistical result; revise them after the schema pilot and power analysis.
            """
        ),
        _code(
            r"""
            # %% hide input
            stages = ["schema pilot", "space-filling core", "boundary stress", "bridge anchors", "adaptive reserve"]
            dimensions = [
                "conductor count / topology",
                "pair gap and facing length",
                "conductor area / perimeter / curvature",
                "ground clearance / shielding",
                "dielectric and metal stack",
                "solver / institution / tool",
                "near-DRC and weak-coupling cases",
            ]
            status = np.array(
                [
                    [3, 3, 2, 2, 2],
                    [2, 3, 3, 2, 3],
                    [2, 3, 3, 2, 3],
                    [2, 3, 3, 2, 3],
                    [2, 3, 2, 3, 3],
                    [2, 2, 2, 3, 3],
                    [2, 2, 3, 2, 3],
                ]
            )
            status_text = np.array([["verify" if value == 3 else "sample" for value in row] for row in status])
            fig = go.Figure(
                go.Heatmap(
                    z=status, x=stages, y=dimensions, text=status_text, texttemplate="%{text}",
                    colorscale=[[0, "#F7F7F7"], [0.66, "#E9C46A"], [1, "#2A9D8F"]],
                    showscale=False,
                    hovertemplate="stage=%{x}<br>dimension=%{y}<br>action=%{text}<extra></extra>",
                )
            )
            fig.update_layout(
                title="What each acquisition stage must cover (planning guide)",
                xaxis={"side": "top", "tickangle": -20},
                yaxis={"autorange": "reversed"},
                template="plotly_white", height=620, margin={"l": 280, "t": 150},
            )
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ### Sweep and track these physical factors

            - conductor count and net-role composition;
            - individual conductor area, perimeter, compactness, curvature, and aspect ratio;
            - every pair's minimum/quantile gap, facing length, overlap, relative orientation,
              shielding, and interaction scale;
            - ground clearance, enclosure size, boundary distance, and reference-ground topology;
            - layer assignment, dielectric thickness and permittivity, metal thickness, and process ID;
            - native layout parameters and their exact mapping to generated polygons;
            - full solver convergence trajectory, adaptive-pass count, element count, residual,
              warnings, runtime, and machine/solver version;
            - DRC status and, for inverse-design studies, whether multiple distinct valid geometries
              realize the same target matrix within tolerance.

            Keep the untouched challenge set outside the adaptive loop. Otherwise active learning
            can make evaluation easier at the same time that it makes training larger.
            """
        ),
        _markdown(
            r"""
            ## 13. Simulation QA before upload

            Saikat should stop the batch and investigate if any required check fails. Tolerances
            must be derived from solver precision and recorded in the run manifest rather than
            silently hard-coded downstream.
            """
        ),
        _code(
            r"""
            # %% hide input
            checks = [
                "GDS hash matches net sidecar",
                "every modeled conductor maps to non-empty geometry",
                "matrix shape equals len(node_order) squared",
                "matrix unit and sign convention are explicit",
                "symmetry residual is below recorded tolerance",
                "diagonal entries are positive",
                "off-diagonal entries are non-positive within tolerance",
                "row shunts and physical diagnostics are recorded",
                "solver convergence and warnings are retained",
                "native design options reproduce the same GDS hash",
                "split-group and bridge-anchor IDs are assigned before training",
            ]
            rationale = [
                "prevents labels from being attached to the wrong geometry",
                "avoids empty or duplicate electrical nodes",
                "makes variable-N labels unambiguous",
                "prevents hidden F/pF/fF or mutual/Maxwell conversion errors",
                "catches extraction or solver failures",
                "checks the expected signed Maxwell convention",
                "checks the expected signed Maxwell convention",
                "preserves physical validity and edge cases",
                "supports reproducibility and uncertainty analysis",
                "keeps the feedback path to the layout tool testable",
                "prevents leakage and identifies controlled domain bridges",
            ]
            fig = go.Figure(
                go.Table(
                    columnwidth=[0.34, 0.66],
                    header={
                        "values": ["required QA check", "why it matters"],
                        "fill_color": "#264653", "font": {"color": "white", "size": 13},
                        "align": "left", "height": 34,
                    },
                    cells={
                        "values": [checks, rationale],
                        "fill_color": [["#F7FAFA", "#EDF6F5"] * 6],
                        "align": "left", "height": 31,
                    },
                )
            )
            fig.update_layout(title="Pre-upload simulation QA checklist", height=650, margin={"l": 20, "r": 20})
            fig.show()
            """,
            hidden=True,
        ),
        _markdown(
            r"""
            ## 14. Delivery structure and first handoff

            Submit one directory per immutable design and a release-level index:

            ```text
            topology-capacitance-release-v1/
              release.json
              catalogue.parquet
              records/
                <design_id>/
                  manifest.json
                  layout.gds
                  design.json
                  nets.json
                  process.json
                  solver.json
                  simulation.json
                  raw_solver_export/
            ```

            Before launching a large sweep, send a **schema pilot** containing at least one valid
            record for each proposed conductor count, topology family, solver, and process stack.
            We should run ingestion, geometry extraction, matrix reconstruction, permutation, and
            round-trip layout checks on that pilot. Only then should the full Sobol/Latin-hypercube
            batches begin.

            ### The next decisive experiment

            A strong proof of generalization is not another within-family random split. It is:

            - pretrain on several topology families and conductor counts from one acquisition domain;
            - adapt with a small, nested support set from a held-out family, $N$, solver, process, or institution;
            - evaluate on a separately held-out morphology block from that target domain;
            - compare against target-only, parameter-only, GDS-only, fixed-bitmap, shuffled-source,
              and full-target oracle controls;
            - report paired intervals, negative-transfer frequency, physical validity, calibration,
              source-budget controls, and solver-verified inverse-design success.

            That experiment will tell us whether the model has learned transferable electrostatic
            interactions rather than memorized NCap geometry or a particular 3 x 3 label convention.
            """
        ),
    ]

    _set_deterministic_cell_ids(notebook)
    return notebook


def _sanitize_execution_metadata(notebook) -> None:
    for cell in notebook["cells"]:
        cell.get("metadata", {}).pop("execution", None)


def _write_notebook(notebook, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    nbf.write(notebook, temporary)
    temporary.replace(output)


def _execute_notebook(notebook, *, kernel_name: str, timeout: int):
    try:
        from nbclient import NotebookClient
    except ImportError as error:
        raise SystemExit("Notebook execution requires nbclient in the active environment.") from error

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": os.fspath(REPOSITORY_ROOT)}},
    )
    executed = client.execute()
    _sanitize_execution_metadata(executed)
    return executed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Tutorial 18 from compact, versioned TopoCap experiment artifacts."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing manifest.json and compact result tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Notebook destination (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute all cells after generation; missing artifacts render waiting panels.",
    )
    parser.add_argument(
        "--strict-results",
        action="store_true",
        help="Fail unless all six empirical result artifacts are present.",
    )
    parser.add_argument(
        "--kernel-name",
        default="squadds-tutorial",
        help="Jupyter kernel used with --execute (default: squadds-tutorial).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-cell execution timeout in seconds (default: 900).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.strict_results:
        missing = _validate_result_directory(arguments.results_dir)
        if missing:
            raise SystemExit("Missing or unreadable result artifacts: " + ", ".join(missing))

    notebook = build_notebook(arguments.results_dir)
    if arguments.execute:
        notebook = _execute_notebook(
            notebook,
            kernel_name=arguments.kernel_name,
            timeout=arguments.timeout,
        )
    _write_notebook(notebook, arguments.output)
    mode = "executed" if arguments.execute else "generated"
    print(f"{mode}: {arguments.output}")


if __name__ == "__main__":
    main()
