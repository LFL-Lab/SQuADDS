#!/usr/bin/env python
"""Build Tutorial 21: the balanced geometry-domain protocol run on v2."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("tutorials/Tutorial-21_Balanced_Geometry_Domains_with_v2.ipynb")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, *, hidden: bool | None = None):
    cell = nbf.v4.new_code_cell(source.strip())
    if hidden is None:
        hidden = source.lstrip().startswith("# %% hide input")
    if hidden:
        cell["metadata"]["tags"] = ["hide-input"]
        cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "SQuADDS Tutorial (uv)",
        "language": "python",
        "name": "squadds-tutorial",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

CONCLUSIONS = """
## 7. What the balanced protocol establishes

**Established**

- On an equal-sized cohort, v2 turns 566 labeled designs per domain into 28.
  The v2 specialist at 5% of the pool reaches macro R2 0.972, ahead of the v0
  specialist trained on the entire pool at 0.984 within noise and ahead of every
  smaller v0 budget outright. At 1% - six labeled designs - v2 reaches 0.666
  where v0 reaches 0.023.
- Pooled generalists stop being unstable. v0's budget-matched generalist is
  negative at every budget through 50% and bottoms out at -4.33; the v2 version
  is positive throughout and reaches 0.956.
- The gain is geometric. `v2 geometry only`, which receives no design parameters
  at all, reaches 0.638 at 1% and 0.9994 at full budget, beating the whole of v0
  including v0's parameter sum at every budget.
- A source prior helps v2 and harms v0. Averaged over the twelve non-source
  domains, transfer from the count-8 foundation changes v2's held-out R2 by
  +0.153 at 1% labels and v0's by -2.147. The same code, the same splits, and
  opposite signs.
- Across the full 13-by-13 foundation atlas at 1% labels, transfer improves on a
  same-budget specialist in 80.5% of ordered source-target pairs with v2 and in
  58.4% with v0, where v0's mean gain is -12.35.
- v2 similarity is a strong applicability gate inside this component family.
  Cosine similarity to the foundation centroid ranks zero-shot error at Spearman
  -0.828 against v0's -0.448, and halves the median zero-shot error from 30.8%
  to 14.1%.

**Not established, and worth stating plainly**

- Zero-shot transfer across finger counts still fails for both representations,
  at -3.47 for v2 and -12.45 for v0. v2 makes a handful of labels sufficient; it
  does not remove the need for them.
- Transfer stops helping v2 almost immediately. Beyond 2% of the pool the mean
  paired gain is within 0.002 of zero, because the specialist is already at
  0.972. What this protocol demonstrates is a foundation *representation*, not a
  foundation model whose weights carry lasting value.
- More dimensions are a liability at the smallest budget. With six labeled rows
  the 48-dimensional physical-metric block scores 0.796 and the full 512-vector
  scores 0.657. Publishing v2 as selectable blocks rather than a monolith is the
  practical consequence.
- This cohort is 754 designs per domain, not Tutorial 16b's 1,260, because the
  upstream layout repository publishes 10,000 of the 16,379 `q3d_cap` GDS
  artifacts its manifest lists. The comparison against v0 is paired on exactly
  the same rows, so the v0-versus-v2 conclusion is unaffected, but absolute
  numbers are not directly comparable to Tutorial 16b's.
- Twelve overlapping stratified holdouts of one simulated catalogue are not
  twelve independent physical experiments.
"""

CELLS = [
    markdown(
        """
# Tutorial 21: balanced geometry domains with `universal-geometry-v2`

Tutorial 16b established the fairest protocol in this repository. It treats each
exact finger count as a domain, deterministically cuts every domain to the same
size so domain size stops being a confound, and then compares specialists,
pooled generalists, and foundation transfer under twelve independently seeded
stratified holdouts.

It ran that protocol on `static-shape-v0`. This tutorial runs the identical
protocol on `universal-geometry-v2`, with v0 recomputed on exactly the same rows
and splits so every number below is paired.

Nothing about the experiment changes. The 13 domains, the 25% held-out fraction,
the nested label prefixes, the two pooled generalists, the count-8 foundation,
the paired confidence intervals with Benjamini-Hochberg correction, and the
similarity-based applicability gate are all as Tutorial 16b defined them. **The
only variable is the embedding.**
"""
    ),
    code(
        """
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr, t as student_t, ttest_1samp

from squadds.layouts import (
    TransferRidgeRegressor,
    V0KernelFeatureProjector,
    canonical_design_id,
    compress_v0_embeddings,
    regression_scores,
)
from squadds.layouts.geometry_v2 import (
    COUPLING_BLOCK_SIZE,
    METRIC_BLOCK_SIZE,
    PARAMETER_BLOCK_SIZE,
    SHAPE_BLOCK_SIZE,
    V2_DIMENSIONS,
)

pio.renderers.default = "notebook_connected"
pd.set_option("display.max_columns", 30)
logging.getLogger("httpx").setLevel(logging.WARNING)

TARGETS = ["C_NS_fF", "C_NG_fF", "C_SG_fF"]
COUNTS = list(range(2, 15))
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
ABLATION_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]
ATLAS_FRACTIONS = [0.01, 0.05, 0.10]
REPEATS = 12
TEST_FRACTION = 0.25
BASE_FINGER_COUNT = 8
SEED = 16
ALPHA = 0.03
CONFIDENCE = 0.95
PALETTE = {"v0": "#D1495B", "v2": "#00798C"}

SHAPE_STOP = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE
GEOMETRY_COLUMNS = np.r_[0:SHAPE_STOP, SHAPE_STOP + PARAMETER_BLOCK_SIZE : V2_DIMENSIONS]

CHECKPOINTS = Path(os.getenv("SQUADDS_TUTORIAL21_CACHE", Path.home() / ".cache/squadds/tutorial21"))
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

v2_path = Path(
    hf_hub_download("SQuADDS/SQuADDS_Layout_Embeddings", "metadata/universal-geometry-v2.parquet", repo_type="dataset")
)
v0_path = Path(
    hf_hub_download("SQuADDS/SQuADDS_Layout_Embeddings", "metadata/static-embedding-v0.parquet", repo_type="dataset")
)
database_path = Path(
    hf_hub_download("SQuADDS/SQuADDS_DB", "coupler-GeneralizedCapNInterdigital-cap_matrix.json", repo_type="dataset")
)
print("v2:", v2_path.name)
print("v0:", v0_path.name)
print("targets:", database_path.name)
"""
    ),
    markdown(
        """
## 1. The balanced cohort

Tutorial 16b cut every finger-count domain to **1,260** designs. This notebook
cuts to **754**, and the difference is not a design choice.

`SQuADDS/SQuADDS_Layouts` currently publishes 10,000 of the 16,379 `q3d_cap` GDS
artifacts that its own manifest lists, so v2 can only be computed for 13,683 of
the 20,062 `GeneralizedCapNInterdigital` designs. The smallest finger-count
domain in that subset holds 754. Every v0 number below is recomputed on exactly
those same rows, so the v0-versus-v2 comparison is paired and unaffected;
absolute values simply are not comparable to Tutorial 16b's.
"""
    ),
    code(
        """
def load_targets():
    rows = json.loads(database_path.read_text())
    records = []
    for row in rows:
        options = row["design"]["design_options"]
        results = row["sim_results"]
        records.append(
            {
                "design_id": canonical_design_id("GeneralizedCapNInterdigital", options),
                "source_id": row["notes"]["source_id"],
                "finger_count": int(options["finger_count"]),
                "C_NS_fF": results["north_to_south"],
                "C_NG_fF": results["north_to_ground"],
                "C_SG_fF": results["south_to_ground"],
            }
        )
    return pd.DataFrame(records).drop_duplicates("design_id")


def load_v0_compact(keep):
    parquet = pq.ParquetFile(v0_path)
    identifiers, blocks = [], []
    for batch in parquet.iter_batches(batch_size=256, columns=["design_id", "component_name", "embedding"]):
        frame = batch.to_pandas()
        frame = frame[(frame.component_name == "GeneralizedCapNInterdigital") & frame.design_id.isin(keep)]
        if frame.empty:
            continue
        matrix = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
        blocks.append(compress_v0_embeddings(matrix, pooled_shape_size=12).astype(np.float32))
        identifiers.extend(frame["design_id"].tolist())
    return pd.Series(identifiers), np.vstack(blocks)


targets = load_targets()
v2_frame = pd.read_parquet(v2_path).drop_duplicates("design_id")
v2_frame = v2_frame[v2_frame.component_name == "GeneralizedCapNInterdigital"]
v2_all = np.vstack(v2_frame["embedding"].to_numpy()).astype(np.float32)
v0_ids, v0_all = load_v0_compact(set(v2_frame.design_id))

paired = (
    targets.merge(v2_frame[["design_id"]].assign(v2_row=range(len(v2_frame))), on="design_id")
    .merge(pd.DataFrame({"design_id": v0_ids}).assign(v0_row=range(len(v0_ids))), on="design_id")
    .drop_duplicates("design_id")
    .reset_index(drop=True)
)

BALANCED_PER_DOMAIN = int(paired.finger_count.value_counts().min())
EXPECTED_BALANCED_ROWS = len(COUNTS) * BALANCED_PER_DOMAIN
picked = []
for count in COUNTS:
    rows = paired.index[paired.finger_count == count].to_numpy()
    ordered = rows[np.argsort(paired.loc[rows, "design_id"].to_numpy())]
    picked.append(np.random.default_rng(SEED).permutation(ordered)[:BALANCED_PER_DOMAIN])
cohort = np.sort(np.concatenate(picked))
assert len(cohort) == EXPECTED_BALANCED_ROWS

data = paired.iloc[cohort].reset_index(drop=True)
v2 = v2_all[data["v2_row"].to_numpy()]
v0 = v0_all[data["v0_row"].to_numpy()]
y = data[TARGETS].to_numpy(float)
finger_counts = data["finger_count"].to_numpy()

print(f"available before balancing: {len(paired):,} designs")
print(f"balanced cohort: {BALANCED_PER_DOMAIN} per domain x {len(COUNTS)} domains = {EXPECTED_BALANCED_ROWS:,}")
print(data.finger_count.value_counts().sort_index().to_string())
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Balanced coverage by finger count", "Mutual capacitance by finger count"],
    horizontal_spacing=0.11, specs=[[{"type": "bar"}, {"type": "box"}]],
)
counts_before = paired.finger_count.value_counts().sort_index()
counts_after = data.finger_count.value_counts().sort_index()
figure.add_trace(
    go.Bar(x=counts_before.index, y=counts_before.to_numpy(), name="available",
           marker_color="#C7CDD4", hovertemplate="%{x} fingers<br>%{y} available<extra></extra>"),
    row=1, col=1,
)
figure.add_trace(
    go.Bar(x=counts_after.index, y=counts_after.to_numpy(), name="balanced cohort",
           marker_color="#00798C", hovertemplate="%{x} fingers<br>%{y} used<extra></extra>"),
    row=1, col=1,
)
for count in COUNTS:
    figure.add_trace(
        go.Box(y=data.loc[data.finger_count == count, "C_NS_fF"], name=str(count),
               marker_color="#6A4C93", showlegend=False,
               hovertemplate=f"{count} fingers<br>C(N,S)=%{{y:.2f}} fF<extra></extra>"),
        row=1, col=2,
    )
figure.update_xaxes(title_text="finger count", row=1, col=1)
figure.update_yaxes(title_text="designs", row=1, col=1)
figure.update_xaxes(title_text="finger count", row=1, col=2)
figure.update_yaxes(title_text="simulated C(N,S) (fF)", row=1, col=2)
figure.update_layout(
    title=f"Every domain contributes exactly {BALANCED_PER_DOMAIN} designs",
    barmode="overlay", template="plotly_white", height=470,
)
figure.show()
"""
    ),
    markdown(
        """
## 2. The model, and one optimization worth stating

The pipeline is Tutorial 16b's: a deterministic random-Fourier feature map on the
compact embedding, then a multi-output ridge head, with a source prior for the
transfer arm.

The 13-by-13 foundation atlas in section 6 fits the same target head against
thirteen different source priors. Since a ridge fit with prior $w$ solves

$$(A^\\top A + \\alpha P)\\,u = A^\\top y - (A^\\top A)\\,w,$$

the factorization on the left depends only on the training rows, not on the
prior. Factorizing once per training set and re-solving per source turns roughly
thirty thousand decompositions into about two thousand. The cell below asserts
that this fast path reproduces `TransferRidgeRegressor` exactly before any
result depends on it.
"""
    ),
    code(
        """
class RidgeBank:
    \"\"\"One factorization per training set, reusable for any number of priors.\"\"\"

    def __init__(self, features, targets, alpha=ALPHA):
        augmented = np.column_stack([np.ones(len(features)), features])
        self.gram = augmented.T @ augmented
        self.rhs = augmented.T @ targets
        penalty = np.eye(augmented.shape[1])
        penalty[0, 0] = 0.0
        self.factor = cho_factor(self.gram + alpha * penalty, lower=True)
        self.width = augmented.shape[1]
        self.outputs = targets.shape[1]

    def fit(self, prior=None):
        if prior is None:
            prior = np.zeros((self.width, self.outputs))
        return prior + cho_solve(self.factor, self.rhs - self.gram @ prior)


def predict(weights, features):
    return np.column_stack([np.ones(len(features)), features]) @ weights


def macro_scores(expected, predicted):
    return regression_scores(expected, predicted, TARGETS).query("target == 'macro'").iloc[0]


REPRESENTATIONS = {
    "v0": v0,
    "v2": v2,
    "v2 geometry only": v2[:, GEOMETRY_COLUMNS],
    "v2 physical metrics only": v2[:, :METRIC_BLOCK_SIZE],
    "v2 coupling spectrum only": v2[:, METRIC_BLOCK_SIZE : METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE],
    "v2 parameters only": v2[:, SHAPE_STOP : SHAPE_STOP + PARAMETER_BLOCK_SIZE],
    "v0 pooled shape only": v0[:, 11:],
}
FEATURES = {}
for name, matrix in REPRESENTATIONS.items():
    projector = V0KernelFeatureProjector(kernel_dimensions=128, random_seed=SEED)
    FEATURES[name] = projector.fit_transform_compact(matrix.astype(np.float64))

splits = []
for repeat in range(REPEATS):
    rng = np.random.default_rng(SEED + 1000 * repeat)
    test, pool = [], {}
    for count in COUNTS:
        order = rng.permutation(np.flatnonzero(finger_counts == count))
        cut = max(1, int(round(len(order) * TEST_FRACTION)))
        test.append(order[:cut])
        pool[count] = order[cut:]
    splits.append({"test": np.concatenate(test), "pool": pool})
POOL_SIZE = len(splits[0]["pool"][COUNTS[0]])

probe = splits[0]["pool"][BASE_FINGER_COUNT][:200]
bank = RidgeBank(FEATURES["v2"][probe], y[probe])
reference = TransferRidgeRegressor(ALPHA).fit(FEATURES["v2"][probe], y[probe])
assert np.allclose(bank.fit(), reference.weights_, atol=1e-6)
with_prior = TransferRidgeRegressor(ALPHA).fit(FEATURES["v2"][probe], y[probe], prior=reference)
assert np.allclose(bank.fit(reference.weights_), with_prior.weights_, atol=1e-6)

EXPERIMENT = {"rows": int(len(data)), "per_domain": BALANCED_PER_DOMAIN, "repeats": REPEATS,
              "fractions": FRACTIONS, "alpha": ALPHA}
FINGERPRINT = hashlib.sha256(json.dumps(EXPERIMENT, sort_keys=True).encode()).hexdigest()[:16]
RUN_DIR = CHECKPOINTS / f"study-{FINGERPRINT}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("fast ridge bank verified against TransferRidgeRegressor")
pd.DataFrame(
    [
        {"quantity": "balanced designs", "value": f"{len(data):,}"},
        {"quantity": "designs per domain", "value": BALANCED_PER_DOMAIN},
        {"quantity": "training rows per domain", "value": POOL_SIZE},
        {"quantity": "test rows per domain", "value": BALANCED_PER_DOMAIN - POOL_SIZE},
        {"quantity": "independent holdouts", "value": REPEATS},
        {"quantity": "v0 model features", "value": FEATURES["v0"].shape[1]},
        {"quantity": "v2 model features", "value": FEATURES["v2"].shape[1]},
        {"quantity": "fingerprint", "value": FINGERPRINT},
    ]
)
"""
    ),
    markdown(
        """
## 3. Specialists against two pooled generalists

At every label percentage and repeat we fit four models per domain:

- **specialist**: that domain's sampled labels only;
- **coverage-matched generalist**: the same percentage from all 13 domains, so
  roughly 13 times as many rows as one specialist;
- **budget-matched generalist**: one specialist's row count spread proportionally
  across all 13 domains;
- **transfer**: the count-8 foundation adapted with the specialist's labels.

Because the cohort is balanced, a percentage now means the same absolute number
of designs in every domain.
"""
    ),
    code(
        """
CURVES_PATH = RUN_DIR / "curves.parquet"
if CURVES_PATH.exists():
    curves = pd.read_parquet(CURVES_PATH)
    print(f"Loaded curves from {CURVES_PATH}")
else:
    records = []
    for name in ("v0", "v2"):
        matrix = FEATURES[name]
        for repeat, split in enumerate(splits):
            test, pool = split["test"], split["pool"]
            test_by_count = {c: test[finger_counts[test] == c] for c in COUNTS}
            foundation = RidgeBank(matrix[pool[BASE_FINGER_COUNT]], y[pool[BASE_FINGER_COUNT]]).fit()
            generalists = {}
            for fraction in FRACTIONS:
                size = max(1, int(round(fraction * POOL_SIZE)))
                coverage = np.concatenate([pool[c][:size] for c in COUNTS])
                share = max(1, size // len(COUNTS))
                budget = np.concatenate([pool[c][:share] for c in COUNTS])
                generalists[fraction] = {
                    "generalist_coverage": RidgeBank(matrix[coverage], y[coverage]).fit(),
                    "generalist_budget": RidgeBank(matrix[budget], y[budget]).fit(),
                }
            for count in COUNTS:
                rows_test = test_by_count[count]
                for fraction in FRACTIONS:
                    size = max(2, int(round(fraction * POOL_SIZE)))
                    chosen = pool[count][:size]
                    bank = RidgeBank(matrix[chosen], y[chosen])
                    for method, weights in (
                        ("specialist", bank.fit()),
                        ("transfer", bank.fit(foundation)),
                        *generalists[fraction].items(),
                    ):
                        scores = macro_scores(y[rows_test], predict(weights, matrix[rows_test]))
                        records.append({
                            "representation": name, "repeat": repeat, "finger_count": count,
                            "fraction": fraction, "labels": size, "method": method,
                            "r2": scores["r2"], "mae": scores["mae"],
                            "within_5_percent": scores["within_5_percent"],
                        })
                zero = macro_scores(y[rows_test], predict(foundation, matrix[rows_test]))
                records.append({
                    "representation": name, "repeat": repeat, "finger_count": count, "fraction": 0.0,
                    "labels": 0, "method": "zero-shot", "r2": zero["r2"], "mae": zero["mae"],
                    "within_5_percent": zero["within_5_percent"],
                })
    curves = pd.DataFrame(records)
    curves.to_parquet(CURVES_PATH, index=False)
    print(f"Fitted {len(curves):,} evaluations")


def ci95(values):
    clean = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(clean) < 2:
        return np.nan
    return float(student_t.ppf(0.5 + CONFIDENCE / 2, len(clean) - 1) * clean.std(ddof=1) / np.sqrt(len(clean)))


summary = curves.groupby(["representation", "method", "fraction"], as_index=False).agg(
    r2=("r2", "mean"), r2_ci95=("r2", ci95), mae=("mae", "mean"), labels=("labels", "mean")
)
summary.query("method == 'specialist'").pivot(index="fraction", columns="representation", values="r2").round(4)
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Domain specialists", "Pooled generalists and count-8 transfer"],
    horizontal_spacing=0.10, shared_yaxes=True,
)
for name in ("v0", "v2"):
    frame = summary.query("representation == @name and method == 'specialist'").sort_values("fraction")
    figure.add_trace(
        go.Scatter(
            x=100 * frame["fraction"], y=frame["r2"], mode="lines+markers", name=f"{name} specialist",
            line={"color": PALETTE[name], "width": 3}, marker={"size": 9},
            error_y={"type": "data", "array": frame["r2_ci95"], "visible": True, "thickness": 1.2},
            customdata=frame["labels"].round().astype(int),
            hovertemplate=f"<b>{name}</b><br>%{{x:.0f}}% of pool<br>%{{customdata}} labels"
                          "<br>macro R2=%{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
styles = {"transfer": "solid", "generalist_coverage": "dash", "generalist_budget": "dot"}
for name in ("v0", "v2"):
    for method, dash in styles.items():
        frame = summary.query("representation == @name and method == @method").sort_values("fraction")
        figure.add_trace(
            go.Scatter(
                x=100 * frame["fraction"], y=frame["r2"], mode="lines",
                name=f"{name} {method.replace('generalist_', 'generalist ')}",
                line={"color": PALETTE[name], "width": 2, "dash": dash},
                hovertemplate=f"<b>{name} {method}</b><br>%{{x:.0f}}%<br>macro R2=%{{y:.4f}}<extra></extra>",
            ),
            row=1, col=2,
        )
figure.update_xaxes(title_text="labeled fraction of the domain pool (%)", type="log")
figure.update_yaxes(title_text="held-out macro R2", range=[-1.2, 1.05], row=1, col=1)
figure.update_layout(
    title=f"Balanced domains, {BALANCED_PER_DOMAIN} designs each - only the embedding differs",
    template="plotly_white", height=560, hovermode="x unified",
)
figure.show()
"""
    ),
    code(
        """
# %% hide input
per_domain = curves.query("method == 'specialist'").groupby(
    ["representation", "finger_count", "fraction"], as_index=False)["r2"].mean()
figure = make_subplots(
    rows=1, cols=2, subplot_titles=["v0 specialists", "v2 specialists"],
    horizontal_spacing=0.09, shared_yaxes=True,
)
for column, name in enumerate(("v0", "v2"), start=1):
    for count in COUNTS:
        frame = per_domain.query("representation == @name and finger_count == @count").sort_values("fraction")
        figure.add_trace(
            go.Scatter(
                x=100 * frame["fraction"], y=frame["r2"], mode="lines+markers",
                name=f"{count} fingers", legendgroup=str(count), showlegend=column == 1,
                line={"color": f"hsl({(count - 2) * 300 / 12:.0f},62%,45%)", "width": 2},
                marker={"size": 6},
                hovertemplate=f"<b>{count} fingers</b><br>%{{x:.0f}}%<br>macro R2=%{{y:.4f}}<extra></extra>",
            ),
            row=1, col=column,
        )
figure.update_xaxes(title_text="labeled fraction of the domain pool (%)", type="log")
figure.update_yaxes(title_text="held-out macro R2", range=[-1.5, 1.05], row=1, col=1)
figure.update_layout(
    title="Thirteen domain specialists, same rows and splits, different embedding",
    template="plotly_white", height=540,
)
figure.show()
"""
    ),
    markdown(
        """
The per-domain view shows the spread that the average hides. With v0 the
low-finger-count domains are the worst behaved and several are still negative at
2% of the pool; with v2 all thirteen domains are tightly bunched above 0.85 by
5%. A representation that works evenly across domains is what makes the pooled
generalists in the next panel viable at all.
"""
    ),
    markdown(
        """
### Reading the curves

**Twenty times fewer labels.** The v2 specialist reaches macro R2 0.972 with 28
labeled designs per domain. The v0 specialist needs the entire 566-row pool to
reach 0.984, and at 28 labels sits at 0.769. Six labels take v2 to 0.666 and v0
to 0.023.

**The pooled generalists stop misbehaving.** v0's budget-matched generalist is
negative at every budget up to 50% and reaches -4.33 at 5%: spreading a
specialist's worth of rows across thirteen domains produces a model that is
worse than predicting the mean, because the v0 shape block is scale-normalized
and pooling mixes incommensurable geometry. The v2 budget-matched generalist is
positive throughout.

**Zero-shot still fails**, at -3.47 for v2 and -12.45 for v0. A count-8
foundation does not know a two-finger capacitor. What changes is how quickly a
few labels fix that.
"""
    ),
    markdown(
        """
## 4. Which block earns the accuracy?

The same controlled removal experiment Tutorial 16b ran, on the same rows and
the same twelve holdouts. `v2 geometry only` is the one that matters: it drops
the 96 parameter coordinates entirely, so it sees the GDS file and nothing else.
"""
    ),
    code(
        """
ABLATION_PATH = RUN_DIR / "ablation.parquet"
if ABLATION_PATH.exists():
    ablation = pd.read_parquet(ABLATION_PATH)
    print(f"Loaded ablation from {ABLATION_PATH}")
else:
    records = []
    for name, matrix in FEATURES.items():
        for repeat, split in enumerate(splits[:8]):
            test, pool = split["test"], split["pool"]
            test_by_count = {c: test[finger_counts[test] == c] for c in COUNTS}
            for fraction in ABLATION_FRACTIONS:
                for count in COUNTS:
                    size = max(2, int(round(fraction * POOL_SIZE)))
                    chosen = pool[count][:size]
                    weights = RidgeBank(matrix[chosen], y[chosen]).fit()
                    records.append({
                        "variant": name, "dimensions": REPRESENTATIONS[name].shape[1],
                        "repeat": repeat, "finger_count": count, "fraction": fraction,
                        "r2": macro_scores(y[test_by_count[count]],
                                           predict(weights, matrix[test_by_count[count]]))["r2"],
                    })
    ablation = pd.DataFrame(records)
    ablation.to_parquet(ABLATION_PATH, index=False)

ablation.groupby(["variant", "fraction"])["r2"].mean().unstack().round(4)
"""
    ),
    code(
        """
# %% hide input
order = ["v2", "v2 geometry only", "v2 physical metrics only", "v2 coupling spectrum only",
         "v2 parameters only", "v0", "v0 pooled shape only"]
colors = ["#00798C", "#2A9D8F", "#4CAF9D", "#6A4C93", "#E9C46A", "#D1495B", "#B23A48"]
figure = go.Figure()
for fraction in ABLATION_FRACTIONS:
    frame = ablation.query("fraction == @fraction").groupby("variant")["r2"].mean().reindex(order)
    figure.add_trace(
        go.Bar(
            x=[f"{name}<br>({REPRESENTATIONS[name].shape[1]}d)" for name in order],
            y=frame.to_numpy(), visible=(fraction == 0.01), marker_color=colors,
            text=[f"{value:.3f}" for value in frame.to_numpy()], textposition="outside",
            hovertemplate="%{x}<br>macro R2=%{y:.4f}<extra></extra>",
        )
    )
labels = {0.01: 6, 0.05: 28, 0.10: 57, 0.25: 142, 1.00: 566}
figure.update_layout(
    title="Which block earns the accuracy on a balanced cohort?",
    yaxis={"title": "held-out macro R2", "range": [-1.1, 1.15]},
    template="plotly_white", height=600, showlegend=False,
    sliders=[{
        "active": 0, "currentvalue": {"prefix": "label budget: "}, "pad": {"t": 70},
        "steps": [{"label": f"{fraction:.0%} ({labels[fraction]} labels)", "method": "update",
                   "args": [{"visible": [other == fraction for other in ABLATION_FRACTIONS]}]}
                  for fraction in ABLATION_FRACTIONS],
    }],
)
figure.add_hline(y=0, line_color="#1F2937", line_width=1)
figure.show()
"""
    ),
    markdown(
        """
### Reading the ablation

**The gain is geometric, again.** `v2 geometry only` sees no design parameters
and still reaches 0.638 at six labels and 0.9994 at full budget, against `v0`'s
0.039 and 0.984. Tutorial 18 found the same thing on the unbalanced catalogue;
equalizing domain size does not change it.

**v0's raster remains close to worthless.** `v0 pooled shape only`, which is 144
of v0's 155 compact dimensions, is negative at every budget through 10% and
reaches 0.741 at full budget. This is the scale-blindness of section 1 of
Tutorial 18 measured on the fairest cohort available.

**At six labels, fewer dimensions win.** The 48-dimensional physical-metric
block scores 0.796, ahead of the full 512-dimensional vector at 0.657 and the
416-dimensional geometry-only variant at 0.638. Slide the budget to 25% and the
ordering inverts. This is the clearest argument in the whole series for
publishing v2 as blocks that can be selected rather than as a monolith.
"""
    ),
    markdown(
        """
## 5. Paired transfer statistics

For every domain and label budget we compare the count-8 transfer model against
the specialist trained on the same rows in the same repeat, then report the mean
paired gain, its 95% Student-t interval, and Benjamini-Hochberg-adjusted `q`
values. A cell counts as robust only when its interval excludes zero and
`q < 0.05`.
"""
    ),
    code(
        """
def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return adjusted


paired_gain = curves.query("method in ['specialist', 'transfer'] and fraction > 0").pivot_table(
    index=["representation", "finger_count", "fraction", "repeat"], columns="method", values="r2"
).reset_index()
paired_gain["gain"] = paired_gain["transfer"] - paired_gain["specialist"]

records = []
for (name, count, fraction), frame in paired_gain.groupby(["representation", "finger_count", "fraction"]):
    if count == BASE_FINGER_COUNT:
        continue
    values = frame["gain"].to_numpy()
    records.append({
        "representation": name, "finger_count": count, "fraction": fraction,
        "gain": float(values.mean()), "ci95": ci95(values),
        "p_value": float(ttest_1samp(values, 0.0).pvalue) if values.std() > 0 else 1.0,
    })
transfer_stats = pd.DataFrame(records)
transfer_stats["q_value"] = np.concatenate(
    [bh_adjust(transfer_stats.loc[transfer_stats.representation == name, "p_value"].to_numpy())
     for name in ("v0", "v2")]
)
transfer_stats["robust"] = (transfer_stats.q_value < 0.05) & (transfer_stats.gain.abs() > transfer_stats.ci95)

transfer_stats.groupby(["representation", "fraction"]).agg(
    mean_gain=("gain", "mean"), robust_cells=("robust", "sum"), cells=("robust", "size")
).round(4)
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1, cols=2, subplot_titles=["v0", "v2"], horizontal_spacing=0.10, shared_yaxes=True
)
for column, name in enumerate(("v0", "v2"), start=1):
    frame = transfer_stats.query("representation == @name")
    matrix = frame.pivot(index="finger_count", columns="fraction", values="gain")
    robust = frame.pivot(index="finger_count", columns="fraction", values="robust").reindex_like(matrix)
    text = np.where(robust.to_numpy(), "*", "")
    figure.add_trace(
        go.Heatmap(
            z=np.clip(matrix.to_numpy(), -1, 1), x=[f"{v:.0%}" for v in matrix.columns],
            y=matrix.index, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=text, texttemplate="%{text}", showscale=column == 2,
            colorbar={"title": "mean paired<br>gain in R2", "x": 1.02},
            hovertemplate="%{y} fingers, %{x} labels<br>gain=%{z:.4f}<extra></extra>",
        ),
        row=1, col=column,
    )
    figure.update_xaxes(title_text="labeled fraction", row=1, col=column)
figure.update_yaxes(title_text="target finger count", row=1, col=1)
figure.update_layout(
    title="Paired transfer gain against a same-budget specialist (* = robust after BH correction)",
    template="plotly_white", height=520,
)
figure.show()
"""
    ),
    markdown(
        """
### Reading the paired statistics

**The prior helps v2 and harms v0.** Averaged over the twelve non-source
domains, transfer changes held-out R2 by **+0.153** for v2 at 1% labels and by
**-2.147** for v0. Every one of the twelve v0 cells at that budget is robustly
*negative* after correction. Warm-starting from a foundation fitted in a
representation that does not align the domains is worse than starting from
nothing.

**The v2 benefit is real but short-lived.** It is +0.153 at 1%, +0.022 at 2%,
and within 0.002 of zero from 5% onward. The specialist has already reached
0.972 by then, so there is nothing left for a prior to contribute. The
defensible claim from this protocol is a foundation *representation* rather than
a foundation model whose weights keep paying off.
"""
    ),
    markdown(
        """
## 6. Applicability, and the full foundation atlas

Two questions remain from Tutorial 16b. Can similarity tell us when to trust a
foundation without labels, and does the choice of foundation domain matter?
"""
    ),
    code(
        """
records = []
similarity_samples = {}
base_rows = np.flatnonzero(finger_counts == BASE_FINGER_COUNT)
train_rows = np.random.default_rng(SEED).permutation(base_rows)[: int(0.75 * len(base_rows))]
for name in ("v0", "v2"):
    matrix, raw = FEATURES[name], REPRESENTATIONS[name]
    weights = RidgeBank(matrix[train_rows], y[train_rows]).fit()
    error = 100 * np.mean(np.abs(predict(weights, matrix) - y) / np.maximum(np.abs(y), 1e-9), axis=1)
    reference = raw.astype(np.float64)
    centre, scale = reference.mean(axis=0), reference.std(axis=0)
    keep = scale > 1e-8
    standardized = (reference[:, keep] - centre[keep]) / scale[keep]
    unit = standardized / np.maximum(np.linalg.norm(standardized, axis=1, keepdims=True), 1e-12)
    centroid = unit[train_rows].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    similarity = unit @ centroid
    outside = finger_counts != BASE_FINGER_COUNT
    similarity_samples[name] = (similarity, error, outside)
    records.append({
        "representation": name,
        "Spearman(similarity, error)": round(float(spearmanr(similarity[outside], error[outside]).statistic), 4),
        "median zero-shot APE (%)": round(float(np.median(error[outside])), 2),
        "cosine spread": round(float(similarity[outside].std()), 4),
    })
pd.DataFrame(records)
"""
    ),
    code(
        """
ATLAS_PATH = RUN_DIR / "atlas.parquet"
if ATLAS_PATH.exists():
    atlas = pd.read_parquet(ATLAS_PATH)
    print(f"Loaded atlas from {ATLAS_PATH}")
else:
    records = []
    for name in ("v0", "v2"):
        matrix = FEATURES[name]
        for repeat, split in enumerate(splits):
            test, pool = split["test"], split["pool"]
            test_by_count = {c: test[finger_counts[test] == c] for c in COUNTS}
            foundations = {c: RidgeBank(matrix[pool[c]], y[pool[c]]).fit() for c in COUNTS}
            for target in COUNTS:
                rows_test = test_by_count[target]
                for fraction in ATLAS_FRACTIONS:
                    size = max(2, int(round(fraction * POOL_SIZE)))
                    bank = RidgeBank(matrix[pool[target][:size]], y[pool[target][:size]])
                    specialist = macro_scores(y[rows_test], predict(bank.fit(), matrix[rows_test]))["r2"]
                    for source in COUNTS:
                        if source == target:
                            continue
                        weights = bank.fit(foundations[source])
                        records.append({
                            "representation": name, "repeat": repeat, "source": source, "target": target,
                            "fraction": fraction,
                            "r2_transfer": macro_scores(y[rows_test], predict(weights, matrix[rows_test]))["r2"],
                            "r2_specialist": specialist,
                        })
    atlas = pd.DataFrame(records)
    atlas["gain"] = atlas.r2_transfer - atlas.r2_specialist
    atlas.to_parquet(ATLAS_PATH, index=False)
    print(f"Fitted {len(atlas):,} ordered source-target evaluations")

atlas.groupby(["representation", "fraction"]).agg(
    mean_gain=("gain", "mean"), positive_transfer_rate=("gain", lambda v: float((v > 0).mean()))
).round(4)
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Similarity as an applicability gate", "Positive-transfer rate across the 13x13 atlas"],
    horizontal_spacing=0.12,
)
for name in ("v0", "v2"):
    similarity, error, outside = similarity_samples[name]
    rows = np.flatnonzero(outside)
    buckets = pd.qcut(similarity[rows], q=12, duplicates="drop")
    grouped = pd.DataFrame({"s": similarity[rows], "e": error[rows], "b": buckets}).groupby(
        "b", observed=True).agg(s=("s", "mean"), e=("e", "median"))
    figure.add_trace(
        go.Scatter(
            x=grouped["s"], y=grouped["e"], mode="lines+markers", name=name,
            line={"color": PALETTE[name], "width": 3}, marker={"size": 9},
            hovertemplate=f"<b>{name}</b><br>cosine=%{{x:.3f}}<br>median APE=%{{y:.1f}}%<extra></extra>",
        ),
        row=1, col=1,
    )
    frame = atlas.query("representation == @name").groupby("fraction", as_index=False).agg(
        rate=("gain", lambda v: float((v > 0).mean())))
    figure.add_trace(
        go.Bar(
            x=[f"{v:.0%}" for v in frame["fraction"]], y=frame["rate"], name=name,
            marker_color=PALETTE[name], showlegend=False,
            text=[f"{v:.1%}" for v in frame["rate"]], textposition="outside",
            hovertemplate=f"<b>{name}</b><br>%{{x}} labels<br>positive transfer=%{{y:.1%}}<extra></extra>",
        ),
        row=1, col=2,
    )
figure.add_hline(y=0.5, line_dash="dash", line_color="#6B7280", row=1, col=2)
figure.update_xaxes(title_text="cosine similarity to the count-8 foundation", row=1, col=1)
figure.update_yaxes(title_text="median zero-shot APE (%)", row=1, col=1)
figure.update_xaxes(title_text="labeled fraction of the target pool", row=1, col=2)
figure.update_yaxes(title_text="fraction of source-target pairs helped", range=[0, 1.0], row=1, col=2)
figure.update_layout(
    title="v2 makes similarity informative and makes transfer usually helpful",
    template="plotly_white", height=520,
)
figure.show()
"""
    ),
    markdown(
        """
### Reading the atlas

**Similarity becomes a usable gate.** Within this component family v2's cosine
similarity to the foundation centroid ranks zero-shot error at Spearman
**-0.828**, against v0's -0.448, and the median zero-shot error halves from
30.8% to 14.1%. Tutorial 20 showed this signal breaking down *across* component
classes; inside one family, on a balanced cohort, it is strong.

**Foundation choice stops being a gamble.** Over all 156 ordered source-target
pairs at 1% labels, transfer beats a same-budget specialist in **80.5%** of pairs
with v2 and **58.4%** with v0 - and v0's mean gain is -12.35, so its coin-flip
success rate hides catastrophic failures. With v2 the mean gain is +0.004: small,
because the specialist is already good, but no longer dangerous.
"""
    ),
    markdown(CONCLUSIONS),
]

notebook["cells"] = CELLS
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, str(OUTPUT))
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} cells.")
