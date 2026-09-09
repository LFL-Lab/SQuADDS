#!/usr/bin/env python
"""Build the executed, omnibus Capacitance-Operator-v3 tutorial."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("tutorials/Tutorial-23_Capacitance_Operator_v3.ipynb")


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, *, hidden: bool = False):
    cell = nbf.v4.new_code_cell(source.strip())
    if hidden:
        cell["metadata"]["tags"] = ["hide-input"]
        cell["metadata"]["jupyter"] = {"source_hidden": True}
    return cell


cells = [
    md(r"""
# Tutorial 23: Capacitance Operator v3 — from invariant geometry to few-shot physics

This single tutorial is the v3 counterpart of Tutorials 18–22. Each old
tutorial has its own section, but every comparison uses the **same 894-design
per-family cohort**, the same target definitions, and the same deterministic
splits. It is intentionally both a working tutorial and a falsification report.

`capacitance-operator-v3` is an additive experimental API. Nothing in v0, v1,
or v2 is changed. Its 256 coordinates factor geometry into a 240-dimensional
fixed-stack core and 16 explicit stack coordinates. Unlike v2's laboratory-frame
moments, v3 is built from distances, measures, eigenvalues, and vector-buffer
topology. Translation, rotation, and reflection are therefore absent in the
continuum; absolute physical scale is retained in separate coordinates.

The central hypothesis is stronger than “similar polygons have similar
capacitance”: a deterministic low-fidelity Maxwell operator can provide a
common physical coordinate system, while supervised learning only corrects
the discrepancy to the high-fidelity solver.

We report three metrics throughout: R2 in `log1p(C)`, and RMSE plus median
absolute error in fF. R2 tests explained variation; RMSE exposes large misses;
median absolute error describes the typical engineering miss.
"""),
    code(r"""
from pathlib import Path
import json, os, subprocess, sys

REPO = Path.cwd().resolve()
if not (REPO / "squadds").is_dir():
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

import squadds
from squadds.layouts import (
    V3_BLOCK_SLICES, V3_DIMENSIONS, V3_FIXED_STACK_DIMENSIONS,
    V3_METRIC_NAMES, V3_NORMALIZED_DISTANCE_EDGES,
    scale_conditioned_coupling,
)
from squadds.layouts.geometry_v2 import (
    METRIC_BLOCK_SIZE as V2_METRICS,
    COUPLING_BLOCK_SIZE as V2_COUPLING,
    SHAPE_BLOCK_SIZE as V2_SHAPE,
)

pio.renderers.default = "notebook_connected"
pd.set_option("display.max_columns", 30)
SEED = 23
FAMILIES = ["CapNInterdigitalTee", "GeneralizedCapNInterdigital", "TransmonCross"]
SHORT = {"CapNInterdigitalTee":"Tee", "GeneralizedCapNInterdigital":"Generalized", "TransmonCross":"Transmon"}
COLORS = {"CapNInterdigitalTee":"#D1495B", "GeneralizedCapNInterdigital":"#00798C", "TransmonCross":"#6A4C93"}

V3_TABLE = Path(os.getenv("SQUADDS_V3_TABLE", REPO/"tutorials/runtime/tutorial23/capacitance-operator-v3-balanced.parquet"))
V2_TABLE = Path(os.getenv("SQUADDS_V2_ALL_TABLE", Path.home()/".cache/squadds/tutorial20/universal-geometry-v2-portcomplete.parquet"))
if not V3_TABLE.is_file():
    subprocess.run([sys.executable, str(REPO/"scripts/build_v3_balanced_table.py")], check=True, cwd=REPO)
assert V2_TABLE.is_file(), f"Run Tutorial 20 once or set SQUADDS_V2_ALL_TABLE: {V2_TABLE}"
print("implementation:", Path(squadds.__file__).resolve())
print("v3 table:", V3_TABLE, "| v2 table:", V2_TABLE)
"""),
    md(r"""
## A. Tutorial 18 equivalent — build and inspect a deterministic catalogue

The source artifact is the unified port-complete GDS release. Every file uses
the same functional layers: ground `(1,0)`, conductor `(1,10)`, and two ordered
ports `(2,0)/(3,0)`. The build script joins by `design_id`, never by row order or
`layout_id`, then encodes exactly 894 designs per family. The table is cached so
rerunning the analysis does not silently re-fit an embedding.

The v3 schema is deterministic per layout. There is no catalogue PCA, learned
normalization, or selected-frequency fit. A new family can therefore be encoded
without changing any existing row.
"""),
    code(r"""
v3_frame = pd.read_parquet(V3_TABLE).drop_duplicates("design_id")
v2_frame = pd.read_parquet(V2_TABLE).drop_duplicates("design_id")
data = v3_frame.merge(v2_frame[["design_id", "embedding"]], on="design_id", suffixes=("_v3", "_v2"))
data = data.sort_values(["component_name", "design_id"]).reset_index(drop=True)
X3 = np.vstack(data.embedding_v3).astype(np.float64)
X2 = np.vstack(data.embedding_v2).astype(np.float64)
assert X3.shape == (2682, V3_DIMENSIONS)
assert data.component_name.value_counts().eq(894).all()
catalogue = data.groupby("component_name").agg(
    designs=("design_id", "size"),
    mutual_median_fF=("mutual_fF", "median"),
    mutual_min_fF=("mutual_fF", "min"),
    mutual_max_fF=("mutual_fF", "max"),
    ground_median_fF=("ground_sum_fF", "median"),
    scale_median_um=("characteristic_scale_um", "median"),
    proxy_median_fF=("proxy_mutual_fF", "median"),
)
catalogue.round(3)
"""),
    code(r"""
fig = make_subplots(rows=1, cols=2, subplot_titles=("Mutual capacitance", "Physical scale"))
for family in FAMILIES:
    d = data[data.component_name == family]
    fig.add_trace(go.Violin(x=[SHORT[family]]*len(d), y=d.mutual_fF, name=SHORT[family],
        legendgroup=family, line_color=COLORS[family], box_visible=True, points=False,
        hovertemplate="%{y:.3f} fF<extra>"+SHORT[family]+"</extra>"), 1, 1)
    fig.add_trace(go.Violin(x=[SHORT[family]]*len(d), y=d.characteristic_scale_um, name=SHORT[family],
        legendgroup=family, line_color=COLORS[family], box_visible=True, points=False, showlegend=False,
        hovertemplate="%{y:.2f} um<extra>"+SHORT[family]+"</extra>"), 1, 2)
fig.update_layout(title="Balanced cohort: target and scale are both family-shifted", height=430)
fig.update_yaxes(title="fF", row=1, col=1); fig.update_yaxes(title="um", row=1, col=2)
fig.show()
"""),
    md(r"""
### The representation contract

The first five blocks describe the fixed-stack problem. The final environment
block is a declared conditioning variable, not “more geometry.” On a sweep in
which every row has the same stack, use `fixed_stack_core`; when substrate or
metal stack varies, include `environment` and validate a stack-held-out split.
This separation prevents fixed thickness/device-size ratios from becoming a
shortcut for component identity.
"""),
    code(r"""
block_rows = []
for name in ["metrics","interactions","operator","topology","electrostatic","environment"]:
    s = V3_BLOCK_SLICES[name]
    block_rows.append({"block":name, "start":s.start, "stop":s.stop, "dimensions":s.stop-s.start,
                       "contract":"explicit stack" if name=="environment" else "fixed-stack core"})
blocks = pd.DataFrame(block_rows)
display(blocks)
fig = go.Figure(go.Bar(x=blocks.dimensions, y=blocks.block, orientation="h", text=blocks.dimensions,
    marker_color=["#00798C"]*5+["#E9C46A"], customdata=blocks[["start","stop","contract"]],
    hovertemplate="%{y}: [%{customdata[0]}, %{customdata[1]})<br>%{customdata[2]}<extra></extra>"))
fig.update_layout(title="CapOp-v3 is 240 core + 16 explicit environment coordinates", height=390,
                  xaxis_title="coordinates", yaxis_autorange="reversed")
fig.show()
"""),
    md(r"""
## B. Tutorial 19 equivalent — how v3 sees all three component families

Tutorial 19 explained v2 as a six-stage pipeline. The v3 version below applies
the same explanation to a generalized IDC, tee IDC, and transverse-cross qubit.

1. **Functional roles:** two ordered conductors, local ground, ports.
2. **Normalize pose, not physics:** re-origin; use invariant distances and
   covariance eigenvalues; split shape from `sqrt(A0+A1)` scale.
3. **Interaction spectra:** soft histograms of terminal–terminal and
   terminal–ground boundary distances normalized by scale.
4. **Green operator:** singular spectra of pairwise logarithmic-kernel blocks.
5. **Topology:** vector buffer growth curves, with no raster orientation.
6. **Electrostatic mean function:** a small grounded boundary-element solve,
   projected onto a passive Maxwell graph Laplacian.

The next figure is the embedding-level equivalent of the component drawings:
use the dropdown to switch family, hover for bin meaning, and note which
features align despite radically different polygons.
"""),
    code(r"""
representatives = {f: data[data.component_name==f].iloc[len(data[data.component_name==f])//2] for f in FAMILIES}
centres = np.sqrt(V3_NORMALIZED_DISTANCE_EDGES[:-1]*V3_NORMALIZED_DISTANCE_EDGES[1:])
traces=[]
for family in FAMILIES:
    v = np.asarray(representatives[family].embedding_v3)
    start=V3_BLOCK_SLICES["interactions"].start
    for i,label in enumerate(["terminal 0 ↔ terminal 1", "terminal 0 ↔ ground", "terminal 1 ↔ ground"]):
        traces.append(go.Scatter(x=centres, y=v[start+32*i:start+32*(i+1)], mode="lines+markers",
            name=label, visible=(family==FAMILIES[0]), legendgroup=label,
            hovertemplate="d/scale=%{x:.4g}<br>soft mass=%{y:.4f}<extra>"+SHORT[family]+"</extra>"))
buttons=[]
for fi,family in enumerate(FAMILIES):
    visible=[False]*len(traces)
    for j in range(3): visible[3*fi+j]=True
    buttons.append(dict(label=SHORT[family], method="update", args=[{"visible":visible},{"title":"Normalized boundary interactions — "+family}]))
fig=go.Figure(traces)
fig.update_layout(title="Normalized boundary interactions — "+FAMILIES[0], height=470,
    xaxis_type="log", xaxis_title="distance / characteristic scale", yaxis_title="soft probability mass",
    updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right")])
fig.show()
"""),
    code(r"""
fig = make_subplots(rows=1, cols=3, subplot_titles=("Green-operator spectrum", "Topology growth", "Maxwell proxy"))
for family in FAMILIES:
    v=np.asarray(representatives[family].embedding_v3)
    op=v[V3_BLOCK_SLICES["operator"]]
    top=v[V3_BLOCK_SLICES["topology"]]
    fig.add_trace(go.Scatter(x=np.arange(16), y=op[:16], mode="lines+markers", name=SHORT[family],
        line_color=COLORS[family], hovertemplate="mode=%{x}<br>value=%{y:.4g}<extra>"+SHORT[family]+"</extra>"),1,1)
    fig.add_trace(go.Scatter(x=np.arange(16), y=top[:16], mode="lines+markers", name=SHORT[family],
        showlegend=False, line_color=COLORS[family], hovertemplate="radius index=%{x}<br>growth=%{y:.4g}<extra>"+SHORT[family]+"</extra>"),1,2)
    d=data[data.component_name==family]
    fig.add_trace(go.Scatter(x=d.proxy_mutual_fF, y=d.mutual_fF, mode="markers", name=SHORT[family],
        showlegend=False, marker=dict(color=COLORS[family], size=5, opacity=.45),
        customdata=d.design_id, hovertemplate="proxy=%{x:.2f} fF<br>solver=%{y:.2f} fF<br>%{customdata}<extra>"+SHORT[family]+"</extra>"),1,3)
fig.update_xaxes(title="singular-value index", row=1,col=1); fig.update_yaxes(type="log", row=1,col=1)
fig.update_xaxes(title="buffer-radius index", row=1,col=2)
fig.update_xaxes(title="low-fidelity proxy (fF)", type="log", row=1,col=3); fig.update_yaxes(title="solver target (fF)", type="log", row=1,col=3)
fig.update_layout(title="Stages 4–6: operator, topology, and electrostatic correction problem", height=450)
fig.show()
"""),
    md(r"""
### Is v2 rotation/translation invariant?

Not by construction. v2 re-origins geometry and many of its distance spectra
are invariant, so translation is largely removed. But it also stores
laboratory-axis moments and raster-derived shape coefficients; an arbitrary
rotation can move those coordinates. v3 replaces them with eigenvalues,
distances, singular values, and vector topology. Finite quadrature introduces
small numerical tolerance at arbitrary angles, but right-angle rotations and
translations are exact in the contract tests.

This does **not** mean scale invariant. Capacitance changes with device scale,
so v3 deliberately stores normalized shape and absolute scale separately.
"""),
    code(r"""
invariance = pd.DataFrame([
    ["translation", "mostly (re-origin)", "yes", "must be nuisance"],
    ["90° rotation", "mixed; axis/raster channels move", "yes", "must be nuisance"],
    ["arbitrary rotation", "mixed; axis/raster channels move", "within quadrature tolerance", "must be nuisance"],
    ["reflection", "not a global contract", "yes", "electrostatics is parity-even"],
    ["uniform scale", "retained but entangled", "retained and factorized", "physical signal"],
], columns=["transformation","v2","v3","reason"])
invariance
"""),
    md(r"""
## C. Tutorial 20 equivalent — strict held-out-family and few-shot transfer

We fit one identical nonlinear random-feature ridge model for every
representation. The scaler and random Fourier map are fit using training rows
only. For the zero-shot test, an entire component family is absent from
training. This is much harder than a random row split and is the correct test
of whether the representation unifies component classes.

Four representations are compared:

- stock v2;
- v2 with its published scale-conditioned coupling transform;
- v3 fixed-stack core (recommended for this fixed-stack cohort);
- complete v3 including explicit environment coordinates (a deliberate
  shortcut-control experiment).
"""),
    code(r"""
class RFFRidge:
    def __init__(self, seed=SEED, features=128, alpha=.3):
        self.seed, self.features, self.alpha = seed, features, alpha
    def fit(self, X, y):
        self.mean=np.mean(X,axis=0); self.scale=np.std(X,axis=0)
        self.scale=np.where(self.scale>1e-6,self.scale,1.0); z=np.clip((X-self.mean)/self.scale,-8,8)
        rng=np.random.default_rng(self.seed)
        sample=z[rng.choice(len(z),size=min(1000,len(z)),replace=False)]; pairs=len(sample)//2
        median_d2=np.median(np.sum((sample[:pairs]-sample[pairs:2*pairs])**2,axis=1)) if pairs else 1.0
        gamma=1/max(2*median_d2,1e-12)
        self.W=rng.normal(scale=np.sqrt(2*gamma), size=(z.shape[1],self.features))
        self.b=rng.uniform(0,2*np.pi,size=self.features)
        phi=np.c_[z,self._map(z)]; design=np.c_[np.ones(len(phi)),phi]
        penalty=np.eye(design.shape[1])*self.alpha; penalty[0,0]=0
        self.coefficients=np.linalg.solve(design.T@design+penalty,design.T@y); return self
    def _map(self,z): return np.sqrt(2/self.features)*np.cos(z@self.W+self.b)
    def predict(self,X):
        z=np.clip((X-self.mean)/self.scale,-8,8); phi=np.c_[z,self._map(z)]
        return np.c_[np.ones(len(phi)),phi]@self.coefficients

def r2_score(actual,predicted):
    denominator=np.sum((actual-np.mean(actual))**2)
    return 1-np.sum((actual-predicted)**2)/denominator

REPRESENTATIONS = {
    "v2": X2,
    "v2 + conditioned scale": scale_conditioned_coupling(X2),
    "v3 fixed-stack core": X3[:, V3_BLOCK_SLICES["fixed_stack_core"]],
    "v3 complete": X3,
}
TARGETS={"mutual":data.mutual_fF.to_numpy(float), "ground sum":data.ground_sum_fF.to_numpy(float)}

def physical_scores(actual_fF, predicted_log):
    predicted=np.maximum(np.expm1(predicted_log),0)
    return {"R2_log":r2_score(np.log1p(actual_fF),predicted_log),
            "RMSE_fF":np.sqrt(np.mean((actual_fF-predicted)**2)),
            "MedAE_fF":np.median(np.abs(actual_fF-predicted))}

def heldout_results():
    rows=[]
    labels=data.component_name.to_numpy()
    for rep,X in REPRESENTATIONS.items():
        for held in FAMILIES:
            train=labels!=held; test=~train
            for target,y in TARGETS.items():
                pred=RFFRidge().fit(X[train],np.log1p(y[train])).predict(X[test])
                rows.append({"representation":rep,"held_out":SHORT[held],"target":target,**physical_scores(y[test],pred)})
    return pd.DataFrame(rows)

heldout=heldout_results()
heldout.round(3)
"""),
    code(r"""
metric_specs=[("R2_log","R2 on log1p(C)"),("RMSE_fF","RMSE (fF)"),("MedAE_fF","median absolute error (fF)")]
fig=make_subplots(rows=1,cols=3,subplot_titles=[s[1] for s in metric_specs])
palette={"v2":"#8D99AE","v2 + conditioned scale":"#F4A261","v3 fixed-stack core":"#00798C","v3 complete":"#6A4C93"}
for col,(metric,title) in enumerate(metric_specs,1):
    view=heldout[heldout.target=="mutual"]
    for rep in REPRESENTATIONS:
        d=view[view.representation==rep]
        fig.add_trace(go.Bar(x=d.held_out,y=d[metric],name=rep,legendgroup=rep,showlegend=(col==1),
            marker_color=palette[rep],customdata=np.c_[d.RMSE_fF,d.MedAE_fF],
            hovertemplate="held out=%{x}<br>value=%{y:.3f}<br>RMSE=%{customdata[0]:.2f}<br>MedAE=%{customdata[1]:.2f}<extra>"+rep+"</extra>"),1,col)
fig.update_layout(barmode="group",title="Zero-shot mutual capacitance: an entire geometry family is unseen",height=470)
fig.add_hline(y=0,line_dash="dot",row=1,col=1)
fig.show()
"""),
    code(r"""
heldout.groupby(["representation","target"])[["R2_log","RMSE_fF","MedAE_fF"]].mean().round(3)
"""),
    md(r"""
### Reading the zero-shot result

The fixed-stack v3 core is the only tested representation with positive mutual-
capacitance R2 in every rotation: **0.970 Tee, 0.639 Generalized, and 0.579
Transmon**. Stock v2 reaches 0.557 and 0.874 on the two couplers but collapses
to -3.162 on the unseen transmon. v3's typical physical errors are **0.98,
0.70, and 1.10 fF** respectively. This is the main v3 result.

There are two cautions. First, v2 remains better than v3 on held-out Generalized
IDC, so v3 has not Pareto-dominated v2 on every class. Its improvement is the
hard Transmon rotation and the all-positive worst case. Second, `v3 complete`
falls to -0.774 and -3.722 on Generalized and Transmon. Because the nominal
stack is fixed, thickness/scale ratios distinguish families without providing
a genuinely sampled stack response. The separation between core and
environment is therefore part of the scientific contract, not an after-the-fact
ablation.
"""),
    md(r"""
### Few-shot adaptation

Zero-shot is the scientific stress test; few-shot is the practical use case.
For each held-out family we reserve a deterministic 30% evaluation set, then
add 0, 4, 8, 16, 32, or 64 labeled target designs to all source-family rows.
Curves show the mean and standard deviation over four target draws.
"""),
    code(r"""
def learning_curves():
    labels=data.component_name.to_numpy(); rows=[]
    budgets=[0,4,8,16,32,64]
    for held_i,held in enumerate(FAMILIES):
        target_idx=np.flatnonzero(labels==held); source_idx=np.flatnonzero(labels!=held)
        base=np.random.default_rng(SEED+held_i).permutation(target_idx)
        evaluation=base[:268]; pool=base[268:]
        for rep,X in REPRESENTATIONS.items():
            for budget in budgets:
                for repeat in range(4):
                    chosen=np.random.default_rng(SEED+100*held_i+10*budget+repeat).choice(pool,size=budget,replace=False) if budget else np.array([],int)
                    train=np.r_[source_idx,chosen]; y=TARGETS["mutual"]
                    pred=RFFRidge(seed=SEED+repeat).fit(X[train],np.log1p(y[train])).predict(X[evaluation])
                    rows.append({"held_out":SHORT[held],"representation":rep,"labels":budget,"repeat":repeat,
                                 **physical_scores(y[evaluation],pred)})
    return pd.DataFrame(rows)

curves=learning_curves()
curve_summary=curves.groupby(["held_out","representation","labels"]).agg(
    R2=("R2_log","mean"), R2_sd=("R2_log","std"), RMSE=("RMSE_fF","mean"), MedAE=("MedAE_fF","mean")).reset_index()
curve_summary.round(3)
"""),
    code(r"""
fig=make_subplots(rows=1,cols=3,subplot_titles=["held out: "+SHORT[f] for f in FAMILIES],shared_yaxes=True)
for col,family in enumerate(FAMILIES,1):
    for rep in REPRESENTATIONS:
        d=curve_summary[(curve_summary.held_out==SHORT[family])&(curve_summary.representation==rep)]
        fig.add_trace(go.Scatter(x=d.labels,y=d.R2,error_y=dict(type="data",array=d.R2_sd,visible=True),
            mode="lines+markers",name=rep,legendgroup=rep,showlegend=(col==1),line_color=palette[rep],
            customdata=np.c_[d.RMSE,d.MedAE],hovertemplate="labels=%{x}<br>R2=%{y:.3f}<br>RMSE=%{customdata[0]:.2f} fF<br>MedAE=%{customdata[1]:.2f} fF<extra>"+rep+"</extra>"),1,col)
    fig.add_hline(y=0,line_dash="dot",row=1,col=col)
fig.update_layout(title="Few-shot mutual-capacitance adaptation",height=460,xaxis_title="target-family labels")
fig.update_yaxes(title="R2 on log1p(C)",row=1,col=1); fig.show()
"""),
    code(r"""
curve_summary[curve_summary.labels.isin([0,8,16,64])].pivot_table(
    index=["held_out","labels"], columns="representation", values="R2"
).round(3)
"""),
    md(r"""
The practical result is as important as zero-shot transfer. On the hard
Transmon rotation, the v3 core rises from 0.579 on the complete held-out set to
about 0.84 with eight labeled target designs, 0.92 with sixteen, and 0.98 with
64. Report the curve, not a cherry-picked budget: label efficiency is the
publishable endpoint, and uncertainty across target draws is visible in the
error bars.
"""),
    md(r"""
## D. Tutorial 21 equivalent — balanced domains and causal block ablations

All domains are already balanced, so a macro mean gives each component class
equal weight. Ablation is performed on the strict held-out-family task, not an
easy within-family random split. This tells us whether a block carries
class-independent physics or merely identifies shapes seen during training.

`fixed-stack core` is the union of metrics, interactions, Green operator,
topology, and electrostatic proxy. The environment-only result is included as a
negative control. A constant stack should not predict capacitance.
"""),
    code(r"""
ABLATIONS={name:X3[:,s] for name,s in V3_BLOCK_SLICES.items() if name in
           ["metrics","interactions","operator","topology","electrostatic","environment","fixed_stack_core"]}
def ablation_results():
    rows=[]; labels=data.component_name.to_numpy(); y=TARGETS["mutual"]
    for block,X in ABLATIONS.items():
        for held in FAMILIES:
            train=labels!=held; test=~train
            pred=RFFRidge().fit(X[train],np.log1p(y[train])).predict(X[test])
            rows.append({"block":block,"held_out":SHORT[held],**physical_scores(y[test],pred)})
    return pd.DataFrame(rows)
ablations=ablation_results()
ablations.pivot(index="block",columns="held_out",values="R2_log").assign(macro=lambda x:x.mean(axis=1)).sort_values("macro",ascending=False).round(3)
"""),
    code(r"""
pivot=ablations.pivot(index="block",columns="held_out",values="R2_log").reindex(columns=[SHORT[f] for f in FAMILIES])
fig=go.Figure(go.Heatmap(z=pivot.to_numpy(),x=pivot.columns,y=pivot.index,zmid=0,colorscale="RdBu",
    text=np.round(pivot.to_numpy(),2),texttemplate="%{text}",hovertemplate="block=%{y}<br>held out=%{x}<br>R2=%{z:.3f}<extra></extra>"))
fig.update_layout(title="Block ablation: zero-shot mutual-capacitance R2",height=490,xaxis_title="unseen family")
fig.show()
"""),
    md(r"""
No single “physics-looking” block explains the result. Invariant scalar metrics
alone are strong (macro R2 0.646), but the combined core reaches 0.730. The
electrostatic block alone is badly miscalibrated across classes even though it
is useful jointly; this is the signature of a mean function needing a
discrepancy model. Interaction and operator blocks also help jointly more than
alone. That non-additivity is why within-family feature importance would be a
misleading explanation of cross-family transfer.
"""),
    md(r"""
### Does the deterministic electrostatic proxy already equal capacitance?

No—and it should not be presented that way. It is a scale-lifted 2D boundary
model, while the labels come from finite 3D simulations and historical solver
setups. A useful mean function may be monotonic within a family while carrying
family-dependent calibration. We therefore inspect rank correlation and the
ratio explicitly rather than hiding calibration inside the embedding.
"""),
    code(r"""
proxy_rows=[]
for family in FAMILIES:
    d=data[data.component_name==family]
    proxy_rows.append({"family":SHORT[family],"Spearman_mutual":spearmanr(d.proxy_mutual_fF,d.mutual_fF).statistic,
                       "median_solver_over_proxy":np.median(d.mutual_fF/d.proxy_mutual_fF),
                       "Spearman_ground":spearmanr(d.proxy_ground_sum_fF,d.ground_sum_fF).statistic})
proxy_report=pd.DataFrame(proxy_rows)
proxy_report.round(3)
"""),
    md(r"""
## E. Tutorial 22 equivalent — similarity, neighbors, and capacitance meaning

Cosine similarity is not physics until a metric and a target are declared. We
standardize the v3 fixed-stack core over the balanced catalogue, normalize each
row, then retrieve the nearest design **from another family**. This asks the
hard question: when v3 calls two unlike component classes close, how close are
their capacitances?

The scatter below shows every design's nearest cross-family neighbor. Points on
the diagonal are successful physical analogies; confident off-diagonal points
are counterexamples that a future metric must learn from.
"""),
    code(r"""
core=REPRESENTATIONS["v3 fixed-stack core"]
Z=(core-core.mean(axis=0))/np.where(core.std(axis=0)>1e-12,core.std(axis=0),1.0)
Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
labels=data.component_name.to_numpy(); nearest=np.empty(len(data),int); similarity=np.empty(len(data))
for i in range(len(data)):
    candidates=np.flatnonzero(labels!=labels[i]); scores=Z[candidates]@Z[i]
    j=np.argmax(scores); nearest[i]=candidates[j]; similarity[i]=scores[j]
neighbor=pd.DataFrame({"family":labels,"source_C":data.mutual_fF,"neighbor_C":data.mutual_fF.to_numpy()[nearest],
    "cosine":similarity,"source_id":data.design_id,"neighbor_id":data.design_id.to_numpy()[nearest],
    "neighbor_family":labels[nearest]})
neighbor["absolute_error_fF"]=(neighbor.source_C-neighbor.neighbor_C).abs()
neighbor["relative_error"]=neighbor.absolute_error_fF/neighbor.source_C.clip(lower=.1)
neighbor.groupby("family")[["cosine","absolute_error_fF","relative_error"]].median().round(3)
"""),
    code(r"""
fig=go.Figure()
for family in FAMILIES:
    d=neighbor[neighbor.family==family]
    fig.add_trace(go.Scatter(x=d.source_C,y=d.neighbor_C,mode="markers",name=SHORT[family],
        marker=dict(color=COLORS[family],size=5,opacity=.5,colorbar=dict(title="cosine")),
        customdata=np.c_[d.cosine,d.neighbor_family,d.source_id,d.neighbor_id],
        hovertemplate="query C=%{x:.2f} fF<br>neighbor C=%{y:.2f} fF<br>cosine=%{customdata[0]:.3f}<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}<extra>"+SHORT[family]+"</extra>"))
limits=[max(neighbor.source_C.min(),.05),max(neighbor.source_C.max(),neighbor.neighbor_C.max())]
fig.add_trace(go.Scatter(x=limits,y=limits,mode="lines",name="equal capacitance",line=dict(color="black",dash="dash")))
fig.update_layout(title="Nearest cross-family geometry: similarity is useful but not equivalence",height=520,
                  xaxis_title="query mutual C (fF)",yaxis_title="neighbor mutual C (fF)")
fig.update_xaxes(type="log");fig.update_yaxes(type="log");fig.show()
"""),
    code(r"""
quantiles=pd.qcut(neighbor.cosine,5,duplicates="drop")
calibration=neighbor.assign(similarity_bin=quantiles).groupby("similarity_bin",observed=True).agg(
    median_cosine=("cosine","median"), median_abs_error_fF=("absolute_error_fF","median"),
    median_relative_error=("relative_error","median"), pairs=("cosine","size")).reset_index()
fig=make_subplots(rows=1,cols=2,subplot_titles=("Absolute error calibration","Relative error calibration"))
fig.add_trace(go.Scatter(x=calibration.median_cosine,y=calibration.median_abs_error_fF,mode="lines+markers",
    customdata=calibration.pairs,hovertemplate="cosine=%{x:.3f}<br>median error=%{y:.2f} fF<br>pairs=%{customdata}<extra></extra>"),1,1)
fig.add_trace(go.Scatter(x=calibration.median_cosine,y=calibration.median_relative_error,mode="lines+markers",
    customdata=calibration.pairs,hovertemplate="cosine=%{x:.3f}<br>median relative error=%{y:.1%}<br>pairs=%{customdata}<extra></extra>"),1,2)
fig.update_xaxes(title="median cross-family cosine");fig.update_yaxes(title="fF",row=1,col=1);fig.update_yaxes(title="fraction",row=1,col=2)
fig.update_layout(title="Empirical meaning of v3 similarity",height=430);fig.show()
"""),
    md(r"""
The neighbor audit identifies the remaining geometry gap. Median cross-family
cosine is roughly 0.51–0.54 for the two capacitor families, with median relative
capacitance error around 34–38%. Transmon's median cosine is only 0.19 and its
nearest cross-family capacitance error is about 179%. Thus the regression result
does not mean raw nearest-neighbor substitution is safe: v3 supplies transferable
coordinates to a learned head, while a calibrated retrieval metric remains a
separate research problem.
"""),
    md(r"""
## F. Conclusions, limits, and the next publishable experiment

### What this run establishes

- v3 is an additive 256-coordinate representation; all v0–v2 APIs and artifacts
  remain untouched.
- Its geometry core has an explicit rigid-motion contract. Translation and
  right-angle rotation are exact in tests; arbitrary rotation is stable to
  finite-quadrature tolerance; scale is intentionally retained and factorized.
- On this balanced, fixed-stack cohort, the **240-coordinate fixed-stack core**
  is the meaningful primary endpoint. It is compared against stock v2 using
  identical rows, models, targets, and held-out-family splits.
- The complete environment-conditioned vector is not automatically better.
  Fixed stack/device-size ratios can expose component identity and hurt blind
  extrapolation. Keeping this negative control is scientifically more useful
  than tuning it away.
- The deterministic Maxwell proxy is a low-fidelity mean function, not a
  surrogate label. Its family-dependent calibration quantifies the remaining
  2D-to-3D discrepancy.
- A high cosine is a retrieval hypothesis, not a guarantee of equal
  capacitance. The neighbor calibration plot converts that slogan into a
  measurable error curve.

### What is not established

- These historical labels do not isolate representation error from solver,
  substrate-stack, boundary-condition, and meshing differences among datasets.
- Three electrostatic families are not enough to claim universal transfer.
- A single fixed stack cannot validate the 16 environmental coordinates.
- The 2D boundary operator cannot capture finite-thickness, fringing-depth, or
  package effects without a learned or simulated discrepancy correction.

### Recommended next acquisition sweep

Run a nested, factorial sweep with identical solver settings and exported GDS:

1. **Shape holdout:** at least two new capacitor topologies (parallel plate/pad,
   concentric or meander) plus the existing three families.
2. **Pose controls:** duplicate 50–100 designs at random translations,
   reflections, and rotations; capacitance should be invariant within solver
   noise and the v3 core should be numerically unchanged.
3. **Scale controls:** uniformly scale 50 base layouts over roughly
   `0.5×, 0.7×, 1×, 1.4×, 2×` while holding normalized geometry fixed.
4. **Gap/overlap axes:** sweep terminal gap and facing boundary length
   independently, including matched-capacitance paths through geometry space.
5. **Stack axes:** cross a smaller geometry subset with substrate permittivity,
   substrate thickness, and metal thickness. Hold out entire stack values when
   validating the environment block.
6. **Fidelity ladder:** compute the deterministic v3 proxy, one standardized 2D
   electrostatic solve, and the same converged 3D solve for every selected row.

The publishable model is then

`C_high fidelity = C_v3 operator + learned discrepancy(geometry core, stack)`,

evaluated with nested leave-one-shape-family-out and leave-one-stack-out splits,
few-shot learning curves, uncertainty coverage, R2, RMSE, and median absolute
error. The strongest claim would not be a new embedding alone; it would be a
demonstrated reduction in labels required to reach a predeclared fF tolerance
on an unseen topology.
"""),
    code(r"""
headline = heldout[(heldout.target=="mutual")].pivot(index="held_out",columns="representation",values="R2_log")
print("Zero-shot mutual-capacitance R2")
display(headline.round(3))
print("\nFixed-stack v3 core engineering errors")
display(heldout[(heldout.target=="mutual") & (heldout.representation=="v3 fixed-stack core")]
        [["held_out","RMSE_fF","MedAE_fF"]].round(3))
print("\nNo repositories or dataset hubs are modified by this tutorial.")
"""),
]

# Drop the intentional empty placeholder used to keep neighboring visual cells readable.
cells = [cell for cell in cells if not (cell.cell_type == "code" and not cell.source.strip())]
notebook = nbf.v4.new_notebook(cells=cells)
notebook["metadata"] = {
    "kernelspec": {"display_name": "SQuADDS Tutorial (uv)", "language": "python", "name": "squadds-tutorial"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"wrote {OUTPUT} ({len(cells)} cells)")
