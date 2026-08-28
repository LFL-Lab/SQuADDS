#!/usr/bin/env python
"""Build Tutorial 20: cross-class transfer with universal-geometry-v2."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUTPUT = Path("tutorials/Tutorial-20_Cross_Class_Transfer_with_v2.ipynb")


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


TRANSFER_READING = """
### Reading the transfer curves

**The representation, not the transfer mechanism, is doing the work.** At 2% of
the `CapNInterdigitalTee` pool - about 13 labeled designs - a v2 model trained
only on those 13 reaches macro R2 0.977, against 0.782 for v0 and 0.599 for the
manually aligned parameter baseline. With the full pool the ordering holds:
0.9999, 0.993, 0.810.

**The source prior adds little once the representation is good.** Transfer beats
target-only by 0.985 against 0.977 at the smallest budget and the two are
indistinguishable from 10% onward. This is the opposite of Tutorial 17's
picture, and the reason is not that transfer got worse but that the target task
got easy: thirteen labels are already enough to fit a v2 head. When someone asks
whether we have a "foundation model", the honest answer from this experiment is
that we have a foundation *representation*, and the pretrained weights on top of
it are close to redundant.

**Zero-shot still fails, for every representation.** The best cross-class
zero-shot score anywhere in the table is v2's -1.38, and negative R2 means worse
than predicting the mean. A model fit on interdigital capacitors does not know
the capacitance of a transmon it has never seen. What v2 buys is that a handful
of labels closes the gap, not that no labels are needed.
"""

HELDOUT_READING = """
### Reading the held-out-class result

This is the strictest test in the SQuADDS tutorials, and the numbers should be
read with its asymmetry in mind: holding out `GeneralizedCapNInterdigital` means
training on 2,827 rows and predicting 13,683, while holding out
`CapNInterdigitalTee` means training on 15,616 and predicting 894.

**v2 is the only representation that ever generalizes to an unseen component
class.** Holding out the Generalized NCap family entirely, a model trained only
on Tee couplers and transmons reaches macro R2 **+0.440** on it, with no labels
from that family at all. Every other representation is negative in every
rotation.

**It does not work in the other two rotations.** Holding out
`CapNInterdigitalTee` gives -0.117, essentially the accuracy of predicting the
mean, and holding out `TransmonCross` gives -1.341. So the correct claim is that
v2 sometimes crosses the class boundary with zero labels and the other
representations never do - not that the problem is solved.

Even where v2 is negative it is negative by one to two orders of magnitude less
than the alternatives: -0.117 against v0's -48.2, and -1.341 against v0's -13.1
and the parameter baseline's -143.6.

**The parameter block earns its place here, and only here.** Tutorial 18 found
that stripping v2's 96 parameter coordinates cost almost nothing within one
class. Across classes it is decisive: geometry-only falls to -8.83 and -37.2
where full v2 holds -0.117 and -1.341. The dimension-typed order statistics -
the smallest length in the design, the largest, the count of each dimension
class - are comparable across schemas in a way the raw geometry blocks alone
apparently are not, and they pin down the absolute size regime of an unfamiliar
device.
"""

RESIDUAL_READING = """
### The residual idea does not work, and the reason is instructive

The hypothesis was that subtracting a crude electrostatic estimate would leave a
smoother, more class-independent quantity. It fails, decisively and in every
rotation: holding out `TransmonCross`, v2 goes from -1.34 predicting log C to
**-44.6** predicting the residual.

The diagnostic is in the correlations printed above. Within each class the proxy
tracks the simulated capacitance extremely well - Spearman +0.920, +0.954, and
+0.987 - but pooled across all three it drops to +0.872. That gap is the whole
story. The proxy is a two-dimensional capacitance per unit length; each class
has its own characteristic depth and scale, so the proxy carries a
**class-dependent offset**. Subtracting it does not remove common structure, it
injects exactly the between-class variation the model is trying to bridge.

A proxy is a good feature and a bad denominator. Left in the input vector, where
v2 puts it, the model is free to use it and to learn the per-class offset.
Subtracted from the target, that offset becomes irreducible error. If this idea
is worth another attempt it needs a proxy that is dimensionally complete - a
three-dimensional solve with the real layer stack - rather than a per-unit-length
stand-in.
"""

CONCLUSIONS = """
## 10. What this experiment establishes

**Established**

- Three SQuADDS families share a mutual-capacitance target, not two. The
  three-class study this enables was previously assumed impossible.
- Across those three classes the design-tool vocabularies intersect in exactly
  one name, `orientation`, a placement angle. A parameter-schema baseline for a
  three-class model does not exist, so a geometry-derived contract is not merely
  better here, it is the only option.
- v2 transforms cross-class label efficiency. Thirteen labeled Tee couplers take
  a v2 model to macro R2 0.977 where v0 reaches 0.782 and the aligned parameter
  baseline reaches 0.599.
- **v2 generalizes to a completely unseen component class in two of three
  rotations** once class size is equalized, reaching +0.859 on held-out
  Generalized NCaps and +0.422 on held-out Tee couplers. No other representation
  is positive anywhere.
- **A new component family needs roughly ten labels.** Five labeled designs from
  an unseen class reach 0.887 and ten reach 0.943, against fifty to a hundred for
  v0.
- A source prior helps only when the representation aligns the classes. At five
  labels adaptation lifts v2 from 0.610 to 0.887 and pushes v0 from 0.261 down to
  -0.502.
- v2's similarity is a much better applicability signal than v0's inside a class,
  reaching Spearman -0.718 on Tee couplers against v0's -0.278.

**Not established, and worth stating plainly**

- Zero-shot cross-class prediction is still not reliable. Two rotations are
  positive; `TransmonCross` stays at -1.891 balanced. The qubit class is
  genuinely harder than the two coupler families, and nothing here fixes it.
- The physics-proxy residual target, proposed as a way to give every class one
  comparable quantity, is worse than predicting capacitance directly in every
  rotation.
- Block ablation run only in-class is misleading. Every v2 block predicts
  capacitance inside a single family at macro R2 0.987 or better, so that setting
  cannot distinguish them; cross-class the same blocks span 0.960 to -3.53. The
  shape spectrum reverses outright, going from one of the best in-class blocks to
  by far the worst across classes.
- Cross-class similarity is not uniformly trustworthy. v2 fixes the NCap-to-NCap
  pair, turning v0's misleading +0.321 into a useful -0.430, but both
  representations have the wrong sign on pairs involving the qubit class. Ranking
  an unfamiliar design against our catalogue is safe within a component family
  and not yet safe across one.
- Only one transfer direction was tested for the curves in section 4, always out
  of the Generalized NCap family, because it is the only family large enough to
  serve as a foundation on the full catalogue.
- All three classes remain electrostatic. Nothing here shows that a capacitance
  representation transfers to the eigenmode quantities `CavityClawRouteMeander`
  reports, which is the next boundary worth crossing.
- The balanced cohort uses 894 designs per class, so it is a smaller and noisier
  experiment than the full catalogue even though it is a fairer one. Both are
  reported above rather than choosing whichever looks better.

**What we would do next**

Two things, in order. First, attack the applicability failure, because the
routing workflow depends on it: fit and publish the frozen whitening transform
that Tutorials 18 and 19 both flagged, then re-measure the per-pair similarity
correlations under it. A metric learned across all four families is the obvious
candidate for making the qubit boundary behave like the NCap one already does.

Second, work out why `TransmonCross` resists. It is the one class whose two
terminals are a large cross and a small claw at very different scales, and it is
the one class where both the held-out prediction and the similarity metric fail.
That is a specific, checkable hypothesis about terminal-scale asymmetry rather
than a general shrug.
"""


BALANCED_INTRO = """
## 6. A balanced cohort: removing class size as a confound

The held-out-class table above has a problem that Tutorial 16b already taught us
to take seriously. The three classes are wildly unequal - 13,683 Generalized
NCaps against 1,933 transmons and 894 Tee couplers - so each rotation changes
*two* things at once: which class is unseen, and how much data the model has.
Holding out the Generalized family trains on 2,827 rows and tests on 13,683;
holding out the Tee couplers trains on 15,616 and tests on 894.

Following Tutorial 16b, we deterministically cut every class to the size of the
smallest, **894 designs each**, and repeat the analysis. Every rotation now
trains on exactly 2 x 894 rows and tests on exactly 894. The only scientific
variable that changes between rotations is which component class is unseen.
"""

BALANCED_CODE = """
BALANCED_PER_CLASS = int(data.component_name.value_counts().min())
EXPECTED_BALANCED_ROWS = len(CLASSES) * BALANCED_PER_CLASS

picked = []
for component in sorted(CLASSES):
    rows = data.index[data.component_name == component].to_numpy()
    ordered = rows[np.argsort(data.loc[rows, "design_id"].to_numpy())]
    picked.append(np.random.default_rng(SEED).permutation(ordered)[:BALANCED_PER_CLASS])
cohort = np.sort(np.concatenate(picked))
assert len(cohort) == EXPECTED_BALANCED_ROWS

balanced = data.iloc[cohort].reset_index(drop=True)
balanced_v2 = v2[cohort]
balanced_v0 = v0[cohort]
balanced_y = balanced[TARGETS].to_numpy(float)
balanced_components = balanced.component_name.to_numpy()
BALANCED_REPRESENTATIONS = {
    "v0": balanced_v0,
    "v2": balanced_v2,
    "v2 geometry only": balanced_v2[:, GEOMETRY_COLUMNS],
}

print(f"balanced cohort: {BALANCED_PER_CLASS} designs per class, {EXPECTED_BALANCED_ROWS} rows total")
print(balanced.component_name.value_counts().to_string())


def balanced_held_out(name, matrix):
    records = []
    for held in sorted(CLASSES):
        train = np.flatnonzero(balanced_components != held)
        test = np.flatnonzero(balanced_components == held)
        features = project(matrix, train)
        model = TransferRidgeRegressor(ALPHA).fit(features[train], balanced_y[train])
        scores = macro(balanced_y[test], model.predict(features[test]))
        records.append(
            {"representation": name, "held_out_class": held, "train_rows": len(train),
             "test_rows": len(test), "r2": scores["r2"], "mae": scores["mae"]}
        )
    return pd.DataFrame(records)


BALANCED_HELD_PATH = RUN_DIR / "balanced_held_out.parquet"
if BALANCED_HELD_PATH.exists():
    balanced_held = pd.read_parquet(BALANCED_HELD_PATH)
else:
    balanced_held = pd.concat(
        [balanced_held_out(name, matrix) for name, matrix in BALANCED_REPRESENTATIONS.items()],
        ignore_index=True,
    )
    balanced_held.to_parquet(BALANCED_HELD_PATH, index=False)

comparison = (
    balanced_held.pivot(index="held_out_class", columns="representation", values="r2")
    .add_suffix(" (balanced)")
    .join(held.pivot(index="held_out_class", columns="representation", values="r2").add_suffix(" (full)"))
)
comparison[[c for c in comparison.columns if c.startswith("v0") or c.startswith("v2 (")]].round(4)
"""

BALANCED_FIGURE = """
# %% hide input
order = sorted(CLASSES)
figure = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Full catalogue (unequal class sizes)", "Balanced cohort (894 designs each)"],
    horizontal_spacing=0.10, shared_yaxes=True,
)
for column, frame in enumerate((held, balanced_held), start=1):
    for name in ("v0", "v2 geometry only", "v2"):
        rows = frame.query("representation == @name").set_index("held_out_class").reindex(order)
        figure.add_trace(
            go.Bar(
                x=order, y=np.clip(rows["r2"].to_numpy(), -3, None), name=name,
                marker_color=PALETTE[name], legendgroup=name, showlegend=column == 1,
                text=[f"{value:.2f}" for value in rows["r2"].to_numpy()], textposition="outside",
                hovertemplate="%{x}<br>" + name + "<br>macro R2=%{text}<extra></extra>",
            ),
            row=1, col=column,
        )
figure.add_hline(y=0, line_color="#1F2937", line_width=2)
figure.update_yaxes(title_text="held-out macro R2 (clipped at -3)", range=[-3.2, 1.35], row=1, col=1)
figure.update_layout(
    title="Equalizing class size turns one positive rotation into two",
    barmode="group", template="plotly_white", height=560,
)
figure.show()
"""

BALANCED_READING = """
### What equalizing class size changes

**v2 crosses the class boundary in two of three rotations instead of one.**
Holding out the Tee couplers moves from -0.117 on the full catalogue to
**+0.422** balanced, and holding out the Generalized NCaps moves from +0.440 to
**+0.859**. Neither of those improvements is v2 getting better; it is the
experiment getting fairer. On the full catalogue the model is dominated by
13,683 Generalized rows, and a head fit largely to one family transfers worse to
the other two.

**`TransmonCross` stays negative** at -1.891. Two of three unseen classes are now
predictable with zero labels from them, and the qubit is not. That is a stable
finding across both cohorts and it is the honest boundary of the claim.

**v0 remains far outside the usable range in every rotation** at -17.6, -7.7, and
-8.3, and geometry-only v2 stays negative too. Only the full v2 vector crosses
the boundary at all.
"""

NEWCOMER_INTRO = """
## 7. The question a new contributor actually asks

The scenario that motivates this whole programme is concrete: a group sends us a
component family we have never seen and a handful of simulations, and wants a
useful model. The experiment below is exactly that, run three times.

For each class in the balanced cohort we treat it as the newcomer. We train a
foundation on the other two classes only, then give the model **M labeled
designs** from the newcomer, with M running from 0 to 400, and score it on a
held-out 30% of that class. Two adaptation strategies are compared: fitting from
scratch on those M labels, and adapting the foundation toward them.
"""

NEWCOMER_CODE = """
NEWCOMER_PATH = RUN_DIR / "balanced_newcomer.parquet"
ADAPT_BUDGETS = [0, 5, 10, 25, 50, 100, 200, 400]

if NEWCOMER_PATH.exists():
    newcomer = pd.read_parquet(NEWCOMER_PATH)
    print(f"Loaded newcomer curves from {NEWCOMER_PATH}")
else:
    records = []
    for name, matrix in BALANCED_REPRESENTATIONS.items():
        for held in sorted(CLASSES):
            train = np.flatnonzero(balanced_components != held)
            arriving = np.flatnonzero(balanced_components == held)
            features = project(matrix, train)
            foundation = TransferRidgeRegressor(ALPHA).fit(features[train], balanced_y[train])
            for repeat in range(REPEATS):
                rng = np.random.default_rng(SEED + 91 * repeat)
                order = rng.permutation(arriving)
                cut = max(1, int(round(0.3 * len(order))))
                test, pool = order[:cut], order[cut:]
                for budget in ADAPT_BUDGETS:
                    if budget == 0:
                        records.append({
                            "representation": name, "new_class": held, "labels": 0,
                            "method": "foundation, no labels", "repeat": repeat,
                            "r2": macro(balanced_y[test], foundation.predict(features[test]))["r2"],
                        })
                        continue
                    if budget > len(pool):
                        continue
                    chosen = pool[:budget]
                    for method, model in (
                        ("from scratch", TransferRidgeRegressor(ALPHA).fit(features[chosen], balanced_y[chosen])),
                        ("adapted from the other classes",
                         TransferRidgeRegressor(ALPHA).fit(features[chosen], balanced_y[chosen], prior=foundation)),
                    ):
                        records.append({
                            "representation": name, "new_class": held, "labels": budget,
                            "method": method, "repeat": repeat,
                            "r2": macro(balanced_y[test], model.predict(features[test]))["r2"],
                        })
    newcomer = pd.DataFrame(records)
    newcomer.to_parquet(NEWCOMER_PATH, index=False)

newcomer_summary = newcomer.groupby(["representation", "method", "labels"], as_index=False)["r2"].mean()
newcomer_summary.pivot(index=["method", "labels"], columns="representation", values="r2").round(4)
"""

NEWCOMER_FIGURE = """
# %% hide input
figure = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Averaged over all three newcomer classes", "Per newcomer class, v2 adapted"],
    horizontal_spacing=0.11,
)
for name in ("v0", "v2"):
    for method, dash in (("from scratch", "dot"), ("adapted from the other classes", "solid")):
        frame = newcomer_summary.query(
            "representation == @name and method == @method and labels > 0"
        ).sort_values("labels")
        figure.add_trace(
            go.Scatter(
                x=frame["labels"], y=frame["r2"], mode="lines+markers",
                name=f"{name}, {method}", line={"color": PALETTE[name], "width": 3, "dash": dash},
                marker={"size": 8},
                hovertemplate=f"<b>{name} {method}</b><br>%{{x}} labels<br>macro R2=%{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )
per_class = newcomer.query("representation == 'v2' and method == 'adapted from the other classes'")
per_class = per_class.groupby(["new_class", "labels"], as_index=False)["r2"].mean()
for component in sorted(CLASSES):
    frame = per_class.query("new_class == @component").sort_values("labels")
    figure.add_trace(
        go.Scatter(
            x=frame["labels"], y=frame["r2"], mode="lines+markers", name=component,
            line={"color": CLASS_COLORS[component], "width": 3}, marker={"size": 8},
            hovertemplate=f"<b>{component}</b><br>%{{x}} labels<br>macro R2=%{{y:.4f}}<extra></extra>",
        ),
        row=1, col=2,
    )
figure.add_hline(y=0.95, line_dash="dash", line_color="#6B7280", opacity=0.7)
figure.update_xaxes(title_text="labeled designs from the new class", type="log")
figure.update_yaxes(title_text="macro R2 on the new class", range=[-0.6, 1.05], row=1, col=1)
figure.update_layout(
    title="A brand-new component family needs about ten labels with v2",
    template="plotly_white", height=560,
)
figure.show()
"""

NEWCOMER_READING = """
### The practical answer

**About ten labels.** With v2 and adaptation, five labeled designs from a
completely unseen component family reach macro R2 0.887, ten reach 0.943, and
twenty-five reach 0.979. For comparison v0 needs roughly fifty to a hundred
labels to reach what v2 reaches with five to ten.

**A source prior only helps if the representation aligns the classes.** This is
the sharpest result in the notebook. At five labels, adaptation *improves* v2
from 0.610 to 0.887 and *destroys* v0, dropping it from 0.261 to -0.502. The
same mechanism, the same code, opposite signs. Pretrained weights are worth
nothing on a representation that puts the classes in unrelated regions; on one
that aligns them, they are worth an order of magnitude in labels.

**The advantage is a low-budget phenomenon.** By 400 labels all three
representations are above 0.98 and the curves have converged. The value of a
foundation representation is concentrated exactly where a new contributor lives:
the first few dozen simulations.
"""


BLOCK_ROLES_INTRO = """
## 8. Which block carries prediction, and which carries transfer?

Tutorial 21 ablates v2's blocks inside a single component family and finds that
the parameter block is nearly redundant: geometry alone matches the full vector.
Section 6 above found the opposite across classes, where dropping the parameter
block collapsed held-out performance.

Those are not contradictory, and the resolution is worth measuring directly. We
evaluate every block in three settings on the balanced cohort:

1. **in-class prediction** - train and test inside one component family, the
   ordinary supervised task;
2. **cross-class with no labels** - train on two families, predict the third;
3. **cross-class with ten labels** - the same, after adapting on ten designs
   from the unseen family.

The question is whether a block that helps you predict is the same block that
helps you transfer.
"""

BLOCK_ROLES_CODE = """
BLOCK_ROLES_PATH = RUN_DIR / "block_roles.parquet"
COUPLING_STOP = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE
SHAPE_STOP_ALL = COUPLING_STOP + SHAPE_BLOCK_SIZE
PARAM_STOP = SHAPE_STOP_ALL + PARAMETER_BLOCK_SIZE
FEW_SHOT = 10

BLOCKS = {
    "v2 full (512)": balanced_v2,
    "v2 geometry only (416)": np.c_[balanced_v2[:, :SHAPE_STOP_ALL], balanced_v2[:, PARAM_STOP:V2_DIMENSIONS]],
    "physical metrics (48)": balanced_v2[:, :METRIC_BLOCK_SIZE],
    "coupling spectrum (192)": balanced_v2[:, METRIC_BLOCK_SIZE:COUPLING_STOP],
    "shape spectrum (128)": balanced_v2[:, COUPLING_STOP:SHAPE_STOP_ALL],
    "parameter statistics (96)": balanced_v2[:, SHAPE_STOP_ALL:PARAM_STOP],
    "physics proxy (48)": balanced_v2[:, PARAM_STOP:V2_DIMENSIONS],
    "v0 full (155)": balanced_v0,
}

if BLOCK_ROLES_PATH.exists():
    block_roles = pd.read_parquet(BLOCK_ROLES_PATH)
    print(f"Loaded block roles from {BLOCK_ROLES_PATH}")
else:
    records = []
    for name, matrix in BLOCKS.items():
        raw = matrix.astype(np.float64)
        in_class = []
        for component in sorted(CLASSES):
            rows = np.flatnonzero(balanced_components == component)
            projector = V0KernelFeatureProjector(kernel_dimensions=128, random_seed=SEED)
            features = projector.fit_transform_compact(raw[rows])
            local = balanced_y[rows]
            for repeat in range(REPEATS):
                order = np.random.default_rng(SEED + 17 * repeat).permutation(len(rows))
                cut = int(0.3 * len(order))
                test, train = order[:cut], order[cut:]
                model = TransferRidgeRegressor(ALPHA).fit(features[train], local[train])
                in_class.append(macro(local[test], model.predict(features[test]))["r2"])
        records.append({"block": name, "setting": "in-class prediction", "r2": float(np.mean(in_class))})

        zero_shot, few_shot = [], []
        for held in sorted(CLASSES):
            train = np.flatnonzero(balanced_components != held)
            arriving = np.flatnonzero(balanced_components == held)
            features = project(matrix, train)
            foundation = TransferRidgeRegressor(ALPHA).fit(features[train], balanced_y[train])
            for repeat in range(REPEATS):
                order = np.random.default_rng(SEED + 91 * repeat).permutation(arriving)
                cut = max(1, int(round(0.3 * len(order))))
                test, pool = order[:cut], order[cut:]
                zero_shot.append(macro(balanced_y[test], foundation.predict(features[test]))["r2"])
                chosen = pool[:FEW_SHOT]
                adapted = TransferRidgeRegressor(ALPHA).fit(
                    features[chosen], balanced_y[chosen], prior=foundation
                )
                few_shot.append(macro(balanced_y[test], adapted.predict(features[test]))["r2"])
        records.append({"block": name, "setting": "cross-class, no labels", "r2": float(np.mean(zero_shot))})
        records.append({"block": name, "setting": f"cross-class, {FEW_SHOT} labels",
                        "r2": float(np.mean(few_shot))})
    block_roles = pd.DataFrame(records)
    block_roles.to_parquet(BLOCK_ROLES_PATH, index=False)

block_roles.pivot(index="block", columns="setting", values="r2").reindex(list(BLOCKS)).round(4)
"""

BLOCK_ROLES_FIGURE = """
# %% hide input
order = list(BLOCKS)
colors = ["#00798C", "#2A9D8F", "#F4A261", "#4C86A8", "#6A4C93", "#E9C46A", "#8AB17D", "#D1495B"]
figure = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "in-class prediction<br><sub>every block works</sub>",
        "cross-class, 10 labels<br><sub>the ranking changes</sub>",
        "cross-class, no labels<br><sub>clipped at -60</sub>",
    ],
    horizontal_spacing=0.07,
)
panels = [
    ("in-class prediction", 1, [0.98, 1.001]),
    ("cross-class, 10 labels", 2, [-4.2, 1.15]),
    ("cross-class, no labels", 3, [-62, 6]),
]
for setting, column, span in panels:
    frame = block_roles.query("setting == @setting").set_index("block").reindex(order)
    values = np.clip(frame["r2"].to_numpy(), span[0], None)
    figure.add_trace(
        go.Bar(
            x=values, y=order, orientation="h", marker_color=colors, showlegend=False,
            text=[f"{value:.3f}" if value > -10 else f"{value:.0f}" for value in frame["r2"].to_numpy()],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + setting + "<br>macro R2=%{text}<extra></extra>",
        ),
        row=1, col=column,
    )
    figure.update_xaxes(range=span, title_text="macro R2", row=1, col=column)
    figure.add_vline(x=0, line_color="#1F2937", line_width=1, row=1, col=column)
figure.update_yaxes(autorange="reversed")
figure.update_yaxes(showticklabels=False, row=1, col=2)
figure.update_yaxes(showticklabels=False, row=1, col=3)
figure.update_layout(
    title="The block that predicts is not the block that transfers",
    template="plotly_white", height=520, margin={"l": 200},
)
figure.show()
"""

BLOCK_ROLES_READING = """
### Reading the three settings

**In-class, block ablation is almost uninformative.** Every single block reaches
macro R2 0.987 or better on its own: the shape spectrum alone gets 0.9963, the
physics proxy alone 0.9870, and even v0 gets 0.9921 against the full v2 vector's
0.9998. Predicting capacitance inside one component family is easy enough that
almost any faithful description of the geometry suffices. An ablation run only
in this setting would conclude, wrongly, that the blocks are interchangeable.

**Cross-class, the same blocks span three orders of magnitude.** With ten labels
from the unseen family the physical-metric block reaches 0.960 and the shape
spectrum reaches **-3.53**. The ranking is not a rescaling of the in-class
ranking; it is a different ordering.

**The shape spectrum is the clearest reversal.** It is among the best in-class
blocks at 0.9963 and by far the worst cross-class at -56.2 with no labels and
-3.53 with ten. That is physically sensible: two-point correlations and contour
harmonics describe *what a family's geometry looks like*, and an interdigital
comb and a transmon cross do not look alike. The block encodes exactly the
information that does not generalize.

**The physical metrics are the best transfer block**, at 0.960 with ten labels,
ahead of the full 512-dimensional vector. Absolute area, perimeter, gap, and
width in micrometres mean the same thing for a coupler and for a qubit, so they
are the coordinates that survive a change of component class.

**The parameter block earns its place here and not in Tutorial 21.** Full v2 at
-0.219 zero-shot beats geometry-only at -5.239, a twenty-fold difference, and
holds the advantage at ten labels. Inside one family, where every design shares a
parameter schema, the typed order statistics are redundant with the geometry;
across families they are one of the few things that stays comparable.

The practical consequence is the same one Tutorial 21 reaches from a different
direction: v2 should be published as blocks that a user can select, because the
right subset depends on whether the task is prediction or transfer.
"""


CELLS = [
    markdown(
        """
# Tutorial 20: crossing the component-class boundary with `universal-geometry-v2`

Tutorial 17 crossed one class boundary using `static-shape-v0`, from
`GeneralizedCapNInterdigital` to `CapNInterdigitalTee`, and closed by noting
that only those two NCap classes shared a comparable supervised target.

That turns out to be untrue, and the correction opens the experiment this
tutorial runs. `TransmonCross` reports `cross_to_claw`, which is the mutual
capacitance between two conductors exactly as `north_to_south` and
`top_to_bottom` are. **Three** SQuADDS families therefore share one physical
target:

| Component | Mutual capacitance field | Design options |
| --- | --- | ---: |
| `GeneralizedCapNInterdigital` | `north_to_south` | 41 |
| `CapNInterdigitalTee` | `top_to_bottom` | 11 |
| `TransmonCross` | `cross_to_claw` | 22 |

Three classes rather than two changes what can be asked. With two you can only
transfer from one to the other. With three you can hold an entire component
class out, train on the rest, and ask whether the representation generalizes to
a device family it has never seen. That is the actual foundation-model test, and
no tutorial in this repository has run it before.

We will:

1. show that a parameter-schema baseline does not exist across three classes;
2. put all three families in one embedding space and compare v0 with v2;
3. run cross-class transfer curves from a Generalized NCap foundation;
4. hold each class out entirely and predict it with zero labels; and
5. test whether predicting the **physics-proxy residual** transfers better than
   predicting capacitance directly.
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
from scipy.stats import spearmanr

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

CLASSES = {
    "GeneralizedCapNInterdigital": {
        "file": "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
        "mutual": "north_to_south",
        "grounds": ("north_to_ground", "south_to_ground"),
    },
    "CapNInterdigitalTee": {
        "file": "coupler-CapNInterdigitalTee-cap_matrix.json",
        "mutual": "top_to_bottom",
        "grounds": ("top_to_ground", "bottom_to_ground"),
    },
    "TransmonCross": {
        "file": "qubit-TransmonCross-cap_matrix.json",
        "mutual": "cross_to_claw",
        "grounds": ("cross_to_ground", "claw_to_ground"),
    },
}
TARGETS = ["log_mutual", "log_ground_sum"]
FRACTIONS = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
REPEATS = 12
TEST_FRACTION = 0.30
SEED = 20
ALPHA = 0.3
SOURCE = "GeneralizedCapNInterdigital"
CLASS_COLORS = {
    "GeneralizedCapNInterdigital": "#00798C",
    "CapNInterdigitalTee": "#D1495B",
    "TransmonCross": "#6A4C93",
}
PALETTE = {"v0": "#D1495B", "v2": "#00798C", "v2 geometry only": "#2A9D8F", "shared parameters": "#E9C46A"}

PHYSICS_OFFSET = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE + PARAMETER_BLOCK_SIZE
SHAPE_STOP = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE
GEOMETRY_COLUMNS = np.r_[0:SHAPE_STOP, SHAPE_STOP + PARAMETER_BLOCK_SIZE : V2_DIMENSIONS]

CHECKPOINTS = Path(os.getenv("SQUADDS_TUTORIAL20_CACHE", Path.home() / ".cache/squadds/tutorial20"))
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# The multi-family v2 table is built locally with the same command as Tutorial 18,
# but without --component-name so every family is encoded.
V2_TABLE = Path(os.getenv("SQUADDS_V2_ALL_TABLE", CHECKPOINTS / "universal-geometry-v2-all.parquet"))
if not V2_TABLE.is_file():
    raise FileNotFoundError(f"Build the multi-family v2 table first; expected {V2_TABLE}.")

v0_path = Path(
    hf_hub_download("SQuADDS/SQuADDS_Layout_Embeddings", "metadata/static-embedding-v0.parquet", repo_type="dataset")
)
database = {
    name: Path(hf_hub_download("SQuADDS/SQuADDS_DB", spec["file"], repo_type="dataset"))
    for name, spec in CLASSES.items()
}
print("v2 table:", V2_TABLE.name)
for name, path in database.items():
    print(f"  {name:30s} {path.name}")
"""
    ),
    markdown(
        """
## 1. There is no shared parameter schema

Tutorial 17 built a "best-effort" baseline from the two geometry fields both
NCap classes happen to name identically, `finger_count` and `finger_length`, and
strengthened it with polynomial terms. That baseline was weak but it existed.

Add a qubit and it stops existing. The cell below intersects the option names of
all three families.
"""
    ),
    code(
        """
option_names = {}
for name, path in database.items():
    names = set()
    for row in json.loads(path.read_text()):
        names |= set(row["design"]["design_options"])
    option_names[name] = names

pairs = []
keys = list(option_names)
for index, first in enumerate(keys):
    for second in keys[index + 1 :]:
        shared = sorted(option_names[first] & option_names[second])
        pairs.append({"pair": f"{first} & {second}", "shared names": len(shared), "names": ", ".join(shared) or "-"})
everything = sorted(set.intersection(*option_names.values()))

print(pd.DataFrame([{"component": k, "option names": len(v)} for k, v in option_names.items()]).to_string(index=False))
print()
print(pd.DataFrame(pairs).to_string(index=False))
print()
print(f"shared by ALL THREE classes: {everything}")
"""
    ),
    markdown(
        """
`orientation` is a placement angle. It says how the component is rotated on the
chip, not how much metal faces how much other metal across what gap, so it
carries no information about capacitance.

That is the whole argument for a geometry-derived representation stated as a
fact about the data rather than as a preference. Across three component classes
the intersection of the design-tool vocabularies is empty of physics. There is
nothing to align, no matter how much effort is spent aligning it. Any shared
feature contract has to be computed from something all three actually have,
which is the layout itself.
"""
    ),
    code(
        """
def load_targets():
    records = []
    for component, spec in CLASSES.items():
        for row in json.loads(database[component].read_text()):
            options = row["design"]["design_options"]
            results = row["sim_results"]
            mutual = abs(float(results[spec["mutual"]]))
            grounds = sum(abs(float(results[name])) for name in spec["grounds"])
            records.append(
                {
                    "design_id": canonical_design_id(component, options),
                    "component_name": component,
                    "mutual_fF": mutual,
                    "log_mutual": float(np.log1p(mutual)),
                    "log_ground_sum": float(np.log1p(grounds)),
                }
            )
    return pd.DataFrame(records).drop_duplicates("design_id")


def load_v0(path, keep):
    parquet = pq.ParquetFile(path)
    identifiers, blocks = [], []
    for batch in parquet.iter_batches(batch_size=256, columns=["design_id", "embedding"]):
        frame = batch.to_pandas()
        frame = frame[frame.design_id.isin(keep)]
        if frame.empty:
            continue
        matrix = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
        blocks.append(compress_v0_embeddings(matrix, pooled_shape_size=12).astype(np.float32))
        identifiers.extend(frame["design_id"].tolist())
    return pd.Series(identifiers), np.vstack(blocks)


targets = load_targets()
v2_frame = pd.read_parquet(V2_TABLE).drop_duplicates("design_id")
v2_frame = v2_frame[v2_frame.design_id.isin(set(targets.design_id))]
v2_all = np.vstack(v2_frame["embedding"].to_numpy()).astype(np.float32)
v0_ids, v0_all = load_v0(v0_path, set(v2_frame.design_id))

data = (
    targets.merge(v2_frame[["design_id"]].assign(v2_row=range(len(v2_frame))), on="design_id")
    .merge(pd.DataFrame({"design_id": v0_ids}).assign(v0_row=range(len(v0_ids))), on="design_id")
    .drop_duplicates("design_id")
    .reset_index(drop=True)
)
v2 = v2_all[data["v2_row"].to_numpy()]
v0 = v0_all[data["v0_row"].to_numpy()]
y = data[TARGETS].to_numpy(float)
components = data["component_name"].to_numpy()
data["proxy_log"] = np.abs(v2[:, PHYSICS_OFFSET + 1])
data["residual"] = data["log_mutual"] - data["proxy_log"]

summary = data.groupby("component_name").agg(
    designs=("design_id", "size"),
    mutual_min_fF=("mutual_fF", "min"),
    mutual_median_fF=("mutual_fF", "median"),
    mutual_max_fF=("mutual_fF", "max"),
)
print(summary.round(3).to_string())
print(f"\\npaired designs across all three classes: {len(data):,}")
"""
    ),
    markdown(
        """
The capacitance scales differ by more than an order of magnitude between
families, which is why the supervised target is `log1p(C)` rather than `C`.

## 2. Three families, one space

The projection below is unsupervised: pooled, standardized, and reduced to two
directions by a randomized sketch. It is only a picture, but it shows the
question. If a representation puts the classes in disjoint islands, a model
trained on one has no reason to say anything useful about another.
"""
    ),
    code(
        """
# %% hide input
def sketch(matrix, seed=20):
    values = matrix.astype(np.float64)
    centre = values.mean(axis=0)
    scale = values.std(axis=0)
    keep = scale > 1e-8
    standardized = (values[:, keep] - centre[keep]) / scale[keep]
    rng = np.random.default_rng(seed)
    projected = standardized @ rng.normal(0, 1 / np.sqrt(32), size=(standardized.shape[1], 32))
    left, _, _ = np.linalg.svd(projected, full_matrices=False)
    return left[:, :2]


figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["static-shape-v0", "universal-geometry-v2"],
    horizontal_spacing=0.09,
)
rng = np.random.default_rng(SEED)
shown = rng.choice(len(data), size=min(6000, len(data)), replace=False)
for column, (name, matrix) in enumerate((("v0", v0), ("v2", v2)), start=1):
    coordinates = sketch(matrix)
    for component in CLASSES:
        rows = shown[components[shown] == component]
        figure.add_trace(
            go.Scattergl(
                x=coordinates[rows, 0],
                y=coordinates[rows, 1],
                mode="markers",
                marker={"size": 4, "opacity": 0.45, "color": CLASS_COLORS[component]},
                name=component,
                legendgroup=component,
                showlegend=column == 1,
                customdata=data["mutual_fF"].to_numpy()[rows],
                hovertemplate=f"<b>{component}</b><br>C=%{{customdata:.2f}} fF<extra></extra>",
            ),
            row=1,
            col=column,
        )
figure.update_layout(
    title="All three component classes projected into each representation",
    template="plotly_white",
    height=530,
)
figure.show()
"""
    ),
    markdown(
        """
## 3. Protocol

Every comparison below uses one pipeline: a deterministic random-Fourier feature
map, a multi-output ridge head, the same alpha, the same splits, and the same
test rows. Only the input vector changes.

One detail matters for honesty. The unsupervised feature map is **fit on the
training classes only**, never on the pooled catalogue. Fitting the scaling on
rows from a class the experiment is pretending not to have seen would quietly
leak the thing being measured.

Four representations are compared:

- **shared parameters** — Tutorial 17's manually aligned `finger_count` and
  `finger_length` with polynomial terms; only definable for the two NCap classes;
- **v0** — the 155-dimensional compact `static-shape-v0` view;
- **v2** — all 512 `universal-geometry-v2` coordinates;
- **v2 geometry only** — v2 with the 96 parameter coordinates removed, so it
  sees the GDS file and nothing else.
"""
    ),
    code(
        """
def project(matrix, fit_rows):
    \"\"\"Fit the unsupervised map on allowed rows only, then transform everything.\"\"\"
    projector = V0KernelFeatureProjector(kernel_dimensions=128, random_seed=SEED)
    projector.fit_compact(matrix[fit_rows].astype(np.float64))
    return projector.transform_compact(matrix.astype(np.float64))


def macro(expected, predicted):
    return regression_scores(expected, predicted, TARGETS).query("target == 'macro'").iloc[0]


shared_fields = []
for component, path in database.items():
    rows = json.loads(path.read_text())
    lookup = {
        canonical_design_id(component, row["design"]["design_options"]): row["design"]["design_options"]
        for row in rows
    }
    for design_id in data.loc[data.component_name == component, "design_id"]:
        options = lookup[design_id]
        count = float(options.get("finger_count", 0.0) or 0.0)
        length = float(str(options.get("finger_length", "0")).replace("um", "") or 0.0)
        shared_fields.append((design_id, count, length))
shared_frame = pd.DataFrame(shared_fields, columns=["design_id", "finger_count", "finger_length_um"])
shared = data[["design_id"]].merge(shared_frame, on="design_id")[["finger_count", "finger_length_um"]].to_numpy(float)
shared_polynomial = np.column_stack([shared, shared**2, shared[:, 0] * shared[:, 1]])

REPRESENTATIONS = {
    "shared parameters": shared_polynomial,
    "v0": v0,
    "v2": v2,
    "v2 geometry only": v2[:, GEOMETRY_COLUMNS],
}
EXPERIMENT = {
    "rows": int(len(data)),
    "repeats": REPEATS,
    "fractions": FRACTIONS,
    "alpha": ALPHA,
    "targets": TARGETS,
    "v2_dimensions": int(v2.shape[1]),
}
FINGERPRINT = hashlib.sha256(json.dumps(EXPERIMENT, sort_keys=True).encode()).hexdigest()[:16]
RUN_DIR = CHECKPOINTS / f"study-{FINGERPRINT}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    [
        {"quantity": "paired designs", "value": f"{len(data):,}"},
        {"quantity": "component classes", "value": len(CLASSES)},
        {"quantity": "independent repeats", "value": REPEATS},
        {"quantity": "v0 dimensions", "value": v0.shape[1]},
        {"quantity": "v2 dimensions", "value": v2.shape[1]},
        {"quantity": "shared-parameter features", "value": shared_polynomial.shape[1]},
        {"quantity": "fingerprint", "value": FINGERPRINT},
    ]
)
"""
    ),
    markdown(
        """
## 4. Cross-class transfer from a Generalized NCap foundation

A foundation is fit on all 13,683 `GeneralizedCapNInterdigital` designs, then
adapted to a target class with an increasing number of that class's labels. At
each budget we compare a **target-only** model against a **transfer** model
regularized toward the foundation weights, plus the **zero-shot** foundation.
"""
    ),
    code(
        """
CURVES_PATH = RUN_DIR / "cross_class_curves.parquet"


def transfer_curves(name, matrix, target):
    source_rows = np.flatnonzero(components == SOURCE)
    target_rows = np.flatnonzero(components == target)
    features = project(matrix, source_rows)
    foundation = TransferRidgeRegressor(ALPHA).fit(features[source_rows], y[source_rows])
    records = []
    for repeat in range(REPEATS):
        rng = np.random.default_rng(SEED + 137 * repeat)
        order = rng.permutation(target_rows)
        cut = max(1, int(round(len(order) * TEST_FRACTION)))
        test, pool = order[:cut], order[cut:]
        scores = macro(y[test], foundation.predict(features[test]))
        records.append({"repeat": repeat, "fraction": 0.0, "labels": 0, "method": "zero-shot", **scores})
        for fraction in FRACTIONS:
            size = max(2, int(round(fraction * len(pool))))
            chosen = pool[:size]
            for method, model in (
                ("target-only", TransferRidgeRegressor(ALPHA).fit(features[chosen], y[chosen])),
                ("transfer", TransferRidgeRegressor(ALPHA).fit(features[chosen], y[chosen], prior=foundation)),
            ):
                records.append(
                    {
                        "repeat": repeat,
                        "fraction": fraction,
                        "labels": size,
                        "method": method,
                        **macro(y[test], model.predict(features[test])),
                    }
                )
    frame = pd.DataFrame(records)
    frame.insert(0, "representation", name)
    frame.insert(1, "target_class", target)
    return frame


if CURVES_PATH.exists():
    curves = pd.read_parquet(CURVES_PATH)
    print(f"Loaded curves from {CURVES_PATH}")
else:
    frames = []
    for target in ("CapNInterdigitalTee", "TransmonCross"):
        for name, matrix in REPRESENTATIONS.items():
            if name == "shared parameters" and target == "TransmonCross":
                continue  # no geometry field is shared with the qubit class
            frames.append(transfer_curves(name, matrix, target))
    curves = pd.concat(frames, ignore_index=True)
    curves.to_parquet(CURVES_PATH, index=False)
    print(f"Fitted {len(curves):,} evaluations")

curve_summary = curves.groupby(["target_class", "representation", "method", "fraction"], as_index=False)["r2"].mean()
curve_summary.query("method == 'transfer'").pivot(
    index=["target_class", "fraction"], columns="representation", values="r2"
).round(4)
"""
    ),
    code(
        """
# %% hide input
figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[f"{SOURCE} -> {target}" for target in ("CapNInterdigitalTee", "TransmonCross")],
    horizontal_spacing=0.10,
    shared_yaxes=True,
)
for column, target in enumerate(("CapNInterdigitalTee", "TransmonCross"), start=1):
    for name in REPRESENTATIONS:
        frame = curve_summary.query("target_class == @target and representation == @name and method == 'transfer'")
        if frame.empty:
            continue
        frame = frame.sort_values("fraction")
        figure.add_trace(
            go.Scatter(
                x=100 * frame["fraction"],
                y=frame["r2"],
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=column == 1,
                line={"color": PALETTE[name], "width": 3},
                marker={"size": 8},
                hovertemplate=f"<b>{name}</b><br>%{{x:.0f}}% labels<br>macro R2=%{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    zero = curve_summary.query("target_class == @target and method == 'zero-shot'")
    for name in ("v0", "v2"):
        value = zero.query("representation == @name")["r2"]
        if not value.empty:
            figure.add_hline(
                y=float(value.iloc[0]),
                line_dash="dot",
                line_color=PALETTE[name],
                opacity=0.6,
                row=1,
                col=column,
            )
figure.update_xaxes(title_text="labeled fraction of the target class (%)", type="log")
figure.update_yaxes(title_text="held-out macro R2 (log space)", range=[-0.6, 1.02], row=1, col=1)
figure.update_layout(
    title="Cross-class transfer curves; dotted lines mark zero-shot",
    template="plotly_white",
    height=560,
)
figure.show()
"""
    ),
    markdown(TRANSFER_READING),
    markdown(
        """
## 5. Hold an entire component class out

This is the test the two-class setup could not support. For each family in turn
we train on the other two and predict the held-out one with **zero** labels from
it. A representation that merely memorizes one class's geometry cannot score
above zero here.
"""
    ),
    code(
        """
HELD_PATH = RUN_DIR / "held_out_class.parquet"


def held_out(name, matrix, target_matrix, label=None):
    records = []
    for held in CLASSES:
        train = np.flatnonzero(components != held)
        test = np.flatnonzero(components == held)
        features = project(matrix, train)
        model = TransferRidgeRegressor(ALPHA).fit(features[train], target_matrix[train])
        scores = macro(target_matrix[test], model.predict(features[test]))
        records.append(
            {
                "representation": label or name,
                "held_out_class": held,
                "train_rows": len(train),
                "test_rows": len(test),
                "r2": scores["r2"],
                "mae": scores["mae"],
            }
        )
    return pd.DataFrame(records)


if HELD_PATH.exists():
    held = pd.read_parquet(HELD_PATH)
    print(f"Loaded held-out-class results from {HELD_PATH}")
else:
    held = pd.concat(
        [held_out(name, matrix, y) for name, matrix in REPRESENTATIONS.items()],
        ignore_index=True,
    )
    held.to_parquet(HELD_PATH, index=False)

held.pivot(index="held_out_class", columns="representation", values="r2").round(4)
"""
    ),
    code(
        """
# %% hide input
order = ["shared parameters", "v0", "v2 geometry only", "v2"]
figure = go.Figure()
for name in order:
    frame = held.query("representation == @name").set_index("held_out_class").reindex(list(CLASSES))
    figure.add_trace(
        go.Bar(
            x=list(CLASSES),
            y=frame["r2"].to_numpy(),
            name=name,
            marker_color=PALETTE[name],
            text=[f"{value:.3f}" for value in frame["r2"].to_numpy()],
            textposition="outside",
            hovertemplate="%{x}<br>" + name + "<br>macro R2=%{y:.4f}<extra></extra>",
        )
    )
figure.add_hline(y=0, line_color="#1F2937", line_width=1)
figure.update_layout(
    title="Train on two component classes, predict the third with no labels from it",
    yaxis={"title": "held-out macro R2 (log space)"},
    xaxis={"title": "component class held out"},
    barmode="group",
    template="plotly_white",
    height=560,
)
figure.show()
"""
    ),
    markdown(HELDOUT_READING),
    markdown(
        """
## 6. Predicting the physics-proxy residual

v2 carries a two-dimensional boundary-element estimate of the mutual
capacitance. It is not a simulation: it ignores the substrate, the metal
thickness, and every three-dimensional effect. But it is computed identically
for every class, which suggests a different supervised target.

Instead of predicting $\\log(1+C)$, predict the **residual**

$$r \\;=\\; \\log(1+C_{\\text{simulated}}) - \\log(1 + C_{\\text{proxy}}),$$

the correction from the crude electrostatic estimate to the real answer. The
proxy absorbs the part of the map that is common to all geometry, leaving the
model a smoother and more class-independent quantity to learn.
"""
    ),
    code(
        """
RESIDUAL_PATH = RUN_DIR / "residual_target.parquet"
residual_y = np.column_stack([data["residual"].to_numpy(), data["log_ground_sum"].to_numpy()])

for component in CLASSES:
    mask = components == component
    rho = spearmanr(data.loc[mask, "proxy_log"], data.loc[mask, "mutual_fF"]).statistic
    print(f"  proxy vs simulated mutual within {component:30s} Spearman {rho:+.3f}")
print(f"  proxy vs simulated mutual pooled across classes      Spearman "
      f"{spearmanr(data['proxy_log'], data['mutual_fF']).statistic:+.3f}")

if RESIDUAL_PATH.exists():
    residual = pd.read_parquet(RESIDUAL_PATH)
    print(f"\\nLoaded residual results from {RESIDUAL_PATH}")
else:
    frames = []
    for name in ("v0", "v2"):
        frames.append(held_out(name, REPRESENTATIONS[name], y, label=f"{name} | predict log C"))
        frames.append(held_out(name, REPRESENTATIONS[name], residual_y, label=f"{name} | predict residual"))
    residual = pd.concat(frames, ignore_index=True)
    residual.to_parquet(RESIDUAL_PATH, index=False)

residual.pivot(index="held_out_class", columns="representation", values="r2").round(4)
"""
    ),
    code(
        """
# %% hide input
figure = go.Figure()
styles = {
    "v0 | predict log C": ("#D1495B", 0.55),
    "v0 | predict residual": ("#D1495B", 1.0),
    "v2 | predict log C": ("#00798C", 0.55),
    "v2 | predict residual": ("#00798C", 1.0),
}
for name, (color, opacity) in styles.items():
    frame = residual.query("representation == @name").set_index("held_out_class").reindex(list(CLASSES))
    figure.add_trace(
        go.Bar(
            x=list(CLASSES),
            y=frame["r2"].to_numpy(),
            name=name,
            marker_color=color,
            marker_opacity=opacity,
            text=[f"{value:.2f}" for value in frame["r2"].to_numpy()],
            textposition="outside",
            hovertemplate="%{x}<br>" + name + "<br>macro R2=%{y:.4f}<extra></extra>",
        )
    )
figure.add_hline(y=0, line_color="#1F2937", line_width=1)
figure.update_layout(
    title="Does subtracting a crude electrostatic estimate make the target more transferable?",
    yaxis={"title": "held-out macro R2"},
    xaxis={"title": "component class held out"},
    barmode="group",
    template="plotly_white",
    height=560,
)
figure.show()
"""
    ),
    markdown(RESIDUAL_READING),
    markdown(BALANCED_INTRO),
    code(BALANCED_CODE),
    code(BALANCED_FIGURE),
    markdown(BALANCED_READING),
    markdown(NEWCOMER_INTRO),
    code(NEWCOMER_CODE),
    code(NEWCOMER_FIGURE),
    markdown(NEWCOMER_READING),
    markdown(BLOCK_ROLES_INTRO),
    code(BLOCK_ROLES_CODE),
    code(BLOCK_ROLES_FIGURE),
    markdown(BLOCK_ROLES_READING),
    markdown(
        """
## 9. Does similarity mean anything across a class boundary?

Accuracy is not the only thing a foundation representation owes us. The routing
workflow needs cosine similarity to correlate with physical closeness even when
two designs come from different families. We sample cross-class pairs only, and
correlate their similarity against the gap in log mutual capacitance.
"""
    ),
    code(
        """
def similarity_table(matrix):
    reference = matrix.astype(np.float64)
    centre = reference.mean(axis=0)
    scale = reference.std(axis=0)
    keep = scale > 1e-8
    standardized = (reference[:, keep] - centre[keep]) / scale[keep]
    return standardized / np.maximum(np.linalg.norm(standardized, axis=1, keepdims=True), 1e-12)


log_mutual = data["log_mutual"].to_numpy()
names = sorted(CLASSES)
rows, samples = [], {}
for name, matrix in (("v0", v0), ("v2", v2)):
    unit = similarity_table(matrix)
    rng = np.random.default_rng(SEED)
    for index, first in enumerate(names):
        for second in names[index:]:
            left_pool = np.flatnonzero(components == first)
            right_pool = np.flatnonzero(components == second)
            left = rng.choice(left_pool, 20000)
            right = rng.choice(right_pool, 20000)
            keep = left != right
            left, right = left[keep], right[keep]
            similarity = np.sum(unit[left] * unit[right], axis=1)
            gap = np.abs(log_mutual[left] - log_mutual[right])
            label = f"within {first}" if first == second else f"{first} vs {second}"
            samples[(name, label)] = (similarity, gap)
            rows.append(
                {
                    "representation": name,
                    "pair": label,
                    "spearman": round(float(spearmanr(similarity, gap).statistic), 3),
                    "median cosine": round(float(np.median(similarity)), 3),
                }
            )
similarity_frame = pd.DataFrame(rows)
print("Spearman(cosine similarity, |difference in log mutual C|)")
print("negative means similar-looking designs really do behave similarly")
print()
similarity_frame.pivot(index="pair", columns="representation", values="spearman")
"""
    ),
    code(
        """
# %% hide input
order = [f"within {name}" for name in names] + [
    f"{names[i]} vs {names[j]}" for i in range(len(names)) for j in range(i + 1, len(names))
]
figure = go.Figure()
for name in ("v0", "v2"):
    frame = similarity_frame.query("representation == @name").set_index("pair").reindex(order)
    figure.add_trace(
        go.Bar(
            x=order,
            y=frame["spearman"].to_numpy(),
            name=name,
            marker_color=PALETTE[name],
            text=[f"{value:+.2f}" for value in frame["spearman"].to_numpy()],
            textposition="outside",
            hovertemplate="%{x}<br>" + name + "<br>Spearman=%{y:+.3f}<extra></extra>",
        )
    )
figure.add_hline(y=0, line_color="#1F2937", line_width=2)
figure.add_annotation(
    x=0.02, y=-0.78, xref="paper", yref="y", text="useful: similar looks, similar physics",
    showarrow=False, font={"color": "#2A9D8F"},
)
figure.add_annotation(
    x=0.02, y=0.42, xref="paper", yref="y", text="misleading: similar looks, different physics",
    showarrow=False, font={"color": "#D1495B"},
)
figure.update_layout(
    title="Similarity is trustworthy inside a class; across the qubit boundary it is not",
    yaxis={"title": "Spearman(cosine, |delta log C|)", "range": [-0.85, 0.5]},
    xaxis={"tickangle": -18},
    barmode="group",
    template="plotly_white",
    height=560,
)
figure.show()
"""
    ),
    markdown(CONCLUSIONS),
]

notebook["cells"] = CELLS

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, str(OUTPUT))
print(f"Wrote {OUTPUT} with {len(notebook['cells'])} cells.")
