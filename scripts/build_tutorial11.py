#!/usr/bin/env python
"""Build ``tutorials/Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.ipynb`` (unexecuted).

Execute it afterwards with ``jupyter nbconvert --to notebook --execute --inplace`` from ``tutorials/``. With an executed copy
as argument, the outputs of unchanged code cells are carried over (prose edits without a re-run).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorials" / "Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.ipynb"
NEXT_URL = "https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-14_Training_on_Your_Own_Designs.html"

LICENSE = """## License

<div style='width: 100%; background-color:#3cb1c2;color:#324344;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'>
    <h3>This code is a part of SQuADDS</h3>
    <p>Developed by Sadman Ahmed Shanto</p>
    <p>This tutorial is written by Sadman Ahmed Shanto</p>
    <p>&copy; Copyright Sadman Ahmed Shanto & Eli Levenson-Falk 2026.</p>
    <p>This code is licensed under the MIT License. You may<br> obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree.</p>
    <p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p>
</div>"""

CELLS = [
    (
        "md",
        """# Tutorial 11: Predicting Capacitance Matrices from a GDS

In this tutorial, we will take a layout (a GDS file, or a component we draw in Qiskit Metal) and predict its full Maxwell capacitance matrix, for any number of conductors, in a couple of seconds on a CPU. By the end of this tutorial, you will be able to:

- Explain why a device physicist wants the full $N \\times N$ capacitance matrix and not just one number per pair of conductors

- Draw a component in Qiskit Metal, export it to a GDS, and look at its conductors before trusting any number

- Load the planar capacitance models with `load_model` and predict a named Maxwell matrix with `CapacitancePredictor`

- Check the physics of a prediction (symmetry, signs, closure, positive definiteness) and read its timing

- Choose between the Accurate and Fast tiers, and compare the model with the two physics priors it is built on

- Predict many layouts at once, control the CPU threads, and use the `squadds-ml` command line

- Turn a prediction into a `SQuADDS_DB`-style `sim_results` row, and compute the geometric embedding of a layout

**Note:** the models need the optional ML extra, `pip install "squadds[ml]"`, and the GDS export from Qiskit Metal needs the `gds` extra, `pip install "squadds[gds]"`.""",
    ),
    ("py", "%load_ext autoreload\n%autoreload 2\n%matplotlib inline"),
    (
        "md",
        """## Why the full capacitance matrix?

Let's say we are laying out a coupler that has to talk to three qubits at once: a five-fingered star in a circular ground pocket, with a pad sitting in three of its valleys (a *multipad star*). Before we send it to the fab, we want to know:

1. the coupling capacitance between the star and each pad (it sets the coupling strength $g$),
2. the self capacitance of every island (it sets the charging energies $E_C$), and
3. the pad-to-pad capacitances we never asked for (they set the crosstalk).

All of these are entries of **one** matrix. For $N$ conductors (ground included), the charges and the voltages are related by the Maxwell capacitance matrix

$$Q_i = \\sum_{j} C_{ij} V_j .$$

The diagonal entry $C_{ii} > 0$ is the self capacitance of conductor $i$ (its charge when it sits at 1 V and everything else is grounded), and the off-diagonal entries are negative, $C_{ij} = -C^{\\rm mutual}_{ij} < 0$. Every signal row sums to a small non-negative number (the field lines that escape to infinity), so $C_{ii} \\geq \\sum_{j \\neq i} |C_{ij}|$.

A circuit Hamiltonian needs every entry at once: the charging energy of island $i$ is $E_{C,i} = \\frac{e^2}{2}\\,(C^{-1})_{ii}$, where $C$ is the signal block (ground removed), and an inverse mixes every entry of the matrix. One two-terminal simulation per pair does not give us that.

Throughout SQuADDS the matrix is in **fF**, with **ground first** (row and column 0, named `G`).""",
    ),
    (
        "md",
        """The planar capacitance models in SQuADDS learn the map from a GDS layout to this full matrix. They were trained on about 45,000 Ansys Q3D simulations of superconducting-circuit components on a silicon substrate ($\\varepsilon_r = 11.45$, 350 µm thick, zero-thickness metal). They read two GDS layers: the **ground plane on layer 1/0** and **every signal conductor on layer 1/10**. Each connected body on 1/10 is one conductor; nets are never merged.

## Setting up

We'll use Qiskit Metal without its GUI, so let's tell it that before importing anything.""",
    ),
    (
        "py",
        """import os

os.environ["QISKIT_METAL_HEADLESS"] = "1"

import warnings

warnings.filterwarnings("ignore")""",
    ),
    (
        "py",
        """import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd""",
    ),
    ("md", "We'll keep every file we write in one folder next to this notebook."),
    ("py", 'work = Path("tutorial11_files")\nwork.mkdir(exist_ok=True)'),
    (
        "md",
        """## Drawing a multipad star

`SQuADDS` ships a Qiskit Metal component for every family of the planar capacitance dataset. The multipad star is `GeneralizedCapStar`: `pad_enables` switches a pad on (`1`) or off (`0`) in each valley, and the star has a short feed line (`cpw_length`) to the north.""",
    ),
    ("py", "from qiskit_metal import designs\n\nfrom squadds.components import GeneralizedCapStar"),
    (
        "md",
        """Importing SQuADDS also loads scqubits, which switches the inline figures of the notebook to vector formats. Let's switch them back to PNG so the plots stay light.""",
    ),
    ("py", 'from matplotlib_inline.backend_inline import set_matplotlib_formats\n\nset_matplotlib_formats("png")'),
    (
        "py",
        """star_options = dict(
    n_points="5",
    star_outer_radius="136.4um",
    star_inner_radius="59.04um",
    tip_radius="2.73um",
    valley_radius="8.8um",
    star_rotation="54",
    gap="18um",
    pocket_radius="160.35um",
    pad_side_gap="6um",
    pad_enables="1,1,0,1,0",
    cpw_length="30um",
    cpw_width="10um",
    cpw_gap="6um",
)""",
    ),
    (
        "py",
        """star_design = designs.DesignPlanar({}, overwrite_enabled=True)
GeneralizedCapStar(star_design, "star", options=star_options)
star_design.rebuild()""",
    ),
    (
        "md",
        """Now let's export it with `squadds.layouts.qmetal_gds`. With `include_ground_domain=True` it writes the layer convention the models read: a square ground plane on 1/0 that reaches 169 µm past the metal, with the component's own etch cut out of it, and every piece of signal metal on 1/10. `minimum_ground_clearance_um` only closes the open end of the feed line (we use the CPW gap, 6 µm).""",
    ),
    ("py", "from squadds.layouts.qmetal_gds import export_qgeometry_gds"),
    (
        "py",
        """star_gds = work / "multipad_star.gds"
export_qgeometry_gds(star_design, star_gds, include_ground_domain=True, minimum_ground_clearance_um=6)
star_gds""",
    ),
    (
        "md",
        """## Look at the metal first

Before we ask for any number, let's look at what the model will actually see. `prepare_layout` reads a GDS exactly the way the predictor does: the ground is one net, and every connected body on the conductor layer is one conductor, named `C1`, `C2`, ... in file order.""",
    ),
    ("py", "from squadds.ml.capacitance import prepare_layout"),
    ("py", "star_layout = prepare_layout(star_gds)\nstar_layout.names"),
    ("md", "Here is a small helper that draws the ground in grey and every conductor in its own color."),
    (
        "py",
        """def plot_layout(layout, labels=None, title="", show_names=True, ax=None):
    \"\"\"Ground in grey (its etch in white), every conductor in its own color, labelled with its name.\"\"\"
    ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
    for poly in getattr(layout.ground, "geoms", [layout.ground]):
        ax.fill(*poly.exterior.xy, color="0.82", lw=0)
        for hole in poly.interiors:
            ax.fill(*hole.xy, color="white", lw=0)
    colors = plt.cm.tab10.colors
    for k, (name, body) in enumerate(zip(layout.names, layout.conductors)):
        for poly in getattr(body, "geoms", [body]):
            ax.fill(*poly.exterior.xy, color=colors[k % 10], lw=0)
        if show_names:
            p = body.representative_point()
            ax.annotate((labels or {}).get(name, name), (p.x, p.y), ha="center", fontsize=9, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)
    return ax""",
    ),
    ("py", 'plot_layout(star_layout, title="multipad star: what the model sees");'),
    (
        "md",
        """Four conductors and the ground: the star (with its feed line) and three pads. `C1`, `C2`, ... are not very telling, so let's name every body after the Qiskit Metal shape it overlaps the most.""",
    ),
    (
        "py",
        """from shapely import affinity


def name_conductors(design, layout):
    \"\"\"{predictor name: Qiskit Metal shape name} for every conductor body (ground -> "ground").\"\"\"
    shapes = {}
    for table in ("poly", "path"):
        df = design.qgeometry.tables[table]
        for row in df[~df["subtract"].astype(bool) & ~df["helper"].astype(bool)].itertuples():
            g = row.geometry if table == "poly" else row.geometry.buffer(row.width / 2, cap_style=2)
            shapes[row.name] = affinity.scale(g, 1000, 1000, origin=(0, 0))  # Metal mm -> GDS µm
    names = {"G": "ground"}
    for name, body in zip(layout.names, layout.conductors):
        names[name] = max(shapes, key=lambda s: body.intersection(shapes[s]).area)
    return names""",
    ),
    ("py", "star_names = name_conductors(star_design, star_layout)\nstar_names"),
    ("py", 'plot_layout(star_layout, labels=star_names, title="multipad star");'),
    (
        "md",
        """Great! That is the metal we expect, and no conductor touches the ground plane. Now we can ask for numbers.

## Predicting the capacitance matrix

The models live on Hugging Face (`SQuADDS/QCap`). `load_model` downloads the checkpoints once, checks the sha256 of every file against the manifest shipped inside SQuADDS and refuses anything that does not match. If you already have a local copy of the models, point the environment variable `SQUADDS_CAPML_MODELS` at its folder and nothing is downloaded.

There are two models, both trained on **all** the data: `QCap` (the default) and `QCap-Fast`.""",
    ),
    ("py", "from squadds.ml.capacitance import CapacitancePredictor, list_models, load_model"),
    ("py", 'accurate = load_model("QCap", tier="accurate")\naccurate.key, accurate.title'),
    (
        "md",
        """A `CapacitancePredictor` wraps a model. `threads` is the number of CPU threads per prediction (2 by default; for a given machine and thread count the results are deterministic).""",
    ),
    ("py", "predictor = CapacitancePredictor(accurate, threads=2)\npredictor"),
    ("py", "res = predictor.predict(star_gds)\nres"),
    (
        "md",
        "`res.to_dataframe()` gives the matrix as a table (fF, ground first). Let's put our names on it.",
    ),
    ("py", "C = res.to_dataframe().rename(index=star_names, columns=star_names)\nC.round(3)"),
    (
        "md",
        """The diagonal holds the self capacitance of every island, the off-diagonal entries are minus the mutual capacitances, and the ground row and column come first, exactly as a Q3D export would give them.

Let's look at the mutual capacitances as a picture, on a log scale, since they span several decades.""",
    ),
    (
        "py",
        """mut = -C.where(~np.eye(len(C), dtype=bool))
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(np.log10(mut), cmap="viridis")
ax.set_xticks(range(len(C)), C.columns, rotation=45, ha="right")
ax.set_yticks(range(len(C)), C.index)
for i in range(len(C)):
    for j in range(len(C)):
        if i != j:
            ax.text(j, i, f"{mut.iloc[i, j]:.3g}", ha="center", va="center", color="w", fontsize=8)
fig.colorbar(im, label="log10 mutual capacitance (fF)")
ax.set_title("multipad star: mutual capacitances (fF)");""",
    ),
    (
        "md",
        """### Physical checks

Every prediction comes with the checks a physical Maxwell matrix must pass:

- `symmetric`: $C_{ij} = C_{ji}$,
- `diagonal_positive` and `offdiag_all_negative`: the Maxwell signs,
- `closure`: every signal row sums to a value $\\geq 0$,
- `positive_definite`: the signal block has only positive eigenvalues (otherwise the electrostatic energy $\\frac{1}{2} V^T C V$ could be negative).""",
    ),
    ("py", "pd.Series(res.checks)"),
    (
        "md",
        """### Timing

`res.timing_s` breaks the wall time into the stages of the pipeline: reading the layout, the geometric embedding (`geometry_s`), the physics prior (`prior_s`, two thin-film field solves) and the networks (`network_s`).""",
    ),
    ("py", 'pd.Series(res.timing_s).drop("cache_hits")'),
    ("py", "res.machine, res.threads"),
    (
        "md",
        """The very first prediction in a fresh Python session also compiles the field solver (numba), so it is slower than the ones that follow. Let's clear the in-memory caches and time it again.""",
    ),
    ("py", "from squadds.ml.capacitance.predictor import clear_cache"),
    (
        "py",
        """clear_cache()
res = predictor.predict(star_gds)
print(f"{res.timing_s['total_s']:.2f} s with {res.threads} threads on {res.machine}")""",
    ),
    ("md", "And asking for the same layout again comes straight from the cache:"),
    (
        "py",
        """again = predictor.predict(star_gds)
print(f"{1e3 * again.timing_s['total_s']:.1f} ms, served from the cache: {again.cached}")""",
    ),
    (
        "md",
        """## How good is it?

The models ship with their reference numbers. One thing to keep in mind when reading them: QCap is trained on **all** the data, test layouts included, so that you get the best model we can make. That means QCap has **no held-out test score of its own**. The honest estimate of what to expect is the score of the same recipe with networks trained on the 35,354 TRAIN layouts only, scored on layouts it never saw. Errors are the layout-mean of the median relative error over the mutual capacitances, with the mean absolute error (MAE) and root-mean-square error (RMSE) in fF.""",
    ),
    (
        "py",
        """splits = {
    "val": "validation",
    "test_infamily": "test: new designs of trained families",
    "test_zeroshot": "test: layouts outside the trained families",
    "heldout_family": "a whole family held out of training",
}
rows = []
for m in list_models():
    if m["model"] != "QCap":
        continue
    for key, label in splits.items():
        r = m["reference"][key]
        rows.append({"tier": m["tier"], "scored on": label, "median %": r["median_pct"],
                     "MAE (fF)": r["mae_ff"], "RMSE (fF)": r["rmse_ff"]})
pd.DataFrame(rows)""",
    ),
    (
        "py",
        "for m in list_models():\n    if m[\"model\"] == \"QCap\":\n        print(f\"{m['tier']}: {m['reference']['recipe']}\\n\")",
    ),
    (
        "md",
        """So on new designs of a component family it was trained on, the Accurate recipe has a median error of 0.177 % (MAE 0.0203 fF, RMSE 0.025 fF), and on layouts outside the trained families 0.587 % (MAE 1.09 fF, RMSE 1.15 fF; those layouts are much larger, hence the larger absolute errors). The reference wall times were measured on an x86 cluster with 2 CPU threads:""",
    ),
    ("py", 'pd.Series({m["tier"]: m["time_x86"] for m in list_models() if m["model"] == "QCap"})'),
    (
        "md",
        """## Accurate vs Fast

The Fast tier replaces the two field solves of the physics prior by a learned surrogate, so it needs no field solve at all. That makes it about ten times faster, but it comes with a warning that we should read before using it:""",
    ),
    ("py", 'fast = load_model("QCap", tier="fast")\nprint(fast.warning)'),
    (
        "md",
        "Let's predict the star with Fast too (once to load it, then once more from cold caches for a fair time).",
    ),
    (
        "py",
        """fast_predictor = CapacitancePredictor(fast, threads=2)
fast_predictor.predict(star_gds)
clear_cache()
res_fast = fast_predictor.predict(star_gds)
res_fast""",
    ),
    ("md", "Here is a small helper that lists the mutual capacitances of a matrix, one row per pair of conductors."),
    (
        "py",
        """def mutuals(C, names):
    \"\"\"The mutual capacitances (fF, positive) of a Maxwell matrix, one entry per pair of conductors.\"\"\"
    C = np.asarray(C)
    return pd.Series({f"{names[i]} to {names[j]}": -C[i, j] for i in range(len(names)) for j in range(i + 1, len(names))})""",
    ),
    (
        "py",
        """names = [star_names[n] for n in res.names]
tiers = pd.DataFrame({"Accurate (fF)": mutuals(res.matrix_fF, names), "Fast (fF)": mutuals(res_fast.matrix_fF, names)})
tiers["Fast vs Accurate (%)"] = 100 * (tiers["Fast (fF)"] / tiers["Accurate (fF)"] - 1)
tiers.round(4)""",
    ),
    (
        "py",
        """pd.Series({"Accurate": res.timing_s["total_s"], "Fast": res_fast.timing_s["total_s"]}, name="wall time (s)")""",
    ),
    (
        "md",
        """Fast took 0.20 s against 2.25 s for Accurate (2 threads on an Intel Xeon E5-2640 v4 cluster node), about eleven times faster. It is not the same answer, though: on this star it differs from Accurate by 2.4 to 3.6 % on the star-to-pad couplings and by 4.6 to 6.9 % on the small pad-to-pad ones. The star is one of the families Fast was trained on, and still its reference numbers above are three to four times worse than Accurate's. A good workflow is to screen many candidates with Fast and to confirm the ones you keep with Accurate.

## Physics priors vs the model

Under the hood, Accurate starts from physics. A thin-film method-of-moments solver computes the capacitance matrix of the layout on a mesh with $p$ panels per gap, and the networks learn only the small residual that the thin-film physics misses. SQuADDS exposes the two physics priors on their own:

- `physics_prior`: one solve at $p = 0.375$ panels per gap (cheap and rough),
- `optimized_physics_prior`: two solves at $p = 1.5$ and $p = 3$, extrapolated entry by entry, $C^{R} = \\exp\\left(2 \\log C_{3} - \\log C_{1.5}\\right)$. This is the prior the Accurate tier starts from.

Neither has a learned parameter. Let's compute both from cold caches and time them.""",
    ),
    ("py", "from squadds.ml.capacitance import physics_prior, optimized_physics_prior"),
    (
        "py",
        """clear_cache()
coarse = physics_prior(star_gds, threads=2)
clear_cache()
optimized = optimized_physics_prior(star_gds, threads=2)""",
    ),
    (
        "py",
        """ladder = pd.DataFrame({
    "coarse prior (fF)": mutuals(coarse.C_fF, names),
    "optimized prior (fF)": mutuals(optimized.C_fF, names),
    "Accurate model (fF)": mutuals(res.matrix_fF, names),
})
ladder["network correction (%)"] = 100 * (ladder["Accurate model (fF)"] / ladder["optimized prior (fF)"] - 1)
ladder.round(4)""",
    ),
    (
        "py",
        """pd.Series({
    "coarse prior": coarse.timing_s["total_s"],
    "optimized prior": optimized.timing_s["total_s"],
    "Accurate (embedding + prior + networks)": res.timing_s["total_s"],
    "Fast (no field solve)": res_fast.timing_s["total_s"],
}, name=f"wall time (s), {res.threads} threads")""",
    ),
    (
        "md",
        """The coarse prior takes 0.10 s, but it sits 3 to 8 % below the optimized prior on most entries (for example 17.30 fF against 18.80 fF from the ground to `pad_0`). The optimized prior takes 1.8 s, most of the Accurate tier's 2.25 s. The networks then correct it by less than 0.3 % on the star-to-pad couplings, by 0.1 to 1.6 % on the couplings to ground, and by about 8 % on the two smallest entries (0.11 fF to 0.12 fF, an absolute change of 0.009 fF). Without a Q3D simulation of this exact star we cannot say from this table alone which column is right; the reference numbers above (and Tutorial 14, where we score the model on Q3D data) are what tell us that.

## A second component: an Xmon with two claws

Let's draw a second device with a different number of conductors: an Xmon cross (`Xmon` in `squadds.components`, the cross without its junction) with a readout claw on the north arm and a bus claw on the west arm (`connector_location="0"`).""",
    ),
    ("py", "from squadds.components import Xmon"),
    (
        "py",
        """claw = dict(connector_type="0", claw_width="19.6um", claw_gap="8.07um", ground_spacing="10.9um",
            claw_cpw_length="26.5um", claw_cpw_width="10um")
xmon_options = dict(
    cross_length="303.7um",
    cross_width="22.6um",
    cross_gap="28.3um",
    connection_pads=dict(
        readout=dict(connector_location="90", claw_length="82.3um", **claw),
        bus=dict(connector_location="0", claw_length="121.5um", **claw),
    ),
)""",
    ),
    (
        "py",
        """xmon_design = designs.DesignPlanar({}, overwrite_enabled=True)
Xmon(xmon_design, "xmon", options=xmon_options)
xmon_design.rebuild()
xmon_gds = work / "xmon_two_claws.gds"
export_qgeometry_gds(xmon_design, xmon_gds, include_ground_domain=True, minimum_ground_clearance_um=8)""",
    ),
    (
        "py",
        """xmon_layout = prepare_layout(xmon_gds)
xmon_names = name_conductors(xmon_design, xmon_layout)
plot_layout(xmon_layout, labels=xmon_names, title="Xmon with two claws");""",
    ),
    (
        "md",
        """And a third one that is much bigger: a five-qubit lattice from the examples that ship with this tutorial. It has 34 conductors plus ground, so it is a $35 \\times 35$ matrix with 595 mutual capacitances.""",
    ),
    (
        "py",
        """lattice_gds = Path("data/tutorial11/lattice_5q.gds")
lattice_layout = prepare_layout(lattice_gds)
print(len(lattice_layout.names), "conductors plus ground")
plot_layout(lattice_layout, title="five-qubit lattice", show_names=False);""",
    ),
    (
        "md",
        """## Batch prediction and threads

`predict` also takes a list of layouts and returns a list of results.""",
    ),
    ("py", "layouts = [star_gds, xmon_gds, lattice_gds]"),
    (
        "py",
        """clear_cache()
t0 = time.perf_counter()
batch = predictor.predict(layouts)
serial_s = time.perf_counter() - t0""",
    ),
    (
        "py",
        """pd.DataFrame({
    "layout": [Path(p).name for p in layouts],
    "conductors with ground": [r.n_conductors for r in batch],
    "total (s)": [r.timing_s["total_s"] for r in batch],
    "checks passed": [r.checks["passed"] for r in batch],
})""",
    ),
    (
        "md",
        """With `workers=3`, the three layouts are predicted in three processes at once, each with `threads` threads. Starting the processes costs a few seconds, and here the lattice alone takes most of the time, so do not expect miracles on this list: workers pay off for long lists of layouts of similar size.""",
    ),
    (
        "py",
        """clear_cache()
t0 = time.perf_counter()
batch_parallel = predictor.predict(layouts, workers=3)
parallel_s = time.perf_counter() - t0
print(f"serial: {serial_s:.1f} s, 3 workers: {parallel_s:.1f} s (wall time, including starting the workers)")""",
    ),
    (
        "py",
        """same = all(np.allclose(a.matrix_fF, b.matrix_fF, rtol=1e-6, atol=0) for a, b in zip(batch, batch_parallel))
print("identical matrices:", same)""",
    ),
    (
        "md",
        "For one big layout, it is the threads that help. Let's predict the lattice with 1, 2 and 4 threads, from cold caches each time.",
    ),
    (
        "py",
        """rows = []
for t in (1, 2, 4):
    clear_cache()
    r = predictor.predict(lattice_gds, threads=t)
    rows.append({"threads": t, "total (s)": r.timing_s["total_s"], "prior (s)": r.timing_s["prior_s"],
                 "network (s)": r.timing_s["network_s"]})
pd.DataFrame(rows).round(2)""",
    ),
    (
        "md",
        """On this node (Intel Xeon E5-2640 v4), the 35-conductor lattice takes 52 s with 1 thread, 35 s with 2 and 22 s with 4. Nearly all of it is the field solve of the physics prior; the networks take about 10 ms.""",
    ),
    (
        "md",
        """## From a matrix to a charging energy

Let's put the Xmon's matrix to work. With our names on it:""",
    ),
    (
        "py",
        """res_xmon = batch[1]
Cx = res_xmon.to_dataframe().rename(index=xmon_names, columns=xmon_names)
Cx.round(3)""",
    ),
    (
        "md",
        """The charging energy of the cross depends on what the claws are connected to. Two textbook limits:

- the claws are held at ground (a low-impedance line): $E_C = \\frac{e^2}{2 C_\\Sigma}$ with $C_\\Sigma = C_{\\rm cross, cross}$, the diagonal entry,
- the claws are floating islands: $E_C = \\frac{e^2}{2} (C^{-1})_{\\rm cross, cross}$ with the inverse of the full signal block.

Both need the full matrix row of the cross, and the second needs every entry.""",
    ),
    (
        "py",
        """from scipy.constants import e, h

signal = Cx.drop(index="ground", columns="ground")
i = list(signal.index).index("cross")
EC_grounded_MHz = e**2 / (2 * signal.iloc[i, i] * 1e-15) / h / 1e6
EC_floating_MHz = e**2 / 2 * np.linalg.inv(signal.to_numpy() * 1e-15)[i, i] / h / 1e6
print(f"E_C/h = {EC_grounded_MHz:.1f} MHz (claws grounded), {EC_floating_MHz:.1f} MHz (claws floating)")""",
    ),
    (
        "md",
        """The two limits differ by only about 0.1 MHz here because the claws couple weakly to the cross (1.9 fF and 2.8 fF, against a self capacitance of 152 fF). On a design with strong couplers they would not, and only the full matrix tells you which case you are in.""",
    ),
    (
        "md",
        """## A prediction as a SQuADDS_DB row

`to_sim_results` returns the `sim_results` shape of the capacitance rows of `SQuADDS_DB`: positive magnitudes `<a>_to_<b>` (the self capacitances on the diagonal, the ground named `ground`), the signed Maxwell matrix as `C_<a>_<b>`, and the units. This means the calculators in `squadds.calcs` take an ML prediction the same way they take a Q3D simulation.""",
    ),
    (
        "py",
        """sim = res.to_sim_results(names={k: v for k, v in star_names.items() if k != "G"})
pd.Series({k: v for k, v in sim.items() if k != "maxwell_matrix"})""",
    ),
    ("py", 'pd.Series(sim["maxwell_matrix"]).head(8)'),
    (
        "md",
        """## From the command line

Everything above is also one shell command away:

```bash
squadds-ml predict my_chip.gds                      # QCap, Accurate tier, CSV to stdout
squadds-ml predict a.gds b.gds --workers 2          # several files
squadds-ml predict my_chip.gds --tier fast --format json
```

Let's try it on our star with the Fast tier.""",
    ),
    ("py", "!squadds-ml predict tutorial11_files/multipad_star.gds --tier fast"),
    (
        "md",
        """## The geometric embedding

The networks do not read polygons. They read the **geometric embedding** of the layout (`planar-capacitance-graph-v1`): one vector of 60 shape and context features per conductor, one vector of 42 features per pair of conductors, and 4 global features of the chip. It is available on its own in `squadds.layouts`, for example to compare designs or to train your own models.""",
    ),
    ("py", "from squadds.layouts import CAPACITANCE_GRAPH_VERSION, encode_capacitance_graph"),
    (
        "py",
        """emb = encode_capacitance_graph(star_gds)
CAPACITANCE_GRAPH_VERSION, {k: v.shape for k, v in emb.items() if hasattr(v, "shape")}""",
    ),
    (
        "py",
        """nodes = pd.DataFrame(emb["node_features"], index=[star_names[n] for n in emb["names"]],
                     columns=emb["node_feature_names"])
nodes.iloc[:, :8].round(3)""",
    ),
    (
        "py",
        """z = (nodes - nodes.mean()) / (nodes.std() + 1e-12)
fig, ax = plt.subplots(figsize=(12, 2.6))
im = ax.imshow(z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax.set_yticks(range(len(z)), z.index)
ax.set_xlabel("conductor feature index")
fig.colorbar(im, label="z-score across conductors")
ax.set_title("multipad star: the 60 conductor features");""",
    ),
    (
        "md",
        """Congrats for making it to the end of this tutorial! 🤗🎉 You have now learned how to predict the full Maxwell capacitance matrix of any planar layout with SQuADDS: we drew a multipad star and an Xmon in Qiskit Metal, looked at their conductors, predicted their matrices with the Accurate and Fast tiers, compared them with the physics priors, predicted a 35-conductor lattice, and turned a prediction into a charging energy and a `SQuADDS_DB` row.

## Next Steps...

How do you know the model is good enough for **your** designs? In [Tutorial 14](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-14_Training_on_Your_Own_Designs.html), we will score it against Q3D simulations of our own designs and learn when (and how) to train on them.""",
    ),
    ("md", LICENSE),
]


def _cell(kind: str, src: str, index: int) -> dict:
    cid = hashlib.sha1(f"{index}:{src}".encode()).hexdigest()[:12]
    if kind == "md":
        return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}
    return {"cell_type": "code", "id": cid, "metadata": {}, "source": src, "execution_count": None, "outputs": []}


def _executed(cells: list[dict], outputs_from: Path | None) -> list[dict]:
    """Copy outputs and execution counts from an executed copy of the notebook, cell by cell, where the code is unchanged
    (so prose can be edited without re-running the notebook)."""
    if outputs_from is None:
        return cells
    done = {}
    for c in json.loads(Path(outputs_from).read_text())["cells"]:
        if c["cell_type"] == "code":
            src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            done.setdefault(src, c)
    for c in cells:
        if c["cell_type"] == "code" and c["source"] in done:
            c["outputs"] = done[c["source"]]["outputs"]
            c["execution_count"] = done[c["source"]]["execution_count"]
    return cells


def build(cells, path: Path, outputs_from: Path | None = None) -> Path:
    nb = {
        "cells": _executed([_cell(k, s, i) for i, (k, s) in enumerate(cells)], outputs_from),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    try:
        import nbformat

        nbformat.validate(nbformat.from_dict(nb))
    except ImportError:
        pass
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    print(build(CELLS, OUT, Path(sys.argv[1]) if len(sys.argv) > 1 else None))
