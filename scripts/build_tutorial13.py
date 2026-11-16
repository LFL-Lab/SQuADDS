#!/usr/bin/env python
"""Build ``tutorials/Tutorial-13_From_Layout_to_Hamiltonian_with_ML.ipynb`` (unexecuted).

Execute it afterwards from ``tutorials/`` (e.g. ``jupyter nbconvert --to notebook --execute --inplace``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorials" / "Tutorial-13_From_Layout_to_Hamiltonian_with_ML.ipynb"
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
        r"""# Tutorial 13: From Layout to Hamiltonian with ML

In this tutorial, we will walk through the design flow of [Tutorial 1](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-1_Getting_Started_with_SQuADDS.html) and [Tutorial 2](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-2_Simulate_interpolated_designs.html) again, but we will replace the Ansys Q3D capacitance simulation with a machine-learning prediction that takes about a second. By the end of this tutorial, you will be able to:

- Query SQuADDS for a qubit-cavity design that hits a target qubit frequency, anharmonicity and readout coupling

- Render the "best-guess" design in Qiskit Metal (no GUI needed) and look at its metal and ports before trusting any number

- Predict its full capacitance matrix with `MLSimulator`, and know how far to trust that prediction

- Turn the prediction into $E_C$, $f_q$, $\alpha$ and $g$ with `squadds.calcs`, and compare them with the Q3D numbers stored in the database

- Check an interpolated design that nobody has simulated yet with `Analyzer.predict_capacitance`

- Query the new qubit and coupler families of the planar capacitance dataset (`Xmon`, `TransmonPocket`, `GeneralizedCapStar`)

**Note:** the ML models need the optional extras, `pip install "squadds[ml,gds]"`.""",
    ),
    ("py", "%load_ext autoreload\n%autoreload 2\n%matplotlib inline"),
    (
        "md",
        """We'll use Qiskit Metal without its GUI in this tutorial, so let's tell it that before importing anything. We'll also silence the (many) deprecation warnings that Qiskit Metal and Shapely like to print.""",
    ),
    (
        "py",
        """import logging
import os
import warnings

os.environ.setdefault("QISKIT_METAL_HEADLESS", "1")
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)""",
    ),
    (
        "md",
        r"""## The design problem

Let's say we want a transmon with

- a qubit frequency $f_q = 4$ GHz and an anharmonicity $\alpha = -200$ MHz,
- read out through a quarter-wave resonator at $f_r = 6.2$ GHz with $\kappa = 120$ kHz,
- with a qubit-resonator coupling $g = 70$ MHz.

Every one of these numbers except $f_r$ and $\kappa$ is set by **capacitance**. For a grounded transmon (an Xmon-style cross) with total capacitance $C_\Sigma$,

$$E_C = \frac{e^2}{2 C_\Sigma}, \qquad f_q \approx \sqrt{8 E_J E_C} - E_C, \qquad \alpha \approx -E_C ,$$

and the coupling grows with the claw-to-cross capacitance $C_g$ roughly as $g \propto (C_g / C_\Sigma)\, (E_J/E_C)^{1/4}$.

In Tutorial 2, checking a design meant an Ansys Q3D run: a license, a Windows machine, and minutes per design. Here we'll ask a model trained on about 45,000 Q3D simulations instead.""",
    ),
    (
        "md",
        """## Query the SQuADDS Database

Let's first find the closest qubit-cavity design in SQuADDS, exactly as in Tutorial 1.""",
    ),
    ("py", "from squadds import Analyzer, SQuADDS_DB"),
    (
        "md",
        """Importing SQuADDS (through scqubits) switches the inline figures to a vector format, so let's switch them back to PNG.""",
    ),
    ("py", 'from matplotlib_inline.backend_inline import set_matplotlib_formats\n\nset_matplotlib_formats("png")'),
    (
        "py",
        """db = SQuADDS_DB()
db.select_system(["qubit", "cavity_claw"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("quarter")
db.show_selections()""",
    ),
    ("py", "df = db.create_system_df()\nanalyzer = Analyzer(db)"),
    ("md", "Our Hamiltonian targets:"),
    (
        "py",
        """target_params = {
    "qubit_frequency_GHz": 4,
    "cavity_frequency_GHz": 6.2,
    "kappa_kHz": 120,
    "resonator_type": "quarter",
    "anharmonicity_MHz": -200,
    "g_MHz": 70,
}""",
    ),
    (
        "py",
        """results = analyzer.find_closest(target_params=target_params, num_top=3, metric="Euclidean", display=True)
results""",
    ),
    (
        "md",
        """Let's keep the closest design and see what its own (Q3D-based) numbers are. `EC` and `EJ` are in GHz, and `cross_to_claw` / `cross_to_ground` are the capacitances (fF) that Q3D gave the contributor of this row.""",
    ),
    (
        "py",
        """best = results.iloc[0]
best[["qubit_frequency_GHz", "anharmonicity_MHz", "g_MHz", "cavity_frequency_GHz", "EC", "EJ",
      "cross_to_claw", "cross_to_ground"]]""",
    ),
    ("md", "The qubit part of the design lives in `design_options_qubit`:"),
    ("py", 'qubit_options = best["design_options_qubit"]\nqubit_options'),
    (
        "md",
        """## Render the design in Qiskit Metal

We pass these options to a `TransmonCross` (as in Tutorial 1) and rebuild the design. No GUI window opens since we are running headless.""",
    ),
    (
        "py",
        """from qiskit_metal import designs
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross""",
    ),
    (
        "py",
        """design = designs.DesignPlanar({}, overwrite_enabled=True)
qubit = TransmonCross(design, "qubit", options=qubit_options)
design.rebuild()""",
    ),
    (
        "md",
        """The capacitance models read a GDS with two layers: the **ground plane on layer 1/0** and **every signal conductor on layer 1/10** (each connected body is one conductor; nets are never merged). `render_gds` exports a Qiskit Metal design in that convention; it is the same rendering `MLSimulator` uses internally.""",
    ),
    (
        "py",
        """from pathlib import Path

from squadds.ml.capacitance.bridge import render_gds

Path("tutorial13_files").mkdir(exist_ok=True)
qubit_gds = render_gds(design, "tutorial13_files/transmon_cross.gds")
qubit_gds""",
    ),
    (
        "md",
        """## Look at the metal (and the ports) first

Before we ask for any number, let's look at what the model will actually see. `prepare_layout` reads a GDS exactly the way the predictor does: the ground is one net, and every conductor body gets a name `C1`, `C2`, ... in file order.""",
    ),
    (
        "py",
        """from squadds.ml.capacitance import prepare_layout

qubit_layout = prepare_layout(str(qubit_gds))
qubit_layout.names""",
    ),
    (
        "md",
        """`C1` and `C2` are not very telling, so here is a small helper that names every body after the Qiskit Metal shape it overlaps the most.""",
    ),
    (
        "py",
        """from shapely import affinity


def name_conductors(design, layout):
    \"\"\"{predictor name: Qiskit Metal shape name} for every conductor body.\"\"\"
    shapes = {}
    for table in ("poly", "path"):
        df = design.qgeometry.tables[table]
        for row in df[~df["subtract"].astype(bool) & ~df["helper"].astype(bool)].itertuples():
            g = row.geometry if table == "poly" else row.geometry.buffer(row.width / 2, cap_style=2)
            shapes[row.name] = affinity.scale(g, 1000, 1000, origin=(0, 0))  # Metal mm -> GDS µm
    names = {}
    for name, body in zip(layout.names, layout.conductors):
        names[name] = max(shapes, key=lambda s: body.intersection(shapes[s]).area)
    return names""",
    ),
    (
        "md",
        """And one that draws the ground in grey (its etch in white), every conductor in its own color, and the Qiskit Metal **pins** of the component as red arrows. The pins are the ports: they mark where a resonator, a feed line or a bus attaches in the full chip.""",
    ),
    (
        "py",
        """import matplotlib.pyplot as plt


def plot_layout(layout, names=None, pins=None, title="", ax=None):
    \"\"\"Ground in grey, every conductor in its own color, and the component's pins (ports) as red arrows.\"\"\"
    ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
    for poly in getattr(layout.ground, "geoms", [layout.ground]):
        ax.fill(*poly.exterior.xy, color="0.82", lw=0)
        for hole in poly.interiors:
            ax.fill(*hole.xy, color="white", lw=0)
    for k, (name, body) in enumerate(zip(layout.names, layout.conductors)):
        for i, poly in enumerate(getattr(body, "geoms", [body])):
            label = f"{name}: {(names or {}).get(name, name)}" if i == 0 else None
            ax.fill(*poly.exterior.xy, color=plt.cm.tab10(k % 10), lw=0, label=label)
    for pin_name, pin in (pins or {}).items():
        (x, y), (dx, dy) = 1e3 * pin["middle"], 60 * pin["normal"]
        ax.annotate(f"port '{pin_name}'", xy=(x, y), xytext=(x + dx, y + dy), color="crimson", fontsize=9,
                    arrowprops=dict(arrowstyle="<-", color="crimson"))
    ax.set_aspect("equal")
    ax.set(xlabel="x (µm)", ylabel="y (µm)", title=title)
    ax.legend(loc="lower left", fontsize=8)
    return ax""",
    ),
    (
        "py",
        """qubit_names = name_conductors(design, qubit_layout)
plot_layout(qubit_layout, qubit_names, qubit.pins, title="TransmonCross: what the model sees");""",
    ),
    (
        "md",
        """Great! Two conductors and the ground: the **cross** (the qubit island; its junction goes from the cross to ground, and it is not part of the capacitance problem) and the **claw** of the readout connector. The `readout` port is where the resonator attaches. That is the metal we expect, so now we can ask for numbers.""",
    ),
    (
        "md",
        r"""## Predict the capacitance matrix with `MLSimulator`

`MLSimulator` has the same call and the same result shape as the capacitance path of `AnsysSimulator` (Tutorial 2), so the rest of the SQuADDS workflow does not care where the numbers came from. Every result carries a **fidelity label**, `"ml:accurate"` or `"ml:fast"`, so a prediction can never be mistaken for a simulation.

**How far can we trust it?** The released model (*QCap*) runs two thin-film method-of-moments field solves and corrects them with a small graph neural network. It was trained on **all** the data, so it has no held-out score of its own. Its recipe, trained on the TRAIN split only (the *paper* model), scores against Q3D (median relative error over the coupling entries, with MAE and RMSE):

| test set | median % | MAE (fF) | RMSE (fF) |
|---|---|---|---|
| test, families seen in training (4,009 layouts) | 0.177 % | 0.0203 | 0.025 |
| a whole family held out of training (4,071 layouts) | 0.408 % | 0.049 | 0.059 |
| zero-shot, unseen kinds of layouts (1,878 layouts) | 0.587 % | 1.092 | 1.152 |
| fresh check of QCap on 39 new layouts | 1.05 % | 0.132 | 0.188 |

It costs about 1 s per small component (median 0.845 s on 2 CPUs of an x86 cluster node, 1.37 s on an Apple M5 Pro laptop for up to 4 conductors). The model serves one stack: silicon ($\varepsilon_r = 11.45$), 350 µm thick, zero-thickness metal.

**Note:** if you have a local copy of the models, point `SQUADDS_CAPML_MODELS` at its folder and nothing is downloaded.""",
    ),
    ("py", "from squadds.ml.capacitance.bridge import MLSimulator"),
    ("py", "ml = MLSimulator()\nml"),
    ("md", "We can hand it the Qiskit Metal component directly:"),
    ("py", 'ml_result = ml.predict(qubit)\nml_result["fidelity"], ml_result["timing"]'),
    (
        "md",
        """The first call of a session is slow: it loads the model and compiles the numerical kernels once (`load_s` and most of `geometry_s` above). Let's ask again: SQuADDS remembers layouts it has already seen, so a repeated design is answered from a cache almost instantly. A **new** layout takes about a second, as we'll see below.""",
    ),
    ("py", 'ml_result = ml.predict(qubit)\nround(ml_result["timing"]["wall_s"], 2)'),
    (
        "md",
        """The result has the familiar `design` / `sim_options` / `sim_results` keys of `AnsysSimulator`, with the `sim_results` named just like the SQuADDS database rows:""",
    ),
    ("py", 'ml_result["sim_results"]'),
    (
        "md",
        """It also tells us whether the predicted matrix passed its physics checks (symmetry, signs, closure, positive definiteness) and whether anything looked unusual about the layout:""",
    ),
    ("py", 'ml_result["checks"], ml_result["warnings"]'),
    (
        "md",
        r"""The full Maxwell matrix (fF) is in `sim_results["maxwell_matrix"]`, keyed by the conductor labels. Let's put it in a DataFrame with readable names. The diagonal entries are positive self capacitances and the off-diagonal entries are $-C^{\rm mutual}_{ij}$.""",
    ),
    (
        "py",
        """import pandas as pd


def maxwell_df(result):
    \"\"\"The signed Maxwell matrix (fF) of an MLSimulator result, named like the SQuADDS sim_results.\"\"\"
    nets = result["net_names"]
    M = result["sim_results"]["maxwell_matrix"]
    names = list(nets.values())
    return pd.DataFrame([[M[f"C_{a}_{b}"] for b in nets] for a in nets], index=names, columns=names)


maxwell_df(ml_result).round(3)""",
    ),
    (
        "md",
        """## How close is it to Q3D?

This design **is** in the database, so we can compare the prediction with the row's own Q3D numbers. Here is a helper that lines them up, in fF and in %, and summarizes the mutual (off-diagonal) entries with the median relative error, the MAE and the RMSE, the metric of the accuracy table above.""",
    ),
    (
        "py",
        """import numpy as np


def compare(row, result):
    \"\"\"Q3D (database row) vs ML for every capacitance both carry, in fF and in %.\"\"\"
    sr = result["sim_results"]
    keys = [k for k in sr if k in row.index and k not in ("units", "maxwell_matrix") and pd.notna(row[k])]
    out = pd.DataFrame({"Q3D (fF)": [float(row[k]) for k in keys], "ML (fF)": [sr[k] for k in keys]}, index=keys)
    out["ML - Q3D (fF)"] = out["ML (fF)"] - out["Q3D (fF)"]
    out["ML - Q3D (%)"] = 100 * out["ML - Q3D (fF)"] / out["Q3D (fF)"]
    mutual = [k for k in keys if k.split("_to_")[0] != k.split("_to_")[1]]
    d = out.loc[mutual, "ML - Q3D (fF)"]
    pct = 100 * (d / out.loc[mutual, "Q3D (fF)"]).abs()
    print(f"mutual entries: median {pct.median():.2f} % | MAE {d.abs().mean():.3f} fF | "
          f"RMSE {np.sqrt((d**2).mean()):.3f} fF | n = {len(mutual)}")
    return out.round(3)""",
    ),
    ("py", "compare(best, ml_result)"),
    (
        "md",
        r"""Before we read too much into these numbers, a few honest caveats:

- This is **one** design, so it is a sanity check, not an accuracy score (the table above is the accuracy score).
- The SQuADDS `TransmonCross` rows were simulated by their contributor with their own Q3D setup. Their rows record no layer stack; the Qiskit Metal / SQuADDS default substrate is 750 µm thick, while the model serves 350 µm. The difference above mixes the model error with this stack difference.
- The rendered chip is not the contributor's chip either: `render_gds` draws a square ground domain around the component. Entries that involve the ground feel that most; the cross-to-claw coupling is the most like-for-like number.
- In these database rows `cross_to_cross` is stored equal to `cross_to_ground`, so it is not the Maxwell diagonal the model predicts; that is why the self terms are left out of the summary line. (SQuADDS builds the `TransmonCross` $C_\Sigma$ from `cross_to_ground` and `cross_to_claw`, so the Hamiltonian below is not affected.)
- These `TransmonCross` geometries are not in the model's training data (the planar capacitance dataset has its own `Xmon` family, which we'll meet below).""",
    ),
    (
        "md",
        r"""## From capacitance to the Hamiltonian

Now let's turn the prediction into Hamiltonian parameters with the same functions SQuADDS uses for Q3D rows (`squadds.calcs.transmon_cross`). For the `TransmonCross`, SQuADDS uses $C_\Sigma = C_{\rm cross,ground} + C_{\rm cross,claw}$, so

$$E_C = \frac{e^2}{2\,(C_{\rm cross,ground} + C_{\rm cross,claw})} .$$

We keep the Josephson energy $E_J$ that SQuADDS picked for our target (the `EJ` column), get $f_q$ and $\alpha$ from scqubits, and compute $g$ from the capacitance matrix with the resonator frequency of the closest design:

$$g = \frac{C_g}{\sqrt{C_\Sigma}} \sqrt{\frac{\hbar\, \omega_r\, e^2}{\det C}} \left(\frac{E_J}{8 E_C}\right)^{1/4} \Big/ \hbar .$$""",
    ),
    (
        "py",
        """from squadds.calcs.transmon_cross import get_transmon_E01_alpha
from squadds.ml.capacitance.bridge import EC_MHz, g_MHz""",
    ),
    (
        "py",
        """EJ = float(best["EJ"])  # GHz
f_r = float(best["cavity_frequency_GHz"])
EC_ml = EC_MHz(ml_result)  # MHz
f_q_ml, alpha_ml = get_transmon_E01_alpha(EJ, EC_ml / 1e3)
g_ml = g_MHz(ml_result, EJ_GHz=EJ, f_r_GHz=f_r, res_type="quarter")""",
    ),
    ("md", "Let's line up the target, the database row (Q3D) and the ML prediction:"),
    (
        "py",
        """hamiltonian = pd.DataFrame(
    {
        "target": [None, target_params["qubit_frequency_GHz"], target_params["anharmonicity_MHz"], target_params["g_MHz"]],
        "closest design, Q3D": [1e3 * best["EC"], best["qubit_frequency_GHz"], best["anharmonicity_MHz"], best["g_MHz"]],
        "closest design, ML": [EC_ml, f_q_ml, alpha_ml, g_ml],
    },
    index=["E_C (MHz)", "f_q (GHz)", "alpha (MHz)", "g (MHz)"],
)
hamiltonian.round(3)""",
    ),
    (
        "md",
        """Awesome! In about a second we went from a layout to the qubit's Hamiltonian, and it agrees with what Q3D told the database within the differences we saw in the capacitances. A plot makes the comparison easier to read:""",
    ),
    (
        "py",
        """fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, key in zip(axes, ["f_q (GHz)", "alpha (MHz)", "g (MHz)"]):
    vals = hamiltonian.loc[key]
    ax.bar(vals.index, vals.values, color=["0.6", "tab:blue", "tab:orange"])
    ax.set_title(key)
    ax.tick_params(axis="x", rotation=20)
fig.tight_layout()""",
    ),
    (
        "md",
        """## An interpolated design that nobody has simulated

The closest design is never exactly on target, so Tutorial 1 used the physics-inspired `ScalingInterpolator` to scale the geometry towards our targets. That gives a **new** design, and in Tutorial 2 the next step was a Q3D run to check it. Let's check it with the model instead.""",
    ),
    ("py", "from squadds.interpolations.physics import ScalingInterpolator"),
    (
        "py",
        "interpolator = ScalingInterpolator(analyzer, target_params)\ndesign_df = interpolator.get_design()\ndesign_df",
    ),
    (
        "md",
        """The interpolated row holds the whole qubit-cavity system. `MLSimulator` predicts single components, so we hand `Analyzer.predict_capacitance` the qubit part and tell it which component class to build:""",
    ),
    (
        "py",
        """interpolated_qubit = pd.DataFrame(
    [{"design_options": design_df.iloc[0]["design_options_qubit"], "component_class": "TransmonCross"}]
)
interpolated_ml = analyzer.predict_capacitance(interpolated_qubit)
interpolated_ml.filter(regex="cross_to_claw_ml|cross_to_ground_ml|fidelity|ml_total_s")""",
    ),
    (
        "md",
        """Let's look at the interpolated cross next to the closest one before we trust its numbers. We'll wrap the render-and-plot steps from above in one small function, since we'll use it a few more times.""",
    ),
    (
        "py",
        """from squadds.ml.capacitance.bridge import build_component


def look_at(component, title, ax=None):
    \"\"\"Render a component to GDS, then plot its metal and ports (returns the GDS path).\"\"\"
    gds = render_gds(component.design, Path("tutorial13_files") / (title.replace(" ", "_") + ".gds"))
    layout = prepare_layout(str(gds))
    plot_layout(layout, name_conductors(component.design, layout), component.pins, title=title, ax=ax)
    return gds""",
    ),
    (
        "py",
        """interpolated_component = build_component(interpolated_qubit.iloc[0])
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
look_at(qubit, "closest design", ax=axes[0])
look_at(interpolated_component, "interpolated design", ax=axes[1]);""",
    ),
    ("md", "Now the Hamiltonian of the interpolated design, with the same $E_J$:"),
    (
        "py",
        """interpolated_result = interpolated_ml.iloc[0]["ml_result"]
EC_interp = EC_MHz(interpolated_result)
f_q_interp, alpha_interp = get_transmon_E01_alpha(EJ, EC_interp / 1e3)
g_interp = g_MHz(interpolated_result, EJ_GHz=EJ, f_r_GHz=target_params["cavity_frequency_GHz"], res_type="quarter")
hamiltonian["interpolated design, ML"] = [EC_interp, f_q_interp, alpha_interp, g_interp]
hamiltonian.round(3)""",
    ),
    (
        "md",
        """Interesting! The interpolation pulled $f_q$ and $\alpha$ right onto our targets, but by the model's numbers $g$ came out a few MHz short of 70 MHz. That is exactly the kind of thing we want to catch *before* we spend a Q3D run (a slightly longer claw would be our next try).

This is the check that used to need a Q3D run per candidate. Since it now takes about a second, we can afford to check every candidate we are curious about, and keep the Q3D (or Palace) runs for the one design we are about to fabricate.""",
    ),
    (
        "md",
        """## New families: the planar capacitance dataset

The models were trained on the **planar capacitance dataset**: about 45,000 Q3D-labelled layouts of interdigital, concentric, star and shielded capacitors, Xmon and `TransmonPocket` qubits and more, every one with its full capacitance matrix. SQuADDS can show these rows in its usual format as a second data source. It is on by default, and a local copy works too, pointed to by the environment variable `SQUADDS_PLANAR_CAP_DATA`.""",
    ),
    ("py", 'db = SQuADDS_DB(sources=("default", "planar_capacitance"))\ndb.enabled_sources()'),
    ("md", "The new configurations show up next to the existing ones:"),
    ("py", "db.get_configs()"),
    (
        "md",
        """### An Xmon with a target frequency and anharmonicity

The `Xmon` rows are a cross in a ground pocket with one or two claws. For these, SQuADDS takes $C_\\Sigma$ as the full Maxwell diagonal of the cross, so a second claw is counted too.""",
    ),
    (
        "py",
        """db.unselect_all()
db.select_system("qubit")
db.select_qubit("Xmon")
xmon_df = db.create_system_df()
xmon_df.shape""",
    ),
    (
        "py",
        """analyzer = Analyzer(db)
xmon_results = analyzer.find_closest({"qubit_frequency_GHz": 4.5, "anharmonicity_MHz": -200}, num_top=3)
xmon_results[["design_id", "qubit_frequency_GHz", "anharmonicity_MHz", "EC", "C_sigma_fF", "cross_to_claw"]]""",
    ),
    (
        "md",
        "Let's look at the closest Xmon first. `build_component` rebuilds a database row with its Qiskit Metal class:",
    ),
    ("py", 'best_xmon = xmon_results.iloc[0]\nlook_at(build_component(best_xmon), "closest Xmon");'),
    (
        "md",
        """And now the prediction, with exactly the call we would use for `AnsysSimulator(analyzer, best_device).simulate()` in Tutorial 2:""",
    ),
    (
        "py",
        """xmon_ml = MLSimulator(analyzer, best_xmon).simulate()
xmon_ml["fidelity"], round(xmon_ml["timing"]["wall_s"], 2)""",
    ),
    ("py", "compare(best_xmon, xmon_ml)"),
    (
        "md",
        """**Careful:** every row of the planar capacitance dataset was part of QCap's training data, so this agreement is a consistency check of the plumbing (rows, rendering, naming), not a measure of accuracy. For accuracy, look at the table further up.

The charging energy from the predicted matrix, with every claw included:""",
    ),
    (
        "py",
        """from squadds.ml.capacitance.bridge import EC_MHz_from_matrix

C_xmon = maxwell_df(xmon_ml)
EC_MHz_from_matrix(C_xmon.values, C_xmon.index, "cross"), 1e3 * best_xmon["EC"]""",
    ),
    (
        "md",
        r"""### A floating transmon: `TransmonPocket`

A `TransmonPocket` has two pads and no grounded island, so the qubit is the **differential** mode of the two pads. With the coupler pads held at ground (their resonators are much larger capacitors),

$$\frac{1}{C_{\rm eff}} = (C^{-1})_{11} + (C^{-1})_{22} - 2\,(C^{-1})_{12}, \qquad E_C = \frac{e^2}{2 C_{\rm eff}},$$

where $C$ is the two-pad block of the Maxwell matrix. SQuADDS uses this for `TransmonPocket` targets.""",
    ),
    (
        "py",
        """db.select_qubit("TransmonPocket")
pocket_df = db.create_system_df()
analyzer = Analyzer(db)
pocket_results = analyzer.find_closest({"qubit_frequency_GHz": 5.3, "anharmonicity_MHz": -250}, num_top=3)
pocket_results[["design_id", "qubit_frequency_GHz", "anharmonicity_MHz", "EC", "C_sigma_fF"]]""",
    ),
    ("py", 'best_pocket = pocket_results.iloc[0]\nlook_at(build_component(best_pocket), "closest TransmonPocket");'),
    (
        "py",
        """pocket_ml = MLSimulator(analyzer, best_pocket).simulate()
C_pocket = maxwell_df(pocket_ml)
C_pocket.round(3)""",
    ),
    (
        "py",
        """EC_MHz_from_matrix(C_pocket.values, C_pocket.index, ["pad_top", "pad_bot"]), 1e3 * best_pocket["EC"]""",
    ),
    (
        "md",
        """### A coupler with a capacitance target: `GeneralizedCapStar`

Couplers don't have a Hamiltonian of their own, so we query them with a capacitance target directly. Let's ask for a star coupler with about 2 fF between the star and its first pad.

One thing to know: some star rows (all of one collection and a few others) were labelled on a GDS that their stored options do not reproduce exactly. They carry `geometry_reproducible = False`, and we'll leave them out whenever we want to rebuild a design from its options.""",
    ),
    (
        "py",
        """db.unselect_all()
db.select_system("coupler")
db.select_coupler("GeneralizedCapStar")
star_df = db.create_system_df()
analyzer = Analyzer(db)
analyzer.df = analyzer.df[analyzer.df["geometry_reproducible"].fillna(True).astype(bool)]
len(star_df), len(analyzer.df)""",
    ),
    (
        "py",
        """star_results = analyzer.find_closest({"star_to_pad_0": 2.0}, num_top=3)
star_results[["design_id", "star_to_pad_0", "star_to_ground", "pad_0_to_ground"]]""",
    ),
    ("py", 'best_star = star_results.iloc[0]\nlook_at(build_component(best_star), "closest GeneralizedCapStar");'),
    (
        "md",
        """`Analyzer.predict_capacitance` predicts every row of the last `find_closest` result in one go (and works the same way on interpolated designs, as we saw above):""",
    ),
    (
        "py",
        """star_ml = analyzer.predict_capacitance()
star_results[["design_id", "star_to_pad_0"]].join(star_ml[["star_to_pad_0_ml", "fidelity", "ml_total_s"]])""",
    ),
    (
        "md",
        """These rows are training data of QCap too, so again: plumbing check, not accuracy.

### What about the Fast tier?

There is also `model="QCap-Fast"`, which replaces the field solves with a learned surrogate and is roughly ten times faster. **Use it only for quick screening of the families it was trained on.** On new designs of a trained family it was *worse than the physics alone*: 3.175 % (MAE 0.583 fF / RMSE 0.824 fF) against 1.765 % (MAE 0.275 fF / RMSE 0.376 fF), and on a held-out family 3.11 % (MAE 0.533 fF / RMSE 0.633 fF) against 1.55 % (MAE 0.166 fF / RMSE 0.205 fF). Use Accurate for anything that matters.""",
    ),
    (
        "py",
        """fast_star = MLSimulator(model="QCap-Fast").predict(best_star)
fast_star["fidelity"], fast_star["sim_results"]["star_to_pad_0"], star_ml.iloc[0]["star_to_pad_0_ml"]""",
    ),
    (
        "md",
        """Finally, let's see what every prediction in this tutorial cost (wall time per design with 2 threads, on the machine this notebook ran on):""",
    ),
    (
        "py",
        """timing = {
    "TransmonCross (closest, repeated: cache)": ml_result["timing"]["wall_s"],
    "TransmonCross (interpolated)": interpolated_ml.iloc[0]["ml_total_s"],
    "Xmon": xmon_ml["timing"]["wall_s"],
    "TransmonPocket": pocket_ml["timing"]["wall_s"],
    "GeneralizedCapStar (accurate)": star_ml.iloc[0]["ml_total_s"],
    "GeneralizedCapStar (fast, first call: loads the model)": fast_star["timing"]["wall_s"],
}
pd.Series(timing, name="wall time (s)").round(2)""",
    ),
    (
        "md",
        f"""Congrats for making it to the end of this tutorial! 🤗🎉 You have now learned how to go from a set of target Hamiltonian parameters to a rendered layout, its full capacitance matrix and its Hamiltonian in about a second per design, how to check interpolated designs without a Q3D run, and how to query the new qubit and coupler families of SQuADDS. Just remember the fidelity label: an `ml:accurate` number is a prediction, and the design you send to the fab still deserves one real simulation.

## Next Steps...

In the next [tutorial]({NEXT_URL}), we will learn how to train the capacitance models on your own designs and simulations.""",
    ),
    ("md", LICENSE),
]


def _cell(kind: str, src: str, index: int) -> dict:
    cid = hashlib.sha1(f"{index}:{src}".encode()).hexdigest()[:12]
    if kind == "md":
        return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}
    return {"cell_type": "code", "id": cid, "metadata": {}, "source": src, "execution_count": None, "outputs": []}


def build(cells, path: Path) -> Path:
    nb = {
        "cells": [_cell(k, s, i) for i, (k, s) in enumerate(cells)],
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
    print(build(CELLS, OUT))
