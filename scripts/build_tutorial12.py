#!/usr/bin/env python
"""Build ``tutorials/Tutorial-12_Planar_Capacitance_Families_in_SQuADDS.ipynb`` (unexecuted).

Execute it afterwards with ``jupyter nbconvert --to notebook --execute --inplace`` from ``tutorials/``. It reads the
planar capacitance dataset (``SQUADDS_PLANAR_CAP_DATA`` for a local copy) and the public ``SQuADDS/SQuADDS_DB``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tutorial11 import LICENSE, ROOT, build  # noqa: E402

OUT = ROOT / "tutorials" / "Tutorial-12_Planar_Capacitance_Families_in_SQuADDS.ipynb"
DOCS = "https://lfl-lab.github.io/SQuADDS/source"

CELLS = [
    (
        "md",
        """# Tutorial 12: Planar Capacitance Families in SQuADDS

In this tutorial, we will meet the new capacitor and qubit families that just arrived in SQuADDS: about 60,000 layouts (paddle, concentric, star and multipad star couplers, shielded concentric capacitors, Xmon and `TransmonPocket` qubits, and more), each one simulated in Ansys Q3D with its **full $N \\times N$ Maxwell capacitance matrix**. By the end of this tutorial, you will be able to:

- Query the planar capacitance dataset with `PlanarCapacitanceDB`: designs by family, by number of conductors and by split

- Rebuild any design with its SQuADDS (Qiskit Metal) component and look at its metal and ports before trusting a number

- Read a design's named Maxwell matrix, its adaptive-pass history, its simulation QC flags and its GDS file

- Use the processed tables (the geometric embedding, the two physics priors and the model input) and compare the priors with Q3D

- Find the designs that look the most like a given one with `nearest`

- Use the same data in the classic SQuADDS workflow: `SQuADDS_DB(sources=("default", "planar_capacitance"))`, `select_coupler`, `select_qubit` and `Analyzer.find_closest`

- Know where the data comes from: the splits (and which test labels are public), the licence and the contributors

**Note:** the dataset is `SQuADDS/SQuADDS_PlanarCapacitance` on Hugging Face. A local copy works too: point the environment variable `SQUADDS_PLANAR_CAP_DATA` at it. Nothing in this tutorial needs the ML extra.""".replace(
            "{DOCS}", DOCS
        ),
    ),
    ("py", "%load_ext autoreload\n%autoreload 2\n%matplotlib inline"),
    (
        "md",
        """## Why a new dataset?

Until now, the capacitance data in SQuADDS came one component at a time: a `TransmonCross` and its claw, an interdigital capacitor between two lines. Real chips keep getting more connected, though. A coupler talks to three qubits at once, a qubit has a readout claw *and* a bus claw, a guard ring sits between two pads. In every one of these, each conductor talks to every other conductor, and the object a device physicist wants is the full **Maxwell capacitance matrix** $C$:

$$Q_i = \\sum_j C_{ij} V_j .$$

For $N$ conductors (the ground plane counts as one), $C$ is a symmetric $N \\times N$ matrix with positive diagonal entries (the self capacitances) and negative off-diagonal entries ($C_{ij} = -C^{\\rm mutual}_{ij}$). In a closed system every row sums to zero once the ground is included ($\\sum_j C_{ij} = 0$), and the signal block is positive definite.

The planar capacitance dataset stores exactly that, for every layout, with a few conventions we will see over and over:

- capacitances in **fF**, Maxwell signs, ground **`G` first**, then the signal conductors (`N`, `S`, `P1`, ...),
- the number of conductors $N$ is read from the layout, and **nets are never merged**,
- the reference label is the last complete adaptive pass of the Q3D solve, on silicon ($\\varepsilon_r = 11.45$, 350 µm thick) with zero-thickness metal.

Let's go meet the families!""",
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
        """import json

import gdstk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import squadds""",
    ),
    (
        "md",
        "Importing SQuADDS switches the inline figures of a notebook to PDF / SVG (a setting it inherits from scqubits), so let's switch back to PNG.",
    ),
    (
        "py",
        """from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("png")""",
    ),
    ("md", "SQuADDS also logs every web request it makes to Hugging Face; let's keep the notebook quiet."),
    (
        "py",
        """import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)""",
    ),
    (
        "md",
        """## Connecting to the dataset

`PlanarCapacitanceDB` is the read-only door to the dataset. It reads one Parquet file at a time and only the columns and rows we ask for, so it never loads more than a query needs (and it needs nothing beyond a plain `pip install squadds`).""",
    ),
    (
        "py",
        """from squadds.ml.capacitance.dataset import PlanarCapacitanceDB

pdb = PlanarCapacitanceDB()""",
    ),
    ("md", "These are the tables we can query:"),
    ("py", "list(pdb.tables())"),
    (
        "md",
        """The first five are the **raw** tables: `designs` (one row per design, with the options that generated it), `layouts` (one row per GDS file), `simulations` (one row per Q3D solve), `labels` (the final Maxwell matrices) and `label_passes` (every adaptive pass). The rest are **processed** tables computed from the layouts: the geometric embedding, two physics priors and the packed input of the planar capacitance models. We'll go through all of them.

## A first look: families, sizes and splits

Let's pull a few columns of the `designs` table for every design:""",
    ),
    (
        "py",
        """designs = pdb.designs(columns=["design_id", "collection", "paper_family", "component_class", "n_conductors", "split"])
len(designs)""",
    ),
    ("py", "designs.head()"),
    (
        "md",
        """Every design has a `design_id` of the form `ref/<collection>/<design>`. The collections `exp6` to `exp15` are parametric sweeps of one component each, so they come with a **SQuADDS component class** that rebuilds them. The other collections (random shape placements, mixed-shape layouts, two-net layouts, lattices, ...) are off-design sets that exist to teach the models physics; they have no Qiskit Metal component.""",
    ),
    (
        "py",
        """sweeps = designs[designs.collection.str.fullmatch(r"exp\\d+") & designs.component_class.notna()]
sweeps.groupby("collection", sort=False).agg(
    family=("paper_family", "first"), component=("component_class", "first"), designs=("design_id", "size")
).sort_index(key=lambda c: c.str[3:].astype(int))""",
    ),
    ("md", "How many conductors do they have? Remember that $N$ counts the ground:"),
    ("py", "pd.crosstab(sweeps.paper_family, sweeps.n_conductors)"),
    (
        "md",
        """And every design sits in one **split**. `train` and `val` are what the models learned from and were tuned on; `test` holds the layouts that were set aside to score them (more on that at the end). We'll use `train` and `val` designs throughout.""",
    ),
    ("py", "pd.crosstab(designs.paper_family, designs.split, margins=True)"),
    (
        "md",
        """`designs(...)` takes filters for the family (matched against `family`, `paper_family` and `model_family`), the split, the number of conductors, the collection and the SQuADDS config. For example, every validation Xmon with two claws ($N = 4$: ground, cross and two claws):""",
    ),
    (
        "py",
        'pdb.designs(collection="exp13", split="val", n_conductors=4)[["design_id", "paper_family", "n_conductors", "split"]]',
    ),
    (
        "md",
        """## Meeting the families

Numbers without a picture are dangerous, so before we read any matrix, we'll **rebuild each design with its SQuADDS component** and look at its metal and its ports. Three small helpers do this.

`build` takes a `design_id`, reads its stored options and instantiates the component in a fresh Qiskit Metal design:""",
    ),
    (
        "py",
        """from qiskit_metal import designs as metal_designs

from squadds.components import get_component_class
from squadds.database.planar_capacitance_view import QGEOMETRY_NETS, design_options_for, net_name


def build(design_id):
    \"\"\"Rebuild a dataset design with its SQuADDS (Qiskit Metal) component class.\"\"\"
    row = pdb.designs(design_ids=[design_id]).iloc[0]
    options = design_options_for(row.collection, row.component_class, pdb.params([design_id])[design_id])
    design = metal_designs.DesignPlanar({}, overwrite_enabled=True)
    component = get_component_class(row.component_class)(design, "component", options=options)
    design.rebuild()
    return component""",
    ),
    (
        "md",
        """`plot_metal` draws the ground in grey (its etch in white), every conductor in its own color with its net label (`N`, `S`, `P1`, ... and the SQuADDS name in brackets), and the **pins** of the component as red arrows. The pins are the ports: they mark where a resonator, a feed line or a bus attaches in the full chip. Qiskit Metal works in mm, so we scale to µm.""",
    ),
    (
        "py",
        """from shapely import affinity
from shapely.ops import unary_union


def metal_of(component):
    \"\"\"Conductor bodies [(net label, geometry)] and ground etch [geometry] of a component, in µm.\"\"\"
    nets, metal, etch = QGEOMETRY_NETS.get(type(component).__name__, {}), [], []
    for table in ("poly", "path"):
        df = component.design.qgeometry.tables[table]
        for r in df[~df["helper"].astype(bool)].itertuples():
            g = r.geometry if table == "poly" else r.geometry.buffer(r.width / 2, cap_style="flat")
            g = affinity.scale(g, 1e3, 1e3, origin=(0, 0))
            (etch if r.subtract else metal).append((nets.get(r.name), g))
    union = unary_union([g for _, g in metal])
    bodies = []
    for body in getattr(union, "geoms", [union]):
        labels = sorted({lab for lab, g in metal if lab and body.intersects(g)})
        bodies.append((labels[0] if len(labels) == 1 else "?", body))
    return bodies, [g for _, g in etch]""",
    ),
    (
        "py",
        """from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry.polygon import orient


def fill(ax, geometry, **kwargs):
    \"\"\"Fill a shapely (multi)polygon, holes included.\"\"\"
    for p in getattr(geometry, "geoms", [geometry]):
        rings = [np.asarray(r.coords) for r in (orient(p).exterior, *orient(p).interiors)]
        codes = [[MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 2) + [MplPath.CLOSEPOLY] for r in rings]
        ax.add_patch(PathPatch(MplPath(np.concatenate(rings), np.concatenate(codes)), lw=0, **kwargs))


def plot_metal(component, net_order, title="", ax=None):
    \"\"\"Ground in grey, etch in white, conductors colored and named, pins (ports) as red arrows.\"\"\"
    ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
    bodies, etch = metal_of(component)
    x0, y0, x1, y1 = unary_union([b for _, b in bodies] + etch).bounds
    pad, cls = 0.12 * max(x1 - x0, y1 - y0), type(component).__name__
    ax.fill([x0 - pad, x1 + pad, x1 + pad, x0 - pad], [y0 - pad, y0 - pad, y1 + pad, y1 + pad], color="0.82", lw=0)
    for g in etch:
        fill(ax, g, color="white")
    for label, body in sorted(bodies, key=lambda b: net_order.index(b[0]) if b[0] in net_order else 99):
        color = plt.cm.tab10(net_order.index(label) % 10 if label in net_order else 9)
        fill(ax, body, color=color, label=f"{label} ({net_name(cls, label)})")
    for name, pin in component.pins.items():
        (x, y), (dx, dy) = 1e3 * pin["middle"], 0.6 * pad * pin["normal"]
        ax.annotate(name, xy=(x, y), xytext=(x + dx, y + dy), color="crimson", fontsize=8,
                    arrowprops=dict(arrowstyle="<-", color="crimson"))
    ax.set_aspect("equal")
    ax.set(xlabel="x (µm)", ylabel="y (µm)", title=title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return ax""",
    ),
    (
        "md",
        """`named_matrix` reads the Q3D label of a design (a pandas DataFrame in fF, ground first) and names its rows and columns the way the SQuADDS rows do (`ground`, `north`, `cross`, `pad_2`, ...). Finally, `pick` chooses the first validation design of a collection, with the largest $N$ unless we ask for one.""",
    ),
    (
        "py",
        """def named_matrix(design_id):
    \"\"\"Q3D Maxwell matrix of a design (fF, ground first), with SQuADDS net names.\"\"\"
    C = pdb.labels([design_id])[design_id]
    cls = pdb.designs(design_ids=[design_id]).component_class.iloc[0]
    names = {lab: net_name(cls, lab) for lab in C.index}
    return C.rename(index=names, columns=names).round(3)


def pick(collection, n_conductors=None):
    \"\"\"First validation design of a collection (largest N unless n_conductors is given).\"\"\"
    d = pdb.designs(collection=collection, split="val", n_conductors=n_conductors)
    d = d.sort_values(["n_conductors", "design_id"], ascending=[False, True])
    return d.design_id.iloc[0]


def show(design_id):
    row = pdb.designs(design_ids=[design_id]).iloc[0]
    title = f"{row.paper_family}: {design_id} (N = {row.n_conductors})"
    return plot_metal(build(design_id), list(row.net_order), title=title)""",
    ),
    (
        "md",
        """### Paddle capacitors

We start with an old friend in a new shape. `GeneralizedCapNInterdigital` is the interdigital capacitor that already lives in SQuADDS; the paddle sweep gives each side its own finger width (`north_finger_width`, `south_finger_width`), so with few, wide fingers the comb turns into two facing paddles. Two signal conductors and the ground: $N = 3$.""",
    ),
    ("py", 'paddle = pick("exp9")\nshow(paddle);'),
    (
        "md",
        """The north body (with its port `north_end`) and the south body (port `south_end`) each end in a CPW stub, so this is a series coupling capacitor between two lines. Here is its matrix:""",
    ),
    ("py", "named_matrix(paddle)"),
    (
        "md",
        """The coupling we would design for is the mutual capacitance $C^{\\rm mutual}_{NS} = -C_{NS}$; the diagonal entries are the total capacitance of each body to everything else. Let's add up each row:""",
    ),
    ("py", "named_matrix(paddle).sum(axis=1).round(6)"),
    (
        "md",
        """The signal rows nearly close: what is left over (here about 0.02 fF) is the capacitance of each body to infinity, since Q3D solves the chip inside a finite box rather than in a perfectly closed ground. The ground row keeps a large remainder, because the ground plane is by far the biggest conductor and has the most capacitance to the far field. This is why we design with the mutual (off-diagonal) entries, and why the ground self term depends on how big the simulated chip is.""",
    ),
    (
        "md",
        """### Concentric capacitors

`GeneralizedCapConcentric` is a round island inside a ring, both inside a round pocket in the ground plane. The ring (`N`, north) is fed by a CPW stub from the north. In some designs the island (`S`, south) is fed by a stub from the south through a gap in the ring; in others, like this one, it floats. The ring screens the island from the ground, so the island-to-ring coupling is large for its footprint.""",
    ),
    ("py", 'concentric = pick("exp10")\nshow(concentric);'),
    ("py", "named_matrix(concentric)"),
    (
        "md",
        """### Star couplers

`GeneralizedCapStar` is a star-shaped island (`S`) in a circular ground pocket, with a pad (`N`, called `pad_0` in SQuADDS) tucked into one of its valleys. The star is the coupler; the pad is where one qubit or resonator attaches.""",
    ),
    ("py", 'star = pick("exp11")\nshow(star);'),
    ("py", "named_matrix(star)"),
    (
        "md",
        """### Multipad star couplers

The same star with a pad in **several** valleys: one coupler that talks to several qubits at once. Now the off-diagonal pad-to-pad entries matter too: they are the direct (unwanted) couplings between the circuits on different pads, the kind of number a pairwise table cannot give us.

**One thing to know:** the multipad star GDS files went through an export step after drawing that the stored options do not replay exactly, so a rebuild matches the component but not every vertex of the solved GDS. Their SQuADDS rows carry `geometry_reproducible = False`; the label always belongs to the GDS file in the dataset.""",
    ),
    ("py", 'multipad = pick("exp12")\nshow(multipad);'),
    ("py", "named_matrix(multipad)"),
    (
        "md",
        """Let's line up the couplings of the star to each of its pads, and the largest pad-to-pad one:""",
    ),
    (
        "py",
        """C = named_matrix(multipad)
pads = [n for n in C.index if n.startswith("pad_")]
pd.Series({**{f"star-{p}": -C.loc["star", p] for p in pads},
           "largest pad-pad": max(-C.loc[a, b] for a in pads for b in pads if a < b)}, name="mutual (fF)").round(3)""",
    ),
    (
        "md",
        """### Xmon qubits

`Xmon` is the cross-shaped transmon in a ground pocket with one or two claws: a readout claw (`N`, `claw`) and, optionally, a bus claw (`P1`, `bus_claw`). The cross (`S`) is the qubit island; its junction goes from the cross to the ground and is not part of the capacitance problem. Its geometry and stack differ from the existing SQuADDS `TransmonCross` rows (continuous sweeps, 350 µm silicon), so it is its own component and its own config. Here is one with both claws:""",
    ),
    ("py", 'xmon = pick("exp13", n_conductors=4)\nshow(xmon);'),
    ("py", "named_matrix(xmon)"),
    (
        "md",
        """For a grounded transmon, the total capacitance of the island is the Maxwell diagonal of the cross, $C_\\Sigma = C_{\\rm cross, cross}$, which counts both claws, and the charging energy is $E_C = e^2 / 2 C_\\Sigma$:""",
    ),
    (
        "py",
        """from scipy.constants import e, h

C_sigma = named_matrix(xmon).loc["cross", "cross"] * 1e-15
print(f"C_sigma = {C_sigma * 1e15:.2f} fF, E_C = {e**2 / (2 * C_sigma) / h / 1e6:.1f} MHz")""",
    ),
    (
        "md",
        """### TransmonPocket qubits

Qiskit Metal's `TransmonPocket` is a **floating** transmon: two pads (`S` = `pad_top`, `P1` = `pad_bot`) in a pocket, with one or two coupler pads (`N` = `readout_pad`, `P2` = `bus_pad`) reaching in. Neither pad is grounded, so the qubit is the differential mode of the two pads, and we need the whole matrix to get its $E_C$ (Tutorial 13 does exactly that).""",
    ),
    ("py", 'pocket = pick("exp14")\nshow(pocket);'),
    ("py", "named_matrix(pocket)"),
    (
        "md",
        """### Shielded concentric capacitors

`ShieldedCapConcentric` is a concentric capacitor (`S` = `inner`, `N` = `outer`) with extra **guard** conductors (`P1`, `P2`, ... = `guard_1`, `guard_2`, ...) around it, each its own net. Guards are how we shield a sensitive node, and their effect is exactly the kind of three-body physics a full matrix captures.""",
    ),
    ("py", 'shielded = pick("exp15")\nshow(shielded);'),
    (
        "md",
        """In this one, the small inner island sits inside two guard rings, and the outer ring (the one with the port) wraps around all of them. Let's see what the guards do:""",
    ),
    ("py", "named_matrix(shielded)"),
    (
        "md",
        """The island talks mostly to the guard right next to it ($C^{\\rm mutual}_{\\rm inner, guard\\_1} \\approx 14$ fF), while its direct coupling to the outer ring is only about 1.2 fF: the guards soak up its field. Whether a guard helps or hurts then depends on what we connect it to, and only the full matrix can tell us.""",
    ),
    (
        "md",
        """We have met all the families. Let's now take **one** design (the two-claw Xmon) and look at everything the dataset knows about it.

## One design, every raw table

### The simulation and its QC flags

`simulations` has one row per solve: the solver and its setup, the stack, the convergence, the solve time, and the label QC flags recomputed from the export (symmetry, signs, row closure, positive definiteness and convergence). The **reference** solve is the one whose last pass is the label.""",
    ),
    (
        "py",
        """sim = pdb.simulations([xmon], role="reference")
qc_columns = ["solver", "eps_r", "substrate_um", "passes_completed", "converged", "delta_pct_final", "cores", "solve_s",
              "qc_symmetric", "qc_signs", "qc_closure", "qc_pd", "qc_convergence", "qc_ok"]
sim[qc_columns].T""",
    ),
    (
        "md",
        """Across the whole dataset, the QC flags let us filter in one line. Here is the fraction of reference solves of each family that pass each check, and how many of them reached their convergence target:""",
    ),
    (
        "py",
        """flags = ["qc_symmetric", "qc_signs", "qc_closure", "qc_pd", "qc_ok"]
sims = pdb.simulations(role="reference", columns=["design_id", "qc_convergence", *flags])
sims = sims.merge(sweeps[["design_id", "paper_family"]], on="design_id")
rates = sims[flags].astype(float).groupby(sims.paper_family).mean().round(4)
rates.join(pd.crosstab(sims.paper_family, sims.qc_convergence.fillna("not recorded")))""",
    ),
    (
        "md",
        """Every matrix passes the symmetry, sign, closure and positive-definiteness checks. What differs is how convergence was recorded. A solve counts as converged when the change between its last two adaptive passes (`delta_pct_final`) is below the solver's own target (`delta_target_pct`). Let's look at those numbers, family by family:""",
    ),
    (
        "py",
        """cols = ["design_id", "passes_completed", "delta_pct_final", "delta_target_pct"]
conv = pdb.simulations(role="reference", columns=cols).merge(sweeps[["design_id", "paper_family"]], on="design_id")
conv.groupby("paper_family")[cols[1:]].median().round(3)""",
    ),
    (
        "md",
        """For the Xmon, `TransmonPocket` and shielded concentric solves, the final change sits just under the 0.05 % target. For the `exp6` to `exp12` sweeps (interdigital, paddle, concentric and both stars), the exports recorded no final change or target, so their convergence is `unrecorded`: neither confirmed nor refuted. They still pass `qc_ok` (which fails only for a truncated solve or one recorded above its target), the label is the last complete pass, and wherever the pass history was kept, `label_passes` (next) lets us judge the convergence ourselves. If a study needs confirmed convergence only, `qc_convergence == "converged"` is the one-line filter.""",
    ),
    (
        "md",
        """### How the label converged

`label_passes` keeps every adaptive pass of the solve, in the same long format as the labels. Let's watch the mutual capacitances of our Xmon settle, relative to their final value:""",
    ),
    (
        "py",
        """passes = pdb.label_passes([xmon])
mutual = passes[passes.a_idx < passes.b_idx]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for (a, b), g in mutual.groupby(["net_a", "net_b"]):
    g = g.sort_values("pass")
    ax.plot(g["pass"], 100 * (g.C_fF / g.C_fF.iloc[-1] - 1), "o-", ms=3, label=f"{a}-{b}")
ax.set(xlabel="adaptive pass", ylabel="change vs final pass (%)", ylim=(-5, 5))
ax.axhline(0, color="k", lw=0.5)
ax.legend(ncol=3, fontsize=8);""",
    ),
    (
        "md",
        """### The GDS that was solved

`gds_path` gives the local path of the GDS file of a design (downloading it from Hugging Face when needed). The dataset layer map is simple: the **ground plane on layer 1/0** and **every signal conductor on layer 1/10**, one connected body per conductor.""",
    ),
    ("py", 'gds = pdb.gds_path(xmon)\ngds.name, f"{gds.stat().st_size / 1e3:.1f} kB"'),
    (
        "py",
        """from shapely.geometry import Polygon


def plot_gds(path, ax):
    \"\"\"Ground (1/0) in grey and each conductor body on 1/10 in its own color.\"\"\"
    cell = gdstk.read_gds(str(path), unit=1e-6).top_level()[0]
    signal = []
    for poly in cell.polygons:
        if (poly.layer, poly.datatype) == (1, 0):
            ax.fill(*poly.points.T, color="0.82", lw=0)
        elif (poly.layer, poly.datatype) == (1, 10):
            signal.append(Polygon(poly.points).buffer(0))
    union = unary_union(signal)
    for k, body in enumerate(getattr(union, "geoms", [union])):
        fill(ax, body, color=plt.cm.Set2(k % 8))
    ax.set_aspect("equal")
    ax.set(xlabel="x (µm)", title=f"{path.name}: {k + 1} conductors + ground")
    return ax""",
    ),
    (
        "py",
        """fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
plot_metal(build(xmon), list(pdb.designs(design_ids=[xmon]).net_order.iloc[0]), "rebuilt with Xmon", ax=axes[0])
plot_gds(gds, axes[1]);""",
    ),
    (
        "md",
        """On the left is the component SQuADDS rebuilds from the stored options; on the right is the file Q3D solved, with its ground plane cut to the chip. Same metal, same three conductors plus ground.

## The processed tables

The raw tables describe what was simulated. The processed tables are what the planar capacitance models read, and they are useful on their own.

### The geometric embedding

Each layout is turned into vectors by the embedding `planar-capacitance-graph-v1`:

- one **conductor vector** per conductor (60 features, ground = node 0): area, perimeter, width, elongation, compactness, solidity, holes, Fourier descriptors of the outline and a granulometry of its widths (the stored table adds a 61st column, a flag for a missing width),
- one **pair vector** per pair of conductors (42 features): minimum distance, facing length, a histogram of separations and its moments, the ground-pair flag, and how crowded the neighbourhood is,
- one **global vector** (4 features): log chip area, log pocket area, log substrate thickness and the effective permittivity.""",
    ),
    (
        "py",
        """emb = pdb.embedding([xmon])[xmon]
{part: emb[part].shape for part in ("nodes", "edges")}, len(emb["globals"])""",
    ),
    (
        "py",
        """emb["nodes"][["node_idx", "net", "log_area", "log_perimeter", "log_width", "compactness", "role_ground"]].round(3)""",
    ),
    ("py", 'emb["edges"][["net_a", "net_b", "is_ground_pair", "log_dmin", "log_cps"]].round(3)'),
    ("py", '{k: v for k, v in emb["globals"].items() if k.startswith(("log_", "eps", "n_"))}'),
    (
        "md",
        """### Two physics priors

The dataset also stores two thin-film method-of-moments (MoM) solutions of every layout, computed from the GDS alone, with no learned parameter:

- `physics_prior`: one solve with $p = 0.375$ panels per gap (cheap and rough),
- `optimized_physics_prior`: two solves at $p = 1.5$ and $p = 3$, extrapolated entry by entry, $C^{R} = \\exp\\left(2 \\log C_{3} - \\log C_{1.5}\\right)$, the $R(1.5, 3)$ prior the planar capacitance models start from.

Both come back as Maxwell matrices, named like the labels. For our Xmon:""",
    ),
    (
        "py",
        """C = pdb.labels([xmon])[xmon]
iu = np.triu_indices(len(C), 1)
matrices = {"coarse prior": pdb.physics_prior([xmon])[xmon],
            "optimized prior": pdb.optimized_physics_prior([xmon])[xmon], "Q3D label": C}
pd.DataFrame({k: -M.loc[C.index, C.index].to_numpy()[iu] for k, M in matrices.items()},
             index=[f"{C.index[i]}-{C.index[j]}" for i, j in zip(*iu)]).round(3)""",
    ),
    (
        "md",
        """(These are the **mutual** capacitances, $-C_{ij}$, for each pair.) How good are the priors in general? Let's score them against the Q3D labels on a handful of validation designs from every family: five per collection. For every mutual entry we compute the error in fF; per family we report the **median relative error** together with the mean absolute error (MAE) and the root-mean-square error (RMSE) in fF, so that a small % on a tiny capacitance is never mistaken for a small error.""",
    ),
    (
        "py",
        """def mutual_errors(design_ids, prior):
    \"\"\"One row per mutual entry: Q3D label vs a stored prior (fF).\"\"\"
    labels, priors, rows = pdb.labels(design_ids), prior(design_ids), []
    for d in design_ids:
        if d not in labels or d not in priors:
            continue
        C = labels[d]
        P = priors[d].loc[C.index, C.index]
        iu = np.triu_indices(len(C), 1)
        rows += [dict(design_id=d, q3d=q, prior=p) for q, p in zip(-C.to_numpy()[iu], -P.to_numpy()[iu])]
    df = pd.DataFrame(rows)
    df["err"] = df.prior - df.q3d
    return df


def summary(df):
    return pd.Series({"median %": 100 * (df.err / df.q3d).abs().median(), "MAE (fF)": df.err.abs().mean(),
                      "RMSE (fF)": np.sqrt((df.err**2).mean()), "entries": len(df), "designs": df.design_id.nunique()})""",
    ),
    (
        "py",
        """sample = sweeps[sweeps.split == "val"].sort_values("design_id").groupby("collection").head(5)
family_of = sample.set_index("design_id").paper_family
scores = {}
for name, prior in [("coarse", pdb.physics_prior), ("optimized", pdb.optimized_physics_prior)]:
    err = mutual_errors(sample.design_id.tolist(), prior)
    scores[name] = err.groupby(err.design_id.map(family_of).rename("family")).apply(summary)
    scores[name].loc["all families"] = summary(err)
pd.concat(scores, axis=1).round(3)""",
    ),
    (
        "md",
        """Over these 50 designs (218 mutual entries), the coarse prior is off by a median 3.5 % (MAE 0.99 fF, RMSE 2.52 fF) and the optimized prior by 0.80 % (MAE 0.19 fF, RMSE 0.34 fF). The coarse prior struggles most on the large ground couplings of the Xmon, where its MAE reaches several fF; with the optimized prior, every family has a median error of 2.2 % or less. The remaining gap is what the networks of the planar capacitance models learn to close (Tutorial 11). This is a small sample (five designs per collection), so it shows the trend, not a benchmark.

What does a prior cost? Both were computed once on x86 cluster nodes and are stored, so reading them is free. Recomputing them for a five-pad star took 0.10 s (coarse) and 1.79 s (optimized) on 2 threads of an Intel Xeon E5-2640 v4 in [Tutorial 11]({DOCS}/tutorials/Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.html), against minutes to an hour for a Q3D solve (the reference solve of our Xmon took 2,467 s on 8 cores).

### The model input

`model_input` packs, per layout, exactly what the released Accurate tier consumes: the conductor vectors `X`, the pair vectors `E`, the global vector `G`, the two optimized-prior solves of every pair `pri` (at $p = 1.5$ and $p = 3$), and `sig`, which marks the signal conductors (the ground is `False`). Everything is flattened into lists, one row per layout:""".replace(
            "{DOCS}", DOCS
        ),
    ),
    (
        "py",
        """mi = pdb.model_input([xmon]).iloc[0]
n = int(mi.n_conductors)
pairs = n * (n - 1) // 2
pd.Series({"N": n, "net_order": list(mi.net_order), "pairs": pairs, "X per conductor": len(mi.X) / n,
           "E per pair": len(mi.E) / pairs, "G": len(mi.G), "pri per pair": len(mi.pri) / pairs, "sig": list(mi.sig)})""",
    ),
    (
        "md",
        """### Designs that look alike: `nearest`

Every layout can be pooled into one vector (the global features, $\\log N$, and the mean and maximum of each conductor feature over the signal conductors), and `nearest` returns the closest designs in that space, by default within the same family. It is a measure of **geometric** similarity, handy to find neighbours of a new idea in the data, not a capacitance distance.""",
    ),
    ("py", "neighbours = pdb.nearest(concentric, k=4)\nneighbours"),
    ("md", "Let's look at our concentric capacitor next to its two nearest neighbours:"),
    (
        "py",
        """fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, d in zip(axes, [concentric, *neighbours.design_id[:2]]):
    plot_metal(build(d), list(pdb.designs(design_ids=[d]).net_order.iloc[0]), d, ax=ax)""",
    ),
    (
        "md",
        """## The same data in the classic SQuADDS workflow

Everything above used the dataset's own tables. The families are also available through the API we know from [Tutorial 1]({DOCS}/tutorials/Tutorial-1_Getting_Started_with_SQuADDS.html): the dataset is a second **source** of `SQuADDS_DB`, converted on load to the usual row format (`design_options`, `sim_options`, `sim_results` with pairwise capacitances `<a>_to_<b>` in fF plus the signed `maxwell_matrix`, `notes`, `contributor`). The source is on by default, and here we name both sources:""".replace(
            "{DOCS}", DOCS
        ),
    ),
    (
        "py",
        """from squadds import Analyzer, SQuADDS_DB

db = SQuADDS_DB(sources=("default", "planar_capacitance"))
db.enabled_sources()""",
    ),
    (
        "md",
        "The new configurations appear next to the existing ones, with the usual `component-ComponentClass-data_type` names:",
    ),
    ("py", "db.get_configs()"),
    (
        "md",
        """### A concentric capacitor with a capacitance target

Couplers don't have a Hamiltonian of their own, so we query them with a capacitance target directly. We select the system and the coupler exactly as for any other SQuADDS component:""",
    ),
    (
        "py",
        """db.unselect_all()
db.select_system("coupler")
db.select_coupler("GeneralizedCapConcentric")
concentric_df = db.create_system_df()
concentric_df[["north_to_south", "north_to_ground", "south_to_ground"]].describe().round(2)""",
    ),
    (
        "md",
        """Let's say our design calls for 25 fF between the ring and the island. `Analyzer.find_closest` finds the closest rows:""",
    ),
    (
        "py",
        """analyzer = Analyzer(db)
results = analyzer.find_closest({"north_to_south": 25.0}, num_top=3)
results[["design_id", "north_to_south", "north_to_ground", "south_to_ground"]]""",
    ),
    (
        "md",
        """Each row still knows its `design_id`, so everything we did above works on it too. Here is the closest one:""",
    ),
    ("py", "best = results.design_id.iloc[0]\nshow(best);"),
    (
        "md",
        """### An Xmon with a target frequency and anharmonicity

For qubits, `find_closest` takes Hamiltonian targets as in Tutorial 1. For the `Xmon` rows, SQuADDS takes $C_\\Sigma$ as the full Maxwell diagonal of the cross, so a second claw is counted too.""",
    ),
    (
        "py",
        """db.unselect_all()
db.select_system("qubit")
db.select_qubit("Xmon")
xmon_df = db.create_system_df()
analyzer = Analyzer(db)
xmon_results = analyzer.find_closest({"qubit_frequency_GHz": 5.0, "anharmonicity_MHz": -210}, num_top=3)
xmon_results[["design_id", "qubit_frequency_GHz", "anharmonicity_MHz", "EC", "C_sigma_fF", "cross_to_claw"]]""",
    ),
    ("py", "show(xmon_results.design_id.iloc[0]);"),
    (
        "md",
        """The `TransmonPocket` rows work the same way with `select_qubit("TransmonPocket")` (with the floating-transmon Hamiltonian), and the other couplers with `select_coupler("GeneralizedCapStar")`, `select_coupler("ShieldedCapConcentric")` and the existing `select_coupler("GeneralizedCapNInterdigital")`, which the paddles join. [Tutorial 13]({DOCS}/tutorials/Tutorial-13_From_Layout_to_Hamiltonian_with_ML.html) takes these rows all the way to a Hamiltonian.

### No row is stored twice

The interdigital sweeps (`exp6`, `exp7`, `exp8`) are, for the most part, the very designs already in `SQuADDS/SQuADDS_DB`. The planar capacitance dataset does not upload them again: those designs carry a `squadds_db_ref` pointing to the existing row (repository, revision, config and row index), and the SQuADDS view leaves them out, so they appear once, from `SQuADDS_DB`. Every design is identified by `geom_hash`, a hash of its polygons rounded to 1 nm, which is how duplicates are found.""".replace(
            "{DOCS}", DOCS
        ),
    ),
    (
        "py",
        """refs = pdb.designs(columns=["design_id", "collection", "squadds_db_ref"])
refs.groupby("collection").squadds_db_ref.apply(lambda s: s.notna().sum()).loc[lambda s: s > 0]""",
    ),
    ("py", "json.loads(refs.squadds_db_ref.dropna().iloc[0])"),
    (
        "md",
        """## Splits, test labels, licence and contributors

### Which labels are public?

The splits are the ones of the paper that introduces the planar capacitance models. `train` and `val` labels are all public. The `test` layouts were set aside to score the models once, so **their geometry, embeddings and priors are public, but their labels are not** (they live in the private held-out repository `SQuADDS/SQuADDS_PlanarCapacitance-heldout`). A test design simply has no label:""",
    ),
    (
        "py",
        """test = pdb.designs(split="test", columns=["design_id", "collection", "paper_family", "notes"])
test_xmon = test[test.collection == "exp13"].design_id.iloc[0]
test_xmon, pdb.labels([test_xmon])""",
    ),
    (
        "md",
        """There is one exception: the 1,904 interdigital test designs (`exp6` to `exp8`) had their matrices public in `SQuADDS/SQuADDS_DB` long before, so their labels are in this dataset too, with a note that says so. Scores on them are therefore not blind.""",
    ),
    ("py", 'test.groupby(test.notes.fillna("").str.contains("labels public")).size().rename("test designs")'),
    (
        "md",
        """We never used a test label in this tutorial, and you should not either if you want to compare a model with the paper's numbers: train on `train`, tune on `val`, and leave `test` alone.

### Licence and contributors

The dataset is released under **CC-BY-4.0**: use it for anything, and credit the people who made it. Every design records its contributor and licence:""",
    ),
    (
        "py",
        """who = pdb.designs(columns=["collection", "contributor_id", "licence_id"])
who.licence_id.value_counts()""",
    ),
    ("md", "And here is who contributed each of the parametric sweeps:"),
    (
        "py",
        """sweep_who = who[who.collection.str.fullmatch(r"exp\\d+")]
pd.crosstab(sweep_who.collection, sweep_who.contributor_id).sort_index(key=lambda c: c.str[3:].astype(int))""",
    ),
    (
        "md",
        """- **Saikat Das** designed and simulated (Q3D) the `exp6` to `exp12` sweeps: the interdigital, paddle, concentric, star and multipad star capacitors.
- **Sadman Ahmed Shanto** made everything else: the Xmon, `TransmonPocket` and shielded concentric sweeps, the off-design sets including the multi-qubit lattice layouts, their Q3D solves, the priors, the embeddings and the curation.

Congrats for making it to the end of this tutorial! 🤗🎉 You have now met the planar capacitance families of SQuADDS, read their full Maxwell matrices, convergence, QC flags and GDS files, used the embedding and the physics priors, and queried the same data through `SQuADDS_DB` and the `Analyzer`.

## Next Steps...

In the next [tutorial](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-13_From_Layout_to_Hamiltonian_with_ML.html), we will take layouts like these from GDS all the way to a Hamiltonian with the planar capacitance models, and plug the predictions into the usual SQuADDS workflow.""",
    ),
    ("md", LICENSE),
]


if __name__ == "__main__":
    print(build(CELLS, OUT, Path(sys.argv[1]) if len(sys.argv) > 1 else None))
