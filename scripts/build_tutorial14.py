#!/usr/bin/env python
"""Build ``tutorials/Tutorial-14_Training_on_Your_Own_Designs.ipynb`` (unexecuted).

Execute it afterwards with ``jupyter nbconvert --to notebook --execute --inplace`` from ``tutorials/``. The collective
retraining section reads the all-data training shards from ``$SQUADDS_CAPML_BASE_SHARDS``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tutorial11 import LICENSE, ROOT, build  # noqa: E402

OUT = ROOT / "tutorials" / "Tutorial-14_Training_on_Your_Own_Designs.ipynb"
NEXT_URL = (
    "https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-15_Contributing_Capacitance_Data_and_Models.html"
)

CELLS = [
    (
        "md",
        """# Tutorial 14: Training on Your Own Designs

In this tutorial, we will answer a question every designer runs into sooner or later: we have simulated a handful of our own designs in Ansys Q3D, so is the released capacitance model already good enough on them, or should we train on them? By the end of this tutorial, you will be able to:

- Turn GDS files and Q3D `cap_matrix.csv` exports into training data, with a label QC that refuses broken simulations

- Cross-validate the zero-shot model against a network trained from scratch on your designs (`fit_in_family`), and read the result as a median % error with the MAE and RMSE in fF

- Retrain the collective model with your designs added to all the public data (`retrain_collective`), and understand why it trains 3 seeds

- Write a SLURM array script to retrain on your own cluster (`squadds-ml train --slurm`)

**Note:** this tutorial needs the optional ML extra, `pip install "squadds[ml]"`. It builds on [Tutorial 11](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.html).""",
    ),
    ("py", "%load_ext autoreload\n%autoreload 2\n%matplotlib inline"),
    (
        "md",
        """## The story

Let's say we are designing Xmon qubits: a cross in a ground pocket, a readout claw on one arm and, on some of them, a bus claw on a second arm. We changed the cross and claw dimensions a bit for our process, simulated ten of these designs in Q3D (each one is a few minutes to an hour of solver time, depending on the mesh), and now we want to simulate a hundred more for a sweep. Before we spend that compute, we want to know whether the ML model is good enough on *our* Xmons.

There are two ways to use the ten simulations:

1. **Zero-shot:** keep the released model as it is, and use our ten designs only to measure how good it is.
2. **From scratch:** train a network on our ten designs alone (the paper's in-family recipe, about a minute per fit on a laptop).

`fit_in_family` cross-validates both and keeps the zero-shot model unless training is *clearly* better. If our family is far from anything the models have seen, a third option is to add our designs to all the public data and retrain the collective model, which we do at the end.

### Our "own" designs

For this tutorial, our ten designs are Xmons from the SQuADDS planar capacitance dataset (family `exp13`) that ship in `data/tutorial14/`. The released `QCap` model is trained on all the public data, these ten included, so its zero-shot score on them is an in-sample score and flatters it. That is fine for showing the workflow. With designs of your own that nobody has seen, the same call gives an honest comparison.""",
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
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd""",
    ),
    ("md", "We'll write everything into one folder, and start from a clean one."),
    (
        "py",
        """work = Path("tutorial14_files")
shutil.rmtree(work, ignore_errors=True)
work.mkdir()""",
    ),
    (
        "md",
        """Each design lives in its own folder with three files: the GDS, the options that generated it (JSON) and the Q3D export, `cap_matrix.csv`.""",
    ),
    (
        "py",
        """data_dir = Path("data/tutorial14/xmon")
folders = sorted(data_dir.iterdir())
[f.name for f in folders]""",
    ),
    ("py", "sorted(p.name for p in folders[0].iterdir())"),
    ("md", "Here are the knobs we turned (µm), straight from the generator options:"),
    (
        "py",
        """params = pd.DataFrame(
    [json.loads(next(f.glob("*.json")).read_text())["generator"]["params"] for f in folders],
    index=[f.name for f in folders],
)
params.round(2)""",
    ),
    (
        "md",
        """### Look at the metal first

As in Tutorial 11, let's look at what the model sees before anything else: the ground on layer 1/0 and every conductor on layer 1/10. These GDS files also carry a small port stub on layer 2/0 at the readout claw: that is how the dataset names the nets the way Q3D did (`N` = the readout claw, `S` = the cross, `P1` = the bus claw).""",
    ),
    ("py", "from squadds.ml.capacitance import prepare_layout"),
    (
        "md",
        """Importing SQuADDS also loads scqubits, which switches the inline figures of the notebook to vector formats. Let's switch them back to PNG so the plots stay light.""",
    ),
    ("py", 'from matplotlib_inline.backend_inline import set_matplotlib_formats\n\nset_matplotlib_formats("png")'),
    (
        "py",
        """def plot_layout(layout, title="", ax=None):
    \"\"\"Ground in grey (its etch in white), every conductor in its own color, labelled with its name.\"\"\"
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    for poly in getattr(layout.ground, "geoms", [layout.ground]):
        ax.fill(*poly.exterior.xy, color="0.82", lw=0)
        for hole in poly.interiors:
            ax.fill(*hole.xy, color="white", lw=0)
    for k, (name, body) in enumerate(zip(layout.names, layout.conductors)):
        for poly in getattr(body, "geoms", [body]):
            ax.fill(*poly.exterior.xy, color=plt.cm.tab10.colors[k % 10], lw=0)
        p = body.representative_point()
        ax.annotate(name, (p.x, p.y), ha="center", fontsize=9, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_title(title)
    return ax""",
    ),
    (
        "py",
        """fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for ax, f in zip(axes, [folders[0], folders[-1]]):
    plot_layout(prepare_layout(next(f.glob("*.gds"))), title=f.name, ax=ax)""",
    ),
    (
        "md",
        """Two of our Xmons, both with a second (bus) claw: four conductors including ground. Six of the ten have the readout claw only (three conductors).

## Reading Q3D labels and label QC

Q3D writes one row per adaptive pass. `read_q3d_csv` keeps the last complete pass, converts it to fF and puts the ground first.""",
    ),
    (
        "py",
        "from squadds.ml.capacitance.training_data import Design, LabelQCError, build_shards, label_qc, read_q3d_csv",
    ),
    (
        "py",
        """names, C, info = read_q3d_csv(folders[0] / "cap_matrix.csv")
info""",
    ),
    ("py", "pd.DataFrame(C, index=names, columns=names).round(3)"),
    (
        "md",
        """Before we train on anything, the labels get checked: symmetry, Maxwell signs, closure (no signal row may sum below zero) and a positive definite signal block. A label that fails is refused, not repaired.""",
    ),
    ("py", "label_qc(C, names)"),
    (
        "md",
        """Let's see QC do its job. Here is the same matrix with a sign typo in one mutual capacitance, the kind of thing that happens when a matrix is copied by hand:""",
    ),
    (
        "py",
        """broken = C.copy()
broken[1, 2] = broken[2, 1] = -broken[1, 2]
try:
    label_qc(broken, names)
except LabelQCError as err:
    print("refused:", err)""",
    ),
    (
        "md",
        """Great, a broken label never reaches the training set.

## Building the training shards

A `Design` pairs a GDS with its label. `build_shards` turns each one into what the networks read: the geometric embedding of the layout, the physics prior (the two thin-film solves of `optimized_physics_prior`) and the QC'd label, matched to the layout's conductors **by name** (nets are never merged). The result is a folder of shards, one file per conductor count.""",
    ),
    (
        "py",
        """designs = [
    Design(gds=str(next(f.glob("*.gds"))), labels=str(f / "cap_matrix.csv"), key=f"my_xmons/{f.name}", family="my_xmon")
    for f in folders
]
designs[0]""",
    ),
    (
        "py",
        """built = build_shards(designs, work / "shards", threads=2)
{k: built[k] for k in ("layouts", "refused", "seconds")}""",
    ),
    ("py", 'from squadds.ml.capacitance.training_data import shard_summary\n\nshard_summary(work / "shards")'),
    (
        "md",
        """**Note:** the priors are computed on *your* machine. On macOS they differ from the released priors (computed on x86 Linux) by up to about $2.4 \\times 10^{-3}$ relative on some layouts. That is harmless for your own models; it is the reason the maintainers compute every released prior on the same x86 machines.

## Zero-shot or from scratch? `fit_in_family`

`fit_in_family` runs a $k$-fold cross-validation over our ten designs. In each fold, two designs are held out and predicted twice:

- by the **zero-shot** model (`base`), which never changes, and
- by a network trained **from scratch** on the other eight designs.

Both candidates are scored on exactly the same held-out layouts, so we can compare them design by design. The from-scratch recipe wins only if it is *materially* better: the layout-bootstrap 95 % interval of the difference must exclude zero **and** the gain must exceed 0.25 percentage points of median error or 0.05 fF of MAE (the equivalence margin of the paper). Otherwise the zero-shot model is kept; it wins ties.

The recipe is the paper's, written out in `TrainConfig` (these are the defaults of the released models):""",
    ),
    ("py", "from squadds.ml.capacitance.train import TrainConfig, fit_in_family\n\nTrainConfig()"),
    (
        "md",
        """Each fit is the full recipe, 18,000 steps. We run the five folds in four processes with 2 CPU threads each.""",
    ),
    (
        "py",
        """cv = fit_in_family(work / "shards", work / "in_family", k_folds=5, base="QCap", threads=2, processes=4,
                   family="my_xmon")
print(cv)""",
    ),
    (
        "md",
        """`cv.report` holds the layout-mean scores of both candidates, per kind of matrix entry: all mutual capacitances, the signal-signal and signal-ground mutuals, and the self capacitances.""",
    ),
    (
        "py",
        """cv.report[["candidate", "kind", "layouts", "median_pct", "mae_fF", "rmse_fF"]].round(4)""",
    ),
    ("md", "The paired difference (from scratch minus zero-shot) with its 95 % interval, and the decision:"),
    (
        "py",
        """diff = cv.decision["paired_from_scratch_minus_zero_shot"]
pd.DataFrame({k: v for k, v in diff.items() if k != "layouts"}).T.round(4)""",
    ),
    ("py", 'cv.decision["chosen"], cv.decision["base"]'),
    (
        "py",
        """pd.Series({k: v for k, v in cv.timing.items() if k != "per_fit_s"})""",
    ),
    ("md", "Let's look at it design by design:"),
    (
        "py",
        """per_layout = pd.read_csv(work / "in_family" / "cv_layouts.csv").query("kind == 'mutual'")
wide = per_layout.pivot(index="key", columns="candidate", values="median_pct")
wide.index = [k.split("/")[-1] for k in wide.index]
ax = wide.plot.bar(figsize=(9, 3.5), rot=45)
ax.set_ylabel("median |error| of the mutuals (%)")
ax.set_title("each design predicted while held out of training");""",
    ),
    (
        "md",
        """On our ten Xmons, the zero-shot QCap has a median error of 0.27 % on the mutual capacitances (MAE 0.120 fF, RMSE 0.173 fF), and a network trained from scratch on the eight designs of each fold gets 0.47 % (MAE 0.260 fF, RMSE 0.375 fF). Training from scratch is *worse*, by 0.20 percentage points (95 % interval 0.10 to 0.31), so `fit_in_family` keeps the zero-shot model. QCap was trained on these ten designs, so its side of the comparison is an in-sample score. Eight designs are simply too few to learn what 45,771 layouts taught the released networks; the gap is largest on the self capacitance of the ground, 1.27 % (MAE 3.61 fF, RMSE 3.61 fF) from scratch against 0.26 % (MAE 0.738 fF, RMSE 0.738 fF) zero-shot. The five cross-validation fits took 77 s of wall time (four processes with 2 threads each on an Apple M5 Pro).

So for our Xmons the answer is: don't train, run the sweep with the model and keep a few Q3D spot checks. Ten designs is a small sample, though, and from-scratch training improves quickly with more designs, so it is worth repeating this check as your set grows.""",
    ),
    (
        "md",
        """## Retraining the collective model

Sometimes your family is far from anything in the public data, the zero-shot error is too large, and ten designs are too few to train on alone. Then the best option is usually to add your designs to **all** the public data and retrain the collective model with QCap Accurate recipe. `retrain_collective` does exactly that:

1. it merges your shards with the all-data training set (refusing a different feature schema or prior version, and duplicate keys),
2. it trains **3 seeds**: three networks from different random initialisations, whose mean is the prediction. One network carries some seed-to-seed noise; the ensemble of three averages most of it out, which is why every released tier is a 3-seed ensemble.

The released models are trained by the maintainers on x86 Linux CPUs, 2 threads per seed. Your retrain on a GPU, on a Mac, or with other library versions gives a slightly different set of networks: training is chaotic, so another CPU, thread count or library version changes the training trajectory, and the predictions move by a few tenths of a percent, about as much as between two seeds, and it is recorded as such in every checkpoint: fine for your own designs, but never mixed into the released data.

The all-data training set is the dataset's `model_input` table. Here we point to a local copy of the QCap shards through the environment variable `SQUADDS_CAPML_BASE_SHARDS`.""",
    ),
    (
        "py",
        """base_shards = Path(os.environ.get("SQUADDS_CAPML_BASE_SHARDS", "qcap_shards"))
base_summary = shard_summary(base_shards)
{k: base_summary[k] for k in ("layouts", "by_split", "n_range", "prior_version")}""",
    ),
    (
        "md",
        """Our stand-in Xmons are public designs, so QCap's training set already contains them (as `exp13/<design>`). To avoid counting them twice, let's drop them from the base first. With designs of your own you skip this step.""",
    ),
    ("py", "from squadds.ml.capacitance.training_data import subset_shards"),
    (
        "py",
        """ours = [f"exp13/{f.name}" for f in folders]
base_clean = subset_shards(base_shards, work / "base_without_ours",
                           keep=lambda d: ~np.isin(d["key"].astype(str), ours))
print(base_summary["layouts"] - shard_summary(base_clean)["layouts"], "of our designs removed from the base")""",
    ),
    (
        "md",
        """Now the retrain itself: three seeds of the full recipe on about 45,800 layouts, in three processes with 2 threads each. On a laptop this takes a while (go get a coffee ☕); on a cluster node it is a few minutes per seed.""",
    ),
    ("py", "from squadds.ml.capacitance.train import retrain_collective"),
    (
        "py",
        """collective, runs = retrain_collective(work / "shards", base_clean, work / "collective", seeds=(0, 1, 2),
                                     processes=3, threads=2)""",
    ),
    (
        "py",
        """pd.DataFrame(runs)[["seed", "steps", "train_seconds", "threads", "device", "machine"]]""",
    ),
    (
        "md",
        """That is about 4.5 minutes per seed here (three seeds in parallel, 2 threads each, Intel Xeon Gold 6130).

`collective` is a model like any other, so it goes straight into a `CapacitancePredictor`. Since we retrained on (almost) the same data with the same recipe as QCap, the two should agree closely on one of our Xmons. This is a reproducibility check, not an accuracy score: both models trained on this design.""",
    ),
    ("py", "from squadds.ml.capacitance import CapacitancePredictor, load_model"),
    (
        "py",
        """gds = designs[-1].gds
qcap = CapacitancePredictor(load_model("QCap", tier="accurate"), threads=2).predict(gds)
retrained = CapacitancePredictor(collective, threads=2).predict(gds)""",
    ),
    (
        "py",
        """def mutuals(C, names):
    \"\"\"The mutual capacitances (fF, positive) of a Maxwell matrix, one entry per pair of conductors.\"\"\"
    C = np.asarray(C)
    return pd.Series({f"{names[i]} to {names[j]}": -C[i, j] for i in range(len(names)) for j in range(i + 1, len(names))})""",
    ),
    (
        "py",
        """check = pd.DataFrame({"QCap (fF)": mutuals(qcap.matrix_fF, qcap.names),
                      "retrained (fF)": mutuals(retrained.matrix_fF, retrained.names)})
check["difference (%)"] = 100 * (check["retrained (fF)"] / check["QCap (fF)"] - 1)
check.round(4)""",
    ),
    (
        "md",
        """The conductors are named in file order here: `C1` is the readout claw, `C2` the cross and `C3` the bus claw. The retrained ensemble agrees with QCap to within about half a percent on the larger entries (1.3 % on the 0.05 fF coupling between the two claws). That is the size of the seed-to-seed spread: our three seeds were trained on a different CPU and with different library versions than the released ones, so their training trajectories differ. Both models trained on this design, so neither column is an accuracy score.""",
    ),
    (
        "md",
        """## Retraining on your own cluster

On a cluster, the natural unit of work is one seed per task. `squadds-ml train --slurm` writes a generic SLURM array script (one array task per seed) with no site-specific accounts, partitions or modules: pass your own with `--partition` and `--account`, or leave them out and let your cluster's defaults apply.""",
    ),
    (
        "py",
        "!squadds-ml train --data tutorial14_files/shards --out tutorial14_files/runs --seeds 0,1,2 --threads 2 --time 01:00:00 --slurm tutorial14_files/train_array.sh",
    ),
    ("md", "Here is the script it wrote (we shorten our working directory to `.` for readability):"),
    (
        "py",
        """script = (work / "train_array.sh").read_text()
print(script.replace(str(Path.cwd().resolve()), ".").replace(sys.executable, "python"))""",
    ),
    (
        "md",
        """Submit it with `sbatch train_array.sh`. When the three tasks are done, `trained_model` turns the run folders into a model for `CapacitancePredictor`:

```python
from squadds.ml.capacitance.train import trained_model

model = trained_model(["tutorial14_files/runs/s0", "tutorial14_files/runs/s1", "tutorial14_files/runs/s2"])
CapacitancePredictor(model).predict("my_next_xmon.gds")
```

The same works from Python with `squadds.ml.capacitance.slurm.write_slurm_array` (for example with `gpus=1` and `device="cuda"`; GPU fits are not bit-identical to the CPU ones either).""",
    ),
    (
        "md",
        f"""Congrats for making it to the end of this tutorial! 🤗🎉 You have now learned how to decide, with numbers, whether the planar capacitance model is good enough on your own designs: we turned GDS files and Q3D exports into QC'd training shards, cross-validated the zero-shot model against training from scratch, retrained the collective model with 3 seeds, and wrote a SLURM script to do it on a cluster.

## Next Steps...

In the next [tutorial]({NEXT_URL}), we will learn how to contribute your simulations and your models back to SQuADDS, so that the next designer starts from a better model.""",
    ),
    ("md", LICENSE),
]


if __name__ == "__main__":
    print(build(CELLS, OUT, Path(sys.argv[1]) if len(sys.argv) > 1 else None))
