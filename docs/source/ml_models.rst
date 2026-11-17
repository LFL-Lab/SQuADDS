ML Models
=========

SQuADDS ships two kinds of ML models:

- **Planar capacitance models** (``squadds.ml.capacitance``, the ``[ml]`` extra): GDS layout -> the full N x N Maxwell
  capacitance matrix of every conductor, on your machine, in about a second. See :ref:`planar-capacitance-models`.
- **Hosted models** on our `Hugging Face org <https://huggingface.co/SQuADDS>`_, served by the SQuADDS ML API on the
  `SQuADDS ML Inference API Space <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_: one HTTP API
  (``/v1``) for scripts and AI agents. See :ref:`hosted-inference`.

Reached a limit of the online demo, or want to predict many layouts? :doc:`ml_local` shows how to run the models on your own computer.

.. contents:: On this page
   :local:
   :depth: 2


.. _planar-capacitance-models:

Planar capacitance models
-------------------------

Resources
^^^^^^^^^

- **Paper:** `QCap: a physics-corrected graph neural network for the capacitance matrix of any planar superconducting device layout <../qcap/>`_, with a short video for each section. By topic:

  - how a layout becomes a graph: `Sec. II B <../qcap/#sec-emb-geometric>`_; the physics prior: `Sec. II C <../qcap/#sec-emb-prior>`_
  - the network and the released models: `Sec. III A <../qcap/#sec-ml-network>`_ and `Sec. III B <../qcap/#sec-ml-models>`_
  - accuracy: `Sec. IV A <../qcap/#sec-res-sota>`_; families the model never saw: `Sec. IV B <../qcap/#sec-res-zeroshot>`_
  - other substrates: `Sec. IV C 1 <../qcap/#sec-res-stack>`_; large chips: `Sec. IV C 2 <../qcap/#sec-res-windows>`_
  - design workflows: `Sec. IV D <../qcap/#sec-res-workflows>`_

- **Models:** `SQuADDS/QCap <https://huggingface.co/SQuADDS/QCap>`_
- **Dataset:** `SQuADDS/SQuADDS_PlanarCapacitance <https://huggingface.co/datasets/SQuADDS/SQuADDS_PlanarCapacitance>`_
- **Browser demo:** `SQuADDS/planar-capacitance <https://huggingface.co/spaces/SQuADDS/planar-capacitance>`_
- **Hosted API:** :ref:`hosted-inference`
- **On your own computer:** :doc:`ml_local`
- **Tutorials:** 11 to 16 (:ref:`planar-capacitance-tutorials`)

What they do
^^^^^^^^^^^^

Given a GDS layout, the models return the **full N x N Maxwell capacitance matrix** in fF: every conductor of the layout,
ground included, for any N. The conductor count is read from the layout and **nets are never merged**. Row and column 0
are the ground net ``G``; the other conductors follow in GDS file order (``C1`` .. ``Cn``). Diagonal entries are self
capacitances (> 0); off-diagonal entries are minus the mutual capacitances (< 0).

Each prediction starts from a physics prior, a thin-film method-of-moments solve of the layout, and a graph network
(the ``planar-capacitance-graph-v1`` embedding of :mod:`squadds.layouts`: 60 conductor, 42 pair and 4 global features)
corrects it. Every result carries physical checks (symmetry, signs, closure, positive definiteness), the timing with
machine and thread count, and the sha256 of the weights used.

Training stack: silicon (eps_r 11.45), 350 um substrate, zero-thickness metal, ground on GDS layer 1/0 and conductors on
1/10 (both configurable).

Other substrates
^^^^^^^^^^^^^^^^

``predict(layout, eps_r=..., thickness_um=...)`` (also ``squadds-ml predict --eps-r --thickness-um`` and the physics priors)
takes any substrate. The route is recorded in ``result.substrate`` (route, the stack the prior was solved at, whether the
stack was tested, the caveat).

- **QCap: prior solved at your stack (calibrated route).** Both thin-film solves run on a dielectric
  slab of your eps_r and thickness; the networks, trained on the training stack only, correct that prior unchanged, and the
  geometric embedding stays at the training stack. Measured on the 15 held-out designs of the stack study (5 interdigital,
  5 shielded concentric, 5 Xmon; networks trained on the TRAIN split, prior v2m, Q3D at each stack), mutual entries, layout-mean median error: 0.21-1.02 %
  (MAE 0.07-0.33 fF, RMSE 0.09-0.45 fF) on every tested stack; Si 10 um 0.69 % (0.115 / 0.164 fF); fused silica (eps_r
  3.8) 0.64 % (0.071 / 0.093 fF); sapphire (eps_r 10.34) 0.26 % (0.076 / 0.106 fF), GaAs (eps_r 12.9) 0.22 % (0.086 /
  0.120 fF), LaAlO3 (eps_r 24) 0.23 % (0.161 / 0.219 fF). The previous fixed-stack prediction was 76.6 %
  (13.7 / 18.8 fF) at 10 um Si and 162.9 % (23.4 / 30.5 fF) on fused silica.
- This is a **bounded-error route, not training-stack accuracy** (training stack 0.23 % (0.080 / 0.111 fF)). It is worse
  than plain permittivity rescaling of the training-stack prediction where that rescaling was already accurate: small
  capacitors (interdigital, paddle) at 50-200 um (interdigital at 50 um 1.39 % (0.098 / 0.114 fF) against 0.53 % (0.043 / 0.045 fF)), thick Si (1000 um 0.65 %
  (0.325 / 0.451 fF) against 0.50 % (0.187 / 0.245 fF)), and the self terms: 0.96-1.22 % (MAE 0.48-0.79 fF, RMSE 0.52-0.86
  fF) at 10-50 um Si and 0.89 % (0.923 / 1.071 fF) at 1000 um, against 0.24 % (0.18 / 0.22 fF) on the training stack. The
  networks never saw the stack vary; training data at more than one stack is the physical fix.
- **Tested range**: Si (eps_r 11.45) at 10-1000 um, or eps_r 3.8-24 at 350 um. Any other combination runs and raises a
  ``UserWarning`` (also in ``result.warnings``): untested, its error is unknown.
- **Cost**: one prior pair per stack, median 1.33 s, max 37 s, plus 0.1 s for the networks (HPC Discovery x86, AMD EPYC,
  2 threads), the same as a training-stack prediction. The prior cache is keyed by the stack.
- **QCap-Fast: training-stack prediction rescaled by (1 + eps_r) / (1 + 11.45); thickness not modelled.** Its learned prior only
  knows the training stack, so it cannot solve one at yours; every off-stack call warns ("QCap-Fast does not solve the prior at
  your stack ..."). Scored on 285 off-stack Q3D solves of 15 held-out designs: 6.36 % (MAE 1.74 fF, RMSE 2.39 fF) in all,
  13.1 % (3.43 / 4.67 fF) across Si thickness 10-1000 um and 1.44 % (0.51 / 0.73 fF) across eps_r 3.8-24 at 350 um, against
  1.29 % (0.48 / 0.71 fF) on the training stack; QCap gives 0.48 % (0.13 / 0.18 fF) on the same solves. **Use
  QCap off the stack** (median 3.6 s per layout against QCap-Fast's 0.16 s; Apple M5 Pro, 2 threads, 10 designs).

Models
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - model
     - what it is
   * - ``QCap`` (**default**)
     - Two thin-film solves (corrected prior with substrate far field, 1.5 and 3 panels per gap), Richardson-extrapolated
       (``optimized_physics_prior``), corrected by a 3-seed ensemble of residual graph networks with a learned closure.
       Networks trained on all 45,771 on-stack Q3D layouts.
   * - ``QCap-Fast``
     - An improved release of the paper's fast model, same architecture: no field solve, a learned surrogate predicts
       the prior and networks correct it. The surrogate is trained on more generated layouts (432,697 against 385,593)
       and the networks are retrained on every labelled design (all 45,771 on-stack layouts, the paper's test split
       included). For quick screening of trained families only (see the warning below).

.. warning::

   **QCap-Fast.** Use ``QCap-Fast`` only for quick screening of the component families it was trained on (interdigital, paddle,
   concentric, star, multipad star, Xmon, TransmonPocket, shielded concentric, two-net, random, procedurally generated
   and synthetic campaign layouts). On new designs of a trained family it was worse than the physics prior alone,
   3.175 % (MAE 0.583 fF / RMSE 0.824 fF) against 1.765 % (MAE 0.275 fF / RMSE 0.376 fF), and the paper's version of it
   was worse than the physics alone on a held-out family too, 3.11 % (MAE 0.533 fF / RMSE 0.633 fF) against 1.55 %
   (MAE 0.166 fF / RMSE 0.205 fF). Use ``QCap`` for anything that matters.

Accuracy and time
^^^^^^^^^^^^^^^^^

Metric: per layout, the median relative error over its mutual entries against Q3D (0.1 fF floor), averaged over layouts,
always with the MAE and RMSE in fF. The QCap and QCap-Fast networks are trained on all data, the paper's test layouts
included, so neither has a **held-out score of its own**. The table gives (a) each model's **own in-sample fit** on the 45,771
layouts it was trained on (a fit, not a generalisation score), (b) its **one never-seen check** (B5: 39 new layouts of one
generated family, 3-4 conductors), and (c) the held-out numbers of the **paper's version** of each model (networks trained
on the 35,354 TRAIN layouts only; for QCap-Fast also the paper's older surrogate). The (c) rows belong to the paper's
versions, not to the released models.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - set (layouts)
     - QCap: median % (MAE / RMSE)
     - QCap-Fast: median % (MAE / RMSE)
   * - this model, in-sample (45,771; trained on them)
     - 0.207 % (0.0679 fF / 0.077 fF)
     - 0.671 % (0.353 fF / 0.395 fF)
   * - this model, B5 never seen (39)
     - 1.05 % (0.132 fF / 0.188 fF)
     - 3.175 % (0.583 fF / 0.824 fF)
   * - B5, physics alone R(1.5, 3) (39)
     - 1.765 % (0.275 fF / 0.376 fF)
     - 1.765 % (0.275 fF / 0.376 fF)
   * - paper's version, validation (4,608)
     - 0.229 % (0.0674 fF / 0.0792 fF)
     - 0.846 % (0.491 fF / 0.549 fF)
   * - paper's version, test, in family (4,009)
     - 0.177 % (0.0203 fF / 0.025 fF)
     - 0.63 % (0.113 fF / 0.1414 fF)
   * - paper's version, test, zero-shot macro (1,878)
     - 0.587 % (1.092 fF / 1.152 fF)
     - 3.576 % (4.608 fF / 5.058 fF)
   * - paper's version, held-out family (4,071)
     - 0.408 % (0.049 fF / 0.059 fF)
     - 3.113 % (0.533 fF / 0.633 fF)

QCap-Fast's surrogate recipe trained without the validation layouts, with the paper's TRAIN-only networks, scores 0.722 %
(0.387 fF / 0.436 fF) on validation (held out), against 0.846 % (0.491 fF / 0.549 fF) for the paper's version.

Every result carries ``result.checks`` (symmetry, off-diagonal signs, smallest eigenvalue of the signal block). When a
check fails, the result's warnings say so and a ``UserWarning`` is raised: do not use that matrix (QCap-Fast is the
one at risk, far outside its training families).

Wall time per layout (median / p90):

- **QCap**: 0.845 / 5.96 s in family and 1.75 / 18.1 s zero-shot on an x86 cluster node with 2 CPUs. With this
  package (median, cold caches): 1.88 s for N <= 4 and 23.0 s for 10 <= N <= 20 on a GH200 node with 2 threads (0.98 s and
  10.5 s with 8 threads); 3.76 s and 46.2 s on an x86 Xeon Silver 4116 with 2 threads.
- **QCap-Fast**: 0.091 / 0.274 s in family and 0.30 / 2.45 s zero-shot (x86, 2 CPUs); with this package 0.093 s for N <= 4
  on a GH200 node and 0.17 s on the Xeon Silver 4116 (2 threads).

The first prediction of a session also downloads and verifies the weights and compiles the thin-film solver (about 40 s,
once per environment thanks to the numba cache); a repeated layout is answered from the cache in milliseconds. The
``threads`` argument sets the solver threads. By default SQuADDS uses all usable cores (at most one per 2.5 GB of RAM and
32; ``SQUADDS_CAPML_THREADS`` sets it too), and results are deterministic for a given thread count and machine.

Known limitation
^^^^^^^^^^^^^^^^

The scene builder repairs polygons with ``shapely.make_valid`` at its default (``method="linework"``), which drops a ground
sliver (0.011-0.53 % of the ground area) where a keyhole cut runs along a hole edge: 7 of the 60,093 dataset layouts, all
``largen`` star_multibody test layouts (N 8-12), and any user layout with that geometry. Stored priors, embeddings and the
released models use the default consistently; ``method="structure"`` would move QCap predictions on those 7 by a
median of 0.050 % per entry (MAE 0.089 fF, RMSE 0.459 fF; max 14.2 %, 7.61 fF) and is planned for a future release.

Install
^^^^^^^

.. code-block:: bash

   pip install "squadds[ml]"

The extra adds ``torch``, ``gdstk``, ``numba``, ``threadpoolctl``, ``shapely`` and ``pyyaml``. ``import squadds`` keeps
working without it; the ML entry points raise an error naming the missing packages and the install command. Prediction
and training run on Linux (x86 and aarch64) and macOS. On Linux aarch64 (e.g. GH200 nodes) ``gdstk`` has no wheel and must
be built from source, and Qiskit Metal's Qt dependency needs a recent glibc, so the component-rebuilding workflows are
easiest on x86 Linux or macOS.

Quick start
^^^^^^^^^^^

.. code-block:: python

   from squadds.ml.capacitance import CapacitancePredictor, load_model

   predictor = CapacitancePredictor(load_model("QCap"))   # or "QCap-Fast"; all cores by default, threads= to limit
   res = predictor.predict("chip.gds")          # also a gdstk Library / Cell, or a list of files
   res.names                                    # ['G', 'C1', 'C2', ...]
   res.matrix_fF                                # N x N Maxwell matrix, fF
   res.checks, res.timing_s, res.machine        # physical checks, time, machine
   res.to_sim_results()                         # the SQuADDS_DB ``sim_results`` shape (feeds squadds.calcs)

From the command line:

.. code-block:: bash

   squadds-ml predict chip.gds                          # QCap, CSV matrix
   squadds-ml predict a.gds b.gds --model QCap-Fast --format json
   squadds-ml predict chip.gds --ground 1/0 --conductor 1/10,2/10 --threads 4

Models are downloaded at a pinned revision and every file is checked against the sha256 in the package manifest; a local
bundle can be used with ``load_model(..., path=...)`` or the ``SQUADDS_CAPML_MODELS`` environment variable.

The physics priors alone (no learned parameter) are available too:

.. code-block:: python

   from squadds.ml.capacitance import physics_prior, optimized_physics_prior

   physics_prior("chip.gds").C_fF               # the physics prior: one solve at 0.375 panels per gap
   optimized_physics_prior("chip.gds").C_fF     # the extrapolated (optimized) prior C^R = R(1.5, 3) of QCap

Priors computed on macOS differ from the released ones (computed on x86 Linux) by up to about 2.4e-3 relative.

Ground and conductors on one layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many layouts draw the ground plane and the conductors on the same GDS layer. Pass the same layer as ground and conductor
(``ground_layers=[(2, 0)], conductor_layers=[(2, 0)]``); a file with a single metal layer and neither default layer is
handled the same way with the default arguments. The metal is split into connected pieces (touching shapes form one
piece) and the ground is picked automatically: the piece whose openings contain every other piece; failing that, the
largest near-rectangular piece on the layout boundary; failing that, the largest piece. The conductors are numbered in
file order and the detected ground comes last, and conductor piece ``k`` is named ``Ck`` in the figure, in
``result.names`` and in the matrix, so the automatic split names its conductors exactly as the two-layer file would,
and names do not move when you change roles. The prediction proceeds with that pick and warns. The result records it in ``result.ground_detection`` (pieces, reason and a
confidence; it is ``low`` when no piece encloses the rest or when the two largest pieces are within 2x in area).

Check the pick on the numbered figure, and if it is wrong, predict again with the ground's piece numbers (several pieces
form one ground net):

.. code-block:: python

   from squadds.ml.capacitance import inspect_layout

   info = inspect_layout("chip_one_layer.gds", ground_layers=[(2, 0)], conductor_layers=[(2, 0)])
   info["figure"]                               # ground grey, conductors coloured, every piece numbered
   info["ground_pieces"], info["reason"], info["confidence"]
   res = predictor.predict("chip_one_layer.gds", ground_layers=[(2, 0)], conductor_layers=[(2, 0)], ground_pieces=[4])

.. code-block:: bash

   squadds-ml predict chip_one_layer.gds --ground 2/0 --conductor 2/0 --ground-pieces 4

Two-layer files are unchanged. On the four flattened test layouts (Xmon, interdigital, the one-qubit lattice and a
16-conductor layout) the detected split is exactly the two-layer one and the predictions are bitwise equal.

Inside the existing SQuADDS workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``MLSimulator`` (``squadds.ml.capacitance.bridge``) has the call and result shape of the Q3D path of
  ``AnsysSimulator``: ``MLSimulator(analyzer, device_dict).simulate()`` returns ``design``,
  ``sim_options`` and ``sim_results`` with ``fidelity="ml:QCap"`` (``model="QCap-Fast"`` for QCap-Fast); ``MLSimulator().predict(...)`` takes a GDS path, a
  Qiskit Metal design or component, or a SQuADDS row.
- ``Analyzer.predict_capacitance(results_df)`` rebuilds the closest (or interpolated) designs with their Qiskit Metal
  component, exports them to GDS and predicts them.
- The Analyzer supports **TransmonPocket** and **Xmon** qubit targets through ``squadds/calcs/floating_transmon.py``:
  ``FloatingTransmonHamiltonian`` (two islands; coupler pads held at ground; the pad-to-pad mode with
  C_eff = C12 + C1Σ C2Σ / (C1Σ + C2Σ); g by capacitive division) and ``XmonHamiltonian`` (one grounded island with any
  number of claws; C_Σ is the cross's Maxwell diagonal, so every claw counts: a two-claw E_C differs from the
  ``TransmonCross`` convention).
- The new families with a Qiskit Metal component are SQuADDS_DB configs: ``SQuADDS_DB()`` enables two sources,
  ``"default"`` (``SQuADDS/SQuADDS_DB``) and ``"planar_capacitance"`` (this dataset). ``select_*``,
  ``create_system_df``, ``Analyzer.find_closest`` and the interpolators work on them unchanged. The dataset also adds
  rows to ``coupler-GeneralizedCapNInterdigital-cap_matrix``, after the existing ones. Existing exp6 / exp7
  interdigital rows rebuild with ``spine_ledge_rule="tangent"`` automatically.
- ``SQuADDS_DB(sources=("default",))`` gives ``SQuADDS/SQuADDS_DB`` alone, as before 1.0. If the planar capacitance
  dataset cannot be read (offline, no access, a missing local copy), ``SQuADDS_DB`` warns once, disables the source and
  returns the ``SQuADDS/SQuADDS_DB`` rows. ``SQUADDS_PLANAR_CAP_DATA=<dir>`` points the source at a local copy.

The dataset
^^^^^^^^^^^

``SQuADDS/SQuADDS_PlanarCapacitance`` holds the layouts (byte-identical GDS), the Q3D simulations and their labels. Read it
with ``PlanarCapacitanceDB`` (``squadds.ml.capacitance.dataset``; pandas and pyarrow only, **no** ``[ml]`` extra needed):

.. code-block:: python

   from squadds.ml.capacitance.dataset import PlanarCapacitanceDB

   db = PlanarCapacitanceDB()                     # pinned commit; revision="main" for the latest; or local_dir=... / env SQUADDS_PLANAR_CAP_DATA
   ids = db.designs(family="concentric", split="val").design_id[:5]
   db.labels(ids)                                 # {design_id: N x N Maxwell matrix (DataFrame, fF, ground first)}
   db.optimized_physics_prior(ids)                # the prior the models consume

Tables (Parquet, long-format N x N in fF, ground first; every processed row records its version and code sha256):

- **raw**: ``designs`` (options, family, SQuADDS component class / config, split), ``layouts`` (GDS path, sha256,
  ``geom_hash``), ``simulations`` (solver, setup, stack, convergence, QC flags), ``labels`` (last adaptive pass),
  ``label_passes`` (every pass);
- **processed**: ``geometric_embedding`` (nodes / edges / globals), ``physics_prior`` (one solve at 0.375 panels
  per gap), ``optimized_physics_prior`` (C^R = R(1.5, 3) with its two solves), ``model_input`` (what
  QCap consumes).

Test-set labels are held out (they are the paper's once-scored evaluation data); their layouts, embeddings and priors are
public.

The paper's synthetic dataset with physics-only targets (no Q3D; surrogate set 385,593 synthetic layouts, generic set
286,134) is a separate Parquet download, ``SQuADDS/SQuADDS_PlanarCapacitance-Synthetic``; the SQuADDS API does not read it.

Retraining on your own compute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Retraining runs anywhere (laptop, GPU box, cluster) with no site-specific tooling (``squadds.ml.capacitance.train``):

- ``build_shards(designs, out_dir)`` (``squadds.ml.capacitance.training_data``) turns your GDS files and Q3D
  ``cap_matrix.csv`` exports into training shards; every label passes QC and is matched to the layout's nets by name.
- ``fit_in_family(designs, out_dir)``: the paper's from-scratch recipe on your k designs, cross-validated against the
  zero-shot QCap (zero-shot is always a candidate and wins ties).
- ``retrain_collective(user_shards, base_shards, out_dir)``: your shards added to the all-data training set, 3 seeds.
- ``fit(...)`` / ``squadds-ml train --data <shards> --out <dir> --seeds 0,1,2``: the paper recipe (18,000 steps).
- ``squadds-ml train ... --slurm job.sh --partition <p> --account <a>`` writes a generic SLURM array script (one seed
  per task).
- ``benchmark.score(model, shards, split="val")`` scores a model on the validation split (median % with MAE and RMSE in
  fF, time, machine, threads). Test scoring is maintainer-only.

Runs record machine, threads, device and package version. Training on a GPU or on macOS is not bit-identical to the
released models (trained on x86 Linux CPUs).

Contributing data and models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data: copy the template ``squadds/database/templates/planar_capacitance_contribution`` (``manifest.yaml``,
``designs.csv``, ``gds/``, ``sims/``), then

.. code-block:: python

   from squadds.ml.capacitance.contribute import contribute_capacitance, validate_capacitance_contribution

   print(validate_capacitance_contribution("my_batch/").to_markdown())
   contribute_capacitance("my_batch/")          # opens a pull request on SQuADDS/SQuADDS_PlanarCapacitance

The validator (``squadds.ml.capacitance.qc``) rejects schema errors, unreadable GDS, a conductor count that differs from
the matrix size (never merge nets), bad net names, asymmetry, wrong Maxwell signs, a signal block that is not positive
definite, a licence other than CC-BY-4.0 / CC0-1.0, and duplicates of held-out test designs; it warns on closure above
5 %, convergence, unusual stacks and other duplicates, strips absolute paths, and can compare every entry with the
``optimized_physics_prior`` (``check_prior=True``). Maintainers merge, compute the priors and add the rows as
``source="community"``, ``split="none"``.

Models: ``contribute_model("you/your-model", revision)`` loads the checkpoints, records their sha256, checks the metadata
(``squadds_model.yaml``: training-data sha256, split used, validation score with median %, MAE and RMSE in fF, time,
machine and threads) and opens a pull request adding the model to ``model_zoo.yaml``. Pass ``val_data=`` to compute the
validation score with ``benchmark.score``. Test-set scores are run by the maintainers, once per declared model, only for
models trained on the TRAIN split.

.. _planar-capacitance-tutorials:

Tutorials
^^^^^^^^^

- `Tutorial 11: Predicting capacitance matrices from GDS <https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.html>`_
- `Tutorial 12: Planar capacitance families in SQuADDS <https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-12_Planar_Capacitance_Families_in_SQuADDS.html>`_
- `Tutorial 13: From layout to Hamiltonian with ML <https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-13_From_Layout_to_Hamiltonian_with_ML.html>`_
- `Tutorial 14: Training on your own designs <https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-14_Training_on_Your_Own_Designs.html>`_
- `Tutorial 15: Contributing capacitance data and models <https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-15_Contributing_Capacitance_Data_and_Models.html>`_
- `Tutorial 16: Inverse design by ML screening <https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-16_Inverse_Design_by_ML_Screening.html>`_

API reference: :doc:`squadds.ml.capacitance` and :doc:`squadds.layouts`. MCP tools: :doc:`mcp_server`.


.. _hosted-inference:

Hosted inference models
-----------------------

The SQuADDS ML API is one HTTP service for the models we host on our `Hugging Face org <https://huggingface.co/SQuADDS>`_. It runs on the `SQuADDS ML Inference API Space <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_. Use it when you want a prediction from a script or an AI agent without installing the model's dependencies.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Resource
     - URL
   * - API host
     - ``https://squadds-squadds-ml-inference-api.hf.space``
   * - Interactive docs
     - `/docs <https://squadds-squadds-ml-inference-api.hf.space/docs>`_ and `/openapi.json <https://squadds-squadds-ml-inference-api.hf.space/openapi.json>`_
   * - Space repo
     - `huggingface.co/spaces/SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_

Which models are available
^^^^^^^^^^^^^^^^^^^^^^^^^^

``GET /v1/models`` is the source of truth for what the API serves. Read it at run time instead of hard-coding model ids. Today it lists ``transmon_cross_hamiltonian_inverse`` (see :ref:`hosted-live-model`) and the planar capacitance models ``qcap`` and ``qcap-fast`` (a GDS file in, the Maxwell matrix out). More models will be added.

Routes
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Route
     - What it returns
   * - ``GET /v1/health``
     - Service status, the state of each model (``ready``, ``loading`` or ``gated``) and the build (SQuADDS commit, registry hash, image time).
   * - ``GET /v1/models``
     - The models you can call: ``id``, ``family``, ``display_name``, ``description``, ``tags``, ``input_kind`` (``json`` or ``file``), ``revision`` and ``status``.
   * - ``GET /v1/models/{model_id}``
     - The listing entry plus the model's contract: input and output names, units, limits and an example request.
   * - ``GET /v1/models/{model_id}/schema``
     - The JSON schema of the request. With ``?format=tool`` it returns a tool definition for an AI agent (``name``, ``description``, ``input_schema``).
   * - ``POST /v1/models/{model_id}/predict``
     - Runs the model and returns the response envelope below.
   * - ``GET /docs``, ``GET /openapi.json``
     - Interactive documentation and the OpenAPI spec, generated from the same schemas.

Predict
^^^^^^^

Send JSON, ``{"inputs": {...}, "options": {...}}``. ``inputs`` may also be a list of objects, a batch of up to the model's ``max_batch_rows``. A model whose ``input_kind`` is ``file`` also accepts ``multipart/form-data`` with a ``file`` field (the bytes) and an ``inputs`` field (the JSON of the other fields), or the file as ``inputs.file_base64`` in a JSON body.

.. code-block:: bash

   curl -X POST \
     https://squadds-squadds-ml-inference-api.hf.space/v1/models/transmon_cross_hamiltonian_inverse/predict \
     -H 'Content-Type: application/json' \
     -d '{"inputs": {"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0}}'

Every model answers with the same envelope. ``outputs`` and ``metadata`` depend on the model family.

.. code-block:: json

   {
     "model_id": "transmon_cross_hamiltonian_inverse",
     "family": "keras_tabular",
     "revision": "26f2e04eb09ccd816e1c39ca7aa90a88bd737171",
     "outputs": {
       "predictions": [
         {
           "design_options.connection_pads.readout.claw_length": 0.00011072495544794947,
           "design_options.connection_pads.readout.ground_spacing": 4.571595582092414e-06,
           "design_options.cross_length": 0.0002005973074119538
         }
       ]
     },
     "metadata": {
       "input_order": ["qubit_frequency_GHz", "anharmonicity_MHz"],
       "output_order": [
         "design_options.connection_pads.readout.claw_length",
         "design_options.connection_pads.readout.ground_spacing",
         "design_options.cross_length"
       ],
       "input_units": {"qubit_frequency_GHz": "GHz", "anharmonicity_MHz": "MHz"},
       "output_units": {
         "design_options.connection_pads.readout.claw_length": "m",
         "design_options.connection_pads.readout.ground_spacing": "m",
         "design_options.cross_length": "m"
       },
       "num_predictions": 1
     },
     "timing_s": {"total": 0.012, "queue": 0.0},
     "cached": false
   }

Errors on ``/v1`` have one shape, ``{"error": {"code": ..., "message": ..., "model_id": ...}}``. The codes are ``invalid_input`` (400), ``auth_required`` and ``forbidden`` (401, 403), ``unknown_model`` (404, also for a model that is not listed), ``too_large`` (413), ``validation_error`` (422), ``busy`` and ``rate_limited`` (429), ``inference_error`` (500) and ``loading`` (503).

Public models need no token. A gated model needs ``Authorization: Bearer <token>`` with the Hugging Face token of a member of the SQuADDS organization.

Python client
^^^^^^^^^^^^^

``squadds.ml.client`` wraps ``/v1`` and needs no extra dependencies:

.. code-block:: python

   from squadds.ml import client

   for m in client.list_models():                  # GET /v1/models
       print(m["id"], m["input_kind"], m["status"])
   client.get_model("transmon_cross_hamiltonian_inverse")   # inputs, outputs, units, limits, example

   res = client.predict(
       "transmon_cross_hamiltonian_inverse",
       inputs={"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0},
   )
   res["outputs"]["predictions"][0]                 # geometry in meters
   # a file model: client.predict(model_id, file="chip.gds", inputs={...})

The client calls ``SQUADDS_ML_API_URL`` when it is set, else the API host above. Pass ``token=`` (or set ``HF_TOKEN``) for a gated model. An HTTP error raises ``RuntimeError`` with the status code and the error object.

For AI agents
^^^^^^^^^^^^^

1. Call ``GET /v1/models`` and pick a model whose ``status`` is ``"ready"``.
2. Call ``GET /v1/models/{model_id}/schema?format=tool`` and register the result as a tool. ``name``, ``description`` and ``input_schema`` are the fields Claude and OpenAI tool definitions use.
3. Send the tool call's arguments to ``POST /v1/models/{model_id}/predict``.
4. Feed the outputs into SQuADDS, Qiskit Metal or your validation flow, in the units given in ``metadata``.

The SQuADDS MCP server offers the same through two tools, ``list_hosted_ml_models`` and ``predict_with_hosted_ml_model`` (see :doc:`mcp_server`).

Legacy routes
^^^^^^^^^^^^^

The first routes of the Space still work, with their original request and response shapes:

- ``GET /health``: liveness.
- ``GET /models``: the deployed models with their ``status`` and input and output contract.
- ``POST /predict``: inference for a given ``model_id`` with its exact input keys.

.. code-block:: bash

   curl -X POST \
     https://squadds-squadds-ml-inference-api.hf.space/predict \
     -H 'Content-Type: application/json' \
     -d '{"model_id":"transmon_cross_hamiltonian_inverse","inputs":{"qubit_frequency_GHz":4.85,"anharmonicity_MHz":-205.0}}'

.. code-block:: json

   {
     "model_id": "transmon_cross_hamiltonian_inverse",
     "display_name": "TransmonCross Hamiltonian to Geometry",
     "predictions": [
       {
         "design_options.connection_pads.readout.claw_length": 0.00011072495544794947,
         "design_options.connection_pads.readout.ground_spacing": 4.571595582092414e-06,
         "design_options.cross_length": 0.0002005973074119538
       }
     ],
     "metadata": {
       "input_order": ["qubit_frequency_GHz", "anharmonicity_MHz"],
       "output_order": [
         "design_options.connection_pads.readout.claw_length",
         "design_options.connection_pads.readout.ground_spacing",
         "design_options.cross_length"
       ],
       "input_units": {
         "qubit_frequency_GHz": "GHz",
         "anharmonicity_MHz": "MHz"
       },
       "output_units": {
         "design_options.connection_pads.readout.claw_length": "m",
         "design_options.connection_pads.readout.ground_spacing": "m",
         "design_options.cross_length": "m"
       },
       "num_predictions": 1
     }
   }

New code should use ``/v1``.

.. _hosted-live-model:

TransmonCross Hamiltonian-to-geometry model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``transmon_cross_hamiltonian_inverse`` predicts TransmonCross (qubit-claw) geometry from target Hamiltonian inputs.

- **On the API:** `model page <https://squadds-squadds-ml-inference-api.hf.space/v1/models/transmon_cross_hamiltonian_inverse>`_
  (inputs, outputs, an example request) and the `interactive API docs <https://squadds-squadds-ml-inference-api.hf.space/docs>`_
- **Paper:** `Component-Level Inverse Design of Transmon Qubits Using Neural Networks <https://arxiv.org/abs/2607.20795>`_ (Seidel et al., 2026)
- **Model:** `SQuADDS/transmon-cross-hamiltonian-inverse <https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse>`_

- **Expected inputs** (SQuADDS-native units): ``qubit_frequency_GHz``, ``anharmonicity_MHz``.
- **Returned outputs** (SI units, meters): ``design_options.connection_pads.readout.claw_length``, ``design_options.connection_pads.readout.ground_spacing``, ``design_options.cross_length``.

Full per-model contract (``X_names``, output order, scalers, ``inference_manifest.json``) lives on the `model repo <https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse>`_.

Acknowledgments
^^^^^^^^^^^^^^^

The first live model (``transmon_cross_hamiltonian_inverse``) was developed in collaboration with Taylor Patti, Nicola Pancotti, Enectali Figueroa-Feliciano, Sara Sussman, Abhishek Chakraborty, Olivia Seidel, Firas Abouzahr, Eli Levenson-Falk, and Sadman Ahmed Shanto — with **Olivia Seidel and Firas Abouzahr** as the primary trainers.


Current limitations
^^^^^^^^^^^^^^^^^^^

- Live today: ``transmon_cross_hamiltonian_inverse``, ``qcap`` and ``qcap-fast``. More models will be added. ``GET /v1/models`` lists what is live.
- Outputs are returned in meters. Convert them to SQuADDS / Qiskit Metal micrometer strings (``"...um"``) before writing them back into ``design_options``.
- The Space runs on shared Hugging Face hardware and answers ``429`` when its queue is full. Do not rely on it for production-scale batch inference.


Contributing a model
^^^^^^^^^^^^^^^^^^^^

Each hosted model is one entry in the API's registry: its id, the adapter that runs it, the weights pinned to a full commit of a Hugging Face model repository with the sha256 of every file, and its limits. To add one:

1. Push the weights to a model repository with a small ``inference_manifest.json`` (input and output names, units and files) and note the commit hash.
2. A model of an existing family (``keras_tabular`` for tabular Keras or scikit-learn surrogates, ``planar_capacitance`` for capacitance releases) needs only a registry entry and one example request with its expected response.
3. A new kind of model needs one adapter class. It declares its input kind (JSON or file) and its request schema, and implements ``describe()`` and ``predict()``. The schema drives ``/docs``, input validation and the agent tool definition, so routes and the Space's Dockerfile stay as they are.
4. Open an issue or pull request on `SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_ with the model repository, its input and output units, and its accuracy on held-out data (MAE and RMSE with any percentage).


Further reading
^^^^^^^^^^^^^^^

- API reference: `/docs <https://squadds-squadds-ml-inference-api.hf.space/docs>`_
- Space README: `huggingface.co/spaces/SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_
- Model repo README + ``inference_manifest.json``: `huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse <https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse>`_
- SQuADDS dataset: `huggingface.co/datasets/SQuADDS/SQuADDS_DB <https://huggingface.co/datasets/SQuADDS/SQuADDS_DB>`_
