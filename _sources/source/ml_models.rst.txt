ML Models
=========

- **Planar capacitance models** (``squadds.ml.capacitance``): GDS in, Maxwell capacitance matrix out, on your
  machine. See :ref:`planar-capacitance-models`.
- **Hosted models**: one HTTP API on our
  `Hugging Face Space <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_, for scripts and AI agents.
  See :ref:`hosted-inference`.

For many layouts, run locally: :doc:`ml_local`.

.. contents:: On this page
   :local:
   :depth: 2


.. _planar-capacitance-models:

Planar capacitance models
-------------------------

Resources
^^^^^^^^^

- **Paper:** `QCap (arXiv:2611.22057) <https://arxiv.org/abs/2611.22057>`_ and its `docsite <../qcap/>`_ (with videos)
- **Models:** `SQuADDS/QCap <https://huggingface.co/SQuADDS/QCap>`_
- **Dataset:** `SQuADDS/SQuADDS_PlanarCapacitance <https://huggingface.co/datasets/SQuADDS/SQuADDS_PlanarCapacitance>`_
- **Browser demo:** `SQuADDS/planar-capacitance <https://huggingface.co/spaces/SQuADDS/planar-capacitance>`_
- **Tutorials:** 11 to 16 (:ref:`planar-capacitance-tutorials`)

What they do
^^^^^^^^^^^^

GDS layout in, N x N Maxwell capacitance matrix (fF) out. Row 0 is ground (``G``), then ``C1`` .. ``Cn`` in GDS
order. Nets are never merged.

A fast physics solve (the *physics prior*) gives a first guess; a graph network corrects it. Each result also carries
physical checks, timing, machine and the weights' sha256.

Trained on silicon (eps_r 11.45, 350 um), zero-thickness metal, ground on layer 1/0, conductors on 1/10. Layers are
configurable.

Models
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - model
     - use it for
   * - ``QCap`` (**default**)
     - Design work.
   * - ``QCap-Fast``
     - Screening known families, about ten times faster (learned prior, no physics solve).

Both are trained on all 45,771 Q3D layouts.

Other substrates
^^^^^^^^^^^^^^^^

``predict(layout, eps_r=3.8, thickness_um=350)``, or ``--eps-r`` / ``--thickness-um`` on the command line.

- ``QCap`` solves the prior at your stack, at no extra cost: 0.21-1.02 % (MAE 0.07-0.33 fF, RMSE 0.09-0.45 fF) on
  tested stacks, vs. 0.23 % (MAE 0.080 fF, RMSE 0.111 fF) on the training stack.
- ``QCap-Fast`` only rescales by (1 + eps_r) / (1 + 11.45) and ignores thickness: 6.36 % (MAE 1.74 fF, RMSE 2.39 fF),
  vs. 0.48 % (MAE 0.13 fF, RMSE 0.18 fF) for ``QCap``. It warns.
- Tested range: Si at 10-1000 um, or eps_r 3.8-24 at 350 um. Outside it you get a warning and unknown error.

``result.substrate`` records what was done. Per-stack numbers: paper Sec. IV C 1.

Accuracy and time
^^^^^^^^^^^^^^^^^

Error vs. Q3D: median relative error of each layout's mutual capacitances, averaged over layouts (MAE / RMSE in fF).
The released models saw all our data except B5 (39 layouts of a new family). "Paper" rows were trained on the 35,354
TRAIN layouts only.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - set (layouts)
     - QCap
     - QCap-Fast
   * - released, in-sample fit (45,771)
     - 0.207 % (0.0679 fF / 0.077 fF)
     - 0.671 % (0.353 fF / 0.395 fF)
   * - released, B5 never seen (39)
     - 1.05 % (0.132 fF / 0.188 fF)
     - 3.175 % (0.583 fF / 0.824 fF)
   * - B5, physics prior alone (39)
     - 1.765 % (0.275 fF / 0.376 fF)
     - 1.765 % (0.275 fF / 0.376 fF)
   * - paper, validation (4,608)
     - 0.229 % (0.0674 fF / 0.0792 fF)
     - 0.846 % (0.491 fF / 0.549 fF)
   * - paper, test in family (4,009)
     - 0.177 % (0.0203 fF / 0.025 fF)
     - 0.63 % (0.113 fF / 0.1414 fF)
   * - paper, test zero-shot (1,878)
     - 0.587 % (1.092 fF / 1.152 fF)
     - 3.576 % (4.608 fF / 5.058 fF)
   * - paper, held-out family (4,071)
     - 0.408 % (0.049 fF / 0.059 fF)
     - 3.113 % (0.533 fF / 0.633 fF)

Time per layout, 2 threads:

- ``QCap``: 0.845 / 5.96 s (median / p90) on an x86 cluster node. With this package, median 1.88 s for N <= 4 and
  23.0 s for 10 <= N <= 20 on a GH200 node; 3.76 s and 46.2 s on an x86 Xeon Silver 4116.
- ``QCap-Fast``: 0.091 / 0.274 s on the cluster node; 0.093 s (GH200) and 0.17 s (Xeon) for N <= 4.

The first call downloads weights and compiles the solver (about 40 s, once); repeated layouts are cached. All cores
are used unless you set ``threads=`` (or ``SQUADDS_CAPML_THREADS``).

If a check in ``result.checks`` fails, SQuADDS warns. Do not use that matrix.

**Known issue:** a keyhole cut along a hole edge can lose a tiny ground sliver (7 of 60,093 dataset layouts). Fix
planned.

Install and quick start
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install "squadds[ml]"

Runs on Linux (x86, aarch64) and macOS. On aarch64, ``gdstk`` must be built from source.

.. code-block:: python

   from squadds.ml.capacitance import CapacitancePredictor, load_model

   predictor = CapacitancePredictor(load_model("QCap"))   # or "QCap-Fast"
   res = predictor.predict("chip.gds")          # also a gdstk Library / Cell, or a list of files
   res.names                                    # ['G', 'C1', 'C2', ...]
   res.matrix_fF                                # N x N Maxwell matrix, fF
   res.checks, res.timing_s, res.machine
   res.to_sim_results()                         # SQuADDS_DB ``sim_results`` shape (feeds squadds.calcs)

.. code-block:: bash

   squadds-ml predict chip.gds                          # QCap, CSV matrix
   squadds-ml predict a.gds b.gds --model QCap-Fast --format json
   squadds-ml predict chip.gds --ground 1/0 --conductor 1/10,2/10 --threads 4

The physics prior alone (no ML):

.. code-block:: python

   from squadds.ml.capacitance import physics_prior, optimized_physics_prior

   physics_prior("chip.gds").C_fF               # one solve
   optimized_physics_prior("chip.gds").C_fF     # the prior QCap corrects

Ground and conductors on one layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass the layer as both. SQuADDS splits the metal into pieces, takes the enclosing piece as ground, and warns. If the
pick is wrong, pass the ground's piece numbers:

.. code-block:: python

   from squadds.ml.capacitance import inspect_layout

   kw = dict(ground_layers=[(2, 0)], conductor_layers=[(2, 0)])
   inspect_layout("chip.gds", **kw)["figure"]       # every piece numbered
   res = predictor.predict("chip.gds", **kw, ground_pieces=[4])

.. code-block:: bash

   squadds-ml predict chip.gds --ground 2/0 --conductor 2/0 --ground-pieces 4

Inside SQuADDS workflows
^^^^^^^^^^^^^^^^^^^^^^^^

- ``MLSimulator`` (``squadds.ml.capacitance.bridge``) is a drop-in for the Q3D path of ``AnsysSimulator``.
- ``Analyzer.predict_capacitance(results_df)`` rebuilds the closest designs and predicts them.
- The Analyzer supports **TransmonPocket** and **Xmon** qubits (``squadds/calcs/floating_transmon.py``).
- ``SQuADDS_DB()`` reads the planar capacitance dataset as a second source; ``SQuADDS_DB(sources=("default",))``
  gives the original database alone.

The dataset
^^^^^^^^^^^

``SQuADDS/SQuADDS_PlanarCapacitance``: GDS layouts, Q3D matrices, geometric embeddings and physics priors. No
``[ml]`` extra needed:

.. code-block:: python

   from squadds.ml.capacitance.dataset import PlanarCapacitanceDB

   db = PlanarCapacitanceDB()
   ids = db.designs(family="concentric", split="val").design_id[:5]
   db.labels(ids)                                 # {design_id: N x N Maxwell matrix, fF, ground first}
   db.optimized_physics_prior(ids)

Test-set matrices are held out (layouts, embeddings and priors are public). Physics-only synthetic set:
``SQuADDS/SQuADDS_PlanarCapacitance-Synthetic``.

Training on your own designs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runs on a laptop, GPU box or cluster (``squadds.ml.capacitance.train``).

- ``build_shards(designs, out_dir)``: GDS files + Q3D ``cap_matrix.csv`` exports to training data.
- ``fit_in_family(designs, out_dir)``: fit on your designs, cross-validated against zero-shot ``QCap``.
- ``retrain_collective(user_shards, base_shards, out_dir)``: retrain on your data plus ours.
- ``squadds-ml train --data <shards> --out <dir>`` (``--slurm`` writes a SLURM array script).
- ``benchmark.score(model, shards, split="val")``: validation score.

See Tutorial 14.

Contributing data and models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the template ``squadds/database/templates/planar_capacitance_contribution``, then:

.. code-block:: python

   from squadds.ml.capacitance.contribute import contribute_capacitance, validate_capacitance_contribution

   print(validate_capacitance_contribution("my_batch/").to_markdown())
   contribute_capacitance("my_batch/")          # opens a pull request on the dataset

The validator checks schema, GDS, conductor count and the physics of every matrix. For a model:
``contribute_model("you/your-model", revision)``. See Tutorial 15.

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

.. _planar-capacitance-citation:

Citing
^^^^^^

If you use QCap or SQuADDS in your work, please cite both papers:
`Shanto, Das and Levenson-Falk, arXiv:2611.22057 (2026) <https://arxiv.org/abs/2611.22057>`__ and
`Shanto et al., Quantum 8, 1465 (2024) <https://doi.org/10.22331/q-2024-09-09-1465>`__.

.. code-block:: bibtex

   @article{Shanto2024squaddsvalidated,
     doi = {10.22331/q-2024-09-09-1465},
     url = {https://doi.org/10.22331/q-2024-09-09-1465},
     title = {{SQ}u{ADDS}: {A} validated design database and simulation workflow for superconducting qubit design},
     author = {Shanto, Sadman and Kuo, Andre and Miyamoto, Clark and Zhang, Haimeng and Maurya, Vivek and Vlachos, Evangelos and Hecht, Malida and Shum, Chung Wa and Levenson-Falk, Eli},
     journal = {{Quantum}},
     issn = {2521-327X},
     publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
     volume = {8},
     pages = {1465},
     month = sep,
     year = {2024}
   }

   @misc{Shanto2026planarcapacitance,
     title = {{QCap}: a physics-corrected graph neural network for the capacitance matrix of any planar superconducting device layout},
     author = {Shanto, Sadman Ahmed and Das, Saikat and Levenson-Falk, Eli},
     year = {2026},
     eprint = {2611.22057},
     archivePrefix = {arXiv},
     primaryClass = {quant-ph},
     url = {https://arxiv.org/abs/2611.22057}
   }


.. _hosted-inference:

Hosted inference models
-----------------------

An HTTP API for the models on our `Hugging Face org <https://huggingface.co/SQuADDS>`_. No install needed.

- **Host:** ``https://squadds-squadds-ml-inference-api.hf.space``
- **Interactive docs:** `/docs <https://squadds-squadds-ml-inference-api.hf.space/docs>`_ and `/openapi.json <https://squadds-squadds-ml-inference-api.hf.space/openapi.json>`_
- **Space:** `SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_

``GET /v1/models`` lists what is live (don't hard-code ids). Today: ``transmon_cross_hamiltonian_inverse``
(see :ref:`hosted-live-model`), ``qcap`` and ``qcap-fast``.

Routes
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Route
     - Returns
   * - ``GET /v1/health``
     - Service status and the state of each model.
   * - ``GET /v1/models``
     - The models you can call, with their ``input_kind`` (``json`` or ``file``) and ``status``.
   * - ``GET /v1/models/{model_id}``
     - The model's inputs, outputs, units, limits and an example request.
   * - ``GET /v1/models/{model_id}/schema``
     - The request's JSON schema; ``?format=tool`` gives a tool definition for an AI agent.
   * - ``POST /v1/models/{model_id}/predict``
     - The prediction.

Predict
^^^^^^^

Send ``{"inputs": {...}}``; ``inputs`` may be a list for a batch. A file model (e.g. a GDS) also takes
``multipart/form-data`` with a ``file`` field.

.. code-block:: bash

   curl -X POST \
     https://squadds-squadds-ml-inference-api.hf.space/v1/models/transmon_cross_hamiltonian_inverse/predict \
     -H 'Content-Type: application/json' \
     -d '{"inputs": {"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0}}'

.. code-block:: json

   {
     "model_id": "transmon_cross_hamiltonian_inverse",
     "outputs": {
       "predictions": [
         {
           "design_options.connection_pads.readout.claw_length": 0.00011072495544794947,
           "design_options.connection_pads.readout.ground_spacing": 4.571595582092414e-06,
           "design_options.cross_length": 0.0002005973074119538
         }
       ]
     },
     "metadata": {"output_units": {"design_options.cross_length": "m", "...": "m"}},
     "timing_s": {"total": 0.012, "queue": 0.0}
   }

Errors: ``{"error": {"code": ..., "message": ...}}``. Gated models need ``Authorization: Bearer <token>`` from a
SQuADDS org member. The old ``/health``, ``/models`` and ``/predict`` routes still work; use ``/v1``.

Python client
^^^^^^^^^^^^^

.. code-block:: python

   from squadds.ml import client

   client.list_models()
   res = client.predict(
       "transmon_cross_hamiltonian_inverse",
       inputs={"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0},
   )
   res["outputs"]["predictions"][0]                 # geometry in meters
   # a file model: client.predict(model_id, file="chip.gds")

``SQUADDS_ML_API_URL`` overrides the host; ``token=`` (or ``HF_TOKEN``) is for gated models.

For AI agents
^^^^^^^^^^^^^

1. ``GET /v1/models`` and pick a model whose ``status`` is ``"ready"``.
2. ``GET /v1/models/{model_id}/schema?format=tool`` and register the result as a tool.
3. Send the tool call's arguments to ``POST /v1/models/{model_id}/predict``.

The SQuADDS MCP server does the same with ``list_hosted_ml_models`` and ``predict_with_hosted_ml_model``
(:doc:`mcp_server`).

.. _hosted-live-model:

TransmonCross Hamiltonian-to-geometry model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``transmon_cross_hamiltonian_inverse``: target ``qubit_frequency_GHz`` and ``anharmonicity_MHz`` in, TransmonCross
claw length, ground spacing and cross length (meters) out. Convert to ``"...um"`` strings for ``design_options``.

- **Paper:** `Component-Level Inverse Design of Transmon Qubits Using Neural Networks <https://arxiv.org/abs/2607.20795>`_ (Seidel et al., 2026)
- **Model:** `SQuADDS/transmon-cross-hamiltonian-inverse <https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse>`_
- **On the API:** `model page <https://squadds-squadds-ml-inference-api.hf.space/v1/models/transmon_cross_hamiltonian_inverse>`_

Developed with Taylor Patti, Nicola Pancotti, Enectali Figueroa-Feliciano, Sara Sussman, Abhishek Chakraborty, Olivia
Seidel, Firas Abouzahr, Eli Levenson-Falk, and Sadman Ahmed Shanto; **Olivia Seidel and Firas Abouzahr** trained it.

Limitations
^^^^^^^^^^^

The Space runs on shared hardware and returns ``429`` when its queue is full. For large batches, run locally
(:doc:`ml_local`).

Contributing a model
^^^^^^^^^^^^^^^^^^^^

Push the weights to a Hugging Face model repo with an ``inference_manifest.json`` (input/output names, units, files).
An existing family (``keras_tabular``, ``planar_capacitance``) needs only a registry entry and an example request; a
new kind needs one adapter class. Then open an issue or PR on
`SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_ with the repo,
units and held-out accuracy (MAE and RMSE with any percentage).
