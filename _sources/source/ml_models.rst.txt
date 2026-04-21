ML Models (Hosted Inference)
============================

SQuADDS hosts ML models trained on the SQuADDS dataset on our `Hugging Face org <https://huggingface.co/SQuADDS>`_, served through the `SQuADDS ML Inference API Space <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_. Use this page as the entry point when you want a stable HTTP surface for inverse-design predictions rather than loading Keras checkpoints yourself.

.. contents:: On this page
   :local:
   :depth: 2


Live Endpoints
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Resource
     - URL
   * - Space repo
     - `huggingface.co/spaces/SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_
   * - API host
     - ``https://squadds-squadds-ml-inference-api.hf.space``
   * - Current model repo
     - `huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse <https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse>`_

API routes:

- ``GET /health`` — liveness.
- ``GET /models`` — list deployed models and their ``status`` / input-output contract.
- ``POST /predict`` — run inference for a given ``model_id`` with its exact input keys.


Recommended Agent Workflow
--------------------------

1. Call ``GET /models`` and inspect the response.
2. Pick a model whose ``status`` is ``"ready"``.
3. Send ``POST /predict`` with that ``model_id`` and the exact input keys it advertises.
4. Feed the returned geometry parameters straight into SQuADDS / Qiskit Metal / validation flows.


Current Live Model
------------------

``transmon_cross_hamiltonian_inverse`` predicts TransmonCross (qubit-claw) geometry from target Hamiltonian inputs.

- **Expected inputs** (SQuADDS-native units): ``qubit_frequency_GHz``, ``anharmonicity_MHz``.
- **Returned outputs** (SI units, meters): ``design_options.connection_pads.readout.claw_length``, ``design_options.connection_pads.readout.ground_spacing``, ``design_options.cross_length``.


Sample Request
^^^^^^^^^^^^^^

.. code-block:: bash

   curl -X POST \
     https://squadds-squadds-ml-inference-api.hf.space/predict \
     -H 'Content-Type: application/json' \
     -d '{"model_id":"transmon_cross_hamiltonian_inverse","inputs":{"qubit_frequency_GHz":4.85,"anharmonicity_MHz":-205.0}}'


Sample Response
^^^^^^^^^^^^^^^

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

Full per-model contract (``X_names``, output order, scalers, ``inference_manifest.json``) lives on the `model repo <https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse>`_.


Acknowledgments
---------------

The first live model (``transmon_cross_hamiltonian_inverse``) was developed in collaboration with Taylor Patti, Nicola Pancotti, Enectali Figueroa-Feliciano, Sara Sussman, Olivia Seidel, Firas Abouzahr, Eli Levenson-Falk, and Sadman Ahmed Shanto — with **Olivia Seidel and Firas Abouzahr** as the primary trainers.


Current Limitations
-------------------

- Only ``transmon_cross_hamiltonian_inverse`` is live today. Resonator and coupled-system inverse models are next — the deployment tooling already knows about those families, so they drop in once checkpoints land.
- Outputs are returned in meters; convert to SQuADDS/Qiskit-Metal micrometer strings (``"…um"``) before writing back into ``design_options``.
- No authentication today; the Space is rate-limited by Hugging Face. Do not rely on it for production-scale batch inference.


Contributing a Model
--------------------

If you have a well-performing SQuADDS-based model, please PR it in. Open an issue or PR against `SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_ with:

- a model checkpoint following the existing ``model/`` / ``scalers/`` / ``inference_manifest.json`` layout,
- the exact input/output columns and units, and
- a short description that can go on the Space's model card.


Further Reading
---------------

- Space README: `huggingface.co/spaces/SQuADDS/squadds-ml-inference-api <https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api>`_
- Model repo README + ``inference_manifest.json``: `huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse <https://huggingface.co/SQuADDS/transmon-cross-hamiltonian-inverse>`_
- SQuADDS dataset: `huggingface.co/datasets/SQuADDS/SQuADDS_DB <https://huggingface.co/datasets/SQuADDS/SQuADDS_DB>`_
