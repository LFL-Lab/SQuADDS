.. _ml-local:

Run the capacitance models on your own computer
===============================================

This page is for you if you reached a limit of the online demo (10 predictions per day, layouts with fewer than 70
conductors including the ground), if you want to predict many layouts, or if you would rather not upload a design. The
demo and the hosted API run on free, shared Hugging Face hardware (two shared vCPUs). The models are the same, and on your
computer they run much faster, on all your cores, with no limits. That is the way we recommend.

1. Install
----------

.. code-block:: bash

   pip install "squadds[ml]"

SQuADDS supports Python 3.11, 3.12 and 3.13. The models run on Linux (x86 and aarch64) and macOS. A fresh virtual
environment is the safest place to install them.

2. Predict one layout in Python
-------------------------------

.. code-block:: python

   from squadds.ml.capacitance import CapacitancePredictor, load_model

   predictor = CapacitancePredictor(load_model("QCap"))
   result = predictor.predict("chip.gds")
   print(result.names, result.matrix_fF)

``result.matrix_fF`` is the full Maxwell capacitance matrix in fF. Row and column 0 are the ground ``G``, then ``C1``,
``C2`` and so on, in GDS file order. For quick screening of the component families it was trained on, use
``load_model("QCap-Fast")``. Use QCap for anything that matters. Its accuracy is in
:ref:`planar-capacitance-models`.

3. Predict from the command line
--------------------------------

.. code-block:: bash

   squadds-ml predict chip.gds                              # QCap, matrix as CSV
   squadds-ml predict chip.gds --model QCap-Fast --format json   # QCap-Fast, matrix and metadata as JSON

4. Layers and substrate
-----------------------

By default the ground plane is on GDS layer 1/0 and the conductors are on layer 1/10. To use other layers:

.. code-block:: python

   result = predictor.predict("chip.gds", ground_layers=[(1, 0)], conductor_layers=[(1, 10), (2, 10)])

.. code-block:: bash

   squadds-ml predict chip.gds --ground 1/0 --conductor 1/10,2/10

If the ground and the conductors are drawn on one layer, give that layer for both. The ground is then detected
automatically and the result warns which piece it picked. "Ground and conductors on one layer" in :doc:`ml_models` shows
how to check the pick and correct it.

The models were trained on silicon (eps_r 11.45), 350 µm thick. For another substrate, give its permittivity and
thickness:

.. code-block:: python

   result = predictor.predict("chip.gds", eps_r=10.34, thickness_um=430)

.. code-block:: bash

   squadds-ml predict chip.gds --eps-r 10.34 --thickness-um 430

QCap solves its physics prior at your substrate. QCap-Fast only rescales its silicon result and
warns, so use QCap on other substrates. The measured errors are in "Other substrates" in :doc:`ml_models`.

5. Many layouts at once
-----------------------

Pass a list of files and the number of processes to use. A list in gives a list of results out, in the same order.

.. code-block:: python

   results = predictor.predict(["a.gds", "b.gds", "c.gds"], workers=4)

.. code-block:: bash

   squadds-ml predict a.gds b.gds c.gds --workers 4

By default SQuADDS uses every core your process may use, at most one per 2.5 GB of RAM and at most 32. With
``--workers`` the cores are split between the processes. To limit it, pass ``threads=`` / ``--threads`` or set
``SQUADDS_CAPML_THREADS``. Results are deterministic for a given thread count and machine, so pass the same ``threads`` to
reproduce a result exactly; between thread counts they agree to rounding (2e-7 relative at most on a 67-conductor grid).

6. What to expect for time
--------------------------

The first prediction downloads and checks the weights and compiles the thin-film solver. That takes about 40 s, once per
environment. A layout you already predicted in the same session comes back from the cache in milliseconds. Measured
median times per layout with 2 threads (N is the size of the matrix, ground included):

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - model
     - machine
     - N 4 or less
     - N 10 to 20
   * - QCap
     - x86 Xeon Silver 4116
     - 3.76 s
     - 46.2 s
   * - QCap
     - GH200 node
     - 1.88 s
     - 23.0 s
   * - QCap-Fast
     - x86 Xeon Silver 4116
     - 0.17 s
     - not measured
   * - QCap-Fast
     - GH200 node
     - 0.093 s
     - not measured

On an Apple M5 Pro laptop with 2 threads, ten designs on substrates other than the training one took a median of 3.6 s
per layout with QCap and 0.16 s with QCap-Fast.

7. Get help
-----------

- Questions, problems and bug reports: `GitHub issues <https://github.com/LFL-Lab/SQuADDS/issues>`_.
- Email: `shanto@usc.edu <mailto:shanto@usc.edu>`__.
- More examples: `Tutorial 11: Predicting capacitance matrices from GDS
  <https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.html>`_ and
  the full reference in :doc:`ml_models`.
