Getting Started with SQuADDS
============================

.. image:: https://img.shields.io/badge/-Setup-blue
   :alt: Setup

This guide helps you get started with ``SQuADDS``

Installation
============

``SQuADDS`` needs Python 3.11 or newer (we test 3.11, 3.12 and 3.13). It uses `uv <https://docs.astral.sh/uv/>`_ for fast, reliable Python package management.

Prerequisites
-------------

Install ``uv`` (if you don't have it already):

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Install from Source (Recommended)
---------------------------------

.. code-block:: bash

   git clone https://github.com/LFL-Lab/SQuADDS.git
   cd SQuADDS
   uv sync

Verify the installation:

.. code-block:: bash

   uv run python -c "import squadds; print(squadds.__file__)"

Install using pip
-----------------

.. code-block:: bash

   pip install SQuADDS

Optional Dependencies
---------------------

Install GDS processing tools:

.. code-block:: bash

   uv sync --extra gds

Install the planar capacitance models (PyTorch, gdstk and the physics prior):

.. code-block:: bash

   uv sync --extra ml

or, with pip:

.. code-block:: bash

   pip install "squadds[ml]"

Install documentation tools:

.. code-block:: bash

   uv sync --extra docs

Install development tools:

.. code-block:: bash

   uv sync --extra dev

Install contribution tools (for contributing data to SQuADDS):

.. code-block:: bash

   uv sync --extra contrib

Install the MCP server for AI agents:

.. code-block:: bash

   uv sync --extra mcp

Install all optional dependencies:

.. code-block:: bash

   uv sync --all-extras

Setting up Jupyter Notebook
---------------------------

To use ``SQuADDS`` in Jupyter notebooks (including VS Code, Cursor, or JupyterLab), register the virtual environment as a Jupyter kernel:

.. code-block:: bash

   # Install ipykernel (included in dev dependencies)
   uv sync --extra dev

   # Register the kernel
   uv run python -m ipykernel install --user --name squadds --display-name "SQuADDS (uv)"

After running these commands:

- **In VS Code/Cursor**: Open a ``.ipynb`` file, click on the kernel selector (top-right), and select **"SQuADDS (uv)"**
- **In JupyterLab/Notebook**: Select **"SQuADDS (uv)"** from the kernel dropdown
- **From command line**: Run notebooks with ``uv run jupyter notebook`` or ``uv run jupyter lab``

Installing Additional Tools
---------------------------

Installing ``SQDMetal``
~~~~~~~~~~~~~~~~~~~~~~~

Once you have ``SQuADDS`` installed, you can install ``SQDMetal`` by:

.. code-block:: bash

   git clone https://github.com/sqdlab/SQDMetal.git
   cd SQDMetal
   pip install .

Installing ``palace``
~~~~~~~~~~~~~~~~~~~~~

``palace`` is a powerful open source electromagnetic simulation tool that can be used with ``SQuADDS``. For detailed installation instructions, please refer to our :doc:`Palace Installation Guide <resources/palace>`.

.. admonition:: Questions?

   Please reach out to `shanto@usc.edu <mailto:shanto@usc.edu>`__ if you face any installation issues.

.. _faq-s:

FAQs
====

We have compiled answers to common questions and issues. If you can't find what you're looking for, feel free to reach out.

Installation
------------

**Q: Which Python versions does** ``SQuADDS`` **support?**

**A:** Python 3.11 or newer. We test 3.11, 3.12 and 3.13. Python 3.10 has reached its end of life, and its last SciPy release cannot load ``scipy.sparse.linalg`` on current macOS.

**Q: Getting** ``ModuleNotFoundError: No module named 'squadds'`` **after installation in Jupyter Notebook. How can I fix this?**

**A:** You may need to restart the kernel after installing ``SQuADDS``. To do this, go to the `Kernel` menu in Jupyter Notebook and select `Restart`.

**Q: Getting** ``ERROR: Failed building wheel for klayout`` **while building from GitHub in Windows**

**A:** The ``gds`` extra pulls in ``klayout``. Install KLayout independently from the website `here <https://www.klayout.de/build.html>`_, then install SQuADDS without the gds extra.

**Q: On Linux aarch64,** ``import qiskit_metal`` **fails. Can I still use SQuADDS?**

**A:** Yes. ``quantum-metal`` needs PySide6, which needs glibc 2.39 on aarch64, so the component-building workflows do not load there. The planar capacitance models (``squadds[ml]``) predict and train there as usual. ``gdstk`` has no aarch64 wheel, so pip builds it from source.

Accessing the Database
----------------------

**Q: I am getting errors while loading the dataset. What should I do?**

**A:** If you encounter errors upon instantiating the `SQuADDS_DB` class, there may be a caching issue. Delete the ``SQuADDS`` datasets from the huggingface cache directory on your local machine. The cache directory is typically located at ``~/.cache/huggingface/datasets/``.

**Q:** ``SQuADDS_DB()`` **now has a** ``planar_capacitance`` **source. How do I get the original database alone?**

**A:** ``SQuADDS_DB()`` loads the planar capacitance dataset next to the original database by default. ``SQuADDS_DB(sources=("default",))`` gives the original database alone. If the planar capacitance dataset cannot be read, ``SQuADDS`` warns once and returns the original rows.

**Q: Can I use the capacitance models and dataset offline, or from my own copy?**

**A:** Yes. Point ``SQUADDS_PLANAR_CAP_DATA`` at a local copy of the planar capacitance dataset and ``SQUADDS_CAPML_MODELS`` at a directory of model bundles. Downloads from Hugging Face pin a revision and check the sha256 of every file.

Designs and Simulations
-----------------------

**Q: Can I give a list of values for a design parameter (e.g. the claw length)?**

**A:** Yes. Any design option can be a list, and every combination is simulated:

.. code-block:: python

   from squadds.simulations import AnsysSimulator, expand_design_options

   simulator = AnsysSimulator(analyzer, best_device)
   simulator.update_design_parameters(claw_length=["150um", "200um"], cross_length=["200um", "250um"])
   results = simulator.simulate()  # four designs, one result each

   designs = expand_design_options({"cross_length": ["200um", "250um"], "cross_gap": "30um"})  # two design dicts

A parameter that appears in both the qubit and the cavity options (such as ``claw_length``) takes one value per design. Use a tuple for an option whose value must stay a fixed sequence.

**Q: How do I build a flux-tunable** ``TransmonCross`` **with a SQUID loop?**

**A:** Set ``style="SQUID_LOOP_Dolan"`` and configure the loop with ``SQUID_LOOP=dict(SQUID_width="40um", SQUID_length="5um", SQUID_offset="5um", JJ_width="150nm", JJ_flip=False)``. The loop and its two Dolan junctions are added as the component ``<qubit name>_SQUID_LOOP_Dolan``. ``style="JJ_Dolan"`` with ``Dolan_JJ=dict(JJ_width=..., JJ_flip=...)`` draws a single Dolan junction (see Tutorial 5).

Planar Capacitance Models
-------------------------

**Q: How do I install the capacitance models? Do I need a GPU?**

**A:** ``pip install "squadds[ml]"`` (or ``uv sync --extra ml``). They run on a CPU, about a second per layout on a laptop. No GPU is needed.

**Q: Which model should I use, QCap or QCap-Fast?**

**A:** ``QCap`` (the default) for anything that matters. ``QCap-Fast`` skips the physics solve and is about ten times quicker, but on a family it has never seen it does worse than the physics prior alone, so use it only to screen the families it was trained on. The numbers are in :ref:`planar-capacitance-models`.

**Q: My layout does not use layers 1/0 and 1/10. What do I do?**

**A:** The ground is read from layer 1/0 and the conductors from 1/10 by default. Pass your own:

.. code-block:: python

   res = predictor.predict("chip.gds", ground_layers=[(1, 0)], conductor_layers=[(1, 10), (2, 10)])

If the ground and the conductors are on one layer, give that layer for both. The ground is then detected automatically, and ``ground_pieces=[...]`` overrides the pick. ``inspect_layout`` shows what the model will read, with every piece numbered, before you predict.

**Q: Why is my matrix larger than I expected?**

**A:** We never merge nets. The matrix is N x N for every conductor the layout has, ground first (row and column 0), in fF, with Maxwell signs (positive diagonal, negative off-diagonal).

**Q: What substrates can I use?**

**A:** The models are trained on 350 µm silicon (eps_r 11.45). They are tested for silicon from 10 to 1000 µm thick, and for eps_r from 3.8 to 24 at 350 µm. Pass ``eps_r=`` and ``thickness_um=`` to ``predict``. Other combinations work but warn, because their error is unknown.

**Q: How large a chip can I predict?**

**A:** Locally there is no conductor limit. On an Apple M5 Pro with its 18 cores, which SQuADDS uses by default, a 3x3 qubit grid (67 conductors) takes about 12 s (peak memory about 20 GB) and a 100-qubit chip (781 conductors) about 200 s. The browser demo and the hosted API run on free, shared Hugging Face compute (two shared vCPUs), so they take smaller layouts and are slower: the API up to 70 conductors plus ground and 20 MB per file, the demo layouts with fewer than 70 conductors, ground included, and 10 of your own layouts per day once you sign in with Hugging Face. For large chips, run SQuADDS on your own machine.

**Q: How many CPU cores does SQuADDS use?**

**A:** All the cores your process may use, at most one per 2.5 GB of RAM (each core adds memory on large layouts) and at most 32. Pass ``threads=`` (or ``--threads`` on the command line), or set ``SQUADDS_CAPML_THREADS``, to use fewer. Results are deterministic for a given thread count and machine; between thread counts they agree to rounding.

**Q: How much memory do the models need?**

**A:** A 10 µm Xmon peaks at about 1.6 GiB. The predictor's caches are bounded at 1 GiB each, set by ``SQUADDS_CAPML_CACHE_BYTES``.

**Q: Exact rebuilds of the multipad star designs differ from the dataset. Why?**

**A:** They need ``shapely`` older than 2.1. Install ``"shapely<2.1"`` for exact rebuilds.

**Q: What is** ``cross_to_cross`` **in the TransmonCross table?**

**A:** It holds the same value as ``cross_to_ground``, the cross-to-ground coupling, not the cross's total self-capacitance. SQuADDS builds the qubit's total capacitance as ``cross_to_ground + cross_to_claw``, so the Hamiltonian results do not use ``cross_to_cross``. A fresh Q3D solve of one row reproduced the stored entries within 0.4 % (MAE 0.27 fF, RMSE 0.32 fF) and put the full self-capacitance 6.75 % (8.0 fF) above the stored ``cross_to_cross``. About 2.2 fF of that is coupling to conductors the table does not list. The simulations used silicon with eps_r 11.45.

Citing, AI Agents and Compute
-----------------------------

**Q: How do I cite SQuADDS?**

**A:** Use the BibTeX in the `Citation <https://github.com/LFL-Lab/SQuADDS#citation>`_ section of the README (also the ``squadds://citation`` resource of the MCP server). Cite our 2024 paper for ``SQuADDS``, and our 2026 paper as well if you use the planar capacitance models or their dataset.

**Q: Can my AI agent use SQuADDS?**

**A:** Yes, through the MCP server. Install it with ``uv sync --extra mcp --extra ml`` and run ``squadds-mcp``. The README's `MCP Server <https://github.com/LFL-Lab/SQuADDS#mcp-server-ai-agent-integration>`_ section has the setup for each client.

**Q: Can I contribute compute?**

**A:** Yes, please. We ask for compute (simulation licences, CPU and GPU time) to grow the datasets and retrain the models. Contact `elevenso@usc.edu <mailto:elevenso@usc.edu>`__.

``.env`` File
-------------

**Q: Why is the** ``.env`` **file needed?**

**A:** The ``.env`` file is needed for making contributions to the ``SQuADDS`` Database. Capacitance data and model contributions add its contributor fields to the submission when it exists. They open a pull request on Hugging Face, so they need a Hugging Face token: pass ``token=``, set ``HF_TOKEN``, or set ``HUGGINGFACE_API_KEY`` in the ``.env``.

**Q: What info should the** ``.env`` **file contain?**

**A:** The ``.env`` file should have the following fields defined.

.. code-block:: bash

   GROUP_NAME=
   PI_NAME=
   INSTITUTION=
   USER_NAME=
   CONTRIB_MISC=
   HUGGINGFACE_API_KEY=
   GITHUB_TOKEN=

You can set these fields via the ``SQuADDS`` API.

.. code-block:: python

   from squadds.core.utils import set_huggingface_api_key, set_github_token
   from squadds.database.utils import create_contributor_info

   create_contributor_info()
   set_huggingface_api_key()
   set_github_token()

**Q: Where is the** ``.env`` **file created or should be placed for it to function properly?**

**A:** The ``.env`` file should be automatically created at the right place within the root directory of the ``SQuADDS`` package. If the ``.env`` file is not automatically created upon installation, you will need to manually create it at this specific location for ``SQuADDS`` to function properly.

To determine the installation root of ``SQuADDS``, and subsequently place or find the ``.env`` file, use the following approach:

.. code-block:: python

   from pathlib import Path
   import squadds

   # Locate the root of the SQuADDS installation
   squadds_root = Path(squadds.__file__).parent.parent

   # installed via pip
   if "site-packages" in str(squadds_root):
      squadds_root = Path(squadds.__file__).parent
   else: # not pypi installed
      pass

   # Path to the expected .env file location
   env_file_path = squadds_root / '.env'
   print(env_file_path)

   if env_file_path.exists():
      print(f"Found .env file at: {env_file_path}")
   else:
      print(".env file not found at the expected location.")
      print(f"To function properly, create a .env file at: {squadds_root}")

.. |SQuADDS| replace:: SQuADDS
