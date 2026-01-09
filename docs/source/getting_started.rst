Getting Started with SQuADDS
============================

.. image:: https://img.shields.io/badge/-Setup-blue
   :alt: Setup

This guide helps you get started with ``SQuADDS``

Installation
============

``SQuADDS`` uses `uv <https://docs.astral.sh/uv/>`_ for fast, reliable Python package management.

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

Install documentation tools:

.. code-block:: bash

   uv sync --extra docs

Install development tools:

.. code-block:: bash

   uv sync --extra dev

Install all optional dependencies:

.. code-block:: bash

   uv sync --all-extras

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

FAQs
====

We have compiled answers to common questions and issues. If you can't find what you're looking for, feel free to reach out.

Installation Issues
-------------------

**Q: Getting** ``ModuleNotFoundError: No module named 'squadds'`` **after installation in Jupyter Notebook. How can I fix this?**

**A:** You may need to restart the kernel after installing ``SQuADDS``. To do this, go to the `Kernel` menu in Jupyter Notebook and select `Restart`.

**Q: Getting** ``ERROR: Failed building wheel for klayout`` **while building from GitHub in Windows**

**A:** Install KLayout independently from the website `here <https://www.klayout.de/build.html>`_, then install SQuADDS without the gds extra.

Accessing the Database
-----------------------

**Q: I am getting errors while loading the dataset. What should I do?**

**A:** If you encounter errors upon instantiating the `SQuADDS_DB` class, there may be a caching issue. Delete the ``SQuADDS`` dataset from the huggingface cache directory on your local machine. The cache directory is typically located at ``~/.cache/huggingface/datasets/``.

``.env`` File 
-------------

**Q: Why is the** ``.env`` **file needed?**

**A:** The ``.env`` file is needed for making contributions to the ``SQuADDS`` Database.

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
