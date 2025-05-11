Getting Started with SQuADDS
============================

.. image:: https://img.shields.io/badge/-Setup-blue
   :alt: Setup

This guide helps you get started with **SQuADDS**

Installation
============

SQuADDS is built on top of `qiskit-metal`, which is a crucial dependency. The installation process depends on whether you already have `qiskit-metal` installed in your environment or not.

1. Installing SQuADDS with Existing `qiskit-metal`
----------------------------------------------------
.. _installing-squadds-with-existing-qiskit-metal:

If you already have `qiskit-metal` installed and running in your environment, you can install SQuADDS in two ways:

a) Using pip:

.. code-block:: bash

   pip install SQuADDS

b) From source:

.. code-block:: bash

   git clone https://github.com/LFL-Lab/SQuADDS.git
   cd SQuADDS
   pip install -e .

2. Installing SQuADDS on a Fresh Environment
-------------------------------------------
.. _installing-squadds-on-a-fresh-environment:

If you don't have `qiskit-metal` installed, you'll need to set up a new environment first. We provide a shell script that:

- Creates a new conda environment with Python 3.10
- Installs `qiskit-metal` and its dependencies
- Sets up the environment for SQuADDS

Run the following script:

.. code-block:: bash

   #!/bin/bash

   # Ensure script fails if any command fails
   set -e

   # Step 1: Download environment.yml from Qiskit-Metal repository
   echo "Downloading environment.yml..."
   curl -O https://raw.githubusercontent.com/Qiskit/qiskit-metal/main/environment.yml

   # Step 2: Set up Miniconda environment
   echo "Setting up Conda environment..."
   conda env create -n <env_name> -f environment.yml
   echo "Conda environment created."

   # Activate the Conda environment
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate <env_name>

   # Step 3: Install Qiskit-Metal
   echo "Installing Qiskit-Metal..."
   python -m pip install --no-deps -e git+https://github.com/Qiskit/qiskit-metal.git#egg=qiskit-metal

After running this script, you'll have a working `qiskit-metal` environment. You can then follow the instructions in :ref:`installing-squadds-with-existing-qiskit-metal` to install SQuADDS.

Installing SQuADDS on Apple Silicon
-----------------------------------

`qiskit-metal` currently lacks full native support for Apple Silicon due to `PySide` compatibility issues. However, you can run SQuADDS on Apple Silicon by emulating the `x86` architecture with Rosetta 2.

First, ensure Rosetta 2 is installed:

.. code-block:: bash

   softwareupdate --install-rosetta

Then, create a new conda environment configured to emulate `x86`:

.. code-block:: bash

   # Create environment with x86 emulation
   CONDA_SUBDIR=osx-64 conda create -n <env_name> python=3.10
   conda activate <env_name>
   conda config --env --set subdir osx-64

This environment will now use Rosetta 2 to run x86 applications. You can then follow the same principles as in :ref:`installing-squadds-on-a-fresh-environment` to set up `qiskit-metal` in this environment.

.. code-block:: bash

   #!/bin/bash

   # Ensure script fails if any command fails
   set -e

   # Step 1: Download environment.yml from Qiskit-Metal repository
   echo "Downloading environment.yml..."
   curl -O https://raw.githubusercontent.com/Qiskit/qiskit-metal/main/environment.yml

   # Step 2: Update the existing conda environment
   echo "Updating Conda environment..."
   conda env update -n <env_name> -f environment.yml
   echo "Conda environment updated."

   # Activate the Conda environment
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate <env_name>

   # Step 3: Install Qiskit-Metal
   echo "Installing Qiskit-Metal..."
   python -m pip install --no-deps -e git+https://github.com/Qiskit/qiskit-metal.git#egg=qiskit-metal

Now, for installing `SQuADDS`, follow the same principles as in :ref:`installing-squadds-with-existing-qiskit-metal`.

.. note::
   The `CONDA_SUBDIR=osx-64` flag tells conda to use x86 packages instead of arm64 packages, and `conda config --env --set subdir osx-64` ensures this setting persists for the environment.

Installing Additional Dependencies
---------------------------------

`SQDMetal` and `palace` are optional dependencies that can be used with `SQuADDS` for additional simulation capabilities.

Installing `SQDMetal`
~~~~~~~~~~~~~~~~~~~~~

Once you have `SQuADDS` and `qiskit-metal` installed, you can install `SQDMetal` by:

.. code-block:: bash

   git clone https://github.com/sqdlab/SQDMetal.git
   cd SQDMetal
   pip install .

Installing `palace`
~~~~~~~~~~~~~~~~~~~

`Palace` is a powerful open source electromagnetic simulation tool that can be used with `SQuADDS`. For detailed installation instructions, please refer to our :doc:`Palace Installation Guide <resources/palace>`.

.. admonition:: Questions?

   Please reach out to `shanto@usc.edu <mailto:shanto@usc.edu>`__ if you face any installation issues.

FAQs
====

We have compiled answers to common questions and issues. If you can't find what you're looking for, feel free to reach out.

Installation Issues
-------------------

**Q: Getting** ``ModuleNotFoundError: No module named 'squadds'`` **after running** `pip install SQuADDS` **in Jupyter Notebook. How can I fix this?**

**A:** You may need to restart the kernel after installing `SQuADDS`. To do this, go to the `Kernel` menu in Jupyter Notebook and select `Restart`.

**Q: Getting** ``ERROR: Failed building wheel for klayout`` **while building from GitHub in Windows**

**A:** This problem can be solved simply by installing KLayout independently from the website `here <https://www.klayout.de/build.html>`_, and commenting out the ``klayout==0.29.0`` in the ``requirements.txt`` file.
The ``requirements.txt`` file can found in the cloned repository. Then re-run the commands. 

Accessing the Database
-----------------------

**Q: I am getting the error** ``Generating train split: 0 examples [00:00, ? examples/s] An error occurred while loading the dataset: An error occurred while generating the dataset`` **for various** ``SQuADDS_DB()`` **methods (e.g.** ``SQuADDS_DB().create_system_df()`` **).**
 
**A:** This is an error we have seen only happening on Windows systems for ``datasets`` library version ``2.20.0``. Downgrading to any versions between ``2.17.0`` and ``2.19.2`` should fix the issue. To downgrade, run the following command:

.. code-block:: bash

   pip install datasets==2.19.2


**Q: I am getting the error** ``KeyError: "Column contributor not in the dataset. Current columns in the dataset: ['image', 'measured_results', 'contrib_info', 'design_code', 'notes', 'sim_results', 'paper_link']"`` **for various** ``SQuADDS_DB()`` **methods (e.g.** ``SQuADDS_DB().view_all_contributors()`` **). Everything was working fine just the other day.**

**A:** This error is due to new datasets (configs) added to ``SQuADDS/SQuADDS_DB`` dataset on 07/04/2024 (🇺🇸 🦅 🎆). To fix this issue please upgrade ``squadds`` to its latest version (or any version greater than or equal to ``0.2.35``).

**Q: If there are errors upon instantiating the** ``SQuADDS_DB`` **class, what should I do?**

**A:** If you encounter errors upon instantiating the `SQuADDS_DB` class, chances are there is an issue with caching. To fix this, please delete the ``SQuADDS`` dataset from the huggingface cache directory on your local machine. The cache directory is typically located at ``~/.cache/huggingface/datasets/``.

``.env`` File 
-------------

**Q: Why is the** ``.env`` **file needed?**

**A:** The ``.env`` file is needed for making contributions to the SQuADDS Database.

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

You can set these fields via the SQuADDS API.

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
