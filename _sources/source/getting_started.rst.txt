Getting Started with SQuADDS
============================

.. image:: https://img.shields.io/badge/-Setup-blue
   :alt: Setup

This guide helps you get started with **SQuADDS**, a database for superconducting qubit and device design and simulation.

Installation
============

There are multiple ways to install SQuADDS. Choose the method that best suits your environment.

.. contents::
   :local:

Install using pip
-----------------

SQuADDS can be installed using pip in an environment with `qiskit-metal` pre-installed.

.. code-block:: bash

   pip install SQuADDS

Install from GitHub
-------------------

Alternatively, you can install SQuADDS from source.

1. **Clone Repository**: Navigate to your chosen directory and clone the repository.

   .. code-block:: bash

      cd <REPO-PATH>
      git clone https://github.com/LFL-Lab/SQuADDS.git

2. **Install Dependencies**: Activate a clean conda environment (with qiskit-metal) and install dependencies.

   .. code-block:: bash

      conda activate <YOUR-ENV>
      cd SQuADDS
      pip install -r requirements.txt
      pip install -e . 

Install using Docker
--------------------

You can use our Docker image to run SQuADDS. The Docker image is available `here <https://github.com/LFL-Lab/SQuADDS/pkgs/container/squadds_env>`__. Instructions on how to use the Docker image can be found `here <https://github.com/LFL-Lab/SQuADDS?tab=readme-ov-file#run-using-docker>`__

Fresh Environment Installation
-------------------------------

For installing SQuADDS (from PyPi) on a completely fresh environment on a UNIX machine, you can use the following shell script.

.. code-block:: bash

   #!/bin/bash

   # Ensure script fails if any command fails
   set -e

   # Step 1: Download environment.yml from Qiskit-Metal repository
   echo "Downloading environment.yml..."
   curl -O https://raw.githubusercontent.com/Qiskit/qiskit-metal/main/environment.yml

   # Step 2: Set up Miniconda environment
   echo "Setting up Conda environment..."
   conda env list
   conda remove --name qiskit-metal-env --all --yes || true
   conda env create -n qiskit-metal-env -f environment.yml
   echo "Conda environment created."

   # Activate the Conda environment
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate qiskit-metal-env

   # Step 3: Install Qiskit-Metal
   echo "Installing Qiskit-Metal..."
   python -m pip install --no-deps -e git+https://github.com/Qiskit/qiskit-metal.git#egg=qiskit-metal

   # Step 4: Install SQuADDS from PyPi
   echo "Installing SQuADDS from pypi"
   pip install SQuADDS

You can also use the GitHub version of SQuADDS by changing Step 4 to:

.. code-block:: bash

   # Step 4: Install SQuADDS from source
   echo "Installing SQuADDS from source"
   # Clone the repository
   git clone https://github.com/LFL-Lab/SQuADDS.git
   cd SQuADDS
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .

.. admonition:: Questions?

   Please reach out to `shanto@usc.edu <mailto:shanto@usc.edu>`__ if you face any installation issues.

FAQs
====

We have compiled answers to common questions and issues. If you can't find what you're looking for, feel free to reach out.

Installation Issues
-------------------

**Q: Getting** ``ModuleNotFoundError: No module named 'squadds'`` **after running** `pip install SQuADDS` **in Jupyter Notebook. How can I fix this?**

**A:** You may need to restart the kernel after installing `SQuADDS`. To do this, go to the `Kernel` menu in Jupyter Notebook and select `Restart`.

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
