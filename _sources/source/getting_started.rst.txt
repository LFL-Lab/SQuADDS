Getting Started with SQuADDS
============================

.. image:: https://img.shields.io/badge/-Setup-blue
   :alt: Setup

Installation
------------

SQuADDS can be installed using pip.

.. code-block:: bash

   pip install SQuADDS

Alternatively, you can install SQuADDS from source.

1. **Clone Repository**: 
   Navigate to your chosen directory and clone the repository.

   .. code-block:: bash

      cd <REPO-PATH>
      git clone https://github.com/LFL-Lab/SQuADDS.git

2. **Install Dependencies**: 
   Activate your Qiskit Metal conda environment and install dependencies.

   .. code-block:: bash

      conda activate <YOUR-ENV>
      cd SQuADDS
      pip install -e .

FAQ
===

Q: **Getting `ModuleNotFoundError: No module named 'squadds'` after running `pip install SQuADDS` in Jupyter Notebook. How can I fix this?**
----------------------------------------------------------------------------------------------------------------------------------------

A: You may need to restart the kernel after installing `SQuADDS`. To do this, go to the `Kernel` menu in Jupyter Notebook and select `Restart`.
