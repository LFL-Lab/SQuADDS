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
   Activate a clean conda environment (with qiskit-metal) and install dependencies.

   .. code-block:: bash

      conda activate <YOUR-ENV>
      cd SQuADDS
      pip install datasets
      pip install pyEPR-quantum
      pip install pyaedt
      pip install -e . 
      pip install -r requirements.txt

.. admonition:: Questions?

   Please reach out to `shanto@usc.edu <mailto:shanto@usc.edu>`__


********************************
FAQ's
********************************

Frequently asked questions.

--------------------
Installation Issues
--------------------

**Q: Getting**``ModuleNotFoundError: No module named 'squadds'``**after running `pip install SQuADDS` in Jupyter Notebook. How can I fix this?**


**A:** You may need to restart the kernel after installing `SQuADDS`. To do this, go to the `Kernel` menu in Jupyter Notebook and select `Restart`.

-----------------------
Accessing the Database
-----------------------

Q: **If there are errors upon instantiating the **``SQuADDS_DB``**class, what should I do?**

**A:** If you encounter errors upon instantiating the ``SQuADDS_DB`` class, chances are there is an issue with caching. To fix this, please delete the ``SQuADDS`` dataset from the huggingface cache directory on your local machine. The cache directory is typically located at ``~/.cache/huggingface/datasets/``.