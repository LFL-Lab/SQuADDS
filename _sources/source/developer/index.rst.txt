.. _dev_notes:

Developer Notes
===============

Everyone is welcome to contribute to SQuADDS. Please review the following section for more information or contact us!

Development Setup
-----------------

SQuADDS uses `uv <https://docs.astral.sh/uv/>`_ for fast, reliable Python package management.

**Prerequisites:**

Install ``uv``:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

**Clone and Setup:**

.. code-block:: bash

   git clone https://github.com/LFL-Lab/SQuADDS.git
   cd SQuADDS
   uv sync --extra dev

**Running Tests:**

.. code-block:: bash

   uv run pytest tests/ -v

**Setting Up Pre-commit Hooks (Recommended):**

.. code-block:: bash

   uv run pre-commit install

This will automatically run formatting and linting every time you commit.

**Running Linter Manually:**

.. code-block:: bash

   uv run ruff check .
   uv run ruff format --check .

**Running All Pre-commit Hooks:**

.. code-block:: bash

   uv run pre-commit run --all-files

**Building Documentation:**

.. code-block:: bash

   uv sync --extra docs
   cd docs
   uv run make html

Contribution Items
------------------

| **Bug Reports** - Please report any bugs you find in the code or documentation by opening an issue on GitHub.

| **Feature Requests** - If you have an idea for a new feature, please open an issue on GitHub.

| **Pull Requests** - We welcome pull requests from the community. If you submit a pull request, please be patient as we review your work. If you have any questions about contributing, please contact us!

| **Documentation** - We welcome contributions to the documentation. If you find any typos or errors, please open an issue on GitHub.

| **Implementation of our wish list** - If you would like to help in implementing the next version of SQuADDS, please look at `wish list <https://github.com/LFL-Lab/SQuADDS/blob/master/wish_list.md>`_ and feel free to contribute!

Please see our `Contributing Guidelines <https://github.com/LFL-Lab/SQuADDS/blob/master/CONTRIBUTING.md>`_ for more information on how to get started.

.. note::

   If at any point you are convinced that something is wrong but the documentation/code says otherwise, you may **absolutely be right**. Please open an issue on GitHub and we will address it as soon as possible.

Project Structure
-----------------

.. code-block:: text

   SQuADDS/
   ├── squadds/           # Main package source code
   │   ├── calcs/         # Calculation modules
   │   ├── components/    # Qiskit Metal component definitions
   │   ├── core/          # Core functionality (db, analysis, utils)
   │   ├── database/      # HuggingFace integration
   │   ├── gds/           # GDS processing utilities
   │   ├── interpolations/# Interpolation algorithms
   │   ├── simulations/   # ANSYS/Palace simulation interfaces
   │   └── ui/            # Streamlit web interface
   ├── tests/             # Test suite
   ├── docs/              # Sphinx documentation
   ├── tutorials/         # Jupyter notebook tutorials
   ├── pyproject.toml     # Project configuration (PEP 621)
   └── uv.lock            # Dependency lock file

Developers
----------

| `Sadman Ahmed Shanto <https://www.sadmanahmedshanto.com>`_ (University of Southern California) - Project Lead 🤖
| `Andre Kuo <https://www.linkedin.com/in/andrekuo>`_ (HRL Laboratories)


Contributors
------------

| **Eli Levenson-Falk, PhD** (University of Southern California) - Eternal Guidance Provider (Principle Investigator) 🙏🏽
| **Clark Miyamoto** (New York University) - Code contributor 💻
| **Madison Howard** (California Institute of Technology) - Bug Hunter 🐛
| **Evangelos Vlachos** (University of Southern California) - Code contributor 💻
| **Kaveh Pezeshki** (Stanford University) - Documentation contributor 📄
| **Anne Whelan** (US Navy) - Documentation contributor 📄
| **Jenny Huang** (Columbia University) - Documentation contributor 📄
| **Connie Miao** (Stanford University) - Data Contributor 📀
| **Malida Hecht** (University of Southern California) - Data contributor 📀
| **Daria Kowsari, PhD** (University of Southern California) - Data contributor 📀
| **Vivek Maurya** (University of Southern California) - Data contributor 📀
| **Haimeng Zhang, PhD** (IBM) - Data contributor 📀
| **Elizabeth Kunz** (University of Southern California) - Documentation 📄 and  Code contributor 💻
| **Adhish Chakravorty** (University of Southern California) - Documentation 📄 and  Code contributor 💻
| **Ethan Zheng** (University of Southern California) - Data contributor 📀 and Bug Hunter 🐛
| **Sara Sussman, PhD** (Fermilab) - Bug Hunter 🐛
| **Priyangshu Chatterjee** (IIT Kharagpur) -  Documentation contributor 📄
| **Abhishek Chakraborty** (Rigetti Computing) - Code contributor 💻
| **Saikat Das** (University of Southern California) - Reviewer ✅
| **Firas Abouzahr** (Northwestern) - Bug Hunter 🐛
