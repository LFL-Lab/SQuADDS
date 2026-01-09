.. _installation-palace:

Palace Installation Guide
=========================

Installation of Palace on HPC systems
=====================================

Follow instructions to install Palace on Linux/HPC systems from the official palace site: https://awslabs.github.io/palace/stable/install/

https://github.com/sqdlab/SQDMetal/blob/main/SQDMetal/PALACE/HPC_documentation.md is also a good resource for installation on HPC systems.

Installation of Palace on Mac OS
================================

Installing Palace on macOS involves using Homebrew to manage dependencies and compiling the software using CMake.

Follow these steps to ensure a smooth installation process. It was tested to work on macOS Darwin 22.3.0 (Apple M1 Pro chip).

Prerequisites
-------------

- **Homebrew**: A package manager for macOS. If not installed, run:

  .. code-block:: bash

     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

- **Xcode Command Line Tools**: Required for development tools. Install by running:

  .. code-block:: bash

     xcode-select --install

Installation Steps
------------------

1. **Update Homebrew**:

   .. code-block:: bash

      brew update

2. **Install Dependencies**:

   Install necessary packages using Homebrew:

   .. code-block:: bash

      brew install cmake gcc open-mpi openblas git

   - **cmake**: For configuring the build process.
   - **gcc**: Provides the Fortran compiler `gfortran`.
   - **open-mpi**: MPI distribution for parallel computations.
   - **openblas**: Provides BLAS and LAPACK libraries.
   - **git**: For cloning the Palace repository.

3. **Set Environment Variables**:

   Ensure CMake can find OpenBLAS:

   .. code-block:: bash

      export CMAKE_PREFIX_PATH=$(brew --prefix openblas)

4. **Clone the Palace Repository**:

   .. code-block:: bash

      git clone https://github.com/awslabs/palace.git
      cd palace

5. **Create a Build Directory**:

   .. code-block:: bash

      mkdir build && cd build

6. **Configure the Build with CMake**:

   Run CMake with appropriate compilers and options:

   .. code-block:: bash

      cmake -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DCMAKE_Fortran_COMPILER=$(brew --prefix gcc)/bin/gfortran \
            -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
            ..

   - **CMAKE_C_COMPILER** and **CMAKE_CXX_COMPILER**: Use default Clang compilers.
   - **CMAKE_Fortran_COMPILER**: Use `gfortran` from Homebrew GCC.
   - **CMAKE_PREFIX_PATH**: Helps CMake locate OpenBLAS.

7. **Build Palace**:

   Compile the software using:

   .. code-block:: bash

      make -j $(sysctl -n hw.ncpu)

   This utilizes all available CPU cores for faster compilation.

8. **Verify the Installation**:

   After building, the Palace executable is in the `bin/` directory:

   .. code-block:: bash

      ls bin/

   You should see an executable named `palace`.

9. **Run an Example**:

   Test the installation by running an example:

   .. code-block:: bash

      cd ../examples/cpw
      ../../build/bin/palace cpw_wave_uniform.json

   This runs the `cpw_wave_uniform` example using the Palace executable.

Optional Steps
--------------

- **Install Palace System-wide**:

  To install Palace to a specific directory (e.g., `/usr/local`), reconfigure with `CMAKE_INSTALL_PREFIX`:

  .. code-block:: bash

     cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
           -DCMAKE_C_COMPILER=clang \
           -DCMAKE_CXX_COMPILER=clang++ \
           -DCMAKE_Fortran_COMPILER=$(brew --prefix gcc)/bin/gfortran \
           -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
           ..

     make -j $(sysctl -n hw.ncpu)
     sudo make install

  This installs the Palace executable to `/usr/local/bin`.

Notes
-----

- **Fortran Compiler**: `gfortran` is required for building some dependencies and is provided by Homebrew GCC.

- **MPI Support**: OpenMPI provides the necessary MPI support for parallel computations.

- **BLAS and LAPACK**: OpenBLAS supplies these libraries, essential for numerical computations.

- **Xcode Command Line Tools**: Needed for Clang compilers and development tools.

Troubleshooting
---------------

- **CMake Cannot Find OpenBLAS**:

  Ensure `CMAKE_PREFIX_PATH` is set correctly:

  .. code-block:: bash

     export CMAKE_PREFIX_PATH=$(brew --prefix openblas)

- **MPI Errors**:

  Confirm that OpenMPI is properly installed and in your PATH.


Installation of Palace on Linux PCs
===================================

Installing Palace on Linux PCs can be somewhat challenging due to its reliance on high-performance libraries that depend on specific CPU instruction sets. The version of OpenMPI included in Ubuntu's standard repositories is not compatible with Palace, necessitating the use of Spack to install a compatible version of OpenMPI.

These scripts were tested on a fresh Ubuntu 22.04 installation as of 11/22/2023.

Installation Steps
------------------

1. Run the following script to install the required dependencies:

.. code-block:: sh

    #!/bin/bash

    # Update and upgrade packages
    sudo apt-get update
    sudo apt-get upgrade

    # Install utilities
    sudo apt-get install gmsh paraview

    # Install Spack prerequisites
    sudo apt-get install build-essential ca-certificates coreutils curl environment-modules gfortran git gpg lsb-release python3 python3-distutils python3-venv unzip zip

    # Install Palace prerequisites
    sudo apt-get install pkg-config build-essential cmake python3 mpi-default-dev

2. Run the following script to install Spack, set up MPI, and build Palace:

.. code-block:: sh

    #!/bin/bash

    # This script will:
    # 1. Install Spack in the current directory
    # 2. Install MPI via Spack
    # 3. Set the system MPI to Spack's MPI
    # 4. Clone the Palace repository
    # 5. Build Palace

    spack_repo="https://github.com/spack/spack.git"
    palace_repo="https://github.com/awslabs/palace.git"

    # Install Spack
    echo 'Installing Spack to:'
    echo $spack_install_dir
    git clone -c feature.manyFiles=true $spack_repo
    . spack/share/spack/setup-env.sh

    # Install MPI
    echo 'Installing MPI'
    spack install mpi

    mpi_info=($(spack find -p mpi))
    mpi_dir=${mpi_info[-1]}
    mpi_bin_dir="$mpi_dir/bin"

    # Set up paths
    echo 'MPI bin directory:'
    echo $mpi_bin_dir
    echo -n 'export PATH="' > setup_palace_env.sh
    echo -n $mpi_bin_dir >> setup_palace_env.sh
    echo ':$PATH"' >> setup_palace_env.sh
    source setup_palace_env.sh

    # Install Palace
    echo 'Installing Palace to:'
    echo $palace_install_dir
    git clone --recurse-submodules $palace_repo
    cd palace
    mkdir build
    cd build
    cmake ..
    make -j

After running these scripts, you should be able to launch Palace by running:

.. code-block:: sh

   $ palace

If no errors are encountered, your installation is likely successful.

Installation of Palace on Windows Systems
=========================================

Using a Prebuilt Executable (Recommended)
-----------------------------------------

If you want to avoid the complex compilation process, you can use a **prebuilt Palace executable** provided by **WELSIM**. The WELSIM company has built and open-sourced **Palace v1.11.0** as part of their simulation software suite.

1. Download the installer from the official GitHub release:

   - https://github.com/WelSimLLC/WelSim-Apps/releases/download/3.1/WelSim31Setup.exe
   - Full release page: https://github.com/WelSimLLC/WelSim-Apps/releases/tag/3.1

2. Run ``WelSim31Setup.exe`` and follow the installation instructions.

3. Once installed, the ``palace.exe`` binary will be located in the installation directory, e.g.:

   .. code-block:: sh

      C:\Program Files\WELSIM\v3.1\palace.exe

**Note**: This version is Palace **v1.11.0**, which is older than the current upstream version, but it should be sufficient for most simulations.

Building Palace on Windows (Not Recommended for Beginners)
----------------------------------------------------------

The installation process involves:

1. Installing necessary compilers and tools
2. Manually downloading and building each dependency
3. Setting up a Visual Studio solution with two projects
4. Linking everything and building

Step 1: Install Tools and Compilers
-----------------------------------

Visual Studio 2022
~~~~~~~~~~~~~~~~~~

1. Download Visual Studio 2022 Community from: https://visualstudio.microsoft.com/vs/community/
2. During installation, ensure the following components are selected:
   - Desktop development with C++
   - MSVC v14.x
   - C++ CMake tools
   - Windows SDK

Intel oneAPI Base & HPC Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
2. Select both **Base Toolkit** and **HPC Toolkit**.
3. After installation, run:

   ::

      "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

Step 2: Build Dependencies
--------------------------

METIS 5.1.0
~~~~~~~~~~~

1. Download from: https://github.com/CIBC-Internal/metis-4.0.3
2. Build:

   ::

      mkdir build && cd build
      cmake .. -G "Visual Studio 17 2022" -A x64
      cmake --build . --config Release

   Output: ``libmetis.lib``, headers in ``include/``

HYPRE 2.25.0
~~~~~~~~~~~~

1. Download from: https://github.com/hypre-space/hypre
2. Build:

   ::

      mkdir build && cd build
      cmake .. -G "Visual Studio 17 2022" -A x64 -DHYPRE_WITH_MPI=OFF
      cmake --build . --config Release

   Output: ``libHYPRE.lib``, headers in ``src/``

libCEED
~~~~~~~

1. Clone the repository:

   ::

      git clone https://github.com/CEED/libCEED
      cd libCEED
      mkdir build && cd build
      cmake .. -G "Visual Studio 17 2022" -A x64
      cmake --build . --config Release

SuperLU_DIST
~~~~~~~~~~~~

1. Clone the repository:

   ::

      git clone https://github.com/xiaoyeli/superlu_dist
      cd superlu_dist
      mkdir build && cd build
      cmake .. -G "Visual Studio 17 2022" -A x64 -DTPL_BLAS_LIBRARIES="path_to_mkl.lib"
      cmake --build . --config Release

STRUMPACK
~~~~~~~~~

1. Clone the repository:

   ::

      git clone https://github.com/pghysels/STRUMPACK
      cd STRUMPACK
      mkdir build && cd build
      cmake .. -G "Visual Studio 17 2022" -A x64
      cmake --build . --config Release

MUMPS
~~~~~

- Building MUMPS on Windows is challenging. If needed, follow the guide at:
  https://github.com/scivision/mumps-builder

SLEPc and PETSc
~~~~~~~~~~~~~~~

- These are difficult to build on Windows.
- Consider using Windows Subsystem for Linux (WSL) or prebuilt binaries.
- Alternatively, remove ``PALACE_WITH_ARPACK`` to skip SLEPc.

Header-only Libraries
~~~~~~~~~~~~~~~~~~~~~

- Download and extract:
  - https://github.com/nlohmann/json
  - https://github.com/fmtlib/fmt
  - https://gitlab.com/libeigen/eigen

Step 3: Set Up Visual Studio Projects
-------------------------------------

1. Create a new **Empty Solution** called ``PalaceVS``
2. Add two projects:
   - ``libpalace``: Static library
   - ``palace``: Console Application
3. Clone Palace repository: https://github.com/awslabs/palace
4. Add files:
   - Put ``src/`` and ``fem/`` into ``libpalace``
   - Put ``main.cpp`` into ``palace``

Step 4: Configure Build Settings
--------------------------------

For both projects:

1. Configuration → C++ → General → Additional Include Directories:
   - Add paths to headers from METIS, HYPRE, libCEED, etc.
   - Add paths to header-only libraries

2. Linker → Input → Additional Dependencies:
   - Add: ``metis.lib;hypre.lib;ceed.lib;superlu.lib;...``

3. Preprocessor Definitions (for libpalace):
   - ``CEED_SKIP_VISIBILITY``
   - ``PALACE_WITH_ARPACK``
   - ``_CRT_SECURE_NO_WARNINGS``

Step 5: Build and Run
---------------------

1. Build ``libpalace`` (Release x64)
2. Build ``palace``
3. Copy required ``.dll`` files into the output folder
4. Run ``palace.exe`` from command line

**Note**: You may need to install the Visual C++ Redistributable package from:

::

   C:\Program Files\Microsoft Visual Studio\{Year}\{Edition}\VC\Redist\MSVC\v{version}\vc_redist.x64.exe

to run the final ``palace.exe`` binary.
