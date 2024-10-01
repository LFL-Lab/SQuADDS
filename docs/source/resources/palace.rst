.. _installation-palace:

Installation of Palace on HPC systems
===========================================

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

   This runs the capacitor example using the Palace executable.

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

Follow these steps for installation:

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
    ```

After running these scripts, you should be able to launch Palace by running:

.. code-block:: sh
   
   $ palace

If no errors are encountered, your installation is likely successful. 

Installation of Palace on Windows Systems
=========================================

Palace is not officially supported on Windows. However, it is possible to compile Palace on Windows using Visual Studio. The following is a guide to compile Palace on Windows inspired by https://welsim.com/.

Compilation Method
------------------

Palace provides the Superbuild compilation method with CMake, which automatically downloads all required libraries and compiles them completely. It compiles effortlessly on Linux. However, on Windows, many core libraries such as ``PETSc, SLEPc, libCEED, MUMPS``, and others require manual compilation. Therefore, the Superbuild mode provided officially cannot compile as smoothly on Windows. Users need to apply the manual method of establishing Visual Studio projects to complete the building.

System and Dependency Libraries
-------------------------------

- **Operating System**: Windows 10, 64-bit
- **Compiler**: Visual Studio 2022 Community, C++17. Intel Fortran Compiler 2022.
- **Palace Version**: 0.11.2

**Dependency Libraries**:

- **Intel MKL**: A popular linear algebra solver, using oneAPI 2022.2.0, consistent with the version of Fortran compiler.
- **METIS**: A mesh partitioning tool for parallel computing, version 5.3.
- **Hypre**: A computational library, version 2.52.
- **nlhmann/json**: Modern C++-based JSON read-write package.
- **{fmt}**: Formatting tool for input-output streams in C/C++.
- **Eigen**: A well-known C++ numerical computing package, has no need for compilation; supports direct header file invocation.
- **libCEED**: A linear algebra computation management terminal that supports parallel computing on various CPUs, GPUs, and clusters.
- **SuperLU_DIST**: The parallel version of SuperLU, a sparse direct linear algebra solver library.
- **STRUMPACK**: An open-source software library for large-scale sparse matrix computing.
- **MUMPS**: An open-source software library from France for solving large-scale sparse linear systems.
- **SLEPc**: A complex number linear algebra solver for eigenvalue problems, based on PETSc.
- **ARPACK-NG**: A complex number linear algebra solver for eigenvalue problems, programmed using Fortran 77 language.
- **GSLIB**: An interpolation solver for high-order spectral elements, optional.

Among these, at least one of the three optional linear solvers ``SuperLU_DIST, STRUMPACK, MUMPS`` must be present. This article uses ``MUMPS``. Additionally, out of the two complex solvers, ``SLEPc`` and ``ARPACK``, at least one is required. Without them, eigenvalue-related computing cannot be performed. This article uses ``ARPACK``.

Visual Studio Projects
----------------------

Establish two projects, namely the static library project ``libpalace``, and the executable file project ``palace``. ``libpalace`` contains all header and source files. ``palace`` is the final generated executable file, containing only a ``main.cpp`` file. This is shown in the figure.

**Project libpalace**

Set the external header file directories.

Add preprocessor macros:

- ``CEED_SKIP_VISIBILITY``
- ``PALACE_WITH_ARPACK``
- ``_CRT_SECURE_NO_WARNINGS``

**Project palace**

The method to add external header files and preprocessor macros is essentially the same as ``libpalace``, so it will not be repeated here. Compiling the executable program requires linking all dependent libraries. The added linked libraries are as follows,

After building, place all dependent dynamic libraries (``*.dll`` files) together with the ``palace.exe`` to run Palace. Test the executable program by running it on the Windows console.

We have open-sourced the building files for Palace, shared at `https://github.com/WelSimLLC/palace <https://github.com/WelSimLLC/palace>`_, and provided the compiled ``palace.exe`` executable file for users to use directly.

You may need to ``C:\Program Files\Microsoft Visual Studio\{Year}\{Licence}\VC\Redist\MSVC\v{version}\vc_redist.x64.exe`` to run the executable program and restart the program.
