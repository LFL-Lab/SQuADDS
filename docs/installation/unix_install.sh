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

# Step 4: Install SQuADDS and its dependencies
echo "Installing SQuADDS and dependencies..."
# Clone the repository
git clone https://github.com/LFL-Lab/SQuADDS.git
cd SQuADDS
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Step 5: Set up environment variables
source ../.env

# Step 6: Run Tests
echo "Running tests..."
python tests/imports_test.py
python tests/mvp_test.py

echo "All tests completed successfully."
