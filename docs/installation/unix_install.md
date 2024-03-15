# Installing SQuADDS on a Fresh Mac/Linux Conda Environment

Create a shell script with the following content. This allows a local development environment for SQuADDS.

```bash
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
```

**If on an M1+ macOS, open your terminal in Rosetta mode.**

1. Ensure you have a `.env` file (at root of SQuADDS) with the following content:

```bash
# .env
GROUP_NAME=
PI_NAME=
INSTITUTION=
USER_NAME=
CONTRIB_MISC=
HUGGINGFACE_API_KEY=
GITHUB_TOKEN=
```

2. Update the path to your `.env` file in the script.

3. Give the script execute permissions (i.e. `chmod +x` it) and run it - `./your_script_name.sh`.
