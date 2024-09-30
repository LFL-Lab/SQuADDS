# Use the official Miniconda3 image
FROM continuumio/miniconda3

# Install required build tools and GUI dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the environment.yml from your repo into the container
COPY environment.yml .

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Activate the conda environment and set it as the default
SHELL ["conda", "run", "-n", "squadds-env", "/bin/bash", "-c"]

# Copy the rest of your repo into the container
COPY . .

# Install SQuADDS from the local source (your repo)
RUN python -m pip install --upgrade pip && pip install -e .

# Install Qiskit Metal from their GitHub repository
RUN python -m pip install --no-deps -e git+https://github.com/Qiskit/qiskit-metal.git#egg=qiskit-metal

# Set the entry point for the Docker container, adjust this to how you run your package
CMD ["conda", "run", "-n", "squadds-env", "python", "-m", "squadds"]
