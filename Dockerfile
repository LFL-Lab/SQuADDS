# Use official Python image
FROM python:3.11-slim

# Install required build tools and GUI dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libgl1 \
    libegl1 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock README.md LICENSE ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy the rest of your repo into the container
COPY . .

# Install the package in editable mode
RUN uv sync --frozen --no-dev

# Set the entry point for the Docker container
CMD ["uv", "run", "python", "-c", "import squadds; print('SQuADDS ready!')"]
