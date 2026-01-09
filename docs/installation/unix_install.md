# Installing SQuADDS on Mac/Linux

SQuADDS uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

## Prerequisites

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
git clone https://github.com/LFL-Lab/SQuADDS.git
cd SQuADDS
uv sync
```

Verify the installation:

```bash
uv run python -c "import squadds; print(squadds.__file__)"
```

## Optional Dependencies

Install GDS processing tools:

```bash
uv sync --extra gds
```

Install all optional dependencies:

```bash
uv sync --all-extras
```

## Setting up Jupyter Notebook

To use SQuADDS in Jupyter notebooks (including VS Code, Cursor, or JupyterLab), you need to register the virtual environment as a Jupyter kernel:

```bash
# Install ipykernel (included in dev dependencies)
uv sync --extra dev

# Register the kernel
uv run python -m ipykernel install --user --name squadds --display-name "SQuADDS (uv)"
```

After running these commands:

1. **In VS Code/Cursor**: Open a `.ipynb` file, click on the kernel selector (top-right), and select **"SQuADDS (uv)"**
2. **In JupyterLab/Notebook**: Select **"SQuADDS (uv)"** from the kernel dropdown
3. **From command line**: Run notebooks with `uv run jupyter notebook` or `uv run jupyter lab`

## Environment Configuration

Create a `.env` file at the root of SQuADDS with the following content (required for contributing to the database):

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

You can also set these fields via the SQuADDS API:

```python
from squadds.core.utils import set_huggingface_api_key, set_github_token
from squadds.database.utils import create_contributor_info

create_contributor_info()
set_huggingface_api_key()
set_github_token()
```
