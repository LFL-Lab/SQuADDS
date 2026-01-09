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
