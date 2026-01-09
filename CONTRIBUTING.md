# Contributing to SQuADDS

Thank you for your interest in contributing to SQuADDS! Here are some guidelines to help you get started.

## Development Setup

SQuADDS uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/LFL-Lab/SQuADDS.git
cd SQuADDS
uv sync --extra dev

# Install pre-commit hooks (recommended - auto-formats on commit)
uv run pre-commit install

# Run tests
uv run pytest tests/ -v

# Run linter manually
uv run ruff check .
uv run ruff format --check .
```

## Making Contributions

If you have improvements or additions to the database, please follow these steps:

- Fork the repository.
- Create a new branch for your contribution (`git checkout -b feature/your-feature-name`).
- Make your changes and ensure tests pass.
- Run the linter: `uv run ruff check . && uv run ruff format .`
- Submit a pull request with the following commit message guidelines.

---

## Commit Message Guidelines

Since 05/31/2024, we started to follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification to ensure that our commit messages are structured and informative. This helps us automatically generate release notes using [Release Drafter](https://github.com/release-drafter/release-drafter).

### Commit Message Format

Each commit message should be structured as follows:

```

<type>: <description>

[optional body]

[optional footer(s)]

```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools and libraries such as documentation generation

### Examples

- **Feature Commit**

```

feat: add support for new quantum device types

```

- **Bug Fix Commit**

```

fix: resolve inconsistencies in edge case simulations

```

- **Breaking Change Commit**

```

feat: overhaul API endpoints

BREAKING CHANGE: Deprecated API endpoints have been removed.

```

### Labels for Pull Requests

When opening a pull request, please use the following labels to help categorize the changes:

- `feature`: For new features
- `bug`: For bug fixes
- `breaking`: For breaking changes

## Using Release Drafter

We use [Release Drafter](https://github.com/release-drafter/release-drafter) to automate the generation of release notes. Release Drafter categorizes changes based on the labels assigned to pull requests.

### Setting Up Release Drafter

The configuration for Release Drafter is stored in the `.github/release-drafter.yml` file. Here’s a summary of our configuration:

```yaml
name-template: 'v$NEXT_PATCH_VERSION'
tag-template: 'v$NEXT_PATCH_VERSION'
categories:
- title: '🚀 New Features'
  labels:
    - 'feature'
- title: '🐛 Bug Fixes'
  labels:
    - 'bug'
- title: '💥 Breaking Changes'
  labels:
    - 'breaking'
change-template: '- $TITLE (#$NUMBER) @$AUTHOR'
version-resolver:
major:
  - 'breaking'
minor:
  - 'feature'
patch:
  - 'fix'
  - 'bug'
  - 'docs'
template: |
## What's Changed
$CHANGES
```

By following these guidelines, you help maintainers streamline the release process and keep the project organized.

If you have any questions or need further assistance, please don't hesitate to reach out!
