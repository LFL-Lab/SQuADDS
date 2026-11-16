"""Machine-learning models for SQuADDS (optional extra: ``pip install squadds[ml]``).

Submodules import torch / gdstk / numba lazily, so ``import squadds`` works without the extra.
"""

from __future__ import annotations

import importlib.util

ML_REQUIREMENTS = ("torch", "gdstk", "numba", "threadpoolctl", "shapely", "yaml")


def missing_ml_dependencies() -> list[str]:
    return [m for m in ML_REQUIREMENTS if importlib.util.find_spec(m) is None]


def require_ml() -> None:
    """Raise ImportError naming the missing packages when the ``[ml]`` extra is not installed."""
    missing = missing_ml_dependencies()
    if missing:
        raise ImportError(
            f"squadds.ml needs the optional ML dependencies (missing: {', '.join(missing)}). "
            "Install them with: pip install 'squadds[ml]'"
        )


def __getattr__(name):
    if name == "capacitance":
        require_ml()
        return importlib.import_module("squadds.ml.capacitance")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ML_REQUIREMENTS", "missing_ml_dependencies", "require_ml"]
