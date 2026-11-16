"""Flagship tiers (registry keys flagship_hub, flagship_tiers).

The tiers run through the shared arm code path (``_core.paper``: scene -> embedding -> priors under the tier's prior stack, or the
learned prior surrogate -> networks), unchanged. `view(reg)` is the registry seen with paper_arms / paper_hub replaced by
flagship_tiers / flagship_hub, so every ``_core.paper`` function serves the tiers as arms.
"""

from __future__ import annotations

import copy

from . import paper as PA


def view(reg):
    if reg.data.get("_flagship_view"):
        return reg
    v = copy.copy(reg)
    v.data = dict(reg.data, paper_arms=reg.data["flagship_tiers"], paper_hub=reg.data["flagship_hub"], _flagship_view=True)
    return v


def tiers(reg) -> dict:
    return view(reg)["paper_arms"]


def served(reg) -> list[str]:
    v = view(reg)
    return [k for k in PA.live(v) if PA.available(k, v)]


def predict(prepared, tier, reg, **kw) -> dict:
    """Full N x N Maxwell matrix (fF; ground row/column 0) of a prepared layout with a flagship tier (``_core.paper.predict``)."""
    v = view(reg)
    res = PA.predict(prepared, tier, v, **kw)
    t = v["paper_arms"][tier]
    res.pop("paper", None)
    res.update(tier=tier, tier_title=t["title"], reference=t.get("reference") or {}, own=t.get("own") or {}, b5=t.get("b5"), b5_physics=t.get("b5_physics"))
    return res


def preload(reg, example=None) -> dict:
    return PA.preload(view(reg), example=example)
