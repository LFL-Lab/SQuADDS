"""exp6/exp7 SQuADDS_DB interdigital rows rebuild on the tangent spine-ledge geometry they were simulated on."""

from __future__ import annotations

import pandas as pd
import pytest

from squadds.components import build_component_from_design
from squadds.components.component_registry import row_campaign

OPTIONS = {"finger_count": "3", "finger_length": "8um", "finger_width": "4um"}


def hf_row(campaign, options=None):
    return {
        "design": {"design_options": dict(options or OPTIONS), "component_class": "GeneralizedCapNInterdigital"},
        "notes": {"source_id": f"{campaign}/cap_0000", "source_campaign": campaign},
    }


@pytest.fixture
def design(headless_qiskit_environment):
    from qiskit_metal import designs

    return designs.DesignPlanar({}, overwrite_enabled=True)


@pytest.mark.parametrize("campaign", ["exp6", "exp7"])
def test_exp6_exp7_rows_get_the_tangent_rule(design, campaign):
    row = hf_row(campaign)
    comp = build_component_from_design(design, row)
    assert comp.options["spine_ledge_rule"] == "tangent"
    assert "spine_ledge_rule" not in row["design"]["design_options"]


@pytest.mark.parametrize("campaign", ["q3d_cap", "exp8", "exp9", None])
def test_other_rows_keep_their_options(design, campaign):
    row = hf_row(campaign) if campaign else {"design": hf_row("x")["design"]}
    comp = build_component_from_design(design, row)
    assert comp.options.get("spine_ledge_rule", "auto") == "auto"


def test_explicit_option_wins(design):
    comp = build_component_from_design(design, hf_row("exp6", OPTIONS | {"spine_ledge_rule": "auto"}))
    assert comp.options["spine_ledge_rule"] == "auto"


def test_flattened_rows_and_campaign_sources(design):
    flat = pd.Series(
        {"design_options": dict(OPTIONS), "coupler_type": "GeneralizedCapNInterdigital", "source_campaign": "exp7"}
    )
    assert build_component_from_design(design, flat, name="c2").options["spine_ledge_rule"] == "tangent"
    assert row_campaign({"notes": {"source_id": "exp6/cap_0001"}}) == "exp6"
    assert row_campaign({"notes": {"collection": "exp7"}}) == "exp7"
    assert row_campaign({"source_campaign": float("nan"), "source_id": None}) is None


def test_tangent_geometry_differs_from_the_default(design):
    from qiskit_metal import designs

    build_component_from_design(design, hf_row("exp6"), name="a")
    other = designs.DesignPlanar({}, overwrite_enabled=True)
    build_component_from_design(other, hf_row("q3d_cap"), name="a")
    a, b = design.qgeometry.tables["poly"], other.qgeometry.tables["poly"]
    assert len(a) == len(b) and not all(x.equals(y) for x, y in zip(a.geometry, b.geometry))
