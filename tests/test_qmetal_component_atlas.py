import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_qmetal_component_atlas.py"
SPEC = spec_from_file_location("build_qmetal_component_atlas", SCRIPT)
atlas = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = atlas
SPEC.loader.exec_module(atlas)


def test_inventory_is_frozen_at_38_production_components():
    assert len(atlas.COMPONENT_SPECS) == 38
    assert len({spec.import_path for spec in atlas.COMPONENT_SPECS}) == 38
    assert sum(spec.recipe == "contextual_route" for spec in atlas.COMPONENT_SPECS) == 6
    assert {spec.class_name for spec in atlas.COMPONENT_SPECS if spec.category == "routes"} == {
        "RouteAnchors",
        "RouteFramed",
        "RouteMeander",
        "RouteMixed",
        "RoutePathfinder",
        "RouteStraight",
    }


def test_pin_assignment_collapses_multiple_pins_and_preserves_unmarked_island():
    islands = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
    ]
    component = SimpleNamespace(
        pins={
            "left_a": {"points": np.asarray([[0.0, 0.4], [0.0, 0.6]])},
            "left_b": {"points": np.asarray([[1.0, 0.4], [1.0, 0.6]])},
        }
    )

    terminals = atlas.assign_terminals(component, islands)

    assert len(terminals) == 2
    assert terminals[0]["pin_names"] == ["left_a", "left_b"]
    assert terminals[0]["assignment_method"] == "pin_to_island"
    assert terminals[1]["pin_names"] == []
    assert terminals[1]["assignment_method"] == "unmarked_conductor_island"
    assert [terminal["marker_layer"] for terminal in terminals] == [2, 3]


def test_geometry_fingerprint_ignores_translation_but_not_shape():
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    moved = Polygon([(7, -3), (8, -3), (8, -2), (7, -2)])
    rectangle = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    empty = Polygon()

    assert atlas.geometry_fingerprint(square, empty) == atlas.geometry_fingerprint(moved, empty)
    assert atlas.geometry_fingerprint(square, empty) != atlas.geometry_fingerprint(rectangle, empty)


def test_route_recipe_is_explicit_and_uses_only_named_fixture_endpoints():
    options = atlas._route_options("RouteStraight")
    assert options["pin_inputs"] == {
        "start_pin": {"component": "atlas_start", "pin": "tie"},
        "end_pin": {"component": "atlas_end", "pin": "tie"},
    }
    assert "anchors" not in options
    assert list(atlas._route_options("RouteMixed")["anchors"]) == [0, 1]


def test_semantic_only_terminations_are_frozen_in_the_inventory():
    names = {spec.class_name for spec in atlas.COMPONENT_SPECS}
    assert {"OpenToGround", "ShortToGround"} <= names
