import pandas as pd
import pytest

from squadds.core.analysis import Analyzer
from squadds.core.analysis_search import (
    SUPPORTED_METRICS,
    filter_df_by_target_params,
    get_H_param_keys_for_system,
    outside_bounds,
    rank_closest_indices,
    remove_resonator_type_from_target_params,
    resolve_metric_strategy,
)
from squadds.core.metrics import (
    ChebyshevMetric,
    CustomMetric,
    EuclideanMetric,
    ManhattanMetric,
    WeightedEuclideanMetric,
)


@pytest.mark.parametrize(
    ("selected_system", "expected"),
    [
        ("qubit", ["qubit_frequency_GHz", "anharmonicity_MHz"]),
        ("cavity_claw", ["resonator_type", "cavity_frequency_GHz", "kappa_kHz"]),
        (
            ["qubit", "cavity_claw"],
            [
                "qubit_frequency_GHz",
                "anharmonicity_MHz",
                "resonator_type",
                "cavity_frequency_GHz",
                "kappa_kHz",
                "g_MHz",
            ],
        ),
        ("coupler", None),
    ],
)
def test_get_H_param_keys_for_system_matches_legacy_mappings(selected_system, expected):
    assert get_H_param_keys_for_system(selected_system) == expected


def test_get_H_param_keys_for_system_rejects_unknown_system():
    with pytest.raises(ValueError, match="Invalid system."):
        get_H_param_keys_for_system("resonator")


def test_remove_resonator_type_from_target_params_mutates_half_wave_dict():
    target_params = {"resonator_type": "half", "cavity_frequency_GHz": 7.0}

    result = remove_resonator_type_from_target_params(target_params, "half", missing_ok=True)

    assert result is target_params
    assert target_params == {"cavity_frequency_GHz": 7.0}


def test_remove_resonator_type_from_target_params_keeps_quarter_wave_dict():
    target_params = {"resonator_type": "quarter", "cavity_frequency_GHz": 7.0}

    result = remove_resonator_type_from_target_params(target_params, "quarter", missing_ok=True)

    assert result is target_params
    assert target_params == {"resonator_type": "quarter", "cavity_frequency_GHz": 7.0}


def test_remove_resonator_type_from_target_params_raises_when_required_key_missing():
    with pytest.raises(KeyError):
        remove_resonator_type_from_target_params({}, "half", missing_ok=False)


def test_outside_bounds_flags_numeric_values_outside_range():
    df = pd.DataFrame({"g_MHz": [10.0, 20.0], "resonator_type": ["quarter", "quarter"]})

    assert outside_bounds(df, {"g_MHz": 25.0}, display=False)


def test_outside_bounds_flags_missing_categorical_match():
    df = pd.DataFrame({"g_MHz": [10.0, 20.0], "resonator_type": ["quarter", "quarter"]})

    assert outside_bounds(df, {"resonator_type": "half"}, display=False)


def test_outside_bounds_rejects_unknown_columns():
    df = pd.DataFrame({"g_MHz": [10.0, 20.0]})

    with pytest.raises(ValueError, match="not a column in dataframe"):
        outside_bounds(df, {"kappa_kHz": 100.0}, display=False)


@pytest.mark.parametrize(
    ("metric_name", "expected_type"),
    [
        ("Euclidean", EuclideanMetric),
        ("Manhattan", ManhattanMetric),
        ("Chebyshev", ChebyshevMetric),
        ("Weighted Euclidean", WeightedEuclideanMetric),
        ("Custom", CustomMetric),
    ],
)
def test_resolve_metric_strategy_returns_expected_type(metric_name, expected_type):
    custom_metric_func = (lambda target, row: 0.0) if metric_name == "Custom" else None

    strategy = resolve_metric_strategy(metric_name, metric_weights={"g_MHz": 2}, custom_metric_func=custom_metric_func)

    assert isinstance(strategy, expected_type)


def test_supported_metrics_stays_in_sync_with_public_analyzer_contract():
    assert SUPPORTED_METRICS == ["Euclidean", "Manhattan", "Chebyshev", "Weighted Euclidean", "Custom"]


def test_filter_df_by_target_params_keeps_only_matching_categorical_rows():
    df = pd.DataFrame(
        {
            "resonator_type": ["quarter", "half", "quarter"],
            "g_MHz": [60.0, 70.0, 80.0],
        },
        index=[10, 20, 30],
    )

    filtered = filter_df_by_target_params(df, {"resonator_type": "quarter", "g_MHz": 60.0})

    assert filtered.index.tolist() == [10, 30]


def test_rank_closest_indices_orders_by_metric_distance():
    df = pd.DataFrame(
        {
            "qubit_frequency_GHz": [4.9, 5.02, 5.2],
            "anharmonicity_MHz": [-180.0, -199.0, -250.0],
        },
        index=[11, 22, 33],
    )
    target_params = {"qubit_frequency_GHz": 5.0, "anharmonicity_MHz": -200.0}
    strategy = resolve_metric_strategy("Euclidean")

    ranked_indices = rank_closest_indices(df, target_params, strategy, num_top=2)

    assert ranked_indices.tolist() == [22, 11]


def test_find_closest_recomputes_when_target_columns_are_missing(monkeypatch):
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.__supported_metrics__ = Analyzer.__supported_metrics__
    analyzer.selected_resonator_type = "quarter"
    analyzer.selected_system = "cavity_claw"
    analyzer.params_computed = True
    analyzer.metric_weights = None
    analyzer.custom_metric_func = None
    analyzer.closest_design_found = False
    analyzer.df = pd.DataFrame({"placeholder": [1]})

    def fake_add_target_params_columns():
        analyzer.df = pd.DataFrame(
            {
                "cavity_frequency_GHz": [7.0],
                "kappa_kHz": [120.0],
                "resonator_type": ["quarter"],
                "design_options": [{"name": "design-a"}],
            }
        )

    monkeypatch.setattr(analyzer, "_add_target_params_columns", fake_add_target_params_columns)
    monkeypatch.setattr(analyzer, "_outside_bounds", lambda df, params, display=True: False)
    monkeypatch.setattr(analyzer, "set_metric_strategy", lambda strategy: setattr(analyzer, "metric_strategy", strategy))

    closest = analyzer.find_closest(
        {"cavity_frequency_GHz": 7.0, "kappa_kHz": 120.0, "resonator_type": "quarter"},
        num_top=1,
        display=False,
    )

    assert closest.iloc[0]["design_options"] == {"name": "design-a"}
