"""Period fidelity and mapping-order-independent execution identities."""
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import dump_bundle, execution_key, load_bundle, pack, signature, unpack
from xdiyo_analytics.reporting import PredictionReporter, StudyResult


@pytest.mark.parametrize("freq", ["M", "Q-MAR", "Y-JUN", "2W-SUN", "2D", "3h", "ns"])
def test_period_scalar_preserves_ordinal_and_frequency(freq):
    expected = pd.Period("2024-02-29 12:34:56.123456789", freq=freq)
    actual = unpack(json.loads(json.dumps(pack(expected))))
    assert type(actual) is pd.Period
    assert actual.ordinal == expected.ordinal
    assert type(actual.freq) is type(expected.freq) and actual.freq == expected.freq
    assert actual == expected


@pytest.mark.parametrize("freq", ["M", "2Q-MAR", "2W-SUN", "3h", "ns"])
@pytest.mark.parametrize("ordinals", [[-11, 0, 2**53 + 7, np.iinfo("int64").min], [], [np.iinfo("int64").min] * 2])
def test_period_index_preserves_ordinals_frequency_name_and_nat(tmp_path, freq, ordinals):
    expected = pd.PeriodIndex(pd.arrays.PeriodArray(np.asarray(ordinals, dtype="int64"),
                              dtype=pd.PeriodDtype(freq=freq)), name=("period", 1))
    path = tmp_path / "period-index.json"
    dump_bundle(path, expected)
    actual = load_bundle(path)
    pd.testing.assert_index_equal(actual, expected, exact=True)
    np.testing.assert_array_equal(actual.asi8, expected.asi8)
    assert actual.freq == expected.freq and actual.name == expected.name
    np.testing.assert_array_equal(actual.isna(), expected.isna())


def period_containers():
    periods = pd.period_range("2024-01", periods=4, freq="2M", name="month")
    index = periods.take([0, 2, 3]).insert(1, pd.NaT)
    columns = pd.period_range("2023Q1", periods=2, freq="Q-MAR", name="forecast")
    frame = pd.DataFrame([[1., 2.], [np.nan, 3.], [4., 5.], [6., 7.]], index=index, columns=columns)
    frame.attrs = {"first": periods[0], "calendar": periods}
    series = pd.Series([periods[0], pd.NaT, periods[2]], dtype=periods.dtype, name="period")
    objects = pd.Series([periods[0], columns[0], pd.NaT], dtype=object, name="mixed_periods")
    categories = pd.Series(pd.Categorical([periods[2], None, periods[0]], categories=periods, ordered=True))
    multi = pd.MultiIndex(levels=[periods, ["a", "b"]], codes=[[0, 2, -1], [0, 1, 0]], names=["month", "side"])
    return dict(frame=frame, series=series, objects=objects, categories=categories, multi=multi,
                missing=pd.Period("NaT", freq="M"))


def assert_period_containers(actual, expected):
    pd.testing.assert_frame_equal(actual["frame"], expected["frame"])
    assert actual["frame"].attrs["first"] == expected["frame"].attrs["first"]
    pd.testing.assert_index_equal(actual["frame"].attrs["calendar"], expected["frame"].attrs["calendar"])
    for key in ("series", "objects", "categories"):
        pd.testing.assert_series_equal(actual[key], expected[key])
        assert [type(item) for item in actual[key]] == [type(item) for item in expected[key]]
    pd.testing.assert_index_equal(actual["multi"], expected["multi"])
    pd.testing.assert_index_equal(actual["multi"].levels[0], expected["multi"].levels[0])
    assert actual["missing"] is pd.NaT


@pytest.mark.parametrize("empty", [False, True])
def test_period_frames_series_categories_levels_and_attrs_round_trip(tmp_path, empty):
    expected = period_containers()
    if empty:
        expected["frame"] = expected["frame"].iloc[:0]
        expected["series"] = expected["series"].iloc[:0]
    path = tmp_path / "period-containers.json"
    dump_bundle(path, expected)
    assert_period_containers(load_bundle(path), expected)


def test_legacy_empty_period_index_dtype_string_remains_readable():
    archived = {"@": "index", "values": [], "dtype": "period[Q-MAR]", "name": "fiscal"}
    pd.testing.assert_index_equal(unpack(archived), pd.PeriodIndex([], freq="Q-MAR", name="fiscal"))


@pytest.mark.parametrize("ordinal", [True, 1.5, [1]])
def test_period_scalar_rejects_non_integer_ordinal(ordinal):
    encoded = pack(pd.Period("2024-01", freq="M"))
    encoded["ordinal"] = ordinal
    with pytest.raises(ValueError, match="integer ordinal"):
        unpack(encoded)


def test_period_frequency_cannot_select_arbitrary_constructors():
    encoded = pack(pd.Period("2024-01", freq="M"))
    encoded["freq"]["name"] = "__import__"
    with pytest.raises(ValueError, match="Unsupported recovery frequency"):
        unpack(encoded)


@dataclass(kw_only=True)
class PeriodTableReporter(PredictionReporter):
    def run(self, context):
        table = pd.DataFrame({"count": [1, 2, 3]}, index=pd.period_range("2024-01", periods=3, freq="M", name="month"))
        return StudyResult("Monthly", tables={"monthly": table})


def test_period_report_publishes_and_loads_with_mandatory_recovery(tmp_path):
    preparation = prepared(holdout=True)
    expected = period_containers()
    preparation.outputs["periods"] = expected
    experiment = FootballExperiment("Period reporting", output_dir=tmp_path)
    post = PostTrainingAnalysis({"Monthly": PeriodTableReporter(type="overall", partition="score")})
    result = experiment.run(preparation, model=ridge(), post_analysis=post)
    loaded = experiment.load(result.record["run_id"])
    assert_period_containers(loaded.prepared.outputs["periods"], expected)
    expected_table = result.post_report.studies[0].result.tables["monthly"]
    pd.testing.assert_frame_equal(loaded.post_report.studies[0].result.tables["monthly"], expected_table)
    numerical = experiment.store.save_run(result.training, result.post_report, name="Numerical periods", config={})
    reloaded = experiment.store.load_run(numerical["run_id"])
    pd.testing.assert_frame_equal(reloaded["report"].studies[0].result.tables["monthly"], expected_table)


def reordered(value):
    if isinstance(value, dict):
        return {key: reordered(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [reordered(item) for item in value]
    return value


def configuration():
    return {"model": {"alpha": .5, "options": {"fit_intercept": True, "seed": 7}},
            "features": ["goals", "corners"], "groups": {2: "two", ("league", 1): {"a": 1, "b": 2}, "2": "text"}}


def test_mapping_signature_and_execution_key_ignore_nested_insertion_order():
    original = configuration()
    reverse = reordered(original)
    assert signature(original) == signature(reverse)
    assert execution_key(original) == execution_key(reverse)
    changed = reordered(original)
    changed["model"]["alpha"] = .75
    assert execution_key(original) != execution_key(changed)
    changed = reordered(original)
    changed["features"].reverse()
    assert execution_key(original) != execution_key(changed)


class ConfigurationKey:
    def __init__(self, config):
        self.config = config

    def cache_key(self):
        return self.config


def test_custom_cache_key_mapping_order_is_canonical():
    assert execution_key(ConfigurationKey(configuration())) == execution_key(ConfigurationKey(reordered(configuration())))


def test_equal_encoded_mapping_keys_have_deterministic_value_tie_break():
    first_nan, second_nan = float("nan"), float("nan")
    original = {first_nan: "first", second_nan: "second"}
    reverse = dict(reversed(list(original.items())))
    assert len(original) == 2
    assert signature(original) == signature(reverse)
    assert execution_key(original) == execution_key(reverse)


def test_packed_frame_mappings_are_canonical_without_reordering_saved_data():
    original = pd.DataFrame({"left": [{"a": 1, "b": 2}, {"c": 3, "d": 4}], "right": [1, 2]})
    original.attrs = {"source": "history", "stat_columns": {"goals": "GF", "corners": "CF"}}
    reverse = original.copy(deep=True)
    reverse.attrs = reordered(original.attrs)
    reverse["left"] = reverse["left"].map(reordered)
    assert pack(original) != pack(reverse)
    assert signature(original) == signature(reverse)
    assert execution_key(original) == execution_key(reverse)
    assert execution_key(original) != execution_key(original.iloc[::-1])
    assert execution_key(original) != execution_key(original[["right", "left"]])
    assert list(unpack(pack(reverse)).attrs) == list(reverse.attrs)
    assert list(unpack(pack(reverse)).iloc[0, 0]) == ["b", "a"]
