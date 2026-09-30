"""Interval endpoints, closure and pandas containers survive recovery."""
from dataclasses import dataclass
from functools import partial
import json

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared
from model_selection_samples import FixedAdapter
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import dump_bundle, load_bundle, pack, unpack
from xdiyo_analytics.reporting import PredictionReporter, StudyResult
from xdiyo_analytics.selection import Candidate


def breaks(kind):
    if kind == "integer":
        return pd.Index([2**53 + 1, 2**53 + 3, 2**53 + 5], dtype="int64")
    if kind == "float":
        return pd.Index([-np.inf, .125, np.inf])
    if kind == "duration":
        return pd.timedelta_range("1ns", periods=3, freq="7ns")
    if kind == "timestamp":
        return pd.date_range("2024-01-15 12:34:56.123456789", periods=3, freq="2D")
    return pd.date_range("2024-01-15 12:34:56.123456789", periods=3, freq="6MS", tz="America/New_York")


@pytest.mark.parametrize("kind", ["integer", "float", "duration", "timestamp", "zone"])
@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_interval_scalar_retains_endpoints_and_closure(kind, closed):
    endpoints = breaks(kind)
    expected = pd.Interval(endpoints[0], endpoints[1], closed=closed)
    actual = unpack(json.loads(json.dumps(pack(expected), allow_nan=False)))
    assert type(actual) is pd.Interval
    assert actual == expected and actual.closed == closed
    assert actual.left == expected.left and actual.right == expected.right
    if kind == "zone":
        assert str(actual.left.tzinfo) == "America/New_York"
        assert actual.left + pd.DateOffset(months=6) == expected.left + pd.DateOffset(months=6)


@pytest.mark.parametrize("kind", ["integer", "float", "duration", "timestamp", "zone"])
@pytest.mark.parametrize("shape", ["full", "empty", "missing"])
def test_interval_index_retains_dtype_name_nullmask_and_endpoint_metadata(tmp_path, kind, shape):
    expected = pd.IntervalIndex.from_breaks(breaks(kind), closed="neither", name=("bins", 1))
    if shape == "empty":
        expected = expected[:0]
    elif shape == "missing":
        expected = expected.insert(1, np.nan)
    path = tmp_path / "intervals.json"
    dump_bundle(path, expected)
    actual = load_bundle(path)
    pd.testing.assert_index_equal(actual, expected, exact=True)
    pd.testing.assert_index_equal(actual.left, expected.left, exact=True)
    pd.testing.assert_index_equal(actual.right, expected.right, exact=True)
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    for side in ("left", "right"):
        if hasattr(getattr(expected, side), "freq"):
            assert getattr(actual, side).freq == getattr(expected, side).freq


def interval_table(empty=False):
    bins = pd.IntervalIndex.from_breaks([0., 2., 4., 6.], closed="right", name="bins")
    table = pd.DataFrame({"cut": pd.cut([1., np.nan, 5.], bins),
                          "intervals": pd.Series([bins[0], np.nan, bins[2]], dtype=bins.dtype),
                          "objects": pd.Series([bins[0], pd.Interval(1, 2, closed="left"), None], dtype=object)},
                         index=bins)
    # Construct by position so the interval row labels do not align away values.
    table["intervals"] = pd.array([bins[0], np.nan, bins[2]], dtype=bins.dtype)
    table["objects"] = np.asarray([bins[0], pd.Interval(1, 2, closed="left"), None], dtype=object)
    table.attrs = {"boundary": bins[0], "bins": bins}
    return table.iloc[:0] if empty else table


def assert_table(actual, expected):
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs["boundary"] == expected.attrs["boundary"]
    pd.testing.assert_index_equal(actual.attrs["bins"], expected.attrs["bins"])
    assert [type(value) for value in actual["objects"]] == [type(value) for value in expected["objects"]]


@pytest.mark.parametrize("empty", [False, True])
def test_interval_frame_series_categories_columns_levels_and_attrs(tmp_path, empty):
    frame = interval_table(empty)
    columns = pd.DataFrame([[1., 2., 3.]], columns=interval_table().index)
    index = pd.MultiIndex(levels=[interval_table().index, ["a", "b"]], codes=[[0, 2, -1], [0, 1, 0]], names=["bin", "side"])
    payload = {"frame": frame, "columns": columns, "level": index, "series": frame["cut"]}
    path = tmp_path / "containers.json"
    dump_bundle(path, payload)
    actual = load_bundle(path)
    assert_table(actual["frame"], frame)
    pd.testing.assert_frame_equal(actual["columns"], columns)
    pd.testing.assert_index_equal(actual["level"], index)
    pd.testing.assert_index_equal(actual["level"].levels[0], index.levels[0])
    pd.testing.assert_series_equal(actual["series"], payload["series"])
    pd.testing.assert_index_equal(actual["series"].cat.categories, payload["series"].cat.categories)


@dataclass(kw_only=True)
class IntervalReporter(PredictionReporter):
    empty: bool = False

    def run(self, context):
        return StudyResult("Intervals", tables={"intervals": interval_table(self.empty)})


class IntervalHistoryAdapter(FixedAdapter):
    def __init__(self, empty):
        super().__init__()
        self.empty = empty

    def fit(self, context):
        super().fit(context)
        self.training_history_ = interval_table(self.empty)
        return self


@pytest.mark.parametrize("empty", [False, True])
def test_interval_reports_history_and_outputs_survive_both_saved_loaders(tmp_path, empty):
    preparation = prepared(holdout=True)
    expected = interval_table(empty)
    preparation.outputs["intervals"] = expected
    experiment = FootballExperiment("Interval artifacts", output_dir=tmp_path)
    candidate = Candidate("Interval history", partial(IntervalHistoryAdapter, empty), config={"empty": empty})
    post = PostTrainingAnalysis({"Intervals": IntervalReporter(type="overall", partition="score", empty=empty)})
    result = experiment.run(preparation, model=candidate, post_analysis=post)
    numerical_record = experiment.store.save_run(result.training, result.post_report, name="Numerical intervals", config={})
    modern = experiment.load(result.record["run_id"])
    numerical = experiment.store.load_run(numerical_record["run_id"])
    assert_table(modern.prepared.outputs["intervals"], expected)
    for actual in (modern.post_report.studies[0].result.tables["intervals"], modern.training.folds[0].training_history,
                   numerical["report"].studies[0].result.tables["intervals"], numerical["training"].folds[0].training_history):
        assert_table(actual, expected)
