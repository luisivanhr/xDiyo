"""Retained histories keep the metadata required to calculate their features."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.data import SeasonData
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment
from xdiyo_analytics.experiments.recovery import dump_bundle, load_bundle, pack, unpack
from xdiyo_analytics.features import Lag, RollingMean, Stat, evaluate_features
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.selection import Candidate
from xdiyo_analytics.splits import TemporalSplit, create_split_plan


class MeanModel:
    def fit(self, context):
        self.mean = context.y.mean()

    def predict(self, context):
        return {"predict": pd.DataFrame({name: value for name, value in self.mean.items()},
                                         index=context.X.index)}


def test_loaded_experiment_history_can_evaluate_the_same_features(tmp_path):
    matches = pd.DataFrame({
        "event_id": range(1, 7), "competition_id": 17, "season_id": 100,
        "home_id": 11, "away_id": 12, "status": "finished",
        "kickoff_utc": [date.timestamp() for date in pd.date_range("2025-01-01", periods=6, tz="UTC")],
        "home_score_current": 2, "away_score_current": 1,
    })
    statistics = pd.DataFrame([
        {"event_id": event, "period": "ALL", "group_name": "Match overview",
         "key": "cornerKicks", "side": side, "team_id": team, "value": float(event + team)}
        for event in range(1, 7) for side, team in [("home", 11), ("away", 12)]
    ])
    provenance = {"sources": [{"league": "Synthetic", "tables": ("matches", "statistics")}],
                  "checked_at": pd.Timestamp("2025-01-07", tz="UTC")}
    history = build_team_history(SeasonData({"matches": matches, "statistics": statistics}, provenance))
    stat = Stat("ALL", "Match overview", "cornerKicks")
    definitions = {"lag": Lag(stat), "mean": RollingMean(stat, 3)}
    features = evaluate_features(history, definitions, keyed=True)
    labels = create_labels(history, {"corners": MatchTotal(stat)})
    dataset = assemble_dataset(features, labels["corners"], layout="match")
    split = create_split_plan(dataset, TemporalSplit(train_size=4, test_size=1, unit="kickoffs"))
    prepared = PreparedExperiment(dataset, split, outputs={"history": history, "features": features})
    original_attrs = deepcopy(history.attrs)
    experiment = FootballExperiment("Reusable retained history", output_dir=tmp_path)
    result = experiment.run(prepared, model=Candidate("Mean", MeanModel))
    loaded = experiment.load(result.record["run_id"])
    restored = loaded.prepared.outputs["history"]
    pd.testing.assert_frame_equal(restored, history)
    assert restored.attrs == original_attrs
    assert restored.attrs["source"] == provenance
    assert loaded.prepared.outputs["features"].attrs == features.attrs
    actual = evaluate_features(restored, definitions, keyed=True)
    pd.testing.assert_frame_equal(actual, features)
    assert actual.attrs == features.attrs
    assert history.attrs == original_attrs
    restored.attrs["source"]["sources"][0]["league"] = "Changed after load"
    assert history.attrs == original_attrs


@pytest.mark.parametrize("kind", ["frame", "empty_frame", "series"])
def test_nested_data_only_attrs_roundtrip_without_column_copies(tmp_path, kind):
    value = (pd.Series([1, 2], name="value") if kind == "series" else
             pd.DataFrame(index=pd.Index([3, 4], name="row")) if kind == "empty_frame" else
             pd.DataFrame({"one": [1, 2], "two": [3, 4]}))
    nested = pd.DataFrame({"label": ["home", "away"]})
    nested.attrs = {"origin": {"version": 2}}
    details = pd.Series([True, False], name="eligible")
    details.attrs = {"rule": ["finished", "known before cutoff"]}
    value.attrs = {"nested": {("ALL", "corners"): [np.array([1, 3], dtype="int64"), nested, details]},
                   "timestamp": pd.Timestamp("2025-01-01", tz="UTC"), "missing": pd.NA}
    encoded = pack(value)
    if kind != "series":
        assert all("attrs" not in column for column in encoded["series"])
    path = tmp_path / "attrs.json"
    dump_bundle(path, value)
    restored = load_bundle(path)
    assert json.loads(path.read_text())["schema"] == 1
    if kind == "series":
        pd.testing.assert_series_equal(restored, value)
    else:
        pd.testing.assert_frame_equal(restored, value)
    attrs = restored.attrs
    assert attrs["timestamp"] == value.attrs["timestamp"]
    assert attrs["missing"] is pd.NA
    array, frame, series = attrs["nested"][("ALL", "corners")]
    np.testing.assert_array_equal(array, np.array([1, 3], dtype="int64"))
    pd.testing.assert_frame_equal(frame, nested)
    pd.testing.assert_series_equal(series, details)
    assert frame.attrs == nested.attrs
    assert series.attrs == details.attrs


@pytest.mark.parametrize("kind", ["frame", "series"])
def test_legacy_frames_and_series_without_attrs_remain_readable(kind):
    value = pd.Series([1, 2], name="value") if kind == "series" else pd.DataFrame({"value": [1, 2]})
    encoded = pack(value)
    encoded.pop("attrs", None)
    if kind == "frame":
        for column in encoded["series"]:
            column.pop("attrs", None)
    restored = unpack(json.loads(json.dumps(encoded)))
    assert restored.attrs == {}
    if kind == "series":
        pd.testing.assert_series_equal(restored, value)
    else:
        pd.testing.assert_frame_equal(restored, value)


@pytest.mark.parametrize("kind", ["frame", "series"])
def test_unsupported_attrs_fail_before_any_pickle_protocol(kind):
    reductions = []
    class UnsafeMetadata:
        def __reduce__(self):
            reductions.append(True)
            raise AssertionError("Attrs must never use pickle or its copy protocol")
    value = pd.Series([1], name="value") if kind == "series" else pd.DataFrame({"value": [1]})
    value.attrs = {"nested": {"callback": UnsafeMetadata()}}
    with pytest.raises(TypeError, match="data-only"):
        pack(value)
    assert reductions == []
