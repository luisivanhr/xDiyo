"""Numerical report and history tables retain their own data-only attributes."""
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from test_post_training_experiments import computed_run
from xdiyo_analytics.experiments import FootballExperiment


def attributes(kind):
    if kind == "empty":
        return {}
    if kind == "simple":
        return {"source": "retained matches"}
    if kind == "null":
        return {"optional": None}
    return {"context": {"window": timedelta(days=2, microseconds=3), "origin": date(2024, 1, 1),
                         "labels": ("home", "away"), 7: pd.Timedelta(9, "ns")},
            "weights": np.array([.25, .75], dtype=np.dtype("f8", metadata={"unit": "share"})),
            "cutoff": pd.Timestamp("2024-01-15 12:00", tz="America/New_York")}


def assert_attrs(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_attrs(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_attrs(left, right)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.dtype.metadata == expected.dtype.metadata
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected
        if isinstance(expected, pd.Timestamp):
            assert str(actual.tzinfo) == str(expected.tzinfo)
            assert actual + pd.DateOffset(months=6) == expected + pd.DateOffset(months=6)


def table(kind, empty_rows):
    result = pd.DataFrame({"step": [1, 2], "loss": [.25, .75]})
    if empty_rows:
        result = result.iloc[:0]
    result.attrs = attributes(kind)
    return result


@pytest.mark.parametrize("kind", ["empty", "simple", "null", "nested"])
@pytest.mark.parametrize("empty_rows", [False, True])
def test_table_attributes_survive_numerical_store_and_experiment_load(tmp_path, kind, empty_rows):
    training, report = computed_run()
    expected = table(kind, empty_rows)
    training.folds[0].training_history = expected
    report.studies[0].result.tables["attributes"] = expected
    experiment = FootballExperiment("Numerical table attributes", output_dir=tmp_path)
    record = experiment.store.save_run(training, report, name="Saved attributes", config={})
    assert "recovery" not in record["artifacts"]
    folder = experiment.store.path / "runs" / record["run_id"]
    assert not (folder / "recovery.json").exists()
    tables = json.loads((folder / "tables.json").read_text(encoding="utf-8"))
    document = next(item["table"] for item in tables if item["name"] == "attributes")
    history = json.loads((folder / "training.json").read_text(encoding="utf-8"))[0]["history"]
    for serialized in (document, history):
        if kind == "empty":
            # This is the historical pandas table format, still read directly.
            assert "encoding" not in serialized and "schema" in serialized and "data" in serialized
        else:
            assert serialized["encoding"] == "xdiyo.data-only.v1"
    loaded = experiment.store.load_run(record["run_id"])
    facade = experiment.load(record["run_id"])
    for actual in (loaded["report"].studies[0].result.tables["attributes"],
                   loaded["training"].folds[0].training_history,
                   facade.post_report.studies[0].result.tables["attributes"], facade.training.folds[0].training_history):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert_attrs(actual.attrs, expected.attrs)
    assert_attrs(expected.attrs, attributes(kind))


class OpaqueMetadata:
    def __deepcopy__(self, memo):
        raise AssertionError("Unsupported attrs must be rejected before pandas copies them.")


@pytest.mark.parametrize("artifact", ["report", "history"])
@pytest.mark.parametrize("opaque", [object, OpaqueMetadata])
def test_unsupported_table_attributes_fail_before_publication(tmp_path, artifact, opaque):
    training, report = computed_run()
    unsupported = table("empty", False)
    unsupported.attrs = {"opaque": opaque()}
    if artifact == "report":
        report.studies[0].result.tables["unsupported"] = unsupported
    else:
        training.folds[0].training_history = unsupported
    experiment = FootballExperiment("Unsupported table attributes", output_dir=tmp_path)
    with pytest.raises(TypeError, match="Recovery cannot encode (object|OpaqueMetadata)"):
        experiment.store.save_run(training, report, name="Unsupported", config={})
    assert experiment.store.read_runs() == []
    assert not list(experiment.store.path.glob("runs/*/run.json"))


@pytest.mark.parametrize("kind", ["complex", "sparse"])
def test_attributes_do_not_bypass_unsupported_table_dtype_rejection(tmp_path, kind):
    training, report = computed_run()
    values = pd.Series([1 + 2j]) if kind == "complex" else pd.Series(pd.arrays.SparseArray([1., 0.]))
    unsupported = pd.DataFrame({"value": values})
    unsupported.attrs = {"source": "nonempty"}
    report.studies[0].result.tables["unsupported"] = unsupported
    experiment = FootballExperiment("Unsupported table dtype", output_dir=tmp_path)
    with pytest.raises(TypeError, match=kind):
        experiment.store.save_run(training, report, name="Unsupported", config={})
    assert experiment.store.read_runs() == []
