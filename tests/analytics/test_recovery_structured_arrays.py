"""Structured NumPy values retain typed fields without serializing raw padding."""
import json

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import dump_bundle, load_bundle, pack, unpack


def layout(kind):
    if kind == "ordinary":
        return np.dtype([("id", ">i8"), ("value", "f8"), ("label", "U5")])
    if kind == "nested":
        return np.dtype([("record", [("code", ">u2"), ("when", "M8[3ns]")]),
                         ("vector", "i2", (2, 3)), ("objects", object, (2,))])
    if kind == "object_temporal":
        return np.dtype([("payload", object), ("when", "M8[ns]"), ("elapsed", "m8[2D]")])
    if kind == "aligned":
        subtype = np.dtype("i4", metadata={"unit": "count"})
        return np.dtype([(("Readable count", "count"), subtype), ("weight", "f8"),
                         ("nested", [("value", "i2"), ("matrix", "f4", (2,))])],
                        align=True, metadata={"source": "fixture", "revision": 2})
    if kind == "zero_subarray":
        return np.dtype({"names": ["empty", "value"], "formats": [("i4", (0,)), "i8"],
                         "offsets": [4, 0], "itemsize": 8})
    if kind == "out_of_order":
        return np.dtype({"names": ["later", "first"], "formats": ["i2", "i8"],
                         "offsets": [24, 0], "itemsize": 32})
    return np.dtype({"names": [], "formats": [], "offsets": [], "itemsize": 8})


def fill(array):
    if array.dtype.fields is not None:
        for name in array.dtype.names:
            fill(array[name])
    elif array.dtype.kind == "O":
        cells = [("tuple", 1), [1, {"nested": True}], {"array": np.array([1, 2]), "date": pd.Timestamp("2024-01-01")}]
        for position, index in enumerate(np.ndindex(array.shape)):
            array[index] = cells[position % len(cells)]
    elif array.dtype.kind in "mM":
        counts = np.arange(array.size, dtype="int64").reshape(array.shape)
        if counts.size:
            counts.flat[-1] = np.iinfo("int64").min
        array[...] = counts.astype(array.dtype)
    elif array.dtype.kind in "iu":
        array[...] = np.arange(array.size).reshape(array.shape) + (2**53 + 1 if array.dtype.itemsize == 8 else 1)
    elif array.dtype.kind == "f":
        array[...] = np.arange(array.size).reshape(array.shape) + .125
    elif array.dtype.kind == "U":
        array[...] = "score"
    return array


def sample(kind, shape):
    return fill(np.zeros(shape, dtype=layout(kind)))


def assert_dtype(actual, expected):
    assert actual == expected
    assert actual.itemsize == expected.itemsize and actual.isalignedstruct == expected.isalignedstruct
    assert actual.metadata == expected.metadata
    if expected.fields is not None:
        assert actual.names == expected.names
        for name in expected.names:
            left, right = actual.fields[name], expected.fields[name]
            assert left[1:] == right[1:]
            assert_dtype(left[0], right[0])
    elif expected.subdtype is not None:
        assert actual.subdtype[1] == expected.subdtype[1]
        assert_dtype(actual.subdtype[0], expected.subdtype[0])


def assert_array(actual, expected):
    assert isinstance(actual, np.ndarray) and actual.shape == expected.shape
    assert_dtype(actual.dtype, expected.dtype)
    # The typed payload checks object cell boundaries as well as temporal units,
    # names, titles and nested arrays; ndarray equality cannot compare those cells.
    assert pack(actual) == pack(expected)
    if expected.dtype.fields is not None:
        for name in expected.dtype.names:
            assert_array(actual[name], expected[name])
    elif expected.dtype.kind == "O":
        for index in np.ndindex(expected.shape):
            assert_cell(actual[index], expected[index])
    else:
        np.testing.assert_array_equal(actual, expected)


def assert_cell(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert_array(actual, expected)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_cell(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_cell(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", ["ordinary", "nested", "object_temporal", "aligned", "out_of_order", "zero_fields", "zero_subarray"])
@pytest.mark.parametrize("shape", [(), (0,), (2,), (2, 2)])
def test_structured_arrays_preserve_layout_shape_and_field_values(tmp_path, kind, shape):
    expected = sample(kind, shape)
    path = tmp_path / "records.json"
    dump_bundle(path, expected)
    assert_array(load_bundle(path), expected)
    assert "structured_array" in path.read_text(encoding="utf-8")


def test_noncontiguous_view_and_structured_scalar_do_not_flatten_records(tmp_path):
    source = sample("nested", (3, 4))
    expected = source[::-1, ::2]
    path = tmp_path / "view.json"
    dump_bundle(path, {"view": expected, "record": expected[0, 0]})
    actual = load_bundle(path)
    assert_array(actual["view"], expected)
    assert isinstance(actual["record"], np.void)
    assert_array(np.asarray(actual["record"]), np.asarray(expected[0, 0]))


def test_old_simple_array_payload_remains_readable():
    archived = {"@": "array", "values": [[1, 2], [3, 4]], "dtype": "int16", "shape": [2, 2]}
    np.testing.assert_array_equal(unpack(archived), np.array([[1, 2], [3, 4]], dtype="int16"))


@pytest.mark.parametrize("problem", ["overlap", "opaque"])
@pytest.mark.parametrize("empty", [False, True])
def test_unsupported_structured_layout_fails_before_completed_publication(tmp_path, problem, empty):
    dtype = (np.dtype({"names": ["a", "b"], "formats": ["i4", "i2"], "offsets": [0, 0], "itemsize": 8})
             if problem == "overlap" else np.dtype([("opaque", "V8")]))
    array = np.zeros(0 if empty else 2, dtype=dtype)
    preparation = prepared(holdout=True)
    preparation.outputs["records"] = array
    experiment = FootballExperiment("Unsupported records", output_dir=tmp_path)
    with pytest.raises(TypeError, match="overlapping|opaque void"):
        experiment.run(preparation, model=ridge())
    assert all(path.parent.name.startswith(".pending-") for path in tmp_path.rglob("recovery.json"))
    for path in tmp_path.rglob("run.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["status"] != "complete"


@pytest.mark.parametrize("empty", [False, True])
def test_structured_arrays_in_outputs_and_frame_attrs_survive_saved_experiment(tmp_path, empty):
    expected = sample("nested", (0,) if empty else (2,))
    frame = pd.DataFrame({"value": [1, 2]})
    frame.attrs = {"records": expected, "scalar": sample("aligned", ())[()]}
    preparation = prepared(holdout=True)
    preparation.outputs["records"] = expected
    preparation.outputs["frame"] = frame
    experiment = FootballExperiment("Structured output", output_dir=tmp_path)
    result = experiment.run(preparation, model=ridge())
    restored = experiment.load(result.record["run_id"])
    assert_array(restored.prepared.outputs["records"], expected)
    actual = restored.prepared.outputs["frame"]
    pd.testing.assert_frame_equal(actual, frame)
    assert_array(actual.attrs["records"], expected)
    assert_array(np.asarray(actual.attrs["scalar"]), np.asarray(frame.attrs["scalar"]))


def test_structured_field_shape_mismatch_is_rejected_without_broadcasting():
    encoded = pack(sample("ordinary", (2,)))
    encoded["fields"][0][1] = pack(np.array(1, dtype=">i8"))
    with pytest.raises(ValueError, match="dtype and shape"):
        unpack(encoded)
