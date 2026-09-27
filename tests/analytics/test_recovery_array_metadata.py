"""Plain array dtype metadata survives recovery without erasing typed cells."""
from datetime import timedelta
import copy

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import dump_bundle, execution_key, load_bundle, pack, signature, unpack


def metadata():
    return {"unit": "meters", "scale": (1, 100),
            "source": {"site": "history", "sides": ["home", "away"]},
            "calendar": np.datetime64("2024-01-01", "D"),
            "offset": timedelta(days=-2, microseconds=3),
            "calibration": np.array([1, 2], dtype="i2"), "labels": {(1, 2): "bin"}}


def assert_nested(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_nested(actual[key], expected[key])
    elif isinstance(expected, np.ndarray):
        assert_array(actual, expected)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested(left, right)
    else:
        if isinstance(expected, np.generic):
            assert actual.dtype == expected.dtype
        assert actual == expected


def assert_array(actual, expected):
    assert isinstance(actual, np.ndarray) and actual.shape == expected.shape
    assert actual.dtype.str == expected.dtype.str
    if expected.dtype.metadata is None:
        assert actual.dtype.metadata is None
    else:
        assert_nested(dict(actual.dtype.metadata), dict(expected.dtype.metadata))
    if expected.dtype.kind == "O":
        for index in np.ndindex(expected.shape):
            assert_nested(actual[index], expected[index])
    elif expected.dtype.kind in "mM":
        np.testing.assert_array_equal(actual.astype("i8"), expected.astype("i8"))
    else:
        np.testing.assert_array_equal(actual, expected)


def sample(dtype, shape):
    dtype = np.dtype(dtype, metadata=metadata())
    result = np.empty(shape, dtype=dtype)
    if dtype.kind == "O":
        values = [("a", 1), [1, {"enabled": True}], np.array([4, 5], dtype="i2"), {"array": np.array([8])}]
        for index, position in enumerate(np.ndindex(shape)):
            result[position] = values[index % len(values)]
    elif dtype.kind in "mM":
        ticks = np.arange(result.size, dtype="i8").reshape(shape)
        if ticks.size:
            ticks.flat[-1] = np.iinfo("i8").min
        result[...] = ticks.astype(dtype)
    elif dtype.kind == "U":
        result[...] = "goal"
    elif dtype.kind == "b":
        result[...] = True
    else:
        result[...] = np.arange(result.size).reshape(shape) + 1
    return result


@pytest.mark.parametrize("dtype", ["f8", ">i8", "bool", "U5", "O", ">M8[3ns]", ">m8[2D]"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3), (2, 0, 3)])
def test_plain_array_dtype_metadata_preserves_cells_shape_and_temporal_units(tmp_path, dtype, shape):
    expected = sample(dtype, shape)
    path = tmp_path / "metadata.json"
    dump_bundle(path, expected)
    assert_array(load_bundle(path), expected)
    assert isinstance(pack(expected)["dtype"], dict)


def test_empty_metadata_is_distinct_from_absent_metadata():
    expected = np.empty((0, 2), dtype=np.dtype("f4", metadata={}))
    actual = unpack(pack(expected))
    assert actual.dtype.metadata is not None and dict(actual.dtype.metadata) == {}
    assert_array(actual, expected)


def test_noncontiguous_object_cells_keep_metadata_and_sequence_boundaries():
    expected = sample("O", (3, 4))[::-1, ::2]
    assert not expected.flags.c_contiguous
    assert_array(unpack(pack(expected)), expected)


@pytest.mark.parametrize("dtype", ["f8", "O", ">M8[3ns]", ">m8[2D]", "V4", "c16"])
def test_metadata_free_empty_arrays_keep_legacy_dtype_strings(dtype):
    expected = np.empty(0, dtype=dtype)
    encoded = pack(expected)
    assert encoded["dtype"] == str(expected.dtype)
    actual = unpack(encoded)
    assert_array(actual, expected)


def test_archived_object_and_temporal_dtype_strings_remain_readable():
    archived = {"@": "array", "dtype": "object", "shape": [1],
                "values": [{"@": "tuple", "values": [1, 2]}]}
    actual = unpack(archived)
    assert actual.dtype.metadata is None and type(actual[0]) is tuple and actual[0] == (1, 2)
    temporal = {"@": "array", "dtype": ">M8[3ns]", "shape": [2], "values": [7, -9223372036854775808]}
    np.testing.assert_array_equal(unpack(temporal).astype("i8"), [7, -9223372036854775808])


@pytest.mark.parametrize("with_metadata", [False, True])
def test_nonempty_complex_arrays_keep_their_existing_unsupported_value_boundary(with_metadata):
    dtype = np.dtype("c16", metadata={"unit": "complex"}) if with_metadata else np.dtype("c16")
    with pytest.raises(TypeError, match="complex"):
        pack(np.array([1 + 2j], dtype=dtype))


def test_array_metadata_changes_identity_but_mapping_order_does_not():
    expected = sample("f8", (2,))
    changed_metadata = copy.deepcopy(dict(expected.dtype.metadata))
    changed_metadata["unit"] = "feet"
    changed = np.array(expected.tolist(), dtype=np.dtype("f8", metadata=changed_metadata))
    reordered = np.array(expected.tolist(), dtype=np.dtype("f8", metadata=dict(reversed(list(expected.dtype.metadata.items())))))
    assert changed.dtype.metadata["unit"] == "feet"
    assert signature(expected) != signature(changed)
    assert execution_key(expected) != execution_key(changed)
    assert signature(expected) == signature(reordered)
    assert execution_key(expected) == execution_key(reordered)


def test_plain_array_metadata_in_outputs_and_frame_attrs_survives_experiment_load(tmp_path):
    values = {name: sample(dtype, shape) for name, dtype, shape in
              [("numeric", "f8", (2,)), ("objects", "O", (2, 3)), ("time", ">M8[3ns]", ()),
               ("duration", ">m8[2D]", (0,))]}
    frame = pd.DataFrame({"score": [1, 2]})
    frame.attrs = values
    preparation = prepared(holdout=True)
    preparation.outputs["arrays"] = values
    preparation.outputs["frame"] = frame
    experiment = FootballExperiment("Array dtype metadata", output_dir=tmp_path)
    result = experiment.run(preparation, model=ridge())
    restored = experiment.load(result.record["run_id"])
    for name, expected in values.items():
        assert_array(restored.prepared.outputs["arrays"][name], expected)
        assert_array(restored.prepared.outputs["frame"].attrs[name], expected)
