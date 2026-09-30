"""Fixed-width byte cells retain raw contents without serializing record padding."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import dump_bundle, load_bundle, pack, unpack


def byte_array(width, shape):
    dtype = np.dtype(f"S{width}", metadata={"unit": "opaque-code"})
    size = int(np.prod(shape, dtype=int)) if shape else 1
    if width == 0:
        return np.ndarray(shape, dtype=dtype)
    pattern = bytes([255, 0, 97, 0, 128, 254, 0])
    data = (pattern * (size * width // len(pattern) + 1))[:size * width]
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)


def assert_byte_array(actual, expected):
    assert type(actual) is np.ndarray
    assert actual.dtype == expected.dtype and actual.dtype.metadata == expected.dtype.metadata
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")
    assert actual.flags.writeable


@pytest.mark.parametrize("width", [0, 1, 4, 17])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3), (2, 0, 3)])
def test_byte_arrays_preserve_width_shape_and_exact_contents(tmp_path, width, shape):
    expected = byte_array(width, shape)
    path = tmp_path / "bytes.json"
    dump_bundle(path, expected)
    assert_byte_array(load_bundle(path), expected)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["payload"]["@"] == "byte_array"


def test_all_byte_values_and_noncontiguous_views_are_lossless():
    expected = np.frombuffer(bytes(range(256)), dtype="S4").reshape(8, 8)[::-1, ::2]
    assert not expected.flags.c_contiguous
    assert_byte_array(unpack(pack(expected)), expected)
    all_values = np.frombuffer(bytes(range(256)), dtype="S256").reshape(())
    assert_byte_array(unpack(pack(all_values)), all_values)


@pytest.mark.parametrize("data", [b"", b"\x00", b"a\x00b\x00\x00", b"\xff\xfe\x80\x00"])
@pytest.mark.parametrize("numpy_scalar", [False, True])
def test_python_and_numpy_byte_scalars_keep_distinct_types_and_nuls(data, numpy_scalar):
    expected = np.bytes_(data) if numpy_scalar else data
    actual = unpack(json.loads(json.dumps(pack(expected))))
    assert type(actual) is type(expected)
    if numpy_scalar:
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    else:
        assert actual == data


def structured_bytes(shape):
    nested = np.dtype([("tag", "S3"), ("tokens", "S2", (2,))])
    dtype = np.dtype({"names": ["code", "nested"], "formats": ["S4", nested],
                     "offsets": [0, 8], "itemsize": 24})
    array = np.empty(shape, dtype=dtype)
    # Poison record padding; only named fields may enter the recovery payload.
    array.reshape(-1).view("u1")[...] = 165
    array["code"] = byte_array(4, shape)
    array["nested"]["tag"] = byte_array(3, shape)
    array["nested"]["tokens"] = byte_array(2, (*shape, 2))
    return array


def assert_structured(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    for left, right in ((actual["code"], expected["code"]),
                        (actual["nested"]["tag"], expected["nested"]["tag"]),
                        (actual["nested"]["tokens"], expected["nested"]["tokens"])):
        assert_byte_array(left, right)


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_nested_structured_fields_and_subarrays_exclude_record_padding(shape):
    expected = structured_bytes(shape)
    encoded = pack(expected)
    actual = unpack(encoded)
    assert_structured(actual, expected)
    raw = json.dumps(encoded)
    assert "a5" not in raw
    if expected.size:
        assert actual.tobytes() != expected.tobytes()  # Restored padding is zeroed.


def test_byte_object_fields_preserve_python_numpy_and_array_values():
    expected = np.empty(3, dtype=[("object", object), ("code", "S4")])
    expected["object"] = [b"a\x00\x00", np.bytes_(b"a\x00\x00"), byte_array(4, ())]
    expected["code"] = byte_array(4, (3,))
    actual = unpack(pack(expected))
    assert actual.dtype == expected.dtype
    assert type(actual["object"][0]) is bytes and actual["object"][0] == expected["object"][0]
    assert type(actual["object"][1]) is np.bytes_
    assert actual["object"][1].dtype == expected["object"][1].dtype
    assert actual["object"][1].tobytes() == expected["object"][1].tobytes()
    assert_byte_array(actual["object"][2], expected["object"][2])
    assert_byte_array(actual["code"], expected["code"])


@pytest.mark.parametrize("empty", [False, True])
def test_byte_records_outputs_and_frame_attrs_publish_and_reload(tmp_path, empty):
    expected = structured_bytes((0,) if empty else (2,))
    frame = pd.DataFrame({"code": pd.Series([b"a\x00", np.bytes_(b"b\x00")], dtype=object)})
    frame.attrs = {"records": expected, "scalar": np.bytes_(b"\xff\x00\x00")}
    preparation = prepared(holdout=True)
    preparation.outputs["records"] = expected
    preparation.outputs["frame"] = frame
    experiment = FootballExperiment("Byte records", output_dir=tmp_path)
    result = experiment.run(preparation, model=ridge())
    restored = experiment.load(result.record["run_id"])
    assert_structured(restored.prepared.outputs["records"], expected)
    actual = restored.prepared.outputs["frame"]
    pd.testing.assert_frame_equal(actual, frame)
    assert_structured(actual.attrs["records"], expected)
    assert actual.attrs["scalar"].dtype == frame.attrs["scalar"].dtype
    assert actual.attrs["scalar"].tobytes() == frame.attrs["scalar"].tobytes()
    assert [type(cell) for cell in actual["code"]] == [bytes, np.bytes_]


@pytest.mark.parametrize("fault", ["short", "long", "dtype", "shape", "negative", "boolean"])
def test_byte_payload_rejects_inconsistent_descriptor(fault):
    encoded = copy.deepcopy(pack(byte_array(4, (2,))))
    if fault == "short": encoded["value"] = encoded["value"][:-2]
    elif fault == "long": encoded["value"] += "00"
    elif fault == "dtype": encoded["dtype"] = {"kind": "plain", "value": "i4"}
    elif fault == "shape": encoded["shape"] = [3]
    elif fault == "negative": encoded["shape"] = [-1]
    else: encoded["shape"] = [True]
    with pytest.raises(ValueError, match="Byte array"):
        unpack(encoded)


def test_legacy_simple_array_payloads_keep_their_decoder():
    np.testing.assert_array_equal(unpack({"@": "array", "values": [1, 2], "dtype": "int16", "shape": [2]}),
                                  np.array([1, 2], dtype="int16"))
    archived = {"@": "array", "values": [], "dtype": "|S4", "shape": [0]}
    assert_byte_array(unpack(archived), np.empty(0, dtype="S4"))


@pytest.mark.parametrize("base", [bytes, np.bytes_])
def test_byte_scalar_subclasses_cannot_silently_drop_custom_state(base):
    class LabeledBytes(base):
        pass

    value = LabeledBytes(b"code\x00")
    value.label = "must not disappear"
    with pytest.raises(TypeError, match="subclasses"):
        pack(value)
    cells = np.empty(1, dtype=object)
    cells[0] = value
    with pytest.raises(TypeError, match="subclasses"):
        pack(cells)
