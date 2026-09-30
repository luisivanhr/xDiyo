"""Unsupported NumPy precision and masks fail explicitly before publication."""
import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from test_post_training_experiments import computed_run
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import execution_key, pack, signature, unpack
from xdiyo_analytics.experiments.store import _json


def value(kind):
    if kind == "longdouble": return np.longdouble("1.25")
    if kind == "clongdouble": return np.clongdouble(1 + 2j)
    if kind == "longdouble_array": return np.array([1, 2], dtype=np.longdouble)
    if kind == "empty_longdouble": return np.empty(0, dtype=np.longdouble)
    if kind == "scalar_longdouble": return np.array(1, dtype=np.longdouble)
    if kind == "empty_clongdouble": return np.empty((2, 0), dtype=np.clongdouble)
    if kind == "structured_empty": return np.empty(0, dtype=[("nested", [("wide", np.longdouble, (2,))])])
    if kind == "masked_integer": return np.ma.array([1, 2], mask=[False, True])
    if kind == "masked_float": return np.ma.array([1., 2.], mask=[True, False])
    if kind == "nomask": return np.ma.array([1, 2], mask=np.ma.nomask)
    if kind == "masked_scalar": return np.ma.array(1, mask=True)
    if kind == "empty_masked": return np.ma.array([], dtype=int)
    return np.ma.masked


KINDS = ["longdouble", "clongdouble", "longdouble_array", "empty_longdouble", "scalar_longdouble",
         "empty_clongdouble", "structured_empty", "masked_integer", "masked_float", "nomask",
         "masked_scalar", "empty_masked", "masked_constant"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("serializer", [pack, _json, signature])
def test_unsupported_numpy_values_have_explicit_conversion_errors(kind, serializer):
    expected = value(kind)
    with pytest.raises(TypeError, match="explicitly.*convert|explicit.*conversion"):
        serializer({"nested": [expected]})
    if isinstance(expected, np.ma.MaskedArray):
        assert isinstance(expected, np.ma.MaskedArray)
        if kind == "masked_integer":
            np.testing.assert_array_equal(expected.data, [1, 2])
            np.testing.assert_array_equal(expected.mask, [False, True])


@pytest.mark.parametrize("kind", ["longdouble", "masked_integer", "masked_constant"])
@pytest.mark.parametrize("container", ["object_array", "frame_cell", "frame_attrs"])
def test_nested_retained_values_reject_without_recursion_or_mask_loss(kind, container):
    unsupported = value(kind)
    if container == "object_array":
        retained = np.empty(1, dtype=object)
        retained[0] = unsupported
    elif container == "frame_cell":
        cells = np.empty(1, dtype=object)
        cells[0] = unsupported
        retained = pd.DataFrame({"cell": cells})
    else:
        retained = pd.DataFrame({"count": [1]})
        retained.attrs = {"nested": {"unsupported": unsupported}}
    with pytest.raises(TypeError, match="longdouble|MaskedArray"):
        pack(retained)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("axis", ["column", "index", "categories"])
def test_extended_precision_pandas_dtypes_reject_even_when_empty(empty, axis):
    values = np.array([] if empty else [1, 2], dtype=np.longdouble)
    if axis == "column":
        retained = pd.DataFrame({"value": values})
    elif axis == "index":
        retained = pd.DataFrame({"value": np.arange(len(values))}, index=pd.Index(values))
    else:
        retained = pd.Series(pd.Categorical([], categories=pd.Index(values)))
    with pytest.raises(TypeError, match="longdouble"):
        pack(retained)


@pytest.mark.parametrize("kind", ["longdouble", "empty_longdouble", "nomask", "masked_integer", "masked_constant"])
@pytest.mark.parametrize("location", ["outputs", "config"])
def test_football_experiment_cannot_publish_unsupported_numpy(tmp_path, kind, location):
    preparation = prepared(holdout=True)
    if location == "outputs":
        preparation.outputs["unsupported"] = value(kind)
    else:
        preparation.config["unsupported"] = value(kind)
    experiment = FootballExperiment("Unsupported NumPy", output_dir=tmp_path)
    with pytest.raises(TypeError, match="longdouble|MaskedArray"):
        experiment.run(preparation, model=ridge())
    assert not any(record["status"] == "complete" for record in experiment.store.read_runs())


@pytest.mark.parametrize("artifact,kind", [
    (artifact, kind)
    for artifact in ["config", "report_cell", "report_attrs", "history_attrs"]
    for kind in ["longdouble", "masked_integer"]
] + [("history_dtype", "longdouble"), ("report_empty_dtype", "longdouble")])
def test_numerical_store_rejects_unsupported_numpy_before_publication(tmp_path, artifact, kind):
    training, report = computed_run()
    config = {}
    unsupported = value(kind)
    if artifact == "config":
        config = {"unsupported": unsupported}
    else:
        table = pd.DataFrame({"value": [1]})
        if artifact.endswith("dtype"):
            table = pd.DataFrame({"value": np.empty(0, dtype=np.longdouble) if "empty" in artifact else np.array([1], dtype=np.longdouble)})
        elif artifact.endswith("cell"):
            cells = np.empty(1, dtype=object)
            cells[0] = unsupported
            table = pd.DataFrame({"value": cells})
        else:
            table.attrs = {"unsupported": unsupported}
        if artifact.startswith("history"):
            training.folds[0].training_history = table
        else:
            report.studies[0].result.tables["unsupported"] = table
    experiment = FootballExperiment("Numerical unsupported NumPy", output_dir=tmp_path)
    with pytest.raises(TypeError, match="longdouble|MaskedArray"):
        experiment.store.save_run(training, report, name="Unsupported", config=config)
    assert experiment.store.read_runs() == []


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ordinary_float_scalars_and_arrays_remain_supported(dtype):
    scalar = dtype(1.25)
    assert unpack(pack(scalar)) == 1.25 and _json(scalar) == 1.25
    for shape in [(), (0,), (2, 3)]:
        array = np.full(shape, 1.25, dtype=dtype)
        actual = unpack(pack(array))
        assert actual.dtype == array.dtype and actual.shape == shape
        np.testing.assert_array_equal(actual, array)
        assert _json(array) == array.tolist()


def test_explicit_mask_resolution_and_precision_conversion_remain_opt_in():
    masked = np.ma.array([1, 2], mask=[False, True])
    converted = masked.filled(-1)
    np.testing.assert_array_equal(unpack(pack(converted)), [1, -1])
    converted_precision = np.asarray(np.array([1.25], dtype=np.longdouble), dtype=np.float64)
    np.testing.assert_array_equal(unpack(pack(converted_precision)), [1.25])


@pytest.mark.parametrize("kind", ["longdouble", "masked_integer", "masked_constant"])
@pytest.mark.parametrize("shape", [(), (0,), (2,)])
def test_preserved_array_dtype_metadata_rejects_unsupported_values(kind, shape):
    dtype = np.dtype("f8", metadata={"nested": {"value": value(kind)}})
    array = np.full(shape, 1.25, dtype=dtype)
    with pytest.raises(TypeError, match="longdouble|MaskedArray"):
        pack(array)
    # Configuration arrays have always used values only; this boundary is unchanged.
    assert _json(array) == array.tolist()


@pytest.mark.parametrize("kind", ["longdouble", "empty_longdouble", "masked_constant", "nomask"])
def test_execution_identity_retains_explicit_conversion_guidance(kind):
    with pytest.raises(TypeError, match="explicitly.*convert"):
        execution_key({"unsupported": value(kind)})


@pytest.mark.parametrize("container", [pd.Series, pd.CategoricalIndex])
def test_categorical_precision_is_rejected_before_value_conversion(monkeypatch, container):
    categories = pd.Index(np.array([1, 2], dtype=np.longdouble))
    retained = container(pd.Categorical([], categories=categories))

    def converted_too_early(self):
        raise AssertionError("unsupported category values were converted before dtype validation")

    monkeypatch.setattr(container, "tolist", converted_too_early)
    with pytest.raises(TypeError, match="longdouble"):
        pack(retained)


def test_rejected_category_leaves_ordinary_experiment_publication_intact(tmp_path):
    categories = pd.Index(np.array([1, 2], dtype=np.longdouble))
    with pytest.raises(TypeError, match="longdouble"):
        pack(pd.Series(pd.Categorical([], categories=categories)))
    experiment = FootballExperiment("Ordinary floats after rejection", output_dir=tmp_path)
    result = experiment.run(prepared(holdout=True), model=ridge())
    restored = experiment.load(result.record["run_id"])
    expected = result.training.folds[0].predictions["predict"]
    assert all(dtype.type is np.float64 for dtype in expected.dtypes)
    pd.testing.assert_frame_equal(restored.training.folds[0].predictions["predict"], expected)
