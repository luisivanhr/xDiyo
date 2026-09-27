"""Named timestamp zones survive data-only recovery and calendar arithmetic."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dateutil import tz
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.experiments.recovery import dump_bundle, load_bundle, pack, unpack


def zone(name, provider):
    return ZoneInfo(name) if provider == "zoneinfo" else tz.gettz(name)


def assert_same_calendar(actual, expected):
    expected = pd.Timestamp(expected)
    assert isinstance(actual, pd.Timestamp)
    assert actual == expected
    assert actual.isoformat() == expected.isoformat()
    assert actual.fold == expected.fold
    assert actual.nanosecond == expected.nanosecond
    for offset in (pd.DateOffset(months=6), pd.DateOffset(days=180)):
        shifted = actual + offset
        reference = expected + offset
        assert shifted == reference
        assert shifted.isoformat() == reference.isoformat()
        assert shifted.utcoffset() == reference.utcoffset()


@pytest.mark.parametrize("provider", ["zoneinfo", "dateutil"])
@pytest.mark.parametrize("name", ["America/New_York", "Europe/Berlin", "Asia/Tokyo"])
@pytest.mark.parametrize("kind", ["timestamp", "datetime"])
def test_named_zone_retains_winter_to_summer_calendar_arithmetic(tmp_path, provider, name, kind):
    expected = pd.Timestamp("2024-01-15 12:34:56.123456789", tz=zone(name, provider))
    if kind == "datetime":
        expected = expected.to_pydatetime(warn=False)
    path = tmp_path / "timestamp.json"
    dump_bundle(path, expected)
    actual = load_bundle(path)
    assert_same_calendar(actual, expected)
    assert str(actual.tzinfo) == str(expected.tzinfo)


@pytest.mark.parametrize("provider", ["zoneinfo", "dateutil"])
@pytest.mark.parametrize("fold", [0, 1])
@pytest.mark.parametrize("kind", ["timestamp", "datetime"])
def test_ambiguous_wall_time_retains_fold_and_instant(provider, fold, kind):
    expected = datetime(2024, 11, 3, 1, 30, tzinfo=zone("America/New_York", provider), fold=fold)
    if kind == "timestamp":
        expected = pd.Timestamp(expected) + pd.Timedelta(123, "ns")
    actual = unpack(pack(expected))
    assert_same_calendar(actual, expected)
    assert actual.fold == fold
    assert str(actual.tzinfo) == str(expected.tzinfo)


@pytest.mark.parametrize("value", [datetime(2024, 1, 2, 3, 4),
    datetime(2024, 1, 2, 3, 4, tzinfo=timezone.utc),
    datetime(2024, 1, 2, 3, 4, tzinfo=timezone(timedelta(hours=5, minutes=30))),
    pd.Timestamp("2024-01-02 03:04:05.123456789"),
    pd.Timestamp("2024-01-02 03:04:05.123456789+05:30")])
def test_naive_and_fixed_offset_encoding_remains_unchanged(value):
    assert pack(value) == {"@": "timestamp", "value": value.isoformat()}
    assert_same_calendar(unpack(pack(value)), value)


@pytest.mark.parametrize("literal", ["2024-02-29", "2024-01-15T12:00:00-05:00", "2024-11-03T01:30:00-04:00"])
def test_legacy_timestamp_without_timezone_metadata_remains_readable(literal):
    actual = unpack({"@": "timestamp", "value": literal})
    expected = pd.Timestamp(literal)
    assert_same_calendar(actual, expected)
    assert type(actual.tzinfo) is type(expected.tzinfo)


def timestamp_payload():
    winter = pd.Timestamp("2024-01-15 12:34:56.123456789", tz=ZoneInfo("America/New_York"))
    ambiguous = pd.Timestamp(datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1))
    frame = pd.DataFrame({"objects": pd.Series([winter, ambiguous, None], dtype=object)})
    frame.attrs = {"clock": {"winter": winter, "ambiguous": ambiguous.to_pydatetime()}}
    return frame


def assert_timestamp_payload(actual, expected):
    pd.testing.assert_frame_equal(actual, expected)
    for index in [0, 1]:
        assert_same_calendar(actual.iloc[index, 0], expected.iloc[index, 0])
        assert str(actual.iloc[index, 0].tzinfo) == "America/New_York"
    for key in expected.attrs["clock"]:
        assert_same_calendar(actual.attrs["clock"][key], expected.attrs["clock"][key])


def test_object_cells_and_nested_attrs_retain_named_zones(tmp_path):
    expected = timestamp_payload()
    path = tmp_path / "frame.json"
    dump_bundle(path, expected)
    assert_timestamp_payload(load_bundle(path), expected)


def test_named_timestamp_outputs_survive_experiment_run_and_load(tmp_path):
    preparation = prepared(holdout=True)
    expected = timestamp_payload()
    preparation.outputs["timestamps"] = expected
    experiment = FootballExperiment("Timestamp zone fidelity", output_dir=tmp_path)
    result = experiment.run(preparation, model=ridge())
    restored = experiment.load(result.record["run_id"])
    assert_timestamp_payload(restored.prepared.outputs["timestamps"], expected)
