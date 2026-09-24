"""Statistics identity, reference and gap-report contracts; no source data I/O."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from xdiyo_analytics.data import StatisticsValidationReport, validate_statistics


EVENT = 2**53 + 1
HOME = 2**53 + 3
AWAY = 2**53 + 5
GRAIN = ["event_id", "period", "group_name", "key", "side"]


@pytest.fixture
def matches():
    # The float column must not coerce large IDs when reference rows are read.
    return pa.table({
        "event_id": [EVENT, 7], "home_id": [HOME, 42], "away_id": [AWAY, 50],
        "kickoff_utc": [1735752600.0, None],
    }).to_pandas(types_mapper=pd.ArrowDtype)


@pytest.fixture
def statistics():
    return pa.table({
        "event_id": [EVENT, EVENT, 7],
        "period": ["ALL", "ALL", "1ST"],
        "group_name": ["Shots", "Shots", "Match overview"],
        "key": ["totalShotsOnGoal", "totalShotsOnGoal", "ballPossession"],
        "side": ["home", "away", "home"],
        "team_id": pa.array([HOME, AWAY, None], type=pa.int64()),
        "value": pa.array([0.0, None, float("nan")], type=pa.float64(), from_pandas=False),
        "display": ["0", "7/10 (70%)", "unknown"],
    }).to_pandas(types_mapper=pd.ArrowDtype)


def _set_object(frame, column, position, value):
    values = frame[column].tolist()
    values[position] = value
    frame[column] = pd.Series(values, index=frame.index, dtype=object)


def test_exact_ids_arrow_null_and_valid_nan_report_without_mutation(statistics, matches):
    statistics.attrs["source"] = {"version": "test", "table": "statistics"}
    matches.attrs["source"] = {"version": "test", "table": "matches"}
    before_stats, before_matches = statistics.copy(deep=True), matches.copy(deep=True)
    before_attrs = deepcopy((statistics.attrs, matches.attrs))
    # Arrow distinguishes a null from a valid NaN; the report includes both.
    assert statistics["value"].array.__arrow_array__().null_count == 1

    report = validate_statistics(statistics, matches)

    assert report == StatisticsValidationReport(
        rows=3, matches_referenced=2, team_id_column_present=True,
        missing_team_id_row_positions=(2,), missing_value_row_positions=(1, 2),
        match_sides_without_statistics=((7, "away"),),
    )
    assert statistics["event_id"].iloc[0] == EVENT
    assert statistics["team_id"].iloc[0] == HOME
    pd.testing.assert_frame_equal(statistics, before_stats)
    pd.testing.assert_frame_equal(matches, before_matches)
    assert (statistics.attrs, matches.attrs) == before_attrs
    with pytest.raises(FrozenInstanceError):
        report.rows = 0


def test_adjacent_and_unsigned_large_ids_stay_distinct():
    events = [2**53, 2**53 + 1, 2**63 + 9]
    home_ids = [2**63 + 21, 2**63 + 22, 2**63 + 23]
    away_ids = [2**63 + 31, 2**63 + 32, 2**63 + 33]
    refs = pa.table({
        "event_id": pa.array(events, type=pa.uint64()),
        "home_id": pa.array(home_ids, type=pa.uint64()),
        "away_id": pa.array(away_ids, type=pa.uint64()),
        "extra_float": [0.5, 0.5, 0.5],
    }).to_pandas(types_mapper=pd.ArrowDtype)
    stats = pa.table({
        "event_id": pa.array(events[:2], type=pa.uint64()),
        "period": ["ALL"] * 2, "group_name": ["Shots"] * 2,
        "key": ["totalShotsOnGoal"] * 2, "side": ["home", "away"],
        "team_id": pa.array([home_ids[0], away_ids[1]], type=pa.uint64()),
        "value": [1.5, 2.5],
    }).to_pandas(types_mapper=pd.ArrowDtype)
    report = validate_statistics(stats, refs)
    assert report.matches_referenced == 2
    assert report.match_sides_without_statistics == (
        (events[0], "away"), (events[1], "home"),
        (events[2], "home"), (events[2], "away"),
    )


@pytest.mark.parametrize("argument", ["statistics", "matches"])
@pytest.mark.parametrize("bad", [None, [], {"event_id": [1]}])
def test_wrong_argument_types_fail_explicitly(statistics, matches, argument, bad):
    args = {"statistics": statistics, "matches": matches}
    args[argument] = bad
    with pytest.raises(TypeError, match=f"{argument} must be a pandas DataFrame"):
        validate_statistics(**args)


@pytest.mark.parametrize("column", GRAIN + ["value"])
def test_each_required_statistics_column_is_checked(statistics, matches, column):
    with pytest.raises(ValueError, match=f"statistics is missing required columns: {column}"):
        validate_statistics(statistics.drop(columns=column), matches)


@pytest.mark.parametrize("column", ["event_id", "home_id", "away_id"])
def test_each_required_match_reference_column_is_checked(statistics, matches, column):
    with pytest.raises(ValueError, match=f"matches is missing required columns: {column}"):
        validate_statistics(statistics, matches.drop(columns=column))


@pytest.mark.parametrize("argument", ["statistics", "matches"])
@pytest.mark.parametrize("column", ["event_id", "unused"])
def test_repeated_required_or_optional_columns_are_rejected(statistics, matches, argument, column):
    args = {"statistics": statistics, "matches": matches}
    frame = args[argument].assign(unused="context")
    args[argument] = pd.concat([frame, frame[[column]]], axis=1)
    with pytest.raises(ValueError, match=f"{argument} has repeated column labels"):
        validate_statistics(**args)


@pytest.mark.parametrize("argument,column", [
    ("statistics", "event_id"), ("matches", "event_id"),
    ("matches", "home_id"), ("matches", "away_id"),
])
@pytest.mark.parametrize("bad", [
    None, pd.NA, True, np.bool_(True), 0, -1, 1.0, 1.25, "7",
    float("nan"), float("inf"), Decimal("7"), [7],
])
def test_identifiers_must_be_exact_positive_integers(statistics, matches, argument, column, bad):
    args = {"statistics": statistics, "matches": matches}
    _set_object(args[argument], column, 1, bad)
    with pytest.raises(ValueError, match=rf"{argument}\.{column}.*row positions \[1\]"):
        validate_statistics(**args)


def test_numpy_integral_objects_are_valid_identifiers(statistics, matches):
    for frame, columns in ((statistics, ["event_id"]), (matches, ["event_id", "home_id", "away_id"])):
        for column in columns:
            frame[column] = pd.Series([np.int64(x) for x in frame[column]], dtype=object)
    assert validate_statistics(statistics, matches).matches_referenced == 2


@pytest.mark.parametrize("column", ["period", "group_name", "key"])
@pytest.mark.parametrize("bad", [None, pd.NA, "", " \t\n", 1, True, ["Shots"]])
def test_statistic_labels_must_be_nonempty_strings(statistics, matches, column, bad):
    _set_object(statistics, column, 1, bad)
    with pytest.raises(ValueError, match=rf"statistics\.{column} requires nonempty strings.*\[1\]"):
        validate_statistics(statistics, matches)


@pytest.mark.parametrize("bad", [None, pd.NA, "", "HOME", "Away", " home", "away ", "neutral", 0, True])
def test_only_exact_home_and_away_sides_are_valid(statistics, matches, bad):
    _set_object(statistics, "side", 1, bad)
    with pytest.raises(ValueError, match=r"statistics.side must be home or away; row positions \[1\]"):
        validate_statistics(statistics, matches)


def test_repeated_keys_in_different_groups_periods_sides_and_matches_are_valid(matches):
    stats = pd.DataFrame([
        (EVENT, "ALL", "Shots", "totalShotsOnGoal", "home", 1),
        (EVENT, "ALL", "Match overview", "totalShotsOnGoal", "home", 1),
        (EVENT, "1ST", "Shots", "totalShotsOnGoal", "home", 1),
        (EVENT, "ALL", "Shots", "totalShotsOnGoal", "away", 1),
        (7, "ALL", "Shots", "totalShotsOnGoal", "home", 1),
        (EVENT, "ALL", "Shots", "TotalShotsOnGoal", "home", 1),
        (EVENT, " ALL ", " Shots ", " totalShotsOnGoal ", "home", 1),
    ], columns=GRAIN + ["value"])
    before = stats.copy(deep=True)
    report = validate_statistics(stats, matches)
    assert (report.rows, report.matches_referenced) == (7, 2)
    pd.testing.assert_frame_equal(stats, before)


def test_duplicate_full_grain_fails_even_with_different_values_or_indices(statistics, matches):
    duplicate = statistics.iloc[[0]].copy()
    duplicate["value"] = 999.0
    duplicate["display"] = "different context"
    stats = pd.concat([statistics, duplicate])
    stats.index = ["first", "second", "third", "duplicate"]
    before = stats.copy(deep=True)
    with pytest.raises(ValueError, match=r"Repeated statistics entries.*row positions \[0, 3\]"):
        validate_statistics(stats, matches)
    pd.testing.assert_frame_equal(stats, before)


@pytest.mark.parametrize("orphan", [8, EVENT - 1])
def test_unknown_matches_are_rejected_including_adjacent_large_ids(statistics, matches, orphan):
    _set_object(statistics, "event_id", 2, orphan)
    with pytest.raises(ValueError, match=r"unknown match IDs at row positions \[2\]"):
        validate_statistics(statistics, matches)


def test_duplicate_reference_events_are_rejected_even_if_not_referenced(statistics, matches):
    refs = pd.concat([matches, matches.iloc[[1]]])
    with pytest.raises(ValueError, match="matches.event_id must be unique"):
        validate_statistics(statistics.iloc[:1], refs)


def test_same_home_and_away_team_rejected_even_if_match_not_referenced(statistics, matches):
    _set_object(matches, "away_id", 1, 42)
    with pytest.raises(ValueError, match="Match home_id and away_id must differ"):
        validate_statistics(statistics.iloc[:1], matches)


def test_reference_validation_is_identity_only(statistics, matches):
    refs = matches.assign(competition_id="unchecked", season_id=-1, kickoff_utc="unknown")
    assert validate_statistics(statistics, refs).matches_referenced == 2


@pytest.mark.parametrize("row,wrong_team,side", [(0, AWAY, "home"), (1, HOME, "away"), (2, HOME, "home")])
def test_supplied_team_must_agree_with_referenced_match_side(statistics, matches, row, wrong_team, side):
    _set_object(statistics, "team_id", row, wrong_team)
    with pytest.raises(ValueError, match=f"does not match the match's {side} team at row position {row}"):
        validate_statistics(statistics, matches)


@pytest.mark.parametrize("bad", [True, np.bool_(True), 0, -1, float(HOME), "42", float("inf"), [42], Decimal("42")])
def test_supplied_nonmissing_team_id_must_be_an_integer(statistics, matches, bad):
    _set_object(statistics, "team_id", 2, bad)
    with pytest.raises(ValueError, match="team_id must be a positive integer when supplied; row position 2"):
        validate_statistics(statistics, matches)


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan"), pd.NaT])
def test_missing_team_ids_are_reported_without_becoming_reference_teams(statistics, matches, missing):
    _set_object(statistics, "team_id", 0, missing)
    before = statistics.copy(deep=True)
    report = validate_statistics(statistics, matches)
    assert report.team_id_column_present is True
    assert report.missing_team_id_row_positions == (0, 2)
    pd.testing.assert_frame_equal(statistics, before)


def test_absent_team_id_column_reports_every_row_without_adding_a_column(statistics, matches):
    stats = statistics.drop(columns="team_id")
    report = validate_statistics(stats, matches)
    assert report.team_id_column_present is False
    assert report.missing_team_id_row_positions == (0, 1, 2)
    assert "team_id" not in stats


def test_scalar_missing_values_and_textual_display_are_separate(matches):
    values = [None, pd.NA, float("nan"), pd.NaT, 0, False, -3, float("inf"), "7/10", [None], {"x": None}]
    stats = pd.DataFrame({
        "event_id": [EVENT] * len(values), "period": "ALL", "group_name": "Synthetic",
        "key": [f"measure{i}" for i in range(len(values))], "side": "home",
        "value": pd.Series(values, dtype=object), "display": "70%",
    })
    before = stats.copy(deep=True)
    report = validate_statistics(stats, matches)
    assert report.missing_value_row_positions == (0, 1, 2, 3)
    assert report.rows == len(values)
    pd.testing.assert_frame_equal(stats, before)


@pytest.mark.parametrize("with_team_column", [False, True])
def test_empty_statistics_report_all_reference_sides(statistics, matches, with_team_column):
    stats = statistics.iloc[:0]
    if not with_team_column:
        stats = stats.drop(columns="team_id")
    report = validate_statistics(stats, matches.iloc[::-1])
    assert report == StatisticsValidationReport(
        rows=0, matches_referenced=0, team_id_column_present=with_team_column,
        missing_team_id_row_positions=(), missing_value_row_positions=(),
        match_sides_without_statistics=((7, "home"), (7, "away"), (EVENT, "home"), (EVENT, "away")),
    )


def test_both_typed_tables_can_be_empty(statistics, matches):
    report = validate_statistics(statistics.iloc[:0], matches.iloc[:0])
    assert (report.rows, report.matches_referenced, report.match_sides_without_statistics) == (0, 0, ())


def test_empty_tables_still_require_schema_and_valid_reference_identities(statistics, matches):
    with pytest.raises(ValueError, match="statistics is missing required columns"):
        validate_statistics(pd.DataFrame(), matches)
    _set_object(matches, "home_id", 1, None)
    with pytest.raises(ValueError, match="matches.home_id.*invalid positive integer IDs"):
        validate_statistics(statistics.iloc[:0], matches)


def test_nonempty_statistics_with_empty_references_are_orphans(statistics, matches):
    with pytest.raises(ValueError, match=r"unknown match IDs at row positions \[0, 1, 2\]"):
        validate_statistics(statistics, matches.iloc[:0])


def test_side_coverage_is_any_row_not_per_measure_period_or_value(statistics, matches):
    extra = statistics.iloc[[2]].copy()
    extra["side"] = "away"
    extra["key"] = "unrelatedMeasure"
    stats = pd.concat([statistics, extra], ignore_index=True)
    report = validate_statistics(stats, matches)
    assert report.match_sides_without_statistics == ()
    assert report.missing_value_row_positions == (1, 2, 3)


def test_gap_positions_ignore_arbitrary_repeated_multiindex_labels(statistics, matches):
    stats = statistics.iloc[[2, 0, 1]].copy()
    stats.index = pd.MultiIndex.from_tuples([("same", 9), ("same", 9), ("last", -10)], names=["label", "order"])
    refs = matches.iloc[::-1].copy()
    refs.index = pd.Index(["ref", "ref"], name="source_label")
    before_stats, before_refs = stats.copy(deep=True), refs.copy(deep=True)
    report = validate_statistics(stats, refs)
    assert report.missing_team_id_row_positions == (0,)
    assert report.missing_value_row_positions == (0, 2)
    assert report.match_sides_without_statistics == ((7, "away"),)
    pd.testing.assert_frame_equal(stats, before_stats)
    pd.testing.assert_frame_equal(refs, before_refs)
