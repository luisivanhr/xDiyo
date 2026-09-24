"""Season pooling and finer gaps, using distinct native IDs and staggered dates."""
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.splits import TemporalSplit
from split_samples import dataset, whole_matches


def seasons(*, layout="match", independent_rounds=False, postpone=False):
    specs = []
    for league in range(2):
        for year in range(6):
            for round_no in range(1, 13):
                specs.append(dict(competition=17 + league,
                                  season=1000 + 100 * league + year,
                                  round=round_no + (100 * league if independent_rounds else 0),
                                  day=365 * year + 37 * league + round_no * 5))
            if postpone and year == 4:
                specs.append(dict(competition=17 + league, season=1000 + 100 * league + year,
                                  round=1, day=365 * year + 37 * league + 59))
    result = dataset(layout=layout, shuffle=True, specs=specs)
    result.metadata["source_season"] = result.metadata.season_id.map(
        lambda value: f"{20 + value % 100}_{21 + value % 100}")
    return result


@pytest.mark.parametrize("layout", ["match", "team_match"])
@pytest.mark.parametrize("window", ["expanding", "sliding"])
def test_pooled_source_seasons_and_explicit_league_seasons(layout, window):
    data = seasons(layout=layout)
    pooled = TemporalSplit(4, unit="seasons", window=window).folds(data)
    separate = TemporalSplit(4, unit="league_seasons", window=window).folds(data)
    assert len(pooled) == 2
    assert len(separate) == 4
    for index, fold in enumerate(pooled):
        train, test = [data.metadata.iloc[rows] for rows in (fold.train, fold.test)]
        expected_train = {f"{20 + i}_{21 + i}" for i in range(index if window == "sliding" else 0, 4 + index)}
        assert set(train.source_season) == expected_train
        assert set(test.source_season) == {f"{24 + index}_{25 + index}"}
        assert set(test.competition_id) == {17, 18}
        assert test.season_id.nunique() == 2
        whole_matches(data, fold)
    for fold in separate:
        selected = data.metadata.iloc[np.r_[fold.train, fold.test]]
        assert selected.competition_id.nunique() == 1
        assert data.metadata.iloc[fold.test].source_season.nunique() == 1
    explicit = TemporalSplit(4, unit="seasons", calendar_by="competition_id", block_by="season_id", window=window).folds(data)
    assert [(tuple(f.train), tuple(f.test)) for f in explicit] == [(tuple(f.train), tuple(f.test)) for f in separate]


@pytest.mark.parametrize("unit", ["seasons", "league_seasons"])
def test_ten_round_gap_excludes_postponed_early_round_and_keeps_nominal_end(unit):
    data = seasons(layout="team_match", postpone=True)
    folds = TemporalSplit(4, unit=unit, gap=10, gap_unit="rounds", score_rounds=(12, None)).folds(data)
    for fold in folds:
        train, test, score = [data.metadata.iloc[rows] for rows in (fold.train, fold.test, fold.score)]
        assert set(test["round"]) == {11, 12}
        assert set(score["round"]) == {12}
        assert not set(train.source_season) & set(test.source_season)
        assert test.source_season.nunique() == 1
        assert fold.metadata["fit_at"] == test.kickoff_at.min()
        gap_rows = data.metadata.iloc[fold.metadata["gap_rows"]]
        assert set(gap_rows["round"]) == set(range(1, 11))
        assert not set(fold.metadata["gap_rows"]) & set(np.r_[fold.train, fold.test])
        whole_matches(data, fold)


def test_round_gap_counts_each_league_independently_not_global_round_names():
    data = seasons(independent_rounds=True)
    fold = TemporalSplit(4, unit="seasons", gap=10, gap_unit="rounds").folds(data)[0]
    test = data.metadata.iloc[fold.test]
    assert set(test.loc[test.competition_id.eq(17), "round"]) == {11, 12}
    assert set(test.loc[test.competition_id.eq(18), "round"]) == {111, 112}


def test_finer_kickoff_gap_cutoffs_and_availability_use_retained_test_boundary():
    data = seasons()
    cutoffs = data.metadata.kickoff_at - pd.Timedelta("2D")
    available = data.metadata.kickoff_at.copy()
    delayed = data.metadata.season_id.eq(1003) & data.metadata["round"].eq(12)
    available.loc[delayed] = pd.Timestamp("2035-01-01", tz="UTC")
    fold = TemporalSplit(4, unit="seasons", gap=10, gap_unit="kickoffs").folds(data, cutoffs=cutoffs, available_at=available)[0]
    test = data.metadata.iloc[fold.test]
    nominal = data.metadata[data.metadata.source_season.eq("24_25")]
    omitted_times = nominal.kickoff_at.drop_duplicates().sort_values().iloc[:10]
    expected = nominal.loc[~nominal.kickoff_at.isin(omitted_times)].index
    assert set(fold.test) == set(expected)
    assert fold.metadata["fit_at"] == cutoffs.iloc[fold.test].min()
    assert not set(np.flatnonzero(delayed)) & set(fold.train)
    assert set(np.flatnonzero(delayed)) <= set(fold.metadata["excluded_train"])


@pytest.mark.parametrize("unit", ["seasons", "league_seasons", "rounds", "kickoffs"])
def test_omitted_gap_unit_replays_explicit_window_unit(unit):
    data = seasons()
    options = dict(train_size=2, test_size=1, gap=1, unit=unit)
    default = TemporalSplit(**options).folds(data)
    explicit = TemporalSplit(**options, gap_unit=unit).folds(data)
    assert [(tuple(f.train), tuple(f.test), tuple(f.score)) for f in default] == [(tuple(f.train), tuple(f.test), tuple(f.score)) for f in explicit]
    if unit in {"seasons", "league_seasons"}:
        first = default[0]
        assert set(data.metadata.iloc[first.test].source_season) == {"23_24"}
        assert set(data.metadata.iloc[first.train].source_season) == {"20_21", "21_22"}


@pytest.mark.parametrize("gap", [-1, 1.5, True, float("nan"), "10"])
def test_gap_requires_nonnegative_integer(gap):
    with pytest.raises(ValueError, match="gap"):
        TemporalSplit(4, unit="seasons", gap=gap, gap_unit="rounds").folds(seasons())


@pytest.mark.parametrize("settings", [dict(gap=12, gap_unit="rounds"), dict(gap=99, gap_unit="kickoffs"), dict(gap=1, gap_unit="weeks")])
def test_invalid_or_consumed_finer_gap_is_rejected(settings):
    with pytest.raises(ValueError, match="gap"):
        TemporalSplit(4, unit="seasons", **settings).folds(seasons())


def test_gap_score_start_and_multiple_test_seasons_intersect():
    data = seasons()
    fold = TemporalSplit(4, test_size=2, unit="seasons", gap=10, gap_unit="rounds", score_start=1, score_rounds=(2, 3)).folds(data)[0]
    test, score = [data.metadata.iloc[rows] for rows in (fold.test, fold.score)]
    assert set(test.loc[test.source_season.eq("24_25"), "round"]) == {11, 12}
    assert set(test.loc[test.source_season.eq("25_26"), "round"]) == set(range(1, 13))
    assert set(score.source_season) == {"25_26"}
    assert set(score["round"]) == {2, 3}
    assert set(fold.score) <= set(fold.test)


def test_coarser_gap_unit_is_rejected_even_when_gap_is_zero():
    with pytest.raises(ValueError, match="finer"):
        TemporalSplit(4, unit="rounds", gap_unit="seasons").folds(seasons())


def test_persistent_ui_exposes_season_modes_and_optional_gap_unit():
    from xdiyo_analytics.ui.inventory import inventory
    from xdiyo_analytics.ui.recipe import catalog_for_ui, node
    fields = {item["name"]: item for item in inventory()["components"]["splits.TemporalSplit"]["fields"]}
    units = {item["value"] if isinstance(item, dict) else item for item in fields["unit"]["choices"]}
    assert {"seasons", "league_seasons"} <= units
    assert {"rounds", "seasons", "kickoffs"} <= set(fields["gap_unit"]["choices"])
    assert fields["gap_unit"]["default"] is None
    model = catalog_for_ui().build(node("splits.TemporalSplit", train_size=4, unit="seasons", gap=10, gap_unit="rounds"))
    fold = model.folds(seasons())[0]
    assert set(seasons().metadata.iloc[fold.test]["round"]) == {11, 12}
