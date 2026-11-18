from test_ui_workflow import ui_recipe
import pytest
from xdiyo_analytics.splits import TemporalSplit
from split_samples import dataset, cases


def test_native_labels_not_storage_aliases_and_chronological_order():
    data = dataset(specs=[
        dict(season=90, experiment_season="2020/2021", day=0),
        dict(season=80, experiment_season="2021/2021", day=2),
        dict(season=70, experiment_season="2022/2022", day=3),
    ])
    data.metadata["source_season"] = "2020"
    folds = TemporalSplit(1, unit="seasons").folds(data)
    assert len(folds) == 2
    assert cases(data, folds[0].train) == {0}
    assert cases(data, folds[0].test) == {1}
    assert folds[0].metadata["train_blocks"] == (("2020/2021",),)
    assert folds[0].metadata["season_policy"] == "sofascore_year_pair_v1"


def test_missing_native_metadata_requires_explicit_legacy():
    data = dataset(specs=[dict(season=2020), dict(season=2021)])
    data.metadata.drop(columns="experiment_season", inplace=True)
    with pytest.raises(ValueError, match="reload scope year-pair metadata"):
        TemporalSplit(1, unit="seasons").folds(data)
    folds = TemporalSplit(1, unit="seasons", block_by=("source_season",)).folds(data)
    assert folds[0].metadata["season_policy"] == "explicit_blocks_v1"


@pytest.mark.parametrize("labels,ids,message", [
    (["2020/2020", "2021/2021"], [1, 1], "conflicting"),
    (["", "2021/2021"], [1, 2], "blank"),
])
def test_native_label_ambiguity_fails_closed(labels, ids, message):
    data = dataset(specs=[dict(season=i, experiment_season=l) for i, l in zip(ids, labels)])
    with pytest.raises(ValueError, match=message):
        TemporalSplit(1, unit="seasons").folds(data)


def test_same_label_across_leagues_is_deliberately_pooled_not_identity():
    data = dataset(specs=[
        dict(competition=17, season=10, experiment_season="2024/2025", day=0),
        dict(competition=18, season=20, experiment_season="2024/2025", day=0),
        dict(competition=17, season=11, experiment_season="2025/2026", day=3),
        dict(competition=18, season=21, experiment_season="2025/2026", day=3),
    ])
    fold = TemporalSplit(1, unit="seasons").folds(data)[0]
    assert cases(data, fold.train) == {0, 1}
    assert cases(data, fold.test) == {2, 3}


def test_pooled_league_seasons_keeps_competition_identity():
    data = dataset(specs=[
        dict(competition=17, season=1, day=0),
        dict(competition=18, season=1, day=1),
        dict(competition=17, season=2, day=2),
    ])
    folds = TemporalSplit(1, unit="league_seasons", calendar_by=()).folds(data)
    assert len(folds) == 2
    assert folds[0].metadata["block_by"] == ("competition_id", "season_id")
    assert cases(data, folds[0].test) == {1}


def test_fold_description_prefers_verbatim_native_label():
    from xdiyo_analytics.ui.server import _fold_description
    data = dataset(specs=[dict(season=2020, experiment_season="2020/2021"), dict(season=2021, experiment_season="2021/2022")])
    fold = TemporalSplit(1, unit="seasons").folds(data)[0]
    description = _fold_description(data.metadata, fold)
    assert "train 2020/2021" in description
    assert "test 2021/2022" in description


def test_match_report_prefers_native_and_retains_explicit_legacy_override():
    from match_results_samples import fixture_context
    from xdiyo_analytics.reporting.match_results import fixture_rows, MatchResultReporter
    ctx = fixture_context()
    ctx.metadata["source_season"] = "2020"
    ctx.metadata["experiment_season"] = "2020/2021"
    rows, _, _ = fixture_rows(ctx, MatchResultReporter(type="overall"))
    assert {r["season"] for r in rows} == {"2020/2021"}
    rows, _, _ = fixture_rows(ctx, MatchResultReporter(type="overall", season_column="source_season"))
    assert {r["season"] for r in rows} == {"2020"}


def test_transition_year_never_substitutes_storage_alias_for_native():
    import pandas as pd
    from xdiyo_analytics.features.transitions import _season_start_year
    assert _season_start_year(pd.DataFrame({"experiment_season":["2020/2021"], "source_season":["2021"]})) == 2020
    assert _season_start_year(pd.DataFrame({"experiment_season":["Special"], "source_season":["2021"]})) is None




def test_round_tickets_use_native_label_instead_of_storage_alias():
    from test_bet_tickets import ledger
    from xdiyo_analytics.evaluation import compose_bets, Parlay
    data = ledger(("win",) * 4)
    data["experiment_season"] = ["2020/2021", "2020/2021", "2021/2021", "2021/2021"]
    # One storage alias must not merge two provider-season cohorts.
    data["source_season"] = "2021"
    tickets, _, _ = compose_bets(data, Parlay(grouping="round"))
    assert set(tickets.experiment_season) == {"2020/2021", "2021/2021"}
    assert len(tickets) == 2


def test_scope_pair_preserves_calendar_and_provider_display():
    import pandas as pd
    from xdiyo_analytics.data.season_convention import attach_season_convention
    raw = pd.DataFrame({"season_label": ["2024"], "season_id": [57353]})
    got = attach_season_convention(raw, {"season_start": 2024, "season_end": 2024})
    assert got.experiment_season.tolist() == ["2024/2024"]
    assert got.season_label.tolist() == ["2024"]
    assert "experiment_season" not in raw
    cross = attach_season_convention(raw, {"season_start": 2024, "season_end": 2025})
    assert cross.experiment_season.tolist() == ["2024/2025"]
    with pytest.raises(ValueError, match="disagrees"):
        attach_season_convention(got, {"season_start": 2024, "season_end": 2025})


def test_loading_and_history_expose_scope_pair_without_native_enrichment(export):
    from xdiyo_analytics.data import load_season
    from xdiyo_analytics.histories import build_team_history
    data = load_season(export.root, "Premier_League_24_25")
    assert set(data.matches.experiment_season) == {"2024/2025"}
    assert data.provenance["season_convention"] == "sofascore_year_pair_v1"
    history = build_team_history(data)
    assert set(history.experiment_season) == {"2024/2025"}



@pytest.mark.parametrize("bad", ["2024", "24/24", "2024/2026", "2024/2023"])
def test_noncanonical_experiment_pair_rejected(bad):
    data = dataset(specs=[dict(season=2024, experiment_season=bad), dict(season=2025)])
    with pytest.raises(ValueError):
        TemporalSplit(1, unit="seasons").folds(data)


def test_distinct_phase_ids_can_share_experiment_pair_without_identity_merge():
    data = dataset(specs=[
        dict(season=1, experiment_season="2024/2024", day=0),
        dict(season=2, experiment_season="2024/2024", day=1),
        dict(season=3, experiment_season="2025/2025", day=3)])
    fold = TemporalSplit(1, unit="seasons").folds(data)[0]
    assert cases(data, fold.train) == {0, 1}
    assert set(data.metadata.iloc[fold.train].season_id) == {1, 2}


def test_prepared_recipe_preserves_year_pair_into_folds(ui_recipe):
    from xdiyo_analytics.ui import prepare_recipe
    prepared = prepare_recipe(ui_recipe)
    assert set(prepared.dataset.metadata.experiment_season) == {"2022/2023", "2023/2024", "2024/2025"}
    assert prepared.split_plan.folds[0].metadata["block_by"] == ("experiment_season",)
    assert prepared.split_plan.folds[0].metadata["test_blocks"] == (("2024/2025",),)
