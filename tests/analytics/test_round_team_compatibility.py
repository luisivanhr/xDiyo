"""Keep workbook round identities and dated canonical/display identities separate."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from xdiyo_analytics.data import assign_fixture_rounds, load_fixture_rounds
from xdiyo_analytics.data.fixture_rounds import FALLBACKS, IDENTITY, SELECTED
from xdiyo_analytics.data.catalogue import (
    Registry, apply_reviewed_team_directory, load_reviewed_team_directory,
)
from xdiyo_analytics.data.catalogue.registry import name_key
from xdiyo_analytics.reporting import TeamCatalog

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def combined():
    return (load_fixture_rounds(ROOT / "data/fixture_rounds/v59_20261108"),
            load_reviewed_team_directory(), TeamCatalog.builtin())


def test_all_round_fixture_sides_have_unique_dated_directory_identity(combined):
    mapping, directory, catalog = combined
    aliases = defaultdict(list)
    for team in directory["teams"]:
        for alias in team["names"]:
            aliases[(team["country"], name_key(alias["name"]))].append(
                (alias["start"], alias["end"], team["team_id"]))
    teams = set()
    for fixture in mapping.selections.itertuples():
        for side in ("home", "away"):
            candidates = {tid for start, end, tid in aliases[
                (fixture.country, name_key(getattr(fixture, side + "_team")))]
                if start <= fixture.kickoff_date <= end}
            assert len(candidates) == 1, (fixture.match_id, side, candidates)
            teams.update(candidates)
    assert len(mapping.selections) * 2 == 122976
    assert len(teams) == 814
    for team_id in teams:
        name, image, error = catalog.display(team_id, badges=True)
        assert name and image and error is None


def test_team_application_does_not_rewrite_round_source_identity_or_fallbacks(combined):
    mapping, directory, catalog = combined
    source = mapping.selections.loc[mapping.selections.match_id.isin(FALLBACKS), list(IDENTITY)].copy()
    before = source.copy(deep=True)
    # Synthetic test scope: applying a country directory grants no membership,
    # fixture identity, season or squad approval in a production publication.
    registry = Registry()
    registry.seed("competition", 9000001, name="Test Australia scope",
                  region="Australia", entity_kind="club_league")
    old_registry = deepcopy(registry.payload)
    reviewed = apply_reviewed_team_directory(registry, directory, catalog=catalog)
    assert registry.payload == old_registry
    assert "fixture" not in reviewed.payload["entities"]
    assert "season" not in reviewed.payload["entities"]
    for fixture in source.itertuples():
        for side in ("home", "away"):
            team_id = reviewed.resolve("team", getattr(fixture, side + "_team"), 9000001, fixture.kickoff_date)
            assert team_id is not None
            assert reviewed.squad(team_id, fixture.kickoff_date) == "needs_review"
            assert catalog.display(team_id, badges=True)[2] is None
    result = assign_fixture_rounds(source, mapping, require_all_targets=False)
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_frame_equal(result[list(IDENTITY)], before)
    assert set(result) == set(IDENTITY) | set(SELECTED)
    assert result.set_index("match_id").season_round.to_dict() == {mid: pair[0] for mid, pair in FALLBACKS.items()}
    assert result.attrs["fixture_rounds"]["selection_semantics"] == "mixed_original_and_provider_conventions"
    assert result.attrs["fixture_rounds"]["model_input_approved"] is False
    assert result.attrs["fixture_rounds"]["feature_asof_approved"] is False
