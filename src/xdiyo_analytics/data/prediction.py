"""Select prediction fixtures already present in ordinary season publications."""

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .loading import SeasonData, load_seasons


def _select_rounds(matches, rounds):
    """Filter fixture rows only, leaving their historical source untouched."""
    if rounds is None:
        return matches
    if "round" not in matches:
        raise KeyError("Selecting prediction rounds requires a round column.")

    def numbers(value):
        if isinstance(value, Integral) and not isinstance(value, bool):
            value = [value]
        if isinstance(value, (str, bytes, Mapping)):
            raise TypeError("Rounds must be integers or a sequence of integers.")
        try:
            result = list(value)
        except TypeError as error:
            raise TypeError("Rounds must be integers or a sequence of integers.") from error
        if any(isinstance(item, bool) or not isinstance(item, Integral) or item < 0 for item in result):
            raise ValueError("Round numbers must be nonnegative integers.")
        return result

    if isinstance(rounds, Mapping):
        if "source_league" not in matches:
            raise KeyError("Per-league rounds require source_league; load through load_seasons or load_prediction_fixtures.")
        mask = pd.Series(False, index=matches.index)
        for league, requested in rounds.items():
            if not isinstance(league, str) or not league:
                raise ValueError("Round mapping keys must be league names.")
            mask |= matches.source_league.eq(league).fillna(False) & matches["round"].isin(numbers(requested))
    else:
        mask = matches["round"].isin(numbers(rounds))
    return matches.loc[mask].copy()


@dataclass
class PredictionFixtures:
    """Completed/ongoing season history plus its explicitly unplayed fixtures.

    data retains completed history and all loaded tables for feature construction.
    fixtures is the selected matches table in kickoff order. Compute features and
    labels on the full history, assemble with drop_missing_targets=False, then
    align(dataset) selects prediction rows by exact fixture and team identities.
    Empty scores/statistics are never interpreted as zeros or as proof of an
    upcoming fixture: selection uses the publication's status field.
    """
    data: SeasonData
    fixtures: pd.DataFrame

    @property
    def completed_matches(self):
        """Completed matches, including early current-season results, for inspection.

        Training/split selection remains explicit; this property does not fit or
        drop other history. Prediction features can use self.data's full history.
        """
        return self.data.matches.loc[self.data.matches.status.eq("finished").fillna(False)].copy()

    def align(self, dataset, *, rounds=None):
        """Return a prediction ModelDataset in fixture order (home then away).

        Preserves one-row match or two-row team-match layout. Reindexes all three
        frames together; retains missing labels and feature values. No model is
        fitted and no statistics or feature transformations are recomputed.
        rounds=None selects all fixtures already in this batch. An integer/list
        selects those round numbers; a mapping such as {'Premier_League': [5, 6],
        'La_Liga': 7} selects only the named leagues and their requested rounds.
        This narrows the existing prediction population, never the source history.
        No matching published fixtures yields an empty, schema-preserving dataset.
        """
        from ..datasets import ModelDataset
        if not isinstance(dataset, ModelDataset):
            raise TypeError("align requires an assembled ModelDataset.")
        if not (dataset.X.index.equals(dataset.y.index) and dataset.X.index.equals(dataset.metadata.index)):
            raise ValueError("Dataset frames must retain their shared row order.")
        fixtures = _select_rounds(self.fixtures, rounds)
        keys = list(dataset.match_columns)
        if not keys or "event_id" not in keys:
            raise ValueError("Dataset needs exact match identity columns including event_id.")
        for frame in (fixtures, dataset.metadata):
            if not set(keys) <= set(frame.columns):
                raise ValueError("Fixtures and prepared data must use matching league/season/event identity columns.")
        if fixtures.duplicated(keys).any() or fixtures[keys].isna().any().any():
            raise ValueError("Prediction fixtures need unique, nonmissing match identities.")
        metadata = dataset.metadata
        if metadata[keys].isna().any().any():
            raise ValueError("Prepared match identities must be nonmissing.")
        groups = {}
        for position, key in enumerate(metadata[keys].itertuples(index=False, name=None)):
            groups.setdefault(key, []).append(position)
        positions = []
        # itertuples keeps integer IDs exact, even when other columns are floats.
        for values in fixtures.itertuples(index=False, name=None):
            fixture = dict(zip(fixtures.columns, values))
            key = tuple(fixture[name] for name in keys)
            rows = groups.get(key, [])
            if dataset.layout == "match":
                if len(rows) != 1:
                    raise ValueError(f"Fixture {key} requires exactly one prepared match row; retain missing targets during assembly.")
                for side in ("home", "away"):
                    if metadata[f"{side}_id"].iloc[rows[0]] != fixture[f"{side}_id"]:
                        raise ValueError("Prepared home/away identities disagree with the published fixture.")
                positions.extend(rows)
            elif dataset.layout == "team_match":
                by_side = {metadata["side"].iloc[position]: position for position in rows}
                if len(rows) != 2 or set(by_side) != {"home", "away"}:
                    raise ValueError(f"Fixture {key} requires both home and away prepared team rows.")
                for side, opponent in (("home", "away"), ("away", "home")):
                    position = by_side[side]
                    if (metadata["team_id"].iloc[position] != fixture[f"{side}_id"]
                            or metadata["opponent_id"].iloc[position] != fixture[f"{opponent}_id"]):
                        raise ValueError("Prepared team/opponent identities disagree with the published fixture.")
                    positions.append(by_side[side])
            else:
                raise ValueError("Prediction layout must be match or team_match.")
        frames = [frame.iloc[positions].copy(deep=True).reset_index(drop=True)
                  for frame in (dataset.X, dataset.y, metadata)]
        return replace(dataset, X=frames[0], y=frames[1], metadata=frames[2], definitions=deepcopy(dataset.definitions))


def select_prediction_fixtures(data, *, statuses=("notstarted",), as_of=None, rounds=None):
    """Select published unplayed fixtures, retaining full data for histories.

    Default uses every notstarted match in the supplied export, including catch-up
    fixtures and different league round numbers. No clock/round guessing or API
    fetching. Optional as_of keeps scheduled kickoffs at/after that UTC instant;
    a naive timestamp is interpreted as UTC. Missing times remain selectable when
    as_of=None and sort last. Source frames are never changed.
    rounds is None, an integer, a list of integers, or a league-name -> round(s)
    mapping. Only prediction fixtures are narrowed; result.data keeps all history.
    """
    if not isinstance(data, SeasonData) or data.matches is None:
        raise TypeError("Prediction fixtures require SeasonData containing matches.")
    matches = data.matches
    if not {"status", "event_id", "home_id", "away_id", "kickoff_utc"} <= set(matches):
        raise ValueError("Match metadata needs status, IDs, teams and kickoff_utc.")
    statuses = (statuses,) if isinstance(statuses, str) else tuple(statuses)
    if not statuses or any(not isinstance(status, str) or not status for status in statuses):
        raise ValueError("Select one or more explicit fixture statuses.")
    mask = matches.status.isin(statuses)
    # Use the same seconds convention as build_team_history, retaining nulls.
    seconds = matches.kickoff_utc.to_numpy(dtype="float64", na_value=np.nan)
    times = pd.to_datetime(seconds, unit="s", utc=True, errors="coerce")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if pd.isna(cutoff):
            raise ValueError("as_of must be a valid timestamp.")
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        mask &= times >= cutoff
    positions = np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))
    order = pd.Series(times[positions]).sort_values(kind="stable", na_position="last").index.to_numpy()
    fixtures = matches.iloc[positions[order]].copy(deep=True).reset_index(drop=True)
    fixtures = _select_rounds(fixtures, rounds).reset_index(drop=True)
    return PredictionFixtures(data, fixtures)


def load_prediction_fixtures(data_root, seasons=None, *, leagues=None,
                             tables=("matches", "statistics", "pregame"), statuses=("notstarted",),
                             as_of=None, rounds=None, include_awarded=False, record_dir=None, verify_hashes=False):
    """Load season exports and align their already-discovered upcoming fixtures.

    data_root may be the usual data/xDiyo_data folder or a daily export folder.
    Completed and upcoming matches can coexist in the SAME season publication;
    completed current-season matches remain available for training in result.data.
    seasons=None discovers the season
    labels published there; pass an explicit label/list to narrow them. Always
    loads matches, while other tables are chosen normally. The collector already
    decides current/next round membership; this helper preserves that population
    and selects unplayed rows rather than inferring a round from empty statistics.
    rounds optionally narrows prediction fixtures to an integer/list or a
    league-name -> round(s) mapping. All loaded completed history remains in data.
    """
    if seasons is None:
        seasons = sorted({match.group(1) for path in Path(data_root).glob("*.manifest.json")
                          if (match := re.search(r"_([0-9]{2}_[0-9]{2})\.manifest\.json$", path.name))})
        if not seasons:
            raise ValueError("No season publications were found.")
    tables = [tables] if isinstance(tables, str) else list(tables)
    tables = list(dict.fromkeys(["matches", *tables]))
    data = load_seasons(data_root, seasons, leagues=leagues, tables=tables, record_dir=record_dir,
                        verify_hashes=verify_hashes, include_awarded=include_awarded)
    return select_prediction_fixtures(data, statuses=statuses, as_of=as_of, rounds=rounds)
