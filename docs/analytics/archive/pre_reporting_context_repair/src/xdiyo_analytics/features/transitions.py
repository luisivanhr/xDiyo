"""Shared season-entry context; policies decide how to use it."""

from .league import LeaguePopulation


def build_team_seasons(history, competition_info, *, season_years=None,
                       complete_memberships=(), overrides=()):
    """Adapt prepared histories to the existing ID-based movement evidence core.

    competition_info maps competition IDs to {'tier': int, 'system': str}.
    season_years optionally maps (competition_id, season_id) to full (start,end)
    years. Otherwise source_season/season_year labels are parsed. Membership is
    observed-partial unless explicitly named in complete_memberships. A new
    appearance alone never establishes promotion/relegation.
    """
    import re
    import pandas as pd
    from utils.team_seasons import Membership, derive_team_seasons
    memberships, overrides = [], list(overrides)
    complete = set(complete_memberships)
    for key, rows in history.groupby(["competition_id", "season_id"], sort=False):
        config = competition_info[key[0]]
        years = None if season_years is None else season_years.get(key)
        if years is None:
            column = "source_season" if "source_season" in rows else "season_year"
            labels = rows[column].dropna().unique()
            if len(labels) != 1:
                raise ValueError(f"Supply season_years for {key}.")
            parts = re.fullmatch(r"(\d{2}|\d{4})(?:[/_\-](\d{2}|\d{4}))?", str(labels[0]))
            if parts is None:
                raise ValueError(f"Supply full season_years for {key}.")
            start = int(parts[1]); start += 2000 if start < 100 else 0
            end = int(parts[2]) if parts[2] else start
            end += (start // 100) * 100 if end < 100 else 0
            if end < start and parts[2] and len(parts[2]) == 2:
                end += 100
            years = start, end
        memberships.append(Membership(
            int(key[0]), int(key[1]), *years, int(config["tier"]),
            frozenset(int(x) for x in rows["team_id"].dropna()), key in complete,
            "prepared_history_membership", system=config["system"],
        ))
    records = derive_team_seasons(memberships, overrides)
    # The older core deliberately omits predecessor provenance for overrides.
    # Preserve explicit predecessor IDs supplied by this caller.
    manual = {(r["competition_id"], r["season_id"], r["team_id"]): r for r in overrides}
    for record in records:
        override = manual.get(tuple(record[k] for k in ("competition_id", "season_id", "team_id")), {})
        for name in ("previous_competition_id", "previous_season_id"):
            if name in override:
                record[name] = override[name]
    return pd.DataFrame(records, dtype=object)


class TransitionContext:
    """Frozen boundaries, prior membership and completed-round counts.

    team_seasons is an optional table keyed by competition_id/season_id/team_id,
    with movement and optional previous_competition_id/previous_season_id.
    season_starts maps league-season keys to datetimes, or is a table with those
    keys plus entry_at. Defaults use the earliest prediction cutoff per season.
    Explicit starts must not follow that first prediction.
    """

    def __init__(self, history, *, cutoffs=None, available_at=None,
                 team_seasons=None, season_starts=None, population=None):
        import pandas as pd
        self.history = history
        self.population = population or LeaguePopulation(history, cutoffs=cutoffs, available_at=available_at)
        self.seasons = list(zip(history["competition_id"].tolist(), history["season_id"].tolist()))
        self.teams = history["team_id"].tolist()
        self.groups = history.groupby(["competition_id", "season_id"], sort=False).indices
        starts = {} if season_starts is None else season_starts
        if hasattr(starts, "columns"):
            if starts.duplicated(["competition_id", "season_id"]).any():
                raise ValueError("Season starts must be unique per competition-season.")
            starts = {(c, s): t for c, s, t in starts[["competition_id", "season_id", "entry_at"]].itertuples(index=False, name=None)}
        self.anchors = {}
        for key, rows in self.groups.items():
            first = self.population.cutoff.iloc[rows].min()
            value = pd.to_datetime(starts.get(key, first), utc=True)
            if pd.notna(first) and pd.notna(value) and value > first:
                raise ValueError("A season entry must not follow its first prediction cutoff.")
            self.anchors[key] = value
        self.predecessors = {}
        for key, time in self.anchors.items():
            previous = [other for other, before in self.anchors.items()
                        if other[0] == key[0] and pd.notna(before) and before < time]
            self.predecessors[key] = max(previous, key=self.anchors.get) if previous else None
        supplied = {}
        if team_seasons is not None:
            table = pd.DataFrame(team_seasons, dtype=object)
            keys = ["competition_id", "season_id", "team_id"]
            if table.duplicated(keys).any():
                raise ValueError("team_seasons must be unique per competition-season-team.")
            supplied = {tuple(r[k] for k in keys): r for r in table.to_dict("records")}
        self.records = {}
        for key, rows in self.groups.items():
            previous = self.predecessors[key]
            old_teams = set() if previous is None else set(history.iloc[self.groups[previous]]["team_id"].tolist())
            for team in dict.fromkeys(history.iloc[rows]["team_id"].tolist()):
                record = dict(supplied.get((*key, team), {}))
                movement = record.get("movement", "retained" if team in old_teams else "unknown")
                if movement not in {"retained", "promoted", "relegated", "unknown", "other_entry"}:
                    raise ValueError(f"Unknown season movement: {movement}")
                pc, ps = record.get("previous_competition_id"), record.get("previous_season_id")
                if pd.isna(pc) or pd.isna(ps):
                    pc, ps = previous if team in old_teams and movement == "retained" else (None, None)
                record.update(competition_id=key[0], season_id=key[1], team_id=team,
                              entry_at=self.anchors[key], movement=movement,
                              previous_competition_id=pc, previous_season_id=ps)
                self.records[(*key, team)] = record
        self._round_counts = {}

    def record(self, row):
        return self.records[(*self.seasons[row], self.teams[row])]

    def prior_rows(self, season, boundary):
        """Select eligible previous-season observations at a frozen boundary."""
        import numpy as np
        if season is None or season not in self.groups:
            return np.array([], dtype=int)
        rows = self.groups[season]
        p = self.population
        mask = p.valid[rows] & (p.kickoff.iloc[rows] < boundary).to_numpy()
        mask &= (p.available.iloc[rows] <= boundary).to_numpy()
        return rows[mask]

    def completed_rounds(self, row, round_keys, *, time=None):
        import pandas as pd
        info, _ = self.population.round_info(round_keys)
        season = self.seasons[row]
        time = self.population.cutoff.iloc[row] if time is None else time
        key = season, time, round_keys
        if key not in self._round_counts:
            self._round_counts[key] = sum(
                self.seasons[positions[0]] == season and pd.notna(close)
                and close <= time and last < time
                for positions, _, close, last in info.values()
            )
        return self._round_counts[key]
