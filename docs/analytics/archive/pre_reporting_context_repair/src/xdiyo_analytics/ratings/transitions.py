"""Rating-owned state transitions and frozen, stream-specific cohorts."""

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol
import math

from .glicko import glicko_state


class RatingTransition(Protocol):
    """Custom adapters own their state semantics; helpers never guess fields.

    apply receives copies of the previous state, season-entry context and
    destination cohort records (team_id, state, position). Return the same state
    fields. Policies used in feature ASTs must be immutable and hashable.
    """
    def apply(self, state, *, context, cohort): ...


@dataclass(frozen=True)
class GlickoTransition:
    """Optional seasonal RD inflation and movement shrinkage; sigma unchanged.

    Promoted teams use bottom-n prior-season destination teams, relegated top-m.
    rank_by='standings' uses the latest eligible pregame position from the
    destination's previous season (not a reconstructed final league table).
    rank_by='rating' uses this exact stream's pre-transition rating. Missing
    cohort information leaves mu unchanged; incoming teams are excluded.
    phi_scale applies at every known prior-state season entry; movement_phi_scale
    is an additional multiplier for promotion/relegation. Both default to one.
    """
    phi_scale: float = 1.
    movement_phi_scale: float = 1.
    shrinkage: float = .5
    top: int = 3
    bottom: int = 3
    rank_by: str = "standings"
    aggregate: str = "mean"

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 1 for x in (self.phi_scale, self.movement_phi_scale)):
            raise ValueError("Uncertainty inflation multipliers must be finite and >= 1.")
        if not math.isfinite(self.shrinkage) or not 0 <= self.shrinkage <= 1:
            raise ValueError("shrinkage must lie in [0, 1].")
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in (self.top, self.bottom)):
            raise ValueError("top and bottom must be positive cohort sizes.")
        if self.rank_by not in {"standings", "rating"} or self.aggregate not in {"mean", "median"}:
            raise ValueError("Choose standings/rating ranking and mean/median aggregation.")

    def apply(self, state, *, context, cohort):
        import statistics
        movement = context["movement"]
        moved = movement in {"promoted", "relegated"}
        rating, rd = state["rating"], state["rd"]
        if context["has_prior_state"]:
            rd *= self.phi_scale
        if moved:
            rd *= self.movement_phi_scale
            if self.rank_by == "standings":
                ranked = [r for r in cohort if r["position"] is not None and math.isfinite(r["position"]) and r["position"] >= 1]
                ranked.sort(key=lambda r: (r["position"], str(r["team_id"])))
            else:
                ranked = sorted(cohort, key=lambda r: (-r["state"]["rating"], str(r["team_id"])))
            chosen = ranked[-self.bottom:] if movement == "promoted" else ranked[:self.top]
            if chosen:
                aggregate = statistics.mean if self.aggregate == "mean" else statistics.median
                prior = aggregate(r["state"]["rating"] for r in chosen)
                rating = (1 - self.shrinkage) * rating + self.shrinkage * prior
        return glicko_state(rating, rd, state["sigma"])


class TransitionReplay:
    """One transition per stream/team/season, before equal-time result updates."""

    def __init__(self, context, scope, policy, initial):
        import pandas as pd
        if set(scope) - {"competition_id", "season_id"}:
            raise ValueError("Season transitions currently support competition_id/season_id rating scopes.")
        self.context, self.scope, self.policy, self.initial = context, scope, policy, initial
        self.events = {}
        seen = set()
        for record in context.records.values():
            if pd.notna(record["entry_at"]):
                identity = record["entry_at"], self.key(record)
                if identity in seen:
                    raise ValueError("This rating scope merges simultaneous season entries for one team; include competition_id in scope.")
                seen.add(identity)
                self.events.setdefault(record["entry_at"], []).append(record)

    def key(self, record, previous=False):
        import pandas as pd
        prefix = "previous_" if previous else ""
        values = tuple(record[prefix + name] for name in self.scope)
        if any(pd.isna(x) for x in values):
            return None
        return (*values, record["team_id"])

    def apply(self, time, states, games_seen, latest, stream):
        import pandas as pd
        context, records = self.context, self.events.get(time, ())
        pending, snapshots = [], []
        # All cohort and previous-state reads happen before any season entry in
        # this batch is committed, so input ordering cannot affect priors.
        for record in records:
            key = self.key(record)
            previous = self.key(record, previous=True)
            state_key = previous if previous in states else key
            old = dict(states.get(state_key, self.initial))
            prior_season = context.predecessors[(record["competition_id"], record["season_id"])]
            cohort, contributing = [], [latest[state_key]] if state_key in latest else []
            if prior_season is not None:
                rows = context.prior_rows(prior_season, time)
                table = context.history.iloc[rows]
                for team, group in table.groupby("team_id", sort=False):
                    if team == record["team_id"]:
                        continue
                    candidate = dict(competition_id=prior_season[0], season_id=prior_season[1], team_id=team)
                    candidate_key = self.key(candidate)
                    if candidate_key not in states:
                        continue
                    # Exclude all other incoming teams from destination cohorts.
                    current = context.records.get((record["competition_id"], record["season_id"], team))
                    if current is not None and current["movement"] in {"promoted", "relegated", "other_entry"}:
                        continue
                    position = None
                    if "team_position" in group:
                        ranked = group.sort_values("kickoff_at", kind="stable")["team_position"].dropna()
                        if len(ranked):
                            position = float(ranked.iloc[-1])
                    cohort.append(dict(team_id=team, state=dict(states[candidate_key]), position=position))
                    if candidate_key in latest:
                        contributing.append(latest[candidate_key])
            details = {**record, "stream": stream, "has_prior_state": state_key in states}
            state = self.policy.apply(old, context=details, cohort=cohort)
            if set(state) != set(self.initial) or not all(math.isfinite(x) for x in state.values()):
                raise ValueError("A rating transition must retain finite numeric state fields.")
            last = max(contributing) if contributing else pd.NaT
            count = games_seen.get(state_key, 0)
            pending.append((key, dict(state), last, count))
            snapshots.append({**dict(zip(self.scope, key[:-1])), "team_id": key[-1],
                              "stream": stream, "recorded_at": time,
                              "latest_kickoff_at": last, "games_seen": count,
                              "snapshot_order": 0, "snapshot_kind": "transition", **state})
        for key, state, last, count in pending:
            states[key] = state
            games_seen[key] = count
            if pd.notna(last):
                latest[key] = last
        return snapshots
