"""Season-entry movement derived from explicit, ID-based membership evidence."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Membership:
    competition_id: int
    season_id: int
    start_year: int
    end_year: int
    tier: int
    teams: frozenset[int]
    complete: bool
    evidence: str
    system: str = "default"

    def __post_init__(self):
        object.__setattr__(self,"teams",frozenset(self.teams))
        if self.start_year < 1800 or self.end_year not in {self.start_year,self.start_year+1}:
            raise ValueError("Supply explicit full season years")
        if self.tier < 1 or self.competition_id < 1 or self.season_id < 1:
            raise ValueError("IDs and league tier must be positive")
        if not self.evidence or any(not isinstance(t,int) or t <= 0 for t in self.teams):
            raise ValueError("Membership needs an evidence label and positive team IDs")


def derive_team_seasons(memberships, overrides=()):
    """No network and no implicit filling of gaps or unobserved divisions.

    Overrides are explicit seeds/corrections keyed by competition_id, season_id,
    team_id and must carry movement plus evidence. Ordinary movement rules apply
    only between adjacent tiers in the same configured league system.
    """
    memberships = list(memberships)
    by_season = {}
    by_year = {}
    for item in memberships:
        key = (item.competition_id,item.season_id)
        year_key = (item.competition_id,item.start_year,item.end_year)
        if key in by_season or year_key in by_year:
            raise ValueError("Duplicate competition-season membership")
        by_season[key] = item
        by_year[year_key] = item
    manual = {}
    for override in overrides:
        key = (override["competition_id"],override["season_id"],override["team_id"])
        if key in manual:
            raise ValueError(f"Conflicting or duplicate override for {key}")
        membership = by_season.get(key[:2])
        if membership is None or key[2] not in membership.teams:
            raise ValueError(f"Override has no matching team-season membership: {key}")
        if override.get("movement") not in {"promoted","relegated","retained","other_entry","unknown"} or not override.get("evidence"):
            raise ValueError("Override needs a supported movement and explicit evidence")
        manual[key] = override
    rows = []
    for current in sorted(memberships,key=lambda m:(m.start_year,m.competition_id)):
        previous = by_year.get((current.competition_id,current.start_year-1,current.end_year-1))
        others = [m for m in memberships if m.system == current.system and
                  m.start_year == current.start_year-1 and m.end_year == current.end_year-1 and
                  m.competition_id != current.competition_id and abs(m.tier-current.tier)==1]
        for team in sorted(current.teams):
            key = (current.competition_id,current.season_id,team)
            movement,source,evidence,prior = "unknown","membership_comparison","missing_or_incomplete_predecessor",None
            if previous and team in previous.teams:
                movement,prior,evidence = "retained",previous,"observed_in_same_competition_in_consecutive_seasons"
            elif previous and previous.complete:
                candidates = [m for m in others if team in m.teams]
                if len(candidates) == 1:
                    prior = candidates[0]
                    movement = "promoted" if prior.tier > current.tier else "relegated"
                    evidence = "absent_from_complete_predecessor_and_observed_in_adjacent_tier"
                else:
                    evidence = "new_entry_origin_unknown" if not candidates else "ambiguous_previous_membership"
            if key in manual:
                override = manual[key]
                movement,source,evidence = override["movement"],"explicit_override",override["evidence"]
                prior = None  # do not attach inferred provenance to an override
            row = {"competition_id":current.competition_id,"season_id":current.season_id,"team_id":team,
                "previous_competition_id":prior.competition_id if prior else None,
                "previous_season_id":prior.season_id if prior else None,
                "movement":movement,"got_promoted":None if movement=="unknown" else movement=="promoted",
                "got_demoted":None if movement=="unknown" else movement=="relegated",
                "evidence":evidence,"source":source+":"+current.evidence,
                "membership_status":"complete" if current.complete else "observed_partial",
                "derivation_version":"1"}
            rows.append(row)
    return rows


def join_team_season_flags(matches, flags):
    """Validate a many-to-one join for each side, preserving row order/count."""
    import pandas as pd
    keys = ["competition_id","season_id","team_id"]
    flags = pd.DataFrame(flags).copy()
    if flags.empty:
        flags = pd.DataFrame(columns=keys+["got_promoted","got_demoted","movement"])
    if flags.duplicated(keys).any():
        raise ValueError("Duplicate team-season flags")
    result = matches.copy()
    reserved = "__xdiyo_row_order"
    if reserved in result:
        raise ValueError("Reserved temporary column already exists")
    result[reserved] = range(len(result))
    for side in ("home","away"):
        selected = flags[keys+["got_promoted","got_demoted","movement"]].rename(columns={
            "team_id":side+"_id","got_promoted":side+"_got_promoted",
            "got_demoted":side+"_got_demoted","movement":side+"_season_entry"})
        collisions = set(selected)-set(["competition_id","season_id",side+"_id"])
        result = result.drop(columns=list(collisions & set(result)))
        result = result.merge(selected,on=["competition_id","season_id",side+"_id"],how="left",validate="many_to_one",sort=False)
        result[side+"_season_entry"] = result[side+"_season_entry"].fillna("unknown")
        for flag in ("got_promoted","got_demoted"):
            result[side+"_"+flag] = result[side+"_"+flag].astype("boolean")
    return result.sort_values(reserved,kind="stable").drop(columns=reserved).reset_index(drop=True)
