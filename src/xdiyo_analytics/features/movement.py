"""Known season-entry flags, resolved by stable team-season identity."""
from dataclasses import dataclass
from collections.abc import Mapping
import pandas as pd

from .expressions import Expr
from .transitions import _validate_record_flags, TransitionContext


@dataclass(frozen=True)
class TeamMovement(Expr):
    """A known 0/1/missing focal-team flag; administrative entry is not retained.

    Supplied team_seasons records override native history for their exact key.
    Missing evidence stays missing; no membership-subset inference makes flags.
    """
    movement: str = 'promoted'

    def __post_init__(self):
        if self.movement not in ('promoted', 'relegated'):
            raise ValueError('TeamMovement selects promoted or relegated.')


def movement_records(history, supplied=None):
    keys = ['competition_id', 'season_id', 'team_id']
    records = {}
    if supplied is not None:
        rows = (pd.DataFrame(supplied, dtype=object).to_dict('records')
                if isinstance(supplied, (pd.DataFrame, Mapping)) else list(supplied))
        for row in rows:
            key = tuple(row[k] for k in keys)
            if any(pd.isna(v) for v in key) or key in records:
                raise ValueError('Movement evidence requires unique nonmissing team-season keys.')
            records[key] = dict(row)
    for key, group in history.groupby(keys, sort=False):
        if key not in records:
            record = dict(zip(keys, key))
            for source, target in [('team_season_entry', 'movement'), ('team_got_promoted', 'got_promoted'),
                                   ('team_got_demoted', 'got_demoted')]:
                if source in group:
                    values = group[source].dropna().unique()
                    if len(values) > 1:
                        raise ValueError('Conflicting native team-season movement evidence.')
                    record[target] = values[0] if len(values) else None
            records[key] = record
    for record in records.values():
        pc, ps = record.get('previous_competition_id'), record.get('previous_season_id')
        if pd.isna(pc) != pd.isna(ps):
            raise ValueError('Supply both predecessor IDs or explicitly null both.')
        pc, ps = record.get('previous_competition_id'), record.get('previous_season_id')
        if pd.isna(pc) != pd.isna(ps):
            raise ValueError('Supply both predecessor IDs or explicitly null both.')
        movement = record.get('movement')
        if movement is None or pd.isna(movement):
            promoted, relegated = record.get('got_promoted'), record.get('got_demoted')
            if pd.notna(promoted) and bool(promoted):
                movement = 'promoted'
            elif pd.notna(relegated) and bool(relegated):
                movement = 'relegated'
            else:
                movement = 'unknown'
        if movement not in ('promoted', 'relegated', 'retained', 'other_entry', 'unknown'):
            raise ValueError(f'Unknown season movement: {movement}')
        record['movement'] = movement
        _validate_record_flags(record)
        for field, label in [('got_promoted', 'promoted'), ('got_demoted', 'relegated')]:
            record[field] = None if movement == 'unknown' else movement == label
    return records


def statistical_context(base, supplied=None):
    """Resolve native/override flags without changing legacy or rating contexts."""
    resolved = movement_records(base.history, supplied)
    records = []
    for key, evidence in resolved.items():
        # Adjacent observed membership may identify own history, but never a flag.
        record = {**base.records.get(key, {}), **evidence}
        if evidence['movement'] in ('promoted', 'relegated', 'other_entry'):
            for name in ('previous_competition_id', 'previous_season_id'):
                if name not in evidence:
                    record[name] = None
        records.append(record)
    return TransitionContext(base.history, team_seasons=records, season_starts=base.anchors,
                             population=base.population)
