"""Cutoff-safe rating diagnostics over retained evaluation fixtures."""

from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist

import numpy as np
import pandas as pd

from .contracts import Artifact, StudyResult
from .post_training import PredictionReporter
from .teams import TeamCatalog, team_key


def _queries(context, metadata, *, use_history=True):
    resources = context.resources if use_history else {}
    history = resources.get('history')
    keys = list(context.match_columns)
    fixtures = metadata.drop_duplicates(keys)
    if history is not None:
        from ..features.history import aligned_times
        source = history.copy()
        source['_rating_cutoff'] = aligned_times(history, resources.get('rating_cutoffs'), default='kickoff_at')
        query = source.merge(fixtures[keys], on=keys, how='inner', validate='many_to_one')
        if len(query[keys].drop_duplicates()) != len(fixtures):
            raise ValueError('Rating history is missing evaluation fixtures.')
    else:
        records = []
        for row in fixtures.to_dict('records'):
            if context.layout == 'team_match':
                home = row['team_id'] if row['side'] == 'home' else row['opponent_id']
                away = row['opponent_id'] if row['side'] == 'home' else row['team_id']
            else:
                home, away = row['home_id'], row['away_id']
            for side, team, opponent in [('home', home, away), ('away', away, home)]:
                records.append({**row, 'side': side, 'team_id': team, 'opponent_id': opponent,
                                'team_name': row.get(f'{side}_name'),
                                '_rating_cutoff': row.get('prediction_cutoff', row['kickoff_at'])})
        query = pd.DataFrame(records)
    if query.duplicated([*keys, 'team_id']).any():
        raise ValueError('Rating history must have one row per fixture and team.')
    if not query.empty:
        query['kickoff_at'] = pd.to_datetime(query.kickoff_at, utc=True, errors='raise')
        query['_rating_cutoff'] = pd.to_datetime(query._rating_cutoff, utc=True, errors='raise')
        if query[['kickoff_at', '_rating_cutoff']].isna().any().any():
            raise ValueError('Rating timelines require kickoff and prediction cutoff timestamps.')
    return query.reset_index(drop=True)


def _model_rating(model):
    """Inspect adapter wrappers without calling predict, update, or fit."""
    from ..ratings import RatingRun
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        run = getattr(model, 'rating_report_run_', None)
        if run is None:
            run = getattr(model, 'run_', None)
        if isinstance(run, RatingRun):
            return run
        model = getattr(model, 'estimator', None)
    return None


def _bounds(mean, sd, level, *, gamma=False):
    if sd is None or not np.isfinite(mean) or not np.isfinite(sd) or sd < 0:
        return None, None
    if sd == 0:
        return mean, mean
    if gamma:
        from scipy.stats import gamma as distribution
        if mean <= 0:
            return None, None
        shape, scale = (mean / sd) ** 2, sd ** 2 / mean
        lo, hi = distribution.ppf([(1-level)/2, (1+level)/2], a=shape, scale=scale)
        return float(lo), float(hi)
    width = NormalDist().inv_cdf((1+level)/2) * sd
    return mean-width, mean+width


@dataclass(kw_only=True)
class RatingReporter(PredictionReporter):
    """Inspect retained rating states at selected test fixtures' prediction cutoffs.

    Glicko displays rating with normal RD intervals. Bayesian ratings display
    attack and defensive vulnerability with Gamma posterior intervals, plus
    shared league home advantage. Lower vulnerability means stronger defense.
    Custom RatingRun numeric fields display without invented uncertainty.
    Select league, rating source, all teams or one team in the rendered viewer;
    badge buttons toggle a team's line and interval together. Distinct fitted
    folds stay separate. No fit, predict, replay or state update is performed.

    FootballExperiment supplies resources automatically. Standalone usage passes
    resources={'ratings': named_runs, 'history': history, 'rating_cutoffs': cutoffs,
    'team_catalog': catalog} to PostTrainingAnalysis.run. History/cutoffs are
    optional when evaluation metadata fully describes the rating queries.
    """
    partition: str = 'test'
    pooling: str | None = 'occurrences'
    ratings: object = None
    interval: float = 0.95
    bayesian_interval: float = 0.80
    show_badges: bool = True
    catalog: object = None

    def run(self, context):
        from ..ratings import RatingRun, BayesianRatingRun
        if context.partition not in {'test', 'score'}:
            raise ValueError('RatingReporter uses test or score evaluation fixtures.')
        if context.pooling == 'mean':
            raise ValueError('RatingReporter cannot average rating states across folds; use occurrences, first or last.')
        for name in ('interval', 'bayesian_interval'):
            level = getattr(self, name)
            if isinstance(level, bool) or not isinstance(level, Real) or not np.isfinite(level) or not 0 < level < 1:
                raise ValueError(f'{name} must be a finite number strictly between 0 and 1.')
        runs = dict(context.resources.get('ratings') or {})
        for fold_id, model in context.models.items():
            run = _model_rating(model)
            if run is not None:
                runs[f'model:fold:{fold_id}'] = run
        selected = list(runs) if self.ratings is None else ([self.ratings] if isinstance(self.ratings, str) else list(self.ratings))
        if len(set(selected)) != len(selected) or set(selected) - set(runs):
            raise ValueError('ratings must select distinct retained rating names: ' + ', '.join(runs))
        result = StudyResult('Rating histories', notes=[
            'States are queried at each evaluation fixture’s prediction cutoff; the horizontal axis is kickoff time.',
            'Only selected evaluation fixtures appear. Separate fold curves are never averaged.',
            f'Glicko: {self.interval:.0%} approximate normal RD bands. Bayesian: {self.bayesian_interval:.0%} equal-tail Gamma posterior bands. These are not match-outcome prediction intervals.',
            'Bayesian defense is defensive vulnerability: lower is stronger. Home advantage is shared within each league, not across disconnected leagues.',
        ])
        if not selected:
            result.notes.append('No retained ratings are available. Prepare rating features or provide named RatingRun resources; older saved runs may need preparation again.')
            return result
        raw_catalog = self.catalog if self.catalog is not None else context.resources.get('team_catalog')
        catalog = raw_catalog if isinstance(raw_catalog, TeamCatalog) else TeamCatalog(raw_catalog or {})
        teams, records = {}, []
        for fold_id, metadata in context.metadata.groupby(level='fold_id', sort=False):
            for name in selected:
                if name.startswith('model:fold:') and name != f'model:fold:{fold_id}':
                    continue
                query = _queries(context, metadata, use_history=not name.startswith('model:fold:'))
                if query.empty:
                    continue
                run = runs[name]
                if not isinstance(run, RatingRun):
                    raise TypeError(f'{name}: expected RatingRun or BayesianRatingRun.')
                bayesian = isinstance(run, BayesianRatingRun)
                fields = list(run.initial_state)
                values = run.features(query, cutoffs=query._rating_cutoff, side='for', fields=fields)
                if bayesian:
                    panels = [('Attack', 'attack_mean', 'attack_sd'),
                              ('Defense · vulnerability', 'defence_vulnerability_mean', 'defence_vulnerability_sd')]
                elif 'rating' in fields:
                    panels = [('Rating', 'rating', 'rd' if 'rd' in fields else None)]
                else:
                    panels = [(field, field, None) for field in fields]
                fixture = (run.fixture_features(query, cutoffs=query._rating_cutoff,
                           fields=('home_advantage_mean', 'home_advantage_sd')) if bayesian else None)
                for i, row in enumerate(query.to_dict('records')):
                    team = team_key(row['team_id'])
                    if team not in teams:
                        display, badge, note = catalog.display(row['team_id'], badges=self.show_badges)
                        if not catalog.entries.get(team, {}).get('name') and pd.notna(row.get('team_name')):
                            display = str(row['team_name'])
                        teams[team] = dict(name=display, badge=badge)
                        if note:
                            result.notes.append(note)
                    league = str(row.get('source_league', row['competition_id']))
                    common = dict(source=name, fold=str(fold_id), league=league,
                                  competition=team_key(row['competition_id']),
                                  kickoff=row['kickoff_at'].isoformat(), cutoff=row['_rating_cutoff'].isoformat(),
                                  event=team_key(row['event_id']))
                    for stream in run.streams:
                        for panel, field, uncertainty in panels:
                            mean = float(values.iloc[i][f'{stream}::team::{field}'])
                            sd = float(values.iloc[i][f'{stream}::team::{uncertainty}']) if uncertainty else None
                            lo, hi = _bounds(mean, sd, self.bayesian_interval if bayesian else self.interval, gamma=bayesian)
                            records.append(dict(**common, stream=str(stream), panel=panel, team=team,
                                                mean=mean, lower=lo, upper=hi))
                    if fixture is not None and row['side'] == 'home':
                        mean, sd = (float(fixture.iloc[i][field]) for field in ('home_advantage_mean', 'home_advantage_sd'))
                        lo, hi = _bounds(mean, sd, self.bayesian_interval, gamma=True)
                        records.append(dict(**common, stream='score', panel='Home advantage', team=None,
                                            mean=mean, lower=lo, upper=hi))
        table = pd.DataFrame(records)
        if table.empty:
            result.notes.append('No rating queries match the selected evaluation fixtures.')
            return result
        table = table.replace([np.inf, -np.inf], np.nan)
        table = table.sort_values(['kickoff', 'source', 'fold'], kind='stable').reset_index(drop=True)
        result.tables['rating_states'] = table
        # JSON-safe data, rather than pre-rendered plots for every league/team.
        payload = dict(rows=table.astype(object).where(table.notna(), None).to_dict('records'),
                       teams=teams, interval=self.interval, bayesian_interval=self.bayesian_interval)
        result.artifacts.append(Artifact('ratings', payload, 'Team ratings and uncertainty'))
        return result
