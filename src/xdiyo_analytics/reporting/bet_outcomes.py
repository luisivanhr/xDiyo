"""Human-readable bet decisions made from retained upstream probabilities."""

from dataclasses import dataclass
import json
from typing import ClassVar
import pandas as pd

from ..evaluation.bet_decisions import prepare_bets
from ..evaluation.betting import evaluate_bets
from .contracts import Artifact, StudyResult
from .match_results import fixture_rows, _plain
from .post_training import PredictionReporter
from .teams import team_key


@dataclass(kw_only=True)
class BetOutcomeReporter(PredictionReporter):
    """Choose bets from saved probabilities; settle and display with team badges.

    Add named offers and a probability policy. output='auto' consumes calibrated
    predict_proba when present, otherwise the retained NB count distribution.
    Offers must describe the predicted label. No model fits or calibration occur
    here. Place this reporter before BetPerformanceReporter(source=its_name),
    using identical analysis scope, rows and pooling, to reuse its exact ledger.
    """
    offers: dict
    policy: object
    partition: str = 'score'
    target: str | None = None
    output: str = 'auto'
    labels: object = None
    history: object = None
    default_odds: float | None = None
    catalog: object = None
    show_badges: bool = True
    page_size: int = 25
    league: object = None
    season: object = None
    team: object = None
    round: object = None
    league_column: str | None = None
    season_column: str | None = None
    supported_types: ClassVar[tuple] = ('per_fold', 'overall')

    def run(self, context):
        if context.partition not in {'test', 'score'} or context.layout not in {'match', 'team_match'}:
            raise ValueError('Bet outcomes require match/team_match evaluation predictions.')
        if isinstance(self.page_size, bool) or not isinstance(self.page_size, int) or self.page_size < 1:
            raise ValueError('page_size must be a positive integer.')
        target = self.target if self.target is not None else (context.y.columns[0] if len(context.y.columns) == 1 else None)
        output = self.output
        if output == 'auto':
            output = 'predict_proba' if 'predict_proba' in context.predictions else 'count_distribution'
        specs, alternatives = prepare_bets(context, self.offers, self.policy, target=target,
                                           output=output, default_odds=self.default_odds)
        ledger, metrics = evaluate_bets(context, specs, labels=self.labels, history=self.history)
        base, teams, notes = fixture_rows(context, self)
        lookup = ledger.set_index(['fold_id', 'row_position', 'bet'])
        groups = dict(tuple(alternatives.groupby(['fold_id', 'row_position'], sort=False)))
        rows = []
        for row, observed in zip(base, context.y[target].tolist()):
            identity = (row['fold_id'], row['row_position'])
            choices = groups[identity]
            selected = choices.loc[choices['take']]
            choice = selected.iloc[0] if len(selected) else None
            settled = lookup.loc[(*identity, choice.bet)] if choice is not None else None
            settlement = settled.settlement if settled is not None else 'no bet'
            detail = choices[['bet', 'description', 'p_win', 'p_push', 'p_loss', 'odds', 'expected_profit', 'reason']]
            record = dict(row, result=_plain(observed), prediction=None,
                          bet=choice.description if choice is not None else 'No bet',
                          probability=_plain(choice.p_win) if choice is not None else None,
                          odds=_plain(choice.odds) if choice is not None else None,
                          profit=_plain(settled.profit) if settled is not None else None,
                          settlement=settlement, status={'win':'correct', 'loss':'incorrect'}.get(settlement, settlement),
                          alternatives=json.dumps([{k:_plain(v) for k,v in r.items()} for r in detail.to_dict('records')], allow_nan=False))
            rows.append(record)
        table = pd.DataFrame(rows, columns=[*base[0], 'result', 'prediction', 'bet', 'probability', 'odds', 'profit', 'settlement', 'status', 'alternatives']) if base else pd.DataFrame(columns=['odds', 'profit'])
        result = StudyResult('Bet outcomes', tables={'bets': table, 'alternatives': alternatives,
                                                     'ledger': ledger, 'bet_metrics': metrics}, notes=list(dict.fromkeys(notes)))
        result.notes.extend([f'Probability source: {output}. Selection uses predictions only; outcomes are used afterward for settlement.',
                             'Green wins; red losses; push, void, unavailable and no bet remain neutral. Filters only change the display.'])
        if self.default_odds is not None:
            result.notes.append(f'Explicit fallback decimal odds: {self.default_odds}. This is a fixed-odds scenario where quotes are absent.')
        result.artifacts.append(Artifact('match_results', table, f'{target}: bet decisions', dict(
            teams=teams, layout=context.layout, page_size=self.page_size, bets=True,
            numeric=False, probabilities=False, odds=bool(table.odds.notna().any()), profit=bool(table.profit.notna().any()),
            initial={key:None if getattr(self,key) is None else team_key(getattr(self,key)) for key in ('league','season','team','round')})))
        return result
