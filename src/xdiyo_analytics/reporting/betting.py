"""Bet settlement/accounting presentation, independent of model fitting."""

from dataclasses import dataclass
import pandas as pd

from ..evaluation.betting import evaluate_bets
from .contracts import Artifact, StudyResult
from .post_training import PredictionReporter
from .studies import _plotting, _style


@dataclass(kw_only=True)
class BetPerformanceReporter(PredictionReporter):
    """Report named BetSpec/dict options with decimal odds and supplied decisions.

    source reuses an earlier BetOutcomeReporter's prepared ledger. For manual
    bets, labels or history is required, exclusively. time_column defaults to kickoff_at:
    cumulative profit is an outcome summary in that order, NOT a cash-flow or
    bankroll simulation. Supply actual settlement times for settlement ordering.
    Unresolved profits remain missing; cumulative values sum known settlements.
    """
    bets: dict | None = None
    labels: object = None
    history: object = None
    time_column: str = "kickoff_at"
    source: str | None = None
    composition: object = None

    def run(self, context):
        if self.source is not None:
            if self.bets is not None:
                raise ValueError("Choose a BetOutcomeReporter source or manual bets, not both.")
            previous = context.previous_results.get(self.source)
            if previous is None or 'alternatives' not in previous.tables or 'ledger' not in previous.tables:
                raise ValueError(f"Bet source {self.source!r} needs an earlier BetOutcomeReporter with identical scope, partition and pooling.")
            ledger, metrics = previous.tables['ledger'].copy(), previous.tables['bet_metrics'].copy()
            if self.composition is not None and 'tickets' in previous.tables:
                raise ValueError("The source already contains tickets; configure composition on only one reporter.")
            if ledger.loc[ledger['take'], 'odds'].isna().any():
                raise ValueError("Bet performance needs odds for selected bets. Add quotes or explicit default_odds to the source reporter.")
        else:
            if self.bets is None:
                raise ValueError("Choose a BetOutcomeReporter source or configure manual bets.")
            ledger, metrics = evaluate_bets(context, self.bets, labels=self.labels, history=self.history)
        result = StudyResult("Bet performance", [Artifact("table", metrics, "Bet metrics")],
                             {"metrics": metrics, "ledger": ledger}, [
                                 "Decimal odds; stake returned for push/void. No fees or bankroll simulation.",
                                 "ROI divides known net profit by settled stakes, including push/void.",
                                 f"Cumulative known profit ordered by {self.time_column}; unresolved bets remain visible."])
        if self.composition is not None:
            from .tickets import add_tickets
            if self.source is not None:
                result.tables['alternatives'] = previous.tables['alternatives'].copy()
            add_tickets(result, self.composition, context)
            ledger, metrics = result.tables['ledger'], result.tables['bet_metrics']
            result.tables['metrics'] = metrics
            result.artifacts[1] = Artifact('table', metrics, 'Ticket metrics')
        elif self.source is not None and 'tickets' in previous.tables:
            for name in ('tickets', 'ticket_legs', 'leg_ledger'):
                result.tables[name] = previous.tables[name].copy()
            result.notes.append('Accounting unit: composed tickets from the source; individual legs are not staked again.')
        if len(ledger):
            ledger[self.time_column] = pd.to_datetime(ledger[self.time_column], utc=True, errors="raise")
            if ledger[self.time_column].isna().any():
                raise ValueError("Bet profit timeline requires nonmissing ordering timestamps.")
            ledger = ledger.sort_values(self.time_column, kind="stable")
            ledger["cumulative_known_profit"] = ledger.groupby("bet", sort=False).profit.transform(lambda s: s.fillna(0).cumsum())
            result.tables["ledger"] = ledger
            go = _plotting()
            figure = go.Figure()
            for bet, rows in ledger.groupby("bet", sort=False):
                figure.add_trace(go.Scatter(x=rows[self.time_column], y=rows.cumulative_known_profit,
                                            mode="lines", name=bet))
            result.artifacts.append(Artifact("plotly", _style(figure, "Cumulative known profit",
                                                               self.time_column, "Profit (stake units)"), "Profit timeline"))
        return result
