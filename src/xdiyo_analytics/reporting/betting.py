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

    labels or history is required, exclusively. time_column defaults to kickoff_at:
    cumulative profit is an outcome summary in that order, NOT a cash-flow or
    bankroll simulation. Supply actual settlement times for settlement ordering.
    Unresolved profits remain missing; cumulative values sum known settlements.
    """
    bets: dict
    labels: object = None
    history: object = None
    time_column: str = "kickoff_at"

    def run(self, context):
        ledger, metrics = evaluate_bets(context, self.bets, labels=self.labels, history=self.history)
        result = StudyResult("Bet performance", [Artifact("table", metrics, "Bet metrics")],
                             {"metrics": metrics, "ledger": ledger}, [
                                 "Decimal odds; stake returned for push/void. No fees or bankroll simulation.",
                                 "ROI divides known net profit by settled stakes, including push/void.",
                                 f"Cumulative known profit ordered by {self.time_column}; unresolved bets remain visible."])
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
