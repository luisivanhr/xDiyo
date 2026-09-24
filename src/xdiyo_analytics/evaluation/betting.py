"""Identity-aligned, flat-stake bet accounting using existing BetOption labels."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from ..labels import BetOption, create_labels
from .metrics import _sample_hash


@dataclass(frozen=True)
class BetSpec:
    """One offered option with decimal odds, explicit decisions and stakes.

    odds/stake accept scalars or Series. take accepts a boolean scalar, Series,
    or callable(context) returning a Series. Series must have the current
    (fold_id,row_position) index, or a unique MultiIndex named by the option's
    observation identity columns. No implicit row-order matching of arrays.
    stake defaults to 1 unit. policy names the externally supplied decision and
    staking rule for comparison; it is not inferred from realized outcomes.
    """
    option: BetOption
    odds: object
    take: object
    stake: object = 1.0
    policy: str = "provided_decisions_fixed_stake"


def _aligned(value, context, keys, name):
    value = value(context) if callable(value) else value
    if np.isscalar(value):
        return pd.Series(value, index=context.y.index, name=name)
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a scalar or an identity-indexed Series.")
    if value.index.equals(context.y.index):
        return value.copy()
    if isinstance(value.index, pd.MultiIndex) and list(value.index.names) == list(keys) and value.index.is_unique:
        query = pd.MultiIndex.from_frame(context.metadata[list(keys)])
        result = value.reindex(query)
        result.index = context.y.index
        return result
    raise ValueError(f"{name} Series must match prediction occurrences or exact option identities.")


def evaluate_bets(context, bets, *, labels=None, history=None):
    """Return (ledger, metrics) without plots or fitting.

    bets maps names to BetSpec or dictionaries accepted by BetSpec. Supply either
    already computed LabelData by the same names or history for create_labels().
    Each option must produce one settlement column. Both match and team_match
    layouts are supported; the label unit must equal the prediction layout.
    Team bets join on team identity as well as match identity, so opposite teams'
    bets remain distinct. No implicit conversion between layouts is performed.
    Settlement joins use explicit identities; missing settlements stay unresolved.

    Decimal odds >1, nonnegative finite stakes and explicit boolean decisions
    are required for placed bets. Missing quotes/stakes remain unresolved; invalid
    finite values raise. Wins pay stake*odds; losses pay zero; push/void return
    stake. No fees, bankroll limits, dynamic staking, quarter lines or parlays are
    simulated. ROI = known net profit / settled stakes, INCLUDING push/void.
    Profit/ROI have partial status while placed bets remain unresolved. No bets
    means profit=0 and ROI undefined. Repeated match options across folds require
    per-fold reporting or an explicit unique-row pooling policy upstream.
    """
    if (labels is None) == (history is None):
        raise ValueError("Supply exactly one of labels or history for bet settlement.")
    specs = {name: value if isinstance(value, BetSpec) else BetSpec(**value) for name, value in bets.items()}
    if any(not isinstance(name, str) or not name or not isinstance(spec.option, BetOption) for name, spec in specs.items()):
        raise ValueError("Bets need nonempty names and existing BetOption expressions.")
    if not specs:
        return pd.DataFrame(), pd.DataFrame()
    if history is not None:
        labels = create_labels(history, {name: spec.option for name, spec in specs.items()})
    if context.metadata.duplicated(list(context.identity_columns)).any():
        raise ValueError("Bet accounting needs unique observations; use per-fold or explicit first/last/mean pooling.")
    ledgers, metrics = [], []
    for name, spec in specs.items():
        label = labels[name]
        if label.unit != context.layout or label.definition != spec.option:
            raise ValueError("Bet settlements must match the requested option and prediction layout.")
        if label.settlement is None or len(label.settlement.columns) != 1:
            raise ValueError("Each bet needs one settlement column; select one concrete statistic/period.")
        if not label.settlement.index.equals(label.metadata.index):
            raise ValueError("Label settlements and metadata must remain aligned.")
        keys = tuple(label.identity_columns)
        source = pd.MultiIndex.from_frame(label.metadata[list(keys)])
        if not source.is_unique:
            raise ValueError("Bet settlement identities must be unique.")
        lookup = pd.Series(label.settlement.iloc[:, 0].to_numpy(), index=source)
        query = pd.MultiIndex.from_frame(context.metadata[list(keys)])
        settlement = lookup.reindex(query).fillna("missing")
        settlement.index = context.y.index
        if not settlement.isin(["win", "loss", "push", "void", "missing"]).all():
            raise ValueError("Unknown bet settlement category.")
        take = _aligned(spec.take, context, keys, "take")
        if take.isna().any() or not take.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("take must supply an explicit boolean decision for every observation.")
        take = take.astype(bool)
        odds = pd.to_numeric(_aligned(spec.odds, context, keys, "odds"), errors="raise").astype(float)
        stakes = pd.to_numeric(_aligned(spec.stake, context, keys, "stake"), errors="raise").astype(float)
        if ((take & odds.notna() & (~np.isfinite(odds) | (odds <= 1))).any() or
                (take & stakes.notna() & (~np.isfinite(stakes) | (stakes < 0))).any()):
            raise ValueError("Placed bets need finite decimal odds >1 and finite nonnegative stakes.")
        settled = take & settlement.ne("missing") & odds.notna() & stakes.notna()
        payout = pd.Series(np.nan, index=context.y.index)
        payout.loc[~take] = 0.0
        payout.loc[settled & settlement.eq("loss")] = 0.0
        payout.loc[settled & settlement.eq("win")] = stakes*odds
        payout.loc[settled & settlement.isin(["push", "void"])] = stakes
        actual_stake = stakes.where(take, 0.0)
        profit = payout-actual_stake
        ledger = context.metadata.copy()
        ledger["bet"] = name
        ledger["odds"] = odds
        ledger["take"] = take
        ledger["stake"] = actual_stake
        ledger["settlement"] = settlement
        ledger["accounting_status"] = np.where(~take, "not_placed", np.where(settled, "settled", "unresolved"))
        ledger["payout"] = payout
        ledger["profit"] = profit
        ledgers.append(ledger.reset_index())
        total_profit = float(profit.loc[settled].sum())
        settled_stakes = float(actual_stake.loc[settled].sum())
        pending = int((take & ~settled).sum())
        values = {"profit": total_profit, "roi": total_profit/settled_stakes if settled_stakes > 0 else np.nan,
                  "bets_placed": int(take.sum()), "settled_bets": int(settled.sum()), "unresolved_bets": pending,
                  "settled_stakes": settled_stakes, "payout": float(payout.loc[settled].sum())}
        for outcome in ("win", "loss", "push", "void"):
            values[outcome+"_count"] = int((settled & settlement.eq(outcome)).sum())
        # Compare offered opportunities/quotes/settlements, not model decisions.
        # The caller's policy identifier states whether decision/staking rules agree.
        comparison = context.metadata.copy()
        comparison["__offered_odds__"] = odds
        sample_hash = _sample_hash(settlement, comparison, pd.Series(True, index=context.y.index))
        parameters = json.dumps({"option": repr(spec.option), "policy": spec.policy,
                                 "stake": float(spec.stake) if np.isscalar(spec.stake) else "supplied_by_policy",
                                 "roi_denominator": "settled_stakes_including_push_void", "fees": 0}, sort_keys=True)
        for key, value in values.items():
            status = "partial" if pending and key in {"profit", "roi"} else "ok"
            if not np.isfinite(value):
                status = "undefined"
            metrics.append(dict(metric=key, calculation=key, target=name, output="bets", value=value,
                                direction="maximize" if key in {"profit", "roi"} else None,
                                n=int(settled.sum()), n_total=len(context.y), n_missing=pending, status=status,
                                parameters=parameters, sample_hash=sample_hash))
    return pd.concat(ledgers, ignore_index=True), pd.DataFrame(metrics)
