"""Prediction-only bet selection. Outcomes are consulted later, at settlement."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..labels import BetOption, Outcome
from .betting import BetSpec, _aligned
from .probabilities import bet_probabilities, negative_binomial_bet_probabilities
from ..odds.selection import OddsSeries


@dataclass(frozen=True)
class BetOffer:
    """An offered BetOption with optional decimal odds and a fixed unit stake."""
    option: BetOption
    odds: object = None
    stake: object = 1.0


@dataclass(frozen=True)
class TightestLine:
    """Tightest O/U line within a fixed probability loss of the safest same-side offer.

    Under chooses the lowest qualifying line; Over chooses the highest. If both
    sides qualify, the higher win probability wins; ties follow offer order.
    min_ev optionally requires expected profit per stake unit and therefore odds.
    """
    min_probability: float = 0.9
    max_probability_loss: float = 0.03
    min_ev: float | None = None


@dataclass(frozen=True)
class HighestExpectedProfit:
    """Choose the qualifying option with greatest expected net profit per unit.

    Decimal odds are required. Ties follow offer order; no qualifying offer means
    no bet. Push returns the stake and contributes zero expected net profit.
    """
    min_probability: float = 0.0
    min_ev: float = 0.0


@dataclass(frozen=True)
class BinaryDrawThreshold:
    """Retain a draw offer when P(non-draw) <= the frozen threshold.

    The predicted label must be this exact draw BetOption, with 1=draw and
    0=non-draw. None selects every offered event (baseline), even when its
    probability is missing. Finite valid prices remain mandatory. No EV cutoff,
    ranking, tuning or independence assumption is applied.
    """
    max_non_draw_probability: float | None = None


def _option_probabilities(context, option, target, output, *, binary_draw=False):
    frame = context.predictions.get(output)
    if frame is None:
        raise ValueError(f"Prediction output {output!r} is unavailable. Retain probabilities when training.")
    if not frame.index.equals(context.y.index):
        raise ValueError("Bet probabilities must align with prediction row identities.")
    if output == "count_distribution":
        if binary_draw:
            raise ValueError('BinaryDrawThreshold requires binary class probabilities.')
        if context.pooling == 'mean':
            # A mixture of NB predictions is not an NB at the average parameters.
            # Resolve each retained component first, then mix probabilities.
            components = []
            positions = context.row_positions
            for fold in context.fold_results.values():
                if output not in fold.predictions:
                    raise ValueError('Mean NB pooling requires each original fold count distribution; use first/last pooling instead.')
                eligible = fold.score_positions if context.partition == 'score' else fold.test_positions
                eligible = pd.Index(eligible).intersection(positions, sort=False)
                values = fold.predictions[output].loc[eligible].xs(target, axis=1, level=0)
                components.append(negative_binomial_bet_probabilities(values['mean'], values['dispersion'], option))
            if not components:
                raise ValueError('Mean NB pooling requires original fold count distributions; use first/last pooling instead.')
            combined = pd.concat(components).groupby(level=0, sort=False).mean().reindex(positions)
            if combined.isna().any().any():
                raise ValueError('Mean NB pooling is missing an original distribution for a selected row.')
            combined.index = frame.index
            return combined
        values = frame.xs(target, axis=1, level=0)
        return negative_binomial_bet_probabilities(values['mean'], values['dispersion'], option)
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Class probabilities need named (target, class) columns.")
    values = frame.xs(target, axis=1, level=0)
    result = pd.DataFrame(np.nan, index=frame.index, columns=['p_win', 'p_push', 'p_loss'])
    # Concatenating different folds introduces absent-class NaNs. Recover each
    # fold's exact declared support; do not fill arbitrary missing cells with 0.
    for fold_id in frame.index.get_level_values('fold_id').unique():
        mask = frame.index.get_level_values('fold_id') == fold_id
        subset = values.loc[mask]
        fold = context.fold_results.get(fold_id)
        if fold is not None and output in fold.predictions:
            support = fold.predictions[output].xs(target, axis=1, level=0).columns
            subset = subset.loc[:, support]
        valid = subset.notna().all(axis=1)
        if valid.any():
            if binary_draw:
                valid_rows = subset.loc[valid]
                p = valid_rows.to_numpy(dtype=float)
                if (not valid_rows.columns.is_unique or not len(valid_rows.columns) or
                        not set(valid_rows.columns).issubset({0, 1}) or not np.isfinite(p).all() or
                        (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)):
                    raise ValueError('Binary draw probabilities need numeric 0=non-draw, 1=draw classes summing to one.')
                mapped = pd.DataFrame({'p_win': valid_rows.get(1, 0.), 'p_push': 0.,
                                       'p_loss': valid_rows.get(0, 0.)}, index=valid_rows.index)
            else:
                mapped = bet_probabilities(subset.loc[valid], option)
            result.loc[subset.index[valid]] = mapped.to_numpy()
    return result


def prepare_bets(context, offers, policy, *, target=None, output='predict_proba', default_odds=None):
    """Return (named BetSpecs, alternatives table) without reading outcomes.

    One reporter represents one target/market. At most one offer is selected per
    prediction row. For team layouts that is one per team. Unknown probability
    rows are skipped explicitly. Odds are optional unless the policy uses EV.
    """
    binary = isinstance(policy, BinaryDrawThreshold)
    if not isinstance(policy, (TightestLine, HighestExpectedProfit, BinaryDrawThreshold)):
        raise TypeError("Choose TightestLine, HighestExpectedProfit or BinaryDrawThreshold.")
    if binary and policy.max_non_draw_probability is not None and (
            isinstance(policy.max_non_draw_probability, bool) or not np.isfinite(policy.max_non_draw_probability)
            or not 0 <= policy.max_non_draw_probability <= 1):
        raise ValueError('max_non_draw_probability must be in [0, 1], or None for the unfiltered baseline.')
    if not binary and (not np.isfinite(policy.min_probability) or not 0 <= policy.min_probability <= 1):
        raise ValueError("min_probability must be between 0 and 1.")
    if isinstance(policy, TightestLine) and (not np.isfinite(policy.max_probability_loss) or not 0 <= policy.max_probability_loss <= 1):
        raise ValueError("max_probability_loss must be between 0 and 1.")
    min_ev = None if binary else policy.min_ev
    if min_ev is not None and not np.isfinite(min_ev):
        raise ValueError("min_ev must be finite or None.")
    if isinstance(policy, HighestExpectedProfit) and policy.min_ev is None:
        raise ValueError("HighestExpectedProfit requires a finite min_ev and decimal odds.")
    target = target if target is not None else (context.y.columns[0] if len(context.y.columns) == 1 else None)
    if target not in context.y:
        raise ValueError("Choose one predicted target for the betting market.")
    if not offers:
        raise ValueError("Add at least one offered bet option.")
    offers = {name: offer if isinstance(offer, BetOffer) else BetOffer(**offer) for name, offer in offers.items()}
    source = next(iter(offers.values())).option.source
    definition = context.definitions.get('label')
    if binary:
        option = next(iter(offers.values())).option
        if (len(offers) != 1 or not isinstance(source, Outcome) or option.selection != 'draw'
                or definition != option):
            raise ValueError('BinaryDrawThreshold needs one draw offer and the exact draw BetOption as the predicted label.')
    elif definition is not None and definition != source:
        raise ValueError("Bet options must use the same label source as the predicted target.")
    rows = []
    for order, (name, offer) in enumerate(offers.items()):
        if not isinstance(name, str) or not name or not isinstance(offer.option, BetOption) or offer.option.source != source:
            raise ValueError("Name each offer and use one common target source per reporter.")
        if isinstance(policy, TightestLine) and offer.option.selection not in {'under', 'over'}:
            raise ValueError("TightestLine supports over/under offers; use HighestExpectedProfit for other options.")
        probabilities = _option_probabilities(context, offer.option, target, output, binary_draw=binary)
        quote = default_odds if offer.odds is None else offer.odds
        quote_metadata = None
        if isinstance(quote, OddsSeries):
            quote.validate_option(offer.option)
            quote_metadata = quote.resolve(context)
            odds = quote_metadata.decimal_odds.astype(float)
        else:
            odds = _aligned(np.nan if quote is None else quote, context, context.identity_columns, 'odds').astype(float)
        stake = _aligned(offer.stake, context, context.identity_columns, 'stake').astype(float)
        if (odds.notna() & (~np.isfinite(odds) | odds.le(1))).any():
            raise ValueError("Decimal odds must be finite and greater than 1.")
        if (~np.isfinite(stake) | stake.lt(0)).any():
            raise ValueError("Stakes must be finite and nonnegative.")
        table = probabilities.assign(bet=name, order=order, selection=offer.option.selection,
                                     line=offer.option.line, odds=odds, stake=stake)
        table['description'] = offer.option.selection.title() + (f' {offer.option.line:g}' if offer.option.line is not None else '')
        table['expected_profit'] = table.p_win * (odds - 1) - table.p_loss
        table['probability_abstention'] = (False if binary and policy.max_non_draw_probability is None
                                          else table.p_win.isna())
        table['eligible'] = (True if policy.max_non_draw_probability is None else
                             table.p_loss.le(policy.max_non_draw_probability)) if binary else table.p_win.ge(policy.min_probability)
        table['reason'] = np.where(table.p_win.isna(), 'Probability unavailable',
                                   np.where(table.eligible, 'Eligible alternative', 'Below minimum probability'))
        if binary:
            table['reason'] = np.where(table.eligible, 'Eligible alternative',
                                      np.where(table.p_loss.isna(), 'Probability unavailable', 'Above non-draw threshold'))
            table.loc[odds.isna(), 'reason'] = 'Missing odds'
            table['eligible'] &= odds.notna()
        if min_ev is not None:
            accepted = table.expected_profit.ge(min_ev)
            table.loc[table.eligible & ~accepted, 'reason'] = 'Missing odds or below minimum expected profit'
            table['eligible'] &= accepted
        if quote_metadata is not None:
            for column in quote_metadata.columns.difference(['decimal_odds']):
                table[column] = quote_metadata[column]
            table.loc[odds.isna(), 'reason'] = quote_metadata.loc[odds.isna(), 'quote_status']
            table['eligible'] &= odds.notna()
        rows.append(table.reset_index())
    alternatives = pd.concat(rows, ignore_index=True)
    alternatives['take'] = False
    for _, group in alternatives.groupby(['fold_id', 'row_position'], sort=False):
        eligible = group.loc[group.eligible]
        if isinstance(policy, TightestLine):
            finalists = []
            for side, offered in group.groupby('selection', sort=False):
                best_probability = offered.p_win.max()
                candidates = eligible.loc[eligible.selection.eq(side) & eligible.p_win.ge(best_probability-policy.max_probability_loss)]
                excluded = eligible.index[eligible.selection.eq(side)].difference(candidates.index)
                alternatives.loc[excluded, 'reason'] = 'Exceeds allowed probability loss'
                if len(candidates):
                    finalists.append(candidates.sort_values(['line', 'order'], ascending=[side == 'under', True]).index[0])
            ranked = group.loc[finalists].sort_values(['p_win', 'order'], ascending=[False, True])
        elif binary:
            ranked = eligible.sort_values('order', kind='stable')
        else:
            ranked = eligible.sort_values(['expected_profit', 'order'], ascending=[False, True])
        if len(ranked):
            winner = ranked.index[0]
            alternatives.loc[winner, ['take', 'reason']] = [True, 'Selected']
    specs = {}
    for name, offer in offers.items():
        selected = alternatives.loc[alternatives.bet.eq(name)].set_index(['fold_id', 'row_position']).reindex(context.y.index)
        specs[name] = BetSpec(offer.option, offer.odds if isinstance(offer.odds, OddsSeries) else selected.odds, selected['take'].astype(bool), selected.stake,
                              policy=f'{policy!r}; output={output}; default_odds={default_odds}')
    return specs, alternatives
