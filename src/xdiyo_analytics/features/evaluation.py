"""Evaluate feature expressions using shared history eligibility and alignment."""

from numbers import Integral, Real

from .expressions import (
    EMA, H2H, ForAgainst, IsHome, Lag, NormalizedStanding,
    RollingMean, RollingStd, RollingZScore, Stat, MatchScore,
)
from .history import eligible_history_rows, league_season_team_counts
from .ratings import MatchResultGlicko, Rating, StatGlicko
from .league import League, LeaveOneOut, LeaguePopulation, reduce_league
from .warmup import WarmStart, SeededEMA, evaluate_warm_start
from .transitions import TransitionContext
from .composition import Constant, ARITHMETIC, operands, constant_frame, arithmetic_frame
from .contextual import RestDays, CalendarFeature, evaluate_context_features
from .spatial import Heatmap, RegionMass, heatmap_values, region_values, finalize_spatial
from .movement import TeamMovement, movement_records, statistical_context


def evaluate_features(history, features, *, group_by=("team_id", "competition_id"),
                      cutoffs=None, available_at=None, team_counts=None, ratings=None,
                      team_seasons=None, season_starts=None, keyed=False, heatmaps=None):
    """Evaluate a mapping of output names to expressions; preserve row/index order.

    Stat references observed values and needs a historical operator at the root;
    IsHome and NormalizedStanding are current-match context and can stand alone.
    Default Stat perspective is 'for'. None periods and 'both' perspectives
    expand into separate columns suffixed with the source identity. One-column
    expressions use exactly the requested output name.

    Heatmap reads exported points supplied through heatmaps=. Lag, rolling
    operators and EMA process its cells using the same temporal eligibility as
    statistics. Raw current-match Heatmap values cannot be prediction features.
    Their venue='same' restricts eligible history to the target venue before the
    match window; venue='all' keeps both. Spatial history is reduced in focal-team
    coordinates, then home-oriented final Away grids rotate by 180 degrees.
    RegionMass integrates source cells in the focal frame. Spatial descriptors
    in result.attrs['spatial_features'] travel through dataset assembly.

    Rolling windows count eligible matches, including matches with missing
    measures. Reductions use available finite values within that window; Lag
    never skips a missing measure to reach an older match. Std defaults to ddof=1.
    Z-score uses the most recent eligible value and INCLUDES it in its window;
    missing latest values, insufficient data and zero spread give NaN. EMA skips
    missing measures without decay and starts at the first observed value.

    Nested temporal expressions evaluate their child at each historical row's
    own configured cutoff, then apply the outer operator to those past outputs.
    H2H applies ordered team/opponent grouping to its wrapped expression and
    enclosing historical operators. No output uses the target's observed stats.

    Default grouping crosses seasons, combines venues and keeps competitions
    separate. See eligible_history_rows for explicit cutoff/availability options.
    Without available_at, earlier finished kickoffs are a retrospective proxy,
    not evidence of historical publication/completion time.

    Standings use distinct team membership per competition/season in the full
    input (or a supplied Series from league_season_team_counts). Missing position
    gives zero by default. Standings have no automatic prior or coverage dependency.

    MatchResultGlicko and StatGlicko build independent state streams once per
    source/configuration in this evaluation. Rating(name) reads a prebuilt
    RatingRun supplied through ratings={name: run}, including custom state fields.
    Rating nodes own their scope; group_by applies to lag/rolling history instead.
    Outputs are aligned training features as well as future-prediction inputs.

    League populations support pooled round/match/day windows and optional LOO.
    WarmStart explicitly enables SeededEMA or a rating transition policy for its
    child only. team_seasons supplies movement evidence; season_starts can set
    transition boundaries before opening fixtures. No wrapper means no warm-up.

    keyed=True replaces the output index with explicit match/team/side identifiers
    for dataset assembly. Values and row order are unchanged. Default False keeps
    the original history index. Keyed output can safely be shuffled before joining
    labels; its keys must remain attached when saving or transforming the frame.
    """
    import numpy as np
    import pandas as pd

    if not features:
        raise ValueError("Provide at least one named feature expression.")
    if not isinstance(keyed, bool):
        raise TypeError("keyed must be a boolean.")
    if isinstance(group_by, str):
        group_by = (group_by,)
    else:
        group_by = tuple(group_by)
    cache, histories, rating_cache = {}, {}, {}
    counts = team_counts
    population = None
    transitions = None
    stat_transitions = None
    movement_evidence = None

    def stat_context():
        nonlocal stat_transitions, movement_evidence
        if stat_transitions is None:
            base = TransitionContext(history, cutoffs=cutoffs, available_at=available_at,
                                     season_starts=season_starts)
            stat_transitions = statistical_context(base, team_seasons)
            movement_evidence = stat_transitions.movement_evidence
        return stat_transitions

    def transition_context():
        nonlocal transitions, population
        if population is None:
            population = LeaguePopulation(history, cutoffs=cutoffs, available_at=available_at)
        if transitions is None:
            transitions = TransitionContext(history, team_seasons=team_seasons,
                                            season_starts=season_starts, population=population)
        return transitions

    def history_rows(scope, venue='all'):
        if venue not in ('all', 'same'):
            raise ValueError('Historical venue must be all or same.')
        key = (scope, venue)
        if key not in histories:
            histories[key] = eligible_history_rows(
                history, group_by=(*group_by, 'side') if venue == 'same' else group_by, head_to_head=scope,
                cutoffs=cutoffs, available_at=available_at,
            )
        return histories[key]

    def positive(value, label):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{label} must be a positive integer.")

    def roles(side):
        choices = {"for": ("team",), "against": ("opponent",), "both": ("team", "opponent")}
        if side not in choices:
            raise ValueError("side must be 'for', 'against' or 'both'.")
        return choices[side]

    def known_reference(node):
        while isinstance(node, (H2H, WarmStart, RegionMass)):
            node = node.source
        if isinstance(node, (Stat, MatchScore, Heatmap, ForAgainst, League, LeaveOneOut)):
            raise ValueError("Z-score reference must be historical or known context.")
        if isinstance(node, ARITHMETIC):
            for child in operands(node):
                known_reference(child)

    def stat_values(node, side):
        metadata = history.attrs.get("stat_columns", {})
        columns = []
        for role in roles(side):
            columns.extend(name for name, info in metadata.items() if
                info["role"] == role and info["group_name"] == node.group and
                info["key"] == node.key and info["field"] == node.field and
                (node.period is None or info["period"] == node.period))
        if not columns:
            raise KeyError(f"Statistic is absent from the selected history: {node!r}")
        return history[columns].copy()

    def evaluate(node, h2h=False, rating_policy=None):
        nonlocal counts, population, movement_evidence
        cache_key = (node, h2h, rating_policy)
        if cache_key in cache:
            return cache[cache_key]
        if isinstance(node, TeamMovement):
            if movement_evidence is None:
                movement_evidence = movement_records(history, team_seasons)
                if any(pd.notna(record.get('previous_competition_id')) for record in movement_evidence.values()):
                    stat_context()  # Validate supplied predecessor identities for flags too.
            field = 'got_promoted' if node.movement == 'promoted' else 'got_demoted'
            values = [movement_evidence[key][field] for key in zip(
                history.competition_id.tolist(), history.season_id.tolist(), history.team_id.tolist())]
            result = (pd.DataFrame({f'was_{node.movement}': pd.array(values, dtype='Float64')}, index=history.index), h2h)
            result[0].attrs['movement_evidence'] = list(movement_evidence.values())
        elif isinstance(node, RestDays):
            kicks = history.kickoff_at
            values = [np.nan if len(rows) == 0 else
                      (kicks.iloc[i] - kicks.iloc[rows[-1]]).total_seconds() / 86400
                      for i, rows in enumerate(history_rows(h2h))]
            result = (pd.DataFrame({'rest_days': values}, index=history.index), h2h)
        elif isinstance(node, CalendarFeature):
            result = (evaluate_context_features(history, {'calendar': node}), h2h)
        elif isinstance(node, Constant):
            result = (constant_frame(node.value, history.index), h2h)
        elif isinstance(node, Real):
            result = (constant_frame(node, history.index), h2h)
        elif isinstance(node, ARITHMETIC):
            left, right = operands(node)
            a, a_scope = evaluate(left, h2h)
            b, b_scope = evaluate(right, h2h)
            if a_scope != b_scope:
                raise ValueError("Arithmetic operands must use the same H2H scope; wrap the combined expression in H2H.")
            result = (arithmetic_frame(node, a, b), a_scope)
        elif isinstance(node, WarmStart):
            if node.policy is None:
                result = evaluate(node.source, h2h)
            elif isinstance(node.policy, SeededEMA):
                result = evaluate_warm_start(node, transition_context() if node.policy.mode == 'legacy' else stat_context(), evaluate, history_rows,
                                             h2h=h2h, group_by=group_by)
            elif isinstance(node.source, (MatchResultGlicko, StatGlicko)):
                result = evaluate(node.source, h2h, node.policy)
            else:
                raise TypeError("WarmStart needs SeededEMA for rolling features or a transition adapter for a rating producer. Apply saved-rating transitions when building the run.")
        elif isinstance(node, H2H):
            result = evaluate(node.source, True)
        elif isinstance(node, Heatmap):
            result = (heatmap_values(history, heatmaps, node), h2h)
        elif isinstance(node, RegionMass):
            source, scope = evaluate(node.source, h2h)
            result = (region_values(source, node.region), scope)
        elif isinstance(node, Stat):
            result = (stat_values(node, "for"), h2h)
        elif isinstance(node, MatchScore):
            columns = ["goals_for" if role == "team" else "goals_against"
                       for role in roles(node.side)]
            # build_team_history retains native current scores and creates
            # missing values when their export fields are absent.
            result = (history[columns].copy(), h2h)
        elif isinstance(node, ForAgainst):
            if isinstance(node.source, (Heatmap, MatchScore)):
                from dataclasses import replace
                result = evaluate(replace(node.source, side=node.side), h2h)
            elif isinstance(node.source, Stat):
                result = (stat_values(node.source, node.side), h2h)
            else:
                raise TypeError("ForAgainst takes a Stat, MatchScore or Heatmap reference.")
        elif isinstance(node, (Rating, MatchResultGlicko, StatGlicko)):
            if h2h:
                raise ValueError("H2H rating streams are not implemented; use H2H with historical statistic operators.")
            if isinstance(node, Rating):
                if ratings is None or node.name not in ratings:
                    raise KeyError(f"Supply the named RatingRun through ratings: {node.name}")
                run = ratings[node.name]
            else:
                from ..ratings import build_ratings
                stat = node.source if isinstance(node, StatGlicko) else None
                if stat is not None and not isinstance(stat, Stat):
                    raise TypeError("StatGlicko takes a Stat reference.")
                higher = node.higher_is_better if isinstance(node, StatGlicko) else True
                definition = (stat, node.engine, node.scope, higher, rating_policy)
                if definition not in rating_cache:
                    rating_cache[definition] = build_ratings(
                        history, stat=stat, engine=node.engine, scope=node.scope,
                        higher_is_better=higher, available_at=available_at,
                        transition=rating_policy,
                        transition_context=transition_context() if rating_policy is not None else None,
                    )
                run = rating_cache[definition]
            result = (run.features(history, cutoffs=cutoffs, side=node.side, fields=node.fields), h2h)
        elif isinstance(node, IsHome):
            values = history["side"].map({"home": 1.0, "away": 0.0})
            result = (values.to_frame("is_home"), h2h)
        elif isinstance(node, NormalizedStanding):
            if counts is None:
                counts = league_season_team_counts(history)
            keys = ["competition_id", "season_id"]
            joined = history[keys].merge(
                counts.rename("team_count").reset_index(), on=keys,
                how="left", sort=False, validate="many_to_one",
            )
            size = joined["team_count"].to_numpy(dtype=float, na_value=np.nan)
            output = {}
            for role in roles(node.side):
                column = f"{role}_position"
                position = (history[column].to_numpy(dtype=float, na_value=np.nan)
                            if column in history else np.full(len(history), np.nan))
                values = np.full(len(history), float(node.missing_value))
                valid = np.isfinite(position) & np.isfinite(size) & (size > 1)
                valid &= (position >= 1) & (position <= size)
                values[valid] = 1 - (position[valid] - 1) / (size[valid] - 1)
                output[f"{role}_standing"] = values
            result = (pd.DataFrame(output, index=history.index), h2h)
        elif isinstance(node, (Lag, RollingMean, RollingStd, RollingZScore, EMA)):
            if node.venue not in ('all', 'same'):
                raise ValueError('Historical venue must be all or same.')
            if isinstance(node.source, (League, LeaveOneOut)):
                if node.venue != 'all':
                    raise ValueError('Same venue is a team-history option, not a league-population option.')
                if h2h:
                    raise ValueError("H2H cannot change a league population.")
                if population is None:
                    population = LeaguePopulation(history, cutoffs=cutoffs, available_at=available_at)
                if isinstance(node, RollingZScore) and node.reference is not None:
                    known_reference(node.reference)
                result = (reduce_league(node, population, evaluate), False)
                cache[cache_key] = result
                return result
            source, scope = evaluate(node.source, h2h)
            candidates = history_rows(scope, node.venue)
            if isinstance(node, Lag):
                positive(node.periods, "periods")
            else:
                positive(node.min_periods, "min_periods")
                if isinstance(node, EMA):
                    if not np.isfinite(node.span) or node.span < 1:
                        raise ValueError("EMA span must be finite and at least 1.")
                else:
                    positive(node.window, "window")
                    if node.min_periods > node.window:
                        raise ValueError("min_periods cannot exceed window.")
                if isinstance(node, (RollingStd, RollingZScore)):
                    if isinstance(node.ddof, bool) or not isinstance(node.ddof, Integral) or node.ddof < 0:
                        raise ValueError("ddof must be a nonnegative integer.")
            values = source.to_numpy(dtype=float, na_value=np.nan, copy=True)
            values[~np.isfinite(values)] = np.nan
            output = np.full(values.shape, np.nan)
            reference = None
            if isinstance(node, RollingZScore) and node.reference is not None:
                known_reference(node.reference)
                reference, _ = evaluate(node.reference, h2h)
                if reference.shape[1] != source.shape[1]:
                    raise ValueError("Z-score reference must match the source column count.")
                reference = reference.to_numpy(dtype=float, na_value=np.nan)
            for row, positions in enumerate(candidates):
                if isinstance(node, Lag):
                    if len(positions) >= node.periods:
                        output[row] = values[positions[-node.periods]]
                    continue
                if not len(positions):
                    continue
                sample = values[positions if isinstance(node, EMA) else positions[-node.window:]]
                for col in range(values.shape[1]):
                    series = sample[:, col]
                    observed = series[np.isfinite(series)]
                    if len(observed) < node.min_periods:
                        continue
                    if isinstance(node, EMA):
                        alpha = 2 / (node.span + 1)
                        mean = observed[0]
                        for value in observed[1:]:
                            mean = alpha * value + (1 - alpha) * mean
                        output[row, col] = mean
                    elif isinstance(node, RollingMean):
                        output[row, col] = observed.mean()
                    elif len(observed) > node.ddof:
                        std = observed.std(ddof=node.ddof)
                        if isinstance(node, RollingStd):
                            output[row, col] = std
                        else:
                            value = series[-1] if reference is None else reference[row, col]
                            if std > 0 and np.isfinite(value):
                                output[row, col] = (value - observed.mean()) / std
            result = (pd.DataFrame(output, index=history.index, columns=source.columns), scope)
            if source.attrs.get('spatial_features'):
                from copy import deepcopy
                specs = deepcopy(source.attrs['spatial_features'])
                for spec in specs.values():
                    spec['operators'].append(dict(operator=type(node).__name__, venue=node.venue,
                                                  window=getattr(node, 'window', None),
                                                  periods=getattr(node, 'periods', None), span=getattr(node, 'span', None)))
                result[0].attrs['spatial_features'] = specs
        else:
            raise TypeError(f"Unsupported feature expression: {type(node).__name__}")
        cache[cache_key] = result
        return result

    def prediction_safe(node):
        if isinstance(node, (H2H, WarmStart, RegionMass)):
            return prediction_safe(node.source)
        if isinstance(node, ARITHMETIC):
            return all(prediction_safe(child) for child in operands(node))
        return not isinstance(node, (Stat, MatchScore, Heatmap, ForAgainst, League, LeaveOneOut))

    outputs, spatial_metadata, warm_audits = [], {}, {}
    for name, node in features.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Feature names must be nonempty strings.")
        if not prediction_safe(node):
            raise ValueError("Observed Stat values need Lag, a rolling operator or EMA before prediction; the same applies to MatchScore and Heatmap sources.")
        frame, _ = evaluate(node)
        if frame.attrs.get('warm_start_audit') is not None:
            warm_audits[name] = frame.attrs['warm_start_audit']
        else:
            # Arithmetic does not carry DataFrame attrs. Keep provenance for
            # warmed descendants, including descendants already in the cache.
            from dataclasses import fields, is_dataclass
            descendants = set()
            def visit(value):
                if not is_dataclass(value) or value in descendants:
                    return
                descendants.add(value)
                for field in fields(value):
                    visit(getattr(value, field.name))
            visit(node)
            child_audits = [audit for (child, _, _), (value, _) in cache.items()
                            if child in descendants for audit in value.attrs.get('warm_start_audit', [])]
            if child_audits:
                warm_audits[name] = child_audits
        names = [name] if frame.shape[1] == 1 else [f"{name}::{col}" for col in frame.columns]
        frame, metadata = finalize_spatial(frame, history, names, name)
        spatial_metadata.update(metadata)
        outputs.append(frame)
    result = pd.concat(outputs, axis=1)
    if not result.columns.is_unique:
        raise ValueError("Feature names produce duplicate output columns.")
    result.attrs = {
        "features": {name: repr(node) for name, node in features.items()},
        "group_by": group_by,
        "availability": "earlier_finished_kickoff_proxy" if available_at is None else "explicit",
        "spatial_features": spatial_metadata,
        "warm_start_audit": warm_audits,
        "movement_evidence": [] if movement_evidence is None else list(movement_evidence.values()),
    }
    if stat_transitions is not None or movement_evidence is not None:
        import hashlib
        digest = hashlib.sha256(pd.util.hash_pandas_object(history, index=True).to_numpy().tobytes())
        digest.update(repr(movement_evidence).encode())
        for (source_node, source_scope, _), (source_frame, _) in cache.items():
            if 'warm_start_source_hash' in source_frame.attrs:
                digest.update(repr((source_node, source_scope, source_frame.attrs['warm_start_source_hash'])).encode())
        if stat_transitions is not None:
            digest.update(repr(stat_transitions.records).encode())
            for times in (stat_transitions.population.cutoff, stat_transitions.population.available):
                digest.update(pd.util.hash_pandas_object(times, index=True).to_numpy().tobytes())
        result.attrs['warm_start_input_hash'] = digest.hexdigest()
    if keyed:
        keys = [name for name in ("source_league", "source_season", "competition_id", "season_id", "event_id") if name in history]
        keys += ["team_id", "side"]
        if "event_id" not in keys:
            raise KeyError("Keyed features require event_id.")
        if history[keys].isna().any().any() or history.duplicated([name for name in keys if name != "side"]).any():
            raise ValueError("Keyed features need unique, nonmissing match/team identifiers.")
        if not history["side"].isin(["home", "away"]).all():
            raise ValueError("Keyed feature sides must be home or away.")
        result.index = pd.MultiIndex.from_frame(history[keys])
        result.attrs["identity_columns"] = tuple(keys)
    return result
