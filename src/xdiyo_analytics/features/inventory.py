"""Compact, count-reconciled inventories for named preset feature columns."""
from collections import Counter
import hashlib
import json
import re


def summarize_feature_columns(columns):
    """Group the preset's columns without emitting thousands of individual rows.

    Counts describe actual input columns, including separate home/away copies.
    Unrecognized names remain visible under an explicit unclassified group.
    The ordered-name digest distinguishes schemas with different column order.
    """
    columns = list(columns)
    if len(columns) != len(set(columns)):
        raise ValueError('Feature inventory requires unique column names.')
    groups = {}
    for column in columns:
        text = column
        stage, venue = 'direct', None
        for prefix in ('sum', 'difference', 'trend', 'calendar', 'league'):
            if text.startswith(prefix + '::'):
                stage, text = prefix, text[len(prefix) + 2:]
                break
        if text.startswith(('home::', 'away::')):
            venue, text = text.split('::', 1)
        warmed = text.startswith('warm::')
        if warmed:
            text = text[6:]
        variant = 'warmed' if warmed else 'baseline'
        family, operation, window, period, stat, role, field = 'unclassified', text, None, None, None, None, None
        match = re.fullmatch(r'(ALL|1ST|2ND)_(.+)_([^_]+)_(for|against)_(lag|mean|std|z|ema)(\d+)(?:_vs_(\d+))?', text)
        if match:
            period, group, key, role, operation, window, long = match.groups()
            family, stat = 'statistic', f'{group}/{key}'
            if long:
                operation, window = 'mean_difference', f'{window}-{long}'
        elif text.startswith('h2h_corners_'):
            family, stat, period = 'H2H', 'Match overview/cornerKicks', 'ALL'
            role, operation, window = re.fullmatch(r'h2h_corners_(for|against)_(mean|std|z)(\d+)', text).groups()
        elif text.startswith(('loo_corners_', 'league_corners_')):
            family, operation, window = re.fullmatch(r'(loo|league)_corners_(mean|std|z)(\d+)', text).groups()
            stat, period = 'Match overview/cornerKicks', 'ALL'
        elif 'glicko' in text:
            family, operation, field = 'rating', text.split('::')[0], text.split('::')[-1]
        elif text == 'standing':
            family, operation = 'standings', 'normalized_position'
        elif text == 'rest_days':
            family, operation = 'context', 'rest_days'
        elif stage == 'calendar':
            family, operation = 'context', text
        elif stage == 'league':
            family, operation, stat = 'identity', 'one_hot_league', text
        key = (stage, variant, family, operation, window, period)
        group = groups.setdefault(key, dict(stage=stage, variant=variant, family=family,
            operation=operation, window=window, period=period, count=0,
            statistics=set(), venues=set(), roles=set(), fields=set()))
        group['count'] += 1
        for name, value in (('statistics', stat), ('venues', venue), ('roles', role), ('fields', field)):
            if value is not None:
                group[name].add(value)
    rows = [{key: sorted(value) if isinstance(value, set) else value for key, value in group.items()}
            for group in groups.values()]
    return dict(total_columns=len(columns),
        ordered_columns_sha256=hashlib.sha256(json.dumps(columns).encode()).hexdigest(),
        stage_counts=dict(Counter({stage: sum(g['count'] for g in rows if g['stage'] == stage)
                                 for stage in dict.fromkeys(g['stage'] for g in rows)})),
        groups=rows, unclassified_columns=sum(g['count'] for g in rows if g['family'] == 'unclassified'))
