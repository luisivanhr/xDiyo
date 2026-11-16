"""Pinned native naming, rule eligibility and reviewed administrative outcomes.

Display names never replace stable storage keys or provider identities. All
overlays live in the selected manifest, so old selection pins cannot acquire
new semantics without an explicit refresh.
"""
import json

POLICY = 'sofascore_native_integration_v1'


def validate_integration(value, scope):
    if (value.get('policy') != POLICY or value.get('competition', {}).get('id') != scope['league_id']
            or value.get('season', {}).get('id') != scope['season_id']):
        raise ValueError('Native integration name/ID scope mismatch')
    for item in (value['competition'], value['season']):
        if type(item.get('id')) is not int or not isinstance(item.get('name'), str) or not item['name']:
            raise ValueError('Verified SofaScore display names are required')
    if not isinstance(value['season'].get('year'), str) or not value['season']['year']:
        raise ValueError('Captured SofaScore season label is required')
    if type(value['rules'].get('enforce')) is not bool:
        raise ValueError('Native rule enforcement metadata must be explicit')
    for key in ('standings', 'countdown'):
        rule = value['rules'][key]
        if rule.get('status') not in ('supported', 'unsupported', 'legacy_verified'):
            raise ValueError('Native competition rule status must be explicit')
        if rule['status'] == 'unsupported' and not rule.get('reason'):
            raise ValueError('Unsupported native rule needs a reason')
        if key == 'countdown' and rule['status'] == 'supported':
            if (rule.get('format') not in ('single','double','triple','quadruple','scotland','japan_transition')
                    or type(rule.get('total_rounds')) is not int or rule['total_rounds'] < 2):
                raise ValueError('Invalid verified native countdown configuration')
    for key, team in value.get('teams', {}).items():
        if str(team.get('id')) != key or not isinstance(team.get('name'), str) or not team['name']:
            raise ValueError('Native team display name/ID mismatch')


def attach_integration(matches, value):
    import pandas as pd
    result = matches.copy()
    for col, name in [('competition_name', value['competition']['name']),
                      ('season_name', value['season']['name']), ('season_label', value['season']['year'])]:
        if col in result:
            raise ValueError(f'Reserved native metadata column: {col}')
        result[col] = pd.Series(name, index=result.index, dtype='string')
    for side in ('home', 'away'):
        result[side+'_display_name'] = pd.Series(
            [value['teams'].get(str(i), {}).get('name', raw)
             for i, raw in zip(result[side+'_id'], result[side+'_name'])], index=result.index, dtype='string')
    result['tournament_name'] = pd.Series([value.get('tournaments', {}).get(str(i), {}).get('name')
        for i in result.tournament_id], index=result.index, dtype='string')
    if value['rules']['enforce']:
        for kind in ('countdown', 'standings'):
            rule = value['rules'][kind]
            result['native_'+kind+'_rule'] = json.dumps(rule, sort_keys=True, separators=(',', ':'))
    for review in value.get('rulings', []):
        mask = result.event_id.eq(review['event_id'])
        if mask.sum() != 1:
            raise ValueError(f"Reviewed administrative fixture missing or duplicated: {review['event_id']}")
        row = result.loc[mask].iloc[0]
        for field in ('home_id', 'away_id', 'kickoff_utc'):
            if row[field] != review[field]:
                raise ValueError(f"Reviewed administrative fixture identity changed: {review['event_id']} {field}")
        if 'provider_is_awarded' not in result:
            result['provider_is_awarded'] = result.is_awarded.copy()
        if review.get('is_awarded') is True:
            result.loc[mask, 'is_awarded'] = True
        for col in ('play_status', 'administrative_home_score', 'administrative_away_score', 'ruling_date'):
            if col in review:
                if col not in result:
                    result[col] = pd.Series(pd.NA, index=result.index, dtype='object')
                result.loc[mask, col] = review[col]
        if 'result_review_json' not in result:
            result['result_review_json'] = pd.Series(pd.NA, index=result.index, dtype='string')
        result.loc[mask, 'result_review_json'] = json.dumps(review, sort_keys=True)
    return result


def native_rule(frame, kind):
    """Return one exact per-season rule; mixed/unsupported scopes fail closed."""
    column = 'native_'+kind+'_rule'
    if column not in frame:
        return None
    if frame[column].isna().any():
        raise ValueError(f'Missing native {kind} rule for some fixtures')
    values = frame[column].unique()
    if len(values) != 1:
        raise ValueError(f'Mixed native {kind} scopes; select one verified independent calendar/table')
    rule = json.loads(values[0])
    if rule['status'] == 'unsupported':
        raise ValueError(f"Unsupported native {kind} rule: {rule['reason']}")
    return rule


def supervised_native_rows(matches, *, cutoff, available_at):
    """Finished/labeled rows with explicit release evidence at a UTC cutoff.

    Unknown releases are ineligible. This helper does not infer publication
    times from kickoff, provider capture time or an administrative ruling date.
    Scores remain unchanged; exclusions are returned as a mask.
    """
    import pandas as pd
    import numpy as np
    from ..features.history import aligned_times, availability_times
    stamp = pd.Timestamp(cutoff)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError('Training cutoff must be a known timezone-aware timestamp')
    frame = matches.copy()
    frame['kickoff_at'] = pd.to_datetime(frame.kickoff_utc, unit='s', utc=True)
    release = availability_times(frame, aligned_times(frame, available_at, default='result_available_at'))
    complete = frame[['home_score_current', 'away_score_current']].notna().all(axis=1)
    complete &= np.isfinite(frame[['home_score_current','away_score_current']].to_numpy(dtype=float,na_value=np.nan)).all(axis=1)
    mask = frame.status.eq('finished') & complete & frame.kickoff_at.lt(stamp) & release.notna() & release.le(stamp)
    if 'is_awarded' in frame:
        mask &= ~frame.is_awarded.fillna(False)
    if 'play_status' in frame:
        mask &= ~frame.play_status.fillna('').str.contains('abandon|unplayed|annul|cancel')
    return mask.fillna(False)
