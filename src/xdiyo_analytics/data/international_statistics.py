"""Lossless, explicit international statistic names in native team coordinates.

No values, missingness, identities, units or availability are inferred here.
The shared mapping is also used by the existing catalogue workbook importer.
"""
from copy import deepcopy
import re
from urllib.parse import quote

from .statistics_backfill import STATISTICS, PERIODS

STATISTIC_NAMING_POLICY = 'international_native_statistic_names_v1'


def international_statistic_mapping(perspective):
    """Return all 60 source-column -> native-column mappings for one team side.

    ``perspective`` identifies the receiving team, not the statistic's source
    side. Away-perspective rows therefore exchange team and opponent channels.
    """
    if perspective not in ('home', 'away'):
        raise ValueError('perspective must be home or away')
    result = {}
    for period, native_period in PERIODS.items():
        for field, (key, _, group, _) in STATISTICS.items():
            for side in ('home', 'away'):
                role = 'team' if side == perspective else 'opponent'
                result[f'{side}_{field}_{period}'] = '::'.join(
                    quote(x, safe=' _') for x in (role, native_period, group, key, 'value'))
    return result


def homogenize_international_statistics(frame, *, perspective):
    """Rename reviewed wide statistics, retaining every other name and value.

    Numeric conversion is deliberately absent. The caller must supply reviewed
    numeric values to numerical consumers. Unknown fields (including odds) are
    untouched. Unreviewed home/away period fields are listed separately, without
    guessing a statistic. Duplicate columns or native-name collisions fail.

    ``attrs['international_statistic_names']['mapping']`` is JSON-serializable
    reversible lineage: invert it and use DataFrame.rename to recover names.
    """
    if not frame.columns.is_unique:
        raise ValueError('Duplicate international column names')
    mapping = {a: b for a, b in international_statistic_mapping(perspective).items() if a in frame}
    collisions = set(mapping.values()) & set(frame.columns)
    if collisions:
        raise ValueError(f'International/native statistic collision: {sorted(collisions)}')
    previous = frame.attrs.get('international_statistic_names')
    if previous is not None:
        if mapping or previous.get('policy') != STATISTIC_NAMING_POLICY or previous.get('perspective') != perspective:
            raise ValueError('Existing statistic naming lineage has a different perspective or mapping; restore source names first')
        return frame.copy(deep=True)
    result = frame.rename(columns=mapping).copy()
    result.attrs = deepcopy(frame.attrs)
    result.attrs['international_statistic_names'] = dict(
        policy=STATISTIC_NAMING_POLICY, perspective=perspective, mapping=mapping,
        unresolved=[c for c in frame.columns if isinstance(c, str)
                    and re.fullmatch(r'(home|away)_.+_(ft|1h|2h)', c)
                    and c not in mapping and c not in {
                        f'{side}_goals_{period}' for side in ('home', 'away') for period in PERIODS}],
    )
    return result
