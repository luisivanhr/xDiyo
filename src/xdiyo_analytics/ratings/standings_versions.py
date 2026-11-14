"""Publication-versioned standings; legacy retrospective policies stay separate."""
from hashlib import sha256
import json

import pandas as pd


KEYS = ('competition_id', 'season_id', 'event_id')
FIELDS = (*KEYS, 'home_id', 'away_id', 'kickoff_at', 'revision', 'available_at',
          'goals_home', 'goals_away', 'points_home', 'points_away', 'eligible', 'source')


def _time(value, name):
    time = pd.Timestamp(value)
    if pd.isna(time) or time.tzinfo is None:
        raise ValueError(f'Result versions require a nonmissing timezone-aware {name}.')
    return time.tz_convert('UTC')


def _journal(history, versions):
    from .standings import _id, _score
    if not isinstance(versions, pd.DataFrame) or set(FIELDS) - set(versions):
        raise ValueError('Result versions require the complete identity, publication, score and provenance schema.')
    if versions.columns.duplicated().any():
        raise ValueError('Result version columns must be unique.')
    records, grouped = [], {}
    for row in versions.to_dict('records'):
        key = tuple(_id(row[k]) for k in KEYS)
        for name in ('home_id', 'away_id'):
            row[name] = _id(row[name])
        row['kickoff_at'] = _time(row['kickoff_at'], 'kickoff_at')
        row['available_at'] = _time(row['available_at'], 'available_at')
        revision = _score(row['revision'])
        if revision is None or isinstance(row['revision'], bool):
            raise ValueError('Result revisions must be consecutive nonnegative integers starting at zero.')
        row['revision'] = revision
        if type(row['eligible']) is not bool:
            raise ValueError('A result version needs explicit boolean eligibility.')
        if not isinstance(row['source'], str) or not row['source'].strip():
            raise ValueError('Every result version needs a nonempty source reference.')
        for field in ('goals_home', 'goals_away', 'points_home', 'points_away'):
            row[field] = _score(row[field])
        if row['eligible'] and (any(row[f] is None for f in ('goals_home', 'goals_away', 'points_home', 'points_away'))
                               or (row['points_home'], row['points_away']) not in ((3, 0), (0, 3), (1, 1), (0, 0))):
            raise ValueError('Eligible versions need complete scores and a supported 3/1/0 points pair.')
        selected = history
        for name, value in zip(KEYS, key):
            selected = selected.loc[selected[name].eq(value)]
        if len(selected) != 2 or set(selected.side) != {'home', 'away'}:
            raise ValueError('A result version must identify exactly one paired source fixture.')
        home = selected.loc[selected.side.eq('home')].iloc[0]
        away = selected.loc[selected.side.eq('away')].iloc[0]
        if (home.team_id != row['home_id'] or home.opponent_id != row['away_id']
                or away.team_id != row['away_id'] or away.opponent_id != row['home_id']
                or any(_time(v, 'source kickoff') != row['kickoff_at'] for v in selected.kickoff_at)):
            raise ValueError('Result version teams/kickoff disagree with the source fixture.')
        # Persist only the declared schema, with optional explanatory provenance.
        owned = {k: row[k] for k in FIELDS}
        owned.update(zip(KEYS, key))
        if 'time_basis' in row:
            if not isinstance(row['time_basis'], str) or not row['time_basis'].strip():
                raise ValueError('A supplied time_basis must be a nonempty string.')
            owned['time_basis'] = row['time_basis']
        grouped.setdefault(key, []).append(owned)
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r['revision'])
        if [r['revision'] for r in rows] != list(range(len(rows))):
            raise ValueError('Result revisions must be unique and consecutive from the initial revision zero.')
        if any(a['available_at'] >= b['available_at'] for a, b in zip(rows, rows[1:])):
            raise ValueError('Result revisions need strictly increasing publication times; simultaneous precedence is ambiguous.')
        records.extend(rows)
    records.sort(key=lambda r: tuple((type(r[k]).__name__, str(r[k])) for k in KEYS) + (r['revision'],))
    return grouped, records


def build_versioned_standings(history, *, team_seasons, result_versions, normalized_ranks,
                              queries, cutoffs, available_at, venues, timing,
                              meetings_per_pair, allow_incomplete_schedule):
    """Resolve each version at the query cutoff, retaining original event order.

    A supplied journal enables strict administrative provenance: every awarded
    or explicitly reviewed fixture in the requested source needs a journal.
    Ordinary result timing retains the caller's declared convention. No precise
    publication times, initial sporting results or rulings are inferred here.
    """
    from . import standings as native
    from ..features.standings import DerivedStandings
    if timing not in native.TIMINGS:
        raise ValueError('Unsupported result-version standings timing.')
    if timing == 'kickoff_day' and available_at is not None:
        raise ValueError('Kickoff-day ordinary results do not accept available_at; journal releases remain explicit.')
    if type(allow_incomplete_schedule) is not bool:
        raise ValueError('allow_incomplete_schedule must be a boolean.')
    if allow_incomplete_schedule and (type(meetings_per_pair) is not int or meetings_per_pair not in (1, 2, 3)):
        raise ValueError('Incomplete independent schedules require explicit meetings_per_pair=1, 2 or 3.')
    if not allow_incomplete_schedule and meetings_per_pair is not None and (type(meetings_per_pair) is not int or meetings_per_pair != 3 or timing != 'kickoff_day'):
        raise ValueError('Only explicit meetings_per_pair=3 with kickoff_day timing is supported.')
    if isinstance(venues, str) or not isinstance(venues, (tuple, list)) or not venues or any(v not in ('all', 'home', 'away') for v in venues):
        raise ValueError('Standings venues must be a nonempty sequence of all, home and/or away.')
    venues = tuple(dict.fromkeys(venues))
    journal, rows = _journal(history, result_versions)
    structural = history.copy(deep=True)
    special = pd.Series(False, index=history.index)
    if 'is_awarded' in history:
        special |= history.is_awarded.fillna(False).astype(bool)
    if 'result_review_json' in history:
        special |= history.result_review_json.notna() & history.result_review_json.ne('')
    for field in ('for', 'against'):
        column = 'standings_goals_' + field
        if column in history:
            special |= history[column].notna() & history[column].ne(history['goals_' + field])
    covered = pd.MultiIndex.from_frame(history[list(KEYS)]).isin(journal)
    if (special.to_numpy() & ~covered).any():
        raise ValueError('Strict result-version timing needs a publication journal for every administrative/reviewed fixture; unknown provenance cannot fall back to a final score.')
    # Reuse native structural/schedule validation. These approved fields are
    # validation only; journal outcomes, never this final representation, drive
    # versioned replay. Raw sporting fields are not changed.
    for key, versions in journal.items():
        mask = pd.Series(True, index=structural.index)
        for k, v in zip(KEYS, key):
            mask &= structural[k].eq(v)
        reference = next((r for r in reversed(versions) if r['eligible']), None)
        scores = (reference['goals_home'], reference['goals_away']) if reference else (0, 0)
        points = (reference['points_home'], reference['points_away']) if reference else (0, 0)
        for side, order in (('home', (0, 1)), ('away', (1, 0))):
            take = mask & structural.side.eq(side)
            structural.loc[take, ['standings_goals_for', 'standings_goals_against']] = [scores[i] for i in order]
            structural.loc[take, ['standings_points_for', 'standings_points_against']] = [points[i] for i in order]
        structural.loc[mask, 'standings_result_available_at'] = versions[-1]['available_at']
        structural.loc[mask, 'result_review_json'] = json.dumps(dict(reviewed=True, source=versions[-1]['source'], policy='explicit_publication_version_journal'))
    scope, members, groups, fixtures, skipped, identity = native._canonical(structural, team_seasons, available_at,
        timing=timing, meetings_per_pair=meetings_per_pair, allow_incomplete_schedule=allow_incomplete_schedule)
    windows = DerivedStandings(normalized_ranks, include_raw=True).normalized_ranks
    query_rows = native._query_keys(history if queries is None else queries, scope, fixtures, cutoffs, timing=timing)
    requested = {}
    for skey, q, _, _ in query_rows:
        requested.setdefault(skey, set()).add(q)
    policy = native.VERSIONED_DAILY_POLICY if timing == 'kickoff_day' else native.VERSIONED_POLICY
    publication_rule = 'original_kickoff_and_result_version_available_at_cutoff'
    persisted = [{k: v.isoformat() if isinstance(v, pd.Timestamp) else v for k, v in r.items()} for r in rows]
    canonical_identity = identity
    identity = sha256(json.dumps(dict(canonical_source=canonical_identity, versions=persisted, policy=policy), sort_keys=True).encode()).hexdigest()
    snapshots, publications = [], []
    extended = venues != ('all',)
    for skey, times in sorted(requested.items(), key=lambda item: repr(item[0])):
        teams = sorted(members[skey], key=lambda t: (type(t).__name__, str(t)))
        matches = sorted(groups[skey], key=lambda m: (m['kickoff'], type(m['event']).__name__, m['event']))
        for q in sorted(times):
            eligible = []
            for match in matches:
                if match['kickoff'] >= q:
                    continue
                key = (*skey[:2], match['event'])
                versions = journal.get(key)
                if versions is None:
                    if match['valid'] and match['available'] <= q:
                        eligible.append(match)
                    continue
                known = [v for v in versions if v['available_at'] <= q]
                if not known or not known[-1]['eligible']:
                    continue
                version = known[-1]
                selected = dict(match, gf=version['goals_home'], ga=version['goals_away'],
                                home_points=version['points_home'], away_points=version['points_away'],
                                result_revision=version['revision'], available=version['available_at'])
                eligible.append(selected)
            for venue in venues:
                histories = {t: [] for t in teams}
                for match in eligible:
                    for team, gf, ga, side in ((match['home'], match['gf'], match['ga'], 'home'), (match['away'], match['ga'], match['gf'], 'away')):
                        if venue != 'all' and side != venue:
                            continue
                        points = match.get('home_points' if side == 'home' else 'away_points')
                        if points is None:
                            points = 3 * int(gf > ga) + int(gf == ga)
                        histories[team].append((1, int(points == 3), int(points == 1), int(points == 0), points, gf, ga, gf - ga))
                pid = len(publications)
                publications.append(dict(scope=list(skey), cutoff=q.isoformat(), publication_rule=publication_rule,
                    eligible_result_count=len(eligible), contributing_events=[m['event'] for m in eligible],
                    result_versions=[dict(event=m['event'], revision=m['result_revision']) for m in eligible if 'result_revision' in m],
                    latest_kickoff=max((m['kickoff'] for m in eligible), default=None).isoformat() if eligible else None,
                    **({'venue': venue} if extended else {})))
                for w in (0, *windows):
                    totals = []
                    for team in teams:
                        past = histories[team][-w:] if w else histories[team]
                        totals.append([sum(row[i] for row in past) for i in range(8)])
                    for team, values, rank in zip(teams, totals, native.competition_ranks(totals)):
                        snapshots.append(dict(zip(scope, skey), team_id=team, recorded_at=q, window=w, rank=rank,
                            team_count=len(teams), effective_count=min(w, len(histories[team])) if w else len(histories[team]),
                            publication_id=pid, **({'venue': venue} if extended else {}), **dict(zip(native.TOTALS, values))))
    metadata = dict(policy=policy, publication_rule=publication_rule, scope=list(scope), windows=list(windows),
        canonical_source_sha256=canonical_identity, source_sha256=identity,
        state_sha256=sha256(json.dumps([identity, sorted(windows), sorted(venues)]).encode()).hexdigest(),
        query_sha256=sha256(repr(sorted((str(k), sorted(map(str, v))) for k, v in requested.items())).encode()).hexdigest(),
        timing=timing, ranking=['points_desc', 'gd_desc', 'gf_desc', 'ga_asc', 'competition_ranks'],
        availability='explicit_admin_versions_with_declared_ordinary_result_timing',
        ordinary_result_timing=timing, result_version_schema=1, result_versions=persisted,
        result_versions_sha256=sha256(json.dumps(persisted, sort_keys=True).encode()).hexdigest(),
        points_rule='journal_points_else_ordinary_score_3_1_0',
        replay='latest_known_version_per_fixture_in_original_kickoff_order',
        skipped=skipped, late_team_insertions=0, table_computations=len(publications),
        persistence='exact_cutoff_lookup_only', schedule='verified_balanced_single_or_double_round_robin')
    if extended:
        metadata['venues'] = list(venues)
    if meetings_per_pair is not None:
        metadata['meetings_per_pair'] = meetings_per_pair
        metadata['schedule'] = 'verified_independent_triple_round_robin_2_1_venues'
    if allow_incomplete_schedule:
        metadata.update(schedule='explicit_independent_regular_incomplete_schedule', allow_incomplete_schedule=True)
    return native.StandingsRun(pd.DataFrame(snapshots), publications, metadata, fixtures)


def validate_saved_versions(metadata, fixtures):
    """Validate the new journal contract and its link to saved fixture identity."""
    paired = []
    expected = set()
    for fixture in fixtures.values():
        key = (*fixture['scope'][:2], fixture['event'])
        review = fixture.get('points_review')
        if isinstance(review, dict) and review.get('policy') == 'explicit_publication_version_journal':
            expected.add(key)
        for side, team, opponent in [('home', fixture['home'], fixture['away']),
                                      ('away', fixture['away'], fixture['home'])]:
            paired.append(dict(zip(KEYS, key), team_id=team, opponent_id=opponent,
                               side=side, kickoff_at=fixture['kickoff']))
    records = metadata['result_versions']
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=FIELDS)
    journal, validated = _journal(pd.DataFrame(paired), frame)
    canonical = [{k: v.isoformat() if isinstance(v, pd.Timestamp) else v for k, v in row.items()}
                 for row in validated]
    if set(journal) != expected or records != canonical:
        raise ValueError('Saved result journal identity, coverage or canonical schema is inconsistent.')
    base = metadata.get('canonical_source_sha256')
    if not isinstance(base, str) or len(base) != 64 or any(c not in '0123456789abcdef' for c in base):
        raise ValueError('Saved result journal needs its canonical source fingerprint.')
    identity = sha256(json.dumps(dict(canonical_source=base, versions=canonical,
                                     policy=metadata['policy']), sort_keys=True).encode()).hexdigest()
    state = sha256(json.dumps([identity, sorted(metadata['windows']),
                               sorted(metadata.get('venues', ['all']))]).encode()).hexdigest()
    if metadata.get('source_sha256') != identity or metadata.get('state_sha256') != state:
        raise ValueError('Saved result journal source/state fingerprint mismatch.')
