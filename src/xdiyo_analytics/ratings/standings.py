"""League-wide score tables, replayed once per requested issuance cutoff.

This deliberately does not use the participant-only pairwise rating engine.
Saved runs are immutable query snapshots, not resumable rating checkpoints.
"""
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.history import aligned_times, availability_times

POLICY = "independent_3_1_0_first_round_complete_v1"
TOTALS = ("played", "wins", "draws", "losses", "points", "gf", "ga", "gd")


def _json_scalar(value):
    """Unbox NumPy scalars without stringifying IDs or relaxing strict JSON."""
    if isinstance(value, np.generic):
        native = value.item()
        if not isinstance(native, np.generic):
            return native
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _id(value):
    if pd.isna(value) or isinstance(value, bool):
        raise ValueError("Standings identities must be nonmissing exact IDs.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not np.isfinite(value) or value % 1 or abs(value) >= 2**53:
            raise ValueError("Standings rejects rounded or nonintegral floating IDs.")
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError("Unsupported standings identity.")


def competition_ranks(totals):
    """Exact lexicographic competition ranks; IDs never break numerical ties."""
    keys = [(-int(t[4]), -int(t[7]), -int(t[5]), int(t[6])) for t in totals]
    ordered = sorted(keys)
    rank = {}
    for i, key in enumerate(ordered):
        rank.setdefault(key, i + 1)
    return [rank[key] for key in keys]


def _score(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0 or value % 1:
        raise ValueError("Standings goals must be nonnegative finite integers or missing.")
    return int(value)


def _equal(a, b):
    return (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b)


def _canonical(history, roster, available_at):
    if roster is None:
        raise ValueError("DerivedStandings requires verified complete team_seasons membership, not teams seen in a fold.")
    scope = ("competition_id", "season_id")
    if "standings_table_id" in history:
        scope += ("standings_table_id",)
        if "standings_rule" not in history or not history.standings_rule.eq("independent_3_1_0").all():
            raise ValueError("Explicit table scopes require standings_rule=independent_3_1_0; carry/halving rules are unsupported.")
    required = (*scope, "event_id", "team_id", "opponent_id", "side", "round", "kickoff_at", "status", "goals_for", "goals_against")
    missing = set(required) - set(history.columns)
    if missing:
        raise ValueError(f"Standings history is missing {sorted(missing)}.")
    if set((*scope, "team_id")) - set(roster.columns):
        raise ValueError("Membership must explicitly identify every standings table scope and team.")
    members = {}
    for r in roster[list(scope) + ["team_id"]].itertuples(index=False, name=None):
        key, team = tuple(_id(x) for x in r[:-1]), _id(r[-1])
        members.setdefault(key, set()).add(team)
    # Only owned narrow columns enter the source fingerprint. No scalar frame
    # access, attrs copies or mutable caller references survive preparation.
    extra = [c for c in ("tournament_id", "stage_json", "is_awarded", "play_status", "result_review_json",
             "standings_goals_for", "standings_goals_against", "standings_result_available_at", "standings_rule") if c in history]
    extra += [c for c in history if c.startswith(("team_score_", "opponent_score_"))]
    frame = history[list(dict.fromkeys((*required, *extra)))].copy()
    frame.attrs = {}
    frame["_available"] = availability_times(history, available_at).to_numpy()
    frame["kickoff_at"] = aligned_times(history, None, default="kickoff_at").to_numpy()
    for c in (*scope, "event_id", "team_id", "opponent_id"):
        frame[c] = pd.Series([_id(v) for v in frame[c].tolist()], index=frame.index, dtype=object)
    fixtures, scopes, skipped = {}, {}, []
    for key, pair in frame.groupby([*scope, "event_id"], sort=False, dropna=False):
        key = tuple(key)
        skey, event = key[:-1], key[-1]
        rows = pair.to_dict("records")
        if len(rows) != 2 or {r['side'] for r in rows} != {"home", "away"}:
            raise ValueError(f"Standings needs exactly two paired team rows for {key}.")
        home, away = sorted(rows, key=lambda r: r['side'] != 'home')
        if home['team_id'] != away['opponent_id'] or away['team_id'] != home['opponent_id'] or home['team_id'] == away['team_id']:
            raise ValueError(f"Conflicting paired standings identities for {key}.")
        for field in ("round", "kickoff_at", "status", "_available", "tournament_id", "stage_json", "is_awarded", "play_status", "result_review_json", "standings_result_available_at"):
            if field in home and not _equal(home[field], away[field]):
                raise ValueError(f"Inconsistent paired {field} for {key}.")
        approved = [r.get(field) for r in (home, away)
                    for field in ('standings_goals_for', 'standings_goals_against')]
        present = [pd.notna(value) for value in approved]
        if any(present) and not all(present):
            raise ValueError(f"Incomplete approved standings score pair for {key}; both scores on both team rows are required.")
        if not any(present) and pd.notna(home.get('standings_result_available_at')):
            raise ValueError(f"Approved standings release for {key} requires a complete approved score pair.")
        for field, other in (("goals_for", "goals_against"), ("standings_goals_for", "standings_goals_against")):
            if field in home and (other not in away or not _equal(home[field], away[other]) or not _equal(away[field], home[other])):
                raise ValueError(f"Inconsistent paired {field} for {key}.")
        teams = members.get(skey, set())
        if len(teams) < 2 or not {home['team_id'], away['team_id']} <= teams:
            raise ValueError(f"Incomplete/conflicting membership or N<2 for {skey}.")
        round_ = home['round']
        if isinstance(round_, bool) or not isinstance(round_, Real) or pd.isna(round_) or round_ < 1 or round_ % 1:
            raise ValueError(f"Reliable positive integer round metadata required for {key}.")
        kick, available = home['kickoff_at'], home['_available']
        if pd.isna(kick):
            raise ValueError(f"Missing scheduled kickoff for {key}.")
        if 'standings_table_id' not in scope and 'stage_json' in home and pd.notna(home['stage_json']):
            stage = json.loads(home['stage_json']) if isinstance(home['stage_json'], str) else home['stage_json']
            if not isinstance(stage, dict) or set(stage) - {'round'}:
                raise ValueError(f"Unsupported stage metadata for {key}; supply verified independent standings_table_id/rule or resolve carry/reset rules.")
        explicit = all(present)
        special = (pd.notna(home.get('is_awarded')) and bool(home.get('is_awarded'))) or (pd.notna(home.get('result_review_json')) and bool(home.get('result_review_json')))
        if special and not explicit:
            raise ValueError(f"Awarded/revised result {key} needs approved standings_goals_for/against and known standings_result_available_at.")
        if explicit:
            release = home.get('standings_result_available_at')
            if pd.isna(release):
                raise ValueError("Approved standings scores need a known ruling/result release time.")
            release = pd.to_datetime(release, utc=True)
            if pd.isna(release):
                raise ValueError('Approved standings scores need a known ruling/result release time.')
            if pd.notna(available):
                available = max(available, release)
        elif any(pd.notna(home[c]) and home[c] != 0 for c in extra if c.startswith(('team_score_', 'opponent_score_'))):
            raise ValueError(f"Extra-time/shootout result {key} needs explicit approved league-result scores.")
        gf = _score(home['standings_goals_for'] if explicit else home['goals_for'])
        ga = _score(home['standings_goals_against'] if explicit else home['goals_against'])
        valid = home['status'] == 'finished' and gf is not None and ga is not None and pd.notna(available)
        play = home.get('play_status')
        if pd.notna(play) and any(t in str(play).lower() for t in ('abandon', 'unplayed', 'cancel')) and not explicit:
            valid = False
        item = dict(scope=skey, event=event, home=home['team_id'], away=away['team_id'],
                    kickoff=kick, available=available, round=int(round_), gf=gf, ga=ga, valid=bool(valid))
        fixtures[key] = item
        scopes.setdefault(skey, []).append(item)
        if not valid:
            skipped.append(dict(scope=list(skey), event=event, reason='incomplete_or_unavailable_result'))
    for skey, matches in scopes.items():
        # Validate a full balanced single/double round-robin schedule, including
        # unfinished fixtures. Mere provider "complete" flags are insufficient.
        teams = members[skey]
        counts = Counter(frozenset((r['home'], r['away'])) for r in matches)
        repetitions = set(counts.values())
        if len(counts) != len(teams)*(len(teams)-1)//2 or len(repetitions) != 1 or not repetitions <= {1, 2}:
            raise ValueError(f"Incomplete schedule or unsupported split/playoff format for {skey}; requires a complete independent single/double round-robin table.")
        if repetitions == {2} and len({(r['home'],r['away']) for r in matches}) != len(matches):
            raise ValueError(f"Conflicting home/away schedule for {skey}.")
        selected = frame.loc[(frame[list(scope)].apply(tuple, axis=1) == skey)]
        if 'standings_table_id' not in scope and 'tournament_id' in selected and selected.tournament_id.nunique(dropna=False) > 1:
            raise ValueError(f"Multiple stages for {skey}; explicit independently verified table scope is required.")
        first = [r for r in matches if r['round'] == 1]
        first_teams = [t for r in first for t in (r['home'], r['away'])]
        if len(first) != len(teams)//2 or len(set(first_teams)) != len(first_teams):
            raise ValueError(f"Incomplete/ambiguous first-round fixture set for {skey}.")
    canonical = sorted((json.dumps(r, default=str, sort_keys=True) for r in frame.to_dict('records')))
    identity = sha256(json.dumps(dict(rows=canonical, members=sorted((str(k), sorted(map(str,v))) for k,v in members.items()), policy=POLICY), sort_keys=True).encode()).hexdigest()
    return scope, members, scopes, fixtures, skipped, identity


def _query_keys(history, scope, fixtures, cutoffs):
    times = aligned_times(history, cutoffs, default='kickoff_at')
    anchors = {}
    for match in fixtures.values():
        key = (*match['scope'], match['round'])
        anchors[key] = min(anchors.get(key, match['kickoff']), match['kickoff'])
    rows = []
    issuance = {}
    columns = [*scope, 'event_id', 'team_id', 'opponent_id', 'kickoff_at']
    for values, requested in zip(history[columns].itertuples(index=False, name=None), times):
        skey = tuple(_id(x) for x in values[:len(scope)])
        event, team, opponent = map(_id, values[len(scope):len(scope)+3])
        match = fixtures.get((*skey, event))
        kick = pd.to_datetime(values[-1], utc=True)
        if match is None or {team, opponent} != {match['home'], match['away']} or kick != match['kickoff']:
            raise ValueError('Query identity/kickoff does not match the complete standings schedule.')
        q = anchors[(*skey, match['round'])] if cutoffs is None else requested
        if pd.isna(q) or q > kick:
            raise ValueError('Standings requires known prediction cutoffs no later than target kickoff.')
        key = (*skey, event)
        if key in issuance and issuance[key] != q:
            raise ValueError('Paired fixture queries must have identical standings prediction cutoffs.')
        issuance[key] = q
        rows.append((skey, q, team, opponent))
    return rows


class StandingsRun:
    """Owned rank snapshots at exact requested cutoffs; safe copies on access.

    Persistence is lookup-only. Never infer continuation from rank snapshots.
    """
    def __init__(self, snapshots, publications, metadata, fixtures):
        self._snapshots = snapshots.copy(deep=True)
        self._publications = deepcopy(publications)
        self._metadata = deepcopy(metadata)
        self._fixtures = deepcopy(fixtures)
        self._lookup = None

    @property
    def snapshots(self):
        return self._snapshots.copy(deep=True)

    @property
    def publications(self):
        return deepcopy(self._publications)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def features(self, history, expression, *, cutoffs=None):
        scope = tuple(self._metadata['scope'])
        if set(expression.normalized_ranks) - set(self._metadata['windows']):
            raise ValueError('Requested form windows are absent from saved standings.')
        venues = ('home', 'away') if expression.venue == 'same' else (expression.venue,)
        if set(venues) - set(self._metadata.get('venues', ['all'])):
            raise ValueError('Requested venues are absent from saved standings; rebuild the required venue tables.')
        queries = _query_keys(history, scope, self._fixtures, cutoffs)
        if self._lookup is None:
            self._lookup = {tuple(r[c] for c in (*scope, 'recorded_at', 'team_id', 'window')) + (r.get('venue', 'all'),):r
                            for r in self._snapshots.to_dict('records')}
        lookup = self._lookup
        columns, output, audits = [], [], []
        roles = {'for':('team',), 'against':('opponent',), 'both':('team','opponent')}[expression.side]
        selected = ([0] if expression.include_raw else []) + list(expression.normalized_ranks)
        for role in roles:
            for w in selected:
                columns.append(role + '::' + ('' if expression.venue == 'all' else expression.venue + '::') + (f'form_rank_{w}' if w else ('season_position_over_n' if expression.normalize else 'season_position')))
        for (skey, q, team, opponent), event in zip(queries, history.event_id.tolist()):
            focal_venue = 'home' if self._fixtures[(*skey, _id(event))]['home'] == team else 'away'
            values, audit = [], []
            for role in roles:
                t = team if role == 'team' else opponent
                venue = expression.venue
                if venue == 'same':
                    venue = focal_venue if role == 'team' else ('away' if focal_venue == 'home' else 'home')
                for w in selected:
                    row = lookup.get((*skey, q, t, w, venue))
                    if row is None:
                        raise ValueError('Saved standings do not cover this exact query cutoff; rebuild from full source history.')
                    rank, n = row['rank'], row['team_count']
                    values.append((n-rank)/(n-1) if w else rank/n if expression.normalize else rank)
                    audit.append(dict(team_id=t, window=w, effective_count=row['effective_count'],
                                      publication_id=row['publication_id'], gate_open=row['gate_open'],
                                      **({'venue': venue} if expression.venue != 'all' else {})))
            output.append(values)
            audits.append(audit)
        result = pd.DataFrame(output, index=history.index, columns=columns, dtype=float)
        result.attrs['standings_audit'] = dict(source=self.metadata, publications=self.publications, rows=audits)
        return result

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if (path/'standings.json').exists() or (path/'snapshots.parquet').exists():
            raise FileExistsError('Standings artifacts already exist.')
        fixtures = []
        for m in self._fixtures.values():
            r = dict(m, scope=list(m['scope']), kickoff=m['kickoff'].isoformat(), available=None if pd.isna(m['available']) else m['available'].isoformat())
            fixtures.append(r)
        body = dict(version=2 if 'venues' in self._metadata else 1, metadata=self.metadata, publications=self.publications, fixtures=fixtures)
        payload = json.dumps(body, default=_json_scalar, allow_nan=False, indent=2)
        self._snapshots.to_parquet(path/'snapshots.parquet', index=False, compression='zstd')
        (path/'standings.json').write_text(payload+'\n', encoding='utf-8')

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        body = json.loads((path/'standings.json').read_text(encoding='utf-8'))
        if body['version'] not in (1, 2) or body['metadata']['policy'] != POLICY:
            raise ValueError('Unsupported standings artifact version/policy.')
        fixtures = {}
        for m in body['fixtures']:
            m['scope'] = tuple(m['scope'])
            m['kickoff'] = pd.Timestamp(m['kickoff'])
            m['available'] = pd.NaT if m['available'] is None else pd.Timestamp(m['available'])
            fixtures[(*m['scope'],m['event'])] = m
        return cls(pd.read_parquet(path/'snapshots.parquet'), body['publications'], body['metadata'], fixtures)


def build_standings(history, *, team_seasons, normalized_ranks=(5,), queries=None, cutoffs=None, available_at=None, venues=('all',)):
    """Build complete tables once per distinct scope/cutoff, without fitting.

    history must include the complete schedule/results. queries may be a subset.
    Default cutoffs freeze each original round at its earliest scheduled kickoff.
    Explicit cutoffs must be actual native issuance times, never outcome times.
    Missing/unfinished results do not consume form slots. Late releases insert in
    kickoff/event chronology; already emitted snapshots are never revised.
    Optional venues builds league-wide home-only/away-only tables, filtering
    before each team's last-w matches. The full first-round gate is unchanged.
    Default all-venue artifacts retain their original schema and state identity.
    """
    from ..features.standings import DerivedStandings
    windows = DerivedStandings(normalized_ranks, include_raw=True).normalized_ranks
    if isinstance(venues, str) or not isinstance(venues, (tuple, list)) or not venues or any(v not in ('all', 'home', 'away') for v in venues):
        raise ValueError('Standings venues must be a nonempty sequence of all, home and/or away.')
    venues = tuple(dict.fromkeys(venues))
    extended = venues != ('all',)
    scope, members, groups, fixtures, skipped, identity = _canonical(history, team_seasons, available_at)
    queries = history if queries is None else queries
    query_rows = _query_keys(queries, scope, fixtures, cutoffs)
    requested = {}
    for skey, q, _, _ in query_rows:
        requested.setdefault(skey, set()).add(q)
    snapshots, publications = [], []
    late = 0
    for skey, times in requested.items():
        teams = sorted(members[skey], key=lambda t:(type(t).__name__,str(t)))
        matches = groups[skey]
        first = {m['event'] for m in matches if m['round'] == 1}
        events = sorted((m for m in matches if m['valid']), key=lambda m:(max(m['available'], m['kickoff']+pd.Timedelta(1,'ns')), m['kickoff'], str(m['event'])))
        for venue in venues:
            histories = {t:[] for t in teams}
            totals = {w:{t:[0]*8 for t in teams} for w in (0,*windows)}
            used, cursor = [], 0
            for q in sorted(times):
                while cursor < len(events) and events[cursor]['available'] <= q and events[cursor]['kickoff'] < q:
                    m = events[cursor]
                    cursor += 1
                    used.append(m['event'])
                    for team, gf, ga, side in ((m['home'],m['gf'],m['ga'],'home'),(m['away'],m['ga'],m['gf'],'away')):
                        if venue != 'all' and venue != side:
                            continue
                        contribution = (1,int(gf>ga),int(gf==ga),int(gf<ga),3*int(gf>ga)+int(gf==ga),gf,ga,gf-ga)
                        past = histories[team]
                        order = (m['kickoff'], type(m['event']).__name__, m['event'])
                        at = bisect_left(past, order, key=lambda entry:entry[0])
                        late += int(at < len(past))
                        for w in (0,*windows):
                            if w and at < len(past)+1-w:
                                continue
                            dropped = past[-w][1] if w and len(past) >= w else (0,)*8
                            totals[w][team] = [a+b-c for a,b,c in zip(totals[w][team],contribution,dropped)]
                        past.insert(at,(order,contribution))
                gate = first <= set(used)
                pid = len(publications)
                publications.append(dict(scope=list(skey), cutoff=q.isoformat(), gate_open=gate,
                    **({"venue": venue} if extended else {}),
                    pending_first_round=sorted(first-set(used),key=str), contributing_events=list(used),
                    latest_kickoff=None if not used else max(fixtures[(*skey,e)]['kickoff'] for e in used).isoformat()))
                for w in (0,*windows):
                    rows = [totals[w][t] if gate else [0]*8 for t in teams]
                    for t, arithmetic, rank in zip(teams, rows, competition_ranks(rows)):
                        snapshots.append(dict(zip(scope,skey), team_id=t, recorded_at=q, window=w, rank=rank,
                            **({"venue": venue} if extended else {}),
                            team_count=len(teams), effective_count=min(w,len(histories[t])) if w else len(histories[t]),
                            gate_open=gate, publication_id=pid, **dict(zip(TOTALS,arithmetic))))
    metadata = dict(policy=POLICY, scope=list(scope), windows=list(windows), source_sha256=identity,
        state_sha256=sha256(json.dumps([identity,sorted(windows),POLICY]).encode()).hexdigest(),
        query_sha256=sha256(repr(sorted((str(k),sorted(str(t) for t in v)) for k,v in requested.items())).encode()).hexdigest(),
        ranking=['points_desc','gd_desc','gf_desc','ga_asc','competition_ranks'],
        availability='earlier_finished_kickoff_proxy_with_reviewed_bounds' if available_at is None else 'explicit_with_reviewed_bounds',
        skipped=skipped, late_team_insertions=late, table_computations=len(publications),
        persistence='exact_cutoff_lookup_only', schedule='verified_balanced_single_or_double_round_robin')
    if extended:
        metadata['venues'] = list(venues)
        metadata['state_sha256'] = sha256(json.dumps([identity, sorted(windows), sorted(venues), POLICY]).encode()).hexdigest()
    return StandingsRun(pd.DataFrame(snapshots), publications, metadata, fixtures)
