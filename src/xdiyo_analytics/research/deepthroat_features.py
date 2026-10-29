"""Approved cross-season form convention, separate from seasonal standings."""
from dataclasses import replace
import numpy as np
import pandas as pd

from ..features import MatchScore, RollingMean, evaluate_features
from ..features.history import aligned_times, availability_times, _eligible_history_index
from ..labels import MatchGoals, Outcome, create_labels
from ..datasets import assemble_dataset

FEATURES = ('home::mean_scored', 'home::mean_conceded', 'home::form_rank',
            'away::mean_scored', 'away::mean_conceded', 'away::form_rank')
POLICY = 'current_season_known_roster_competition_history_average_ties_v1'


def windows(values, minimum=1):
    from numbers import Integral
    values = tuple(values)
    if not values or any(isinstance(n, bool) or not isinstance(n, Integral) or n < minimum for n in values):
        raise ValueError(f'Expected nonempty integer candidates >= {minimum}.')
    if len(set(values)) != len(values):
        raise ValueError('Candidate values must be distinct.')
    return tuple(sorted(int(n) for n in values))


def normalized_ranks(totals):
    """Rows are points, GF, GA; exact ties share their average ordinal rank."""
    if len(totals) < 2:
        raise ValueError('Rank normalization requires at least two active teams.')
    keys = [(-int(p), -int(gf-ga), -int(gf)) for p,gf,ga in totals]
    ordered = sorted(keys)
    ranks = {}
    for key in set(keys):
        positions = [i+1 for i, value in enumerate(ordered) if value == key]
        ranks[key] = sum(positions)/len(positions)
    return np.array([(len(keys)-ranks[k])/(len(keys)-1) for k in keys])


def prepare_bundles(history, roster, *, n_values=range(9,101), cutoffs=None, available_at=None):
    """Build six-input native match datasets; no fitting or selection.

    Roster requires competition_id, season_id, team_id, known_at. Membership
    timestamps are explicit evidence/assumptions supplied by the caller. Missing
    paired results are removed from *both* histories before counting windows.
    Native eligibility handles result completion, cutoffs and excluded targets.
    The optional availability argument has evaluate_features semantics.
    """
    ns = windows(n_values, 6)
    h = history.copy(deep=True).reset_index(drop=True)
    cutoff = aligned_times(history, cutoffs, default='kickoff_at').reset_index(drop=True)
    release = availability_times(history, available_at).reset_index(drop=True)
    required = ['competition_id','season_id','team_id','known_at']
    if set(required)-set(roster):
        raise ValueError('Roster needs competition_id, season_id, team_id and explicit known_at.')
    members = roster[required].copy()
    if members[required[:3]].isna().any().any() or members.duplicated(required[:3]).any():
        raise ValueError('Roster membership identities must be unique and nonmissing.')
    members['known_at'] = pd.to_datetime(members.known_at, utc=True, errors='raise')
    if members.known_at.isna().any() or cutoff.isna().any():
        raise ValueError('Membership availability and prediction cutoffs must be known.')
    # Native label validation checks paired identities and score perspectives.
    goal_label = create_labels(h, {'goals': MatchGoals()})['goals']
    keys = list(goal_label.identity_columns)
    paired = h[keys].copy()
    paired['cutoff'] = cutoff
    mismatch = paired.groupby(keys, sort=False).cutoff.nunique(dropna=False).ne(1)
    if mismatch.any():
        fixture = mismatch[mismatch].index[0]
        raise ValueError(f'Both sides of a fixture must have identical prediction cutoffs; inconsistent fixture {fixture!r}.')
    paired['release'] = release
    grouped = paired.groupby(keys, sort=False).release
    # Neither perspective can consume a result before both are released.
    release = grouped.transform('max').where(grouped.transform('count').eq(2))
    scores = h[['goals_for','goals_against']].to_numpy(dtype=float, na_value=np.nan)
    complete = np.isfinite(scores).all(axis=1)
    h.loc[~complete, 'status'] = 'incomplete_score'
    expected_result = np.full(len(h), None, dtype=object)
    gf_complete, ga_complete = scores[complete].T
    expected_result[complete] = np.where(gf_complete > ga_complete, 'W', np.where(gf_complete < ga_complete, 'L', 'D'))
    if 'result' in h and (h.loc[complete & h.status.eq('finished'),'result'] != expected_result[complete & h.status.eq('finished')]).any():
        raise ValueError('Outcome labels disagree with paired goal scores.')
    h['result'] = expected_result
    definitions = {f'{name}_{n}': RollingMean(MatchScore(side=side), n, min_periods=6)
                   for n in ns for name,side in [('mean_scored','for'),('mean_conceded','against')]}
    means = evaluate_features(h, definitions, cutoffs=cutoff, available_at=release, keyed=True)
    # One bounded native history query per member/cutoff, reused for all n.
    h['_research_cutoff'] = cutoff
    probes, contexts = [], []
    for (comp, season, time), positions in h.groupby(['competition_id','season_id','_research_cutoff'],sort=False).indices.items():
        active = members[members.competition_id.eq(comp) & members.season_id.eq(season) & members.known_at.le(time)]
        teams = active.team_id.tolist()
        if len(teams) < 2 or not set(h.iloc[positions].team_id) <= set(teams):
            raise ValueError('Prediction teams require a known current-season roster of at least two teams.')
        start = len(probes)
        template = h.iloc[positions[0]].to_dict()
        for team in teams:
            probes.append({**template, 'team_id':team, 'status':'research_query',
                           'kickoff_at':time, '_research_cutoff':time})
        contexts.append((positions,teams,start,len(probes)))
    probe_frame = pd.DataFrame(probes)
    query_history = pd.concat([h,probe_frame], ignore_index=True)
    query_release = pd.concat([release,pd.Series(pd.NaT,index=range(len(probes)),dtype='datetime64[ns, UTC]')],ignore_index=True)
    index = _eligible_history_index(query_history, cutoffs=query_history._research_cutoff, available_at=query_release)
    gf = h.goals_for.to_numpy(dtype=float,na_value=np.nan)
    ga = h.goals_against.to_numpy(dtype=float,na_value=np.nan)
    points = np.where(gf>ga,3,np.where(gf==ga,1,0))
    rank = {n: np.full(len(h),np.nan) for n in ns}
    count = np.zeros(len(h),dtype=int)
    for positions,teams,start,end in contexts:
        totals = {n: [] for n in ns}
        counts = {}
        for i,team in enumerate(teams,start):
            rows = index.window(len(h)+i,max(ns))
            counts[team] = len(rows)
            sums = np.cumsum(np.column_stack([points[rows],gf[rows],ga[rows]])[::-1],axis=0)
            for n in ns:
                totals[n].append(sums[min(n,len(rows))-1] if len(rows) else [0,0,0])
        row_teams = h.iloc[positions].team_id.tolist()
        count[positions] = [counts[t] for t in row_teams]
        for n in ns:
            mapping = dict(zip(teams,normalized_ranks(totals[n])))
            rank[n][positions] = [mapping[t] for t in row_teams]
    labels = create_labels(h, {'goals':MatchGoals(),'outcome':Outcome(perspective='home')})
    # A paired release bound accompanies the development-selection gate.
    paired_release = h[keys].copy(); paired_release['release'] = release
    releases = paired_release.groupby(keys,sort=False).release.max()
    counts = means.iloc[:,:0].copy(); counts['complete_history_count'] = count
    count_data = assemble_dataset(counts,labels['goals'],layout='match')
    result = {}
    for n in ns:
        values = means[[f'mean_scored_{n}',f'mean_conceded_{n}']].rename(columns={f'mean_scored_{n}':'mean_scored',f'mean_conceded_{n}':'mean_conceded'})
        values['form_rank'] = rank[n]
        values.attrs = {}  # Identity remains in the native keyed index.
        data = assemble_dataset(values,labels['goals'],layout='match')
        data.X = data.X[list(FEATURES)]
        data.metadata['research_eligible'] = count_data.X.ge(6).all(axis=1) & np.isfinite(data.X).all(axis=1)
        data.metadata['label_available_at'] = releases.reindex(pd.MultiIndex.from_frame(data.metadata[keys])).to_numpy()
        data.metadata['prediction_cutoff'] = cutoff[h.side.eq('home')].to_numpy()
        data.definitions.update(display_name='Deepthroat-gg', membership_policy=POLICY, window=n,
            features={k:v for k,v in definitions.items() if k.endswith(f'_{n}')},
            short_history='at least six complete paired results; actual count denominator',
            availability='native kickoff proxy with completion bounds' if available_at is None else str(available_at))
        outcome_data = assemble_dataset(values, labels['outcome'], layout='match')
        pd.testing.assert_frame_equal(outcome_data.metadata[keys], data.metadata[keys])
        outcomes = replace(data, y=outcome_data.y, target_perspective=outcome_data.target_perspective,
                           definitions={**data.definitions, 'label': labels['outcome'].definition})
        result[n] = (data,outcomes)
    return result
