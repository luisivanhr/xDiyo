"""Bounded four-publication integration with independent record-level membership."""
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import json
import numpy as np
import pandas as pd
from support import ROOT, SCRATCH, guard, save, sha
from xdiyo_analytics.data import load_seasons, select_stats
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import IsHome, Lag, Stat, evaluate_features
from xdiyo_analytics.labels import TeamValue, MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.splits import TemporalSplit, MatchKFold, GroupKFold, CPCV, create_split_plan, reconstruct_paths

source = guard()
data = load_seasons(ROOT/'data/xDiyo_data', ['23_24','24_25'], leagues=['Premier_League','Bundesliga'], tables=['matches','statistics'], record_dir=ROOT/'experiment/initial_population/selections', verify_hashes=True)
protected = {}
for item in data.provenance['sources']:
    path = __import__('pathlib').Path(item['manifest_path'])
    manifest = json.loads(path.read_text())
    protected[str(path)] = sha(path)
    for name in ['matches','statistics']:
        file = path.parent / manifest['tables'][name]['file']
        protected[str(file)] = sha(file)
for league in ['Premier_League','Bundesliga']:
    for season in ['23_24','24_25']:
        for file in [ROOT/f'experiment/initial_population/selections/{league}_{season}.json', ROOT/f'data/xDiyo_data/{league}_{season}.manifest.json']:
            protected[str(file)] = sha(file)
history = build_team_history(select_stats(data,stats=[('ALL','Match overview','cornerKicks')]))
corners = Stat('ALL','Match overview','cornerKicks')
features = evaluate_features(history,{'venue':IsHome(),'previous_corners':Lag(corners)},keyed=True)
labels = create_labels(history,{'team':TeamValue(corners),'total':MatchTotal(corners)})
checks = {}; checked_memberships = 0; reconstructed_values = 0

def records(dataset):
    result = {}
    for position, row in enumerate(dataset.metadata.to_dict('records')):
        key = tuple(row[name] for name in dataset.match_columns)
        result.setdefault(key, dict(row, positions=[]))['positions'].append(position)
    return result

def positions(match_records, keys):
    return sorted(position for key in keys for position in match_records[key]['positions'])

def temporal_oracle(model, splitter, *, lead=pd.Timedelta(0), explicit=False):
    rows = records(model)
    calendars = {}
    for key, row in rows.items():
        calendar = () if splitter.unit == 'kickoffs' else (row['competition_id'],)
        block = (row['kickoff_at'],) if splitter.unit == 'kickoffs' else (row['season_id'],) if splitter.unit == 'seasons' else (row['season_id'],row['round'])
        calendars.setdefault(calendar,{}).setdefault(block,[]).append(key)
    expected = []
    for calendar, blocks in calendars.items():
        ordered = sorted(blocks, key=lambda b:min(rows[key]['kickoff_at'] for key in blocks[b]))
        step = splitter.test_size if splitter.step is None else splitter.step
        for end in range(splitter.train_size,len(ordered)-splitter.gap,step):
            test_blocks = ordered[end+splitter.gap:end+splitter.gap+splitter.test_size]
            if len(test_blocks) < splitter.test_size and not splitter.allow_partial_test: break
            begin = 0 if splitter.window == 'expanding' else end-splitter.train_size
            candidates = [key for b in ordered[begin:end] for key in blocks[b]]
            test = [key for b in test_blocks for key in blocks[b]]
            boundary = min(rows[key]['kickoff_at']-lead for key in test) - pd.Timedelta(splitter.gap_time or 0)
            train = []
            for key in candidates:
                row = rows[key]
                available = row['kickoff_at'] + (pd.Timedelta('3h') if explicit else pd.Timedelta(0))
                if row['kickoff_at'] < boundary and available <= boundary and model.y.iloc[row['positions']].notna().all().all(): train.append(key)
            score = [key for b in test_blocks[splitter.score_start:] for key in blocks[b]]
            if splitter.score_rounds is not None:
                low, high = splitter.score_rounds
                score = [key for key in score if (low is None or rows[key]['round'] >= low) and (high is None or rows[key]['round'] <= high)]
            expected.append((positions(rows,train),positions(rows,test),positions(rows,score)))
    return expected

for name, label in labels.items():
    model = assemble_dataset(features,label,layout=label.unit)
    order = np.random.default_rng(292).permutation(len(model.X))
    model.X, model.y, model.metadata = [frame.iloc[order].reset_index(drop=True) for frame in [model.X,model.y,model.metadata]]
    before = deepcopy(model)
    match_records = records(model)
    configs = {
        'rounds': TemporalSplit(20,test_size=6,step=6,score_start=1,score_rounds=(5,34),allow_partial_test=True),
        'lead_2d': TemporalSplit(20,test_size=6,step=6,allow_partial_test=True),
        'seasons': TemporalSplit(1,unit='seasons',score_rounds=(5,None)),
        'pooled_sliding': TemporalSplit(20,test_size=8,step=50,window='sliding',unit='kickoffs',gap=1,gap_time='2h'),
        'match_kfold': MatchKFold(5,shuffle=True,random_state=291),
        'group_kfold': GroupKFold(4),
        'cpcv': CPCV(6,2,embargo='1D'),
    }
    for kind, splitter in configs.items():
        options = {}
        if kind == 'lead_2d': options = {'cutoffs':model.metadata.kickoff_at-pd.Timedelta('2D'),'available_at':model.metadata.kickoff_at+pd.Timedelta('3h')}
        if kind == 'cpcv': options = {'information_start':model.metadata.kickoff_at-pd.Timedelta('2D'),'available_at':model.metadata.kickoff_at+pd.Timedelta('3h')}
        plan = create_split_plan(model,splitter,**options)
        if isinstance(splitter,TemporalSplit):
            expected = temporal_oracle(model,splitter,lead=pd.Timedelta('2D') if kind == 'lead_2d' else pd.Timedelta(0),explicit=kind=='lead_2d')
            assert len(expected) == len(plan.folds)
            for fold, answer in zip(plan.folds,expected):
                assert (fold.train.tolist(),fold.test.tolist(),fold.score.tolist()) == answer
        for fold in plan.folds:
            for member in [fold.train,fold.test,fold.score]:
                chosen = set(member)
                assert all(not (chosen & set(row['positions'])) or set(row['positions']) <= chosen for row in match_records.values())
                checked_memberships += len(member)
            assert not set(fold.train) & set(fold.test) and set(fold.score) <= set(fold.test)
        if kind in ['match_kfold','group_kfold']:
            assert sorted(int(row) for fold in plan.folds for row in fold.test) == list(range(len(model.X)))
        if kind == 'group_kfold':
            for fold in plan.folds:
                a = set(zip(model.metadata.iloc[fold.train].competition_id,model.metadata.iloc[fold.train].season_id))
                b = set(zip(model.metadata.iloc[fold.test].competition_id,model.metadata.iloc[fold.test].season_id))
                assert not a & b and len(b) == 1
        if kind == 'cpcv':
            assert len(plan.folds) == 15 and splitter.n_paths == 5
            assert [f.metadata['test_blocks'] for f in plan.folds] == list(combinations(range(6),2))
            # Independent event-interval comparisons and embargo boundaries for every candidate.
            info = {key:(row['kickoff_at'],row['kickoff_at']-pd.Timedelta('2D'),row['kickoff_at']+pd.Timedelta('3h')) for key,row in match_records.items()}
            groups = model.groups.tolist()
            for fold in plan.folds:
                selected = set(groups[row] for row in fold.test)
                candidates = set(match_records)-selected
                purged = {key for key in candidates if any(info[key][1] <= info[test][2] and info[test][1] <= info[key][2] for test in selected)}
                runs = []
                for block in fold.metadata['test_blocks']:
                    if runs and block==runs[-1][-1]+1:runs[-1].append(block)
                    else:runs.append([block])
                ends = [max(info[groups[row]][2] for block in run for row in fold.metadata['block_rows'][block]) for run in runs]
                embargoed = {key for key in candidates if any(end < info[key][0] <= end+pd.Timedelta('1D') for end in ends)}
                assert fold.train.tolist() == positions(match_records,candidates-purged-embargoed)
            sentinels = [pd.DataFrame({'position_check':f.test.astype(float),'missing_check':np.where(f.test%7==0,np.nan,1.)},index=f.test) for f in plan.folds]
            reconstructed = reconstruct_paths(plan,sentinels)
            assert len(reconstructed)==5*len(model.X)
            for _, path in reconstructed.groupby('path_id'):
                assert path.row_position.tolist()==sorted(range(len(model.X)),key=lambda row:model.metadata.kickoff_at.iloc[row])
                assert path.position_check.tolist()==path.row_position.astype(float).tolist()
                assert path.missing_check.isna().sum()==len(range(0,len(model.X),7))
            reconstructed_values += len(reconstructed)*2
        checks[name+':'+kind] = {'folds':len(plan.folds),'train_rows':sum(len(f.train) for f in plan.folds),'test_rows':sum(len(f.test) for f in plan.folds),'score_rows':sum(len(f.score) for f in plan.folds)}
    for field in ['X','y','metadata']:pd.testing.assert_frame_equal(getattr(model,field),getattr(before,field))
    assert model.definitions==before.definitions
assert all(sha(path)==digest for path,digest in protected.items())
assert guard()==source
record = {'status':'verified','checked_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source,'publications':len(data.provenance['sources']),'matches':len(data.matches),'history_rows':len(history),'checks':checks,'membership_positions_checked':checked_memberships,'reconstructed_prediction_cells':reconstructed_values,'prepared_source_sha256':protected,'inputs_unchanged':True,'timing_note':'Explicit availability is a synthetic kickoff-plus-three-hours scenario, not observed completion/publication evidence. Two-day prediction lead is a sensitivity scenario. CPCV predictions are numeric alignment sentinels, not fitted forecasts.','reference':'Independent Python match dictionaries, ordered block membership, pairwise interval overlap and contiguous-run embargo checks.'}
save('real_check.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','prepared_source_sha256','checks']},indent=2))
