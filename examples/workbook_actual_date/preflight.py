"""Independent frozen-population and causal-boundary gate before CUDA launch."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from experiment_inputs import load_stage
from run import STAGES, FOLDS, FAMILIES, config, sha, write

ROOT = Path(__file__).resolve().parent
ORIGINAL = {8,17,18,23,34,35,36,37,38,44,53,54,182}

def run():
    records, reference, compatibility = [], {}, {}
    for stage in STAGES:
        data = load_stage(stage)
        store = data['store']
        cfg = config(FAMILIES[0], store)
        for name,frame in (('schedule',store.schedule),('history',store.history),('phase',store.phase_context.frame)):
            part=frame.loc[frame.competition_id.isin(ORIGINAL)].sort_values(['event_id']+(['side'] if 'side' in frame else [])).reset_index(drop=True)
            if stage=='original13':compatibility[name]=part
            else:pd.testing.assert_frame_equal(part,compatibility[name],check_exact=True)
        groups = [('development', data['dev_fit'], data['dev_validation'])]
        groups += [(fold, data['folds'][fold]['train'], data['folds'][fold]['test']) for fold in FOLDS]
        for label, train, test in groups:
            assert len(train) and len(test), (stage, label, 'empty')
            assert not set(train.event_id) & set(test.event_id), (stage,label,'overlap')
            for frame in (train,test):
                assert not frame.event_id.duplicated().any()
                assert np.isfinite(frame[['goals_for','goals_against']].to_numpy(dtype=float)).all()
                assert frame[['goals_for','goals_against']].ge(0).all().all()
                assert frame.home_id.ne(frame.away_id).all()
            cutoff = min(cfg.history_cutoff(t) for t in store.issue_times(test,cfg))
            available = store.label_availability(train,cfg)
            assert available.notna().all() and available.max() < cutoff, (stage,label,'label leakage',str(available.max()),str(cutoff))
            assert cfg.forecast_policy == 'actual_playing_date_v1'
            for frame in (train,test):
                actual=store.issue_times(frame,cfg)
                expected=frame.kickoff_at.dt.tz_convert(cfg.timezone).dt.normalize().dt.tz_convert('UTC')
                assert np.array_equal(actual,expected)
                store.assert_targets_unobserved(frame,actual,cfg)
            if label != 'development':
                year = 2021 + int(label.rsplit('_',1)[1])
                start, end = pd.Timestamp(f'{year}-07-01',tz='Europe/Berlin'), pd.Timestamp(f'{year+1}-07-01',tz='Europe/Berlin')
                dates = pd.to_datetime(test.kickoff_at,utc=True).dt.tz_convert('Europe/Berlin')
                assert dates.ge(start).all() and dates.lt(end).all(), (stage,label,'test dates')
                assert pd.to_datetime(train.kickoff_at,utc=True).lt(start).all()
                common = test[test.competition_id.isin(ORIGINAL)].sort_values('event_id').reset_index(drop=True)
                fields = ['event_id','competition_id','home_id','away_id','kickoff_at','issuance_at','goals_for','goals_against','round','phase_key']
                if stage == 'original13': reference[label] = common[fields]
                else: pd.testing.assert_frame_equal(common[fields],reference[label],check_exact=True)
            records.append(dict(stage=stage,fold=label,train=len(train),test=len(test),
                leagues=int(test.competition_id.nunique()),train_label_max=str(available.max()),test_cutoff_min=str(cutoff)))
            for position in (0,len(test)//2,len(test)-1):
                query=test.iloc[[position]];cut=store.issue_times(query,cfg).iloc[0]
                snapshot=store.snapshot(cut,cfg)
                assert snapshot.issuance==snapshot.cutoff==cut
                assert snapshot.edges.kickoff_at.lt(cut).all()
                assert snapshot.edges.result_available_at.lt(cut).all()
                assert not set(query.event_id)&set(snapshot.edges.event_id)
        for family in FAMILIES:
            cfg = config(family,store)
            assert cfg.statistics == config(FAMILIES[0],store).statistics
    # Real CUDA smoke artifacts are required independently of structural checks.
    smoke = []
    for stage in STAGES:
        for family in FAMILIES:
            path = ROOT/'results'/'smoke'/stage/family/'17'/'fit.json'
            if not path.exists():
                raise RuntimeError('Missing CUDA one-step smoke: '+str(path))
            record = json.loads(path.read_text())
            assert record['smoke'] is True
            smoke.append(dict(path=str(path),sha256=sha(path)))
    write(ROOT/'results'/'PREFLIGHT.json',dict(status='PASS',populations=records,smoke=smoke,
        scope='Full population integrity, strict causal train-label boundaries and actual-date cutoffs across every fold, paired original13 identity, real one-update CUDA smoke on 32 fitting and 32 validation fixtures for all nine stage/family combinations. Not a final experiment acceptance.'))

if __name__ == '__main__': run()
