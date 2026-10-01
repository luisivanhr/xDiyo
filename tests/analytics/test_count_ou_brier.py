"""Native count-tail scoring, identity preservation and retained trial replay."""
from copy import deepcopy
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.evaluation import Metric, evaluate_metrics
from xdiyo_analytics.evaluation.count_scores import DEFAULT_OU_LINES
from xdiyo_analytics.reporting import PerformanceReporter
from xdiyo_analytics.selection import Candidate, MetricSelection, ModelSelection
from xdiyo_analytics.experiments import ExperimentStore
from xdiyo_analytics.ui import catalog_for_ui
from test_post_training_metrics import frames


def test_hand_calculation_and_unseen_target():
    y, p = frames([10, 5, 25], [[.2, .3, .5]] * 3, classes=[0, 10, 20])
    result = evaluate_metrics(y, p, ['count_ou_brier']).iloc[0]
    assert result.value == pytest.approx(.245)
    assert (result.n, result.n_total, result.n_missing, result.status) == (3, 3, 0, 'ok')
    for i, expected in enumerate([.145, .445, .145]):
        assert evaluate_metrics(y.iloc[[i]], {k:v.iloc[[i]] for k,v in p.items()},
                                ['count_ou_brier']).iloc[0].value == pytest.approx(expected)
    # One line: P(under)=.2, observed events are 0,1,0 respectively.
    request = Metric('count_ou_brier', parameters={'lines':[7.5]}, output='calibrated', key='ou')
    row = evaluate_metrics(y, {'calibrated':p['predict_proba']}, [request]).iloc[0]
    assert row.value == pytest.approx((.04+.64+.04)/3)
    assert row.metric == 'ou' and row.output == 'calibrated'
    assert json.loads(row.parameters) == {'lines':[7.5]}


@pytest.mark.parametrize('lines', [[], None, True, '6.5', [True], [np.nan], [np.inf],
    [0.5,0.5], [0], [-.5], [7], [7.25], ['7.5'], [6.5,7.5,6.5], {6.5:1}, {6.5,7.5}])
def test_invalid_lines(lines):
    y,p = frames([25], [[.5,.5]], classes=[1,24])
    with pytest.raises(ValueError, match='lines'):
        evaluate_metrics(y,p,[Metric('count_ou_brier', parameters={'lines':lines})])


@pytest.mark.parametrize('target', [None, pd.NA, np.nan, np.inf, -1, .5, True, '25'])
def test_invalid_targets_never_masked(target):
    y,p = frames([target], [[.5,.5]], classes=[1,24])
    with pytest.raises(ValueError, match='Targets'):
        evaluate_metrics(y,p,['count_ou_brier'])


@pytest.mark.parametrize('classes', [[0,0], [-1,1], [.5,1], [0,np.inf], ['0','1'], [False,True]])
def test_invalid_count_classes(classes):
    y,p = frames([1], [[.5,.5]], classes=classes)
    with pytest.raises(ValueError): evaluate_metrics(y,p,['count_ou_brier'])


@pytest.mark.parametrize('vector', [[np.nan,.5], [np.inf,.5], [-.1,1.1], [.2,.3]])
def test_invalid_probability_vectors_never_masked(vector):
    y,p = frames([25], [vector], classes=[1,24])
    with pytest.raises(ValueError, match='complete finite'):
        evaluate_metrics(y,p,['count_ou_brier'])


def test_one_class_and_normalization_tolerance_without_renormalizing():
    y,p = frames([25], [[1.]], classes=[10])
    assert evaluate_metrics(y,p,['count_ou_brier']).iloc[0].value == .5
    y,p = frames([25], [[.2,.8000005]], classes=[0,20])
    expected = (.2**2 + (.8000005-1)**2)/2
    assert evaluate_metrics(y,p,['count_ou_brier']).iloc[0].value == pytest.approx(expected, abs=1e-15)


def training_sample(*, differing=False, repeat=False):
    from xdiyo_analytics.training import FoldResult, TrainingResult
    from post_training_samples import RecordedModel
    folds=[]
    for fid,positions,truth in [(3,[0],[10]),(8,[0,2] if repeat else [1,2],[10,25] if repeat else [5,25])]:
        y=pd.DataFrame({'target':truth},index=pd.Index(positions,name='row_position'))
        classes=[0,10,30] if differing and fid==8 else [0,10,20]
        values=[.2,.3,.5]
        p=pd.DataFrame([values]*len(y),index=y.index,columns=pd.MultiIndex.from_product([['target'],classes]))
        meta=pd.DataFrame({'event_id':np.array(positions,dtype='uint64')+2**63},index=y.index)
        folds.append(FoldResult(fid,RecordedModel(),np.array([],dtype=int),np.array(positions),np.array(positions),
            ('synthetic',),('target',),{'predict_proba':p},y,meta))
    return TrainingResult(folds,'match',('event_id',),('event_id',),'total',{})


def report(training, pooling=None, mode='overall'):
    return PostTrainingAnalysis({'selection_metrics':PerformanceReporter(type=mode, partition='score',
        pooling=pooling, metrics=[Metric('count_ou_brier',key='ou_brier')])}).run(training)


@pytest.mark.parametrize('differing', [False,True])
def test_unequal_folds_pool_matches_preserving_identity(differing):
    training=training_sample(differing=differing)
    original=deepcopy(training.folds[0].predictions)
    pooled=report(training).studies[0]
    row=pooled.result.tables['metrics'].iloc[0]
    assert row.value==pytest.approx(.245) and row.value!=pytest.approx(.22)
    assert (row.n,row.n_total,row.n_missing,row.status)==(3,3,0,'ok')
    assert pooled.scope.row_position.tolist()==[0,1,2]
    assert row.sample_hash==report(training_sample()).studies[0].result.tables['metrics'].iloc[0].sample_hash
    per=report(training,mode='per_fold').studies
    assert [s.result.tables['metrics'].iloc[0].value for s in per]==pytest.approx([.145,.295])
    pd.testing.assert_frame_equal(original['predict_proba'],training.folds[0].predictions['predict_proba'])


def test_source_nan_invalid_and_mean_explicitly_unsupported():
    training=training_sample(differing=True)
    training.folds[1].predictions['predict_proba'].iloc[0,0]=np.nan
    with pytest.raises(ValueError,match='complete finite'):report(training)
    with pytest.raises(ValueError,match='mean pooling'):report(training_sample(),pooling='mean')


@pytest.mark.parametrize('pooling,n', [('occurrences',3),('first',2),('last',2)])
def test_repeated_rows_obey_existing_policy(pooling,n):
    training=training_sample(differing=True,repeat=True)
    row=report(training,pooling).studies[0].result.tables['metrics'].iloc[0]
    assert row.n==n and row.n_missing==0
    assert row.value==pytest.approx(.145)
    with pytest.raises(ValueError,match='explicit pooling'):report(training)


def test_recipe_catalog_and_metric_inventory_roundtrip():
    from pathlib import Path
    import xdiyo_analytics.ui.build_inventory as builder
    catalog=catalog_for_ui()
    request=Metric('count_ou_brier',target='total_corners',output='predict_proba',
        parameters={'lines':list(DEFAULT_OU_LINES)},key='ou_brier',direction='minimize')
    selector={'study':'selection_metrics','type':'overall','partition':'score','target':'total_corners','output':'predict_proba'}
    decision=MetricSelection('ou_brier',direction='minimize',selector=selector)
    fragment={'metrics':['mae',catalog.encode(request)],'decision':catalog.encode(decision)}
    restored=catalog.build(json.loads(json.dumps(fragment)),{})
    assert restored['metrics'][1]==request and restored['decision']==decision
    inventory=json.loads(Path(builder.__file__).with_name('inventory.json').read_text(encoding='utf-8'))
    metric=next(m for m in inventory['metrics'] if m['name']=='count_ou_brier')
    assert metric['kind']=='probability' and metric['direction']=='minimize'
    assert metric['fields'][0]['kind']=='list' and metric['fields'][0]['item']['kind']=='number'
    assert metric['fields'][0]['default']==list(DEFAULT_OU_LINES)


def test_target_expansion_alignment_and_empty_population():
    y,p=frames([25],[[.2,.3,.5]],classes=[0,10,20])
    y['second']=5
    p['predict_proba']=pd.concat([p['predict_proba'],p['predict_proba'].rename(columns={'target':'second'},level=0)],axis=1)
    rows=evaluate_metrics(y,p,['count_ou_brier'])
    assert rows.target.tolist()==['target','second']
    assert rows.value.tolist()==pytest.approx([.145,.445])
    empty=evaluate_metrics(y.iloc[:0],{k:v.iloc[:0] for k,v in p.items()},['count_ou_brier'])
    assert empty.status.tolist()==['no_valid_observations']*2
    p['predict_proba'].index=pd.Index([999])
    with pytest.raises(ValueError,match='index/order'):evaluate_metrics(y,p,['count_ou_brier'])


def test_first_last_choose_source_distributions_and_no_source_is_not_zero_filled():
    training=training_sample(differing=True,repeat=True)
    training.folds[1].predictions['predict_proba'].iloc[0]=[1.,0.,0.]
    assert report(training,'first').studies[0].result.tables['metrics'].iloc[0].value==pytest.approx(.145)
    assert report(training,'last').studies[0].result.tables['metrics'].iloc[0].value==pytest.approx((.5+.145)/2)
    assert report(training,'occurrences').studies[0].result.tables['metrics'].iloc[0].value==pytest.approx((.5+.145*2)/3)
    from xdiyo_analytics.analysis.post_training import _population
    y,p,meta=_population(training,training.folds,'score','occurrences')
    with pytest.raises(ValueError,match='complete finite'):
        evaluate_metrics(y,p,['count_ou_brier'],metadata=meta)


FITS=[]


class CountAdapter:
    def __init__(self, probability):self.probability=probability
    def fit(self,context):
        FITS.append(1)
        return self
    def predict(self,context):
        index=context.X.index
        return {'predict':pd.DataFrame({'target':[10]*len(index)},index=index),
                'predict_proba':pd.DataFrame([[1-self.probability,self.probability]]*len(index),
                    index=index,columns=pd.MultiIndex.from_product([['target'],[0,20]]))}


@dataclass
class CountFactory:
    probability: float
    def __call__(self):return CountAdapter(self.probability)


def test_native_selection_replays_changed_objective_without_refitting(tmp_path,monkeypatch):
    from model_selection_samples import sample,plan,development
    data=sample()
    data.y['target']=25  # outside the fitted support for every scoring row
    candidates=[Candidate(name,CountFactory(p),config={'p':p})
                for name,p in [('weak',.2),('best_first',.8),('best_tie',.8)]]
    store=ExperimentStore(tmp_path,'Count replay')
    FITS.clear()
    first=ModelSelection(candidates,metrics=['mae']).run(data,plan(data),
        development_positions=development(data),experiment=store,resume=True)
    assert first.winner.candidate.name=='weak' and len(FITS)==6
    def no_fit(*args,**kwargs):raise AssertionError('Cached inner candidates must not be refitted')
    # Patch the runner, leaving candidate fingerprints unchanged.
    import xdiyo_analytics.selection.core as core
    monkeypatch.setattr(core,'_fit_candidate',no_fit)
    request=Metric('count_ou_brier',key='ou_brier',target='target',output='predict_proba')
    decision=MetricSelection('ou_brier',selector={'study':'selection_metrics','type':'overall',
        'partition':'score','target':'target','output':'predict_proba'})
    second=ModelSelection(candidates,metrics=['mae',request],decision=decision).run(data,plan(data),
        development_positions=development(data),experiment=store,resume=True)
    assert second.winner.candidate.name=='best_first' and len(FITS)==6
    assert [t.saved_run_id for t in first.trials]==[t.saved_run_id for t in second.trials]
    assert [next(m['value'] for m in t.record['metrics'] if m['metric']=='ou_brier' and m['type']=='overall')
            for t in second.trials]==pytest.approx([.64,.04,.04])
