from dataclasses import replace
import math
import numpy as np
import pytest
from model_selection_samples import trial
from xdiyo_analytics.selection import FitStatistics,information_criteria,MetricSelection,WeightedSelection,ParsimonySelection

def test_scalar_direction_and_exact_ties_follow_candidate_order():
    values=[trial('z',{'mse':2.}),trial('a',{'mse':2.}),trial('m',{'mse':5.})]
    decision=MetricSelection('mse').decide(values)
    assert decision.winner_id=='z'
    assert decision.table.set_index('run_id').loc['z','selected']
    assert MetricSelection('mse',direction='maximize').decide(values).winner_id=='m'

def test_weighted_percentile_utilities_are_independent_hand_calculation():
    values=[trial('A',{'mse':1.,'accuracy':.2}),trial('B',{'mse':2.,'accuracy':.9}),trial('C',{'mse':3.,'accuracy':.5})]
    decision=WeightedSelection({'mse':1.,'accuracy':3.}).decide(values)
    assert decision.winner_id=='B'
    actual=decision.table.set_index('run_id')
    assert actual.loc[['A','B','C'],'score'].tolist()==[.25,.875,.375]
    assert actual.loc[['A','B','C'],'utility::mse'].tolist()==[1.,.5,0.]

def test_fixed_utility_clips_and_missing_required_evidence_stays_unranked():
    values=[trial('A',{'mse':-1.,'accuracy':.2}),trial('B',{'mse':2.,'accuracy':.9}),trial('missing',{'mse':0.})]
    decision=WeightedSelection({'mse':1.,'accuracy':1.},scaling='fixed',reference_scales={'mse':(0.,4.),'accuracy':(0.,1.)}).decide(values)
    frame=decision.table.set_index('run_id')
    assert decision.winner_id=='B' and frame.loc['A','score']==pytest.approx(.6)
    assert frame.loc['B','score']==pytest.approx(.7) and frame.loc['missing','status'].startswith('unranked')
    assert np.isnan(frame.loc['missing','score'])

def test_parsimony_absolute_boundary_selected_flag_can_differ_from_rank():
    values=[trial('best',{'mse':1.},complexity={'n_parameters':8}),
        trial('simple',{'mse':1.25},complexity={'n_parameters':2}),trial('too_far',{'mse':1.26},complexity={'n_parameters':1})]
    decision=ParsimonySelection('mse',tolerance=.25).decide(values)
    table=decision.table.set_index('run_id')
    assert decision.winner_id=='simple' and table.loc['simple','selected'] and table.loc['simple','rank']==2
    assert not table.loc['best','selected'] and not table.loc['too_far','within_tolerance']

def test_parsimony_complexity_then_metric_then_candidate_order():
    values=[trial('best_missing',{'mse':1.}),trial('z',{'mse':1.2},complexity={'size':2}),
        trial('worse',{'mse':1.3},complexity={'size':2}),trial('a',{'mse':1.2},complexity={'size':2})]
    assert ParsimonySelection('mse',complexity='size',tolerance=.5).decide(values).winner_id=='z'

@pytest.mark.parametrize('rule',[MetricSelection('mse'),WeightedSelection({'mse':1}),ParsimonySelection('mse')])
def test_incompatible_samples_never_choose_across_groups(rule):
    with pytest.raises(ValueError,match='incompatible'):
        rule.decide([trial('a',{'mse':1.},sample_hash='a'),trial('b',{'mse':2.},sample_hash='b')])

def test_metric_selector_disambiguates_target_and_missing_all_raises():
    a,b=trial('a',{'mse':2.}),trial('b',{'mse':1.})
    for t in (a,b):t.record['metrics'].append(dict(t.record['metrics'][0],target='other',value=100.))
    with pytest.raises(ValueError,match='ambiguous'):MetricSelection('mse').decide([a,b])
    assert MetricSelection('mse',selector={'target':'target'}).decide([a,b]).winner_id=='b'
    with pytest.raises(ValueError,match='No candidate'):MetricSelection('mse').decide([trial('x',{})])

def test_aic_bic_independent_arithmetic_allows_declared_effective_complexity():
    result=information_criteria(FitStatistics(-12.5,2.5,20,'gaussian with constants; k includes variance'))
    assert result['aic']==30. and result['bic']==pytest.approx(25+2.5*math.log(20))
    assert information_criteria(FitStatistics(1.,0.,1,'known model'))=={'aic':-2.,'bic':-2.}

@pytest.mark.parametrize('field,value',[('log_likelihood',np.nan),('log_likelihood',np.inf),('log_likelihood',True),
    ('n_parameters',-1),('n_parameters',np.inf),('n_parameters','two'),('n_observations',0),('n_observations',-1),
    ('n_observations',2.5),('n_observations',True),('likelihood_id',''),('likelihood_id',None)])
def test_invalid_fit_statistics_fail(field,value):
    with pytest.raises((ValueError,TypeError)):
        information_criteria(replace(FitStatistics(-10.,2.,10,'test convention'),**{field:value}))

def test_fit_statistics_provider_contract_is_explicit():
    with pytest.raises(TypeError):information_criteria({'log_likelihood':-10,'n_parameters':2,'n_observations':10})
