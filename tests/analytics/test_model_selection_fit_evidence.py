from dataclasses import replace
import math
import numpy as np
import pandas as pd
import pytest
from model_selection_samples import sample,plan,development,FixedAdapter,candidate
from split_samples import rows_for
from xdiyo_analytics.selection import Candidate,FitStatistics,ModelSelection,MetricSelection,ParsimonySelection

class GaussianAdapter(FixedAdapter):
    def __init__(self,linear=False):super().__init__();self.linear=linear
    def fit(self,context):
        super().fit(context)
        x=context.X.signal.to_numpy();y=context.y.iloc[:,0].to_numpy()
        self.design=np.c_[np.ones(len(x)),x] if self.linear else np.ones((len(x),1))
        self.beta=np.linalg.lstsq(self.design,y,rcond=None)[0]
        self.residual=y-self.design@self.beta
        return self
    def predict(self,context):
        x=context.X.signal.to_numpy();design=np.c_[np.ones(len(x)),x] if self.linear else np.ones((len(x),1))
        return {'predict':pd.DataFrame({'target':design@self.beta},index=context.X.index)}
    def fit_statistics(self):
        n=len(self.residual)
        ll=-.5*n*math.log(2*math.pi*4)-np.dot(self.residual,self.residual)/8
        return FitStatistics(float(ll),len(self.beta),n,'Gaussian known variance 4; constants included; k=regression coefficients')

def gaussian(name,linear=False,**kwargs):return Candidate(name,lambda:GaussianAdapter(linear),config={'linear':linear,'variance':4},**kwargs)

def test_per_fit_aic_bic_and_explicit_arithmetic_mean_with_unequal_fit_sizes():
    data=sample();splits=plan(data)
    result=ModelSelection([gaussian('constant'),gaussian('linear',True)],metrics='mse',
        information_criteria=True,decision=MetricSelection('aic')).run(data,splits,development_positions=development(data))
    assert result.winner.candidate.name=='linear'
    for trial in result.trials:
        expected=[]
        for fold in splits.folds:
            x=data.X.signal.iloc[fold.train].to_numpy();y=data.y.target.iloc[fold.train].to_numpy();n=len(y)
            if trial.candidate.name=='constant':fitted=np.full(n,y.mean());k=1
            else:
                slope=np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean())
                fitted=y.mean()+slope*(x-x.mean());k=2
            ll=-.5*n*math.log(8*math.pi)-sum((y-fitted)**2)/8
            expected.append((ll,-2*ll+2*k,-2*ll+k*math.log(n)))
        fits=trial.report.studies[-1].result.tables['fit_statistics']
        np.testing.assert_allclose(fits[['log_likelihood','aic','bic']].to_numpy(),expected,rtol=1e-11,atol=1e-11)
        for key in ('aic','bic'):
            metric=next(m for m in trial.record['metrics'] if m['metric']==key)
            column=1 if key=='aic' else 2
            assert metric['value']==pytest.approx(np.mean(np.asarray(expected)[:,column]))
            assert metric['calculation']=='mean_fold_'+key and metric['n']==2
            assert metric['value']!=pytest.approx(np.sum(np.asarray(expected)[:,column]))
        assert trial.complexity['n_parameters']==(2 if trial.candidate.name=='linear' else 1)

@pytest.mark.parametrize('difference',['likelihood_id','n_observations','fitted_sample'])
def test_information_criterion_comparison_rejects_mismatched_convention_or_fit_population(difference):
    data=sample();kwargs={}
    def provider(model):
        stats=model.fit_statistics()
        if difference=='likelihood_id':return replace(stats,likelihood_id='Other units or normalization')
        if difference=='n_observations':return replace(stats,n_observations=stats.n_observations+1)
        return stats
    if difference=='fitted_sample':kwargs['validation']={0:rows_for(data,[4]),1:rows_for(data,[7])}
    values=[gaussian('one'),gaussian('two',True,fit_statistics=provider,**kwargs)]
    with pytest.raises(ValueError,match='incompatible'):
        ModelSelection(values,information_criteria=True,decision=MetricSelection('bic')).run(data,plan(data),development_positions=development(data))

def test_fit_stat_callback_overrides_method_and_declares_sampling_units():
    data=sample(layout='team_match')
    def provider(model):return FitStatistics(-3.,2.5,len(model.context.X)//2,'joint paired sampling unit; effective k=2.5')
    result=ModelSelection([gaussian('paired',fit_statistics=provider)],information_criteria=True,
        decision=MetricSelection('aic')).run(data,plan(data),development_positions=development(data))
    fits=result.winner.report.studies[-1].result.tables['fit_statistics']
    assert fits.n_observations.tolist()==[5,8] and fits.aic.tolist()==[11.,11.]
    assert result.winner.complexity['n_parameters']==2.5

def test_unrequested_fit_statistics_are_never_called():
    data=sample()
    def unavailable(model):raise AssertionError('fit statistics were not requested')
    result=ModelSelection([candidate('numeric',fit_statistics=unavailable)],metrics='mse').run(data,plan(data),development_positions=development(data))
    assert result.winner.report.studies[-1].result.tables['fit_statistics'].empty

@pytest.mark.parametrize('problem',['missing_provider','invalid_stat','complexity_disagreement'])
def test_invalid_or_missing_fit_evidence_can_be_recorded_as_failed_trial(problem):
    data=sample()
    bad=candidate('bad') if problem=='missing_provider' else gaussian('bad',fit_statistics=lambda m:FitStatistics(np.nan,1,5,'test')) if problem=='invalid_stat' else gaussian('bad',complexity=lambda m:{'n_parameters':99})
    result=ModelSelection([bad,gaussian('good',True)],information_criteria=True,decision=MetricSelection('aic'),on_error='record').run(data,plan(data),development_positions=development(data))
    assert result.trials[0].error and result.trials[0].record['status']=='failed'
    assert result.winner.candidate.name=='good'

def test_missing_complexity_on_one_fold_does_not_average_available_folds_only():
    data=sample()
    incomplete=candidate('incomplete',complexity=lambda model:{} if model.context.fold_id==1 else {'size':1.})
    complete=candidate('complete',complexity=lambda model:{'size':2.+model.context.fold_id})
    result=ModelSelection([incomplete,complete],metrics='mse',decision=ParsimonySelection('mse',complexity='size')).run(data,plan(data),development_positions=development(data))
    assert np.isnan(result.trials[0].complexity['size'])
    assert result.trials[1].complexity['size']==2.5 and result.winner.candidate.name=='complete'

@pytest.mark.parametrize('measure',[{'size':-1},{'size':np.inf},{'size':True},{'':1}])
def test_complexity_values_have_explicit_nonnegative_finite_contract(measure):
    data=sample()
    with pytest.raises(ValueError,match='Complexity'):
        ModelSelection([candidate('bad',complexity=lambda m:measure)],metrics='mse').run(data,plan(data),development_positions=development(data))
