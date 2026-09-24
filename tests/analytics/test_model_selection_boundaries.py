from copy import deepcopy
import numpy as np
import pytest
from model_selection_samples import sample,plan,development,candidate,FixedAdapter
from split_samples import rows_for
from xdiyo_analytics.reporting import PerformanceReporter
from xdiyo_analytics.selection import Candidate,ModelSelection
from xdiyo_analytics.splits import Fold,SplitPlan

@pytest.mark.parametrize('problem',['overlap','score_not_test','partial_train','partial_test','partial_score'])
def test_declarative_inner_partitions_raise_before_model_construction(problem):
    data=sample(layout='team_match');splits=plan(data);calls=[]
    fold=splits.folds[0]
    if problem=='overlap':fold.test=np.r_[fold.test,rows_for(data,[1])]
    elif problem=='score_not_test':fold.score=rows_for(data,[3])
    else:
        field=problem.removeprefix('partial_');setattr(fold,field,getattr(fold,field)[:-1])
    def build():calls.append(1);return FixedAdapter()
    with pytest.raises(ValueError,match='disjoint|subset|whole matches'):
        ModelSelection([Candidate('x',build)],metrics='mse',on_error='record').run(data,splits,development_positions=development(data))
    assert calls==[]

@pytest.mark.parametrize('problem',['reserved_name','wrong_partition','validation_outside_development'])
def test_scope_configuration_cannot_be_recorded_as_an_ordinary_failed_fit(problem):
    data=sample();calls=[];options={};candidate_options={}
    if problem=='reserved_name':options['evidence_reporters']={'fit_evidence':PerformanceReporter(type='overall',partition='score',metrics=['mae'])}
    elif problem=='wrong_partition':options['evidence_reporters']={'extra':PerformanceReporter(type='overall',partition='test',metrics=['mae'])}
    else:candidate_options['validation']=rows_for(data,[14])
    def build():calls.append(1);return FixedAdapter()
    with pytest.raises(ValueError):
        ModelSelection([Candidate('x',build,**candidate_options)],metrics='mse',on_error='record',**options).run(data,plan(data),development_positions=development(data))
    assert calls==[]

@pytest.mark.parametrize('pooling',[None,'occurrences','first','last','mean'])
def test_repeated_score_rows_require_explicit_pooling_and_use_prediction_level_arithmetic(pooling):
    data=sample();made=[]
    splits=SplitPlan([Fold(rows_for(data,range(5)),rows_for(data,[6,7]),rows_for(data,[7])),
                      Fold(rows_for(data,range(6)),rows_for(data,[7,10,11]),rows_for(data,[7,10,11]))],len(data.X),np.arange(len(data.X)))
    def build():
        model=FixedAdapter(value=10.*len(made));made.append(model);return model
    search=ModelSelection([Candidate('x',build)],metrics='mse',pooling=pooling)
    if pooling is None:
        with pytest.raises(ValueError,match='pooling'):search.run(data,splits,development_positions=development(data))
        return
    result=search.run(data,splits,development_positions=development(data))
    cases=[7,7,10,11] if pooling=='occurrences' else [7,10,11]
    predicted=[0.,10.,10.,10.] if pooling=='occurrences' else [dict(first=0.,last=10.,mean=5.)[pooling],10.,10.]
    truth=data.y.target.iloc[rows_for(data,cases)].to_numpy()
    if pooling=='occurrences':truth=data.y.target.iloc[[7,7,10,11]].to_numpy()
    expected=np.mean((truth-np.array(predicted))**2)
    assert result.comparison['raw::mse'].iloc[0]==pytest.approx(expected)
    assert result.winner.training.folds[0].score_positions.tolist()==[7]
    assert result.winner.training.folds[1].score_positions.tolist()==[7,10,11]
