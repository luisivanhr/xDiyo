from copy import deepcopy
import numpy as np
import pytest
from model_selection_samples import sample,FixedAdapter
from split_samples import rows_for,unchanged
from xdiyo_analytics.selection import Candidate,ModelSelection
from xdiyo_analytics.splits import Fold,SplitPlan

def outer_plan(data):
    folds=[Fold(rows_for(data,range(6))[::-1],rows_for(data,[6,7]),rows_for(data,[7])),
           Fold(rows_for(data,range(8,14)),rows_for(data,[14,15])[::-1],rows_for(data,[14,15]))]
    return SplitPlan(folds,len(data.X),np.arange(len(data.X)))

def inner_plan(data,*,one=False):
    cases=list(dict.fromkeys(data.metadata.case.tolist()))
    rows=lambda ids:rows_for(data,[cases[i] for i in ids])
    folds=[Fold(rows([0,1]),rows([2,3]),rows([3]))]
    if not one:folds.append(Fold(rows([0,1,2,3]),rows([4,5]),rows([4,5])))
    return SplitPlan(folds,len(data.X),np.arange(len(data.X)))

@pytest.mark.parametrize('layout',['match','team_match'])
@pytest.mark.parametrize('source_kind',['one_shot','callable'])
def test_nested_search_independent_winners_original_rows_and_no_outer_access(layout,source_kind):
    data=sample(layout=layout,shuffle=True);data.y['target']=np.where(data.metadata.case<8,0.,10.)
    original=deepcopy(data);seen=[];sources=[];models=[]
    def make(value):
        model=FixedAdapter(value);models.append(model);return model
    def candidates():
        sources.append(1)
        for value in (0.,10.):yield Candidate(str(value),lambda value=value:make(value),config={'constant':value})
    def inner(scoped):
        seen.append((set(scoped.metadata.case),scoped.X.index.tolist(),len(scoped.X)))
        assert set(scoped.metadata.case) in [set(range(6)),set(range(8,14))]
        assert scoped.X.index.tolist()==list(range(len(scoped.X)))
        scoped.definitions['inspection_only']='copied'
        return inner_plan(scoped)
    source=candidates if source_kind=='callable' else candidates()
    result=ModelSelection(source,metrics='mse').run_nested(data,outer_plan(data),inner)
    assert [s.winner.candidate.name for s in result.selections.values()]==['0.0','10.0']
    assert len(sources)==(2 if source_kind=='callable' else 1)
    assert len(models)==10 and len({id(m) for m in models})==10
    assert result.comparison.outer_fold.tolist().count(0)==2 and result.comparison.outer_fold.tolist().count(1)==2
    for outer_id,(stored,expected) in enumerate(zip(result.training.folds,outer_plan(data).folds)):
        assert stored.fold_id==outer_id and stored.test_positions.tolist()==expected.test.tolist()
        assert stored.score_positions.tolist()==expected.score.tolist()
        assert stored.predictions['predict'].index.tolist()==expected.test.tolist()
        assert stored.metadata.event_id.tolist()==data.metadata.iloc[expected.test].event_id.tolist()
        np.testing.assert_allclose(stored.predictions['predict'].target,0. if outer_id==0 else 10.)
        assert set(stored.model.context.metadata.case)==set(data.metadata.iloc[expected.train].case)
    unchanged(data,original)

def test_changing_outer_outcomes_does_not_change_nested_winners():
    data=sample();data.y['target']=np.where(data.metadata.case<8,0.,10.)
    changed=deepcopy(data);changed.y.iloc[rows_for(data,[6,7,14,15])]=123456.
    def candidates():return [Candidate(str(v),lambda v=v:FixedAdapter(v)) for v in (0.,10.)]
    search=ModelSelection(candidates,metrics='mse')
    first=search.run_nested(data,outer_plan(data),inner_plan)
    second=search.run_nested(changed,outer_plan(changed),inner_plan)
    assert [s.winner.candidate.name for s in first.selections.values()]==[s.winner.candidate.name for s in second.selections.values()]
    for a,b in zip(first.training.folds,second.training.folds):
        assert not a.y_true.equals(b.y_true)
        assert a.predictions['predict'].equals(b.predictions['predict'])

def test_freshness_guard_spans_separate_outer_iterations():
    data=sample();made=[]
    def factory():
        if len(made)==2:return made[0]
        value=FixedAdapter();made.append(value);return value
    with pytest.raises(ValueError,match='fresh|reuse'):
        ModelSelection([Candidate('reused across outer folds',factory)],metrics='mse').run_nested(data,outer_plan(data),lambda d:inner_plan(d,one=True))
    assert len(made)==2

def test_inner_factory_must_return_plan_for_local_population():
    data=sample()
    with pytest.raises(ValueError,match='supplied development dataset'):
        ModelSelection([Candidate('x',FixedAdapter)],metrics='mse').run_nested(data,outer_plan(data),lambda d:SplitPlan([],len(data.X),np.arange(len(data.X))))
