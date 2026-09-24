from copy import deepcopy
import pytest
from model_selection_samples import candidate,FixedAdapter
from xdiyo_analytics.selection import Candidate,GridCandidates

def test_grid_cartesian_conditional_empty_and_stable_names():
    grids=[{'alpha':[.1,1.],'ratio':[.2,.8]}, {}, {'kind':['ridge']}]
    build=lambda parameters:Candidate('model',FixedAdapter,config={'description':'explicit'})
    first=list(GridCandidates(grids,build));second=list(GridCandidates(grids,build))
    expected=[{'alpha':.1,'ratio':.2},{'alpha':.1,'ratio':.8},{'alpha':1.,'ratio':.2},
              {'alpha':1.,'ratio':.8},{},{'kind':'ridge'}]
    assert [c.config['grid_parameters'] for c in first]==expected
    assert [c.name for c in first]==[f'model [{i}]' for i in range(1,7)]
    assert [c.config for c in first]==[c.config for c in second]

def test_grid_copies_values_before_builder_and_between_candidates():
    original={'layers':[[1,2],[3,4]]};before=deepcopy(original)
    def build(params):
        params['layers'].append(99)
        return Candidate('net',FixedAdapter,config={'builder_value':params})
    values=list(GridCandidates(original,build));values[0].config['grid_parameters']['layers'].append(88)
    assert original==before and values[1].config['grid_parameters']=={'layers':[3,4]}
    assert values[0].config['builder_value']['layers']==[1,2,99]

@pytest.mark.parametrize('parameters',[{'x':[]},{'x':'abc'},{'x':b'ab'},{'x':{'a':1}},[{1:[2]}],[None]])
def test_bad_grid_definitions_fail(parameters):
    with pytest.raises((ValueError,TypeError)):list(GridCandidates(parameters,lambda p:candidate('x')))

def test_grid_builder_must_return_candidate():
    with pytest.raises(TypeError,match='Candidate'):list(GridCandidates({},lambda p:object()))

@pytest.mark.parametrize('changes',[{'name':''},{'name':' '},{'model_factory':None},{'config':[]},
    {'config':{'a':float('nan')}},{'config':{'a':object()}},{'features_from':'pick'}])
def test_candidate_requires_explicit_serializable_identity_and_factory(changes):
    args=dict(name='good',model_factory=FixedAdapter);args.update(changes)
    with pytest.raises((ValueError,TypeError)):Candidate(**args)
