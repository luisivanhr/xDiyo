from functools import partial
import types
import pandas as pd
import pytest
from xdiyo_analytics.experiments.recovery import execution_key,signature

def function(source,filename='<ipython-input-1>',padding=0,**globals):
    scope={'__name__':'__notebook__',**globals}
    exec(compile('\n'*padding+source,filename,'exec'),scope)
    return scope['factory']

def test_notebook_locations_do_not_change_nested_code_identity():
    source='def factory(alpha=.1):\n    def inner(x):\n        return x+alpha\n    return inner\n'
    one=function(source);two=function(source,'<ipython-input-999>',80)
    assert execution_key(one)==execution_key(two)
    assert execution_key(one)!=execution_key(function(source.replace('x+alpha','x+alpha+1')))

@pytest.mark.parametrize('factor',['default','keyword_default','closure','global','partial'])
def test_factory_configuration_changes_invalidate_identity(factor):
    if factor=='default':
        one=function('def factory(alpha=.1): return alpha');two=function('def factory(alpha=.2): return alpha')
    elif factor=='keyword_default':
        one=function('def factory(*,alpha=.1): return alpha');two=function('def factory(*,alpha=.2): return alpha')
    elif factor=='closure':
        def make(alpha):return lambda:alpha
        one,two=make(.1),make(.2)
    elif factor=='global':
        one=function('def factory(): return ALPHA',ALPHA=.1);two=function('def factory(): return ALPHA',ALPHA=.2)
    else:
        base=function('def factory(alpha): return alpha');one,two=partial(base,.1),partial(base,.2)
    assert execution_key(one)!=execution_key(two)

def test_custom_opaque_callable_declares_configuration_without_being_called():
    class Factory:
        def __init__(self,alpha):self.alpha=alpha;self.calls=0
        def cache_key(self):return {'alpha':self.alpha}
        def __call__(self):self.calls+=1;raise AssertionError('identity must not fit')
    one,two=Factory(.1),Factory(.1)
    assert execution_key(one)==execution_key(two)
    one.calls=200
    assert execution_key(one)==execution_key(two)
    one.alpha=.2
    assert execution_key(one)!=execution_key(two)
    assert two.calls==0

def test_unknown_opaque_configuration_requires_explicit_identity():
    class Unknown:pass
    with pytest.raises(TypeError,match='cache_key'):execution_key(Unknown())

def test_recursive_function_globals_and_module_version_are_bounded():
    source='def factory(x=0): return factory(x-1) if x else 0'
    assert execution_key(function(source))==execution_key(function(source,'other',5))
    mod=types.ModuleType('synthetic_framework');mod.__version__='1'
    old=execution_key(mod);mod.__version__='2'
    assert execution_key(mod)!=old

def test_notebook_adapter_method_implementation_is_part_of_identity():
    def adapter(constant,filename):
        scope={'__name__':'__notebook__'}
        exec(compile(f'class Adapter:\n    def predict(self,x): return x+{constant}\n',filename,'exec'),scope)
        return scope['Adapter']
    assert execution_key(adapter(1,'cell1'))==execution_key(adapter(1,'cell99'))
    assert execution_key(adapter(1,'cell1'))!=execution_key(adapter(2,'cell1'))

def notebook_adapter(kind,scale,filename='<ipython-input-1>',padding=0):
    if kind=='class_attribute':
        source=f'class Adapter:\n    scale = {scale}\n'
    elif kind=='base_attribute':
        source=f'class Base:\n    scale = {scale}\nclass Adapter(Base):\n    pass\n'
    elif kind=='base_method':
        source=(f'class Base:\n    def coefficient(self): return {scale}\n'
                'class Adapter(Base):\n    @property\n    def scale(self): return self.coefficient()\n')
    else:
        source=f'class Adapter:\n    @property\n    def scale(self): return {scale}\n'
    source+=('    def fit(self,context):\n        self.columns = context.y.columns\n'
             '    def predict(self,context):\n'
             '        return {"predict": pd.DataFrame(self.scale,index=context.X.index,columns=self.columns)}\n')
    scope={'__name__':'__notebook__','pd':pd}
    exec(compile('\n'*padding+source,filename,'exec'),scope)
    return scope['Adapter']

@pytest.mark.parametrize('kind',['class_attribute','base_attribute','base_method','property'])
def test_notebook_class_state_and_bases_control_actual_final_reuse(tmp_path,monkeypatch,kind):
    from football_experiment_samples import prepared,post
    from xdiyo_analytics.experiments import FootballExperiment
    from xdiyo_analytics.experiments import football
    from xdiyo_analytics.selection import Candidate
    calls=[]
    fit=football._fit_candidate
    def count_fit(*args,**kwargs):
        calls.append(True)
        return fit(*args,**kwargs)
    monkeypatch.setattr(football,'_fit_candidate',count_fit)
    experiment=FootballExperiment('Notebook state '+kind,output_dir=tmp_path)
    data=prepared(holdout=True)
    first=experiment.run(data,model=Candidate('Notebook',notebook_adapter(kind,1)),post_analysis=post())
    unchanged=experiment.run(data,model=Candidate('Notebook',notebook_adapter(kind,1,'cell99',30)),post_analysis=post())
    assert unchanged.reused and unchanged.record['run_id']==first.record['run_id']
    assert len(calls)==1
    changed=experiment.run(data,model=Candidate('Notebook',notebook_adapter(kind,2,'cell100',40)),post_analysis=post())
    assert not changed.reused and changed.record['run_id']!=first.record['run_id']
    assert len(calls)==2
    assert (first.training.folds[0].predictions['predict']==1).all().all()
    assert (changed.training.folds[0].predictions['predict']==2).all().all()
    assert len(experiment.store.read_runs(role='final'))==2

@pytest.mark.parametrize('binding',['staticmethod','classmethod'])
def test_notebook_class_explicit_key_describes_opaque_state_without_construction(binding):
    def adapter(revision,filename):
        arguments='' if binding=='staticmethod' else 'cls'
        source=(f'class Adapter:\n    opaque = object()\n    @'+binding+'\n'
                f'    def cache_key({arguments}): return {{"revision": {revision}}}\n'
                '    def __init__(self): raise AssertionError("must not construct")\n')
        scope={'__name__':'__notebook__'}
        exec(compile(source,filename,'exec'),scope)
        return scope['Adapter']
    assert execution_key(adapter(1,'cell1'))==execution_key(adapter(1,'cell99'))
    assert execution_key(adapter(1,'cell1'))!=execution_key(adapter(2,'cell1'))

def test_notebook_class_descriptor_binding_and_recursive_state_have_stable_identity():
    def adapter(binding,filename,padding=0):
        scope={'__name__':'__notebook__'}
        source=(f'class Adapter:\n    @{binding}\n    def coefficient(cls): return 2\n'
                'Adapter.self = Adapter\nAdapter.options = {"owner": Adapter, "scale": 2}\n')
        exec(compile('\n'*padding+source,filename,'exec'),scope)
        return scope['Adapter']
    one=adapter('staticmethod','cell1')
    assert execution_key(one)==execution_key(adapter('staticmethod','cell99',30))
    assert execution_key(one)!=execution_key(adapter('classmethod','cell1'))
    one.options['scale']=3
    assert execution_key(one)!=execution_key(adapter('staticmethod','cell1'))

def test_notebook_class_opaque_state_requires_an_explicit_class_key():
    scope={'__name__':'__notebook__'}
    exec(compile('class Adapter:\n    opaque = object()\n','cell1','exec'),scope)
    with pytest.raises(TypeError,match='cache_key'):
        execution_key(scope['Adapter'])

@pytest.mark.parametrize('kind',['slots','annotations','dataclass'])
def test_notebook_class_generated_metadata_does_not_hide_declared_state(kind):
    def adapter(scale,filename,slots=('fitted',)):
        source='from typing import ClassVar\nfrom dataclasses import dataclass\n'
        if kind=='dataclass':
            source+='@dataclass(repr=False)\n'
        source+=f'class Adapter:\n    scale: ClassVar[int] = {scale}\n    features: list[str]\n'
        if kind=='slots':
            source+=f'    __slots__ = {slots!r}\n'
        source+='    def coefficient(self): return self.scale\n'
        scope={'__name__':'__notebook__'}
        exec(compile(source,filename,'exec'),scope)
        return scope['Adapter']
    assert execution_key(adapter(2,'cell1'))==execution_key(adapter(2,'cell99'))
    assert execution_key(adapter(2,'cell1'))!=execution_key(adapter(3,'cell1'))
    if kind=='slots':
        assert execution_key(adapter(2,'cell1'))!=execution_key(adapter(2,'cell1',('fitted','extra')))
