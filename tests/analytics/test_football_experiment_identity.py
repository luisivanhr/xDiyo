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

@pytest.mark.parametrize('factor',['default','keyword_default','closure','global','partial','code'])
@pytest.mark.parametrize('explicit_key',[False,True])
def test_factory_configuration_changes_invalidate_identity(factor,explicit_key):
    if factor=='default':
        one=function('def factory(alpha=.1): return alpha');two=function('def factory(alpha=.2): return alpha')
    elif factor=='keyword_default':
        one=function('def factory(*,alpha=.1): return alpha');two=function('def factory(*,alpha=.2): return alpha')
    elif factor=='closure':
        def make(alpha):return lambda:alpha
        one,two=make(.1),make(.2)
    elif factor=='global':
        one=function('def factory(): return ALPHA',ALPHA=.1);two=function('def factory(): return ALPHA',ALPHA=.2)
    elif factor=='code':
        one=function('def factory(): return 1');two=function('def factory(): return 2')
    else:
        base=function('def factory(alpha): return alpha');one,two=partial(base,.1),partial(base,.2)
    if explicit_key:
        for factory in (one,two):
            target=factory.func if isinstance(factory,partial) else factory
            target.cache_key=lambda:{'revision':1}
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

def test_local_model_class_global_change_refits_and_preserves_unchanged_reuse(tmp_path,monkeypatch):
    from football_experiment_samples import prepared,post
    from xdiyo_analytics.experiments import FootballExperiment
    from xdiyo_analytics.experiments import football
    from xdiyo_analytics.selection import Candidate
    source=('def factory():\n'
            '    class Adapter:\n'
            '        def fit(self,context): self.columns = context.y.columns\n'
            '        def predict(self,context):\n'
            '            return {"predict": pd.DataFrame(SCALE,index=context.X.index,columns=self.columns)}\n'
            '    return Adapter()\n')
    factory=function(source,pd=pd,SCALE=1)
    calls=[]
    fit=football._fit_candidate
    def count_fit(*args,**kwargs):
        calls.append(True)
        return fit(*args,**kwargs)
    monkeypatch.setattr(football,'_fit_candidate',count_fit)
    experiment=FootballExperiment('Nested notebook globals',output_dir=tmp_path)
    data=prepared(holdout=True)
    first=experiment.run(data,model=Candidate('Local model',factory),post_analysis=post())
    unchanged=function(source,'cell99',30,SCALE=1,pd=pd,UNUSED='new unrelated value')
    restored=experiment.run(data,model=Candidate('Local model',unchanged),post_analysis=post())
    assert restored.reused and restored.record['run_id']==first.record['run_id'] and len(calls)==1
    factory.__globals__['SCALE']=2
    changed=experiment.run(data,model=Candidate('Local model',factory),post_analysis=post())
    assert not changed.reused and changed.record['run_id']!=first.record['run_id'] and len(calls)==2
    assert (first.training.folds[0].predictions['predict']==1).all().all()
    assert (changed.training.folds[0].predictions['predict']==2).all().all()

@pytest.mark.parametrize('body',[
    '    def inner(): return SCALE + OFFSET\n    return inner\n',
    '    return (SCALE * item + OFFSET for item in [1,2])\n',
    '    def inner(): return [SCALE * item + OFFSET for item in [1,2]]\n    return inner\n',
])
def test_nested_function_and_comprehension_globals_are_stable_and_complete(body):
    source='def factory():\n'+body
    one=function(source,SCALE=2,OFFSET=1,UNUSED=3)
    two=function(source,'cell99',30,UNUSED=4,OFFSET=1,SCALE=2)
    assert execution_key(one)==execution_key(two)
    assert execution_key(one)!=execution_key(function(source,SCALE=3,OFFSET=1,UNUSED=3))
    assert execution_key(one)!=execution_key(function(source,SCALE=2,OFFSET=2,UNUSED=3))
    assert list(signature(one)['globals'])==['OFFSET','SCALE']

def test_globals_referenced_only_by_nested_code_keep_cycles_bounded():
    source=('def factory():\n    def inner(): return OTHER()\n    return inner\n'
            'def other():\n    def inner(): return factory()\n    return inner\n'
            'OTHER = other\n')
    assert execution_key(function(source))==execution_key(function(source,'cell99',30))

def test_nested_class_attributes_do_not_capture_unread_same_named_globals():
    source=('def factory():\n    class Adapter:\n        fitted = False\n'
            '        def fit(self): self.fitted = True\n'
            '        def predict(self): return self.fitted\n    return Adapter()\n')
    one=function(source,fitted=object(),fit=object(),predict=object())
    two=function(source,'cell99',30,predict=object(),fit=object(),fitted=object())
    assert execution_key(one)==execution_key(two)
    assert list(signature(one)['globals'])==['__name__']


def _attributed_factory():
    from model_selection_samples import FixedAdapter
    owner=_attributed_factory
    if hasattr(owner,'calls'):
        owner.calls+=1
    return FixedAdapter(owner.scale)


@pytest.mark.parametrize('origin',['sourceful','notebook'])
@pytest.mark.parametrize('explicit_key',[False,True])
def test_function_factory_state_controls_actual_reuse_and_predictions(tmp_path,monkeypatch,origin,explicit_key):
    from football_experiment_samples import prepared
    from xdiyo_analytics.experiments import FootballExperiment,football
    from xdiyo_analytics.selection import Candidate
    source=('def factory():\n    from model_selection_samples import FixedAdapter\n'
            '    if hasattr(factory,"calls"): factory.calls+=1\n'
            '    return FixedAdapter(factory.scale)\n')
    factory=_attributed_factory if origin=='sourceful' else function(source)
    monkeypatch.setattr(factory,'scale',1,raising=False)
    if explicit_key:
        monkeypatch.setattr(factory,'calls',0,raising=False)
        monkeypatch.setattr(factory,'diagnostics',object(),raising=False)
        monkeypatch.setattr(factory,'cache_key',lambda:{'scale':factory.scale},raising=False)
    fits=[]
    fit=football._fit_candidate
    def count_fit(*args,**kwargs):
        fits.append(True)
        return fit(*args,**kwargs)
    monkeypatch.setattr(football,'_fit_candidate',count_fit)
    experiment=FootballExperiment('Function state '+origin,output_dir=tmp_path)
    data=prepared(holdout=True)
    candidate=Candidate('Attributed function',factory)
    first=experiment.run(data,model=candidate)
    if explicit_key:
        assert factory.calls==1
        factory.calls+=100
        factory.diagnostics=object()
    unchanged=experiment.run(data,model=candidate)
    assert unchanged.reused and unchanged.record['run_id']==first.record['run_id'] and len(fits)==1
    factory.scale=2
    changed=experiment.run(data,model=candidate)
    assert not changed.reused and changed.record['run_id']!=first.record['run_id'] and len(fits)==2
    assert (first.training.folds[0].predictions['predict']==1).all().all()
    assert (changed.training.folds[0].predictions['predict']==2).all().all()
    assert experiment.run(data,model=candidate).reused and len(fits)==2


@pytest.mark.parametrize('key_kind',['function','partial','callable'])
def test_function_keys_bound_aliases_and_self_references_without_diagnostic_state(key_kind):
    def make(filename,padding=0):
        factory=function('def factory(): return factory.scale + ALIAS.scale + OWNERS["current"].scale',filename,padding,OWNERS={})
        factory.__globals__['ALIAS']=factory
        factory.__globals__['OWNERS']['current']=factory
        factory.scale=2
        factory.calls=0
        factory.diagnostics=object()
        def key(owner):return {'scale':owner.scale}
        class Key:
            def __call__(self):return key(factory)
        factory.cache_key=(lambda:key(factory)) if key_kind=='function' else partial(key,factory) if key_kind=='partial' else Key()
        return factory
    one,two=make('cell1'),make('cell99',30)
    assert execution_key(one)==execution_key(two)
    one.calls=200
    one.diagnostics=object()
    assert execution_key(one)==execution_key(two)
    one.scale=3
    assert execution_key(one)!=execution_key(two)


def test_function_attributes_and_recursive_state_are_stable_and_complete():
    def make(filename):
        factory=function('def factory(): return ALIAS.scale',filename)
        factory.__globals__['ALIAS']=factory
        factory.scale=2
        factory.self=factory
        factory.options={'owner':factory,'nested':{'alpha':1}}
        factory.options['cycle']=factory.options
        factory.transform=function('def factory(value): return value + 1',filename)
        return factory
    one,two=make('cell1'),make('cell99')
    assert execution_key(one)==execution_key(two)
    one.options['nested']['alpha']=2
    assert execution_key(one)!=execution_key(two)
    one.options['nested']['alpha']=1
    one.transform=function('def factory(value): return value + 2')
    assert execution_key(one)!=execution_key(two)


def test_same_named_global_rebound_to_another_object_remains_a_dependency():
    factory=function('def factory(): return factory.scale')
    other=function('def factory(): return 0')
    other.scale=1
    factory.__globals__['factory']=other
    before=execution_key(factory)
    other.scale=2
    assert execution_key(factory)!=before



def _source_backed_class_factory(binding, inherited=False):
    from model_selection_samples import FixedAdapter
    owner = {}

    class Factory:
        scale = 1
        calls = 0
        diagnostics = object()

        def __new__(cls):
            cls.calls += 1
            return FixedAdapter(cls.scale)

    if binding == 'staticmethod':
        Factory.cache_key = staticmethod(lambda: {'scale': owner['factory'].scale})
    else:
        Factory.cache_key = classmethod(lambda cls: {'scale': cls.scale})
    factory = Factory
    if inherited:
        class InheritedFactory(Factory):
            scale = 1
        Factory.scale = -1
        factory = InheritedFactory
    owner['factory'] = factory
    return factory


@pytest.mark.parametrize('binding', ['staticmethod', 'classmethod'])
@pytest.mark.parametrize('inherited', [False, True])
def test_source_backed_class_key_controls_actual_reuse_and_predictions(tmp_path, monkeypatch, binding, inherited):
    from football_experiment_samples import prepared
    from xdiyo_analytics.experiments import FootballExperiment, football
    from xdiyo_analytics.selection import Candidate
    factory = _source_backed_class_factory(binding, inherited)
    fits = []
    fit = football._fit_candidate

    def count_fit(*args, **kwargs):
        fits.append(True)
        return fit(*args, **kwargs)

    monkeypatch.setattr(football, '_fit_candidate', count_fit)
    experiment = FootballExperiment('Source-backed class key', output_dir=tmp_path)
    data = prepared(holdout=True)
    candidate = Candidate('Keyed class', factory)
    first = experiment.run(data, model=candidate)
    assert factory.calls == 1
    factory.calls += 100
    factory.diagnostics = object()
    unchanged = experiment.run(data, model=candidate)
    assert unchanged.reused and unchanged.record['run_id'] == first.record['run_id']
    assert len(fits) == 1 and factory.calls == 101
    factory.scale = 2
    changed = experiment.run(data, model=candidate)
    assert not changed.reused and changed.record['run_id'] != first.record['run_id']
    assert len(fits) == 2 and factory.calls == 102
    assert (first.training.folds[0].predictions['predict'] == 1).all().all()
    assert (changed.training.folds[0].predictions['predict'] == 2).all().all()
    assert experiment.run(data, model=candidate).reused and len(fits) == 2


@pytest.mark.parametrize('binding', ['staticmethod', 'classmethod'])
def test_source_backed_class_key_preserves_source_identity_and_bounds_cycles(tmp_path, monkeypatch, binding):
    import sys
    factory = _source_backed_class_factory(binding, inherited=True)
    module = types.ModuleType('keyed_source_probe')
    source = tmp_path / 'factory.py'
    source.write_text('class Factory: pass\n', encoding='utf-8')
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(factory, '__module__', module.__name__)
    key = {'owner': factory, 'scale': 1}
    key['cycle'] = key
    if binding == 'staticmethod':
        factory.cache_key = staticmethod(lambda: key)
    else:
        factory.cache_key = classmethod(lambda cls: {**key, 'actual_class': cls})
    before = signature(factory)
    assert before['source'] and 'class_key' in before
    factory.calls = 200
    factory.diagnostics = object()
    assert signature(factory) == before
    key['scale'] = 2
    assert signature(factory) != before
    key['scale'] = 1
    source.write_text('class Factory: changed = True\n', encoding='utf-8')
    assert signature(factory) != before


@pytest.mark.parametrize('binding', ['instance', 'property', 'descriptor'])
def test_source_backed_class_identity_does_not_invoke_non_class_key_descriptors(binding):
    def unexpected(*args):
        raise AssertionError('must not invoke an instance method or descriptor')

    class Descriptor:
        __get__ = unexpected

    class Factory:
        pass

    Factory.cache_key = unexpected if binding == 'instance' else property(unexpected) if binding == 'property' else Descriptor()
    description = signature(Factory)
    assert description['source'] and 'class_key' not in description



@pytest.mark.parametrize('binding', ['staticmethod', 'classmethod'])
@pytest.mark.parametrize('inherited', [False, True])
def test_source_backed_metaclass_keys_keep_python_binding_without_attribute_hooks(monkeypatch, binding, inherited):
    class Meta(type):
        scale = 1

    if binding == 'staticmethod':
        Meta.cache_key = staticmethod(lambda: {'scale': metaclass.scale})
    else:
        Meta.cache_key = classmethod(lambda cls: {'owner': cls.__name__, 'scale': cls.scale})
    metaclass = Meta
    if inherited:
        class ChildMeta(Meta):
            scale = 2
        metaclass = ChildMeta

    class Factory(metaclass=metaclass):
        scale = 0

    expected = Factory.cache_key()
    before = signature(Factory)
    assert before['class_key'] == signature(expected)

    def guarded_getattribute(cls, name):
        if name == 'cache_key':
            raise AssertionError('signature must inspect known descriptors without invoking lookup hooks')
        return type.__getattribute__(cls, name)

    monkeypatch.setattr(metaclass, '__getattribute__', guarded_getattribute)
    assert signature(Factory) == before
    metaclass.scale = 3
    assert signature(Factory) != before
    assert signature(Factory)['class_key'] == signature({**expected, 'scale': 3})
