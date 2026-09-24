from functools import partial
import types
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
