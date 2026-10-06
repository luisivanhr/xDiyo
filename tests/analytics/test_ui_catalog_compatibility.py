"""One version-dependent constructor field, without losing persistent UI hints."""
from copy import deepcopy
import inspect
import json
import pytest

from xdiyo_analytics.ui.catalog import Catalog
from xdiyo_analytics.ui.inventory import component_schema

KEY = 'sklearn.compose.ColumnTransformer'


@pytest.mark.parametrize('default', [True, False, 'deprecated'])
@pytest.mark.parametrize('saved_field', [True,False])
def test_installed_parameter_presence_default_and_recipe_build(monkeypatch,default,saved_field):
    # Simulate constructor contracts from supported releases without installing
    # or downgrading a scientific environment solely for presentation tests.
    class Transformer:
        def __init__(self, transformers, *, force_int_remainder_cols=default):
            self.transformers=transformers
            self.force_int_remainder_cols=force_int_remainder_cols
    saved=deepcopy(component_schema(KEY))
    if saved_field:
        saved['fields'].append(dict(name='force_int_remainder_cols',kind='text',default='stale',
                                   help='Retained field help',primary=False))
    original=deepcopy(saved)
    monkeypatch.setattr('xdiyo_analytics.ui.inventory.component_schema',lambda key:saved)
    catalog=Catalog().register(KEY,Transformer,category='preprocessor')
    # Registration cached the signature; ordinary schema reads must not inspect.
    original_signature=inspect.signature
    monkeypatch.setattr(inspect,'signature',lambda *a,**k:pytest.fail('Uncached signature lookup'))
    schema=catalog.schema()[0]
    fields={f['name']:f for f in schema['fields']}
    assert fields['force_int_remainder_cols']['default']==default
    assert fields['force_int_remainder_cols']['kind']==('boolean' if isinstance(default,bool) else 'value')
    assert fields['force_int_remainder_cols']['help']==('Retained field help' if saved_field else
        'Compatibility option from the installed scikit-learn constructor. Keep its default; deprecated releases ignore it.')
    for field in original['fields']:
        if field['name']!='force_int_remainder_cols':
            assert fields[field['name']]==field
    assert saved==original
    monkeypatch.setattr(inspect,'signature',original_signature)
    recipe={'component':KEY,'params':{'transformers':[],'force_int_remainder_cols':default}}
    assert catalog.build(json.loads(json.dumps(recipe))).force_int_remainder_cols==default
    fields['force_int_remainder_cols']['help']='consumer mutation'
    assert catalog.schema()[0]['fields'][-1]['help']!='consumer mutation'


def test_removed_parameter_not_exposed_from_stale_inventory(monkeypatch):
    class Transformer:
        def __init__(self,transformers):self.transformers=transformers
    saved=deepcopy(component_schema(KEY))
    saved['fields'].append(dict(name='force_int_remainder_cols',default=True))
    monkeypatch.setattr('xdiyo_analytics.ui.inventory.component_schema',lambda key:saved)
    catalog=Catalog().register(KEY,Transformer,category='preprocessor')
    assert 'force_int_remainder_cols' not in {f['name'] for f in catalog.schema()[0]['fields']}
    assert saved['fields'][-1]['name']=='force_int_remainder_cols'


def test_actual_installed_constructor_contract():
    from xdiyo_analytics.ui import catalog_for_ui
    catalog=catalog_for_ui()
    parameter=inspect.signature(catalog.entries[KEY]['constructor']).parameters.get('force_int_remainder_cols')
    fields={f['name']:f for s in catalog.schema() if s['id']==KEY for f in s['fields']}
    assert ('force_int_remainder_cols' in fields)==(parameter is not None)
    if parameter is not None:
        assert fields['force_int_remainder_cols']['default']==parameter.default
