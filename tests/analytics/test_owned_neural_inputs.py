from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest
import torch
from test_heatmap_cnn import standalone_context
from test_graph_preparation_memory import assert_exact
from xdiyo_analytics.neural import HeatmapCNNModel, HeatmapCNNConfig, Reconstruction
from xdiyo_analytics.neural.adapter import CNNBackend
from xdiyo_analytics.neural.prepared import OwnedPopulation


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('aux', [None, Reconstruction(), Reconstruction(loss='kl')])
def test_owned_loss_updates_equal_mutable_validating_path(task, aux):
    torch.set_num_threads(1)
    c = standalone_context(task)
    spec = HeatmapCNNModel(task=task, encoder=HeatmapCNNConfig(reconstruction=aux), batch_size=3)
    a, b = CNNBackend(spec, 13, 'cpu'), CNNBackend(spec, 13, 'cpu')
    a.initialize(c); b.initialize(c)
    b.train = tuple(b.train)  # Legacy arbitrary-mutable population contract.
    for _ in range(3):
        assert a.step() == b.step()
        assert_exact(a.model.state_dict(), b.model.state_dict())
        assert_exact(a.optimizer.state_dict(), b.optimizer.state_dict())
        for x, y in zip(a.model.parameters(), b.model.parameters()):
            assert torch.equal(x.grad, y.grad)


def test_owned_aliases_generation_and_invalidation():
    c = standalone_context()
    backend = CNNBackend(HeatmapCNNModel(), 1, 'cpu'); backend.initialize(c)
    population = backend.train
    expected = population[0]
    c.X.iloc[:] = 99
    exposed = population[0]; exposed[:] = 8
    torch.as_tensor(population[0]).fill_(9)
    batch = population.batch(slice(0, 2)); batch[0][:] = 10
    np.testing.assert_array_equal(population[0], expected)
    with pytest.raises(ValueError): population._arrays[0].setflags(write=True)
    with pytest.raises(FrozenInstanceError): population.kl_validated = True
    other = OwnedPopulation(tuple(population), binding=population.binding)
    assert other.generation == population.generation
    changed = list(population); changed[0][0, 0, 0, 0] = 1
    assert OwnedPopulation(changed, binding=population.binding).generation != population.generation
    backend.spec = replace(backend.spec, numeric_features=('strength',))
    with pytest.raises(ValueError, match='initialize again'): backend.loss(population)


def test_owned_kl_validated_once_but_custom_inputs_still_rejected(monkeypatch):
    c = standalone_context()
    spec = HeatmapCNNModel(encoder=HeatmapCNNConfig(reconstruction=Reconstruction(loss='kl')))
    backend = CNNBackend(spec, 1, 'cpu'); backend.initialize(c)
    def forbidden(*a, **k): raise AssertionError('revalidated immutable targets')
    monkeypatch.setattr(backend.model.encoder, 'reconstruction_errors', forbidden)
    assert np.isfinite(backend.loss(backend.train)).all()
    monkeypatch.undo()
    invalid = list(backend.train); invalid[0][3, 0] = -1
    with pytest.raises(ValueError, match='nonnegative'): backend.loss(invalid)
    with pytest.raises(ValueError, match='nonnegative'):
        OwnedPopulation(invalid, binding='test', validate_kl=True)
    invalid[0][3, 0] = 0
    with pytest.raises(ValueError, match='positive mass'):
        OwnedPopulation(invalid, binding='test', validate_kl=True)


def test_graph_plan_nested_metadata_is_isolated():
    import pandas as pd
    from types import SimpleNamespace
    from xdiyo_analytics.graphs.streaming import GroupPlan
    meta = pd.DataFrame({'notes': [{'parts': ['original']}]})
    context = SimpleNamespace(metadata=meta, X=pd.DataFrame({'x':[1.]}))
    plan = GroupPlan(None, SimpleNamespace(fixture_features=('x',)), context, pd.Series([1]))
    meta.at[0, 'notes']['parts'][0] = 'changed'
    context.X.iloc[0, 0] = 99
    assert plan.metadata.at[0, 'notes']['parts'] == ['original']
    assert plan.X.iloc[0, 0] == 1


def test_native_store_export_matches_public_tables_and_keeps_attrs_isolated(tmp_path):
    import pandas as pd
    from test_temporal_graph import sample
    from xdiyo_analytics.graphs import TemporalEventStore
    source = sample()
    history, memberships, schedule = source.history, source.team_seasons, source.schedule
    history.attrs['notes'] = {'parts': ['original']}
    store = TemporalEventStore(history, memberships, schedule=schedule)
    history.attrs['notes']['parts'][0] = 'changed'
    exposed = store.history
    exposed.attrs['notes']['parts'][0] = 'also changed'
    assert store.history.attrs['notes']['parts'] == ['original']
    store._export_tables(tmp_path)
    for name in ('history', 'team_seasons', 'schedule'):
        expected = tmp_path/('public-'+name+'.parquet')
        getattr(store, name).to_parquet(expected, compression='zstd', index=False)
        assert expected.read_bytes() == (tmp_path/(name+'.parquet')).read_bytes()
        pd.testing.assert_frame_equal(pd.read_parquet(expected), pd.read_parquet(tmp_path/(name+'.parquet')))
