"""Independent complete-transfer reference for optimized native inspection."""
from pathlib import Path
import hashlib
import os
from types import MethodType
import numpy as np
import pandas as pd
import pytest
import torch

REFERENCE_SHA = "84875df59b9dda1fae809b7df60b150e4541aae2a22bdfd7e741c2175f38f73d"

@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('spatial',[False,True])
def test_select_before_transfer_matches_frozen_inspection(monkeypatch,device,spatial):
    if device.startswith('cuda') and os.environ.get('XDIYO_CUDA_CHECKS')!='1':
        pytest.skip('Explicit free-GPU gate required')
    from test_temporal_graph import sample, config, context
    from test_heatmap_cnn import graph_setup
    from xdiyo_analytics.graphs import TemporalGraphAdapter
    from xdiyo_analytics.training import TrainingControl
    import xdiyo_analytics.graphs.inspection as inspection
    torch.set_num_threads(1)
    if spatial:
        spec,store=graph_setup(deterministic=True);cfg=spec.config
    else:
        store=sample();cfg=config(deterministic=True)
    adapter=TemporalGraphAdapter(cfg,store,device=device)
    adapter.fit_controlled(context(store,[1,2,3,4]),TrainingControl(max_steps=2,seed=7))
    queries=pd.DataFrame({'team_id':[2,1,3,2], 'issuance_at':pd.to_datetime(['2020-01-25','2020-01-25','2020-01-25','2020-01-28'],utc=True)},index=[70,3,28,6])
    expected_embeddings=adapter.encode_teams(queries)
    expected_attention=adapter.attention_weights(queries)
    raw=(Path(__file__).parent/'reference/inspection_before_6ae5003.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==REFERENCE_SHA
    namespace={};exec(compile(raw,'frozen_inspection','exec'),namespace)
    def legacy_encode(backend, time, positions, queries, records, replay=None):
        assert replay is None
        return namespace['_encode_group'](backend, time, positions, queries, records)
    def legacy_attention(adapter, time, positions, queries, records, replay=None):
        assert replay is None
        return namespace['_attention_group'](adapter, time, positions, queries, records)
    monkeypatch.setattr(adapter.backend_,'_encode_group',MethodType(legacy_encode,adapter.backend_))
    monkeypatch.setattr(inspection,'_attention_group',legacy_attention)
    pd.testing.assert_frame_equal(expected_embeddings,adapter.encode_teams(queries),check_exact=True)
    pd.testing.assert_frame_equal(expected_attention,adapter.attention_weights(queries),check_exact=True)


def test_spatial_native_cache_masks_aliases_and_store_generation(monkeypatch):
    from test_heatmap_cnn import graph_setup
    from test_temporal_graph import context
    from xdiyo_analytics.graphs.adapter import _Backend
    from test_graph_preparation_memory import assert_exact
    spec,store=graph_setup()
    backend=_Backend(spec.config,store,3,'cpu');backend.initialize(context(store,[1,2,3,4]))
    prep=backend.preprocessor
    time=pd.Timestamp('2020-01-20',tz='UTC')
    snapshot=store.snapshot(time,spec.config,{1,2})
    reference=prep.tensors(snapshot,'cpu')
    actual=prep.native_tensors(store.snapshot(time,spec.config,{1,2}),'cpu',store)
    assert_exact(reference,actual)
    maps,flags=prep.cached_maps(snapshot,store);maps.fill(99);flags.fill(0)
    assert_exact(reference,prep.native_tensors(store.snapshot(time,spec.config,{1,2}),'cpu',store))
    from dataclasses import replace
    with pytest.raises(ValueError,match='different event store'):
        prep.cached_maps(replace(snapshot,store_version='changed'),store)
