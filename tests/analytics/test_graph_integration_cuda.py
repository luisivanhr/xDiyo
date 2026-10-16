"""Explicit idle-GPU integration gates; no production/research data."""
import os
import pytest
import torch
from test_graph_integration_gates import (
    MAIN, FEATURE, threads,
    test_combined_resident_exact_updates_and_no_learned_cache as resident_gate,
    test_extensions_fresh_process_continuation as resume_gate,
    test_pinned_source_two_update_parity_and_legacy_reload as pinned_gate,
)

pytestmark = pytest.mark.skipif(os.environ.get('XDIYO_CUDA_CHECKS') != '1',
    reason='Requires explicit authorization and an idle GPU')

@pytest.mark.parametrize('heatmaps', [False, True])
@pytest.mark.parametrize('memory', [False, True])
@pytest.mark.parametrize('route', ['auxiliary', 'combined', 'bottleneck'])
def test_cuda_combined_residency(monkeypatch, heatmaps, memory, route):
    assert torch.cuda.is_available()
    resident_gate(monkeypatch, heatmaps, memory, route, device='cuda:0')

@pytest.mark.parametrize('kind', ['gat_memory', 'gat_statistics', 'gat_combined'])
@pytest.mark.parametrize('boundary', ['', 'early_stopping'])
def test_cuda_extension_cold_process_resume(tmp_path, kind, boundary):
    assert torch.cuda.is_available()
    resume_gate(tmp_path, kind, boundary, device='cuda:0')

@pytest.mark.parametrize('reference,kind', [(MAIN,'plain'), (MAIN,'cnn'),
    (FEATURE,'gat_memory'), (FEATURE,'gat_statistics'), (FEATURE,'gat_combined')])
def test_cuda_pinned_sources_and_reload(tmp_path, reference, kind):
    assert torch.cuda.is_available()
    pinned_gate(tmp_path, reference, kind, device='cuda:0')
