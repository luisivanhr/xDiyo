"""Deterministic, descriptor-first dependency packing with explicit byte ledgers.

Limits concern execution workspace plus a reserve for the existing source-cache
capacity, not the resident store, model, optimizer, allocator reservations, or
third-party workspaces. Reserving cache capacity even when warm keeps plans stable.
Full-population statistics observations are the existing native resident cache,
built once by supervision before admission; packing neither duplicates nor owns
that cache. Pass-local detached scalar receipts are reserved separately below.
Estimates intentionally overcount tensor lifetimes. They are admission estimates,
not a proof of process RSS or driver allocation bounds; native fallbacks retain
native memory requirements. No OOM retry or available-memory-based regrouping is
performed. Consumers must release a yielded group before requesting the next.
"""
from dataclasses import dataclass
from contextlib import nullcontext
from hashlib import sha256
import json


@dataclass(frozen=True)
class Ledger:
    descriptors: int
    preparation: int
    inputs: int
    concatenation: int
    activations: int
    targets: int
    source_cache: int
    host_metadata: int
    one_hot_workspace: int
    scalar_receipts: int
    source_map_workspace: int
    host_bytes: int
    device_bytes: int


def descriptor_bound(plan):
    """Include eligible rows, codes, masks, layer selections and Python overhead."""
    edges = len(plan.store._history)
    nodes = len(plan.store._source_nodes)
    fixtures = max((len(positions) for _, positions in getattr(plan, 'descriptors', ())), default=0)
    # Query-only teams can be absent from source_nodes. Reserve two cold team
    # IDs per fixture plus metadata/selected-feature staging before any selection.
    fixture_columns = len(getattr(plan.config, 'fixture_features', ()))
    queries = fixtures * (2 * 1024 + 512 + 16 * fixture_columns)
    return 4096 + edges * (64 + 24 * plan.config.layers) + nodes * 512 + queries


def _value(selection, name):
    return selection[name] if isinstance(selection, dict) else getattr(selection, name)


def _shape(selection, native=False):
    if native:
        return int(_value(selection, 'native_edge_count')), int(_value(selection, 'full_node_count'))
    return len(_value(selection, 'layer_rows')[0]), len(_value(selection, 'teams'))


def estimate(backend, selections, *, backward=False, packed=False, native=False, receipt_groups=0):
    """Conservative coexistence ledger including host staging and target buffers.

    All edge/node intermediates are charged for every layer during backward.
    32 float planes per head/channel cover saved activations, their gradients,
    advanced-index copies, pointwise temporaries and reduction buffers. Target
    masks/indices/values/distribution and quantile temporaries are additionally
    charged even when observations are sparse. CPU tensor workspace is host RAM.
    """
    cfg = backend.config
    pre = backend.preprocessor
    edge_width = 2 * len(pre.edge_columns) + len(pre.leagues) + 2
    fixture_width = 2 * len(pre.fixture_columns) + len(pre.leagues) + 1
    edges = nodes = fixtures = descriptors = 0
    preparation_edges = preparation_nodes = eligibility_scan = 0
    snapshot_mode = getattr(getattr(cfg, 'execution', None), 'mode', None) == 'dependency'
    for selection in selections:
        e, n = _shape(selection, native)
        edges += e
        nodes += n
        fixtures += len(_value(selection, 'positions'))
        descriptors += int(_value(selection, 'host_nbytes')) + 4096
        pe, pn = _shape(selection, native or snapshot_mode)
        preparation_edges += pe
        preparation_nodes += pn
        # Lean preserves validation and support over FULL native eligibility.
        # pandas duration arrays, source-code gathers and support masks may all
        # coexist even when the model's dependency cone is very small.
        eligibility_scan += int(_value(selection, 'native_edge_count')) * 192
    inputs = edges * (4 * edge_width + 24 + 8 * cfg.layers) + nodes * 32 + fixtures * (4 * fixture_width + 24)
    # Float64 encoding, masks, source slicing and mapping metadata coexist with
    # their final float32 tensors. Snapshot mode may additionally copy frames.
    preparation = preparation_edges * (24 * edge_width + 128) + preparation_nodes * 512 + fixtures * (24 * fixture_width + 128)
    preparation += eligibility_scan
    # No second cache is created. Reserve the existing cache's complete allowed
    # capacity and cold validation chunk work irrespective of current warmness.
    store = getattr(backend, 'store', None)
    source_count = len(store._history) if store is not None else max(
        int(_value(s, 'native_edge_count')) for s in selections)
    # Lean preparation maps the COMPLETE immutable source vocabulary, including
    # teams seen only in future/ineligible rows. Retained cone node counts cannot
    # bound that dense int64 source_to_local array. Preparations are sequential.
    source_node_count = len(store._source_nodes) if store is not None else max(
        int(_value(s, 'full_node_count')) for s in selections)
    source_map_workspace = 8 * source_node_count
    preparation += source_map_workspace
    cache_capacity = source_count * (4 * edge_width + 1)
    source_cache = cache_capacity if cache_capacity <= getattr(pre, '_edge_cache_limit', 128 * 1024**2) else 0
    preparation += min(source_count, 1024) * (32 * edge_width + 128)
    # GraphPreprocessor.one_hot constructs a float64 identity matrix before
    # row selection. It is quadratic in league vocabulary, even for one fixture.
    # Calls within a batch are sequential, so one maximum identity is reserved.
    one_hot_workspace = 8 * (len(pre.leagues) + 1)**2
    preparation += one_hot_workspace
    concatenation = inputs * 2 if packed else 0  # inputs + cat destination + index offset scratch
    d, h, layers = cfg.embedding_dim, cfg.heads, cfg.layers
    activations = 4 * (edges * (32 * d * h + 16 * h + 8 * d) + nodes * (16 * d * h + 16 * d))
    activations *= layers if backward else 2
    statistics = cfg.statistics_forecast
    target_count = 0 if statistics is None else len(statistics.targets)
    summary_width = 0 if statistics is None else sum(
        len(t.quantiles) if t.family == 'quantile' else 1 if t.family == 'point' else 2
        for t in statistics.targets)
    # Charge the original fixture vector and statistics summary together even
    # for auxiliary/bottleneck routes: a transparent upper envelope of both heads.
    activations += fixtures * (32 * (4 * d + fixture_width + cfg.hidden_width + summary_width) + 128)
    target_width = 0 if statistics is None else sum(max(8, len(t.quantiles) * 8) for t in statistics.targets)
    targets = fixtures * (256 + 32 * target_width) + 4096
    # Tensor wrappers and autograd Nodes remain on HOST even for CUDA tensors.
    # Heuristic operation-count reserves intentionally exceed current graph and
    # target loops; they are not portable ABI-size guarantees. Piece wrappers
    # coexist until disjoint concatenation finishes.
    host_metadata = ((128 * layers + 64 * target_count + 64) * 1024 if backward else
                     (32 * layers + 16 * target_count + 32) * 512)
    host_metadata += len(selections) * 128 * 512
    # Use native issuance count, never packed batch count, for a stable upper
    # envelope of retained detached loss scalars, list entries and final stack /
    # host float conversion. CUDA minimum allocation bins are overreserved too.
    scalar_receipts = int(receipt_groups) * 3072
    host = descriptors + preparation + source_cache + host_metadata + int(receipt_groups) * 2048
    device = inputs + concatenation + activations + targets + int(receipt_groups) * 1024
    if snapshot_mode and not native:
        # The staged snapshot implementation first materializes full native
        # inputs, then indexes the cone. Charge both through the transition.
        native_inputs = preparation_edges * (4 * edge_width + 24) + preparation_nodes * 32
        device += native_inputs
    if str(backend.device).startswith('cpu'):
        host += device
        device = 0
    else:
        host += inputs + concatenation  # host staging retained through transfers
    return Ledger(descriptors, preparation, inputs, concatenation, activations, targets, source_cache,
                  host_metadata, one_hot_workspace, scalar_receipts, source_map_workspace, host, device)


def pack_groups(pieces):
    """Disjoint union; identical team IDs at different issuances never merge."""
    import numpy as np
    import torch
    node_offsets = np.cumsum([0] + [len(p[1]['league']) for p in pieces[:-1]])
    edge_offsets = np.cumsum([0] + [len(p[1]['receiver']) for p in pieces[:-1]])
    data = {key: torch.cat([p[1][key] for p in pieces])
            for key in ('x', 'log_weight', 'gate', 'league', 'entry')}
    for key in ('receiver', 'sender'):
        data[key] = torch.cat([p[1][key] + int(offset) for p, offset in zip(pieces, node_offsets)])
    rows = []
    for layer in range(len(pieces[0][1]['_dependency_rows'])):
        if all(p[1]['_dependency_rows'][layer] is None for p in pieces):
            rows.append(None)
        else:
            rows.append(torch.cat([
                (torch.arange(len(p[1]['receiver']), device=p[1]['receiver'].device)
                 if p[1]['_dependency_rows'][layer] is None else p[1]['_dependency_rows'][layer]) + int(offset)
                for p, offset in zip(pieces, edge_offsets)]))
    data['_dependency_rows'] = rows
    data['_group_lengths'] = tuple(len(p[0]) for p in pieces)
    return (np.concatenate([p[0] for p in pieces]), data,
            torch.cat([p[2] + int(offset) for p, offset in zip(pieces, node_offsets)]),
            torch.cat([p[3] + int(offset) for p, offset in zip(pieces, node_offsets)]),
            torch.cat([p[4] for p in pieces]))


def _fits(backend, selections, execution, backward, packed, receipt_groups=0):
    candidate = estimate(backend, selections, backward=backward, packed=packed, receipt_groups=receipt_groups)
    # A pack may not exceed the largest native singleton workspace envelope.
    native = [estimate(backend, [s], backward=backward, native=True, receipt_groups=receipt_groups) for s in selections]
    host_limit = min(execution.host_memory_bytes, max(v.host_bytes for v in native))
    device_limit = min(execution.device_memory_bytes, max(v.device_bytes for v in native))
    return candidate.host_bytes <= host_limit and candidate.device_bytes <= device_limit, candidate


def _headroom(backend, execution):
    """Fail before yielding/backward, without changing the deterministic plan."""
    if not execution.headroom_bytes:
        return
    try:
        import psutil
    except ImportError as exc:
        raise ImportError('Dependency execution headroom checks require psutil; install xdiyo-analytics[graphs].') from exc
    if psutil.virtual_memory().available < execution.headroom_bytes + execution.host_memory_bytes:
        raise RuntimeError('Dependency execution host headroom unavailable; execution plan was not changed.')
    if str(backend.device).startswith('cuda'):
        import torch
        free, _ = torch.cuda.mem_get_info(backend.device)
        if free < execution.headroom_bytes + execution.device_memory_bytes:
            raise RuntimeError('Dependency execution device headroom unavailable; execution plan was not changed.')


def stream_batches(backend, groups, backward=False):
    """Yield ``(normal_group, original_indices, fallback_reason)`` in order.

    Only bounded descriptors are held before admission. A rejected pending
    descriptor never constructs host feature tensors or device tensors. Trace
    contains bounded counters and a streaming hash, not a retained per-group log.
    Unsupported modes fall back explicitly; runtime allocation errors propagate.
    """
    from .dependency import select_dependencies, prepare_dependency_group
    execution = backend.config.execution
    trace = dict(requested_mode='native' if execution is None else execution.mode,
                 groups=0, batches=0, fallback_groups=0, max_host_bytes=0, max_device_bytes=0,
                 descriptor_limit_bytes=0, fallback_reasons={}, plan_sha256='')
    backend._execution_trace = trace
    digest = sha256()

    def native_lease(index, replay=None):
        if hasattr(backend, '_group_lease'):
            return backend._group_lease(groups, index, replay)
        return nullcontext((groups[index], False))

    def record(indices, reason, ledger=None):
        trace['groups'] += len(indices)
        trace['batches'] += 1
        if reason:
            trace['fallback_groups'] += len(indices)
            trace['fallback_reasons'][reason] = trace['fallback_reasons'].get(reason, 0) + len(indices)
        if ledger:
            trace['max_host_bytes'] = max(trace['max_host_bytes'], ledger.host_bytes)
            trace['max_device_bytes'] = max(trace['max_device_bytes'], ledger.device_bytes)
        digest.update(json.dumps((indices, reason, None if ledger is None else (ledger.host_bytes, ledger.device_bytes))).encode())
        trace['plan_sha256'] = digest.hexdigest()

    reason = None
    if execution is None or execution.mode == 'native':
        reason = 'native_requested'
    elif backend.config.heatmaps is not None or backend.config.boundary_memory is not None:
        reason = 'unsupported_cnn_or_boundary'
    elif getattr(getattr(groups, '_resident', None), 'pool', None) is not None:
        reason = 'input_residency_enabled'
    elif not hasattr(groups, 'plan'):
        reason = 'unsupported_group_provider'
    if reason:
        replay = backend._replay(groups) if hasattr(backend, '_replay') else None
        for index in range(len(groups)):
            record((index,), reason)
            with native_lease(index, replay) as (group, _):
                yield group, (index,), reason
            del group
        return
    _headroom(backend, execution)
    bound = descriptor_bound(groups.plan)
    trace['descriptor_limit_bytes'] = bound
    if bound > execution.host_memory_bytes:
        for index in range(len(groups)):
            record((index,), 'descriptor_budget')
            with native_lease(index) as (group, _):
                yield group, (index,), 'descriptor_budget'
            del group
        return
    pending = []
    pending_indices = []

    def admitted(selections, packed):
        return _fits(backend, selections, execution, backward, packed, receipt_groups=len(groups))

    def materialize():
        pieces = [prepare_dependency_group(groups, i, selection=s, lean=execution.mode != 'dependency')
                  for i, s in zip(pending_indices, pending)]
        return pack_groups(pieces) if len(pieces) > 1 else pieces[0]

    for index in range(len(groups)):
        # Reserve worst-case construction of the next descriptor before doing so.
        if pending and sum(int(_value(s, 'host_nbytes')) + 4096 for s in pending) + bound > execution.host_memory_bytes:
            _, ledger = admitted(pending, len(pending) > 1)
            group = materialize()
            indices = tuple(pending_indices)
            pending.clear(); pending_indices.clear()
            record(indices, None, ledger)
            yield group, indices, None
            del group
        selection = select_dependencies(groups.plan, index)
        accepted, single = admitted([selection], False)
        empty = (not len(_value(selection, 'layer_rows')[0]) and
                 int(_value(selection, 'native_edge_count')) > 0)
        dense = sum(len(rows) for rows in _value(selection, 'layer_rows')) >= (
            int(_value(selection, 'native_edge_count')) * backend.config.layers)
        accepted = accepted and not dense and not empty
        if pending:
            fits, ledger = admitted(pending + [selection], True)
            if execution.mode != 'dependency_packed' or len(pending) >= execution.max_graphs or not fits or not accepted:
                # Drop the unadmitted descriptor before tensor materialization.
                del selection
                _, ledger = admitted(pending, len(pending) > 1)
                group = materialize()
                indices = tuple(pending_indices)
                pending.clear(); pending_indices.clear()
                record(indices, None, ledger)
                yield group, indices, None
                del group
                selection = select_dependencies(groups.plan, index)
        if not accepted:
            del selection
            reason = 'empty_dependency' if empty else 'dense_dependency' if dense else 'workspace_budget'
            record((index,), reason)
            with native_lease(index) as (group, _):
                yield group, (index,), reason
            del group
        else:
            pending.append(selection)
            pending_indices.append(index)
            # The pending list is the sole owner: otherwise the generator local
            # keeps the last descriptor alive across a flush and next admission.
            del selection
    if pending:
        _, ledger = admitted(pending, len(pending) > 1)
        group = materialize()
        indices = tuple(pending_indices)
        pending.clear(); pending_indices.clear()
        record(indices, None, ledger)
        yield group, indices, None
