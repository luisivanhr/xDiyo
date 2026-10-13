"""Read-only, cutoff-exact learned edge attention, with no refitting."""
import numpy as np
import pandas as pd
from .adapter import _torch, torch_threads
from .store import _ids, _times
from .streaming import release_temporary_frames
from .execution import configured_execution


def _attention_group(adapter, time, positions, queries, records, replay=None):
    backend = adapter.backend_
    teams = queries.iloc[positions].team_id.tolist()
    if replay is None:
        snap = backend.store.snapshot(time, adapter.config, teams)
        data = backend.preprocessor.tensors(snap, adapter.device)
    else:
        snap, data = replay.snapshot(time, teams)
    _, layers = backend.model.encode(data, return_attention=True)
    selected = np.flatnonzero(snap.edges.team_id.isin(teams))
    edges = list(snap.edges.iloc[selected].itertuples(index=False))
    for layer, alpha in enumerate(layers):
        weights = alpha.cpu().numpy()
        for i, edge in zip(selected, edges):
            node = snap.receiver[i]
            for head, weight in enumerate(weights[i]):
                records.append(dict(team_id=int(edge.team_id), opponent_id=int(edge.opponent_id),
                    event_id=int(edge.event_id), competition_id=int(edge.competition_id),
                    season_id=int(edge.season_id), kickoff_at=edge.kickoff_at,
                    issuance_at=time, history_cutoff=snap.cutoff, layer=layer, head=head,
                    attention=float(weight), log_decay=float(snap.log_weight[i]),
                    support=float(snap.support[node]), gate=float(snap.gate[node]),
                    model_version=backend.model_version, event_store_version=backend.store.version))
                if replay is not None:
                    records[-1].update(boundary_at=replay.start, evidence_scope='explicit_segment_edges')


@configured_execution
@release_temporary_frames
def attention_weights(adapter, queries):
    if not queries.index.is_unique:
        raise ValueError('Attention queries require unique row indices.')
    _ids(queries, ('team_id',))
    issue = _times(queries.issuance_at)
    backend = adapter.backend_
    backend._boundary(issue)
    records = []
    torch = _torch()
    was_training = backend.model.training
    try:
        backend.model.eval()
        with torch_threads(adapter.inner_threads), torch.no_grad():
            from .memory import BoundaryReplay
            replay = BoundaryReplay(backend, issue.max()) if adapter.config.boundary_memory is not None and len(issue) else None
            for time, positions in issue.groupby(issue, sort=True).indices.items():
                _attention_group(adapter, time, positions, queries, records, replay)
    finally:
        backend.model.train(was_training)
    return pd.DataFrame(records)
