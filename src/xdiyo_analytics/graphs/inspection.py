"""Read-only, cutoff-exact learned edge attention, with no refitting."""
import numpy as np
import pandas as pd
from .adapter import _torch, torch_threads
from .store import _ids, _times


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
            for time, positions in issue.groupby(issue, sort=True).indices.items():
                teams = queries.iloc[positions].team_id.tolist()
                snap = backend.store.snapshot(time, adapter.config, teams)
                _, layers = backend.model.encode(backend.preprocessor.tensors(snap, adapter.device), return_attention=True)
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
    finally:
        backend.model.train(was_training)
    return pd.DataFrame(records)
