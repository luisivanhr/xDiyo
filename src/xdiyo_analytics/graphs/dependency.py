"""Transient exact backwards dependency cones for scalar temporal graphs."""
from dataclasses import dataclass
import math
import sys
import numpy as np
import pandas as pd
from .streaming import release_temporary_frames


@dataclass(frozen=True)
class DependencySelection:
    issuance: object
    positions: np.ndarray
    eligible_rows: np.ndarray
    layer_rows: tuple
    teams: tuple
    full_node_count: int
    native_edge_count: int
    host_nbytes: int


def select_dependencies(plan, index):
    if plan.config.heatmaps is not None or plan.config.boundary_memory is not None:
        raise ValueError('Dependency execution supports scalar full-history graphs only.')
    store = plan.store
    issuance, positions = plan.descriptors[index]
    meta = pd.DataFrame(plan.metadata, copy=False).iloc[positions]
    queried = set(meta.home_id.tolist()) | set(meta.away_id.tolist())
    eligible = store._eligible_rows(plan.config.history_cutoff(issuance))
    receiver = store._receiver_codes[eligible]
    sender = store._sender_codes[eligible]
    frontier = {store._source_node_index[t] for t in queried if t in store._source_node_index}
    reverse = []
    for _ in range(plan.config.layers):
        mask = np.isin(receiver, tuple(frontier))
        reverse.append(eligible[mask])
        frontier.update(sender[mask].tolist())
    layers = tuple(reversed(reverse))
    teams = tuple(sorted(queried | {store._source_nodes[c] for c in frontier}))
    full_nodes = len(queried | {store._source_nodes[c] for c in np.unique(receiver)})
    size = positions.nbytes + eligible.nbytes + sum(r.nbytes for r in layers)
    size += sys.getsizeof(teams) + sum(sys.getsizeof(t) for t in teams)
    return DependencySelection(issuance, positions, eligible, layers, teams,
                               full_nodes, len(eligible), size)


def _indices(data, selection, base, device):
    import torch
    data['_dependency_rows'] = [None if np.array_equal(rows, base) else
        torch.as_tensor(np.searchsorted(base, rows), dtype=torch.long, device=device)
        for rows in selection.layer_rows]


@release_temporary_frames
def prepare_dependency_group(groups, index, selection=None, lean=True):
    """Prepare one cone, preserving native causal validation on cold caches."""
    import torch
    selection = select_dependencies(groups.plan, index) if selection is None else selection
    plan, prep, device = groups.plan, groups.preprocessor, groups.device
    store, cfg = plan.store, plan.config
    base = selection.layer_rows[0]
    if not lean:
        positions, data, home, away, fixture = groups[index]
        native_teams = tuple(sorted(set(selection.teams) | {
            store._source_nodes[c] for c in np.unique(store._receiver_codes[selection.eligible_rows])}))
        lookup = {t:i for i,t in enumerate(native_teams)}
        nodes = torch.as_tensor([lookup[t] for t in selection.teams], device=device)
        edges = torch.as_tensor(np.searchsorted(selection.eligible_rows, base), device=device)
        for key in ('x', 'receiver', 'sender', 'log_weight'):
            data[key] = data[key][edges]
        mapping = torch.full((len(native_teams),), -1, dtype=torch.long, device=device)
        mapping[nodes] = torch.arange(len(nodes), device=device)
        data['receiver'], data['sender'] = mapping[data['receiver']], mapping[data['sender']]
        for key in ('league', 'entry', 'gate'):
            data[key] = data[key][nodes]
        home, away = mapping[home], mapping[away]
        _indices(data, selection, base, device)
        return positions, data, home, away, fixture
    # Only immutable source arrays and one issuance's transient arrays persist.
    meta = pd.DataFrame(plan.metadata, copy=False).iloc[selection.positions]
    teams = selection.teams
    node = {t:i for i,t in enumerate(teams)}
    source_to_local = np.full(len(store._source_nodes), -1, dtype=np.int64)
    for t, i in node.items():
        c = store._source_node_index.get(t)
        if c is not None:
            source_to_local[c] = i
    receiver = source_to_local[store._receiver_codes[base]]
    sender = source_to_local[store._sender_codes[base]]
    counts = np.zeros(len(teams), dtype=np.int64)
    leagues, entries = [], []
    for i, team in enumerate(teams):
        membership = store._membership_by_team.get(team)
        n = np.searchsorted(membership[0], selection.issuance.value, side='right') if membership is not None else 0
        counts[i] = n
        leagues.append(membership[1][n-1] if n else None)
        entries.append(membership[2][n-1] if n else 'unknown')
    # Retain pandas overflow checking and exact native division order.
    all_age = (selection.issuance-store._history.kickoff_at.iloc[selection.eligible_rows]).dt.total_seconds().to_numpy()/86400.
    age = all_age[np.searchsorted(selection.eligible_rows, base)]
    crossed = (counts[receiver]-store._entries_at_event[base]).astype(float)
    logw = -math.log(2)*age/cfg.half_life_days + crossed*math.log(cfg.seasonal_retention)
    all_receiver = source_to_local[store._receiver_codes[selection.eligible_rows]]
    keep = all_receiver >= 0
    support_receiver = all_receiver[keep]
    support_rows = selection.eligible_rows[keep]
    support_crossed = (counts[support_receiver]-store._entries_at_event[support_rows]).astype(float)
    support_logw = -math.log(2)*all_age[keep]/cfg.half_life_days + support_crossed*math.log(cfg.seasonal_retention)
    support = np.bincount(support_receiver, weights=np.exp(support_logw), minlength=len(teams))
    del all_age, age, crossed, all_receiver, keep, support_receiver, support_rows, support_crossed, support_logw
    missing = np.iinfo(np.int64).min
    latest = np.full(len(teams), missing, dtype=np.int64)
    np.maximum.at(latest, receiver, store._kickoff_ns[base])
    home = np.asarray([node[t] for t in meta.home_id], dtype=np.int64)
    away = np.asarray([node[t] for t in meta.away_id], dtype=np.int64)
    def rest(i):
        return ((selection.issuance-pd.Timestamp(latest[i], tz='UTC')).total_seconds()/86400
                if latest[i] != missing else np.nan)
    neutral = meta['neutral_venue'].to_numpy(dtype=float, na_value=np.nan) if 'neutral_venue' in meta else np.full(len(meta), np.nan)
    if np.any(np.isfinite(neutral) & ~np.isin(neutral, [0,1])):
        raise ValueError('neutral_venue must be 0, 1 or missing.')
    numeric = np.column_stack((neutral, [rest(i) for i in home], [rest(i) for i in away], support[home], support[away]))
    if cfg.fixture_features:
        numeric = np.column_stack((numeric, plan.X.iloc[selection.positions].to_numpy(dtype=float, na_value=np.nan)))
    from .config import ENTRY_ORDER
    def tensor(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)
    encoded = prep.source_edges(store, base, validate_rows=selection.eligible_rows)
    data = dict(x=tensor(encoded), receiver=tensor(receiver, torch.long), sender=tensor(sender, torch.long),
                log_weight=tensor(logw, torch.float64), gate=tensor(support/(support+cfg.support_kappa)),
                league=tensor([prep.leagues.get(v,0) for v in leagues], torch.long),
                entry=tensor([ENTRY_ORDER.index(v) for v in entries], torch.long))
    _indices(data, selection, base, device)
    return selection.positions, data, tensor(home, torch.long), tensor(away, torch.long), tensor(prep.fixture(meta,numeric))
