"""Pass-local, chronological detached boundary states. No learned cache survives."""
from contextlib import contextmanager
import pandas as pd


def boundaries(store, config, until):
    mode = config.boundary_memory
    end = config.history_cutoff(until)
    if store._history.empty:
        return []  # No evidence to summarize; retain the ordinary current prior.
    first = config.history_cutoff(store._history.kickoff_at.min())
    if mode.policy == 'season':
        return sorted({config.history_cutoff(t) for t in store._memberships.entry_at.drop_duplicates()
                       if first < config.history_cutoff(t) <= end})
    anchor = pd.Timestamp(mode.anchor).tz_convert(config.timezone)
    if anchor != anchor.normalize():
        raise ValueError('Fixed-segment anchor must be local midnight in the model timezone.')
    start = first.tz_convert(config.timezone)
    n = (start.date()-anchor.date()).days // mode.days
    start = anchor + pd.DateOffset(days=(n+1)*mode.days)
    result = []
    while start <= end:
        result.append(start.tz_convert('UTC'))
        start += pd.DateOffset(days=mode.days)
    return result


def segment_start(store, config, time):
    bs = boundaries(store, config, time)
    return bs[-1] if bs else None


class BoundaryReplay:
    """One evaluated weight version, monotone queries, detached stable-ID state.

    An edge is explicit iff eligible(now) and not eligible(segment_start).
    In-segment queries never change the fixed segment-start state.
    """
    def __init__(self, backend, until):
        self.backend = backend
        self.boundaries = iter(boundaries(backend.store, backend.config, until))
        self.next = next(self.boundaries, None)
        self.start = None
        self.state = {}
        self.last = None

    def tensors(self, snap, *, reconstruction_targets=False):
        import torch
        b = self.backend
        data = b.preprocessor.native_tensors(snap, b.device, b.store,
                                             reconstruction_targets=reconstruction_targets)
        return self.initial_state(data, snap.teams)

    def initial_state(self, data, teams):
        import torch
        prior = self.backend.model.prior(data['league'], data['entry'])
        # Existing carries replace the prior; unseen teams retain trainable priors.
        data['initial_state'] = torch.stack([self.state[t] if t in self.state else prior[i]
                                             for i, t in enumerate(teams)]) if teams else prior
        return data

    @contextmanager
    def lease(self, groups, index):
        time, _ = groups.plan.descriptors[index]
        self.advance(time)
        with groups.segment_lease(index, self.start, self.state) as (group, teams):
            # The lease owns a transient copy. Never mutate its immutable cache.
            self.initial_state(group[1], teams)
            yield group

    def advance(self, time):
        import torch
        b = self.backend
        cutoff = b.config.history_cutoff(time)
        if self.last is not None and cutoff < self.last:
            raise ValueError('Boundary replay requires chronological cutoffs.')
        while self.next is not None and self.next <= cutoff:
            boundary = self.next
            with torch.no_grad():
                snap = b.store.snapshot(boundary, b.config, self.state, _since=self.start,
                                        _columns=b.preprocessor.snapshot_columns(b.store))
                data = self.tensors(snap)
                z = b.model.encode(data)
                prior = b.model.prior(data['league'], data['entry'])
                retention = b.config.boundary_memory.retention
                # Retention applies only on actual boundary transitions, once.
                z = retention*z + (1-retention)*prior
                self.state = {t: z[i].detach().clone() for i, t in enumerate(snap.teams)}
            self.start = boundary
            self.next = next(self.boundaries, None)
        self.last = cutoff

    def snapshot(self, time, teams=()):
        self.advance(time)
        b = self.backend
        snap = b.store.snapshot(time, b.config, set(teams) | set(self.state), _since=self.start,
                                _columns=b.preprocessor.snapshot_columns(b.store))
        return snap, self.tensors(snap)

    def group(self, groups, index):
        import torch
        b = self.backend
        time, _ = groups.plan.descriptors[index]
        self.advance(time)
        positions, snap, home, away, meta, numeric = groups.plan.raw(
            index, columns=b.preprocessor.snapshot_columns(b.store), since=self.start, extra_teams=self.state)
        return (positions, self.tensors(snap, reconstruction_targets=groups.reconstruction_targets), torch.as_tensor(home, device=b.device),
                torch.as_tensor(away, device=b.device),
                torch.as_tensor(b.preprocessor.fixture(meta, numeric), dtype=torch.float32, device=b.device))
