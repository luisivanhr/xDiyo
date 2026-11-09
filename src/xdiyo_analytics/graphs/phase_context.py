"""Named, immutable native countdown/rank context, outside match statistics.

The table contains one home-oriented row per fixture at its original forecast
cutoff. It is an exact-query artifact, not a mechanism for recomputing ranks
from a later graph snapshot. Unsupported native scopes remain missing.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ('competition_id', 'season_id', 'event_id')
WINDOWS = (1, 5, 10, 20)
VALUES = ('countdown', *(f'{side}_rank_{w}' for w in WINDOWS for side in ('home', 'away')))
POLICY = 'native_phase_form_rank_original_forecast_cutoff_v1'
CORRECTED_POLICY = 'native_phase_form_rank_available_results_original_forecast_cutoff_v2'
DAILY_POLICY = 'native_phase_form_rank_retrospective_kickoff_day_v3'
SUPPORTED_POLICIES = (POLICY, CORRECTED_POLICY, DAILY_POLICY)


@dataclass(frozen=True)
class PhaseRankFeatures:
    route: str
    source_id: str
    window: int | None = None
    policy: str = POLICY

    def __post_init__(self):
        routes = ('CD', 'D1', 'D5', 'D10', 'D20', 'E1', 'DE1', 'CD_DW')
        if self.route not in routes or self.policy not in SUPPORTED_POLICIES:
            raise ValueError('Unsupported native phase/rank route or policy.')
        required = {'D1': 1, 'D5': 5, 'D10': 10, 'D20': 20, 'DE1': 1}.get(self.route)
        if self.route == 'CD_DW':
            if type(self.window) is not int or self.window not in WINDOWS:
                raise ValueError('Combined context requires an explicit declared rank window.')
        elif self.window != required or (self.window is not None and type(self.window) is not int):
            raise ValueError('Rank window disagrees with the declared route.')
        if (not isinstance(self.source_id, str) or len(self.source_id) != 64
                or any(c not in '0123456789abcdef' for c in self.source_id)):
            raise ValueError('Native context requires an exact SHA256 source identity.')

    @property
    def fixture_columns(self):
        columns = ('countdown',) if self.route in ('CD', 'CD_DW') else ()
        if self.window is not None:
            columns += (f'home_rank_{self.window}', f'away_rank_{self.window}')
        return columns

    @property
    def edge_columns(self):
        return ('receiver_prematch_rank_1', 'opponent_prematch_rank_1') if self.route in ('E1', 'DE1') else ()

    def schema(self):
        return dict(**asdict(self), missing_value=.5, missing_mask='one_per_coordinate',
                    encoding='bounded_passthrough_v1', fixture_columns=list(self.fixture_columns),
                    edge_columns=list(self.edge_columns))


def bounded(values):
    values = np.asarray(values, dtype=float)
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError('Native phase/rank coordinates must lie in [0,1] or be missing.')
    missing = np.isnan(values)
    return np.concatenate((np.where(missing, .5, values), missing.astype(float)), axis=1)


class PhaseRankLookup:
    """Owned exact-cutoff lookup. Public access returns copies; no labels enter it."""
    def __init__(self, frame, *, provenance, policy=POLICY):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError('Unsupported native context publication policy.')
        self.policy = policy
        columns = [*KEYS, 'home_id', 'away_id', 'kickoff_at', 'cutoff_at', *VALUES]
        if set(frame) != set(columns):
            raise ValueError('Native context table must contain exactly the declared identity and bounded fields.')
        from .store import _ids, _times
        data = frame[columns].copy(deep=True).reset_index(drop=True)
        data.attrs = {}
        _ids(data, (*KEYS, 'home_id', 'away_id'))
        for column in ('kickoff_at', 'cutoff_at'):
            data[column] = _times(data[column])
        if data.duplicated(list(KEYS)).any() or data.home_id.eq(data.away_id).any():
            raise ValueError('Native context needs unique fixtures with correctly oriented teams.')
        if data.cutoff_at.gt(data.kickoff_at).any():
            raise ValueError('Native context cutoff cannot follow target kickoff.')
        for column in VALUES:
            data[column] = data[column].to_numpy(dtype=float, na_value=np.nan)
        bounded(data[list(VALUES)])
        self._frame = data.sort_values(list(KEYS), kind='stable').reset_index(drop=True)
        self._index = pd.MultiIndex.from_frame(self._frame[list(KEYS)])
        self._provenance = json.loads(json.dumps(provenance, sort_keys=True))
        digest = sha256(json.dumps(dict(policy=self.policy, provenance=self._provenance), sort_keys=True).encode())
        digest.update(pd.util.hash_pandas_object(self._frame, index=False).values.tobytes())
        self.source_id = digest.hexdigest()

    def __deepcopy__(self, memo):
        # Immutable source data may be shared by CPU input-cache preparations.
        return self

    @property
    def frame(self):
        return self._frame.copy(deep=True)

    def state(self):
        """Data-only representation for native prepared-experiment recovery."""
        return dict(frame=self.frame, provenance=json.loads(json.dumps(self._provenance)),
                    policy=self.policy, source_id=self.source_id)

    @classmethod
    def from_state(cls, state):
        if set(state) != {'frame', 'provenance', 'policy', 'source_id'}:
            raise ValueError('Incomplete prepared phase context state.')
        result = cls(state['frame'], provenance=state['provenance'], policy=state['policy'])
        if result.source_id != state['source_id']:
            raise ValueError('Prepared phase context source identity changed.')
        return result

    def _rows(self, frame):
        indices = self._index.get_indexer(pd.MultiIndex.from_frame(frame[list(KEYS)]))
        if (indices < 0).any():
            raise ValueError('Fixture absent from the frozen native context source.')
        rows = self._frame.iloc[indices]
        if not np.array_equal(pd.to_datetime(frame.kickoff_at, utc=True).array.asi8,
                              rows.kickoff_at.array.asi8):
            # Normalize timestamp resolution before comparing exact instants.
            actual = pd.to_datetime(frame.kickoff_at, utc=True).astype('datetime64[ns, UTC]')
            if not np.array_equal(actual.array.asi8, rows.kickoff_at.array.asi8):
                raise ValueError('Context fixture kickoff disagrees with its source.')
        return rows

    def fixture(self, frame, specification):
        rows = self._rows(frame)
        for side in ('home', 'away'):
            if not np.array_equal(frame[f'{side}_id'], rows[f'{side}_id']):
                raise ValueError('Context home/away identity disagrees with its source.')
        if '_phase_cutoff' not in frame:
            raise ValueError('Decoder ranks require an explicit graph history cutoff.')
        cutoff = pd.to_datetime(frame._phase_cutoff, utc=True).astype('datetime64[ns, UTC]')
        if not np.array_equal(cutoff.array.asi8, rows.cutoff_at.array.asi8):
            raise ValueError('Decoder context requires the exact saved forecast cutoff.')
        return bounded(rows[list(specification.fixture_columns)])

    def edges(self, frame):
        rows = self._rows(frame)
        home = frame.side.eq('home').to_numpy()
        if not frame.side.isin(('home', 'away')).all():
            raise ValueError('Historical context requires home/away edge perspectives.')
        for field, h, a in (('team_id', 'home_id', 'away_id'), ('opponent_id', 'away_id', 'home_id')):
            if not np.array_equal(frame[field], np.where(home, rows[h], rows[a])):
                raise ValueError('Historical context perspective disagrees with fixture identity.')
        rates = rows[['home_rank_1', 'away_rank_1']].to_numpy(float, copy=True)
        rates[~home] = rates[~home, ::-1]
        return bounded(rates)

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self._frame.to_parquet(directory/'phase_context.parquet', index=False)
        (directory/'phase_context.json').write_text(json.dumps(dict(source_id=self.source_id,
            policy=self.policy, provenance=self._provenance), indent=2)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory/'phase_context.json').read_text(encoding='utf-8'))
        obj = cls(pd.read_parquet(directory/'phase_context.parquet'), provenance=meta['provenance'], policy=meta['policy'])
        if obj.source_id != meta['source_id']:
            raise ValueError('Native context fingerprint mismatch.')
        return obj
