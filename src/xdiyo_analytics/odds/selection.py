"""Pinned, identity-aligned odds sources for existing native bet reporters."""

from dataclasses import dataclass, replace
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .crosswalk import MATCH_KEYS, exact_id, read_crosswalk, read_manifest, verified_path
from .workbook import _hash, load_odds

SHEETS = {'corners': 'Corners_Closing_Odds', 'yellow_cards': 'Cards_Closing_Odds',
          '1x2': 'Odds', 'goals': 'Odds', 'btts': 'Odds', 'asian_handicap': 'Odds'}
TIMING_NOTE = ('Opening/closing are vendor labels without quote timestamps or bookmaker identity. '
               'This is a price scenario, not verified executable entry timing. '
               'Tickets do not establish a common bookmaker or simultaneous prices.')


@lru_cache(maxsize=8)
def _partition(snapshot, league, season, sheet, fingerprint):
    return load_odds(snapshot, league=league, season=season, sheet=sheet)


@dataclass(frozen=True)
class OddsSeries:
    """Read the odds database and supply native identity-aligned prices.

    Paths may be relocated together. Seasons are explicit. No fallback, outcome
    matching, provider mixing or synthetic filling occurs. Construction pins the
    manifests automatically; serialized pins reject modified contracts.
    """
    snapshot: str
    crosswalk: str
    seasons: tuple
    market: str
    selection: str
    quote_type: str = 'closing'
    line: float | None = None
    leagues: tuple | None = None
    settlement_confirmed: bool = False
    snapshot_hash: str | None = None
    crosswalk_hash: str | None = None

    def __post_init__(self):
        from .database import current_database
        object.__setattr__(self, 'snapshot', str(current_database(self.snapshot)))
        if not self.seasons or isinstance(self.seasons, str):
            raise ValueError('Select odds seasons explicitly')
        object.__setattr__(self, 'seasons', tuple(self.seasons))
        if self.leagues is not None:
            if isinstance(self.leagues, str) or not self.leagues:
                raise ValueError('Select leagues or use None for all snapshot leagues')
            object.__setattr__(self, 'leagues', tuple(self.leagues))
        if self.market not in SHEETS or self.quote_type not in ('opening', 'closing'):
            raise ValueError('Unsupported market or quote type')
        if self.quote_type == 'opening' and self.market in ('corners', 'yellow_cards', 'asian_handicap'):
            raise ValueError('This source has no opening prices for the selected market')
        if self.market in ('corners', 'yellow_cards', 'goals', 'asian_handicap'):
            if self.line is None or not np.isfinite(self.line) or Decimal(str(self.line))*2 % 1:
                raise ValueError('Choose an integer or half line; quarter-line settlement is unsupported')
        elif self.line is not None:
            raise ValueError('This market does not have a line')
        selections = {'1x2': ('home','draw','away'), 'btts': ('yes','no'), 'asian_handicap': ('home','away')}
        if self.selection not in selections.get(self.market, ('over','under')):
            raise ValueError('Selection is incompatible with the market')
        manifest = read_manifest(self.snapshot)
        crosswalk = read_manifest(self.crosswalk, crosswalk=True)
        for name in ('snapshot', 'crosswalk'):
            digest = _hash(Path(getattr(self, name))/'manifest.json')
            pin = getattr(self, name+'_hash')
            if pin is not None and digest != pin:
                raise ValueError(f'{name} manifest changed since it was pinned')
            object.__setattr__(self, name+'_hash', digest)
        if crosswalk['snapshot'] != manifest['snapshot'] or crosswalk['snapshot_manifest_sha256'] != self.snapshot_hash:
            raise ValueError('Crosswalk does not match the pinned snapshot')

    def cache_key(self):
        self._check_pins()
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def _check_pins(self):
        for name in ('snapshot','crosswalk'):
            if _hash(Path(getattr(self, name))/'manifest.json') != getattr(self, name+'_hash'):
                raise ValueError(f'{name} manifest changed since it was pinned')

    def quotes(self):
        """Return selected provider quotes, without settlement or prediction data."""
        self._check_pins()
        manifest = read_manifest(self.snapshot)
        sheet = SHEETS[self.market]
        frames = []
        for league, season in manifest['league_seasons']:
            if season not in self.seasons or (self.leagues is not None and league not in self.leagues):
                continue
            prefix = f'league={league}/season={season}/{sheet}/quotes-'
            items = [item for item in manifest['files'] if item['path'].startswith(prefix)]
            if not items:
                continue
            # Verify even cached inputs. The bounded cache avoids reparsing a
            # partition separately for every offered line.
            for item in items:
                verified_path(self.snapshot, item)
            fingerprint = tuple(item['sha256'] for item in items)
            frame = _partition(str(Path(self.snapshot).resolve()), league, season, sheet, fingerprint)
            mask = frame.market.eq(self.market) & frame.selection.eq(self.selection) & frame.quote_type.eq(self.quote_type)
            mask &= frame.line.isna() if self.line is None else pd.to_numeric(frame.line, errors='coerce').eq(self.line)
            frames.append(frame.loc[mask].copy())
        if not frames:
            return pd.DataFrame(columns=['match_id','decimal_odds','status','source_column'])
        result = pd.concat(frames, ignore_index=True)
        if result.match_id.duplicated().any():
            raise ValueError('Multiple quotes for a fixture; no implicit best-price selection')
        return result

    def validate_option(self, option):
        from ..labels import MatchTotal, Outcome
        if self.market in ('btts', 'asian_handicap'):
            raise ValueError('BTTS/Asian handicap prices are retained, but native event/settlement integration is unsupported')
        if not self.settlement_confirmed:
            raise ValueError('Confirm native target equivalence to vendor regulation-time settlement before using imported odds')
        source = option.source
        if self.market == '1x2':
            if not isinstance(source, Outcome) or source.perspective not in ('home','away') or not source.higher_is_better:
                raise ValueError('1X2 needs a home/away regulation-time goals Outcome')
            if source.source is not None and (source.source.key != 'goals' or source.source.period != 'ALL' or source.source.field != 'value'):
                raise ValueError('1X2 cannot use corners or another statistic outcome')
            selected = {'draw':'draw', 'win':source.perspective, 'loss':'away' if source.perspective=='home' else 'home'}.get(option.selection)
            if selected != self.selection or option.draw != 'loss':
                raise ValueError('1X2 selection/perspective must match the quote; draw-no-bet is a different market')
        else:
            expected = {'corners':'cornerKicks', 'yellow_cards':'yellowCards', 'goals':'goals'}[self.market]
            if not isinstance(source, MatchTotal) or source.source.key != expected or source.source.period != 'ALL' or source.source.field != 'value':
                raise ValueError(f'{self.market} odds require the full-match {expected} MatchTotal target')
            if option.selection != self.selection or option.line != self.line:
                raise ValueError('Bet selection and line must match the selected quote')
            if float(self.line).is_integer() and option.on_equal != 'push':
                raise ValueError('Integer total lines require push settlement')

    def resolve(self, context):
        """Prices and provenance indexed exactly like the prediction occurrences."""
        self._check_pins()
        if getattr(context, 'layout', 'match') != 'match':
            raise ValueError('Imported total/1X2 odds require a match layout; team markets are unsupported')
        if not set(MATCH_KEYS) <= set(context.metadata):
            raise ValueError('Imported odds need source_league, source_season, competition_id, season_id and event_id metadata')
        manifest = read_manifest(self.snapshot)
        walk = read_crosswalk(self.crosswalk, manifest['snapshot'])
        walk = walk.loc[walk.status.eq('matched')].copy()
        if walk.duplicated(list(MATCH_KEYS)).any():
            raise ValueError('Crosswalk native identities are ambiguous')
        quotes = self.quotes().rename(columns={'match_id':'vendor_match_id', 'status':'quote_status'})
        keep = ['vendor_match_id','decimal_odds','quote_status','source_column']
        joined = walk.merge(quotes[keep], on='vendor_match_id', how='left', validate='one_to_one')
        metadata = context.metadata[list(MATCH_KEYS)].copy()
        for key in MATCH_KEYS:
            metadata[key] = metadata[key].map(exact_id)
            joined[key] = joined[key].map(exact_id)
        joined = joined.set_index(list(MATCH_KEYS))
        query = pd.MultiIndex.from_frame(metadata)
        aligned = joined.reindex(query).copy()
        aligned.index = context.y.index
        aligned['quote_status'] = aligned.quote_status.fillna('unavailable_quote')
        aligned.loc[aligned.vendor_match_id.isna(), 'quote_status'] = 'unmatched_fixture'
        aligned['quote_provider'] = 'imported_odds'
        aligned['quote_snapshot'] = manifest['snapshot']
        aligned['quote_crosswalk'] = read_manifest(self.crosswalk, crosswalk=True)['version']
        aligned['quote_type'] = self.quote_type
        aligned['quote_market'] = self.market
        aligned['quote_selection'] = self.selection
        aligned['quote_line'] = self.line
        aligned['quote_timing'] = 'vendor_label_only_unknown_timestamp'
        aligned['quote_mapping_reason'] = aligned.reason
        aligned['quote_schedule_evidence'] = aligned.get('schedule_evidence')
        aligned['quote_vendor_scheduled_at'] = aligned.vendor_scheduled_at
        aligned['quote_snapshot_hash'] = self.snapshot_hash
        aligned['quote_crosswalk_hash'] = self.crosswalk_hash
        return aligned[['decimal_odds','vendor_match_id','source_column', *[c for c in aligned if c.startswith('quote_')]]]

    def __call__(self, context):
        return self.resolve(context).decimal_odds


def paired_quotes(source):
    """Outer-join both timings; keep separate missing-open and missing-close flags."""
    if source.market in ('corners','yellow_cards','asian_handicap'):
        raise ValueError('Paired opening/closing prices are unavailable for this market')
    pieces = []
    for timing in ('opening','closing'):
        frame = replace(source, quote_type=timing).quotes()
        pieces.append(frame[['match_id','decimal_odds','status']].rename(columns={
            'decimal_odds':timing, 'status':timing+'_status'}).set_index('match_id'))
    result = pieces[0].join(pieces[1], how='outer', validate='one_to_one')
    for timing in ('opening','closing'):
        result['missing_'+timing] = result[timing].isna()
    result['paired'] = ~(result.missing_opening | result.missing_closing)
    return result
