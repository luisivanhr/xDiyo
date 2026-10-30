"""Native JSON scalar persistence, independent of pandas group-key boxing."""
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import DerivedStandings
from xdiyo_analytics.ratings import StandingsRun, build_standings
from test_derived_standings import population


@pytest.mark.parametrize('venues',[('all',),('all','home','away')])
@pytest.mark.parametrize('dtype',[np.int64,np.uint64])
def test_numpy_scope_and_event_ids_roundtrip(tmp_path,venues,dtype):
    # Exercise large exact integers without routing any identifier through float.
    base=2**63+10 if dtype is np.uint64 else 2**53+10
    h,m=population(4,base=base)
    for frame in (h,m):
        for column in ('competition_id','season_id','team_id','opponent_id','event_id'):
            if column in frame: frame[column]=frame[column].astype(dtype)
    # Also test nested skipped-result provenance, not just successful events.
    h.loc[h.event_id.eq(h.event_id.iloc[-1]),['goals_for','goals_against']]=np.nan
    native=build_standings(h,team_seasons=m,normalized_ranks=[3],venues=venues,available_at='release')
    # pandas 2.2 naturally boxes group keys as NumPy scalars. Force the same
    # representation on newer pandas so this regression never silently stops
    # exercising the JSON boundary.
    def boxed(value):
        if isinstance(value,(bool,np.bool_)): return np.bool_(value)
        if isinstance(value,(int,np.integer)): return dtype(value)
        if isinstance(value,(float,np.floating)): return np.float32(value)
        if isinstance(value,dict): return {k:boxed(v) for k,v in value.items()}
        if isinstance(value,tuple): return tuple(boxed(v) for v in value)
        if isinstance(value,list): return [boxed(v) for v in value]
        return value
    run=StandingsRun(native.snapshots,boxed(native.publications),boxed(native.metadata),
        {key:boxed(record) for key,record in native._fixtures.items()})
    before=run.metadata
    run.save(tmp_path)
    restored=StandingsRun.load(tmp_path)
    pd.testing.assert_frame_equal(restored.snapshots,native.snapshots)
    assert restored.metadata==before==run.metadata
    assert restored.publications==run.publications
    assert isinstance(run.metadata['table_computations'],np.integer)
    for venue in (*venues, *(['same'] if len(venues)>1 else [])):
        expr=DerivedStandings([3],include_raw=True,side='both',venue=venue)
        pd.testing.assert_frame_equal(restored.features(h,expr),run.features(h,expr))
    body=json.loads((tmp_path/'standings.json').read_text(encoding='utf-8'))
    assert body['version']==3
    assert type(body['fixtures'][0]['home']) is int
    assert body['fixtures'][0]['home']==base+1
    assert type(body['fixtures'][0]['scope'][0]) is int
    assert type(body['publications'][0]['eligible_result_count']) is int
    assert type(body['metadata']['skipped'][0]['event']) is int


@pytest.mark.parametrize('value',[np.float32(.25),np.int32(17),np.uint64(2**63+5),np.bool_(True)])
def test_numpy_metadata_scalars_remain_typed(tmp_path,value):
    h,m=population(4)
    native=build_standings(h,team_seasons=m)
    metadata={**native.metadata,'numeric_evidence':value}
    run=StandingsRun(native.snapshots,native.publications,metadata,native._fixtures)
    run.save(tmp_path)
    result=json.loads((tmp_path/'standings.json').read_text())['metadata']['numeric_evidence']
    assert result==value.item() and type(result) is type(value.item())


@pytest.mark.parametrize('bad,error',[(np.float32(np.nan),ValueError),(np.float32(np.inf),ValueError),(object(),TypeError)])
def test_invalid_metadata_is_not_stringified_or_published(tmp_path,bad,error):
    h,m=population(4)
    native=build_standings(h,team_seasons=m)
    run=StandingsRun(native.snapshots,native.publications,{**native.metadata,'invalid':bad},native._fixtures)
    with pytest.raises(error): run.save(tmp_path)
    assert not (tmp_path/'standings.json').exists()
    assert not (tmp_path/'snapshots.parquet').exists()
