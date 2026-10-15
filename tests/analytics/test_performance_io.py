"""Independent output and lifetime checks for the library performance branch."""
import gc
import json
import weakref
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.experiments.recovery import pack, _iter_pack_json, dump_bundle, load_bundle


@pytest.mark.parametrize('size', [0, 3, 2050])
def test_streaming_recovery_identical_bytes_types_and_attrs(tmp_path, size):
    frame = pd.DataFrame({'a': pd.array(([1,None,0]*((size+2)//3))[:size],dtype='Int64'),
                          'b': pd.array((['é',None,'']*((size+2)//3))[:size], dtype='string'),
                          'c': pd.Categorical((['x',None,'z']*((size+2)//3))[:size], categories=['z','x'], ordered=True)})
    frame['time'] = pd.date_range('2024-01-01', periods=size, tz='Asia/Tokyo')
    frame.attrs = {'nested': {'geometry':[1, None, np.nan]}}
    payload = {'frame': frame, 'series': frame.a, 'tuple': (np.inf, -np.inf, pd.NaT, pd.NA),
               'empty': [], 'array': np.array([[1., np.nan]])}
    expected = json.dumps(pack(payload), ensure_ascii=False, allow_nan=False)
    assert ''.join(_iter_pack_json(payload)) == expected
    path=tmp_path/'bundle.json';dump_bundle(path,payload)
    assert path.read_text(encoding='utf-8') == '{"schema": 1, "payload": '+expected+'}'
    restored=load_bundle(path)
    pd.testing.assert_frame_equal(restored['frame'], frame)
    pd.testing.assert_series_equal(restored['series'], frame.a)
    with pytest.raises(FileExistsError): dump_bundle(path, payload)


def test_shared_strings_detached_and_rich_text_preserved(monkeypatch):
    from xdiyo_analytics.odds import workbook
    archive=BytesIO()
    with ZipFile(archive,'w') as zip:
        zip.writestr('xl/sharedStrings.xml', '<sst xmlns="'+workbook.NS[1:-1]+'">'+
            ''.join('<si><r><t xml:space="preserve"> a </t></r><r><t>b</t></r></si>' for _ in range(3000))+'</sst>')
    archive.seek(0)
    book=workbook._Workbook.__new__(workbook._Workbook);book.archive=ZipFile(archive);book.strings={}
    original=workbook.ET.iterparse
    roots=[]
    def parse(*args,**kwargs):
        first=True
        for event,element in original(*args,**kwargs):
            if first: roots.append(element); first=False
            yield event,element
    monkeypatch.setattr(workbook.ET,'iterparse',parse)
    book.resolve([2999]);assert book.strings == {2999:' a b'}
    assert len(roots[0]) == 0
    with pytest.raises(ValueError, match='Invalid shared-string'): book.resolve([3000])
    book.archive.close()


def test_csv_cache_tracks_content_mutation_and_preserves_links():
    from xdiyo_analytics.reporting.viewer import _csv_link
    cache={};frame=pd.DataFrame({'x':[1.]})
    assert _csv_link(frame,'one',cache=cache).replace('one.csv','two.csv') == _csv_link(frame,'two',cache=cache)
    assert len(cache)==1
    before=_csv_link(frame,'one',cache=cache);frame.iloc[0,0]=2
    assert _csv_link(frame,'one',cache=cache) != before and len(cache)==2


def test_odds_operation_over_eight_partitions_bounded_and_released():
    from xdiyo_analytics.odds._cache import operation, table, _active, _Tables
    calls=[]; refs=[]
    def load(key):
        calls.append(key); frame=pd.DataFrame({'price':[float(key)]})
        refs.append(weakref.ref(frame)); return frame
    @operation
    def nested():
        for key in range(12):
            assert table(key, lambda: load(key)).price.iloc[0] == key
    @operation
    def run():
        nested(); nested()
        assert len(calls)==12 and len(_active.get().entries)==12
    run();gc.collect()
    assert _active.get() is None and all(ref() is None for ref in refs)
    small=_Tables(limit=150)
    for key in range(20): small.get(key, lambda: pd.DataFrame({'x':[key]}))
    assert small.bytes <= 150 and len(small.entries) <= 1
    # Changed content identities cannot hit an old entry.
    cache=_Tables();a=cache.get(('path','hash1'),lambda: pd.DataFrame({'x':[1]}))
    b=cache.get(('path','hash2'),lambda: pd.DataFrame({'x':[2]}))
    assert a.x.iloc[0] == 1 and b.x.iloc[0] == 2
    @operation
    def fail():
        table('failure',lambda: load(99)); raise RuntimeError('cancel')
    with pytest.raises(RuntimeError):fail()
    assert _active.get() is None


def test_custom_crosswalk_nested_cells_do_not_mutate_cached_table(tmp_path, monkeypatch):
    import hashlib
    from xdiyo_analytics.odds import crosswalk
    from xdiyo_analytics.odds._cache import operation
    source = pd.DataFrame({'status': ['matched'], 'custom': [{'notes': ['original']}],
                           'array': [np.array([1, 2])]})
    path = tmp_path/'fixtures.parquet'; source.to_parquet(path, index=False)
    # Logical identity is calculated on the actual parquet representation.
    stored = pd.read_parquet(path)
    manifest = {'snapshot': 'test', 'version': hashlib.sha256(stored.to_json(orient='records', date_format='iso').encode()).hexdigest()[:24],
                'files': [{'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
    monkeypatch.setattr(crosswalk, 'read_manifest', lambda *a, **k: manifest)
    calls = []; read = pd.read_parquet
    def recorded(*a, **k):
        calls.append(1); return read(*a, **k)
    monkeypatch.setattr(pd, 'read_parquet', recorded)
    @operation
    def run():
        first = crosswalk.read_crosswalk(tmp_path, 'test')
        first.at[0, 'custom']['notes'][0] = 'changed'
        first.at[0, 'array'][0] = 99
        second = crosswalk.read_crosswalk(tmp_path, 'test')
        assert second.at[0, 'custom']['notes'][0] == 'original'
        assert second.at[0, 'array'][0] == 1
        assert calls == [1]
    run()


@pytest.mark.parametrize('size', [0, 3, 2050])
def test_streaming_recovery_matches_frozen_reference_encoder(size):
    import hashlib
    from pathlib import Path
    import types
    path = Path(__file__).parent/'reference/recovery_01617da.txt'
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == '10e608476ff89666a2a1440ee3ec8e21071b94b687fb2b7e549fbfe2d2367ceb'
    reference = types.ModuleType('xdiyo_analytics.experiments._frozen_recovery')
    reference.__package__ = 'xdiyo_analytics.experiments'
    exec(compile(source, str(path), 'exec'), reference.__dict__)
    frame = pd.DataFrame({'integer': pd.array(([1, None, 0]*((size+2)//3))[:size], dtype='Int64'),
                          'text': pd.array((['é', None, '']*((size+2)//3))[:size], dtype='string'),
                          'nested': ([{'x': [1, None, np.nan]}]*size)})
    frame['category'] = pd.Categorical((['a', None, 'b']*((size+2)//3))[:size], categories=['b','a'], ordered=True)
    frame['time'] = pd.date_range('2024-01-01', periods=size, tz='Asia/Tokyo')
    frame.attrs = {'lineage': {'missing': pd.NA, 'scales': np.array([1., np.nan])}}
    values = dict(frame=frame, series=frame.integer, empty=pd.DataFrame(),
                  scalars=(np.inf, -np.inf, pd.NA, pd.NaT), array=np.array([[1., np.nan]]))
    assert ''.join(_iter_pack_json(values)) == json.dumps(reference.pack(values), ensure_ascii=False, allow_nan=False)
